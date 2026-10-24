use crate::{
    ContextBudget, ContextTemplate, GenerationResponse, KoboldClient, Preset, PromptBuilder,
    PromptData, PromptMessage, PromptRole, StoryStringPosition, StoryStringRole,
    SystemPromptTemplate,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{future::Future, pin::Pin};

/// Wire roles are deliberately separate from persisted application roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl From<PromptMessage> for ChatMessage {
    fn from(message: PromptMessage) -> Self {
        Self {
            role: match message.role {
                PromptRole::System => ChatRole::System,
                PromptRole::User => ChatRole::User,
                PromptRole::Assistant => ChatRole::Assistant,
            },
            content: message.content,
        }
    }
}

/// Structured messages only: do not wrap an already formatted text prompt here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatGenerationRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default = "empty_parameters")]
    pub parameters: Value,
}

fn empty_parameters() -> Value {
    json!({})
}

impl ChatGenerationRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            stop_sequences: Vec::new(),
            parameters: empty_parameters(),
        }
    }

    /// Translate ST sampling parameters to KoboldCpp's chat endpoint names.
    pub fn from_preset(messages: Vec<ChatMessage>, preset: &Preset) -> Result<Self> {
        let mut parameters = preset.kobold_request("", &[], 1)?;
        let map = parameters
            .as_object_mut()
            .expect("preset produces an object");
        for key in ["prompt", "max_context_length", "stop_sequence", "trim_stop"] {
            map.remove(key);
        }
        if let Some(value) = map.remove("max_length") {
            map.insert("max_tokens".into(), value);
        }
        if let Some(value) = map.remove("mirostat") {
            map.insert("mirostat_mode".into(), value);
        }
        if let Some(value) = map.remove("sampler_seed") {
            map.insert("seed".into(), value);
        }
        Ok(Self {
            messages,
            parameters,
            stop_sequences: Vec::new(),
        })
    }

    pub(crate) fn payload(&self, max_context: u64) -> Result<Value> {
        if self.messages.is_empty() {
            bail!("Chat Completion requires at least one message");
        }
        let mut map = self
            .parameters
            .as_object()
            .cloned()
            .context("chat parameters must be an object")?;
        // These would replace the structured input, alter the token budget, or
        // request a response format this text-only adapter cannot consume.
        for key in [
            "prompt",
            "messages",
            "memory",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "images",
            "audio",
            "max_length",
            "max_completion_tokens",
            "max_context_length",
            "stop",
            "stop_sequence",
            "stream",
            "n",
            "continue_assistant_turn",
        ] {
            if map.contains_key(key) {
                bail!(
                    "Chat parameter {key:?} is reserved or unsupported; use the structured request fields"
                );
            }
        }
        map.entry("max_tokens".to_owned()).or_insert(256.into());
        if map["max_tokens"].as_u64().filter(|n| *n > 0).is_none() {
            bail!("max_tokens must be a positive integer");
        }
        map.insert("messages".into(), serde_json::to_value(&self.messages)?);
        map.insert("stop".into(), serde_json::to_value(&self.stop_sequences)?);
        map.insert("max_context_length".into(), max_context.into());
        map.insert("stream".into(), false.into());
        map.insert("n".into(), 1.into());
        // A trailing historical assistant turn must not become a prefill.
        map.insert("continue_assistant_turn".into(), false.into());
        Ok(Value::Object(map))
    }
}

/// Backend-rendered text and budgeted token count (template tokens plus the
/// request's explicit per-image reserve for multimodal requests).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatPreview {
    pub prompt: String,
    #[serde(rename = "value")]
    pub tokens: u64,
}

pub trait ChatTokenCounter: Send + Sync {
    fn preview_chat<'a>(
        &'a self,
        request: &'a ChatGenerationRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatPreview>> + Send + 'a>>;
}

#[derive(Debug)]
pub struct FittedChatPrompt {
    pub request: ChatGenerationRequest,
    pub preview: ChatPreview,
    pub input_budget: u64,
    pub dropped_messages: usize,
}

impl PromptBuilder {
    /// Render semantic chat messages without local instruct tokens. The backend
    /// owns the chat template and assistant generation prefix.
    pub fn build_chat(
        &self,
        context: &ContextTemplate,
        system: &SystemPromptTemplate,
        data: &PromptData,
    ) -> Result<Vec<ChatMessage>> {
        if !context.chat_start.is_empty() {
            bail!(
                "context.chat_start is text-completion formatting; use a chat context without chat_start"
            );
        }
        let parts = self.render_parts(context, system, data)?;
        let history = if data.group_roleplay {
            Self::group_roleplay_messages(&parts.messages, data)
        } else {
            parts.messages
        };
        let mut messages: Vec<ChatMessage> = history.into_iter().map(Into::into).collect();
        if !parts.story.is_empty() {
            let (at, role) = match context.story_string_spec().position {
                StoryStringPosition::Top => (0, ChatRole::System),
                StoryStringPosition::InChat { depth, role } => (
                    messages.len().saturating_sub(depth),
                    match role {
                        StoryStringRole::System => ChatRole::System,
                        StoryStringRole::User => ChatRole::User,
                        StoryStringRole::Assistant => ChatRole::Assistant,
                    },
                ),
            };
            messages.insert(
                at,
                ChatMessage {
                    role,
                    content: parts.story,
                },
            );
        }
        if !parts.post_history.is_empty() {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: parts.post_history,
            });
        }
        Ok(messages)
    }

    /// Fit original history before grouping/injection, counting the server's
    /// chat template rather than a local text-completion approximation.
    pub async fn build_chat_with_budget(
        &self,
        context: &ContextTemplate,
        system: &SystemPromptTemplate,
        data: &PromptData,
        counter: &(impl ChatTokenCounter + ?Sized),
        options: &ChatGenerationRequest,
        budget: ContextBudget,
    ) -> Result<FittedChatPrompt> {
        let limit = budget.input_tokens()?;
        let response = options
            .parameters
            .get("max_tokens")
            .cloned()
            .unwrap_or(256.into());
        if response.as_u64() != Some(budget.response_tokens) {
            bail!("Chat response reservation must match max_tokens");
        }
        let latest = data
            .messages
            .iter()
            .rposition(|m| m.role != PromptRole::System);
        let removable: Vec<usize> = data
            .messages
            .iter()
            .enumerate()
            .filter(|(i, m)| Some(*i) != latest && m.role != PromptRole::System)
            .map(|(i, _)| i)
            .collect();
        let measure = async |dropped: usize| -> Result<FittedChatPrompt> {
            let mut candidate = data.clone();
            candidate.messages = data
                .messages
                .iter()
                .enumerate()
                .filter(|(i, _)| removable[..dropped].binary_search(i).is_err())
                .map(|(_, m)| m.clone())
                .collect();
            let mut request = options.clone();
            request.messages = self.build_chat(context, system, &candidate)?;
            let preview = counter.preview_chat(&request).await?;
            Ok(FittedChatPrompt {
                request,
                preview,
                input_budget: limit,
                dropped_messages: dropped,
            })
        };
        let full = measure(0).await?;
        if full.preview.tokens <= limit {
            return Ok(full);
        }
        let mut fitting = measure(removable.len()).await?;
        if fitting.preview.tokens > limit {
            bail!(
                "Chat prompt cannot fit: mandatory content and latest message use {} tokens; input budget is {}",
                fitting.preview.tokens,
                limit
            );
        }
        let (mut low, mut high) = (1, removable.len());
        while low < high {
            let mid = low + (high - low) / 2;
            let candidate = measure(mid).await?;
            if candidate.preview.tokens <= limit {
                high = mid;
                fitting = candidate;
            } else {
                low = mid + 1;
            }
        }
        Ok(fitting)
    }
}

impl KoboldClient {
    async fn preview_chat_payload(
        &self,
        request: &crate::ChatCompletionRequest,
        payload: &Value,
    ) -> Result<ChatPreview> {
        let mut payload = payload.clone();
        payload["special"] = true.into();
        let mut preview = self
            .http
            .post(format!(
                "{}/api/extra/tokencount",
                self.base_url.trim_end_matches('/')
            ))
            .json(&payload)
            .send()
            .await
            .context("Chat tokenization request failed")?
            .error_for_status()
            .context("Chat tokenization HTTP error")?
            .json::<ChatPreview>()
            .await
            .context(
                "Backend must support chat messages and return the rendered prompt from tokencount",
            )?;
        if preview.prompt.is_empty() || preview.tokens == 0 {
            bail!("Backend did not render/tokenize the chat messages; refusing unsafe generation");
        }
        preview.tokens = preview
            .tokens
            .checked_add(request.image_tokens()?)
            .context("Multimodal token budget overflow")?;
        Ok(preview)
    }

    pub async fn preview_chat(
        &self,
        request: &(impl crate::ChatRequest + ?Sized),
    ) -> Result<ChatPreview> {
        let request = request.to_completion_request();
        self.preview_chat_payload(&request, &request.payload(self.max_context_length())?)
            .await
    }

    /// Send structured messages to /v1/chat/completions. This does not invoke the
    /// local instruct renderer or convert a flat prompt back to messages.
    pub async fn generate_chat(
        &self,
        request: &(impl crate::ChatRequest + ?Sized),
    ) -> Result<GenerationResponse> {
        let response = self.generate_chat_full(request).await?;
        if !response.tool_calls.is_empty() {
            bail!("Chat returned tool calls; use generate_chat_full to handle them explicitly");
        }
        Ok(GenerationResponse {
            text: response
                .content
                .context("Chat completion returned no text content")?,
        })
    }

    pub async fn generate_chat_full(
        &self,
        request: &(impl crate::ChatRequest + ?Sized),
    ) -> Result<crate::ChatGenerationResponse> {
        let request = request.to_completion_request();
        let payload = request.payload(self.max_context_length())?;
        let limit = ContextBudget::new(
            self.max_context_length(),
            payload["max_tokens"].as_u64().unwrap(),
        )
        .input_tokens()?;
        let preview = self.preview_chat_payload(&request, &payload).await?;
        if preview.tokens > limit {
            bail!(
                "Chat prompt uses {} tokens, input budget is {}; use build_chat_with_budget",
                preview.tokens,
                limit
            );
        }
        let response = self
            .http
            .post(format!(
                "{}/v1/chat/completions",
                self.base_url.trim_end_matches('/')
            ))
            .json(&payload)
            .send()
            .await
            .context("Chat completion request failed")?
            .error_for_status()
            .context("Chat completion HTTP error")?
            .json::<Value>()
            .await
            .context("Invalid chat completion response")?;
        let response = crate::ChatGenerationResponse::decode(response)?;
        if matches!(request.tool_choice, Some(crate::ToolChoice::None))
            && !response.tool_calls.is_empty()
        {
            bail!("Backend returned tool calls despite tool_choice none");
        }
        if let Some(crate::ToolChoice::Function(name)) = &request.tool_choice
            && (response.tool_calls.is_empty()
                || response.tool_calls.iter().any(|c| &c.function.name != name))
        {
            bail!("Backend did not honor forced tool choice {name}");
        }
        if matches!(request.tool_choice, Some(crate::ToolChoice::Required))
            && response.tool_calls.is_empty()
        {
            bail!("Backend did not honor required tool choice");
        }
        Ok(response)
    }
}

impl ChatTokenCounter for KoboldClient {
    fn preview_chat<'a>(
        &'a self,
        request: &'a ChatGenerationRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatPreview>> + Send + 'a>> {
        Box::pin(KoboldClient::preview_chat(self, request))
    }
}
