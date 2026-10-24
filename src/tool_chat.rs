//! Structured tool protocol. Execution is deliberately separate from HTTP.
use crate::{
    ChatGenerationRequest, ChatMessage, ChatPreview, ChatRole, ContextBudget, KoboldClient,
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionDefinition,
}
impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            kind: ToolType::Function,
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.function.name.is_empty(),
            "Tool name must not be empty"
        );
        ensure!(
            self.function.parameters.is_object() || self.function.parameters.is_boolean(),
            "Tool parameters must be a JSON Schema object or boolean"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}
impl Serialize for ToolChoice {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Auto => "auto".serialize(serializer),
            Self::None => "none".serialize(serializer),
            Self::Required => "required".serialize(serializer),
            Self::Function(name) => {
                json!({"type":"function","function":{"name":name}}).serialize(serializer)
            }
        }
    }
}
impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value.as_str() {
            Some("auto") => Ok(Self::Auto),
            Some("none") => Ok(Self::None),
            Some("required") => Ok(Self::Required),
            _ if value["type"] == "function" => value["function"]["name"]
                .as_str()
                .filter(|n| !n.is_empty())
                .map(|n| Self::Function(n.into()))
                .ok_or_else(|| serde::de::Error::custom("Missing function name")),
            _ => Err(serde::de::Error::custom("Invalid tool_choice")),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// Retained verbatim, including malformed JSON. Parse explicitly.
    pub arguments: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionCall,
}
impl ToolCall {
    pub fn arguments<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_str(&self.function.arguments).with_context(|| {
            format!(
                "Invalid arguments for tool {} (call {})",
                self.function.name, self.id
            )
        })
    }
}

/// Separate from ChatMessage so existing two-field struct literals keep compiling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionMessage {
    System {
        content: crate::ChatContent,
    },
    User {
        content: crate::ChatContent,
    },
    Assistant {
        content: Option<crate::ChatContent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(
            default,
            deserialize_with = "deserialize_calls",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        /// Preserve additional assistant fields returned by the server.
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}
fn deserialize_calls<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<ToolCall>, D::Error> {
    Ok(Option::<Vec<ToolCall>>::deserialize(deserializer)?.unwrap_or_default())
}
impl ChatCompletionMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: crate::ChatContent::Text(content.into()),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: crate::ChatContent::Text(content.into()),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: Some(crate::ChatContent::Text(content.into())),
            reasoning_content: None,
            tool_calls: Vec::new(),
            extra: Map::new(),
        }
    }
    pub fn tool(id: impl Into<String>, result: &Value) -> Self {
        Self::Tool {
            tool_call_id: id.into(),
            content: result.to_string(),
        }
    }
    pub fn content(&self) -> Option<&str> {
        match self {
            Self::System { content } | Self::User { content } => content.as_text(),
            Self::Tool { content, .. } => Some(content),
            Self::Assistant { content, .. } => {
                content.as_ref().and_then(crate::ChatContent::as_text)
            }
        }
    }
    pub fn user_parts(parts: Vec<crate::ContentPart>) -> Self {
        Self::User {
            content: crate::ChatContent::Parts(parts),
        }
    }
    pub fn text_parts(&self) -> Vec<&str> {
        match self {
            Self::System { content } | Self::User { content } => content.text_parts(),
            Self::Assistant { content, .. } => {
                content.as_ref().map(|c| c.text_parts()).unwrap_or_default()
            }
            Self::Tool { content, .. } => vec![content],
        }
    }
    pub fn image_count(&self) -> usize {
        match self {
            Self::System { content } | Self::User { content } => content.image_count(),
            Self::Assistant { content, .. } => content.as_ref().map_or(0, |c| c.image_count()),
            Self::Tool { .. } => 0,
        }
    }
    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Self::Assistant { tool_calls, .. } => tool_calls,
            _ => &[],
        }
    }
}
impl From<ChatMessage> for ChatCompletionMessage {
    fn from(m: ChatMessage) -> Self {
        match m.role {
            ChatRole::System => Self::system(m.content),
            ChatRole::User => Self::user(m.content),
            ChatRole::Assistant => Self::assistant(m.content),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatCompletionMessage>,
    /// Extra context reservation per image: Kobold's textual preview does not
    /// count vision embeddings. Tune to the loaded vision encoder's budget.
    #[serde(default = "image_reserve")]
    pub image_token_reserve: u64,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default = "parameters")]
    pub parameters: Value,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
}
fn parameters() -> Value {
    json!({})
}
fn image_reserve() -> u64 {
    4096
}
impl ChatCompletionRequest {
    /// Tools are disabled by default (no definitions are sent).
    pub fn new(messages: Vec<ChatCompletionMessage>) -> Self {
        Self {
            messages,
            image_token_reserve: image_reserve(),
            stop_sequences: vec![],
            parameters: parameters(),
            tools: vec![],
            tool_choice: None,
            chat_template_kwargs: Map::new(),
        }
    }
    pub fn disable_tools(&mut self) {
        self.tools.clear();
        self.tool_choice = Some(ToolChoice::None);
    }
    pub(crate) fn image_tokens(&self) -> Result<u64> {
        let count: usize = self.messages.iter().map(|m| m.image_count()).sum();
        (count as u64)
            .checked_mul(self.image_token_reserve)
            .context("Image token reservation overflow")
    }
    pub(crate) fn payload(&self, max_context: u64) -> Result<Value> {
        self.units()?;
        let mut payload = ChatGenerationRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: String::new(),
            }],
            stop_sequences: self.stop_sequences.clone(),
            parameters: self.parameters.clone(),
        }
        .payload(max_context)?;
        let mut names = HashSet::new();
        for tool in &self.tools {
            tool.validate()?;
            ensure!(
                names.insert(&tool.function.name),
                "Duplicate tool definition: {}",
                tool.function.name
            );
        }
        match &self.tool_choice {
            Some(ToolChoice::Required) => {
                ensure!(!self.tools.is_empty(), "Required tool choice needs tools")
            }
            Some(ToolChoice::Function(name)) => {
                ensure!(names.contains(name), "Forced tool is not defined: {name}")
            }
            _ => {}
        }
        payload["messages"] = serde_json::to_value(&self.messages)?;
        if !self.tools.is_empty() {
            payload["tools"] = serde_json::to_value(&self.tools)?;
        }
        if let Some(choice) = &self.tool_choice {
            payload["tool_choice"] = serde_json::to_value(choice)?;
        }
        if !self.chat_template_kwargs.is_empty() {
            ensure!(
                payload.get("chat_template_kwargs").is_none(),
                "Set chat_template_kwargs in either parameters or the structured field, not both"
            );
            payload["chat_template_kwargs"] = serde_json::to_value(&self.chat_template_kwargs)?;
        }
        if let Some(kwargs) = payload.get("chat_template_kwargs") {
            let kwargs = kwargs
                .as_object()
                .context("chat_template_kwargs must be an object")?;
            for key in [
                "messages",
                "tools",
                "tool_choice",
                "add_generation_prompt",
                "continue_final_message",
            ] {
                ensure!(
                    !kwargs.contains_key(key),
                    "Reserved chat_template_kwargs field: {key}"
                );
            }
        }
        Ok(payload)
    }
    /// Complete assistant/calls/results exchanges are indivisible trimming units.
    fn units(&self) -> Result<Vec<std::ops::Range<usize>>> {
        ensure!(
            !self.messages.is_empty(),
            "Chat Completion requires at least one message"
        );
        let mut units = Vec::new();
        let mut seen = HashSet::new();
        let mut i = 0;
        while i < self.messages.len() {
            let start = i;
            ensure!(
                !matches!(self.messages[i], ChatCompletionMessage::Tool { .. }),
                "Orphan tool result at message {i}"
            );
            if let ChatCompletionMessage::Assistant { extra, .. } = &self.messages[i] {
                for key in [
                    "role",
                    "content",
                    "reasoning_content",
                    "tool_calls",
                    "tool_call_id",
                ] {
                    ensure!(
                        !extra.contains_key(key),
                        "Reserved assistant extension field: {key}"
                    );
                }
            }
            let calls = self.messages[i].tool_calls();
            let mut pending = HashSet::new();
            for call in calls {
                ensure!(
                    !call.id.is_empty() && !call.function.name.is_empty(),
                    "Empty tool call ID or name"
                );
                ensure!(
                    seen.insert(call.id.as_str()),
                    "Duplicate tool call ID: {}",
                    call.id
                );
                pending.insert(call.id.as_str());
            }
            i += 1;
            while !pending.is_empty() {
                match self.messages.get(i) {
                    Some(ChatCompletionMessage::Tool { tool_call_id, .. })
                        if pending.remove(tool_call_id.as_str()) =>
                    {
                        i += 1
                    }
                    _ => bail!(
                        "Incomplete tool exchange at message {start}: missing, duplicate or mismatched results"
                    ),
                }
            }
            units.push(start..i);
        }
        Ok(units)
    }
}
impl From<ChatGenerationRequest> for ChatCompletionRequest {
    fn from(request: ChatGenerationRequest) -> Self {
        Self {
            messages: request.messages.into_iter().map(Into::into).collect(),
            parameters: request.parameters,
            stop_sequences: request.stop_sequences,
            ..Self::new(vec![])
        }
    }
}
impl ChatGenerationRequest {
    pub fn with_tools(self, tools: Vec<ToolDefinition>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            tools,
            ..self.into()
        }
    }
}

mod sealed {
    pub trait Sealed {}
}
/// Implemented by the legacy and extended requests; not an arbitrary payload hook.
pub trait ChatRequest: sealed::Sealed {
    fn to_completion_request(&self) -> ChatCompletionRequest;
}
impl sealed::Sealed for ChatGenerationRequest {}
impl sealed::Sealed for ChatCompletionRequest {}
impl ChatRequest for ChatGenerationRequest {
    fn to_completion_request(&self) -> ChatCompletionRequest {
        self.clone().into()
    }
}
impl ChatRequest for ChatCompletionRequest {
    fn to_completion_request(&self) -> ChatCompletionRequest {
        self.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatGenerationResponse {
    pub message: ChatCompletionMessage,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub finish_reason: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}
impl ChatGenerationResponse {
    pub(crate) fn decode(value: Value) -> Result<Self> {
        #[derive(Deserialize)]
        struct Response {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: ChatCompletionMessage,
            #[serde(default)]
            finish_reason: Option<String>,
        }
        let response: Response =
            serde_json::from_value(value).context("Invalid chat completion response")?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .context("Chat completion returned no choices")?;
        let ChatCompletionMessage::Assistant {
            content,
            reasoning_content,
            tool_calls,
            ..
        } = &choice.message
        else {
            bail!("Chat completion returned a non-assistant role")
        };
        let text = content.as_ref().and_then(|c| {
            let parts = c.text_parts();
            (!parts.is_empty()).then(|| parts.concat())
        });
        ensure!(
            text.is_some() || !tool_calls.is_empty(),
            "Chat completion returned neither text nor tool calls"
        );
        let mut ids = HashSet::new();
        for call in tool_calls {
            ensure!(
                !call.id.is_empty() && !call.function.name.is_empty() && ids.insert(&call.id),
                "Invalid or duplicate tool call ID/name"
            );
        }
        ensure!(
            choice.finish_reason.as_deref() != Some("tool_calls") || !tool_calls.is_empty(),
            "finish_reason tool_calls without calls"
        );
        Ok(Self {
            content: text,
            reasoning_content: reasoning_content.clone(),
            tool_calls: tool_calls.clone(),
            message: choice.message,
            finish_reason: choice.finish_reason,
        })
    }
}

#[derive(Debug)]
pub struct FittedChatCompletion {
    pub request: ChatCompletionRequest,
    pub preview: ChatPreview,
    pub dropped_messages: usize,
    pub input_budget: u64,
}
impl KoboldClient {
    /// Fit rich history, retaining systems and the most recent entire exchange.
    /// Tool schemas and template kwargs remain present in every server preview.
    pub async fn fit_chat(&self, request: &ChatCompletionRequest) -> Result<FittedChatCompletion> {
        let payload = request.payload(self.max_context_length())?;
        let input_budget = ContextBudget::new(
            self.max_context_length(),
            payload["max_tokens"].as_u64().context("max_tokens")?,
        )
        .input_tokens()?;
        let units = request.units()?;
        let latest = units.iter().rposition(|u| {
            !matches!(
                request.messages[u.start],
                ChatCompletionMessage::System { .. }
            )
        });
        let mut candidate = request.clone();
        let mut dropped_messages = 0;
        // Linear rather than binary: arbitrary Jinja templates need not have
        // monotonic token counts when turns are removed.
        let mut removable = units.iter().enumerate().filter(|(i, u)| {
            Some(*i) != latest
                && !matches!(
                    request.messages[u.start],
                    ChatCompletionMessage::System { .. }
                )
        });
        loop {
            let preview = self.preview_chat(&candidate).await?;
            if preview.tokens <= input_budget {
                return Ok(FittedChatCompletion {
                    request: candidate,
                    preview,
                    dropped_messages,
                    input_budget,
                });
            }
            let (_, unit) = removable.next().context("Chat prompt cannot fit: mandatory instructions, tools and latest exchange exceed context")?;
            candidate
                .messages
                .drain(unit.start - dropped_messages..unit.end - dropped_messages);
            dropped_messages += unit.len();
        }
    }
}

// Accept JSON emitted with different whitespace/key order. Fail closed for
// templates that transform or omit tool data: token counts alone cannot prove
// that a backend actually included the tools in the counted prompt.
