use std::{future::Future, pin::Pin};

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::preset::Preset;
use crate::{ContextBudget, TokenCounter};

#[derive(Clone)]
pub struct KoboldClient {
    http: Client,
    base_url: String,
    max_context_length: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationResponse {
    pub text: String,
}

#[derive(Deserialize)]
struct KoboldResponse {
    results: Vec<GenerateResult>,
}

#[derive(Deserialize)]
struct GenerateResult {
    text: String,
}

impl KoboldClient {
    pub fn context_budget(&self, preset: &Preset) -> Result<ContextBudget> {
        Ok(ContextBudget::new(
            self.max_context_length,
            preset.generation_length()?,
        ))
    }

    /// Uses the loaded model's tokenizer, including special tokens. Fail closed
    /// when tokenization is unavailable; character-based estimates are unsafe.
    pub async fn count_tokens(&self, prompt: &str) -> Result<u64> {
        #[derive(Deserialize)]
        struct Count {
            value: u64,
        }
        let result = self
            .http
            .post(format!(
                "{}/api/extra/tokencount",
                self.base_url.trim_end_matches('/')
            ))
            .json(&serde_json::json!({"prompt": prompt, "special": true}))
            .send()
            .await
            .context("KoboldCpp tokenization request failed")?
            .error_for_status()
            .context("KoboldCpp tokenization HTTP error")?
            .json::<Count>()
            .await
            .context("KoboldCpp returned an invalid token count")?;
        if result.value == 0 && !prompt.is_empty() {
            bail!("KoboldCpp returned zero tokens for a nonempty prompt");
        }
        Ok(result.value)
    }

    pub async fn connect(base_url: impl Into<String>) -> Result<Self> {
        Self::connect_with_client(base_url, Client::new()).await
    }

    /// Supply a client configured with the application's timeouts, proxy or TLS.
    pub async fn connect_with_client(base_url: impl Into<String>, http: Client) -> Result<Self> {
        let client = Self {
            http,
            base_url: base_url.into(),
            max_context_length: 0,
        };
        let max_context_length = client.load_max_context_length().await?;
        Ok(Self {
            max_context_length,
            ..client
        })
    }

    pub async fn generate(
        &self,
        prompt: &str,
        preset: &Preset,
        stop_sequences: &[String],
    ) -> Result<String> {
        let request = preset.kobold_request(prompt, stop_sequences, self.max_context_length)?;
        self.generate_json(&request).await
    }

    /// KoboldCpp's maximum prompt context detected during `connect`.
    pub fn max_context_length(&self) -> u64 {
        self.max_context_length
    }

    async fn generate_json(&self, request: &Value) -> Result<String> {
        let response_tokens = request["max_length"]
            .as_u64()
            .filter(|n| *n > 0)
            .context("max_length must be a positive integer")?;
        let limit = ContextBudget::new(self.max_context_length, response_tokens).input_tokens()?;
        let mut tokens = self
            .count_tokens(request["prompt"].as_str().context("prompt must be text")?)
            .await?;
        if let Some(memory) = request
            .get("memory")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            tokens = tokens
                .checked_add(self.count_tokens(memory).await?)
                .context("token count overflow")?;
        }
        if tokens > limit {
            bail!(
                "Промпт превышает контекст: {} токенов при бюджете {}. Используйте PromptBuilder::build_with_budget или сократите входные данные.",
                tokens,
                limit
            );
        }
        let url = format!("{}/api/v1/generate", self.base_url.trim_end_matches('/'));

        let response = self
            .http
            .post(url)
            .json(&request)
            .send()
            .await
            .context("не удалось подключиться к KoboldCpp")?
            .error_for_status()
            .context("KoboldCpp вернул HTTP-ошибку")?
            .json::<KoboldResponse>()
            .await
            .context("не удалось распарсить ответ KoboldCpp")?;

        response
            .results
            .into_iter()
            .next()
            .map(|result| result.text)
            .context("KoboldCpp вернул пустой results")
    }

    /// Generate with native KoboldCpp sampler parameters, without an ST preset.
    pub async fn generate_request(
        &self,
        request: &GenerationRequest,
    ) -> Result<GenerationResponse> {
        let mut params = request
            .parameters
            .as_object()
            .cloned()
            .context("model generation parameters must be a JSON object")?;
        params.insert("prompt".into(), Value::String(request.prompt.clone()));
        params
            .entry("max_length".to_owned())
            .or_insert(Value::from(256));
        params.insert(
            "max_context_length".into(),
            Value::Number(self.max_context_length.into()),
        );
        params.insert(
            "stop_sequence".into(),
            Value::Array(
                request
                    .stop_sequences
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
        params.insert("trim_stop".into(), Value::Bool(true));
        Ok(GenerationResponse {
            text: self.generate_json(&Value::Object(params)).await?,
        })
    }

    async fn load_max_context_length(&self) -> Result<u64> {
        let primary = self.get_value("/api/extra/true_max_context_length").await;
        let value = match primary {
            Ok(value) => value,
            Err(primary_error) => self
                .get_value("/api/v1/config/max_context_length")
                .await
                .with_context(|| {
                    format!("не удалось получить размер контекста KoboldCpp: {primary_error:#}")
                })?,
        };
        if value == 0 {
            bail!("KoboldCpp вернул нулевой размер контекста")
        }
        Ok(value)
    }

    async fn get_value(&self, path: &str) -> Result<u64> {
        #[derive(Deserialize)]
        struct ValueResponse {
            value: u64,
        }

        let url = format!("{}{path}", self.base_url.trim_end_matches('/'));
        let response = self
            .http
            .get(url)
            .send()
            .await
            .context("не удалось подключиться к KoboldCpp")?
            .error_for_status()
            .context("KoboldCpp вернул HTTP-ошибку")?
            .json::<ValueResponse>()
            .await
            .context("KoboldCpp вернул некорректный размер контекста")?;
        Ok(response.value)
    }
}

/// Backend-neutral request envelope. `parameters` is the model backend's
/// native JSON parameter object; prompt and stop sequences are kept separate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default = "empty_parameters")]
    pub parameters: Value,
}

fn empty_parameters() -> Value {
    serde_json::json!({})
}

impl GenerationRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            stop_sequences: Vec::new(),
            parameters: empty_parameters(),
        }
    }
}

/// Async generation interface that applications can implement for another
/// backend, or use through [`KoboldClient`].
pub trait ModelClient: Send + Sync {
    fn generate<'a>(
        &'a self,
        request: &'a GenerationRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GenerationResponse>> + Send + 'a>>;
}

impl ModelClient for KoboldClient {
    fn generate<'a>(
        &'a self,
        request: &'a GenerationRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GenerationResponse>> + Send + 'a>> {
        Box::pin(self.generate_request(request))
    }
}

impl TokenCounter for KoboldClient {
    fn count_tokens<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64>> + Send + 'a>> {
        Box::pin(KoboldClient::count_tokens(self, text))
    }
}
