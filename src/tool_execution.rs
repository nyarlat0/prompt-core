//! Explicit, bounded tool execution. Requires the `tool-execution` feature.
use crate::{
    ChatCompletionMessage, ChatCompletionRequest, ChatGenerationResponse, KoboldClient, ToolCall,
    ToolChoice, ToolDefinition,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, future::Future, pin::Pin, time::Duration};

tokio::task_local! { static EXECUTING: (); }

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>>;
type Handler = dyn for<'a> Fn(ToolContext<'a>, Value) -> ToolFuture<'a> + Send + Sync;

#[derive(Clone, Copy)]
pub struct ToolContext<'a> {
    client: &'a KoboldClient,
}
impl<'a> ToolContext<'a> {
    pub fn client(self) -> &'a KoboldClient {
        self.client
    }
    /// Separate prompts/settings, same client/model. Deliberately no tools.
    pub async fn generate_child(
        self,
        mut request: ChatCompletionRequest,
    ) -> Result<ChatGenerationResponse> {
        request.disable_tools();
        self.client.generate_chat_full(&request).await
    }
}

#[derive(Debug, Clone)]
pub struct ToolExecutionOptions {
    /// Number of assistant replies containing tools. A final answer is still
    /// requested after the last allowed tool round.
    pub max_rounds: usize,
    pub max_calls: usize,
    pub handler_timeout: Duration,
    /// Bounds HTTP/tokenization as well as all handlers in this run.
    pub total_timeout: Duration,
}
impl Default for ToolExecutionOptions {
    fn default() -> Self {
        Self {
            max_rounds: 8,
            max_calls: 32,
            handler_timeout: Duration::from_secs(120),
            total_timeout: Duration::from_secs(600),
        }
    }
}

#[derive(Default)]
pub struct ToolRegistry {
    handlers: BTreeMap<String, (ToolDefinition, Box<Handler>)>,
}
impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    /// Handler validates argument types/domain constraints (e.g. using
    /// serde_json::from_value). JSON Schema is sent to the model, not executed.
    pub fn register<F>(&mut self, definition: ToolDefinition, handler: F) -> Result<()>
    where
        F: for<'a> Fn(ToolContext<'a>, Value) -> ToolFuture<'a> + Send + Sync + 'static,
    {
        definition.validate()?;
        let name = definition.function.name.clone();
        ensure!(
            !self.handlers.contains_key(&name),
            "Tool already registered: {name}"
        );
        self.handlers.insert(name, (definition, Box::new(handler)));
        Ok(())
    }
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.handlers.values().map(|(d, _)| d.clone()).collect()
    }
    fn arguments(&self, call: &ToolCall) -> Result<Value> {
        ensure!(
            self.handlers.contains_key(&call.function.name),
            "Unknown tool {} (call {})",
            call.function.name,
            call.id
        );
        let args: Value = call.arguments()?;
        ensure!(
            args.is_object(),
            "Arguments for {} must be a JSON object (call {})",
            call.function.name,
            call.id
        );
        Ok(args)
    }
    async fn invoke(
        &self,
        client: &KoboldClient,
        call: &ToolCall,
        timeout: Duration,
    ) -> Result<Value> {
        let args = self.arguments(call)?;
        let (_, handler) = &self.handlers[&call.function.name];
        tokio::time::timeout(timeout, handler(ToolContext { client }, args))
            .await
            .with_context(|| format!("Tool {} (call {}) timed out", call.function.name, call.id))?
            .with_context(|| format!("Tool {} (call {}) failed", call.function.name, call.id))
    }
    /// Explicitly execute one call. No implicit execution occurs in the client.
    pub async fn execute(
        &self,
        client: &KoboldClient,
        call: &ToolCall,
        timeout: Duration,
    ) -> Result<Value> {
        ensure!(
            EXECUTING.try_with(|_| ()).is_err(),
            "Recursive tool execution is disabled; use a tool-free child generation"
        );
        EXECUTING
            .scope((), self.invoke(client, call, timeout))
            .await
    }
    /// Appends accepted assistant messages, tool results and the final answer. On a
    /// handler error, records an error result and skipped results, then aborts.
    /// Timeout/cancellation may leave a pending exchange; never blindly retry
    /// side-effectful tools. Inspect the transcript before resuming.
    pub async fn run(
        &self,
        client: &KoboldClient,
        request: &mut ChatCompletionRequest,
        options: &ToolExecutionOptions,
    ) -> Result<ChatGenerationResponse> {
        ensure!(
            EXECUTING.try_with(|_| ()).is_err(),
            "Recursive tool execution is disabled; use a tool-free child generation"
        );
        EXECUTING
            .scope((), async {
                tokio::time::timeout(
                    options.total_timeout,
                    self.run_inner(client, request, options),
                )
                .await
                .context("Tool loop total timeout")?
            })
            .await
    }
    async fn run_inner(
        &self,
        client: &KoboldClient,
        request: &mut ChatCompletionRequest,
        options: &ToolExecutionOptions,
    ) -> Result<ChatGenerationResponse> {
        // Do not auto-enable registry definitions: the request chooses tools.
        for definition in &request.tools {
            let (registered, _) = self
                .handlers
                .get(&definition.function.name)
                .with_context(|| format!("Unknown tool {}", definition.function.name))?;
            ensure!(
                definition == registered,
                "Tool definition differs from registered handler: {}",
                definition.function.name
            );
        }
        let mut rounds = 0;
        let mut calls = 0usize;
        loop {
            let fitted = client.fit_chat(request).await?;
            let response = client.generate_chat_full(&fitted.request).await?;
            if response.tool_calls.is_empty() {
                request.messages.push(response.message.clone());
                return Ok(response);
            }
            // Check the entire batch before running any side effects.
            ensure!(
                !matches!(request.tool_choice, Some(ToolChoice::None)),
                "Tools disabled for request"
            );
            ensure!(
                rounds < options.max_rounds,
                "Tool round limit exceeded ({})",
                options.max_rounds
            );
            calls = calls
                .checked_add(response.tool_calls.len())
                .context("Tool call count overflow")?;
            ensure!(
                calls <= options.max_calls,
                "Tool call limit exceeded ({})",
                options.max_calls
            );
            let mut ids = std::collections::HashSet::new();
            for call in request
                .messages
                .iter()
                .flat_map(ChatCompletionMessage::tool_calls)
            {
                ids.insert(&call.id);
            }
            for call in &response.tool_calls {
                ensure!(ids.insert(&call.id), "Duplicate tool call ID: {}", call.id);
                ensure!(
                    request
                        .tools
                        .iter()
                        .any(|d| d.function.name == call.function.name),
                    "Unknown or unavailable tool {}",
                    call.function.name
                );
                self.arguments(call)?;
            }
            rounds += 1;
            request.messages.push(response.message.clone());
            let mut failure = None;
            for call in &response.tool_calls {
                let result = if failure.is_some() {
                    json!({"error":"Skipped after earlier tool failure"})
                } else {
                    match self.invoke(client, call, options.handler_timeout).await {
                        Ok(value) => value,
                        Err(error) => {
                            let value = json!({"error":format!("{error:#}")});
                            failure = Some(error);
                            value
                        }
                    }
                };
                request
                    .messages
                    .push(ChatCompletionMessage::tool(&call.id, &result));
            }
            if let Some(error) = failure {
                return Err(error);
            }
        }
    }
}
