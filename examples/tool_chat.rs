//! Optional live smoke example: cargo run --example tool_chat --features tool-execution -- URL
use prompt_core::{
    ChatCompletionMessage as Message, ChatCompletionRequest, KoboldClient, ToolDefinition,
    ToolExecutionOptions, ToolRegistry,
};
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://localhost:5001".into());
    let client = KoboldClient::connect(url).await?;
    let mut registry = ToolRegistry::new();
    registry.register(
        ToolDefinition::new(
            "summarize",
            "Summarize text using a separate reasoning request",
            json!({
                "type":"object", "properties":{"text":{"type":"string"}}, "required":["text"]
            }),
        ),
        |ctx, args| {
            Box::pin(async move {
                #[derive(serde::Deserialize)]
                struct Args {
                    text: String,
                }
                let args: Args = serde_json::from_value(args)?;
                let mut child = ChatCompletionRequest::new(vec![
                    Message::system("Summarize in one sentence."),
                    Message::user(args.text),
                ]);
                child.parameters = json!({"max_tokens":128,"temperature":0.2});
                child
                    .chat_template_kwargs
                    .insert("enable_thinking".into(), false.into());
                Ok(json!({"summary":ctx.generate_child(child).await?.content}))
            })
        },
    )?;
    let mut request = ChatCompletionRequest::new(vec![Message::user(
        "Call summarize for this text: Rust supports asynchronous functions. Then report its result.",
    )]);
    request.tools = registry.definitions();
    let response = registry
        .run(&client, &mut request, &ToolExecutionOptions::default())
        .await?;
    println!("{}", response.content.unwrap_or_default());
    Ok(())
}
