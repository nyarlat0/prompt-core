# prompt-core

Rust library for prompt rendering, SillyTavern-compatible prompt assembly and
KoboldCpp generation. No Telegram or database dependency.

The bot uses this crate through a Cargo path dependency. To move it into another
repository, copy this entire directory: its manifest does not inherit workspace
settings or depend on files outside this directory.

## Dependencies and features

```toml
[dependencies]
prompt-core = { path = "../prompt-core" }
serde_json = "1"
anyhow = "1"
```

Enable `features = ["roleplay"]` for character cards, lorebooks and Russian
lemmatization. Chess and other applications can use the default build without
the morphology dictionary. Async HTTP operations require a Tokio runtime in
the calling application.

## Arbitrary templates and custom expressions

JSON fields are available directly as `{{fen}}` or `{{position.side}}`.
Register an expression when a placeholder needs computation. The callback
receives the current rendering's root JSON and resolved positional arguments.
It can also capture application-owned state.

```rust
use prompt_core::TemplateEngine;
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let mut templates = TemplateEngine::new();
    templates.register_expression("active_color", |data, _args| {
        data["position"]["side"].as_str()
            .map(str::to_owned)
            .ok_or_else(|| "position.side is required".into())
    });

    let rendered = templates.render(
        "Play as {{active_color}}. Board: {{fen}}",
        &json!({
            "position": { "side": "black" },
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        }),
    )?;
    println!("{rendered}");
    Ok(())
}
```

Expressions support `{{name}}` and `{{name argument}}`. Callback errors
propagate to the caller. For block helpers, named arguments, or typed
subexpressions, use `register_helper` with
`prompt_core::handlebars::HelperDef`.

Rendering does not HTML-escape the prompt. Handlebars built-ins such as
`{{#if}}` and `{{#each}}` work normally. `{{trim}}` is reserved for the
SillyTavern macro that removes adjacent line breaks. Substituted values are
not recursively evaluated as templates.

## SillyTavern prompt builder

`PromptBuilder` reuses the same expression registry across all rendering
stages. `PromptData.custom` supplies per-request application data at template
root and under `custom`. Built-in fields take precedence at template root;
the original custom fields remain available under `custom`.

```rust
use prompt_core::{
    ContextTemplate, InstructTemplate, PromptBuilder, PromptData, SystemPromptTemplate,
};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let mut builder = PromptBuilder::new();
    builder.register_expression("game_state", |data, _| {
        Ok(data["state"].as_str().unwrap_or("unknown").to_owned())
    });
    let context: ContextTemplate = serde_json::from_value(json!({
        "name": "custom", "story_string": "{{system}}\n{{game_state}}"
    }))?;
    let instruct: InstructTemplate = serde_json::from_value(json!({
        "name": "chat", "input_sequence": "<user>",
        "system_sequence": "<system>", "output_sequence": "<model>"
    }))?;
    let system: SystemPromptTemplate = serde_json::from_value(json!({
        "name": "assistant", "content": "Respond to {{user}}."
    }))?;
    let data = PromptData {
        user: "Ivan".into(),
        custom: json!({"state": "White to move"}),
        ..Default::default()
    };
    let prompt = builder.build(&context, &instruct, &system, &data)?;
    println!("{}", prompt.text);
    Ok(())
}
```

A configured `TemplateEngine` can also be passed to
`PromptBuilder::with_engine`. `BuiltPrompt` contains the final text and stop
sequences. `PromptMessage` contains role, optional name and text; persistence
and transport metadata belong to the application.

## KoboldCpp and other backends

`KoboldClient::connect` discovers the backend's maximum context length and
uses it on every request. `connect_with_client` accepts a `reqwest::Client`
configured by the caller (timeouts, proxy, TLS). The library does not print
prompts or responses; the application controls logging.

For SillyTavern sampler JSON, use `Preset::from_value` or `PresetStore::load`.
The adapter translates ST keys to KoboldCpp keys:

```rust,no_run
use prompt_core::{KoboldClient, Preset};
use serde_json::json;

async fn generate() -> anyhow::Result<String> {
    let client = KoboldClient::connect("http://localhost:5001").await?;
    let preset = Preset::from_value(json!({"genamt": 128, "temp": 0.8}))?;
    client.generate("Your prompt", &preset, &["<user>".into()]).await
}
```

For native backend parameters use `GenerationRequest` and
`KoboldClient::generate_request`. The object-safe `ModelClient` interface
allows an application to supply another backend:

```rust
use prompt_core::{GenerationRequest, ModelClient};
use serde_json::json;

async fn complete(client: &dyn ModelClient, prompt: String) -> anyhow::Result<String> {
    let mut request = GenerationRequest::new(prompt);
    // Native KoboldCpp parameter names; other adapters define their own keys.
    request.parameters = json!({"temperature": 0.8, "max_length": 128});
    request.stop_sequences.push("<user>".into());
    Ok(client.generate(&request).await?.text)
}
```

`parameters` is a JSON object. For KoboldCpp, prompt, stop sequences and the
detected maximum context length override corresponding parameter keys.

## Roleplay content (optional)

With `roleplay` enabled, `prompt_core::content` exposes character card
loading (PNG/JSON), lorebook loading and activation, and the Russian lemmatizer.
Applications select active books, participants and their order; activated
text can be passed through `PromptData.wi_before` and `wi_after`.
