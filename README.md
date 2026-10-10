# prompt-core

Rust library for prompt rendering, SillyTavern-compatible prompt assembly and
KoboldCpp generation with model-aware context budgeting.

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

System prompt JSON may also contain `user_prompt` (default: empty). It uses
the same template environment as `content` and `post_history`, including custom
directives. Nonempty rendered text becomes a final User message after history
and before the system `post_history`. It is ephemeral, is included in token
budgeting, and remains present when old history messages are dropped.

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

## Context management

Use `PromptBuilder::build_with_budget` to fit conversation history into the
model context before generation. It returns a `FittedPrompt` containing the
exact prompt, token count, input budget and number of omitted messages.

```rust,no_run
use prompt_core::{
    ContextTemplate, FittedPrompt, InstructTemplate, KoboldClient,
    Preset, PromptBuilder, PromptData, SystemPromptTemplate,
};

async fn prepare(
    client: &KoboldClient,
    builder: &PromptBuilder,
    context: &ContextTemplate,
    instruct: &InstructTemplate,
    system: &SystemPromptTemplate,
    data: &PromptData,
    preset: &Preset,
) -> anyhow::Result<FittedPrompt> {
    builder.build_with_budget(
        context, instruct, system, data, client, client.context_budget(preset)?,
    ).await
}
```

The input budget is `context_tokens - response_tokens - safety_tokens`.
`ContextBudget::new` reserves 8 safety tokens by default. The response reserve
matches the requested generation length: ST `genamt` or native `max_length`,
defaulting explicitly to 256 if absent; zero and invalid values are rejected.

The fitter removes oldest non-system messages as whole messages, preserving
the latest conversational message and every system message. Description,
scenario, personas, activated lore, story string and post-history instructions
remain intact. Each candidate is rebuilt, including grouped turns and story
injection positions, then tokenized as a complete string. The original
`PromptData` and its full history are never modified. Lore activation is
supplied by the caller and is not recalculated after trimming.

After an initial full-prompt check, a binary search over history suffixes
limits tokenizer round trips. Only an actually measured fitting prompt is
returned. Custom expressions should be deterministic while fitting; unusual
templates with non-monotonic sizes may retain less history than theoretically
possible. If mandatory content plus the latest message does not fit, fitting
returns an error with token counts instead of cutting instructions or message
text. Reducing the response limit or mandatory content is the caller's choice.

`TokenCounter` is backend-independent and can be implemented for other model
providers. `KoboldClient` uses its loaded model's
[`/api/extra/tokencount`](https://github.com/LostRuins/koboldcpp/wiki)
endpoint with special tokens enabled. Tokenizer failures abort the request;
there is no character-count fallback.

Direct `generate` and `generate_request` calls also validate the final input
against the budget before contacting the generation endpoint. They reject
oversized raw prompts, since an opaque string has no safe message boundaries.
Use the fitter for automatic history reduction. Native KoboldCpp `memory`,
when supplied, is counted separately and included in the input budget.
The library does not automatically retry a rejected generation or modify
persistent conversation storage.

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
