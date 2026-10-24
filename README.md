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

## Chat Completion

Select the structured API with `KoboldClient::generate_chat`; existing
`generate`, `generate_request` and `ModelClient` calls remain Text Completion.
Chat requests go to `/v1/chat/completions` and decode
`choices[0].message.content`. This is the KoboldCpp chat adapter, not a generic
client for arbitrary OpenAI-compatible servers: context discovery and chat
tokenization require KoboldCpp endpoints.

`PromptBuilder::build_chat` renders story fields and custom expressions through
the same pipeline as the text builder, then returns `ChatMessage` values with
lowercase `system`, `user`, and `assistant` wire roles. The top story is
system content; in-chat story injections retain their role and depth.
Group history keeps only the active character's turns as assistant, merging
other turns with double newlines. Ephemeral `user_prompt` and system
post-history instructions preserve their positions.

Local instruct tokens and assistant prefixes are not included in message
contents. The server applies its configured model chat template exactly once.
Message names are not sent as API `name` fields or prepended to the content.
A text-only `context.chat_start` is rejected; use a context without it.
Instruct-derived and context-derived text stop strings are not inferred for
chat: supply `ChatGenerationRequest.stop_sequences` explicitly. Story-string
contents should contain semantic text, not manually embedded instruct tags.

```rust,no_run
use prompt_core::{
    ChatGenerationRequest, ContextTemplate, KoboldClient, Preset,
    PromptBuilder, PromptData, SystemPromptTemplate,
};

async fn chat(
    client: &KoboldClient,
    builder: &PromptBuilder,
    context: &ContextTemplate,
    system: &SystemPromptTemplate,
    data: &PromptData,
    preset: &Preset,
) -> anyhow::Result<String> {
    // The fitter fills messages from PromptData. Set native sampling parameters
    // or chat_template_kwargs on options.parameters if needed.
    let options = ChatGenerationRequest::from_preset(Vec::new(), preset)?;
    let fitted = builder.build_chat_with_budget(
        context, system, data, client, &options, client.context_budget(preset)?,
    ).await?;
    println!("{}", fitted.preview.prompt);
    Ok(client.generate_chat(&fitted.request).await?.text)
}
```

For custom applications, use `ChatGenerationRequest::new(messages)` without
the ST builder. Parameters use chat endpoint names such as `max_tokens`,
`temperature`, `model`, `adapter`, and `chat_template_kwargs`.
`from_preset` translates ST samplers, including the response limit and
Mirostat naming. The adapter requests one non-streaming text response and a
new assistant turn, rather than continuation of the last historical assistant
message. Conflicting raw prompt/length fields are rejected. For multimodal
content, convert to `ChatCompletionRequest` after building the text messages.
Tools use the extended structured API described below; raw parameter overrides
for tools remain forbidden.

`preview_chat` sends the same messages and settings to the backend's
`/api/extra/tokencount` renderer and returns its final prompt and token count.
Budget fitting uses that count, including server template overhead, while
preserving mandatory blocks and the latest conversational message. Generation
performs the same preflight again. Missing preview support, empty/invalid
counts, overflow, or dropped/changed/reordered non-whitespace message content
produce an error before generation. There is no text-mode fallback.

The server must have a suitable model chat template configured. Role
boundaries are supplied as structured messages, but exact token delimiters
and assistant prefixes belong to that server template and need not match a
local `InstructTemplate`. The content-order check detects common lossy
templates; it does not prove semantic equivalence of arbitrary server
templates or guard against a model/template being replaced between requests.
Use `preview_chat` or `FittedChatPrompt.preview` to inspect the actual input.
Never put `BuiltPrompt.text` into a single chat message: doing so applies
chat formatting twice.

## Roleplay content (optional)

With `roleplay` enabled, `prompt_core::content` exposes character card
loading (PNG/JSON), lorebook loading and activation, and the Russian lemmatizer.
Applications select active books, participants and their order; activated
text can be passed through `PromptData.wi_before` and `wi_after`.


## Tool calling

The protocol types and `KoboldClient::generate_chat_full` are always available.
For the optional Rust executor, enable `features = ["tool-execution"]` and use a
Tokio runtime with its time driver enabled. No agent framework or MCP dependency
is required. Existing `ChatMessage` and `ChatGenerationRequest` struct literals
remain source-compatible. `ChatGenerationRequest::with_tools` converts to
`ChatCompletionRequest`; `From<ChatGenerationRequest>` also preserves presets,
stops and generation parameters. Text Completion is unchanged.

`ChatCompletionMessage` adds assistant `tool_calls`, optional text/reasoning and
`role: "tool"` results linked by ID. `ChatGenerationResponse` exposes `message`,
`content`, `reasoning_content`, `finish_reason`, and `tool_calls`. The original
assistant message, including unknown extension fields, can be replayed. Missing
finish reasons on older backends remain `None`. Null content with calls is valid.
`ToolCall::arguments::<T>()` parses the original argument string with contextual
errors; decoding the response itself never requires valid argument JSON.
`generate_chat` still returns text, but errors on tool calls instead of silently
turning them into text. Receiving a call never runs a function automatically.

```rust,no_run
use prompt_core::{
    ChatCompletionMessage as Message, ChatGenerationRequest, ChatMessage,
    ChatRole, KoboldClient, ToolChoice, ToolDefinition,
};
use serde_json::{Value, json};

async fn manual(client: &KoboldClient) -> anyhow::Result<()> {
    let tool = ToolDefinition::new("normalize", "Normalize supplied text", json!({
        "type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"], "additionalProperties": false
    }));
    let mut request = ChatGenerationRequest::new(vec![ChatMessage {
        role: ChatRole::User, content: "Normalize HELLO".into(),
    }]).with_tools(vec![tool]);
    request.tool_choice = Some(ToolChoice::Auto);
    request.chat_template_kwargs.insert("enable_thinking".into(), false.into());
    request.parameters["max_tokens"] = json!(256);

    let response = client.generate_chat_full(&request).await?;
    // Store the whole assistant message, including all call IDs and reasoning.
    request.messages.push(response.message);
    for call in response.tool_calls {
        anyhow::ensure!(call.function.name == "normalize", "Unknown tool");
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Args { text: String }
        let args: Args = call.arguments()?;
        let result: Value = json!({"text": args.text.to_lowercase()});
        request.messages.push(Message::tool(call.id, &result));
    }
    // Only continue after supplying ALL results. For a bounded automatic
    // continuation, use ToolRegistry below.
    Ok(())
}
```

`ToolChoice` supports `Auto`, `None`, `Required`, and `Function(name)` using the
OpenAI wire format. Required/forced choices are checked against the response;
servers that ignore them cause an error. `request.disable_tools()` removes tool
definitions and sets `none`. The typed `chat_template_kwargs` map supports
`enable_thinking` and model-specific options. Legacy parameters can still carry
that map, but setting it in both places is an error. Overrides of messages,
tools, generation-prefix controls and other reserved fields are rejected.

### Rust handlers and child generation

```rust,no_run
# #[cfg(feature = "tool-execution")]
# async fn example(client: &prompt_core::KoboldClient) -> anyhow::Result<()> {
use prompt_core::{
    ChatCompletionMessage as Message, ChatCompletionRequest, ToolDefinition,
    ToolExecutionOptions, ToolRegistry,
};
use serde_json::json;
use std::time::Duration;

let mut registry = ToolRegistry::new();
registry.register(
    ToolDefinition::new("summarize", "Summarize the supplied text", json!({
        "type":"object", "properties":{"text":{"type":"string"}},
        "required":["text"]
    })),
    |ctx, arguments| Box::pin(async move {
        #[derive(serde::Deserialize)]
        struct Args { text: String }
        let args: Args = serde_json::from_value(arguments)?;
        let mut child = ChatCompletionRequest::new(vec![
            Message::system("Give a one-sentence summary."),
            Message::user(args.text),
        ]);
        child.parameters = json!({"temperature":0.2, "max_tokens":128});
        child.chat_template_kwargs.insert("enable_thinking".into(), false.into());
        // Same KoboldClient, same loaded model; separate messages/settings.
        // This helper always disables child tools, even if provided by mistake.
        let answer = ctx.generate_child(child).await?;
        Ok(json!({"summary": answer.content}))
    }),
)?;

let mut request = ChatCompletionRequest::new(vec![
    Message::user("Use summarize to summarize: Rust supports async functions."),
]);
request.tools = registry.definitions();
let options = ToolExecutionOptions {
    max_rounds: 4, max_calls: 8,
    handler_timeout: Duration::from_secs(120),
    total_timeout: Duration::from_secs(600),
};
let answer = registry.run(client, &mut request, &options).await?;
println!("{}", answer.content.unwrap_or_default());
// request.messages now includes calls, results and the final assistant reply.
# Ok(())
# }
```

Register ordinary functions in the same way: return `Ok(json!(my_function(...)))`
inside the async handler. Arguments must be JSON objects. Handlers validate types
and application constraints with Serde or a validator of the application's
choice; the library does **not** implement a JSON Schema validation engine.
`registry.execute(client, &call, timeout).await` explicitly executes a single
registered call and returns its JSON result. It does not modify history.

`run` uses only tools listed in the request, verifies their registered definitions,
prechecks each batch, runs calls sequentially, and supplies one result per ID in
response order. HTTP calls hold no client mutex, so a handler can reuse
`ctx.client()` for custom child requests (including Text Completion) without
reentrant deadlock. Callers own scheduling of independent concurrent runs on the
same server. `generate_child` provides the safe tool-free default.

Limits apply per run. An extra tool round beyond `max_rounds`, or a batch exceeding
`max_calls`, fails before executing that batch. Nested executor invocation on the
same Tokio task is rejected; child HTTP generation is allowed. Application code
must not bypass this guard by spawning unbounded independent executor tasks.
Handlers must be cooperative async code: use appropriate isolation for blocking
work; Tokio timeouts cannot interrupt a synchronous function blocking its thread.

Unknown tools, invalid JSON, handler failures and timeouts are errors, not model
text. Handler failures append an error result and skipped results for the rest of
the batch, then abort without another generation. A total timeout/cancellation
can leave a pending exchange in the caller's transcript. Inspect it before
resuming; incomplete exchanges are rejected before HTTP, and side effects are
never automatically retried or rolled back. Handler panics are application bugs
and follow the application's normal panic policy.

### Context and backend requirements

For tools, build initial semantic messages with `PromptBuilder::build_chat`,
convert with `ChatGenerationRequest::with_tools`, then call
`client.fit_chat(&request).await?`. The older `build_chat_with_budget` remains for
legacy tool-free requests. Do not budget first and then add uncounted schemas or
results. `ToolRegistry::run` calls `fit_chat` before every generation, leaving the
full caller transcript intact. Use `FittedChatCompletion.preview` to inspect the
exact input and token count sent for that turn.

Both `preview_chat` and `generate_chat_full` accept old or extended requests.
Every candidate includes tools, calls, results and template kwargs in
`/api/extra/tokencount`; generation repeats the exact server preflight. No local
approximate tokenizer is used. `fit_chat` preserves all system messages and the
latest conversational unit. An assistant with calls and its complete contiguous
set of results is one indivisible unit. Missing, duplicated or orphaned IDs fail
validation. If mandatory content cannot fit, generation fails.

Run a tool-capable model on KoboldCpp with `--jinja --jinjatools` (aliases may
vary by version), using its appropriate tool-aware Jinja template. See the
[official flags documentation](https://koboldcpp.net/docs/reference/flags/)
and [KoboldCpp source](https://github.com/LostRuins/koboldcpp/blob/concedo/koboldcpp.py).
Older backends/templates may ignore tools or format them through a legacy adapter.
The preview must return the complete rendered prompt and its token count.

The library no longer inspects the rendered Jinja prompt to confirm that
tool definitions, arguments, reasoning or individual message contents appear
as literal substrings. That format-dependent check incorrectly rejected valid
Qwen XML-style tool calls and other backend-specific templates. The backend
owns its template. Its token count, mandatory context budget, structured
tool-call parsing, tool-call IDs, and transcript integrity checks remain.
Reasoning fields are preserved on the wire; their rendering is controlled
by the server template. Ensure your selected backend template genuinely
supports tools, because the library does not attempt to verify its text.
Do not configure server-side parameter overrides that replace the request
or switch the loaded model during a run.

An optional live example (not run by tests):

```sh
cargo run -p prompt-core --example tool_chat --features tool-execution -- http://localhost:5001
```

Standard tests use local mock HTTP servers and require no model process.

## Multimodal Chat Completion

`PromptBuilder`, `PromptData`, `PromptMessage` and `ChatMessage` remain text-only.
Existing string constructors and text request JSON are unchanged. Attach images
at the wire-request boundary, after rendering the text template:

```rust
use prompt_core::{ChatCompletionMessage, ChatCompletionRequest, ContentPart};
let request = ChatCompletionRequest::new(vec![
    ChatCompletionMessage::user_parts(vec![
        ContentPart::text("Describe this board."),
        ContentPart::image_url("data:image/png;base64,..."),
    ]),
]);
```

`ChatContent` serializes as a string or an ordered array of tagged `text` /
`image_url` parts. `ImageUrl` accepts an unchanged URL/data URI and optional
`detail: auto|low|high`. PNG data URIs use standard base64. There is no implicit
download, image resizing, or base64 tokenization. HTTP(S) image URLs are represented
on the OpenAI wire, but backend support varies: KoboldCpp's current adapter uses
data URIs, so applications should use those.

Text constructors (`system`, `user`, `assistant`, `tool`) remain compatible;
direct rich-message struct literals now use `ChatContent` (`"text".into()`).
`content()` borrows a plain string, while `text_parts()` accesses textual parts
without discarding the raw multimodal message. Response `content` concatenates
text parts in order; `message` retains the original parts, `reasoning_content`,
tool calls and extension fields. Image-only replies are not mistaken for text.

Use the same `client.fit_chat(&request)` and `generate_chat_full(&fitted.request)`
as tools/reasoning. `generate_chat` also accepts the rich request for text-only
responses. Tools, template kwargs and reasoning options travel unchanged with
images. Trimming removes entire messages/exchanges, never individual image
parts; the latest conversational unit and system messages remain mandatory.

**Vision context accounting:** KoboldCpp's `tokencount` renders image markers but
does not count vision embeddings. `image_token_reserve` defaults to **4096 per
image**, added to every preview's token count, both during fitting and the final
preflight check. It is local accounting, not an API generation parameter. Configure
it for the actual model/encoder/resolution; 4096 is a conservative starting value,
not a universal upper bound or exact count. Base64 bytes are never counted as text.
If the mandatory image+text+output budget cannot fit, generation fails, rather
than silently removing the image. The server must load a vision-capable model
and its matching multimodal projector. A successful HTTP request alone cannot
prove the model actually understood the picture.

Tests cover wire roundtrips, text compatibility, response parts, tools/reasoning
coexistence, context trimming and mandatory-image overflow using a local Kobold
protocol server. Optional live smoke test:

```sh
KOBOLD_URL=http://aistation:5001 cargo test -p prompt-core --test multimodal live_kobold_png -- --ignored
```
