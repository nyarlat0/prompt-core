use prompt_core::*;
use serde_json::{Value, json};
mod support;
use support::server;

fn definition() -> ToolDefinition {
    ToolDefinition::new(
        "lookup",
        "Look up a value",
        json!({"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}),
    )
}
fn call(id: &str) -> ToolCall {
    serde_json::from_value(json!({"id":id,"type":"function","function":{"name":"lookup","arguments":"{\"key\":\"value\"}"}})).unwrap()
}
fn assistant(calls: Vec<ToolCall>) -> ChatCompletionMessage {
    serde_json::from_value(json!({"role":"assistant","content":null,"reasoning_content":"private reasoning","tool_calls":calls,"vendor_field":"preserved"})).unwrap()
}
fn request() -> ChatCompletionRequest {
    let mut r = ChatCompletionRequest::new(vec![ChatCompletionMessage::user("QUESTION")]);
    r.tools = vec![definition()];
    r.parameters = json!({"max_tokens":16});
    r
}
fn preview(r: &ChatCompletionRequest, tokens: u64) -> Value {
    // Simulate Jinja JSON formatting, deliberately different whitespace.
    let mut prompt = serde_json::to_string_pretty(&r.tools).unwrap();
    for m in &r.messages {
        prompt.push_str("<turn>");
        prompt.push_str(m.content().unwrap_or_default());
        for c in m.tool_calls() {
            prompt.push_str(&c.function.name);
            prompt
                .push_str(&serde_json::to_string_pretty(&c.arguments::<Value>().unwrap()).unwrap());
        }
    }
    prompt.push_str("<assistant>");
    json!({"prompt":prompt,"value":tokens})
}
fn reply(message: ChatCompletionMessage, reason: &str) -> Value {
    json!({"choices":[{"message":message,"finish_reason":reason}]})
}
fn connect_step() -> (&'static str, Value, Value) {
    (
        "/api/extra/true_max_context_length",
        json!({}),
        json!({"value":1024}),
    )
}
async fn client(url: String) -> KoboldClient {
    KoboldClient::connect_with_client(url, reqwest::Client::builder().no_proxy().build().unwrap())
        .await
        .unwrap()
}

#[test]
fn wire_choices_raw_arguments_and_conversion() {
    for (choice, wire) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::None, json!("none")),
        (ToolChoice::Required, json!("required")),
        (
            ToolChoice::Function("lookup".into()),
            json!({"type":"function","function":{"name":"lookup"}}),
        ),
    ] {
        assert_eq!(serde_json::to_value(&choice).unwrap(), wire);
        assert_eq!(serde_json::from_value::<ToolChoice>(wire).unwrap(), choice);
    }
    let mut c = call("a");
    c.function.arguments = "{broken".into();
    assert!(
        c.arguments::<Value>()
            .unwrap_err()
            .to_string()
            .contains("call a")
    );
    assert_eq!(
        serde_json::to_value(c).unwrap()["function"]["arguments"],
        "{broken"
    );
    let old = ChatGenerationRequest::new(vec![ChatMessage {
        role: ChatRole::User,
        content: "hello".into(),
    }]);
    let new = old.with_tools(vec![definition()]);
    assert_eq!(new.messages[0], ChatCompletionMessage::user("hello"));
}

#[tokio::test]
async fn full_response_and_wire_payload() {
    let mut r = request();
    r.tool_choice = Some(ToolChoice::Required);
    r.chat_template_kwargs
        .insert("enable_thinking".into(), false.into());
    let expected = json!({"messages":r.messages,"tools":r.tools,"tool_choice":"required","chat_template_kwargs":{"enable_thinking":false},"max_tokens":16,"stream":false,"n":1});
    let m = assistant(vec![call("a"), call("b")]);
    let (url, h) = server(vec![
        connect_step(),
        ("/api/extra/tokencount", expected.clone(), preview(&r, 30)),
        (
            "/v1/chat/completions",
            expected,
            reply(m.clone(), "tool_calls"),
        ),
    ]);
    let response = client(url).await.generate_chat_full(&r).await.unwrap();
    assert_eq!(response.message, m);
    assert_eq!(response.content, None);
    assert_eq!(
        response.reasoning_content.as_deref(),
        Some("private reasoning")
    );
    assert_eq!(response.tool_calls.len(), 2);
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    h.join().unwrap();
}

#[tokio::test]
async fn legacy_text_wrapper_does_not_flatten_calls() {
    let r = ChatGenerationRequest::new(vec![ChatMessage {
        role: ChatRole::User,
        content: "QUESTION".into(),
    }]);
    let (url, h) = server(vec![
        connect_step(),
        (
            "/api/extra/tokencount",
            json!({}),
            json!({"prompt":"QUESTION","value":10}),
        ),
        (
            "/v1/chat/completions",
            json!({}),
            reply(assistant(vec![call("a")]), "tool_calls"),
        ),
    ]);
    assert!(
        client(url)
            .await
            .generate_chat(&r)
            .await
            .unwrap_err()
            .to_string()
            .contains("generate_chat_full")
    );
    h.join().unwrap();
}

#[tokio::test]
async fn budget_preserves_atomic_exchange_systems_and_counts_tools_and_results() {
    let mut r = request();
    r.messages = vec![
        ChatCompletionMessage::system("RULES"),
        assistant(vec![call("a"), call("b")]),
        ChatCompletionMessage::tool("a", &json!({"large":"OLD"})),
        ChatCompletionMessage::tool("b", &json!("OLD2")),
        ChatCompletionMessage::user("QUESTION"),
        ChatCompletionMessage::system("POST"),
    ];
    let mut trimmed = r.clone();
    trimmed.messages.drain(1..4);
    let (url, h) = server(vec![
        connect_step(),
        (
            "/api/extra/tokencount",
            json!({"tools":r.tools,"messages":r.messages}),
            preview(&r, 1100),
        ),
        (
            "/api/extra/tokencount",
            json!({"tools":trimmed.tools,"messages":trimmed.messages}),
            preview(&trimmed, 100),
        ),
    ]);
    let fit = client(url).await.fit_chat(&r).await.unwrap();
    assert_eq!(fit.dropped_messages, 3);
    assert_eq!(fit.request.messages, trimmed.messages);
    assert_eq!(r.messages.len(), 6);
    h.join().unwrap();
}

#[tokio::test]
async fn schema_and_latest_results_cannot_be_trimmed_to_force_fit() {
    let mut r = request();
    r.messages.extend([
        assistant(vec![call("a")]),
        ChatCompletionMessage::tool("a", &json!("LARGE")),
    ]);
    let mut trimmed = r.clone();
    trimmed.messages.remove(0);
    let (url, h) = server(vec![
        connect_step(),
        (
            "/api/extra/tokencount",
            json!({"tools":r.tools}),
            preview(&r, 1200),
        ),
        (
            "/api/extra/tokencount",
            json!({"messages":trimmed.messages,"tools":r.tools}),
            preview(&trimmed, 1100),
        ),
    ]);
    assert!(
        client(url)
            .await
            .fit_chat(&r)
            .await
            .unwrap_err()
            .to_string()
            .contains("cannot fit")
    );
    h.join().unwrap();
}

#[tokio::test]
async fn rejects_dangling_exchanges_and_conflicts() {
    let (url, h) = server(vec![
        connect_step(),
        (
            "/api/extra/tokencount",
            json!({}),
            json!({"prompt":"QUESTION","value":10}),
        ),
    ]);
    let c = client(url).await;
    for messages in [
        vec![ChatCompletionMessage::tool("a", &json!(1))],
        vec![assistant(vec![call("a")])],
        vec![
            assistant(vec![call("a")]),
            ChatCompletionMessage::tool("wrong", &json!(1)),
        ],
    ] {
        let r = ChatCompletionRequest::new(messages);
        assert!(c.preview_chat(&r).await.is_err());
    }
    for key in [
        "tools",
        "tool_choice",
        "messages",
        "prompt",
        "memory",
        "max_context_length",
        "stream",
    ] {
        let mut r = request();
        r.parameters[key] = json!({});
        assert!(
            c.preview_chat(&r)
                .await
                .unwrap_err()
                .to_string()
                .contains("reserved")
        );
    }
    let mut r = request();
    r.chat_template_kwargs.insert("messages".into(), json!([]));
    assert!(c.preview_chat(&r).await.is_err());
    // The backend's own tokenizer owns Jinja rendering. A missing tool
    // definition in the textual preview is no longer a fatal failure.
    c.preview_chat(&request()).await.unwrap();
    h.join().unwrap();
}

#[tokio::test]
async fn malformed_responses_are_errors_not_panics() {
    for response in [
        json!({"choices":[]}),
        json!({"choices":[{"message":{"role":"user","content":"bad"}}]}),
        json!({"choices":[{"message":{"role":"assistant","content":null}}]}),
        json!({"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"invalid"}]}}]}),
    ] {
        let r = request();
        let (url, h) = server(vec![
            connect_step(),
            ("/api/extra/tokencount", json!({}), preview(&r, 10)),
            ("/v1/chat/completions", json!({}), response),
        ]);
        assert!(client(url).await.generate_chat_full(&r).await.is_err());
        h.join().unwrap();
    }
}

#[cfg(feature = "tool-execution")]
mod execution {
    use super::*;
    use std::time::Duration;
    fn generation(
        steps: &mut Vec<(&'static str, Value, Value)>,
        r: &ChatCompletionRequest,
        m: ChatCompletionMessage,
        reason: &str,
        fit: bool,
    ) {
        let expected = json!({"messages":r.messages,"tools":if r.tools.is_empty(){Value::Null}else{json!(r.tools)},"tool_choice":r.tool_choice,"chat_template_kwargs":if r.chat_template_kwargs.is_empty(){Value::Null}else{json!(r.chat_template_kwargs)}});
        if fit {
            steps.push(("/api/extra/tokencount", expected.clone(), preview(r, 50)));
        }
        steps.push(("/api/extra/tokencount", expected.clone(), preview(r, 50)));
        steps.push(("/v1/chat/completions", expected, reply(m, reason)));
    }
    #[tokio::test]
    async fn multiple_tools_nested_same_client_and_final_answer() {
        let mut registry = ToolRegistry::new();
        registry
            .register(definition(), |ctx, args| {
                Box::pin(async move {
                    assert_eq!(args["key"], "value");
                    let mut child = ChatCompletionRequest::new(vec![
                        ChatCompletionMessage::system("CHILD"),
                        ChatCompletionMessage::user("SUBQUESTION"),
                    ]);
                    child.parameters = json!({"max_tokens":8,"temperature":0.1});
                    child
                        .chat_template_kwargs
                        .insert("enable_thinking".into(), false.into());
                    let response = ctx.generate_child(child).await?;
                    Ok(json!({"answer":response.content}))
                })
            })
            .unwrap();
        let mut r = request();
        let m = assistant(vec![call("a"), call("b")]);
        let mut steps = vec![connect_step()];
        generation(&mut steps, &r, m.clone(), "tool_calls", true);
        let mut child = ChatCompletionRequest::new(vec![
            ChatCompletionMessage::system("CHILD"),
            ChatCompletionMessage::user("SUBQUESTION"),
        ]);
        child.parameters = json!({"max_tokens":8,"temperature":0.1});
        child
            .chat_template_kwargs
            .insert("enable_thinking".into(), false.into());
        child.disable_tools();
        generation(
            &mut steps,
            &child,
            ChatCompletionMessage::assistant("analysis a"),
            "stop",
            false,
        );
        generation(
            &mut steps,
            &child,
            ChatCompletionMessage::assistant("analysis b"),
            "stop",
            false,
        );
        let mut continued = r.clone();
        continued.messages.extend([
            m.clone(),
            ChatCompletionMessage::tool("a", &json!({"answer":"analysis a"})),
            ChatCompletionMessage::tool("b", &json!({"answer":"analysis b"})),
        ]);
        generation(
            &mut steps,
            &continued,
            ChatCompletionMessage::assistant("FINAL"),
            "stop",
            true,
        );
        let (url, h) = server(steps);
        let c = client(url).await;
        let result = registry
            .run(&c, &mut r, &ToolExecutionOptions::default())
            .await
            .unwrap();
        assert_eq!(result.finish_reason.as_deref(), Some("stop"));
        assert_eq!(result.content.as_deref(), Some("FINAL"));
        assert_eq!(&r.messages[..4], &continued.messages);
        assert_eq!(r.messages[1], m);
        h.join().unwrap();
    }
    #[tokio::test]
    async fn errors_timeouts_and_recursion_are_explicit() {
        let (url, h) = server(vec![connect_step()]);
        let c = client(url).await;
        let mut registry = ToolRegistry::new();
        registry
            .register(definition(), |_, _| {
                Box::pin(async { std::future::pending::<anyhow::Result<Value>>().await })
            })
            .unwrap();
        let mut bad = call("a");
        bad.function.name = "unknown".into();
        assert!(
            registry
                .execute(&c, &bad, Duration::from_millis(10))
                .await
                .unwrap_err()
                .to_string()
                .contains("Unknown")
        );
        bad = call("a");
        bad.function.arguments = "broken".into();
        assert!(
            registry
                .execute(&c, &bad, Duration::from_millis(10))
                .await
                .unwrap_err()
                .to_string()
                .contains("arguments")
        );
        assert!(
            registry
                .execute(&c, &call("a"), Duration::from_millis(10))
                .await
                .unwrap_err()
                .to_string()
                .contains("timed out")
        );
        let mut recursive = ToolRegistry::new();
        recursive
            .register(definition(), |ctx, _| {
                Box::pin(async move {
                    ToolRegistry::new()
                        .execute(ctx.client(), &call("nested"), Duration::from_secs(1))
                        .await
                })
            })
            .unwrap();
        let err = recursive
            .execute(&c, &call("a"), Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("Recursive"));
        h.join().unwrap();
    }
    #[tokio::test]
    async fn round_and_call_limits_stop_without_executing_excess_calls() {
        for max_calls in [0, 10] {
            let mut registry = ToolRegistry::new();
            registry
                .register(definition(), |_, _| Box::pin(async { Ok(json!("done")) }))
                .unwrap();
            let mut r = request();
            let mut steps = vec![connect_step()];
            generation(
                &mut steps,
                &r,
                assistant(vec![call("a")]),
                "tool_calls",
                true,
            );
            if max_calls > 0 {
                let mut next = r.clone();
                next.messages.extend([
                    assistant(vec![call("a")]),
                    ChatCompletionMessage::tool("a", &json!("done")),
                ]);
                generation(
                    &mut steps,
                    &next,
                    assistant(vec![call("b")]),
                    "tool_calls",
                    true,
                );
            }
            let (url, h) = server(steps);
            let c = client(url).await;
            let options = ToolExecutionOptions {
                max_rounds: 1,
                max_calls,
                ..Default::default()
            };
            assert!(
                registry
                    .run(&c, &mut r, &options)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("limit")
            );
            h.join().unwrap();
        }
    }
    #[tokio::test]
    async fn disabled_tools_and_handler_error_transcript() {
        let mut registry = ToolRegistry::new();
        registry
            .register(definition(), |_, _| {
                Box::pin(async { anyhow::bail!("handler failure") })
            })
            .unwrap();
        let mut disabled = request();
        disabled.disable_tools();
        let mut steps = vec![connect_step()];
        generation(
            &mut steps,
            &disabled,
            ChatCompletionMessage::assistant("TEXT"),
            "stop",
            true,
        );
        let (url, h) = server(steps);
        assert_eq!(
            registry
                .run(
                    &client(url).await,
                    &mut disabled,
                    &ToolExecutionOptions::default()
                )
                .await
                .unwrap()
                .content
                .as_deref(),
            Some("TEXT")
        );
        h.join().unwrap();
        let mut r = request();
        let mut steps = vec![connect_step()];
        generation(
            &mut steps,
            &r,
            assistant(vec![call("a"), call("b")]),
            "tool_calls",
            true,
        );
        let (url, h) = server(steps);
        let error = registry
            .run(&client(url).await, &mut r, &ToolExecutionOptions::default())
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("handler failure"));
        assert_eq!(r.messages.len(), 4);
        assert!(r.messages[2].content().unwrap().contains("handler failure"));
        assert!(r.messages[3].content().unwrap().contains("Skipped"));
        h.join().unwrap();
    }
}

#[tokio::test]
async fn malformed_argument_string_survives_http_decoding() {
    let r = request();
    let mut broken = call("bad-json");
    broken.function.arguments = "{\"key\": unfinished".into();
    let (url, h) = server(vec![
        connect_step(),
        ("/api/extra/tokencount", json!({}), preview(&r, 10)),
        (
            "/v1/chat/completions",
            json!({}),
            reply(assistant(vec![broken.clone()]), "tool_calls"),
        ),
    ]);
    let response = client(url).await.generate_chat_full(&r).await.unwrap();
    assert_eq!(response.tool_calls[0], broken);
    assert!(response.tool_calls[0].arguments::<Value>().is_err());
    h.join().unwrap();
}

#[tokio::test]
async fn forced_choice_and_disabled_choice_cannot_be_ignored() {
    for choice in [
        ToolChoice::Required,
        ToolChoice::Function("lookup".into()),
        ToolChoice::None,
    ] {
        let mut r = request();
        let m = if choice == ToolChoice::None {
            assistant(vec![call("a")])
        } else {
            ChatCompletionMessage::assistant("ignored")
        };
        r.tool_choice = Some(choice);
        let (url, h) = server(vec![
            connect_step(),
            (
                "/api/extra/tokencount",
                json!({"tool_choice":r.tool_choice}),
                preview(&r, 10),
            ),
            (
                "/v1/chat/completions",
                json!({"tool_choice":r.tool_choice}),
                reply(m, "stop"),
            ),
        ]);
        assert!(client(url).await.generate_chat_full(&r).await.is_err());
        h.join().unwrap();
    }
}

#[tokio::test]
async fn full_generation_checks_budget_but_does_not_parse_jinja_tool_transcript() {
    let mut r = request();
    r.messages.extend([
        assistant(vec![call("a")]),
        ChatCompletionMessage::tool("a", &json!("RESULT")),
    ]);
    // Finite model context remains an actual backend constraint.
    let (url, h) = server(vec![
        connect_step(),
        (
            "/api/extra/tokencount",
            json!({"tools":r.tools,"messages":r.messages}),
            preview(&r, 1001),
        ),
    ]);
    assert!(client(url).await.generate_chat_full(&r).await.is_err());
    h.join().unwrap();

    // Even a lossy or custom Jinja template should not be second-guessed
    // by the library: the backend and model decide how to render its calls.
    let mut nonstandard = preview(&r, 40);
    nonstandard["prompt"] = json!("QUESTION\n<assistant tool=lookup><arg omitted/>\nRESULT");
    let (url, h) = server(vec![
        connect_step(),
        (
            "/api/extra/tokencount",
            json!({"tools":r.tools,"messages":r.messages}),
            nonstandard.clone(),
        ),
        (
            "/v1/chat/completions",
            json!({"tools":r.tools,"messages":r.messages}),
            reply(ChatCompletionMessage::assistant("FINAL"), "stop"),
        ),
    ]);
    assert_eq!(client(url).await.generate_chat_full(&r).await.unwrap().content.as_deref(),Some("FINAL"));
    h.join().unwrap();
}

#[test]
fn absent_calls_can_be_null() {
    let m: ChatCompletionMessage =
        serde_json::from_value(json!({"role":"assistant","content":"ok","tool_calls":null}))
            .unwrap();
    assert!(m.tool_calls().is_empty());
    assert_eq!(m.content(), Some("ok"));
}
