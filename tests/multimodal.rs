use prompt_core::*;
use serde_json::json;
mod support;

// Valid 1x1 PNG, deliberately small for protocol tests.
const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=";

fn message() -> ChatCompletionMessage {
    ChatCompletionMessage::user_parts(vec![
        ContentPart::text("Describe"),
        ContentPart::image_url(PNG),
        ContentPart::text("briefly"),
    ])
}

#[test]
fn wire_roundtrip_and_text_compatibility() {
    assert_eq!(
        serde_json::to_value(ChatCompletionMessage::user("hello")).unwrap(),
        json!({"role":"user","content":"hello"})
    );
    let expected = json!({"role":"user","content":[{"type":"text","text":"Describe"},{"type":"image_url","image_url":{"url":PNG}},{"type":"text","text":"briefly"}]});
    assert_eq!(serde_json::to_value(message()).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<ChatCompletionMessage>(expected).unwrap(),
        message()
    );
    for url in [PNG, "https://example.com/board.png"] {
        let part = ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: Some(ImageDetail::High),
            },
        };
        assert_eq!(
            serde_json::to_value(&part).unwrap()["image_url"]["detail"],
            "high"
        );
        assert_eq!(
            serde_json::from_value::<ContentPart>(serde_json::to_value(&part).unwrap()).unwrap(),
            part
        );
    }
    assert!(
        serde_json::from_value::<ContentPart>(json!({"type":"image_url","image_url":{"url":7}}))
            .is_err()
    );
    assert!(
        serde_json::from_value::<ContentPart>(
            json!({"type":"image_url","image_url":{"url":PNG,"detail":"huge"}})
        )
        .is_err()
    );
}

#[tokio::test]
async fn kobold_images_tools_reasoning_share_fit_and_generation_pipeline() {
    let tool = ToolDefinition::new("lookup", "lookup", json!({"type":"object","properties":{}}));
    let mut request = ChatCompletionRequest::new(vec![
        ChatCompletionMessage::system("SYS"),
        ChatCompletionMessage::user("OLD"),
        message(),
    ]);
    request.tools = vec![tool.clone()];
    request.parameters = json!({"max_tokens":100,"preserve_thinking":true});
    request.image_token_reserve = 200;
    request
        .chat_template_kwargs
        .insert("enable_thinking".into(), true.into());
    let mut trimmed = request.clone();
    trimmed.messages.remove(1);
    let prompt = format!(
        "{} SYS Describe (Attached Image 1) briefly",
        serde_json::to_string(&tool.function).unwrap()
    );
    let expected = json!({"messages":trimmed.messages,"tools":trimmed.tools,"preserve_thinking":true,"chat_template_kwargs":{"enable_thinking":true}});
    let reply = json!({"choices":[{"message":{"role":"assistant","content":[{"type":"text","text":"part1"},{"type":"text","text":"part2"}],"reasoning_content":"reasoning","tool_calls":[{"id":"call1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]});
    let (url, handle) = support::server(vec![
        (
            "/api/extra/true_max_context_length",
            json!({}),
            json!({"value":1024}),
        ),
        (
            "/api/extra/tokencount",
            json!({"messages":request.messages}),
            json!({"value":800,"prompt":format!("{} SYS OLD Describe (Attached Image 1) briefly",serde_json::to_string(&tool.function).unwrap())}),
        ),
        (
            "/api/extra/tokencount",
            expected.clone(),
            json!({"value":100,"prompt":prompt}),
        ),
        (
            "/api/extra/tokencount",
            expected.clone(),
            json!({"value":100,"prompt":prompt}),
        ),
        ("/v1/chat/completions", expected, reply),
    ]);
    let client = KoboldClient::connect_with_client(
        url,
        reqwest::Client::builder().no_proxy().build().unwrap(),
    )
    .await
    .unwrap();
    let fitted = client.fit_chat(&request).await.unwrap();
    assert_eq!(fitted.dropped_messages, 1);
    assert_eq!(fitted.preview.tokens, 300);
    assert_eq!(fitted.request.messages.last().unwrap(), &message());
    let response = client.generate_chat_full(&fitted.request).await.unwrap();
    assert_eq!(response.content.as_deref(), Some("part1part2"));
    assert_eq!(response.reasoning_content.as_deref(), Some("reasoning"));
    assert_eq!(response.tool_calls.len(), 1);
    assert!(serde_json::to_value(response.message).unwrap()["content"].is_array());
    handle.join().unwrap();
}

#[tokio::test]
async fn latest_image_is_never_silently_trimmed_and_text_is_checked() {
    for prompt in ["Describe (Attached Image 1) briefly", "Describe only"] {
        let mut request = ChatCompletionRequest::new(vec![message()]);
        request.parameters = json!({"max_tokens":100});
        request.image_token_reserve = 1000;
        let (url, handle) = support::server(vec![
            (
                "/api/extra/true_max_context_length",
                json!({}),
                json!({"value":1024}),
            ),
            (
                "/api/extra/tokencount",
                json!({"messages":request.messages}),
                json!({"value":100,"prompt":prompt}),
            ),
        ]);
        let client = KoboldClient::connect_with_client(
            url,
            reqwest::Client::builder().no_proxy().build().unwrap(),
        )
        .await
        .unwrap();
        assert!(client.fit_chat(&request).await.is_err());
        handle.join().unwrap();
    }
}

/// Opt-in real server smoke test. Needs a vision model and its matching mmproj.
#[tokio::test]
#[ignore = "requires KOBOLD_URL and a running vision-capable KoboldCpp"]
async fn live_kobold_png() {
    let url = std::env::var("KOBOLD_URL").expect("set KOBOLD_URL");
    let client = KoboldClient::connect_with_client(
        url,
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap(),
    )
    .await
    .unwrap();
    let mut request = ChatCompletionRequest::new(vec![message()]);
    request.parameters = json!({"max_tokens":64,"temperature":0,"thinking_budget_tokens":0});
    let fitted = client.fit_chat(&request).await.unwrap();
    let response = client.generate_chat_full(&fitted.request).await.unwrap();
    assert!(!response.content.unwrap_or_default().trim().is_empty());
    // A successful HTTP response alone does not prove visual comprehension.
    assert_eq!(fitted.request.messages[0].image_count(), 1);
}
