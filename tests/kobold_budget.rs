use prompt_core::{GenerationRequest, KoboldClient, Preset};
use serde_json::json;

mod support;
use support::server;

#[tokio::test]
async fn preset_and_native_calls_count_before_generating_and_share_reserve() {
    let (url, handle) = server(vec![
        (
            "/api/extra/true_max_context_length",
            json!({}),
            json!({"value":1024}),
        ),
        (
            "/api/extra/tokencount",
            json!({"prompt":"hello", "special":true}),
            json!({"value":10}),
        ),
        (
            "/api/v1/generate",
            json!({"prompt":"hello", "max_length":100, "max_context_length":1024}),
            json!({"results":[{"text":"one"}]}),
        ),
        (
            "/api/extra/tokencount",
            json!({"prompt":"world", "special":true}),
            json!({"value":10}),
        ),
        (
            "/api/v1/generate",
            json!({"prompt":"world", "max_length":256, "max_context_length":1024}),
            json!({"results":[{"text":"two"}]}),
        ),
    ]);
    let client = KoboldClient::connect_with_client(
        url,
        reqwest::Client::builder().no_proxy().build().unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        client
            .generate(
                "hello",
                &Preset::from_value(json!({"genamt":100})).unwrap(),
                &[]
            )
            .await
            .unwrap(),
        "one"
    );
    assert_eq!(
        client
            .generate_request(&GenerationRequest::new("world"))
            .await
            .unwrap()
            .text,
        "two"
    );
    handle.join().unwrap();
}

#[tokio::test]
async fn overflow_or_invalid_token_count_never_generates() {
    for count in [json!(1000), json!(-1), json!(0)] {
        let (url, handle) = server(vec![
            (
                "/api/extra/true_max_context_length",
                json!({}),
                json!({"value":1024}),
            ),
            (
                "/api/extra/tokencount",
                json!({"prompt":"hello"}),
                json!({"value":count}),
            ),
        ]);
        let client = KoboldClient::connect_with_client(
            url,
            reqwest::Client::builder().no_proxy().build().unwrap(),
        )
        .await
        .unwrap();
        let error = client
            .generate_request(&GenerationRequest::new("hello"))
            .await
            .unwrap_err();
        assert!(
            !error.to_string().contains("подключиться"),
            "unexpected generation attempt: {error}"
        );
        handle.join().unwrap();
    }
}
