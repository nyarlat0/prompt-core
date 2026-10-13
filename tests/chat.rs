use prompt_core::*;
use serde_json::json;
mod support;
use support::server;

fn templates() -> (ContextTemplate, SystemPromptTemplate) {
    (
        serde_json::from_value(json!({"name":"chat","story_string":"{{system}}|{{description}}|{{wiBefore}}|{{custom.game}}"})).unwrap(),
        serde_json::from_value(json!({"name":"sys","content":"RULES {{char}}","user_prompt":"REQUEST {{char}}","post_history":"POST"})).unwrap()
    )
}

fn data() -> PromptData {
    PromptData {
        character: "Alice".into(),
        description: "CARD".into(),
        wi_before: "LORE".into(),
        custom: json!({"game":"CHESS"}),
        group_roleplay: true,
        messages: vec![
            PromptMessage::user("Иван", "one"),
            PromptMessage::assistant("Bob", "two"),
            PromptMessage::assistant("Alice", "three"),
            PromptMessage::user("Анна", "four"),
        ],
        ..Default::default()
    }
}

#[test]
fn chat_preserves_semantic_order_grouping_macros_and_post_history() {
    let (context, system) = templates();
    let messages = PromptBuilder::new()
        .build_chat(&context, &system, &data())
        .unwrap();
    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([
            {"role":"system","content":"RULES Alice|CARD|LORE|CHESS"},
            {"role":"user","content":"one\n\ntwo"},
            {"role":"assistant","content":"three"},
            {"role":"user","content":"four\n\nREQUEST Alice"},
            {"role":"system","content":"POST"}
        ])
    );
}

#[test]
fn in_chat_story_keeps_role_and_depth_after_grouping() {
    let (mut context, system) = templates();
    context.story_string_position = 1;
    context.story_string_depth = 1;
    context.story_string_role = 1;
    let messages = PromptBuilder::new()
        .build_chat(&context, &system, &data())
        .unwrap();
    assert_eq!(messages[2].role, ChatRole::User);
    assert!(messages[2].content.starts_with("RULES"));
    assert_eq!(messages[3].content, "four\n\nREQUEST Alice");
    assert_eq!(messages[4].role, ChatRole::System);
    context.chat_start = "<chat>".into();
    assert!(
        PromptBuilder::new()
            .build_chat(&context, &system, &data())
            .is_err()
    );
}

#[test]
fn text_completion_format_is_unchanged_and_not_reused_for_chat() {
    let (context, system) = templates();
    let instruct: InstructTemplate = serde_json::from_value(json!({
        "name":"test", "input_sequence":"<u>", "output_sequence":"<a>", "system_sequence":"<s>"
    }))
    .unwrap();
    let builder = PromptBuilder::new();
    let text = builder
        .build(&context, &instruct, &system, &data())
        .unwrap()
        .text;
    assert_eq!(
        text,
        "RULES Alice|CARD|LORE|CHESS<u>one\n\ntwo<a>three<u>four\n\nREQUEST Alice<s>POST<a>"
    );
    let chat = builder.build_chat(&context, &system, &data()).unwrap();
    assert!(
        chat.iter()
            .all(|m| !m.content.contains("<u>") && !m.content.contains("<a>"))
    );
}

#[tokio::test]
async fn sends_identical_structured_payload_to_counter_and_chat_endpoint() {
    let messages = vec![
        ChatMessage {
            role: ChatRole::System,
            content: "RULES".into(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: "Привет\n\nмир".into(),
        },
        ChatMessage {
            role: ChatRole::System,
            content: "POST".into(),
        },
    ];
    let mut request = ChatGenerationRequest::from_preset(
        messages,
        &Preset::from_value(json!({
            "genamt":100, "temp":0.7,"mirostat_mode":2,"mirostat_tau":5
        }))
        .unwrap(),
    )
    .unwrap();
    request.stop_sequences = vec!["STOP".into()];
    request.parameters["chat_template_kwargs"] = json!({"enable_thinking":false});
    let expected = json!({
        "messages": request.messages, "prompt":null, "max_length":null, "max_tokens":100,
        "max_context_length":1024, "temperature":0.7, "mirostat_mode":2,
        "stop":["STOP"],"stream":false, "n":1, "continue_assistant_turn":false,
        "chat_template_kwargs":{"enable_thinking":false}
    });
    let mut count_expected = expected.clone();
    count_expected["special"] = true.into();
    let (url, handle) = server(vec![
        (
            "/api/extra/true_max_context_length",
            json!({}),
            json!({"value":1024}),
        ),
        (
            "/api/extra/tokencount",
            count_expected,
            json!({"value":50,"prompt":"<s>RULES</s><u>Привет\n\nмир</u><s>POST</s><a>"}),
        ),
        (
            "/v1/chat/completions",
            expected,
            json!({"choices":[{"message":{"role":"assistant","content":"Ответ"}}]}),
        ),
    ]);
    let client = KoboldClient::connect_with_client(
        url,
        reqwest::Client::builder().no_proxy().build().unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(client.generate_chat(&request).await.unwrap().text, "Ответ");
    handle.join().unwrap();
}

#[tokio::test]
async fn rejects_bad_previews_without_sending_generation() {
    for preview in [
        json!({"value":1000,"prompt":"hello"}), // template overhead also counts
        json!({"value":0,"prompt":""}),         // old backend ignores messages
        json!({"value":3}),                     // old backend doesn't return rendered prompt
        json!({"value":3,"prompt":"lost content"}),
    ] {
        let (url, handle) = server(vec![
            (
                "/api/extra/true_max_context_length",
                json!({}),
                json!({"value":1024}),
            ),
            ("/api/extra/tokencount", json!({"prompt":null}), preview),
        ]);
        let client = KoboldClient::connect_with_client(
            url,
            reqwest::Client::builder().no_proxy().build().unwrap(),
        )
        .await
        .unwrap();
        let request = ChatGenerationRequest::new(vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
        }]);
        let error = client.generate_chat(&request).await.unwrap_err();
        assert!(
            !error.to_string().contains("completion request failed"),
            "{error}"
        );
        handle.join().unwrap();
    }
}

struct ChatCounter;
impl ChatTokenCounter for ChatCounter {
    fn preview_chat<'a>(
        &'a self,
        req: &'a ChatGenerationRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<ChatPreview>> + Send + 'a>>
    {
        Box::pin(async move {
            let prompt = req
                .messages
                .iter()
                .map(|m| format!("<{:?}>{}", m.role, m.content))
                .collect::<String>()
                + "<assistant>";
            Ok(ChatPreview {
                tokens: prompt.chars().count() as u64,
                prompt,
            })
        })
    }
}

#[tokio::test]
async fn chat_budget_counts_template_and_preserves_mandatory_messages() {
    let (context, system) = templates();
    let mut data = data();
    data.messages.insert(0, PromptMessage::system("PIN"));
    data.messages
        .insert(0, PromptMessage::user("old", "x".repeat(1000)));
    let options = ChatGenerationRequest::new(Vec::new());
    let fit = PromptBuilder::new()
        .build_chat_with_budget(
            &context,
            &system,
            &data,
            &ChatCounter,
            &options,
            ContextBudget::new(550, 256),
        )
        .await
        .unwrap();
    assert_eq!(fit.dropped_messages, 1);
    assert!(fit.preview.tokens <= fit.input_budget);
    assert!(fit.preview.prompt.contains("PIN"));
    assert!(fit.preview.prompt.contains("REQUEST Alice"));
    assert!(fit.preview.prompt.contains("POST"));
    assert_eq!(data.messages.len(), 6);
    assert!(
        PromptBuilder::new()
            .build_chat_with_budget(
                &context,
                &system,
                &data,
                &ChatCounter,
                &options,
                ContextBudget::new(270, 256)
            )
            .await
            .is_err()
    );
}

#[tokio::test]
async fn rejects_reserved_fields_before_tokenization() {
    let (url, handle) = server(vec![(
        "/api/extra/true_max_context_length",
        json!({}),
        json!({"value":1024}),
    )]);
    let client = KoboldClient::connect_with_client(
        url,
        reqwest::Client::builder().no_proxy().build().unwrap(),
    )
    .await
    .unwrap();
    for key in [
        "prompt",
        "messages",
        "max_length",
        "stream",
        "continue_assistant_turn",
        "memory",
        "tools",
    ] {
        let mut request = ChatGenerationRequest::new(vec![ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
        }]);
        request.parameters[key] = json!("invalid");
        assert!(
            client
                .generate_chat(&request)
                .await
                .unwrap_err()
                .to_string()
                .contains("reserved")
        );
    }
    handle.join().unwrap();
}

#[tokio::test]
async fn post_history_cannot_be_silently_moved_to_the_front() {
    let (url, handle) = server(vec![
        (
            "/api/extra/true_max_context_length",
            json!({}),
            json!({"value":1024}),
        ),
        (
            "/api/extra/tokencount",
            json!({}),
            json!({"value":10,"prompt":"<s>POST<u>hello<a>"}),
        ),
    ]);
    let client = KoboldClient::connect_with_client(
        url,
        reqwest::Client::builder().no_proxy().build().unwrap(),
    )
    .await
    .unwrap();
    let request = ChatGenerationRequest::new(vec![
        ChatMessage {
            role: ChatRole::User,
            content: "hello".into(),
        },
        ChatMessage {
            role: ChatRole::System,
            content: "POST".into(),
        },
    ]);
    assert!(
        client
            .generate_chat(&request)
            .await
            .unwrap_err()
            .to_string()
            .contains("reordered")
    );
    handle.join().unwrap();
}
