use anyhow::Result;
use prompt_core::*;
use serde_json::json;
use std::{future::Future, pin::Pin};

// Deliberately simple deterministic test counter, not a production tokenizer.
struct Characters;
impl TokenCounter for Characters {
    fn count_tokens<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64>> + Send + 'a>> {
        Box::pin(async move { Ok(text.chars().count() as u64) })
    }
}

fn templates(in_chat: bool) -> (ContextTemplate, InstructTemplate, SystemPromptTemplate) {
    (
        serde_json::from_value(json!({
            "name":"test", "story_string":"{{system}}/{{wiBefore}}/{{description}}",
            "story_string_position": if in_chat {1} else {0}, "story_string_depth":2
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "name":"test", "input_sequence":"<u>", "first_input_sequence":"<first>",
            "output_sequence":"<a>", "system_sequence":"<s>",
            "input_suffix":"</u>", "output_suffix":"</a>", "system_suffix":"</s>"
        }))
        .unwrap(),
        serde_json::from_value(json!({
            "name":"test", "content":"RULES", "post_history":"POST"
        }))
        .unwrap(),
    )
}

#[tokio::test]
async fn fits_exact_boundary_without_mutating_or_dropping_history() {
    let (c, i, s) = templates(false);
    let builder = PromptBuilder::new();
    let data = PromptData {
        messages: vec![PromptMessage::user("Иван", "Привет 🌍")],
        ..Default::default()
    };
    let plain = builder.build(&c, &i, &s, &data).unwrap();
    let size = plain.text.chars().count() as u64;
    let fit = builder
        .build_with_budget(
            &c,
            &i,
            &s,
            &data,
            &Characters,
            ContextBudget::new(size + 18, 10),
        )
        .await
        .unwrap();
    assert_eq!(fit.prompt.text, plain.text);
    assert_eq!(fit.prompt_tokens, fit.input_budget);
    assert_eq!(fit.dropped_messages, 0);
    assert_eq!(data.messages.len(), 1);
}

#[tokio::test]
async fn trims_old_messages_but_preserves_system_latest_lore_and_post_history() {
    for group in [false, true] {
        let (c, i, s) = templates(group);
        let builder = PromptBuilder::new();
        let data = PromptData {
            character: "Alice".into(),
            group_roleplay: group,
            wi_before: "LORE".into(),
            description: "CARD".into(),
            messages: vec![
                PromptMessage::user("Иван", "старое".repeat(80)),
                PromptMessage::system("PINNED"),
                PromptMessage::assistant("Bob", "старый ответ".repeat(60)),
                PromptMessage::user("Иван", "новое действие"),
            ],
            ..Default::default()
        };
        let mut expected_data = data.clone();
        expected_data.messages = vec![data.messages[1].clone(), data.messages[3].clone()];
        let expected = builder.build(&c, &i, &s, &expected_data).unwrap();
        let limit = expected.text.chars().count() as u64;
        let fit = builder
            .build_with_budget(
                &c,
                &i,
                &s,
                &data,
                &Characters,
                ContextBudget::new(limit + 18, 10),
            )
            .await
            .unwrap();
        assert_eq!(fit.prompt.text, expected.text);
        assert_eq!(fit.dropped_messages, 2);
        for protected in ["RULES", "LORE", "CARD", "POST", "PINNED", "новое действие"]
        {
            assert!(fit.prompt.text.contains(protected));
        }
        assert_eq!(data.messages.len(), 4);
        assert!(data.messages[0].content.len() > 80);
    }
}

#[tokio::test]
async fn mandatory_content_or_latest_message_overflow_is_an_error() {
    let (c, i, s) = templates(false);
    let data = PromptData {
        messages: vec![PromptMessage::user("user", "too long".repeat(100))],
        ..Default::default()
    };
    let err = PromptBuilder::new()
        .build_with_budget(&c, &i, &s, &data, &Characters, ContextBudget::new(64, 16))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("обязательные блоки"));
    let empty = PromptData {
        wi_before: "LORE".repeat(100),
        ..Default::default()
    };
    assert!(
        PromptBuilder::new()
            .build_with_budget(&c, &i, &s, &empty, &Characters, ContextBudget::new(64, 16))
            .await
            .is_err()
    );
}

#[test]
fn invalid_budgets_and_generation_lengths_are_rejected() {
    assert!(ContextBudget::new(32, 32).input_tokens().is_err());
    assert!(ContextBudget::new(32, 0).input_tokens().is_err());
    assert!(ContextBudget::new(32, u64::MAX).input_tokens().is_err());
    for value in [json!(0), json!(-1), json!("bad")] {
        assert!(
            Preset::from_value(json!({"genamt":value}))
                .unwrap()
                .generation_length()
                .is_err()
        );
    }
    assert_eq!(
        Preset::from_value(json!({}))
            .unwrap()
            .kobold_request("", &[], 1024)
            .unwrap()["max_length"],
        256
    );
}
