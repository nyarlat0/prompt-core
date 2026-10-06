use prompt_core::{
    ContextTemplate, InstructTemplate, PromptBuilder, PromptData, SystemPromptTemplate,
    TemplateEngine,
};
use serde_json::json;

#[test]
fn expressions_receive_request_data_and_arguments_without_html_escaping() {
    let mut engine = TemplateEngine::new();
    engine.register_expression("move", |data, args| {
        Ok(format!(
            "{} -> {}",
            data["side"].as_str().unwrap(),
            args[0].as_str().unwrap()
        ))
    });
    let template = "{{move target}} | {{fen}}";
    assert_eq!(
        engine
            .render(
                template,
                &json!({"side":"white", "target":"e4", "fen":"<board>"})
            )
            .unwrap(),
        "white -> e4 | <board>"
    );
    assert_eq!(
        engine
            .render(
                template,
                &json!({"side":"black", "target":"e5", "fen":"&board"})
            )
            .unwrap(),
        "black -> e5 | &board"
    );
}

#[test]
fn builder_uses_extensions_in_all_template_stages() {
    let mut builder = PromptBuilder::new();
    builder.register_expression("position", |data, _| {
        data["fen"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| "missing fen".into())
    });
    let context: ContextTemplate = serde_json::from_value(json!({
        "name":"chess", "story_string":"{{system}}|{{description}}|{{custom.fen}}|{{position}}"
    }))
    .unwrap();
    let instruct: InstructTemplate = serde_json::from_value(json!({
        "name":"test", "macro":true, "output_sequence":"<model {{position}}>",
        "system_sequence":"<system>", "stop_sequence":"<stop {{position}}>"
    }))
    .unwrap();
    let system: SystemPromptTemplate = serde_json::from_value(json!({
        "name":"test", "content":"{{position}}", "post_history":"{{position}}"
    }))
    .unwrap();
    let data = PromptData {
        custom: json!({"fen":"board&", "system":"must not replace system"}),
        description: "{{position}}".into(),
        ..Default::default()
    };
    let built = builder.build(&context, &instruct, &system, &data).unwrap();
    assert_eq!(
        built.text,
        "board&|board&|board&|board&<system>board&<model board&>"
    );
    assert_eq!(built.stop_sequences, vec!["<stop board&>"]);
}

#[test]
fn expression_errors_propagate_to_caller() {
    let mut engine = TemplateEngine::new();
    engine.register_expression("invalid", |_, _| Err("illegal position".into()));
    assert!(
        engine
            .render("{{invalid}}", &json!({}))
            .unwrap_err()
            .to_string()
            .contains("illegal position")
    );
}
