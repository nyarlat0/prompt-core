use std::collections::HashSet;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::{ContextTemplate, InstructTemplate, SystemPromptTemplate, TemplateEngine};

use super::{
    BuiltPrompt, PromptData, PromptMessage, PromptRole, StoryStringPosition, StoryStringRole,
};

/// Builds SillyTavern-style prompts using an application-configurable template
/// registry. Register custom Handlebars helpers before calling `build`.
pub struct PromptBuilder {
    engine: TemplateEngine,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self {
            engine: TemplateEngine::new(),
        }
    }

    pub fn with_engine(engine: TemplateEngine) -> Self {
        Self { engine }
    }

    pub fn register_helper(
        &mut self,
        name: &str,
        helper: impl handlebars::HelperDef + Send + Sync + 'static,
    ) {
        self.engine.register_helper(name, helper);
    }

    /// Register an application expression in every prompt template stage.
    pub fn register_expression<F>(&mut self, name: &str, expression: F)
    where
        F: Fn(&Value, &[Value]) -> Result<String, String> + Send + Sync + 'static,
    {
        self.engine.register_expression(name, expression);
    }

    pub fn render_text(&self, template: &str, data: &PromptData) -> Result<String> {
        let mut env = Self::base_environment(data);
        let personas = self.engine.render(&data.personas, &env)?;
        env["personas"] = Value::String(personas);
        self.engine.render(template, &env)
    }

    pub fn build(
        &self,
        context: &ContextTemplate,
        instruct: &InstructTemplate,
        sysprompt: &SystemPromptTemplate,
        data: &PromptData,
    ) -> Result<BuiltPrompt> {
        let mut base_env = Self::base_environment(data);
        let personas = self.engine.render(&data.personas, &base_env)?;
        base_env["personas"] = Value::String(personas.clone());

        /*
         * System Prompt сам тоже может содержать:
         *
         * {{user}}
         * {{char}}
         * и т.д.
         */
        let system = self
            .render_joined(
                &[&sysprompt.content, &data.character_system_prompt],
                &base_env,
            )
            .context("failed to render system prompt")?;

        /*
         * Character/persona fields SillyTavern тоже
         * прогоняет через macro replacement.
         */
        let description = self.engine.render(&data.description, &base_env)?;

        let personality = self.engine.render(&data.personality, &base_env)?;

        let persona = self.engine.render(&data.persona, &base_env)?;

        let scenario = self.engine.render(&data.scenario, &base_env)?;

        let wi_before = self.engine.render(&data.wi_before, &base_env)?;

        let wi_after = self.engine.render(&data.wi_after, &base_env)?;

        /*
         * Теперь окружение именно для Context Template.
         */
        let mut story_env = json!({
            "user": data.user,
            "char": data.character,
            "personas": personas,
            "characters": data.characters,
            "all-chars": data.all_chars,
            "unactive-chars": data.unactive_chars,

            "system": system,

            "description": description,
            "personality": personality,
            "persona": persona,
            "scenario": scenario,

            "creatorNotes": self.engine.render(&data.creator_notes, &base_env)?,
            "alternateGreetings": data.alternate_greetings,

            "wiBefore": wi_before,
            "wiAfter": wi_after,

            // SillyTavern поддерживает оба имени.
            "loreBefore": wi_before,
            "loreAfter": wi_after,

            "anchorBefore": data.anchor_before,
            "anchorAfter": data.anchor_after,

            "mesExamples": data.mes_examples,
            "mesExamplesRaw": data.mes_examples_raw,
        });
        story_env["custom"] = data.custom.clone();
        Self::extend_with_custom_fields(&mut story_env, &data.custom);

        let story = self
            .engine
            .render(&context.story_string, &story_env)
            .context("failed to render story string")?;

        let mut prompt = String::new();

        let story_spec = context.story_string_spec();

        if matches!(story_spec.position, StoryStringPosition::Top) {
            self.append_story_string(&mut prompt, &story, instruct, &story_env)?;
        }

        self.append_chat_start(&mut prompt, context, &story_env)?;

        let story_injection = match story_spec.position {
            StoryStringPosition::Top => None,

            StoryStringPosition::InChat { role, .. } => Some((story.as_str(), role)),
        };

        let user_prompt = self.engine.render(&sysprompt.user_prompt, &base_env)?;
        let mut messages = data.messages.clone();
        if !user_prompt.trim().is_empty() {
            messages.push(PromptMessage {
                role: PromptRole::User,
                name: None,
                content: user_prompt,
            });
        }
        self.append_history(
            &mut prompt,
            &messages,
            instruct,
            &base_env,
            data,
            story_injection,
            context.story_string_depth,
        )?;

        /*
         * post_history в ST добавляется после истории
         * перед следующей репликой модели.
         */
        let post_history = self.render_joined(
            &[&sysprompt.post_history, &data.character_post_history],
            &base_env,
        )?;

        if !post_history.is_empty() {
            let message = PromptMessage::system(post_history);

            let formatted =
                self.format_message(&message, instruct, &base_env, data, false, true)?;

            prompt.push_str(&formatted);
        }

        /*
         * Последняя строка prompt:
         *
         * <|turn>model
         *
         * После неё KoboldCpp начинает completion.
         */
        prompt.push_str(&self.generation_prefix(instruct, &base_env, data)?);

        let stop_sequences = self.stop_sequences(context, instruct, &base_env, data)?;

        Ok(BuiltPrompt {
            text: prompt,
            stop_sequences,
        })
    }

    fn base_environment(data: &PromptData) -> Value {
        let mut env = json!({
            "user": data.user,
            "char": data.character,
            "persona": data.persona,
            "personas": data.personas,
            "characters": data.characters,
            "all-chars": data.all_chars,
            "unactive-chars": data.unactive_chars,
            "custom": data.custom,
        });
        Self::extend_with_custom_fields(&mut env, &data.custom);
        env
    }

    fn extend_with_custom_fields(environment: &mut Value, custom: &Value) {
        let (Some(environment), Some(custom)) = (environment.as_object_mut(), custom.as_object())
        else {
            return;
        };
        for (key, value) in custom {
            environment
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
    }

    fn render_joined(&self, parts: &[&str], env: &Value) -> Result<String> {
        let mut rendered = Vec::new();

        for part in parts {
            let value = self.engine.render(part, env)?;
            if !value.is_empty() {
                rendered.push(value);
            }
        }

        Ok(rendered.join("\n\n"))
    }

    fn append_story_string(
        &self,
        prompt: &mut String,
        story: &str,
        instruct: &InstructTemplate,
        env: &Value,
    ) -> Result<()> {
        if story.is_empty() {
            return Ok(());
        }

        let separator = if instruct.wrap { "\n" } else { "" };

        let prefix =
            self.render_sequence(&instruct.story_string_prefix, instruct, env, "System")?;

        let suffix =
            self.render_sequence(&instruct.story_string_suffix, instruct, env, "System")?;

        if !prefix.is_empty() {
            prompt.push_str(&prefix);
            prompt.push_str(separator);
        }

        prompt.push_str(story);

        if !suffix.is_empty() {
            prompt.push_str(&suffix);
        }

        Ok(())
    }

    fn append_chat_start(
        &self,
        prompt: &mut String,
        context: &ContextTemplate,
        env: &Value,
    ) -> Result<()> {
        if context.chat_start.is_empty() {
            return Ok(());
        }

        let chat_start = self.engine.render(&context.chat_start, env)?;

        prompt.push_str(&chat_start);

        if !prompt.ends_with('\n') {
            prompt.push('\n');
        }

        Ok(())
    }

    fn append_history(
        &self,
        prompt: &mut String,
        messages: &[PromptMessage],
        instruct: &InstructTemplate,
        env: &Value,
        data: &PromptData,
        story_injection: Option<(&str, StoryStringRole)>,
        story_depth: i32,
    ) -> Result<()> {
        let grouped_messages;
        let messages = if data.group_roleplay {
            grouped_messages = Self::group_roleplay_messages(messages, data);
            &grouped_messages
        } else {
            messages
        };
        let last_user = messages
            .iter()
            .rposition(|message| message.role == PromptRole::User);

        let insertion_at = story_injection.map(|_| {
            if story_depth <= 0 {
                messages.len()
            } else {
                messages.len().saturating_sub(story_depth as usize)
            }
        });

        if insertion_at == Some(0) {
            self.append_story_injection(prompt, story_injection, instruct, env, data)?;
        }

        for (index, message) in messages.iter().enumerate() {
            let formatted = self.format_message(
                message,
                instruct,
                env,
                data,
                index == 0,
                Some(index) == last_user,
            )?;

            prompt.push_str(&formatted);

            if insertion_at == Some(index + 1) {
                self.append_story_injection(prompt, story_injection, instruct, env, data)?;
            }
        }

        Ok(())
    }

    fn group_roleplay_messages(
        messages: &[PromptMessage],
        data: &PromptData,
    ) -> Vec<PromptMessage> {
        let mut grouped: Vec<PromptMessage> = Vec::new();

        for message in messages {
            let role = if message.role == PromptRole::System {
                PromptRole::System
            } else if message.role == PromptRole::Assistant
                && message.name.as_deref() == Some(data.character.as_str())
            {
                PromptRole::Assistant
            } else {
                PromptRole::User
            };

            if role == PromptRole::User {
                if let Some(previous) = grouped.last_mut()
                    && previous.role == PromptRole::User
                {
                    if !previous.content.is_empty() {
                        previous.content.push_str("\n\n");
                    }
                    previous.content.push_str(&message.content);
                } else {
                    grouped.push(PromptMessage::user(
                        data.user.clone(),
                        message.content.clone(),
                    ));
                }
            } else {
                grouped.push(PromptMessage {
                    role,
                    name: message.name.clone(),
                    content: message.content.clone(),
                });
            }
        }

        grouped
    }

    fn append_story_injection(
        &self,
        prompt: &mut String,
        story_injection: Option<(&str, StoryStringRole)>,
        instruct: &InstructTemplate,
        env: &Value,
        data: &PromptData,
    ) -> Result<()> {
        let Some((content, role)) = story_injection else {
            return Ok(());
        };

        if content.is_empty() {
            return Ok(());
        }

        let role = match role {
            StoryStringRole::System => PromptRole::System,
            StoryStringRole::User => PromptRole::User,
            StoryStringRole::Assistant => PromptRole::Assistant,
        };

        let message = match role {
            PromptRole::System => PromptMessage::system(content),
            PromptRole::User => PromptMessage::user("User", content),
            PromptRole::Assistant => PromptMessage::assistant("Assistant", content),
        };

        prompt.push_str(&self.format_message(&message, instruct, env, data, false, false)?);

        Ok(())
    }

    fn format_message(
        &self,
        message: &PromptMessage,
        instruct: &InstructTemplate,
        env: &Value,
        data: &PromptData,
        is_first: bool,
        is_last_user: bool,
    ) -> Result<String> {
        let prefix = match message.role {
            PromptRole::User => {
                if is_last_user && !instruct.last_input_sequence.is_empty() {
                    &instruct.last_input_sequence
                } else if is_first && !instruct.first_input_sequence.is_empty() {
                    &instruct.first_input_sequence
                } else {
                    &instruct.input_sequence
                }
            }

            PromptRole::Assistant => {
                if is_first && !instruct.first_output_sequence.is_empty() {
                    &instruct.first_output_sequence
                } else {
                    &instruct.output_sequence
                }
            }

            PromptRole::System => {
                if instruct.system_same_as_user {
                    &instruct.input_sequence
                } else {
                    &instruct.system_sequence
                }
            }
        };

        let suffix = match message.role {
            PromptRole::User => &instruct.input_suffix,

            PromptRole::Assistant => &instruct.output_suffix,

            PromptRole::System => {
                if instruct.system_same_as_user {
                    &instruct.input_suffix
                } else {
                    &instruct.system_suffix
                }
            }
        };

        let fallback_name = match message.role {
            PromptRole::User => &data.user,
            PromptRole::Assistant => &data.character,
            PromptRole::System => "System",
        };

        let name = message.name.as_deref().unwrap_or(fallback_name);

        let prefix = self.render_sequence(prefix, instruct, env, name)?;

        let mut suffix = self.render_sequence(suffix, instruct, env, name)?;

        if suffix.is_empty() && instruct.wrap {
            suffix.push('\n');
        }

        let separator = if instruct.wrap { "\n" } else { "" };

        /*
         * В ST:
         *
         * always => имя всегда.
         * force   => группы / past personas.
         *
         * Групповых character cards у нас ещё нет,
         * поэтому FORCE пока имени не добавляет.
         */
        let include_name = !data.group_roleplay
            && message.role != PromptRole::System
            && instruct.names_behavior == "always";

        let content = if include_name && !name.is_empty() {
            format!("{name}: {}", message.content)
        } else {
            message.content.clone()
        };

        let mut parts = Vec::new();

        if !prefix.is_empty() {
            parts.push(prefix);
        }

        parts.push(format!("{content}{suffix}"));

        Ok(parts.join(separator))
    }

    fn generation_prefix(
        &self,
        instruct: &InstructTemplate,
        env: &Value,
        data: &PromptData,
    ) -> Result<String> {
        let sequence = if !instruct.last_output_sequence.is_empty() {
            &instruct.last_output_sequence
        } else {
            &instruct.output_sequence
        };

        let sequence = self.render_sequence(sequence, instruct, env, &data.character)?;

        let include_name = !data.group_roleplay && instruct.names_behavior == "always";

        let separator = if instruct.wrap { "\n" } else { "" };

        let mut result = String::new();

        if instruct.wrap {
            result.push_str(separator);
        }

        result.push_str(&sequence);

        if include_name && !data.character.is_empty() {
            result.push_str(separator);

            result.push_str(&format!("{}:", data.character));
        } else if instruct.wrap {
            result.push_str(separator);
        }

        Ok(result)
    }

    fn render_sequence(
        &self,
        sequence: &str,
        instruct: &InstructTemplate,
        env: &Value,
        name: &str,
    ) -> Result<String> {
        if !instruct.macros {
            return Ok(sequence.to_string());
        }

        let mut sequence_env = env.clone();

        if let Some(object) = sequence_env.as_object_mut() {
            object.insert("name".into(), Value::String(name.to_string()));
        }

        self.engine.render(sequence, &sequence_env)
    }
    fn stop_sequences(
        &self,
        context: &ContextTemplate,
        instruct: &InstructTemplate,
        env: &Value,
        data: &PromptData,
    ) -> Result<Vec<String>> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();

        let mut add = |value: String| {
            if value.trim().is_empty() {
                return;
            }

            if seen.insert(value.clone()) {
                result.push(value);
            }
        };

        /*
         * names_as_stop_strings в обычной
         * генерации ST не даёт модели начать
         * реплику пользователя.
         */
        if context.names_as_stop_strings && !data.user.is_empty() {
            add(format!("\n{}:", data.user));
        }

        let mut sequences = vec![instruct.stop_sequence.as_str()];

        if instruct.sequences_as_stop_strings {
            sequences.extend([
                instruct.input_sequence.as_str(),
                instruct.output_sequence.as_str(),
                instruct.first_output_sequence.as_str(),
                instruct.last_output_sequence.as_str(),
                instruct.system_sequence.as_str(),
                instruct.last_system_sequence.as_str(),
            ]);
        }

        for sequence in sequences {
            let rendered = self.render_sequence(sequence, instruct, env, "System")?;

            /*
             * SillyTavern тоже разбивает
             * sequence по newline.
             */
            for part in rendered.split('\n') {
                if part.trim().is_empty() {
                    continue;
                }

                let part = if instruct.wrap {
                    format!("\n{part}")
                } else {
                    part.to_string()
                };

                add(part);
            }
        }

        if context.use_stop_strings {
            if !context.chat_start.is_empty() {
                let value = self.engine.render(&context.chat_start, env)?;

                if !value.is_empty() {
                    add(format!("\n{value}"));
                }
            }

            if !context.example_separator.is_empty() {
                let value = self.engine.render(&context.example_separator, env)?;

                if !value.is_empty() {
                    add(format!("\n{value}"));
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::PromptBuilder;
    use crate::{
        ContextTemplate, InstructTemplate, PromptData, PromptMessage, SystemPromptTemplate,
    };

    #[test]
    fn group_roleplay_merges_users_and_hides_character_names() {
        let context: ContextTemplate = serde_json::from_value(json!({
            "name": "context",
            "story_string": ""
        }))
        .unwrap();
        let instruct: InstructTemplate = serde_json::from_value(json!({
            "name": "instruct",
            "input_sequence": "<user>",
            "output_sequence": "<model>",
            "names_behavior": "always"
        }))
        .unwrap();
        let sysprompt: SystemPromptTemplate = serde_json::from_value(json!({
            "name": "sysprompt",
            "content": ""
        }))
        .unwrap();
        let data = PromptData {
            user: "Иван".to_string(),
            character: "Alice".to_string(),
            group_roleplay: true,
            messages: vec![
                PromptMessage::user("Иван", "первый пользовательский ход"),
                PromptMessage::assistant("Alice", "ход Alice"),
                PromptMessage::user("Анна", "второй пользовательский ход"),
                PromptMessage::assistant("Bob", "ход Bob"),
            ],
            ..Default::default()
        };

        let built = PromptBuilder::new()
            .build(&context, &instruct, &sysprompt, &data)
            .unwrap();

        assert!(!built.text.contains("Alice:"));
        assert!(!built.text.contains("Bob:"));
        assert!(
            built
                .text
                .contains("второй пользовательский ход\n\nход Bob")
        );
        assert!(built.text.ends_with("<model>"));
    }

    #[test]
    fn renders_group_roster_directives_with_hyphens() {
        let data = PromptData {
            all_chars: "## Иван\nСледопыт".to_string(),
            unactive_chars: "## Шарлотта\nМаг".to_string(),
            ..Default::default()
        };

        let rendered = PromptBuilder::new().render_text(
            "{{#if all-chars}}{{all-chars}}{{/if}}\n---\n{{#if unactive-chars}}{{unactive-chars}}{{/if}}",
            &data,
        )
        .unwrap();

        assert_eq!(rendered, "## Иван\nСледопыт\n---\n## Шарлотта\nМаг");
    }

    #[test]
    fn renders_post_history_as_system_message() {
        let context: ContextTemplate = serde_json::from_value(json!({
            "name": "context",
            "story_string": ""
        }))
        .unwrap();
        let instruct: InstructTemplate = serde_json::from_value(json!({
            "name": "instruct",
            "input_sequence": "<user>",
            "system_sequence": "<system>",
            "output_sequence": "<model>",
            "input_suffix": "",
            "system_suffix": "",
            "output_suffix": ""
        }))
        .unwrap();
        let sysprompt: SystemPromptTemplate = serde_json::from_value(json!({
            "name": "sysprompt",
            "content": "",
            "post_history": "Инструкция после истории"
        }))
        .unwrap();
        let data = PromptData {
            messages: vec![PromptMessage::user("Иван", "ход пользователя")],
            ..Default::default()
        };

        let built = PromptBuilder::new()
            .build(&context, &instruct, &sysprompt, &data)
            .unwrap();

        assert!(built.text.contains("<system>Инструкция после истории"));
        assert!(!built.text.contains("<user>Инструкция после истории"));
    }
}
