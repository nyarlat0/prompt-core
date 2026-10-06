use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub name: Option<String>,
    pub content: String,
}

impl PromptMessage {
    pub fn user(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::User,
            name: Some(name.into()),
            content: content.into(),
        }
    }

    pub fn assistant(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::Assistant,
            name: Some(name.into()),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::System,
            name: None,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PromptData {
    /// Extra application-specific JSON fields exposed at template root (and
    /// under `custom`). Standard SillyTavern fields keep precedence on clashes.
    pub custom: Value,
    pub user: String,
    pub character: String,
    pub persona: String,
    pub personas: String,
    pub characters: String,
    pub all_chars: String,
    pub unactive_chars: String,
    pub group_roleplay: bool,

    pub description: String,
    pub personality: String,
    pub scenario: String,

    pub creator_notes: String,
    pub character_system_prompt: String,
    pub character_post_history: String,
    pub alternate_greetings: Vec<String>,

    pub wi_before: String,
    pub wi_after: String,

    pub anchor_before: String,
    pub anchor_after: String,

    pub mes_examples: String,
    pub mes_examples_raw: String,

    pub messages: Vec<PromptMessage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoryStringRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoryStringPosition {
    Top,
    InChat { depth: usize, role: StoryStringRole },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoryStringSpec {
    pub position: StoryStringPosition,
}

impl StoryStringSpec {
    pub fn from_context(position: i32, depth: i32, role: i32) -> Self {
        if position == 0 {
            return Self {
                position: StoryStringPosition::Top,
            };
        }

        let role = match role {
            1 => StoryStringRole::User,
            2 => StoryStringRole::Assistant,
            _ => StoryStringRole::System,
        };

        Self {
            position: StoryStringPosition::InChat {
                depth: depth.max(0) as usize,
                role,
            },
        }
    }
}

#[derive(Debug)]
pub struct BuiltPrompt {
    pub text: String,
    pub stop_sequences: Vec<String>,
}
