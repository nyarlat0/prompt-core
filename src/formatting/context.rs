use serde::{Deserialize, Serialize};

use super::NamedTemplate;
use crate::StoryStringSpec;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContextTemplate {
    pub name: String,

    pub story_string: String,

    #[serde(default)]
    pub example_separator: String,

    #[serde(default)]
    pub chat_start: String,

    #[serde(default)]
    pub use_stop_strings: bool,

    #[serde(default)]
    pub names_as_stop_strings: bool,

    #[serde(default)]
    pub story_string_position: i32,

    #[serde(default)]
    pub story_string_depth: i32,

    #[serde(default)]
    pub story_string_role: i32,

    #[serde(default)]
    pub always_force_name2: bool,

    #[serde(default)]
    pub trim_sentences: bool,

    #[serde(default)]
    pub single_line: bool,
}

impl NamedTemplate for ContextTemplate {
    fn name(&self) -> &str {
        &self.name
    }
}

impl ContextTemplate {
    pub fn story_string_spec(&self) -> StoryStringSpec {
        StoryStringSpec::from_context(
            self.story_string_position,
            self.story_string_depth,
            self.story_string_role,
        )
    }
}
