use serde::{Deserialize, Serialize};

use super::NamedTemplate;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InstructTemplate {
    pub name: String,

    #[serde(default)]
    pub input_sequence: String,

    #[serde(default)]
    pub output_sequence: String,

    #[serde(default)]
    pub system_sequence: String,

    #[serde(default)]
    pub input_suffix: String,

    #[serde(default)]
    pub output_suffix: String,

    #[serde(default)]
    pub system_suffix: String,

    #[serde(default)]
    pub first_input_sequence: String,

    #[serde(default)]
    pub last_input_sequence: String,

    #[serde(default)]
    pub first_output_sequence: String,

    #[serde(default)]
    pub last_output_sequence: String,

    #[serde(default)]
    pub last_system_sequence: String,

    #[serde(default)]
    pub stop_sequence: String,

    #[serde(default)]
    pub user_alignment_message: String,

    #[serde(default)]
    pub story_string_prefix: String,

    #[serde(default)]
    pub story_string_suffix: String,

    #[serde(default)]
    pub wrap: bool,

    #[serde(default, rename = "macro")]
    pub macros: bool,

    #[serde(default)]
    pub skip_examples: bool,

    #[serde(default)]
    pub system_same_as_user: bool,

    #[serde(default)]
    pub sequences_as_stop_strings: bool,

    #[serde(default)]
    pub names_behavior: String,

    #[serde(default)]
    pub activation_regex: String,
}

impl NamedTemplate for InstructTemplate {
    fn name(&self) -> &str {
        &self.name
    }
}
