use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::NamedTemplate;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemPromptTemplate {
    pub name: String,

    #[serde(default)]
    pub content: String,

    #[serde(default)]
    pub post_history: String,

    #[serde(default)]
    pub extensions: Value,
}

impl NamedTemplate for SystemPromptTemplate {
    fn name(&self) -> &str {
        &self.name
    }
}
