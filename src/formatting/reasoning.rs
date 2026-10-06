use serde::{Deserialize, Serialize};

use super::NamedTemplate;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReasoningTemplate {
    pub name: String,

    #[serde(default)]
    pub prefix: String,

    #[serde(default)]
    pub suffix: String,

    #[serde(default)]
    pub separator: String,
}

impl NamedTemplate for ReasoningTemplate {
    fn name(&self) -> &str {
        &self.name
    }
}

impl ReasoningTemplate {
    pub fn strip_from_output(&self, output: &str) -> String {
        if self.prefix.is_empty() || self.suffix.is_empty() {
            return output.to_string();
        }

        let mut result = output.to_string();

        loop {
            let Some(start) = result.find(&self.prefix) else {
                break;
            };

            let content_start = start + self.prefix.len();

            let Some(relative_end) = result[content_start..].find(&self.suffix) else {
                // Незакрытый reasoning block.
                // Не уничтожаем потенциально полезный текст.
                break;
            };

            let mut end = content_start + relative_end + self.suffix.len();

            /*
             * ST reasoning template может задавать
             * separator между reasoning и ответом.
             */
            if !self.separator.is_empty() && result[end..].starts_with(&self.separator) {
                end += self.separator.len();
            }

            result.replace_range(start..end, "");
        }

        result.trim_start().to_string()
    }
}
