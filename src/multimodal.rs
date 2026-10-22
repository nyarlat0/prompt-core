//! Wire-level content. PromptBuilder deliberately remains text-only.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    /// HTTP(S) URL or data:image/png;base64,...; transmitted unchanged.
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }
}
impl From<String> for ChatContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}
impl From<&str> for ChatContent {
    fn from(value: &str) -> Self {
        Self::Text(value.into())
    }
}
impl ChatContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(_) => None,
        }
    }
    pub fn text_parts(&self) -> Vec<&str> {
        match self {
            Self::Text(text) => vec![text],
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
        }
    }
    pub fn image_count(&self) -> usize {
        match self {
            Self::Text(_) => 0,
            Self::Parts(parts) => parts
                .iter()
                .filter(|p| matches!(p, ContentPart::ImageUrl { .. }))
                .count(),
        }
    }
    /// Append without losing the original text or its order.
    pub fn push(&mut self, part: ContentPart) {
        if let Self::Text(text) = self {
            *self = Self::Parts(vec![ContentPart::text(std::mem::take(text))]);
        }
        if let Self::Parts(parts) = self {
            parts.push(part);
        }
    }
}
