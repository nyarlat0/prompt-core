#![doc = include_str!("../README.md")]
//! Reusable prompt construction and text-generation building blocks.
//!
//! Applications can use [`TemplateEngine`] with any JSON data model and add
//! their own Handlebars helpers, or use [`PromptBuilder`] and the SillyTavern
//! compatible template types for chat/RP prompts. [`KoboldClient`] and
//! [`Preset`] provide the KoboldCpp adapter.

mod builder;
#[cfg(feature = "roleplay")]
pub mod content;
mod formatting;
mod kobold;
mod preset;
mod template_engine;
mod types;

pub use builder::PromptBuilder;
pub use formatting::NamedTemplate;
pub use formatting::{ContextTemplate, InstructTemplate, ReasoningTemplate, SystemPromptTemplate};
pub use handlebars;
pub use kobold::{GenerationRequest, GenerationResponse, KoboldClient, ModelClient};
pub use preset::{Preset, PresetStore};
pub use template_engine::TemplateEngine;
pub use types::{
    BuiltPrompt, PromptData, PromptMessage, PromptRole, StoryStringPosition, StoryStringRole,
    StoryStringSpec,
};
