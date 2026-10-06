mod context;
mod instruct;
mod reasoning;
mod sysprompt;

pub use context::ContextTemplate;
pub use instruct::InstructTemplate;
pub use reasoning::ReasoningTemplate;
pub use sysprompt::SystemPromptTemplate;

/// Shared name interface used by applications that load named templates.
pub trait NamedTemplate {
    fn name(&self) -> &str;
}
