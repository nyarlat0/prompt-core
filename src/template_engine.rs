use anyhow::Result;
use handlebars::{
    Context, Handlebars, Helper, HelperDef, HelperResult, Output, RenderContext, RenderError,
    RenderErrorReason, no_escape,
};
use serde::Serialize;
use serde_json::Value;

const TRIM_MARKER: &str = "__PROMPT_CORE_STORY_TRIM_MARKER__";

/// A reusable Handlebars registry. Helpers registered here are available in
/// every template rendered by this engine, including prompt and format parts.
/// HTML escaping is disabled because templates produce model input, not HTML.
pub struct TemplateEngine {
    registry: Handlebars<'static>,
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateEngine {
    pub fn new() -> Self {
        let mut registry = Handlebars::new();
        registry.register_escape_fn(no_escape);
        Self { registry }
    }

    /// Register a Handlebars helper, usable as `{{name}}` or
    /// `{{name argument key=value}}` in any template rendered by this engine.
    /// Implement [`HelperDef`] for custom expressions that need logic or
    /// access to the template context.
    pub fn register_helper(&mut self, name: &str, helper: impl HelperDef + Send + Sync + 'static) {
        self.registry.register_helper(name, Box::new(helper));
    }

    /// Register a compact custom expression. The closure receives the full
    /// template JSON context and positional expression arguments, so
    /// `register_expression("legal_moves", ...)` enables `{{legal_moves}}`
    /// as well as expressions such as `{{legal_moves fen}}`.
    pub fn register_expression<F>(&mut self, name: &str, expression: F)
    where
        F: Fn(&Value, &[Value]) -> Result<String, String> + Send + Sync + 'static,
    {
        self.register_helper(name, ExpressionHelper(expression));
    }

    pub fn render<T: Serialize>(&self, template: &str, data: &T) -> Result<String> {
        if template.is_empty() {
            return Ok(String::new());
        }

        // SillyTavern's trim marker removes adjacent line breaks after render.
        let template = template.replace("{{trim}}", TRIM_MARKER);
        let rendered = self.registry.render_template(&template, data)?;
        Ok(Self::apply_trim_markers(rendered))
    }

    fn apply_trim_markers(mut text: String) -> String {
        while let Some(pos) = text.find(TRIM_MARKER) {
            let bytes = text.as_bytes();
            let mut start = pos;
            while start > 0 && matches!(bytes[start - 1], b'\n' | b'\r') {
                start -= 1;
            }
            let mut end = pos + TRIM_MARKER.len();
            while end < bytes.len() && matches!(bytes[end], b'\n' | b'\r') {
                end += 1;
            }
            text.replace_range(start..end, "");
        }
        text
    }
}

struct ExpressionHelper<F>(F);

impl<F> HelperDef for ExpressionHelper<F>
where
    F: Fn(&Value, &[Value]) -> Result<String, String> + Send + Sync,
{
    fn call<'reg: 'rc, 'rc>(
        &self,
        helper: &Helper<'rc>,
        _registry: &'reg Handlebars<'reg>,
        context: &'rc Context,
        _render_context: &mut RenderContext<'reg, 'rc>,
        output: &mut dyn Output,
    ) -> HelperResult {
        let arguments: Vec<Value> = helper
            .params()
            .iter()
            .map(|parameter| parameter.value().clone())
            .collect();
        let rendered = (self.0)(context.data(), &arguments)
            .map_err(|message| RenderError::from(RenderErrorReason::Other(message)))?;
        output.write(&rendered)?;
        Ok(())
    }
}
