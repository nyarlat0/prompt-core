use crate::{
    BuiltPrompt, ContextTemplate, InstructTemplate, PromptBuilder, PromptData, PromptRole,
    SystemPromptTemplate,
};
use anyhow::{Context, Result, bail};
use std::{future::Future, pin::Pin};

/// Counts the complete rendered input using the target model's tokenizer.
pub trait TokenCounter: Send + Sync {
    fn count_tokens<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<u64>> + Send + 'a>>;
}

#[derive(Debug, Clone, Copy)]
pub struct ContextBudget {
    pub context_tokens: u64,
    pub response_tokens: u64,
    /// Additional headroom for backend-added tokens.
    pub safety_tokens: u64,
}

impl ContextBudget {
    pub fn new(context_tokens: u64, response_tokens: u64) -> Self {
        Self {
            context_tokens,
            response_tokens,
            safety_tokens: 8,
        }
    }

    pub fn input_tokens(self) -> Result<u64> {
        if self.response_tokens == 0 {
            bail!("Длина генерации должна быть больше нуля");
        }
        self.context_tokens
            .checked_sub(self.response_tokens)
            .and_then(|n| n.checked_sub(self.safety_tokens))
            .filter(|n| *n > 0)
            .context("Размер контекста недостаточен для ответа и запаса служебных токенов")
    }
}

#[derive(Debug)]
pub struct FittedPrompt {
    pub prompt: BuiltPrompt,
    pub prompt_tokens: u64,
    pub input_budget: u64,
    pub dropped_messages: usize,
}

impl PromptBuilder {
    /// Keep a recent suffix of conversational history, all system messages and
    /// the latest conversational message. Never mutate the caller's history.
    /// Every candidate is rebuilt with its own grouping, prefixes and injection
    /// positions, then counted as a whole. Expressions should be deterministic.
    pub async fn build_with_budget(
        &self,
        context: &ContextTemplate,
        instruct: &InstructTemplate,
        system: &SystemPromptTemplate,
        data: &PromptData,
        counter: &(impl TokenCounter + ?Sized),
        budget: ContextBudget,
    ) -> Result<FittedPrompt> {
        let limit = budget.input_tokens()?;
        let latest = data
            .messages
            .iter()
            .rposition(|m| m.role != PromptRole::System);
        let removable: Vec<usize> = data
            .messages
            .iter()
            .enumerate()
            .filter(|(i, m)| Some(*i) != latest && m.role != PromptRole::System)
            .map(|(i, _)| i)
            .collect();
        let build = |drop_count: usize| {
            let mut candidate = data.clone();
            candidate.messages = data
                .messages
                .iter()
                .enumerate()
                .filter(|(i, _)| removable[..drop_count].binary_search(i).is_err())
                .map(|(_, m)| m.clone())
                .collect();
            self.build(context, instruct, system, &candidate)
        };
        let measure =
            async |prompt: BuiltPrompt, dropped_messages: usize| -> Result<FittedPrompt> {
                let prompt_tokens = counter
                    .count_tokens(&prompt.text)
                    .await
                    .context("Не удалось подсчитать токены промпта")?;
                Ok(FittedPrompt {
                    prompt,
                    prompt_tokens,
                    input_budget: limit,
                    dropped_messages,
                })
            };
        let full = measure(build(0)?, 0).await?;
        if full.prompt_tokens <= limit {
            return Ok(full);
        }
        let mut fitting = measure(build(removable.len())?, removable.len()).await?;
        if fitting.prompt_tokens > limit {
            bail!(
                "Промпт не помещается в контекст: обязательные блоки и последнее сообщение занимают {} токенов, доступно {} (контекст {}, ответ {}, запас {}). Сократите инструкции, lorebooks или последнее сообщение.",
                fitting.prompt_tokens,
                limit,
                budget.context_tokens,
                budget.response_tokens,
                budget.safety_tokens
            );
        }
        // Binary search avoids one HTTP tokenization call per historical message.
        // Only return an actually measured fitting candidate, even if a custom
        // template makes token counts non-monotonic.
        let mut low = 1;
        let mut high = removable.len();
        while low < high {
            let mid = low + (high - low) / 2;
            let candidate = measure(build(mid)?, mid).await?;
            if candidate.prompt_tokens <= limit {
                high = mid;
                fitting = candidate;
            } else {
                low = mid + 1;
            }
        }
        Ok(fitting)
    }
}
