use rsmorphy::{MorphAnalyzer, prelude::Source, rsmorphy_dict_ru};

const MIN_MORPHOLOGY_WORD_LENGTH: usize = 3;

#[derive(Debug)]
pub struct RussianLemmatizer {
    morph: MorphAnalyzer,
}

impl RussianLemmatizer {
    pub fn new() -> Self {
        Self {
            morph: MorphAnalyzer::from_file(rsmorphy_dict_ru::DICT_PATH),
        }
    }

    pub fn lemmatize_text(&self, text: &str) -> Vec<String> {
        tokenize(text)
            .into_iter()
            .map(|token| {
                if token.chars().count() < MIN_MORPHOLOGY_WORD_LENGTH
                    || token.chars().all(|character| character.is_ascii_digit())
                {
                    return token;
                }
                self.morph
                    .parse(&token)
                    .into_iter()
                    .next()
                    .map(|parse| parse.lex.get_lemma(&self.morph).get_word().into_owned())
                    .unwrap_or(token)
            })
            .collect()
    }

    pub fn key_matches(&self, text_tokens: &[String], key: &str) -> bool {
        let key_tokens = self.lemmatize_text(key);
        !key_tokens.is_empty()
            && text_tokens
                .windows(key_tokens.len())
                .any(|window| window == key_tokens.as_slice())
    }
}

impl Default for RussianLemmatizer {
    fn default() -> Self {
        Self::new()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if character.is_alphanumeric() || matches!(character, '-' | '_') {
            current.push(character.to_lowercase().next().unwrap_or(character));
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

#[cfg(test)]
mod tests {
    use super::RussianLemmatizer;

    #[test]
    fn russian_inflections_match_their_lemmas() {
        let lemmatizer = RussianLemmatizer::new();
        let tokens = lemmatizer.lemmatize_text("Кошки бегают по комнатам");

        assert!(tokens.iter().any(|token| token == "кошка"));
        assert!(tokens.iter().any(|token| token == "бегать"));
        assert!(lemmatizer.key_matches(&tokens, "комната"));
        assert!(!lemmatizer.key_matches(&tokens, "собака"));

        let level_tokens = lemmatizer.lemmatize_text("он стоял на уровне 0");
        assert!(lemmatizer.key_matches(&level_tokens, "Уровень 0"));
    }

    #[test]
    fn short_tokens_do_not_reach_rsmorphy() {
        let lemmatizer = RussianLemmatizer::new();
        let tokens = lemmatizer.lemmatize_text("Я и он в чате 0 12");

        assert_eq!(tokens, ["я", "и", "он", "в", "чат", "0", "12"]);
    }
}
