use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::RussianLemmatizer;

#[derive(Debug, Clone)]
pub struct LorebookEntry {
    pub name: String,
    pub keys: Vec<String>,
    pub secondary_keys: Vec<String>,
    pub content: String,
    pub constant: bool,
    pub enabled: bool,
    pub position: i32,
    pub order: i32,
}

#[derive(Debug, Clone)]
pub struct Lorebook {
    name: String,
    scan_depth: usize,
    entries: Vec<LorebookEntry>,
    raw: Value,
}

#[derive(Debug, Clone, Default)]
pub struct LorebookActivation {
    pub before: String,
    pub after: String,
    pub matched_entries: Vec<String>,
}

impl Lorebook {
    fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("cannot read lorebook {}", path.display()))?;
        let raw: Value = serde_json::from_str(&text)
            .with_context(|| format!("invalid lorebook {}", path.display()))?;

        let object = raw
            .as_object()
            .context("lorebook must contain a JSON object")?;
        let fallback_name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("lorebook");
        let name = string(object, "name");

        Ok(Self {
            name: if name.is_empty() {
                fallback_name.to_owned()
            } else {
                name
            },
            scan_depth: positive_usize(object.get("scan_depth")).unwrap_or(20),
            entries: parse_entries(object.get("entries"))?,
            raw,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn scan_depth(&self) -> usize {
        self.scan_depth
    }

    pub fn raw(&self) -> &Value {
        &self.raw
    }

    fn activate(
        &self,
        scan_tokens: &[String],
        lemmatizer: &RussianLemmatizer,
        manual_keys: &[String],
    ) -> LorebookActivation {
        let manual_tokens = manual_keys
            .iter()
            .map(|key| lemmatizer.lemmatize_text(key))
            .collect::<Vec<_>>();
        let mut selected = Vec::new();

        for entry in &self.entries {
            if !entry.enabled || entry.content.trim().is_empty() {
                continue;
            }

            let mut keys = entry
                .keys
                .iter()
                .chain(entry.secondary_keys.iter())
                .filter(|key| !key.trim().is_empty());
            let matched = entry.constant
                || keys
                    .clone()
                    .any(|key| lemmatizer.key_matches(scan_tokens, key))
                || keys.any(|key| {
                    manual_tokens
                        .iter()
                        .any(|manual| !manual.is_empty() && lemmatizer.key_matches(manual, key))
                });

            if !matched {
                continue;
            }
            selected.push(entry);
        }

        let mut before = String::new();
        let mut after = String::new();

        for entry in &selected {
            let destination = if entry.position == 0 {
                &mut before
            } else {
                &mut after
            };

            if !destination.is_empty() {
                destination.push_str("\n\n");
            }
            destination.push_str(&entry.content);
        }

        LorebookActivation {
            before,
            after,
            matched_entries: selected.iter().map(|entry| entry.name.clone()).collect(),
        }
    }
}

pub struct LorebookStore {
    lorebooks: HashMap<String, Lorebook>,
    lemmatizer: RussianLemmatizer,
}

impl LorebookStore {
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let mut lorebooks = HashMap::new();
        let lemmatizer = RussianLemmatizer::new();

        if !directory.exists() {
            return Ok(Self {
                lorebooks,
                lemmatizer,
            });
        }

        for entry in fs::read_dir(directory)
            .with_context(|| format!("cannot read lorebook directory {}", directory.display()))?
        {
            let path = entry?.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }

            let lorebook = Lorebook::from_path(&path)?;
            let name = lorebook.name().to_owned();

            if lorebooks.insert(name.clone(), lorebook).is_some() {
                bail!("duplicate lorebook name: {name}");
            }
        }

        Ok(Self {
            lorebooks,
            lemmatizer,
        })
    }

    pub fn get(&self, name: &str) -> Option<&Lorebook> {
        self.lorebooks.get(name)
    }

    pub fn activate(
        &self,
        name: &str,
        scan_text: &str,
        manual_keys: &[String],
    ) -> Option<LorebookActivation> {
        let lorebook = self.lorebooks.get(name)?;
        let scan_tokens = self.lemmatizer.lemmatize_text(scan_text);
        Some(lorebook.activate(&scan_tokens, &self.lemmatizer, manual_keys))
    }

    /// Activate the selected books, then apply one-way cross-activation links.
    /// Cross links scan directly activated source content once, without recursion.
    pub fn activate_many(
        &self,
        active: &[String],
        scan_text: &str,
        manual_keys: &[String],
        relations: &[(String, String)],
    ) -> LorebookActivation {
        let mut result = LorebookActivation::default();
        let mut activations = HashMap::new();
        for name in active {
            let Some(activation) = self.activate(name, scan_text, manual_keys) else {
                continue;
            };
            append_lorebook_activation(&mut result, &activation, name, false);
            activations.insert(name.clone(), activation);
        }
        for (source, target) in relations {
            let Some(source_activation) = activations.get(source) else {
                continue;
            };
            if source_activation.matched_entries.is_empty() {
                continue;
            }
            let source_content =
                format!("{}\n{}", source_activation.before, source_activation.after);
            let Some(target_activation) = self.activate(target, &source_content, manual_keys)
            else {
                continue;
            };
            append_lorebook_activation(&mut result, &target_activation, target, true);
        }
        result
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.lorebooks.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

fn append_lorebook_activation(
    destination: &mut LorebookActivation,
    source: &LorebookActivation,
    lorebook_name: &str,
    avoid_duplicates: bool,
) {
    for (target, content) in [
        (&mut destination.before, &source.before),
        (&mut destination.after, &source.after),
    ] {
        if content.is_empty() {
            continue;
        }
        if avoid_duplicates && target.contains(content) {
            continue;
        }
        if !target.is_empty() {
            target.push_str("\n\n");
        }
        target.push_str(content);
    }
    for entry in &source.matched_entries {
        let qualified = format!("{lorebook_name}/{entry}");
        if !destination.matched_entries.contains(&qualified) {
            destination.matched_entries.push(qualified);
        }
    }
}

fn parse_entries(value: Option<&Value>) -> Result<Vec<LorebookEntry>> {
    let values: Vec<&Value> = match value {
        Some(Value::Object(map)) => map.values().collect(),
        Some(Value::Array(values)) => values.iter().collect(),
        Some(_) => bail!("lorebook entries must be an object or array"),
        None => Vec::new(),
    };

    let mut entries = values
        .into_iter()
        .map(parse_entry)
        .collect::<Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.order);
    Ok(entries)
}

fn parse_entry(value: &Value) -> Result<LorebookEntry> {
    let object = value
        .as_object()
        .context("lorebook entry must be an object")?;
    let keys = strings(object, &["key", "keys"]);
    let secondary_keys = strings(object, &["keysecondary", "secondary_keys"]);
    let name = string(object, "name");
    let content = string(object, "content");
    let enabled = object
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true)
        && !object
            .get("disable")
            .and_then(Value::as_bool)
            .unwrap_or(false);

    Ok(LorebookEntry {
        name: if name.is_empty() {
            content.lines().next().unwrap_or("unnamed entry").to_owned()
        } else {
            name
        },
        keys,
        secondary_keys,
        content,
        constant: bool(object, "constant"),
        enabled,
        position: number(object, "position").unwrap_or(0) as i32,
        order: number(object, "order")
            .or_else(|| number(object, "insertion_order"))
            .unwrap_or(0) as i32,
    })
}

fn string(object: &serde_json::Map<String, Value>, key: &str) -> String {
    object
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn strings(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Vec<String> {
    for key in keys {
        if let Some(values) = object.get(*key).and_then(Value::as_array) {
            return values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect();
        }
        if let Some(value) = object.get(*key).and_then(Value::as_str)
            && !value.trim().is_empty()
        {
            return vec![value.to_owned()];
        }
    }
    Vec::new()
}

fn bool(object: &serde_json::Map<String, Value>, key: &str) -> bool {
    object.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn number(object: &serde_json::Map<String, Value>, key: &str) -> Option<i64> {
    object.get(key).and_then(Value::as_i64)
}

fn positive_usize(value: Option<&Value>) -> Option<usize> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::LorebookStore;

    #[test]
    fn activates_only_matching_or_constant_entries() {
        let path = PathBuf::from(format!(
            "/tmp/ai-tg-bot-lorebook-test-{}.json",
            std::process::id()
        ));
        let json = r#"
        {
          "name": "test",
          "entries": {
            "match": {"key": ["кошка"], "content": "matched"},
            "constant": {"key": [], "constant": true, "content": "always"},
            "other": {"key": ["собака"], "content": "wrong"},
            "probability": {"key": ["кошка"], "probability": 0, "content": "also matched"}
          }
        }
        "#;
        fs::write(&path, json).unwrap();

        let store = LorebookStore::load(path.parent().unwrap()).unwrap();
        let activation = store.activate("test", "Кошки смотрят в окно", &[]).unwrap();

        assert!(activation.before.contains("matched"));
        assert!(activation.before.contains("always"));
        assert!(activation.before.contains("also matched"));
        assert!(!activation.before.contains("wrong"));
        assert!(
            activation.before.find("matched").unwrap() < activation.before.find("always").unwrap()
        );
        assert!(
            activation.before.find("always").unwrap()
                < activation.before.find("also matched").unwrap()
        );

        let manual = store
            .activate("test", "нет подходящих слов", &["кошка".to_string()])
            .unwrap();
        assert!(manual.before.contains("matched"));
        assert!(manual.before.contains("also matched"));

        let _ = fs::remove_file(path);
    }
}
