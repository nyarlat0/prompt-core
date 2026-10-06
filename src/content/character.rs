use std::{
    collections::HashMap,
    fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct CharacterCard {
    name: String,
    description: String,
    personality: String,
    scenario: String,
    first_message: String,
    message_examples: String,
    creator_notes: String,
    system_prompt: String,
    post_history_instructions: String,
    alternate_greetings: Vec<String>,
    raw: Value,
}

impl CharacterCard {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = if path.extension().and_then(|extension| extension.to_str()) == Some("png") {
            let bytes = fs::read(path)
                .with_context(|| format!("cannot read character card {}", path.display()))?;
            extract_png_card(&bytes)
                .with_context(|| format!("cannot read chara metadata from {}", path.display()))?
        } else {
            let text = fs::read_to_string(path)
                .with_context(|| format!("cannot read character card {}", path.display()))?;
            serde_json::from_str(&text)
                .with_context(|| format!("invalid character card {}", path.display()))?
        };

        Self::from_json(raw, path.to_owned())
    }

    fn from_json(raw: Value, source: PathBuf) -> Result<Self> {
        let data = raw
            .get("data")
            .filter(|value| value.is_object())
            .unwrap_or(&raw);
        let object = data
            .as_object()
            .context("character card data must be a JSON object")?;

        let name = string_field(object, "name");
        if name.trim().is_empty() {
            bail!("character card {} has no name", source.display());
        }

        Ok(Self {
            name,
            description: string_field(object, "description"),
            personality: string_field(object, "personality"),
            scenario: string_field(object, "scenario"),
            first_message: first_message(object),
            message_examples: string_field_aliases(object, &["mes_example", "mes_examples"]),
            creator_notes: string_field_aliases(
                object,
                &["creator_notes", "creatorcomment", "creator_comments"],
            ),
            system_prompt: string_field(object, "system_prompt"),
            post_history_instructions: string_field(object, "post_history_instructions"),
            alternate_greetings: string_array_field(object, "alternate_greetings"),
            raw,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn personality(&self) -> &str {
        &self.personality
    }

    pub fn scenario(&self) -> &str {
        &self.scenario
    }

    pub fn first_message(&self) -> &str {
        &self.first_message
    }

    pub fn message_examples(&self) -> &str {
        &self.message_examples
    }

    pub fn creator_notes(&self) -> &str {
        &self.creator_notes
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub fn post_history_instructions(&self) -> &str {
        &self.post_history_instructions
    }

    pub fn alternate_greetings(&self) -> &[String] {
        &self.alternate_greetings
    }

    pub fn raw(&self) -> &Value {
        &self.raw
    }
}

pub struct CharacterStore {
    characters: HashMap<String, CharacterCard>,
}

impl CharacterStore {
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let mut characters = HashMap::new();

        if !directory.exists() {
            return Ok(Self { characters });
        }

        for entry in fs::read_dir(directory)
            .with_context(|| format!("cannot read character directory {}", directory.display()))?
        {
            let path = entry?.path();
            let extension = path.extension().and_then(|extension| extension.to_str());

            if !matches!(extension, Some("png") | Some("json")) {
                continue;
            }

            let card = CharacterCard::from_path(&path)?;
            let name = card.name().to_owned();

            if characters.insert(name.clone(), card).is_some() {
                bail!("duplicate character name: {name}");
            }
        }

        Ok(Self { characters })
    }

    pub fn get(&self, name: &str) -> Option<&CharacterCard> {
        self.characters.get(name)
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.characters.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

fn string_field(object: &serde_json::Map<String, Value>, name: &str) -> String {
    object
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn string_field_aliases(object: &serde_json::Map<String, Value>, names: &[&str]) -> String {
    names
        .iter()
        .map(|name| string_field(object, name))
        .find(|value| !value.is_empty())
        .unwrap_or_default()
}

fn string_array_field(object: &serde_json::Map<String, Value>, name: &str) -> Vec<String> {
    object
        .get(name)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn first_message(object: &serde_json::Map<String, Value>) -> String {
    string_field_aliases(object, &["first_mes", "first_message"])
}

fn extract_png_card(bytes: &[u8]) -> Result<Value> {
    const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

    if !bytes.starts_with(PNG_SIGNATURE) {
        bail!("file is not a PNG");
    }

    let mut cursor = Cursor::new(&bytes[8..]);

    loop {
        let length = read_u32(&mut cursor)? as usize;
        let mut kind = [0; 4];
        cursor.read_exact(&mut kind)?;

        let mut data = vec![0; length];
        cursor.read_exact(&mut data)?;
        let mut crc = [0; 4];
        cursor.read_exact(&mut crc)?;

        if &kind == b"tEXt" {
            if let Some(separator) = data.iter().position(|byte| *byte == 0)
                && &data[..separator] == b"chara"
            {
                let value = &data[separator + 1..];
                let decoded = BASE64
                    .decode(value)
                    .context("invalid base64 in PNG chara metadata")?;
                return serde_json::from_slice(&decoded)
                    .context("invalid JSON in PNG chara metadata");
            }
        }

        if &kind == b"IEND" {
            break;
        }
    }

    bail!("PNG has no chara tEXt chunk")
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut bytes = [0; 4];
    cursor.read_exact(&mut bytes)?;
    Ok(u32::from_be_bytes(bytes))
}
