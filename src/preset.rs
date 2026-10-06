use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

#[derive(Debug, Clone)]
pub struct Preset {
    raw: Value,
}

impl Preset {
    /// Explicit response reservation; missing genamt defaults to 256 tokens.
    pub fn generation_length(&self) -> Result<u64> {
        match self.raw.get("genamt") {
            None => Ok(256),
            Some(value) => value
                .as_u64()
                .filter(|n| *n > 0)
                .context("preset genamt must be a positive integer"),
        }
    }
    /// Construct a SillyTavern sampling preset without filesystem access.
    pub fn from_value(raw: Value) -> Result<Self> {
        if !raw.is_object() {
            bail!("preset must contain a JSON object");
        }
        Ok(Self { raw })
    }

    pub fn raw(&self) -> &Value {
        &self.raw
    }

    pub fn kobold_request(
        &self,
        prompt: &str,
        stop_sequences: &[String],
        max_context_length: u64,
    ) -> Result<Value> {
        let src = self
            .raw
            .as_object()
            .context("preset must contain a JSON object")?;

        let mut dst = Map::new();

        dst.insert("prompt".into(), Value::String(prompt.to_string()));

        /*
         * SillyTavern:
         *   genamt     = generation length
         *   max_length = context length
         *
         * KoboldCpp:
         *   max_length         = generation length
         *   max_context_length = context length
         */
        copy(src, &mut dst, "genamt", "max_length");
        dst.insert("max_length".into(), self.generation_length()?.into());
        copy(src, &mut dst, "max_length", "max_context_length");
        dst.insert(
            "max_context_length".into(),
            Value::Number(max_context_length.into()),
        );

        // Основные samplers.
        copy(src, &mut dst, "temp", "temperature");
        copy(src, &mut dst, "top_p", "top_p");
        copy(src, &mut dst, "top_k", "top_k");
        copy(src, &mut dst, "top_a", "top_a");
        copy(src, &mut dst, "tfs", "tfs");
        copy(src, &mut dst, "min_p", "min_p");

        // В ST называется typical_p.
        copy(src, &mut dst, "typical_p", "typical");

        // Repetition penalty.
        copy(src, &mut dst, "rep_pen", "rep_pen");
        copy(src, &mut dst, "rep_pen_range", "rep_pen_range");
        copy(src, &mut dst, "rep_pen_slope", "rep_pen_slope");

        // Presence/frequency penalties.
        copy(src, &mut dst, "freq_pen", "frequency_penalty");
        copy(src, &mut dst, "presence_pen", "presence_penalty");

        // Smoothing.
        copy(src, &mut dst, "smoothing_factor", "smoothing_factor");

        // DRY.
        copy(src, &mut dst, "dry_multiplier", "dry_multiplier");
        copy(src, &mut dst, "dry_base", "dry_base");
        copy(src, &mut dst, "dry_allowed_length", "dry_allowed_length");
        copy(src, &mut dst, "dry_penalty_last_n", "dry_penalty_last_n");

        /*
         * SillyTavern почему-то хранит dry_sequence_breakers
         * строкой, внутри которой лежит JSON-массив.
         */
        if let Some(value) = src.get("dry_sequence_breakers") {
            match value {
                Value::Array(_) => {
                    dst.insert("dry_sequence_breakers".into(), value.clone());
                }

                Value::String(s) => {
                    if let Ok(parsed) = serde_json::from_str::<Value>(s)
                        && parsed.is_array()
                    {
                        dst.insert("dry_sequence_breakers".into(), parsed);
                    }
                }

                _ => {}
            }
        }

        // Mirostat.
        copy(src, &mut dst, "mirostat_mode", "mirostat");
        copy(src, &mut dst, "mirostat_tau", "mirostat_tau");
        copy(src, &mut dst, "mirostat_eta", "mirostat_eta");

        // XTC / N-Sigma.
        copy(src, &mut dst, "xtc_threshold", "xtc_threshold");
        copy(src, &mut dst, "xtc_probability", "xtc_probability");
        copy(src, &mut dst, "nsigma", "nsigma");

        // Старый числовой sampler order KoboldCpp.
        copy(src, &mut dst, "sampler_order", "sampler_order");

        // Ban EOS.
        if let Some(Value::Bool(value)) = src.get("ban_eos_token") {
            dst.insert("use_default_badwordsids".into(), Value::Bool(*value));
        }

        // SillyTavern: skip special.
        // KoboldCpp: render special.
        if let Some(Value::Bool(skip)) = src.get("skip_special_tokens") {
            dst.insert("render_special".into(), Value::Bool(!skip));
        }

        dst.insert("trim_stop".into(), Value::Bool(true));

        dst.insert(
            "stop_sequence".into(),
            Value::Array(stop_sequences.iter().cloned().map(Value::String).collect()),
        );

        Ok(Value::Object(dst))
    }
}

fn copy(src: &Map<String, Value>, dst: &mut Map<String, Value>, from: &str, to: &str) {
    if let Some(value) = src.get(from) {
        dst.insert(to.to_string(), value.clone());
    }
}

pub struct PresetStore {
    presets: HashMap<String, Preset>,
    default: String,
}

impl PresetStore {
    pub fn load(directory: impl AsRef<Path>, default: impl Into<String>) -> Result<Self> {
        let directory = directory.as_ref();
        let default = default.into();

        let mut presets = HashMap::new();

        for entry in fs::read_dir(directory)
            .with_context(|| format!("cannot read preset directory {}", directory.display()))?
        {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }

            let name = path
                .file_stem()
                .and_then(|x| x.to_str())
                .context("invalid preset filename")?
                .to_string();

            let contents = fs::read_to_string(&path)
                .with_context(|| format!("cannot read preset {}", path.display()))?;

            let raw: Value = serde_json::from_str(&contents)
                .with_context(|| format!("invalid JSON in {}", path.display()))?;

            if !raw.is_object() {
                bail!("preset {} is not a JSON object", path.display());
            }

            presets.insert(name, Preset { raw });
        }

        if !presets.contains_key(&default) {
            bail!(
                "default preset {:?} not found in {}",
                default,
                directory.display()
            );
        }

        Ok(Self { presets, default })
    }

    pub fn get(&self, name: &str) -> Option<&Preset> {
        self.presets.get(name)
    }

    pub fn default_name(&self) -> &str {
        &self.default
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.presets.keys().cloned().collect();

        names.sort();
        names
    }
}
