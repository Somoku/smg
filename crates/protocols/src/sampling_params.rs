use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use validator::Validate;

use super::common::StringOrArray;

// ============================================================================
// PR 9: StructuredOutputsParams (plan.md §9.1)
// ============================================================================

/// vLLM-style structured output parameters.
///
/// Exactly one of the constraint fields must be set. This provides richer
/// structured output control than the flat `json_schema`/`regex`/`ebnf` fields
/// on `SamplingParams`, supporting additional modes: `choice`, `grammar`,
/// `json_object`, and `structural_tag`.
///
/// **Conflict rule**: If `structured_outputs` is set on `SamplingParams`, the
/// flat fields (`json_schema`, `regex`, `ebnf`) must all be `None`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_structured_outputs_params"))]
pub struct StructuredOutputsParams {
    /// JSON schema (string or object)
    pub json: Option<serde_json::Value>,
    /// Regex pattern
    pub regex: Option<String>,
    /// List of allowed string choices
    pub choice: Option<Vec<String>>,
    /// Grammar / EBNF string
    pub grammar: Option<String>,
    /// Force JSON object output
    pub json_object: Option<bool>,
    /// Structural tag (e.g., Harmony models)
    pub structural_tag: Option<String>,

    // Options
    pub disable_fallback: Option<bool>,
    pub disable_any_whitespace: Option<bool>,
    pub disable_additional_properties: Option<bool>,
    pub whitespace_pattern: Option<String>,
}

fn validate_structured_outputs_params(
    params: &StructuredOutputsParams,
) -> Result<(), validator::ValidationError> {
    let count = [
        params.json.is_some(),
        params.regex.is_some(),
        params.choice.is_some(),
        params.grammar.is_some(),
        params.json_object.is_some(),
        params.structural_tag.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    if count > 1 {
        return Err(validator::ValidationError::new(
            "structured_outputs: set exactly one of json/regex/choice/grammar/json_object/structural_tag",
        ));
    }

    Ok(())
}

// ============================================================================
// PR 9: Extended SamplingParams (plan.md §9.2)
// ============================================================================

/// Sampling parameters for text generation.
///
/// Supports fields from SGLang, vLLM, and TRT-LLM backends. Cross-backend
/// conflict validation ensures users don't set mutually exclusive fields.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_sampling_params"))]
pub struct SamplingParams {
    /// Temperature for sampling (must be >= 0.0, no upper limit)
    #[validate(range(min = 0.0))]
    pub temperature: Option<f32>,

    /// Maximum number of new tokens to generate (must be >= 0).
    /// SGLang calls this `max_new_tokens`; vLLM calls it `max_tokens`.
    // PR 9 §9.2a: serde alias for vLLM compatibility
    #[serde(alias = "max_tokens")]
    #[validate(range(min = 0))]
    pub max_new_tokens: Option<u32>,

    /// Top-p nucleus sampling (0.0 < top_p <= 1.0)
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// Top-k sampling (-1 to disable, or >= 1)
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: Option<i32>,

    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,

    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,

    pub stop: Option<StringOrArray>,
    pub ignore_eos: Option<bool>,
    pub skip_special_tokens: Option<bool>,
    pub json_schema: Option<String>,
    pub regex: Option<String>,
    pub ebnf: Option<String>,

    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,

    /// Minimum number of new tokens.
    /// SGLang calls this `min_new_tokens`; vLLM calls it `min_tokens`.
    // PR 9 §9.2a: serde alias for vLLM compatibility
    #[serde(alias = "min_tokens")]
    pub min_new_tokens: Option<u32>,

    pub stop_token_ids: Option<Vec<u32>>,

    /// Whether to include stop strings in the output text.
    /// SGLang calls this `no_stop_trim`; vLLM calls it `include_stop_str_in_output`.
    /// `no_stop_trim` is the canonical name; `include_stop_str_in_output` accepted as alias.
    // PR 9 §9.2a: serde alias for vLLM compatibility
    #[serde(alias = "include_stop_str_in_output")]
    pub no_stop_trim: Option<bool>,

    pub n: Option<u32>,

    /// Random seed (SGLang convention, unsigned).
    /// **Conflict**: Cannot be set simultaneously with `seed`.
    pub sampling_seed: Option<u64>,

    // ── PR 9 §9.2b: NEW fields for multi-backend support ──

    /// Number of log probabilities per output token. -1 for all vocab.
    /// Carried inside SamplingParams for vLLM; for SGLang, mapped to
    /// top-level GenerateRequest fields as a fallback.
    pub logprobs: Option<i32>,

    /// Number of log probabilities per prompt token. -1 for all vocab.
    pub prompt_logprobs: Option<i32>,

    /// Random seed for reproducibility (vLLM/TRT-LLM naming, signed).
    /// **Conflict**: Cannot be set simultaneously with `sampling_seed`.
    /// The type is `i64` to accommodate vLLM's signed semantics
    /// (`seed = -1` means "no seed" in vLLM).
    pub seed: Option<i64>,

    /// Token ID to logit bias mapping (-100 to 100). Standard OpenAI field.
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Prompt truncation: -1 for model max, >= 1 for explicit limit.
    /// vLLM-specific.
    pub truncate_prompt_tokens: Option<i32>,

    /// Whether to add spaces between special tokens.
    pub spaces_between_special_tokens: Option<bool>,

    /// vLLM-style structured outputs object. Supports `choice`, `grammar`,
    /// `json_object`, and `structural_tag` in addition to `json`/`regex`.
    /// **Conflict**: Cannot be set simultaneously with flat `json_schema`,
    /// `regex`, or `ebnf` fields.
    #[validate(nested)]
    pub structured_outputs: Option<StructuredOutputsParams>,

    /// Arbitrary extra arguments for backend-specific passthrough.
    /// Not validated by SMG; forwarded to the backend as-is.
    pub extra_args: Option<serde_json::Map<String, serde_json::Value>>,
}

// ============================================================================
// Shared Validation Functions
// ============================================================================

/// Validates top_p: 0.0 < top_p <= 1.0 (can't use range validator for open interval)
pub fn validate_top_p_value(top_p: f32) -> Result<(), validator::ValidationError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(validator::ValidationError::new(
            "top_p must be in (0, 1] - greater than 0.0 and at most 1.0",
        ));
    }
    Ok(())
}

/// Validates top_k: -1 (disabled) or >= 1 (special -1 case - can't use range validator)
pub fn validate_top_k_value(top_k: i32) -> Result<(), validator::ValidationError> {
    if top_k != -1 && top_k < 1 {
        return Err(validator::ValidationError::new(
            "top_k must be -1 (disabled) or at least 1",
        ));
    }
    Ok(())
}

// ============================================================================
// PR 9: Cross-backend conflict validation (plan.md §9.3)
// ============================================================================

/// Validation function for SamplingParams - cross-field and cross-backend validation
fn validate_sampling_params(params: &SamplingParams) -> Result<(), validator::ValidationError> {
    // 1. Cross-field validation: min_new_tokens <= max_new_tokens
    if let (Some(min), Some(max)) = (params.min_new_tokens, params.max_new_tokens) {
        if min > max {
            return Err(validator::ValidationError::new(
                "min_new_tokens cannot exceed max_new_tokens",
            ));
        }
    }

    // 2. Validate mutually exclusive structured output constraints
    //    Flat fields (json_schema, regex, ebnf) conflict with nested structured_outputs
    let has_flat_constraint =
        params.json_schema.is_some() || params.regex.is_some() || params.ebnf.is_some();
    let has_nested_constraint = params.structured_outputs.is_some();

    if has_flat_constraint && has_nested_constraint {
        return Err(validator::ValidationError::new(
            "cannot set both flat constraint fields (json_schema/regex/ebnf) and structured_outputs",
        ));
    }

    // Existing flat mutual exclusivity check
    if has_flat_constraint {
        let flat_count = [
            params.json_schema.is_some(),
            params.regex.is_some(),
            params.ebnf.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count();

        if flat_count > 1 {
            return Err(validator::ValidationError::new(
                "only one of json_schema, regex, or ebnf can be set",
            ));
        }
    }

    // 3. Validate seed conflict: sampling_seed (u64) and seed (i64) are the same
    //    concept with different types — must not be set simultaneously
    if params.sampling_seed.is_some() && params.seed.is_some() {
        return Err(validator::ValidationError::new(
            "cannot set both sampling_seed (SGLang u64) and seed (vLLM i64) simultaneously",
        ));
    }

    // 4. Validate logprobs: -1 or >= 0
    if let Some(lp) = params.logprobs {
        if lp < -1 {
            return Err(validator::ValidationError::new(
                "logprobs must be -1 (all vocab) or >= 0",
            ));
        }
    }
    if let Some(plp) = params.prompt_logprobs {
        if plp < -1 {
            return Err(validator::ValidationError::new(
                "prompt_logprobs must be -1 (all vocab) or >= 0",
            ));
        }
    }

    // 5. Validate truncate_prompt_tokens: -1 or >= 1
    if let Some(v) = params.truncate_prompt_tokens {
        if v == 0 || v < -1 {
            return Err(validator::ValidationError::new(
                "truncate_prompt_tokens must be -1 (model max) or >= 1",
            ));
        }
    }

    Ok(())
}

// ============================================================================
// PR 9: Seed resolution helper (plan.md §9.2)
// ============================================================================

/// Resolve the effective seed, preferring `seed` (i64) over `sampling_seed` (u64).
/// Returns None if neither is set.
/// Callers should validate that both are not set simultaneously before calling.
pub fn resolve_seed(params: &SamplingParams) -> Option<i64> {
    // seed takes priority if set (already validated that both aren't set)
    params
        .seed
        .or_else(|| params.sampling_seed.map(|s| s as i64))
}

// ============================================================================
// PR 9: Tests (plan.md §9.8)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_and_structured_outputs_conflict() {
        let params = SamplingParams {
            json_schema: Some("{}".to_string()),
            structured_outputs: Some(StructuredOutputsParams {
                regex: Some(".*".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_structured_outputs_single_constraint() {
        let params = SamplingParams {
            structured_outputs: Some(StructuredOutputsParams {
                choice: Some(vec!["a".to_string(), "b".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_structured_outputs_multiple_constraints_rejected() {
        let so = StructuredOutputsParams {
            json: Some(serde_json::json!({"type": "object"})),
            regex: Some(".*".to_string()),
            ..Default::default()
        };
        assert!(so.validate().is_err());
    }

    #[test]
    fn test_logprobs_validation() {
        // Invalid: -2
        let params = SamplingParams {
            logprobs: Some(-2),
            ..Default::default()
        };
        assert!(params.validate().is_err());

        // Valid: -1 (all vocab)
        let params = SamplingParams {
            logprobs: Some(-1),
            ..Default::default()
        };
        assert!(params.validate().is_ok());

        // Valid: 5
        let params = SamplingParams {
            logprobs: Some(5),
            ..Default::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_prompt_logprobs_validation() {
        let params = SamplingParams {
            prompt_logprobs: Some(-2),
            ..Default::default()
        };
        assert!(params.validate().is_err());

        let params = SamplingParams {
            prompt_logprobs: Some(-1),
            ..Default::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_truncate_prompt_tokens_validation() {
        // Invalid: 0
        let params = SamplingParams {
            truncate_prompt_tokens: Some(0),
            ..Default::default()
        };
        assert!(params.validate().is_err());

        // Valid: -1 (model max)
        let params = SamplingParams {
            truncate_prompt_tokens: Some(-1),
            ..Default::default()
        };
        assert!(params.validate().is_ok());

        // Valid: 100
        let params = SamplingParams {
            truncate_prompt_tokens: Some(100),
            ..Default::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_seed_conflict_rejected() {
        let params = SamplingParams {
            sampling_seed: Some(42),
            seed: Some(42),
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_seed_only_sampling_seed() {
        let params = SamplingParams {
            sampling_seed: Some(42),
            ..Default::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_seed_only_seed() {
        let params = SamplingParams {
            seed: Some(123),
            ..Default::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_seed_vllm_negative_one() {
        // vLLM uses seed = -1 to mean "no seed"
        let params = SamplingParams {
            seed: Some(-1),
            ..Default::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_no_stop_trim_alias_deserialization() {
        // no_stop_trim is the canonical field
        let json = r#"{"no_stop_trim": true}"#;
        let params: SamplingParams = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(params.no_stop_trim, Some(true));

        // include_stop_str_in_output is the alias, maps to no_stop_trim
        let json = r#"{"include_stop_str_in_output": false}"#;
        let params: SamplingParams = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(params.no_stop_trim, Some(false));
    }

    #[test]
    fn test_max_tokens_alias_deserialization() {
        let json = r#"{"max_tokens": 256}"#;
        let params: SamplingParams = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(params.max_new_tokens, Some(256));

        let json = r#"{"max_new_tokens": 128}"#;
        let params: SamplingParams = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(params.max_new_tokens, Some(128));
    }

    #[test]
    fn test_min_tokens_alias_deserialization() {
        let json = r#"{"min_tokens": 10}"#;
        let params: SamplingParams = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(params.min_new_tokens, Some(10));
    }

    #[test]
    fn test_resolve_seed_prefers_seed() {
        let params = SamplingParams {
            seed: Some(42),
            ..Default::default()
        };
        assert_eq!(resolve_seed(&params), Some(42));
    }

    #[test]
    fn test_resolve_seed_falls_back_to_sampling_seed() {
        let params = SamplingParams {
            sampling_seed: Some(99),
            ..Default::default()
        };
        assert_eq!(resolve_seed(&params), Some(99));
    }

    #[test]
    fn test_resolve_seed_neither_set() {
        let params = SamplingParams::default();
        assert_eq!(resolve_seed(&params), None);
    }

    #[test]
    fn test_existing_flat_mutual_exclusivity() {
        // Existing behavior: json_schema + regex conflict
        let params = SamplingParams {
            json_schema: Some("{}".to_string()),
            regex: Some(".*".to_string()),
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_default_params_valid() {
        let params = SamplingParams::default();
        assert!(params.validate().is_ok());
    }
}
