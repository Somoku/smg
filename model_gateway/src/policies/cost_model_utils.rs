//! Cost-model utilities for load-aware routing policies.
//!
//! The cost model is keyed by TP/PP configuration strings (e.g., `"TP8_PP1"`)
//! and provides linear latency estimation based on token count and request count.
//!
//! ## Latency formula
//!
//! ```text
//! latency = attn_latency_b + attn_latency_k * token_num
//!         + max(other_threshold, other_latency_b + other_latency_k * request_num)
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CostModelEntry
// ---------------------------------------------------------------------------

/// A single cost model entry for a specific TP/PP configuration.
///
/// The latency formula is:
/// ```text
/// attn_latency_b + attn_latency_k * token_num
///   + max(other_threshold, other_latency_b + other_latency_k * request_num)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModelEntry {
    /// Threshold for the "other" (non-attention) latency component.
    pub other_threshold: f64,
    /// Base latency for non-attention compute.
    pub other_latency_b: f64,
    /// Per-request latency coefficient for non-attention compute.
    pub other_latency_k: f64,
    /// Base latency for attention compute.
    pub attn_latency_b: f64,
    /// Per-token latency coefficient for attention compute.
    pub attn_latency_k: f64,
}

impl CostModelEntry {
    /// Estimate forward-pass latency for the given request and token counts.
    #[inline]
    pub fn estimate_latency(&self, request_num: i64, token_num: i64) -> f64 {
        let attn = self.attn_latency_b + self.attn_latency_k * token_num as f64;
        let other = self.other_latency_b + self.other_latency_k * request_num as f64;
        attn + f64::max(self.other_threshold, other)
    }

    /// Estimate throughput: `request_num / latency`.
    ///
    /// Returns `0.0` when `request_num ≤ 0` or latency is non-positive.
    #[inline]
    pub fn estimate_throughput(&self, request_num: i64, token_num: i64) -> f64 {
        if request_num <= 0 {
            return 0.0;
        }
        let latency = self.estimate_latency(request_num, token_num);
        if latency <= 0.0 {
            return 0.0;
        }
        request_num as f64 / latency
    }
}

// ---------------------------------------------------------------------------
// CostModel
// ---------------------------------------------------------------------------

/// Cost model indexed by TP/PP configuration key (e.g., `"TP8_PP1"`).
///
/// Load from a JSON file whose top-level keys are TP/PP strings and whose
/// values are [`CostModelEntry`] objects:
///
/// ```json
/// {
///   "TP8_PP1": {
///     "other_threshold": 0.5,
///     "other_latency_b": 0.1,
///     "other_latency_k": 0.02,
///     "attn_latency_b": 0.05,
///     "attn_latency_k": 0.001
///   }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CostModel {
    entries: HashMap<String, CostModelEntry>,
}

impl CostModel {
    /// Load a [`CostModel`] from a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error (without fallback) if the file cannot be read or the
    /// JSON cannot be parsed.
    pub fn load_from_file(path: &str) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read cost-model file: {path}"))?;
        let entries: HashMap<String, CostModelEntry> = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse cost-model JSON from: {path}"))?;
        tracing::info!(path, entries = entries.len(), "loaded cost model");
        Ok(Self { entries })
    }

    /// Load a [`CostModel`] from a JSON string.
    pub fn load_from_str(json_str: &str) -> Result<Self, serde_json::Error> {
        let entries: HashMap<String, CostModelEntry> = serde_json::from_str(json_str)?;
        Ok(Self { entries })
    }

    /// Return the entry for `tp_pp_key` if present.
    #[inline]
    pub fn get(&self, tp_pp_key: &str) -> Option<&CostModelEntry> {
        self.entries.get(tp_pp_key)
    }

    /// Number of entries in the model.
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the model has no entries.
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry() -> CostModelEntry {
        CostModelEntry {
            other_threshold: 0.5,
            other_latency_b: 0.1,
            other_latency_k: 0.02,
            attn_latency_b: 0.05,
            attn_latency_k: 0.001,
        }
    }

    fn sample_json() -> &'static str {
        r#"{
            "TP8_PP1": {
                "other_threshold": 0.5,
                "other_latency_b": 0.1,
                "other_latency_k": 0.02,
                "attn_latency_b": 0.05,
                "attn_latency_k": 0.001
            },
            "TP4_PP2": {
                "other_threshold": 0.3,
                "other_latency_b": 0.2,
                "other_latency_k": 0.03,
                "attn_latency_b": 0.1,
                "attn_latency_k": 0.002
            }
        }"#
    }

    // ── CostModelEntry ──────────────────────────────────────────────────

    #[test]
    fn estimate_latency_other_below_threshold() {
        let e = sample_entry();
        // attn = 0.05 + 0.001 * 1000 = 1.05
        // other = 0.1 + 0.02 * 10 = 0.3  → max(0.5, 0.3) = 0.5
        // total = 1.55
        assert!((e.estimate_latency(10, 1000) - 1.55).abs() < 1e-10);
    }

    #[test]
    fn estimate_latency_other_exceeds_threshold() {
        let e = sample_entry();
        // attn = 0.05 + 0.001 * 100 = 0.15
        // other = 0.1 + 0.02 * 50 = 1.1  → max(0.5, 1.1) = 1.1
        // total = 1.25
        assert!((e.estimate_latency(50, 100) - 1.25).abs() < 1e-10);
    }

    #[test]
    fn estimate_latency_zero_inputs() {
        let e = sample_entry();
        // attn = 0.05, other = 0.1 → max(0.5, 0.1) = 0.5 → total = 0.55
        assert!((e.estimate_latency(0, 0) - 0.55).abs() < 1e-10);
    }

    #[test]
    fn estimate_throughput_basic() {
        let e = sample_entry();
        // latency = 1.55  →  throughput = 10 / 1.55
        assert!((e.estimate_throughput(10, 1000) - 10.0 / 1.55).abs() < 1e-10);
    }

    #[test]
    fn estimate_throughput_zero_requests() {
        let e = sample_entry();
        assert_eq!(e.estimate_throughput(0, 1000), 0.0);
    }

    #[test]
    fn estimate_throughput_zero_latency_guard() {
        let e = CostModelEntry {
            other_threshold: 0.0,
            other_latency_b: 0.0,
            other_latency_k: 0.0,
            attn_latency_b: 0.0,
            attn_latency_k: 0.0,
        };
        assert_eq!(e.estimate_throughput(10, 100), 0.0);
    }

    // ── CostModel ───────────────────────────────────────────────────────

    #[test]
    fn load_from_str_success() {
        let m = CostModel::load_from_str(sample_json()).unwrap();
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty());
    }

    #[test]
    fn load_from_str_invalid_json() {
        assert!(CostModel::load_from_str("not json").is_err());
    }

    #[test]
    fn load_from_str_empty_object() {
        let m = CostModel::load_from_str("{}").unwrap();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
    }

    #[test]
    fn get_existing_key() {
        let m = CostModel::load_from_str(sample_json()).unwrap();
        let e = m.get("TP8_PP1").unwrap();
        assert!((e.other_threshold - 0.5).abs() < 1e-10);
        assert!((e.attn_latency_k - 0.001).abs() < 1e-10);
    }

    #[test]
    fn get_missing_key() {
        let m = CostModel::load_from_str(sample_json()).unwrap();
        assert!(m.get("TP16_PP1").is_none());
    }

    #[test]
    fn load_from_file_nonexistent() {
        assert!(CostModel::load_from_file("/nonexistent/smg_cost_model.json").is_err());
    }

    #[test]
    fn load_from_file_roundtrip() {
        let path = std::env::temp_dir().join("smg_test_cost_model.json");
        std::fs::write(&path, sample_json()).unwrap();
        let m = CostModel::load_from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(m.len(), 2);
        assert!(m.get("TP8_PP1").is_some());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn entry_latency_and_throughput_via_model() {
        let m = CostModel::load_from_str(sample_json()).unwrap();
        let e = m.get("TP4_PP2").unwrap();
        // attn = 0.1 + 0.002 * 500 = 1.1
        // other = 0.2 + 0.03 * 20 = 0.8  → max(0.3, 0.8) = 0.8
        // total = 1.9
        assert!((e.estimate_latency(20, 500) - 1.9).abs() < 1e-10);
    }
}
