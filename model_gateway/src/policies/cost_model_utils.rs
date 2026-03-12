// PR 2: Cost Model Utilities (plan.md §2.1–§2.2)
//!
//! JSON-based cost model for TP/PP-aware latency estimation.
//!
//! The cost model is keyed by TP/PP configuration strings (e.g., "TP8_PP1")
//! and provides linear latency estimation based on token count and request count.
//!
//! ## Latency Formula
//!
//! ```text
//! latency = attn_latency_b + attn_latency_k * token_num
//!         + max(other_threshold, other_latency_b + other_latency_k * request_num)
//! ```

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

// PR 2 §2.1: CostModelEntry — a single cost model entry for a specific TP/PP configuration.
/// A single cost model entry for a specific TP/PP configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModelEntry {
    /// Threshold for the "other" (non-attention) latency component
    pub other_threshold: f64,
    /// Base latency for non-attention compute
    pub other_latency_b: f64,
    /// Per-request latency coefficient for non-attention compute
    pub other_latency_k: f64,
    /// Base latency for attention compute
    pub attn_latency_b: f64,
    /// Per-token latency coefficient for attention compute
    pub attn_latency_k: f64,
}

impl CostModelEntry {
    // PR 2 §2.1: Estimate latency for given token count and request count.
    /// Estimate latency for given token count and request count.
    ///
    /// Formula:
    /// ```text
    /// attn_latency_b + attn_latency_k * token_num
    ///   + max(other_threshold, other_latency_b + other_latency_k * request_num)
    /// ```
    #[inline]
    pub fn estimate_latency(&self, request_num: i64, token_num: i64) -> f64 {
        let attn = self.attn_latency_b + self.attn_latency_k * token_num as f64;
        let other = self.other_latency_b + self.other_latency_k * request_num as f64;
        attn + f64::max(self.other_threshold, other)
    }

    // PR 2 §2.1: Estimate throughput: request_num / latency.
    /// Estimate throughput: request_num / latency.
    #[inline]
    pub fn estimate_throughput(&self, request_num: i64, token_num: i64) -> f64 {
        let latency = self.estimate_latency(request_num, token_num);
        if latency <= 0.0 {
            return 0.0;
        }
        request_num as f64 / latency
    }
}

// PR 2 §2.2: CostModel — cost model indexed by TP/PP configuration key.
/// Cost model indexed by TP/PP configuration key.
///
/// Loaded from a JSON file with structure:
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
    // PR 2 §2.2: Load cost model from a JSON file path.
    /// Load cost model from a JSON file path.
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let path = Path::new(path);
        let content = std::fs::read_to_string(path)?;
        let entries: HashMap<String, CostModelEntry> = serde_json::from_str(&content)?;
        info!(
            "Loaded cost model with {} entries from {path:?}",
            entries.len()
        );
        Ok(Self { entries })
    }

    // PR 2 §2.2: Load cost model from a JSON string.
    /// Load cost model from a JSON string.
    pub fn load_from_str(json_str: &str) -> Result<Self, serde_json::Error> {
        let entries: HashMap<String, CostModelEntry> = serde_json::from_str(json_str)?;
        Ok(Self { entries })
    }

    /// Get cost model entry for a given TP/PP key (e.g., "TP8_PP1").
    pub fn get(&self, tp_pp_key: &str) -> Option<&CostModelEntry> {
        self.entries.get(tp_pp_key)
    }

    /// Check if the model has an entry for the given key.
    pub fn contains(&self, tp_pp_key: &str) -> bool {
        self.entries.contains_key(tp_pp_key)
    }

    /// Get the number of entries in the cost model.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cost model is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// PR 2 §2.1–§2.2: Tests
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

    // ── §2.1 CostModelEntry tests ──

    #[test]
    fn test_estimate_latency_basic() {
        let entry = sample_entry();
        // attn = 0.05 + 0.001 * 1000 = 1.05
        // other = 0.1 + 0.02 * 10 = 0.3
        // max(0.5, 0.3) = 0.5
        // total = 1.05 + 0.5 = 1.55
        let latency = entry.estimate_latency(10, 1000);
        assert!((latency - 1.55).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_latency_other_exceeds_threshold() {
        let entry = sample_entry();
        // attn = 0.05 + 0.001 * 100 = 0.15
        // other = 0.1 + 0.02 * 50 = 1.1
        // max(0.5, 1.1) = 1.1
        // total = 0.15 + 1.1 = 1.25
        let latency = entry.estimate_latency(50, 100);
        assert!((latency - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_latency_zero_inputs() {
        let entry = sample_entry();
        // attn = 0.05 + 0.001 * 0 = 0.05
        // other = 0.1 + 0.02 * 0 = 0.1
        // max(0.5, 0.1) = 0.5
        // total = 0.05 + 0.5 = 0.55
        let latency = entry.estimate_latency(0, 0);
        assert!((latency - 0.55).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_throughput_basic() {
        let entry = sample_entry();
        let tp = entry.estimate_throughput(10, 1000);
        // latency = 1.55 (from above)
        // throughput = 10 / 1.55
        assert!((tp - 10.0 / 1.55).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_throughput_zero_requests() {
        let entry = sample_entry();
        let tp = entry.estimate_throughput(0, 1000);
        // latency > 0, but request_num = 0, so throughput = 0 / latency = 0
        assert!((tp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_throughput_zero_latency_guard() {
        // Craft a scenario where latency could be zero or negative
        let entry = CostModelEntry {
            other_threshold: 0.0,
            other_latency_b: 0.0,
            other_latency_k: 0.0,
            attn_latency_b: 0.0,
            attn_latency_k: 0.0,
        };
        let tp = entry.estimate_throughput(10, 100);
        // latency = 0.0, guard returns 0.0
        assert!((tp - 0.0).abs() < 1e-10);
    }

    // ── §2.2 CostModel tests ──

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

    #[test]
    fn test_load_from_str_success() {
        let model = CostModel::load_from_str(sample_json()).unwrap();
        assert_eq!(model.len(), 2);
        assert!(!model.is_empty());
    }

    #[test]
    fn test_load_from_str_invalid_json() {
        let result = CostModel::load_from_str("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_str_empty_object() {
        let model = CostModel::load_from_str("{}").unwrap();
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
    }

    #[test]
    fn test_get_existing_key() {
        let model = CostModel::load_from_str(sample_json()).unwrap();
        let entry = model.get("TP8_PP1");
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert!((entry.other_threshold - 0.5).abs() < 1e-10);
        assert!((entry.attn_latency_k - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_get_missing_key() {
        let model = CostModel::load_from_str(sample_json()).unwrap();
        assert!(model.get("TP16_PP1").is_none());
    }

    #[test]
    fn test_contains() {
        let model = CostModel::load_from_str(sample_json()).unwrap();
        assert!(model.contains("TP8_PP1"));
        assert!(model.contains("TP4_PP2"));
        assert!(!model.contains("TP2_PP4"));
    }

    #[test]
    fn test_len_and_is_empty() {
        let model = CostModel::load_from_str(sample_json()).unwrap();
        assert_eq!(model.len(), 2);
        assert!(!model.is_empty());

        let empty = CostModel::load_from_str("{}").unwrap();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_load_from_file_nonexistent() {
        let result = CostModel::load_from_file("/nonexistent/path/cost_model.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_file_roundtrip() {
        // Write sample JSON to a temp file, then load it
        let dir = std::env::temp_dir();
        let path = dir.join("smg_test_cost_model.json");
        std::fs::write(&path, sample_json()).unwrap();

        let model = CostModel::load_from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(model.len(), 2);
        assert!(model.contains("TP8_PP1"));

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_get_entry_and_estimate() {
        let model = CostModel::load_from_str(sample_json()).unwrap();
        let entry = model.get("TP4_PP2").unwrap();
        // attn = 0.1 + 0.002 * 500 = 1.1
        // other = 0.2 + 0.03 * 20 = 0.8
        // max(0.3, 0.8) = 0.8
        // total = 1.1 + 0.8 = 1.9
        let latency = entry.estimate_latency(20, 500);
        assert!((latency - 1.9).abs() < 1e-10);
    }
}
