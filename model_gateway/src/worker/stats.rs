use std::collections::HashMap;

use axum::{
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

fn default_timestamp() -> DateTime<Utc> {
    Utc::now()
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineSchedulerStats {
    /// Per-request-id token maps deserialized from the engine heartbeat.
    ///
    /// These maps are kept for debugging and detailed per-request inspection.
    #[serde(default)]
    pub req_id_to_prompt_token_num: HashMap<String, usize>,
    #[serde(default)]
    pub req_id_to_response_token_num: HashMap<String, usize>,
    #[serde(default)]
    pub num_running_reqs: usize,
    #[serde(default)]
    pub num_waiting_reqs: usize,
    #[serde(default)]
    pub kv_cache_usage: f64,
    /// For hot-path policy selection use [`EngineSchedulerStats::total_prompt_tokens`]
    /// and [`EngineSchedulerStats::total_response_tokens`] instead — they are
    /// pre-aggregated at deserialisation time so that `total_token_num()` and
    /// `token_num_with_budget()` are O(1) reads.
    ///
    /// Sum of all values in `req_id_to_prompt_token_num`.
    /// Pre-aggregated at construction / deserialisation; not serialised.
    #[serde(skip)]
    pub total_prompt_tokens: usize,
    /// Sum of all values in `req_id_to_response_token_num`.
    /// Pre-aggregated at construction / deserialisation; not serialised.
    #[serde(skip)]
    pub total_response_tokens: usize,
}

impl EngineSchedulerStats {
    pub fn recompute_aggregates(&mut self) {
        self.total_prompt_tokens = self.req_id_to_prompt_token_num.values().sum();
        self.total_response_tokens = self.req_id_to_response_token_num.values().sum();
    }
}

impl Default for EngineSchedulerStats {
    fn default() -> Self {
        Self {
            req_id_to_prompt_token_num: HashMap::new(),
            req_id_to_response_token_num: HashMap::new(),
            num_running_reqs: 0,
            num_waiting_reqs: 0,
            kv_cache_usage: 0.0,
            total_prompt_tokens: 0,
            total_response_tokens: 0,
        }
    }
}

impl<'de> Deserialize<'de> for EngineSchedulerStats {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Intermediate helper that only holds the wire fields (serde-derived).
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default)]
            req_id_to_prompt_token_num: HashMap<String, usize>,
            #[serde(default)]
            req_id_to_response_token_num: HashMap<String, usize>,
            #[serde(default)]
            num_running_reqs: usize,
            #[serde(default)]
            num_waiting_reqs: usize,
            #[serde(default)]
            kv_cache_usage: f64,
        }

        let wire = Wire::deserialize(deserializer)?;
        let total_prompt_tokens = wire.req_id_to_prompt_token_num.values().sum();
        let total_response_tokens = wire.req_id_to_response_token_num.values().sum();
        Ok(Self {
            req_id_to_prompt_token_num: wire.req_id_to_prompt_token_num,
            req_id_to_response_token_num: wire.req_id_to_response_token_num,
            num_running_reqs: wire.num_running_reqs,
            num_waiting_reqs: wire.num_waiting_reqs,
            kv_cache_usage: wire.kv_cache_usage,
            total_prompt_tokens,
            total_response_tokens,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    #[serde(default = "default_timestamp")]
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub scheduler_stats: EngineSchedulerStats,
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            timestamp: default_timestamp(),
            scheduler_stats: EngineSchedulerStats::default(),
        }
    }
}

impl EngineStats {
    pub fn waiting_queue_size(&self) -> usize {
        self.scheduler_stats.num_waiting_reqs
    }

    pub fn running_queue_size(&self) -> usize {
        self.scheduler_stats.num_running_reqs
    }

    pub fn waiting_and_running_queue_size(&self) -> usize {
        self.waiting_queue_size() + self.running_queue_size()
    }

    pub fn total_token_num(&self) -> usize {
        self.scheduler_stats.total_prompt_tokens + self.scheduler_stats.total_response_tokens
    }

    /// For each running request the contribution is:
    ///   `prompt_tokens + ceil((response_tokens + 1) / budget) * budget`
    ///
    /// The `+1` ensures that a request generating zero response tokens still
    /// occupies at least one budget unit (avoids treating in-flight requests
    /// as free).
    ///
    /// Iterates over the per-request prompt-token map for an accurate per-request
    /// calculation.  Returns `0` when no requests are tracked.
    pub fn token_num_with_budget(&self, request_budget: usize) -> usize {
        let budget = request_budget.max(1);
        self.scheduler_stats
            .req_id_to_prompt_token_num
            .iter()
            .map(|(req_id, prompt_tokens)| {
                let response_tokens = self
                    .scheduler_stats
                    .req_id_to_response_token_num
                    .get(req_id)
                    .copied()
                    .unwrap_or(0);
                prompt_tokens + (response_tokens + 1).div_ceil(budget) * budget
            })
            .sum()
    }
}

#[derive(Debug, Clone)]
pub enum EngineStatsUpdateOutcome {
    Applied,
    Stale { reason: String },
    Rejected { reason: String },
}

/// Request body for `POST /workers/update_stats`.
pub type WorkerStatsUpdateRequest = Vec<WorkerStatsUpdateRequestItem>;

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerStatsUpdateRequestItem {
    #[serde(default)]
    /// Direct worker ID, or base worker ID when `dp_rank` is present.
    pub worker_id: String,
    #[serde(default)]
    /// Not be present when `worker_id` is a direct worker ID.
    pub dp_rank: Option<usize>,
    #[serde(flatten)]
    pub stats: EngineStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerStatsUpdateResult {
    pub total: usize,
    pub updated: usize,
    pub stale_ignored: usize,
    pub rejected: usize,
    pub results: Vec<WorkerStatsUpdateResultItem>,
}

impl IntoResponse for WorkerStatsUpdateResult {
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerStatsUpdateResultItem {
    pub status: String,
    pub worker_id: String,
    pub url: String,
    pub applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Build an [`EngineStats`] with the given per-request token maps and queue sizes.
    fn make_stats(
        prompt: HashMap<&str, usize>,
        response: HashMap<&str, usize>,
        running: usize,
        waiting: usize,
    ) -> EngineStats {
        let req_id_to_prompt_token_num: HashMap<String, usize> = prompt
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let req_id_to_response_token_num: HashMap<String, usize> = response
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let total_prompt_tokens = req_id_to_prompt_token_num.values().sum();
        let total_response_tokens = req_id_to_response_token_num.values().sum();
        EngineStats {
            timestamp: Utc::now(),
            scheduler_stats: EngineSchedulerStats {
                req_id_to_prompt_token_num,
                req_id_to_response_token_num,
                num_running_reqs: running,
                num_waiting_reqs: waiting,
                kv_cache_usage: 0.0,
                total_prompt_tokens,
                total_response_tokens,
            },
        }
    }

    #[test]
    fn queue_sizes() {
        let s = make_stats(HashMap::new(), HashMap::new(), 3, 2);
        assert_eq!(s.running_queue_size(), 3);
        assert_eq!(s.waiting_queue_size(), 2);
        assert_eq!(s.waiting_and_running_queue_size(), 5);
    }

    #[test]
    fn total_token_num_sums_pre_aggregated() {
        // 100 prompt + 15 response = 115
        let s = make_stats(
            HashMap::from([("r1", 100)]),
            HashMap::from([("r1", 15)]),
            1,
            0,
        );
        assert_eq!(s.total_token_num(), 115);
    }

    #[test]
    fn total_token_num_empty() {
        let s = EngineStats::default();
        assert_eq!(s.total_token_num(), 0);
    }

    #[test]
    fn token_num_with_budget_empty_map_returns_zero() {
        let s = EngineStats::default();
        assert_eq!(s.token_num_with_budget(16), 0);
        assert_eq!(s.token_num_with_budget(1), 0);
    }

    #[test]
    fn token_num_with_budget_single_request() {
        // prompt=100, response=15, budget=16
        // contribution = 100 + ceil((15+1)/16)*16 = 100 + 1*16 = 116
        let s = make_stats(
            HashMap::from([("r1", 100)]),
            HashMap::from([("r1", 15)]),
            1,
            0,
        );
        assert_eq!(s.token_num_with_budget(16), 116);
    }

    #[test]
    fn token_num_with_budget_response_zero_counts_one_unit() {
        // prompt=50, response=0, budget=16
        // contribution = 50 + ceil((0+1)/16)*16 = 50 + 1*16 = 66
        let s = make_stats(HashMap::from([("r1", 50)]), HashMap::new(), 1, 0);
        assert_eq!(s.token_num_with_budget(16), 66);
    }

    #[test]
    fn token_num_with_budget_exact_multiple() {
        // prompt=100, response=32, budget=16
        // contribution = 100 + ceil((32+1)/16)*16 = 100 + 3*16 = 148
        let s = make_stats(
            HashMap::from([("r1", 100)]),
            HashMap::from([("r1", 32)]),
            1,
            0,
        );
        assert_eq!(s.token_num_with_budget(16), 148);
    }

    #[test]
    fn token_num_with_budget_multiple_requests() {
        // r1: prompt=100, response=15, budget=16 → 100 + ceil(16/16)*16 = 116
        // r2: prompt=200, response=0,  budget=16 → 200 + ceil(1/16)*16  = 216
        // total = 332
        let s = make_stats(
            HashMap::from([("r1", 100), ("r2", 200)]),
            HashMap::from([("r1", 15)]),
            2,
            0,
        );
        assert_eq!(s.token_num_with_budget(16), 332);
    }

    #[test]
    fn token_num_with_budget_zero_budget_treated_as_one() {
        // budget=0 → treated as 1 → ceil((15+1)/1)*1 = 16
        // contribution = 100 + 16 = 116
        let s = make_stats(
            HashMap::from([("r1", 100)]),
            HashMap::from([("r1", 15)]),
            1,
            0,
        );
        assert_eq!(s.token_num_with_budget(0), 116);
    }

    #[test]
    fn deserialise_aggregates_pre_computed() {
        let json = r#"{
            "req_id_to_prompt_token_num":   {"a": 10, "b": 20},
            "req_id_to_response_token_num": {"a": 5},
            "num_running_reqs": 2,
            "num_waiting_reqs": 1,
            "kv_cache_usage": 0.3
        }"#;
        let s: EngineSchedulerStats = serde_json::from_str(json).unwrap();
        assert_eq!(s.total_prompt_tokens, 30);
        assert_eq!(s.total_response_tokens, 5);
        assert_eq!(s.num_running_reqs, 2);
        assert_eq!(s.num_waiting_reqs, 1);
    }

    #[test]
    fn recompute_aggregates_after_manual_modification() {
        let mut s = EngineSchedulerStats::default();
        s.req_id_to_prompt_token_num.insert("x".to_string(), 42);
        s.req_id_to_response_token_num.insert("x".to_string(), 8);
        s.recompute_aggregates();
        assert_eq!(s.total_prompt_tokens, 42);
        assert_eq!(s.total_response_tokens, 8);
    }
}
