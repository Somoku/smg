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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineSchedulerStats {
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
        self.scheduler_stats
            .req_id_to_prompt_token_num
            .values()
            .sum::<usize>()
            + self
                .scheduler_stats
                .req_id_to_response_token_num
                .values()
                .sum::<usize>()
    }

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

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerStatsUpdateRequest {
    pub updates: Vec<WorkerStatsUpdateRequestItem>,
}

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
