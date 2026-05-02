use axum::{
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

/// Batch request body for `POST /workers/update_weight_version`.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerWeightVersionUpdateRequest {
    pub updates: Vec<WorkerWeightVersionUpdateRequestItem>,
}

/// Single runtime weight version update targeting a worker.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerWeightVersionUpdateRequestItem {
    /// Direct worker ID, or base worker ID when `dp_rank` is present.
    pub worker_id: String,
    /// Target DP rank under `worker_id` when updating a DP worker.
    #[serde(default)]
    pub dp_rank: Option<usize>,
    pub weight_version: u64,
}

/// Aggregate result of a batch runtime weight version update.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerWeightVersionUpdateResult {
    pub total: usize,
    pub updated: usize,
    pub rejected: usize,
    pub results: Vec<WorkerWeightVersionUpdateResultItem>,
}

impl IntoResponse for WorkerWeightVersionUpdateResult {
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}

/// Per-worker runtime weight version update result.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerWeightVersionUpdateResultItem {
    pub status: String,
    pub worker_id: String,
    pub url: String,
    pub weight_version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}