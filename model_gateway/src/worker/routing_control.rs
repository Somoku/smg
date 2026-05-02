use axum::{
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

/// DP rank selector accepted by worker pause/resume requests.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DpRankInput {
    Single(usize),
    Multiple(Vec<usize>),
}

impl DpRankInput {
    pub fn to_ranks(&self) -> Result<Vec<usize>, String> {
        let mut ranks = match self {
            Self::Single(rank) => vec![*rank],
            Self::Multiple(ranks) => ranks.clone(),
        };

        if ranks.is_empty() {
            return Err("dp_rank list cannot be empty".to_string());
        }

        ranks.sort_unstable();
        ranks.dedup();
        Ok(ranks)
    }
}

/// Single target in a worker pause/resume request.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerRoutingControlTargetRequest {
    /// Direct worker ID. If `dp_rank` is present, this is treated as the base worker ID.
    #[serde(default)]
    pub worker_id: Option<String>,
    /// Base worker ID for targeting all DP ranks or a specific `dp_rank`.
    #[serde(default)]
    pub base_worker_id: Option<String>,
    #[serde(default)]
    pub dp_rank: Option<DpRankInput>,
}

/// Request body for `POST /workers/pause` and `POST /workers/resume`.
pub type WorkerRoutingControlRequest = Vec<WorkerRoutingControlTargetRequest>;

/// Aggregate result of a batch worker routing-control operation.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerRoutingControlResult {
    pub action: String,
    pub total: usize,
    pub updated: usize,
    pub rejected: usize,
    pub results: Vec<WorkerRoutingControlResultItem>,
}

impl IntoResponse for WorkerRoutingControlResult {
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}

/// Per-worker result within a batch worker routing-control operation.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerRoutingControlResultItem {
    pub status: String,
    pub worker_id: String,
    pub url: String,
    pub paused: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_worker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}
