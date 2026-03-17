//! Worker Service - Business logic layer for worker operations
//!
//! This module provides a clean separation between HTTP concerns (in routers)
//! and business logic for worker management. The service orchestrates
//! WorkerRegistry and JobQueue operations.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::worker::{WorkerErrorResponse, WorkerInfo, WorkerSpec, WorkerUpdateRequest};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, warn};

use crate::{
    config::RouterConfig,
    core::{
        worker::worker_to_info, worker_registry::WorkerId, EngineStats, EngineStatsUpdateOutcome,
        Job, JobQueue, Worker, WorkerRegistry,
    },
};

/// Shared map tracking `(worker_id, dp_rank) → version_tag` after sync.
///
/// Used by the routing loop for version-based filtering and loopback
/// version updates. Shared between `AppContext`, `WorkerService`, and
/// `RoutingLoopRuntime`.
pub type InstanceVersionMap = Arc<Mutex<HashMap<(String, usize), i64>>>;

/// Error types for worker service operations
#[derive(Debug)]
pub enum WorkerServiceError {
    /// Worker with given ID was not found
    NotFound { worker_id: String },
    /// Invalid worker ID format (expected UUID)
    InvalidId { raw: String, message: String },
    /// Job queue not initialized
    QueueNotInitialized,
    /// Failed to submit job to queue
    QueueSubmitFailed { message: String },
}

impl WorkerServiceError {
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::NotFound { .. } => "WORKER_NOT_FOUND",
            Self::InvalidId { .. } => "BAD_REQUEST",
            Self::QueueNotInitialized => "INTERNAL_SERVER_ERROR",
            Self::QueueSubmitFailed { .. } => "INTERNAL_SERVER_ERROR",
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound { .. } => StatusCode::NOT_FOUND,
            Self::InvalidId { .. } => StatusCode::BAD_REQUEST,
            Self::QueueNotInitialized => StatusCode::INTERNAL_SERVER_ERROR,
            Self::QueueSubmitFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for WorkerServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { worker_id } => write!(f, "Worker {worker_id} not found"),
            Self::InvalidId { raw, message } => {
                write!(
                    f,
                    "Invalid worker_id '{raw}' (expected UUID). Error: {message}"
                )
            }
            Self::QueueNotInitialized => write!(f, "Job queue not initialized"),
            Self::QueueSubmitFailed { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for WorkerServiceError {}

impl IntoResponse for WorkerServiceError {
    fn into_response(self) -> Response {
        let error = WorkerErrorResponse {
            error: self.to_string(),
            code: self.error_code().to_string(),
        };
        (self.status_code(), Json(error)).into_response()
    }
}

/// Result of creating a worker (async job submission)
#[derive(Debug)]
pub struct CreateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
    pub location: String,
}

impl IntoResponse for CreateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "url": self.url,
            "location": self.location,
            "message": "Worker addition queued for background processing"
        });
        (
            StatusCode::ACCEPTED,
            [(http::header::LOCATION, self.location)],
            Json(response),
        )
            .into_response()
    }
}

/// Result of deleting a worker (async job submission)
#[derive(Debug)]
pub struct DeleteWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for DeleteWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker removal queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of updating a worker (async job submission)
#[derive(Debug)]
pub struct UpdateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for UpdateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker update queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of listing workers
#[derive(Debug)]
pub struct ListWorkersResult {
    pub workers: Vec<WorkerInfo>,
    pub total: usize,
    pub prefill_count: usize,
    pub decode_count: usize,
    pub regular_count: usize,
}

impl IntoResponse for ListWorkersResult {
    fn into_response(self) -> Response {
        let response = json!({
            "workers": self.workers,
            "total": self.total,
            "stats": {
                "prefill_count": self.prefill_count,
                "decode_count": self.decode_count,
                "regular_count": self.regular_count,
            }
        });
        Json(response).into_response()
    }
}

/// Wrapper for WorkerInfo to implement IntoResponse
pub struct GetWorkerResponse(pub WorkerInfo);

impl IntoResponse for GetWorkerResponse {
    fn into_response(self) -> Response {
        Json(self.0).into_response()
    }
}

// ============================================================================
// PR 1 §1.5: Engine stats push endpoint types
// ============================================================================

// PR 1 §1.5: Batch request for engine stats updates
/// Batch request body for `POST /workers/stats`.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerStatsUpdateRequest {
    pub updates: Vec<WorkerStatsTargetUpdateRequest>,
}

// PR 1 §1.5: Individual stats update targeting a single worker
/// Single worker stats update within a batch request.
///
/// Target resolution modes (mutually exclusive):
/// - `worker_id`: direct worker UUID
/// - `base_worker_id` + `dp_rank`: DP-aware resolution
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerStatsTargetUpdateRequest {
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub base_worker_id: Option<String>,
    #[serde(default)]
    pub dp_rank: Option<usize>,
    #[serde(flatten)]
    pub stats: EngineStats,
}

// PR 1 §1.5: Result of batch stats update
/// Aggregate result of a batch stats update.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateWorkerStatsResult {
    pub total: usize,
    pub updated: usize,
    pub stale_ignored: usize,
    pub rejected: usize,
    pub results: Vec<WorkerStatsUpdateItemResult>,
}

impl IntoResponse for UpdateWorkerStatsResult {
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}

// PR 1 §1.5: Per-item result for stats update
/// Per-worker result within a batch stats update response.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerStatsUpdateItemResult {
    pub status: String,
    pub worker_id: String,
    pub url: String,
    pub applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_worker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale_reason: Option<String>,
}

// ============================================================================
// PR 1 §1.5: Version tag update endpoint types
// ============================================================================

// PR 1 §1.5: Batch request for version tag updates
/// Batch request body for `POST /workers/version_tag`.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerVersionTagUpdateRequest {
    pub updates: Vec<WorkerVersionTagTargetUpdateRequest>,
}

// PR 1 §1.5: Individual version tag update
/// Single worker version tag update within a batch request.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerVersionTagTargetUpdateRequest {
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub base_worker_id: Option<String>,
    #[serde(default)]
    pub dp_rank: Option<usize>,
    pub version_tag: i64,
}

// PR 1 §1.5: Result of batch version tag update
/// Aggregate result of a batch version tag update.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateWorkerVersionTagResult {
    pub total: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub results: Vec<WorkerVersionTagUpdateItemResult>,
}

impl IntoResponse for UpdateWorkerVersionTagResult {
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}

// PR 1 §1.5: Per-item result for version tag update
/// Per-worker result within a batch version tag update response.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerVersionTagUpdateItemResult {
    pub status: String,
    pub worker_id: String,
    pub url: String,
    pub version_tag: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_worker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

// ============================================================================
// PR 1 §1.5: Routing control (pause/resume) endpoint types
// ============================================================================

// PR 1 §1.5: DP rank input (single or multiple)
/// DP rank specifier — single rank or a list of ranks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DpRankInput {
    Single(usize),
    Multiple(Vec<usize>),
}

impl DpRankInput {
    fn to_ranks(&self) -> Result<Vec<usize>, WorkerServiceError> {
        let mut ranks = match self {
            Self::Single(rank) => vec![*rank],
            Self::Multiple(ranks) => ranks.clone(),
        };

        if ranks.is_empty() {
            return Err(WorkerServiceError::InvalidId {
                raw: "dp_rank".to_string(),
                message: "dp_rank list cannot be empty".to_string(),
            });
        }

        ranks.sort_unstable();
        ranks.dedup();
        Ok(ranks)
    }
}

// PR 1 §1.5: Single routing control target
/// Target for a pause/resume operation.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerRoutingControlTargetRequest {
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub base_worker_id: Option<String>,
    #[serde(default)]
    pub dp_rank: Option<DpRankInput>,
}

// PR 1 §1.5: Batch routing control request
/// Request body for `POST /workers/pause` and `POST /workers/resume`.
pub type WorkerRoutingControlRequest = Vec<WorkerRoutingControlTargetRequest>;

// PR 1 §1.5: Result of batch routing control operation
/// Aggregate result of a batch pause/resume operation.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateWorkerRoutingResult {
    pub action: String,
    pub total: usize,
    pub updated: usize,
    pub rejected: usize,
    pub results: Vec<WorkerRoutingControlItemResult>,
}

impl IntoResponse for UpdateWorkerRoutingResult {
    fn into_response(self) -> Response {
        Json(self).into_response()
    }
}

// PR 1 §1.5: Per-item result for routing control
/// Per-worker result within a batch pause/resume response.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerRoutingControlItemResult {
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

// PR 1 §1.5: Type alias for resolved worker list to reduce type complexity
type ResolvedWorkers = Vec<(WorkerId, Arc<dyn Worker>)>;

/// Worker Service - Orchestrates worker business logic
///
/// This service provides a clean API for worker operations, separating
/// business logic from HTTP concerns. Handlers in server.rs become thin
/// wrappers that translate between HTTP and this service.
pub struct WorkerService {
    worker_registry: Arc<WorkerRegistry>,
    job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
    router_config: RouterConfig,
    /// Shared map tracking `(worker_id, dp_rank) → version_tag` after sync.
    instance_to_version_after_sync: InstanceVersionMap,
}

impl WorkerService {
    /// Create a new WorkerService
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
        router_config: RouterConfig,
        instance_to_version_after_sync: InstanceVersionMap,
    ) -> Self {
        Self {
            worker_registry,
            job_queue,
            router_config,
            instance_to_version_after_sync,
        }
    }

    /// Get a reference to the shared instance-to-version map.
    pub fn instance_to_version_map(&self) -> &InstanceVersionMap {
        &self.instance_to_version_after_sync
    }

    /// Parse and validate a worker ID string
    pub fn parse_worker_id(raw: &str) -> Result<WorkerId, WorkerServiceError> {
        uuid::Uuid::parse_str(raw)
            .map(|_| WorkerId::from_string(raw.to_string()))
            .map_err(|e| WorkerServiceError::InvalidId {
                raw: raw.to_string(),
                message: e.to_string(),
            })
    }

    /// Get the job queue, returning an error if not initialized
    fn get_job_queue(&self) -> Result<&Arc<JobQueue>, WorkerServiceError> {
        self.job_queue
            .get()
            .ok_or(WorkerServiceError::QueueNotInitialized)
    }

    pub async fn create_worker(
        &self,
        config: WorkerSpec,
    ) -> Result<CreateWorkerResult, WorkerServiceError> {
        if self.router_config.api_key.is_some() && config.api_key.is_none() {
            warn!(
                "Adding worker {} without API key while router has API key configured. \
                Worker will be accessible without authentication. \
                If the worker requires the same API key as the router, please specify it explicitly.",
                config.url
            );
        }

        let worker_url = config.url.clone();
        let worker_id = self.worker_registry.reserve_id_for_url(&worker_url);

        let job = Job::AddWorker {
            config: Box::new(config),
        };

        self.get_job_queue()?
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        let location = format!("/workers/{}", worker_id.as_str());

        Ok(CreateWorkerResult {
            worker_id,
            url: worker_url,
            location,
        })
    }

    /// List all workers with their info
    pub fn list_workers(&self) -> ListWorkersResult {
        let workers = self.worker_registry.get_all_with_ids();
        let worker_infos: Vec<WorkerInfo> = workers
            .iter()
            .map(|(worker_id, worker)| {
                let mut info = worker_to_info(worker);
                info.id = worker_id.as_str().to_string();
                info
            })
            .collect();

        let stats = self.worker_registry.stats();

        ListWorkersResult {
            workers: worker_infos,
            total: stats.total_workers,
            prefill_count: stats.prefill_workers,
            decode_count: stats.decode_workers,
            regular_count: stats.regular_workers,
        }
    }

    pub fn get_worker(&self, worker_id_raw: &str) -> Result<GetWorkerResponse, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;
        let job_queue = self.get_job_queue()?;

        if let Some(worker) = self.worker_registry.get(&worker_id) {
            let worker_url = worker.url().to_string();
            let mut worker_info = worker_to_info(&worker);
            worker_info.id = worker_id.as_str().to_string();
            if let Some(status) = job_queue.get_status(&worker_url) {
                worker_info.job_status = Some(status);
            }
            return Ok(GetWorkerResponse(worker_info));
        }

        if let Some(worker_url) = self.worker_registry.get_url_by_id(&worker_id) {
            if let Some(status) = job_queue.get_status(&worker_url) {
                return Ok(GetWorkerResponse(WorkerInfo::pending(
                    worker_id.as_str(),
                    worker_url,
                    Some(status),
                )));
            }
        }

        Err(WorkerServiceError::NotFound {
            worker_id: worker_id_raw.to_string(),
        })
    }

    /// Delete a worker by ID (submits async job)
    pub async fn delete_worker(
        &self,
        worker_id_raw: &str,
    ) -> Result<DeleteWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::RemoveWorker { url: url.clone() };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(DeleteWorkerResult { worker_id, url })
    }

    /// Update a worker by ID (submits async job)
    pub async fn update_worker(
        &self,
        worker_id_raw: &str,
        update: WorkerUpdateRequest,
    ) -> Result<UpdateWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::UpdateWorker {
            url: url.clone(),
            update: Box::new(update),
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
    }

    // ========================================================================
    // PR 1 §1.5: Worker resolution helpers for DP-aware targeting
    // ========================================================================

    // PR 1 §1.5: Resolve a single worker by base_worker_id + dp_rank
    fn resolve_worker_by_base_and_dp(
        &self,
        base_worker_id_raw: &str,
        dp_rank: usize,
    ) -> Result<(WorkerId, Arc<dyn Worker>), WorkerServiceError> {
        let base_worker_id = Self::parse_worker_id(base_worker_id_raw)?;
        let base_url = self
            .worker_registry
            .get_url_by_id(&base_worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: base_worker_id_raw.to_string(),
            })?;

        let target_url = format!("{base_url}@{dp_rank}");
        if let Some((worker_id, worker)) = self
            .worker_registry
            .get_all_with_ids()
            .into_iter()
            .find(|(_, worker)| worker.url() == target_url)
        {
            return Ok((worker_id, worker));
        }

        // Backward-compatible fallback for non-DP workers.
        if dp_rank == 0 {
            if let Some(worker) = self.worker_registry.get_by_url(&base_url) {
                return Ok((base_worker_id, worker));
            }
        }

        Err(WorkerServiceError::NotFound {
            worker_id: format!("{base_worker_id_raw}@{dp_rank}"),
        })
    }

    // PR 1 §1.5: Resolve worker from a stats update target (worker_id or base+dp)
    fn resolve_worker_from_target(
        &self,
        target: &WorkerStatsTargetUpdateRequest,
    ) -> Result<(WorkerId, Arc<dyn Worker>), WorkerServiceError> {
        match (
            target.worker_id.as_deref(),
            target.base_worker_id.as_deref(),
            target.dp_rank,
        ) {
            (Some(worker_id_raw), None, None) => {
                let worker_id = Self::parse_worker_id(worker_id_raw)?;
                let worker = self.worker_registry.get(&worker_id).ok_or_else(|| {
                    WorkerServiceError::NotFound {
                        worker_id: worker_id_raw.to_string(),
                    }
                })?;
                Ok((worker_id, worker))
            }
            (None, Some(base_worker_id_raw), Some(dp_rank)) => {
                self.resolve_worker_by_base_and_dp(base_worker_id_raw, dp_rank)
            }
            _ => Err(WorkerServiceError::InvalidId {
                raw: "worker stats target".to_string(),
                message:
                    "target must specify either `worker_id`, or both `base_worker_id` and `dp_rank`"
                        .to_string(),
            }),
        }
    }

    // PR 1 §1.5: Resolve all DP workers from a base_worker_id
    fn resolve_workers_by_base_worker_id(
        &self,
        base_worker_id_raw: &str,
    ) -> Result<ResolvedWorkers, WorkerServiceError> {
        let base_worker_id = Self::parse_worker_id(base_worker_id_raw)?;
        let base_url = self
            .worker_registry
            .get_url_by_id(&base_worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: base_worker_id_raw.to_string(),
            })?;

        let prefix = format!("{base_url}@");
        let mut workers: ResolvedWorkers = self
            .worker_registry
            .get_all_with_ids()
            .into_iter()
            .filter(|(_, worker)| worker.url().starts_with(&prefix))
            .collect();

        // Backward-compatible fallback for non-DP worker URLs.
        if workers.is_empty() {
            if let Some(worker) = self.worker_registry.get_by_url(&base_url) {
                workers.push((base_worker_id, worker));
            }
        }

        if workers.is_empty() {
            return Err(WorkerServiceError::NotFound {
                worker_id: base_worker_id_raw.to_string(),
            });
        }

        Ok(workers)
    }

    // PR 1 §1.5: Resolve workers from a routing control target (worker_id, base_worker_id, or base+dp)
    fn resolve_workers_from_routing_control_target(
        &self,
        target: &WorkerRoutingControlTargetRequest,
    ) -> Result<ResolvedWorkers, WorkerServiceError> {
        match (
            target.worker_id.as_deref(),
            target.base_worker_id.as_deref(),
            target.dp_rank.as_ref(),
        ) {
            (Some(worker_id_raw), None, None) => {
                let worker_id = Self::parse_worker_id(worker_id_raw)?;
                let worker = self
                    .worker_registry
                    .get(&worker_id)
                    .ok_or_else(|| WorkerServiceError::NotFound {
                        worker_id: worker_id_raw.to_string(),
                    })?;
                Ok(vec![(worker_id, worker)])
            }
            (None, Some(base_worker_id_raw), None) => {
                self.resolve_workers_by_base_worker_id(base_worker_id_raw)
            }
            (None, Some(base_worker_id_raw), Some(dp_rank_input)) => {
                let ranks = dp_rank_input.to_ranks()?;
                let mut resolved = Vec::with_capacity(ranks.len());
                for rank in ranks {
                    resolved
                        .push(self.resolve_worker_by_base_and_dp(base_worker_id_raw, rank)?);
                }
                Ok(resolved)
            }
            _ => Err(WorkerServiceError::InvalidId {
                raw: "worker routing control target".to_string(),
                message: "target must specify exactly one mode: `worker_id`, `base_worker_id`, or `base_worker_id` with `dp_rank`"
                    .to_string(),
            }),
        }
    }

    // ========================================================================
    // PR 1 §1.5: Endpoint business logic
    // ========================================================================

    // PR 1 §1.5: Build a per-item result from an EngineStatsUpdateOutcome
    fn build_stats_update_result(
        worker_id: &WorkerId,
        worker_url: String,
        outcome: EngineStatsUpdateOutcome,
        base_worker_id: Option<String>,
        dp_rank: Option<usize>,
    ) -> WorkerStatsUpdateItemResult {
        match outcome {
            EngineStatsUpdateOutcome::Applied => WorkerStatsUpdateItemResult {
                status: "updated".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url: worker_url,
                applied: true,
                base_worker_id,
                dp_rank,
                stale_reason: None,
            },
            EngineStatsUpdateOutcome::Stale { reason } => WorkerStatsUpdateItemResult {
                status: "stale_ignored".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url: worker_url,
                applied: false,
                base_worker_id,
                dp_rank,
                stale_reason: Some(reason),
            },
            EngineStatsUpdateOutcome::Rejected { reason } => WorkerStatsUpdateItemResult {
                status: "rejected".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url: worker_url,
                applied: false,
                base_worker_id,
                dp_rank,
                stale_reason: Some(reason),
            },
        }
    }

    // PR 1 §1.5: Batch engine stats update
    /// Update engine stats for one or more workers.
    ///
    /// Called by `POST /workers/stats`. Each update item targets a single worker
    /// via `worker_id` or `base_worker_id` + `dp_rank`. Stats are applied with
    /// staleness checking using the configured threshold.
    pub fn update_worker_stats(&self, update: WorkerStatsUpdateRequest) -> UpdateWorkerStatsResult {
        let total = update.updates.len();
        let mut results = Vec::with_capacity(total);
        let staleness_threshold_ms = self.router_config.engine_stats_staleness_threshold_ms;

        for item in update.updates {
            match self.resolve_worker_from_target(&item) {
                Ok((worker_id, worker)) => {
                    let worker_url = worker.url().to_string();
                    let outcome = worker.update_engine_stats(item.stats, staleness_threshold_ms);
                    results.push(Self::build_stats_update_result(
                        &worker_id,
                        worker_url,
                        outcome,
                        item.base_worker_id,
                        item.dp_rank,
                    ));
                }
                Err(err) => {
                    results.push(WorkerStatsUpdateItemResult {
                        status: "rejected".to_string(),
                        worker_id: item
                            .worker_id
                            .or_else(|| item.base_worker_id.clone())
                            .unwrap_or_default(),
                        url: String::new(),
                        applied: false,
                        base_worker_id: item.base_worker_id,
                        dp_rank: item.dp_rank,
                        stale_reason: Some(err.to_string()),
                    });
                }
            }
        }

        let updated = results.iter().filter(|r| r.status == "updated").count();
        let stale_ignored = results
            .iter()
            .filter(|r| r.status == "stale_ignored")
            .count();
        let rejected = results.iter().filter(|r| r.status == "rejected").count();

        UpdateWorkerStatsResult {
            total,
            updated,
            stale_ignored,
            rejected,
            results,
        }
    }

    // PR 1 §1.5: Apply routing control (pause/resume) to resolved workers
    fn apply_worker_routing_control(
        &self,
        update: WorkerRoutingControlRequest,
        paused: bool,
    ) -> UpdateWorkerRoutingResult {
        let mut results = Vec::new();
        let mut seen_worker_ids = HashSet::new();

        for target in update {
            let base_worker_id = target.base_worker_id.clone();
            let requested_ranks: Option<Vec<usize>> =
                target.dp_rank.as_ref().and_then(|r| r.to_ranks().ok());

            match self.resolve_workers_from_routing_control_target(&target) {
                Ok(workers) => {
                    for (worker_id, worker) in workers {
                        if !seen_worker_ids.insert(worker_id.as_str().to_string()) {
                            continue;
                        }

                        worker.set_paused(paused);

                        results.push(WorkerRoutingControlItemResult {
                            status: "updated".to_string(),
                            worker_id: worker_id.as_str().to_string(),
                            url: worker.url().to_string(),
                            paused,
                            base_worker_id: base_worker_id.clone(),
                            dp_rank: worker.dp_rank().or_else(|| {
                                if requested_ranks.as_ref().is_some_and(|r| r.len() == 1) {
                                    requested_ranks.as_ref().and_then(|r| r.first().copied())
                                } else {
                                    None
                                }
                            }),
                            reason: None,
                        });
                    }
                }
                Err(err) => {
                    results.push(WorkerRoutingControlItemResult {
                        status: "rejected".to_string(),
                        worker_id: target
                            .worker_id
                            .or_else(|| target.base_worker_id.clone())
                            .unwrap_or_default(),
                        url: String::new(),
                        paused,
                        base_worker_id: target.base_worker_id,
                        dp_rank: requested_ranks
                            .as_ref()
                            .and_then(|ranks| (ranks.len() == 1).then(|| ranks[0])),
                        reason: Some(err.to_string()),
                    });
                }
            }
        }

        let updated = results.iter().filter(|r| r.status == "updated").count();
        let rejected = results.iter().filter(|r| r.status == "rejected").count();

        UpdateWorkerRoutingResult {
            action: if paused {
                "paused".to_string()
            } else {
                "resumed".to_string()
            },
            total: results.len(),
            updated,
            rejected,
            results,
        }
    }

    // PR 1 §1.5: Pause routing for workers
    /// Pause routing for one or multiple workers.
    ///
    /// Called by `POST /workers/pause`. Paused workers are excluded from
    /// load-balancing selection via `is_available()`.
    pub fn pause_workers(&self, update: WorkerRoutingControlRequest) -> UpdateWorkerRoutingResult {
        self.apply_worker_routing_control(update, true)
    }

    // PR 1 §1.5: Resume routing for workers
    /// Resume routing for one or multiple workers.
    ///
    /// Called by `POST /workers/resume`.
    pub fn resume_workers(&self, update: WorkerRoutingControlRequest) -> UpdateWorkerRoutingResult {
        self.apply_worker_routing_control(update, false)
    }

    // PR 1 §1.5: Batch version tag update
    /// Batch update worker `version_tag` labels via the existing worker update workflow.
    ///
    /// Called by `POST /workers/version_tag`. Each update item targets a single
    /// worker and sets its `version_tag` label to the specified value.
    pub async fn update_worker_version_tag(
        &self,
        update: WorkerVersionTagUpdateRequest,
    ) -> UpdateWorkerVersionTagResult {
        let total = update.updates.len();
        let mut results = Vec::with_capacity(total);

        for item in update.updates {
            let base_worker_id = item.base_worker_id.clone();
            let dp_rank = item.dp_rank;
            let version_tag = item.version_tag;

            // Reuse stats target resolution by constructing a compatible target
            let target = WorkerStatsTargetUpdateRequest {
                worker_id: item.worker_id.clone(),
                base_worker_id: item.base_worker_id.clone(),
                dp_rank: item.dp_rank,
                stats: EngineStats::default(),
            };

            match self.resolve_worker_from_target(&target) {
                Ok((worker_id, worker)) => {
                    let update_payload: WorkerUpdateRequest = match serde_json::from_value(json!({
                        "labels": {
                            "version_tag": version_tag.to_string()
                        }
                    })) {
                        Ok(payload) => payload,
                        Err(err) => {
                            results.push(WorkerVersionTagUpdateItemResult {
                                status: "rejected".to_string(),
                                worker_id: worker_id.as_str().to_string(),
                                url: worker.url().to_string(),
                                version_tag,
                                base_worker_id,
                                dp_rank,
                                reason: Some(format!(
                                    "Failed to build worker update payload: {err}"
                                )),
                            });
                            continue;
                        }
                    };

                    match self.update_worker(worker_id.as_str(), update_payload).await {
                        Ok(update_result) => {
                            // PR 1 §1.6c: Update instance_to_version_after_sync map
                            let resolved_dp_rank = worker.dp_rank().unwrap_or(0);
                            self.instance_to_version_after_sync.lock().insert(
                                (worker_id.as_str().to_string(), resolved_dp_rank),
                                version_tag,
                            );
                            debug!(
                                worker_id = worker_id.as_str(),
                                dp_rank = resolved_dp_rank,
                                version_tag,
                                "Updated instance_to_version_after_sync map"
                            );

                            results.push(WorkerVersionTagUpdateItemResult {
                                status: "accepted".to_string(),
                                worker_id: update_result.worker_id.as_str().to_string(),
                                url: update_result.url,
                                version_tag,
                                base_worker_id,
                                dp_rank,
                                reason: None,
                            });
                        }
                        Err(err) => {
                            results.push(WorkerVersionTagUpdateItemResult {
                                status: "rejected".to_string(),
                                worker_id: worker_id.as_str().to_string(),
                                url: worker.url().to_string(),
                                version_tag,
                                base_worker_id,
                                dp_rank,
                                reason: Some(err.to_string()),
                            });
                        }
                    }
                }
                Err(err) => {
                    results.push(WorkerVersionTagUpdateItemResult {
                        status: "rejected".to_string(),
                        worker_id: item
                            .worker_id
                            .or_else(|| item.base_worker_id.clone())
                            .unwrap_or_default(),
                        url: String::new(),
                        version_tag,
                        base_worker_id,
                        dp_rank,
                        reason: Some(err.to_string()),
                    });
                }
            }
        }

        let accepted = results.iter().filter(|r| r.status == "accepted").count();
        let rejected = results.iter().filter(|r| r.status == "rejected").count();

        UpdateWorkerVersionTagResult {
            total,
            accepted,
            rejected,
            results,
        }
    }

    // ========================================================================
    // PR 1 §1.6c: Worker registration hook for version_tag label
    // ========================================================================
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use super::*;
    use crate::{
        config::RouterConfig,
        core::{
            worker::{EngineSchedulerStats, EngineStats, EngineStatsSnapshot, Worker},
            worker_builder::BasicWorkerBuilder,
            worker_registry::WorkerRegistry,
            JobQueue,
        },
    };

    /// Helper: create a WorkerService with an empty registry and default config.
    fn create_test_service() -> (WorkerService, Arc<WorkerRegistry>) {
        let registry = Arc::new(WorkerRegistry::new());
        let job_queue = Arc::new(std::sync::OnceLock::<Arc<JobQueue>>::new());
        let config = RouterConfig::default();
        let version_map: InstanceVersionMap = Arc::new(Mutex::new(HashMap::new()));
        let service = WorkerService::new(registry.clone(), job_queue, config, version_map);
        (service, registry)
    }

    /// Helper: create a WorkerService with a custom staleness threshold.
    fn create_test_service_with_staleness(
        threshold_ms: u64,
    ) -> (WorkerService, Arc<WorkerRegistry>) {
        let registry = Arc::new(WorkerRegistry::new());
        let job_queue = Arc::new(std::sync::OnceLock::<Arc<JobQueue>>::new());
        let config = RouterConfig {
            engine_stats_staleness_threshold_ms: threshold_ms,
            ..Default::default()
        };
        let version_map: InstanceVersionMap = Arc::new(Mutex::new(HashMap::new()));
        let service = WorkerService::new(registry.clone(), job_queue, config, version_map);
        (service, registry)
    }

    /// Helper: register a worker in the registry, returning its ID.
    fn register_worker(registry: &WorkerRegistry, url: &str) -> WorkerId {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(url).build());
        registry.register(worker)
    }

    /// Helper: register a DP-aware worker.
    fn register_dp_worker(
        registry: &WorkerRegistry,
        url: &str,
        dp_rank: usize,
        dp_size: usize,
    ) -> WorkerId {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .dp_config(dp_rank, dp_size)
                .build(),
        );
        registry.register(worker)
    }

    /// Helper: build a fresh EngineStats with a recent timestamp.
    fn fresh_engine_stats() -> EngineStats {
        let now = chrono::Utc::now();
        EngineStats {
            snapshot: EngineStatsSnapshot {
                timestamp: now.to_rfc3339(),
                scheduler_stats: EngineSchedulerStats {
                    num_running_reqs: 5,
                    num_waiting_reqs: 3,
                    kv_cache_usage: 0.42,
                    ..Default::default()
                },
            },
        }
    }

    // ====================================================================
    // update_worker_stats tests
    // ====================================================================

    #[test]
    fn test_update_worker_stats_success_by_worker_id() {
        let (service, registry) = create_test_service();
        let worker_id = register_worker(&registry, "http://worker1:8000");

        let stats = fresh_engine_stats();
        let request = WorkerStatsUpdateRequest {
            updates: vec![WorkerStatsTargetUpdateRequest {
                worker_id: Some(worker_id.as_str().to_string()),
                base_worker_id: None,
                dp_rank: None,
                stats,
            }],
        };

        let result = service.update_worker_stats(request);
        assert_eq!(result.total, 1);
        assert_eq!(result.updated, 1);
        assert_eq!(result.stale_ignored, 0);
        assert_eq!(result.rejected, 0);
        assert!(result.results[0].applied);
        assert_eq!(result.results[0].status, "updated");

        // Verify stats are stored on worker
        let worker = registry.get(&worker_id).expect("worker exists");
        assert_eq!(worker.engine_stats().running_queue_size(), 5);
        assert_eq!(worker.engine_stats().waiting_queue_size(), 3);
    }

    #[test]
    fn test_update_worker_stats_worker_not_found() {
        let (service, _registry) = create_test_service();

        let request = WorkerStatsUpdateRequest {
            updates: vec![WorkerStatsTargetUpdateRequest {
                worker_id: Some("00000000-0000-0000-0000-000000000001".to_string()),
                base_worker_id: None,
                dp_rank: None,
                stats: fresh_engine_stats(),
            }],
        };

        let result = service.update_worker_stats(request);
        assert_eq!(result.total, 1);
        assert_eq!(result.updated, 0);
        assert_eq!(result.rejected, 1);
        assert!(!result.results[0].applied);
        assert_eq!(result.results[0].status, "rejected");
    }

    #[test]
    fn test_update_worker_stats_invalid_worker_id() {
        let (service, _registry) = create_test_service();

        let request = WorkerStatsUpdateRequest {
            updates: vec![WorkerStatsTargetUpdateRequest {
                worker_id: Some("not-a-uuid".to_string()),
                base_worker_id: None,
                dp_rank: None,
                stats: fresh_engine_stats(),
            }],
        };

        let result = service.update_worker_stats(request);
        assert_eq!(result.rejected, 1);
        assert!(!result.results[0].applied);
    }

    #[test]
    fn test_update_worker_stats_stale_rejected() {
        // Use a threshold of 0 to force staleness rejection of any non-future timestamp
        let (service, registry) = create_test_service_with_staleness(0);
        let worker_id = register_worker(&registry, "http://worker1:8000");

        // First update with current timestamp
        let stats = fresh_engine_stats();
        let request1 = WorkerStatsUpdateRequest {
            updates: vec![WorkerStatsTargetUpdateRequest {
                worker_id: Some(worker_id.as_str().to_string()),
                base_worker_id: None,
                dp_rank: None,
                stats,
            }],
        };
        let r1 = service.update_worker_stats(request1);
        assert_eq!(r1.updated, 1);

        // Second update with older timestamp
        let old_stats = EngineStats {
            snapshot: EngineStatsSnapshot {
                timestamp: "2020-01-01T00:00:00Z".to_string(),
                scheduler_stats: EngineSchedulerStats::default(),
            },
        };
        let request2 = WorkerStatsUpdateRequest {
            updates: vec![WorkerStatsTargetUpdateRequest {
                worker_id: Some(worker_id.as_str().to_string()),
                base_worker_id: None,
                dp_rank: None,
                stats: old_stats,
            }],
        };
        let r2 = service.update_worker_stats(request2);
        // Should be stale or rejected — not updated
        assert_eq!(r2.updated, 0);
        assert!(r2.stale_ignored > 0 || r2.rejected > 0);
    }

    #[test]
    fn test_update_worker_stats_batch_mixed() {
        let (service, registry) = create_test_service();
        let worker_id = register_worker(&registry, "http://worker1:8000");

        let request = WorkerStatsUpdateRequest {
            updates: vec![
                // Valid update
                WorkerStatsTargetUpdateRequest {
                    worker_id: Some(worker_id.as_str().to_string()),
                    base_worker_id: None,
                    dp_rank: None,
                    stats: fresh_engine_stats(),
                },
                // Invalid: worker does not exist
                WorkerStatsTargetUpdateRequest {
                    worker_id: Some("00000000-0000-0000-0000-000000000099".to_string()),
                    base_worker_id: None,
                    dp_rank: None,
                    stats: fresh_engine_stats(),
                },
            ],
        };

        let result = service.update_worker_stats(request);
        assert_eq!(result.total, 2);
        assert_eq!(result.updated, 1);
        assert_eq!(result.rejected, 1);
    }

    #[test]
    fn test_update_worker_stats_by_base_worker_id_and_dp_rank() {
        let (service, registry) = create_test_service();
        // Register base worker first so URL mapping exists
        let base_id = register_worker(&registry, "http://worker1:8000");
        // Register DP worker
        let _dp_id = register_dp_worker(&registry, "http://worker1:8000", 0, 4);

        let request = WorkerStatsUpdateRequest {
            updates: vec![WorkerStatsTargetUpdateRequest {
                worker_id: None,
                base_worker_id: Some(base_id.as_str().to_string()),
                dp_rank: Some(0),
                stats: fresh_engine_stats(),
            }],
        };

        let result = service.update_worker_stats(request);
        assert_eq!(result.total, 1);
        assert_eq!(result.updated, 1);
    }

    #[test]
    fn test_update_worker_stats_invalid_target_mode() {
        let (service, _registry) = create_test_service();

        // Neither worker_id nor base_worker_id specified
        let request = WorkerStatsUpdateRequest {
            updates: vec![WorkerStatsTargetUpdateRequest {
                worker_id: None,
                base_worker_id: None,
                dp_rank: None,
                stats: fresh_engine_stats(),
            }],
        };

        let result = service.update_worker_stats(request);
        assert_eq!(result.rejected, 1);
    }

    // ====================================================================
    // pause_workers / resume_workers tests
    // ====================================================================

    #[test]
    fn test_workers_pause_sets_flag() {
        let (service, registry) = create_test_service();
        let worker_id = register_worker(&registry, "http://worker1:8000");

        let request: WorkerRoutingControlRequest = vec![WorkerRoutingControlTargetRequest {
            worker_id: Some(worker_id.as_str().to_string()),
            base_worker_id: None,
            dp_rank: None,
        }];

        let result = service.pause_workers(request);
        assert_eq!(result.action, "paused");
        assert_eq!(result.updated, 1);
        assert_eq!(result.rejected, 0);
        assert!(result.results[0].paused);

        // Verify worker is actually paused
        let worker = registry.get(&worker_id).expect("worker exists");
        assert!(worker.is_paused());
    }

    #[test]
    fn test_workers_resume_clears_flag() {
        let (service, registry) = create_test_service();
        let worker_id = register_worker(&registry, "http://worker1:8000");

        // Pause first
        let pause_request: WorkerRoutingControlRequest = vec![WorkerRoutingControlTargetRequest {
            worker_id: Some(worker_id.as_str().to_string()),
            base_worker_id: None,
            dp_rank: None,
        }];
        service.pause_workers(pause_request);

        // Verify paused
        let worker = registry.get(&worker_id).expect("worker exists");
        assert!(worker.is_paused());

        // Resume
        let resume_request: WorkerRoutingControlRequest = vec![WorkerRoutingControlTargetRequest {
            worker_id: Some(worker_id.as_str().to_string()),
            base_worker_id: None,
            dp_rank: None,
        }];
        let result = service.resume_workers(resume_request);
        assert_eq!(result.action, "resumed");
        assert_eq!(result.updated, 1);
        assert!(!result.results[0].paused);

        // Verify worker is no longer paused
        assert!(!worker.is_paused());
    }

    #[test]
    fn test_workers_pause_makes_unavailable() {
        let (service, registry) = create_test_service();
        let worker_id = register_worker(&registry, "http://worker1:8000");

        // Worker should be available before pause
        let worker = registry.get(&worker_id).expect("worker exists");
        assert!(worker.is_available());

        // Pause
        let request: WorkerRoutingControlRequest = vec![WorkerRoutingControlTargetRequest {
            worker_id: Some(worker_id.as_str().to_string()),
            base_worker_id: None,
            dp_rank: None,
        }];
        service.pause_workers(request);

        // Worker should be unavailable after pause
        assert!(!worker.is_available());
    }

    #[test]
    fn test_workers_pause_worker_not_found() {
        let (service, _registry) = create_test_service();

        let request: WorkerRoutingControlRequest = vec![WorkerRoutingControlTargetRequest {
            worker_id: Some("00000000-0000-0000-0000-000000000001".to_string()),
            base_worker_id: None,
            dp_rank: None,
        }];

        let result = service.pause_workers(request);
        assert_eq!(result.rejected, 1);
        assert_eq!(result.updated, 0);
    }

    #[test]
    fn test_workers_pause_batch_multiple() {
        let (service, registry) = create_test_service();
        let worker_id1 = register_worker(&registry, "http://worker1:8000");
        let worker_id2 = register_worker(&registry, "http://worker2:8000");

        let request: WorkerRoutingControlRequest = vec![
            WorkerRoutingControlTargetRequest {
                worker_id: Some(worker_id1.as_str().to_string()),
                base_worker_id: None,
                dp_rank: None,
            },
            WorkerRoutingControlTargetRequest {
                worker_id: Some(worker_id2.as_str().to_string()),
                base_worker_id: None,
                dp_rank: None,
            },
        ];

        let result = service.pause_workers(request);
        assert_eq!(result.updated, 2);
        assert_eq!(result.rejected, 0);

        let w1 = registry.get(&worker_id1).expect("worker1");
        let w2 = registry.get(&worker_id2).expect("worker2");
        assert!(w1.is_paused());
        assert!(w2.is_paused());
    }

    #[test]
    fn test_workers_pause_deduplicates() {
        let (service, registry) = create_test_service();
        let worker_id = register_worker(&registry, "http://worker1:8000");

        // Send same worker twice
        let request: WorkerRoutingControlRequest = vec![
            WorkerRoutingControlTargetRequest {
                worker_id: Some(worker_id.as_str().to_string()),
                base_worker_id: None,
                dp_rank: None,
            },
            WorkerRoutingControlTargetRequest {
                worker_id: Some(worker_id.as_str().to_string()),
                base_worker_id: None,
                dp_rank: None,
            },
        ];

        let result = service.pause_workers(request);
        // Dedup: only 1 updated, the second is silently skipped
        assert_eq!(result.updated, 1);
    }

    #[test]
    fn test_workers_pause_by_base_worker_id() {
        let (service, registry) = create_test_service();
        // Register base worker
        let base_id = register_worker(&registry, "http://worker1:8000");
        // Register DP workers under the same base URL
        let _dp0 = register_dp_worker(&registry, "http://worker1:8000", 0, 4);
        let _dp1 = register_dp_worker(&registry, "http://worker1:8000", 1, 4);

        // Pause all DP workers by base_worker_id
        let request: WorkerRoutingControlRequest = vec![WorkerRoutingControlTargetRequest {
            worker_id: None,
            base_worker_id: Some(base_id.as_str().to_string()),
            dp_rank: None,
        }];

        let result = service.pause_workers(request);
        // Should pause the DP workers (found by prefix matching)
        assert!(result.updated >= 1);
        assert_eq!(result.rejected, 0);
    }

    // ====================================================================
    // instance_to_version_map accessor tests
    // ====================================================================

    #[test]
    fn test_instance_version_map_shared() {
        let registry = Arc::new(WorkerRegistry::new());
        let job_queue = Arc::new(std::sync::OnceLock::<Arc<JobQueue>>::new());
        let config = RouterConfig::default();
        let version_map: InstanceVersionMap = Arc::new(Mutex::new(HashMap::new()));

        let service = WorkerService::new(registry, job_queue, config, version_map.clone());

        // Insert via the map directly
        version_map.lock().insert(("w1".to_string(), 0), 100);

        // Verify readable through the service accessor
        let service_map = service.instance_to_version_map();
        assert_eq!(service_map.lock().get(&("w1".to_string(), 0)), Some(&100));
    }

    // ====================================================================
    // Worker resolution edge-case tests
    // ====================================================================

    #[test]
    fn test_resolve_worker_worker_id_only() {
        let (service, registry) = create_test_service();
        let worker_id = register_worker(&registry, "http://worker1:8000");

        let target = WorkerStatsTargetUpdateRequest {
            worker_id: Some(worker_id.as_str().to_string()),
            base_worker_id: None,
            dp_rank: None,
            stats: EngineStats::default(),
        };

        let result = service.resolve_worker_from_target(&target);
        assert!(result.is_ok());
        let (resolved_id, _) = result.expect("resolved");
        assert_eq!(resolved_id.as_str(), worker_id.as_str());
    }

    #[test]
    fn test_resolve_worker_invalid_target_both_ids() {
        let (service, _registry) = create_test_service();

        // Both worker_id and base_worker_id — invalid
        let target = WorkerStatsTargetUpdateRequest {
            worker_id: Some("00000000-0000-0000-0000-000000000001".to_string()),
            base_worker_id: Some("00000000-0000-0000-0000-000000000002".to_string()),
            dp_rank: Some(0),
            stats: EngineStats::default(),
        };

        let result = service.resolve_worker_from_target(&target);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_worker_base_id_without_dp_rank_is_invalid() {
        let (service, _registry) = create_test_service();

        // base_worker_id without dp_rank — invalid for stats target
        let target = WorkerStatsTargetUpdateRequest {
            worker_id: None,
            base_worker_id: Some("00000000-0000-0000-0000-000000000001".to_string()),
            dp_rank: None,
            stats: EngineStats::default(),
        };

        let result = service.resolve_worker_from_target(&target);
        assert!(result.is_err());
    }

    // ====================================================================
    // DpRankInput tests
    // ====================================================================

    #[test]
    fn test_dp_rank_input_single() {
        let input = DpRankInput::Single(3);
        let ranks = input.to_ranks().expect("valid");
        assert_eq!(ranks, vec![3]);
    }

    #[test]
    fn test_dp_rank_input_multiple() {
        let input = DpRankInput::Multiple(vec![2, 0, 1]);
        let ranks = input.to_ranks().expect("valid");
        // Should be sorted and deduped
        assert_eq!(ranks, vec![0, 1, 2]);
    }

    #[test]
    fn test_dp_rank_input_multiple_with_duplicates() {
        let input = DpRankInput::Multiple(vec![1, 1, 2, 2, 3]);
        let ranks = input.to_ranks().expect("valid");
        assert_eq!(ranks, vec![1, 2, 3]);
    }

    #[test]
    fn test_dp_rank_input_empty_list_is_error() {
        let input = DpRankInput::Multiple(vec![]);
        let result = input.to_ranks();
        assert!(result.is_err());
    }

    // ====================================================================
    // parse_worker_id tests
    // ====================================================================

    #[test]
    fn test_parse_worker_id_valid() {
        let result = WorkerService::parse_worker_id("00000000-0000-0000-0000-000000000001");
        assert!(result.is_ok());
        assert_eq!(
            result.expect("valid").as_str(),
            "00000000-0000-0000-0000-000000000001"
        );
    }

    #[test]
    fn test_parse_worker_id_invalid() {
        let result = WorkerService::parse_worker_id("not-a-uuid");
        assert!(result.is_err());
    }

    // ====================================================================
    // list_workers tests
    // ====================================================================

    #[test]
    fn test_list_workers_empty() {
        let (service, _registry) = create_test_service();
        let result = service.list_workers();
        assert_eq!(result.total, 0);
        assert!(result.workers.is_empty());
    }

    #[test]
    fn test_list_workers_with_workers() {
        let (service, registry) = create_test_service();
        register_worker(&registry, "http://worker1:8000");
        register_worker(&registry, "http://worker2:8000");

        let result = service.list_workers();
        assert_eq!(result.total, 2);
        assert_eq!(result.workers.len(), 2);
    }
}
