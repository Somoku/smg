//! Worker Service - Business logic layer for worker operations
//!
//! This module provides a clean separation between HTTP concerns (in routers)
//! and business logic for worker management. The service orchestrates
//! WorkerRegistry and JobQueue operations.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::worker::{WorkerErrorResponse, WorkerInfo, WorkerSpec, WorkerUpdateRequest};
use serde_json::json;
use tracing::warn;

use crate::{
    config::RouterConfig,
    worker::{
        registry::WorkerId, worker::worker_to_info, EngineStatsUpdateOutcome, Worker,
        WorkerRegistry, WorkerStatsUpdateRequest, WorkerStatsUpdateResult,
        WorkerStatsUpdateResultItem,
        WorkerWeightVersionUpdateRequest, WorkerWeightVersionUpdateResult,
        WorkerWeightVersionUpdateResultItem,
    },
    workflow::{Job, JobQueue},
};

/// Error types for worker service operations
#[derive(Debug)]
pub enum WorkerServiceError {
    /// Worker with given ID was not found
    NotFound { worker_id: String },
    /// Invalid worker ID format (expected UUID)
    InvalidId { raw: String, message: String },
    /// Bad request (e.g., URL mismatch in PUT)
    BadRequest { message: String },
    /// Worker with this URL already exists (duplicate POST)
    Conflict { url: String, worker_id: WorkerId },
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
            Self::BadRequest { .. } => "BAD_REQUEST",
            Self::Conflict { .. } => "WORKER_ALREADY_EXISTS",
            Self::QueueNotInitialized => "INTERNAL_SERVER_ERROR",
            Self::QueueSubmitFailed { .. } => "INTERNAL_SERVER_ERROR",
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound { .. } => StatusCode::NOT_FOUND,
            Self::InvalidId { .. } => StatusCode::BAD_REQUEST,
            Self::BadRequest { .. } => StatusCode::BAD_REQUEST,
            Self::Conflict { .. } => StatusCode::CONFLICT,
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
            Self::BadRequest { message } => write!(f, "{message}"),
            Self::Conflict { url, worker_id } => {
                let id = worker_id.as_str();
                write!(
                    f,
                    "Worker already exists at URL '{url}' with ID {id}. \
                    Use PUT /workers/{id} to replace or PATCH /workers/{id} to update."
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

/// Worker Service - Orchestrates worker business logic
///
/// This service provides a clean API for worker operations, separating
/// business logic from HTTP concerns. Handlers in server.rs become thin
/// wrappers that translate between HTTP and this service.
pub struct WorkerService {
    worker_registry: Arc<WorkerRegistry>,
    job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
    router_config: RouterConfig,
}

impl WorkerService {
    /// Create a new WorkerService
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
        router_config: RouterConfig,
    ) -> Self {
        Self {
            worker_registry,
            job_queue,
            router_config,
        }
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

        // Reserve (or retrieve) a stable ID for the 202 response.
        // If this URL already has an active worker, reject with 409.
        let worker_id = if let Some(ref manual_id) = config.id {
            WorkerId::from_string(manual_id.clone())
        } else {
            self.worker_registry.reserve_id_for_url(&worker_url)
        };
        if self.worker_registry.get(&worker_id).is_some() {
            return Err(WorkerServiceError::Conflict {
                url: worker_url,
                worker_id,
            });
        }

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

    /// Replace a worker by ID (full replace, re-runs registration workflow)
    pub async fn replace_worker(
        &self,
        worker_id_raw: &str,
        config: WorkerSpec,
    ) -> Result<UpdateWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        // Validate that the URL in the request body matches the existing worker.
        // URL changes are not supported via replace — use DELETE + POST instead.
        if config.url != url {
            return Err(WorkerServiceError::BadRequest {
                message: format!(
                    "URL mismatch: worker has URL '{url}' but request body has '{}'. \
                    URL changes are not supported via PUT. Use DELETE + POST instead.",
                    config.url
                ),
            });
        }

        // Re-run the full registration workflow (model discovery, etc.)
        // The workflow uses register_or_replace() which does overwrite-then-diff.
        let job = Job::AddWorker {
            config: Box::new(config),
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
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

        let job = Job::RemoveWorker {
            url: url.clone(),
            expected_revision: None,
        };

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

    pub async fn update_worker_stats(
        &self,
        update: WorkerStatsUpdateRequest,
    ) -> WorkerStatsUpdateResult {
        let update_num = update.updates.len();
        let mut results = Vec::with_capacity(update_num);
        let staleness_threshold_ms = self.router_config.engine_stats_staleness_threshold_ms;

        // Process each update
        for item in update.updates {
            match self.resolve_worker_by_id_and_dp(&item.worker_id, item.dp_rank) {
                Ok((worker_id, worker)) => {
                    let url = worker.url().to_string();
                    let dp_rank = item.dp_rank;
                    let outcome = worker.update_engine_stats(item.stats, staleness_threshold_ms);
                    results.push(Self::build_stats_update_result(
                        &worker_id, url, dp_rank, outcome,
                    ));
                }
                Err(err) => {
                    results.push(WorkerStatsUpdateResultItem {
                        status: "rejected".to_string(),
                        worker_id: item.worker_id,
                        url: String::new(),
                        applied: false,
                        dp_rank: item.dp_rank,
                        stale_reason: Some(err.to_string()),
                    });
                }
            }
        }

        // Count the results
        let updated = results.iter().filter(|r| r.status == "updated").count();
        let stale_ignored = results
            .iter()
            .filter(|r| r.status == "stale_ignored")
            .count();
        let rejected = results.iter().filter(|r| r.status == "rejected").count();

        WorkerStatsUpdateResult {
            total: update_num,
            updated,
            stale_ignored,
            rejected,
            results,
        }
    }

    pub fn update_worker_weight_version(
        &self,
        update: WorkerWeightVersionUpdateRequest,
    ) -> WorkerWeightVersionUpdateResult {
        let update_num = update.updates.len();
        let mut results = Vec::with_capacity(update_num);

        // Process each update
        for item in update.updates {
            let weight_version = item.weight_version;
            match self.resolve_worker_by_id_and_dp(&item.worker_id, item.dp_rank) {
                Ok((worker_id, worker)) => {
                    if worker.update_dyn_weight_version(weight_version) {
                        results.push(WorkerWeightVersionUpdateResultItem {
                            status: "updated".to_string(),
                            worker_id: worker_id.as_str().to_string(),
                            url: worker.url().to_string(),
                            weight_version,
                            dp_rank: item.dp_rank,
                            reason: None,
                        });
                    } else {
                        results.push(WorkerWeightVersionUpdateResultItem {
                            status: "rejected".to_string(),
                            worker_id: worker_id.as_str().to_string(),
                            url: worker.url().to_string(),
                            weight_version,
                            dp_rank: item.dp_rank,
                            reason: Some(
                                "worker does not support dynamic weight version updates"
                                    .to_string(),
                            ),
                        });
                    }
                }
                Err(err) => {
                    results.push(WorkerWeightVersionUpdateResultItem {
                        status: "rejected".to_string(),
                        worker_id: item.worker_id,
                        url: String::new(),
                        weight_version,
                        dp_rank: item.dp_rank,
                        reason: Some(err.to_string()),
                    });
                }
            }
        }

        // Count the number of updated and rejected results
        let updated = results.iter().filter(|r| r.status == "updated").count();
        let rejected = results.iter().filter(|r| r.status == "rejected").count();

        WorkerWeightVersionUpdateResult {
            update_num,
            updated,
            rejected,
            results,
        }
    }

    fn resolve_worker_by_id_and_dp(
        &self,
        worker_id_raw: &str,
        dp_rank: Option<usize>,
    ) -> Result<(WorkerId, Arc<dyn Worker>), WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;
        let Some(dp_rank) = dp_rank else {
            let worker = self.worker_registry.get(&worker_id).ok_or_else(|| {
                WorkerServiceError::NotFound {
                    worker_id: worker_id_raw.to_string(),
                }
            })?;
            if worker.is_dp_aware() {
                return Err(WorkerServiceError::BadRequest {
                    message: format!(
                        "worker_id '{}' points to a DP worker; include the base worker_id and dp_rank instead",
                        worker_id_raw
                    ),
                });
            }
            return Ok((worker_id, worker));
        };

        if self
            .worker_registry
            .get(&worker_id)
            .is_some_and(|worker| worker.is_dp_aware())
        {
            return Err(WorkerServiceError::BadRequest {
                message: format!(
                    "worker_id '{}' points to a DP worker; use the base worker_id with dp_rank",
                    worker_id_raw
                ),
            });
        }

        let base_url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;
        let target_url = format!("{base_url}@{dp_rank}");
        let target_id = self
            .worker_registry
            .get_id_by_url(&target_url)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: format!("{worker_id_raw}@{dp_rank}"),
            })?;
        let worker =
            self.worker_registry
                .get(&target_id)
                .ok_or_else(|| WorkerServiceError::NotFound {
                    worker_id: target_id.as_str().to_string(),
                })?;
        if worker.dp_rank() != Some(dp_rank) {
            return Err(WorkerServiceError::BadRequest {
                message: format!("worker at URL '{}' is not DP rank {dp_rank}", worker.url()),
            });
        }

        Ok((target_id, worker))
    }

    fn build_stats_update_result(
        worker_id: &WorkerId,
        url: String,
        dp_rank: Option<usize>,
        outcome: EngineStatsUpdateOutcome,
    ) -> WorkerStatsUpdateResultItem {
        match outcome {
            EngineStatsUpdateOutcome::Applied => WorkerStatsUpdateResultItem {
                status: "updated".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url,
                applied: true,
                dp_rank,
                stale_reason: None,
            },
            EngineStatsUpdateOutcome::Stale { reason } => WorkerStatsUpdateResultItem {
                status: "stale_ignored".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url,
                applied: false,
                dp_rank,
                stale_reason: Some(reason),
            },
            EngineStatsUpdateOutcome::Rejected { reason } => WorkerStatsUpdateResultItem {
                status: "rejected".to_string(),
                worker_id: worker_id.as_str().to_string(),
                url,
                applied: false,
                dp_rank,
                stale_reason: Some(reason),
            },
        }
    }
}
