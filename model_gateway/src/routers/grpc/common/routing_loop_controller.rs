// PR 5A §5A.1a-d: Routing loop controller endpoints for pause, resume, status, and filter.
// PR 10 §10.8: Handler wrappers use RoutingLoopRuntime directly (RoutingLoopState removed).
//!
//! Provides HTTP handlers for managing the PSRL routing loop lifecycle:
//! - `routing_loop_pause()` — pause dispatch
//! - `routing_loop_resume()` — resume dispatch
//! - `routing_loop_status()` — query running/paused state
//! - `routing_loop_filter()` — list queued requests matching a version_tag threshold

use std::sync::{atomic::Ordering, Arc};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use tracing::info;

use crate::routers::routing_loop_utils::RoutingLoopRuntime;

// ── Query parameters ────────────────────────────────────────────────────

// PR 5A §5A.1d: Query parameter for routing_loop_filter.
/// Query parameter for the `/routing_loop/filter` endpoint.
#[derive(Debug, Deserialize)]
pub struct RoutingLoopFilterQuery {
    pub version_tag: i64,
}

// ── Axum handler wrappers (for server.rs route registration) ────────────

/// Helper response for when routing loop is not enabled.
fn routing_loop_not_enabled() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": "routing loop not enabled"
        })),
    )
        .into_response()
}

/// `POST /routing_loop/pause`
pub async fn routing_loop_pause(State(state): State<Arc<crate::server::AppState>>) -> Response {
    // PR 10 §10.8: Read from routing_loop_runtime (replaces routing_loop_state).
    match state.context.routing_loop_runtime.as_ref() {
        Some(runtime) => handle_pause_runtime(runtime).await,
        None => routing_loop_not_enabled(),
    }
}

/// `POST /routing_loop/resume`
pub async fn routing_loop_resume(State(state): State<Arc<crate::server::AppState>>) -> Response {
    // PR 10 §10.8: Read from routing_loop_runtime (replaces routing_loop_state).
    match state.context.routing_loop_runtime.as_ref() {
        Some(runtime) => handle_resume_runtime(runtime).await,
        None => routing_loop_not_enabled(),
    }
}

/// `GET /routing_loop/status`
pub async fn routing_loop_status(State(state): State<Arc<crate::server::AppState>>) -> Response {
    // PR 10 §10.8: Read from routing_loop_runtime (replaces routing_loop_state).
    match state.context.routing_loop_runtime.as_ref() {
        Some(runtime) => handle_status_runtime(runtime).await,
        None => routing_loop_not_enabled(),
    }
}

/// `GET /routing_loop/filter?version_tag=N`
pub async fn routing_loop_filter(
    State(state): State<Arc<crate::server::AppState>>,
    Query(query): Query<RoutingLoopFilterQuery>,
) -> Response {
    // PR 10 §10.8: Use routing_loop_runtime and iterate actual MultiPriorityRequestQueue.
    match state.context.routing_loop_runtime.as_ref() {
        Some(runtime) => handle_filter_runtime(runtime, query.version_tag).await,
        None => routing_loop_not_enabled(),
    }
}

// ── Runtime-level handler helpers (PR 10 §10.8) ─────────────────────────

/// Pause handler operating directly on `RoutingLoopRuntime`.
async fn handle_pause_runtime(runtime: &RoutingLoopRuntime) -> Response {
    runtime.is_paused.store(true, Ordering::Relaxed);
    let queue_size = runtime.request_queue.lock().await.len();
    info!(
        paused = true,
        queue_size,
        is_routing = runtime.is_routing.load(Ordering::Relaxed),
        "routing_loop_pause called"
    );
    (
        StatusCode::OK,
        Json(serde_json::json!({ "running": false })),
    )
        .into_response()
}

/// Resume handler operating directly on `RoutingLoopRuntime`.
async fn handle_resume_runtime(runtime: &RoutingLoopRuntime) -> Response {
    runtime.is_paused.store(false, Ordering::Relaxed);
    let queue_size = runtime.request_queue.lock().await.len();
    info!(
        paused = false,
        queue_size,
        is_routing = runtime.is_routing.load(Ordering::Relaxed),
        "routing_loop_resume called"
    );
    (StatusCode::OK, Json(serde_json::json!({ "running": true }))).into_response()
}

/// Status handler operating directly on `RoutingLoopRuntime`.
async fn handle_status_runtime(runtime: &RoutingLoopRuntime) -> Response {
    let is_routing = runtime.is_routing.load(Ordering::Relaxed);
    let is_paused = runtime.is_paused.load(Ordering::Relaxed);
    let queue_size = runtime.request_queue.lock().await.len();
    tracing::debug!(
        is_routing,
        is_paused,
        queue_size,
        "routing_loop_status queried"
    );
    (
        StatusCode::OK,
        Json(serde_json::json!({ "is_routing": is_routing })),
    )
        .into_response()
}

/// Filter handler operating directly on `RoutingLoopRuntime`.
///
/// PR 10 §10.8: Iterates the actual `MultiPriorityRequestQueue` instead of
/// the stub `Vec<QueuedRequestMeta>` from `RoutingLoopState`.
async fn handle_filter_runtime(runtime: &RoutingLoopRuntime, version_tag: i64) -> Response {
    tracing::debug!(
        version_tag,
        paused = runtime.is_paused.load(Ordering::Relaxed),
        is_routing = runtime.is_routing.load(Ordering::Relaxed),
        "routing_loop_filter requested"
    );

    let queue = runtime.request_queue.lock().await;
    let queue_size = queue.len();

    // Iterate all entries in the multi-priority queue matching version_tag threshold.
    let requests: Vec<serde_json::Value> = queue
        .iter_requests()
        .filter(|entry| {
            entry
                .routing_meta
                .as_ref()
                .is_some_and(|m| m.version_tag <= version_tag)
        })
        .filter_map(|entry| {
            entry.routing_meta.as_ref().map(|m| {
                serde_json::json!({
                    "request_id": m.request_id,
                    "model_id": entry.ctx.input.model_id.as_deref().unwrap_or("unknown"),
                    "is_validate": m.is_validate,
                    "version_tag": m.version_tag,
                })
            })
        })
        .collect();

    tracing::debug!(
        version_tag,
        matched_count = requests.len(),
        queue_size,
        "routing_loop_filter finished"
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "version_tag_threshold": version_tag,
            "count": requests.len(),
            "requests": requests,
        })),
    )
        .into_response()
}
