// PR 5A §5A.1a-d: Routing loop controller endpoints for pause, resume, status, and filter.
//!
//! Provides HTTP handlers for managing the PSRL routing loop lifecycle:
//! - `routing_loop_pause()` — pause dispatch
//! - `routing_loop_resume()` — resume dispatch
//! - `routing_loop_status()` — query running/paused state
//! - `routing_loop_filter()` — list queued requests matching a version_tag threshold
//!
//! These handlers operate on a shared `RoutingLoopState` that is integrated
//! into `AppContext` when `enable_routing_loop = true`.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use tokio::sync::Mutex;
use tracing::info;

use crate::routers::routing_loop_utils::RoutingMeta;

// ── Shared state ────────────────────────────────────────────────────────

// PR 5A §5A.1a: Shared routing loop state for controller endpoints.
/// Shared state backing the routing loop controller endpoints.
///
/// This struct is the minimal state required for pause/resume/status/filter.
/// The full `RoutingLoopRuntime` (PR 6) will compose this state with queue
/// infrastructure, PS Manager client, and dispatch channels.
pub struct RoutingLoopState {
    /// Whether the routing loop is currently dispatching requests.
    pub is_routing: AtomicBool,
    /// Whether the routing loop has been administratively paused.
    pub is_paused: AtomicBool,
    /// Lightweight metadata for queued requests, used by the filter endpoint.
    pub request_queue: Mutex<Vec<QueuedRequestMeta>>,
}

// PR 5A §5A.1d: Lightweight metadata for a queued routing loop request.
/// Lightweight metadata snapshot of a queued routing-loop request.
///
/// Used by `routing_loop_filter()` to report matching entries without
/// exposing full request bodies or response channels.
#[derive(Debug, Clone)]
pub struct QueuedRequestMeta {
    pub routing_meta: Option<RoutingMeta>,
    pub model_id: Option<String>,
}

impl RoutingLoopState {
    /// Create a new routing loop state with routing inactive and unpaused.
    pub fn new() -> Self {
        Self {
            is_routing: AtomicBool::new(false),
            is_paused: AtomicBool::new(false),
            request_queue: Mutex::new(Vec::new()),
        }
    }
}

impl Default for RoutingLoopState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Query parameters ────────────────────────────────────────────────────

// PR 5A §5A.1d: Query parameter for routing_loop_filter.
/// Query parameter for the `/routing_loop/filter` endpoint.
#[derive(Debug, Deserialize)]
pub struct RoutingLoopFilterQuery {
    pub version_tag: i64,
}

// ── Core logic (testable without AppState) ──────────────────────────────

// PR 5A §5A.1a: Pause routing loop dispatch.
/// Pause the routing loop. Sets `is_paused = true` and returns `{ "running": false }`.
pub async fn handle_pause(state: &RoutingLoopState) -> Response {
    state.is_paused.store(true, Ordering::Relaxed);

    let queue_size = {
        let queue = state.request_queue.lock().await;
        queue.len()
    };

    info!(
        paused = true,
        queue_size,
        is_routing = state.is_routing.load(Ordering::Relaxed),
        "routing_loop_pause called"
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "running": false,
        })),
    )
        .into_response()
}

// PR 5A §5A.1b: Resume routing loop dispatch.
/// Resume the routing loop. Sets `is_paused = false` and returns `{ "running": true }`.
pub async fn handle_resume(state: &RoutingLoopState) -> Response {
    state.is_paused.store(false, Ordering::Relaxed);

    let queue_size = {
        let queue = state.request_queue.lock().await;
        queue.len()
    };

    info!(
        paused = false,
        queue_size,
        is_routing = state.is_routing.load(Ordering::Relaxed),
        "routing_loop_resume called"
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "running": true,
        })),
    )
        .into_response()
}

// PR 5A §5A.1c: Query routing loop status.
/// Returns `{ "is_routing": bool }`.
pub async fn handle_status(state: &RoutingLoopState) -> Response {
    let is_routing = state.is_routing.load(Ordering::Relaxed);
    let is_paused = state.is_paused.load(Ordering::Relaxed);

    let queue_size = {
        let queue = state.request_queue.lock().await;
        queue.len()
    };

    tracing::debug!(
        is_routing,
        is_paused,
        queue_size,
        "routing_loop_status queried"
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "is_routing": is_routing,
        })),
    )
        .into_response()
}

// PR 5A §5A.1d: Filter queued requests by version_tag.
/// List queued requests with `version_tag <= threshold`.
///
/// Returns `{ "version_tag_threshold": N, "count": M, "requests": [...] }`.
pub async fn handle_filter(state: &RoutingLoopState, version_tag: i64) -> Response {
    tracing::debug!(
        version_tag,
        paused = state.is_paused.load(Ordering::Relaxed),
        is_routing = state.is_routing.load(Ordering::Relaxed),
        "routing_loop_filter requested"
    );

    let queue = state.request_queue.lock().await;
    let queue_size = queue.len();

    let requests: Vec<serde_json::Value> = queue
        .iter()
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
                    "model_id": entry.model_id,
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
pub async fn routing_loop_pause(
    State(state): State<Arc<crate::server::AppState>>,
) -> Response {
    match state.context.routing_loop_state.as_ref() {
        Some(rl_state) => handle_pause(rl_state).await,
        None => routing_loop_not_enabled(),
    }
}

/// `POST /routing_loop/resume`
pub async fn routing_loop_resume(
    State(state): State<Arc<crate::server::AppState>>,
) -> Response {
    match state.context.routing_loop_state.as_ref() {
        Some(rl_state) => handle_resume(rl_state).await,
        None => routing_loop_not_enabled(),
    }
}

/// `GET /routing_loop/status`
pub async fn routing_loop_status(
    State(state): State<Arc<crate::server::AppState>>,
) -> Response {
    match state.context.routing_loop_state.as_ref() {
        Some(rl_state) => handle_status(rl_state).await,
        None => routing_loop_not_enabled(),
    }
}

/// `GET /routing_loop/filter?version_tag=N`
pub async fn routing_loop_filter(
    State(state): State<Arc<crate::server::AppState>>,
    Query(query): Query<RoutingLoopFilterQuery>,
) -> Response {
    match state.context.routing_loop_state.as_ref() {
        Some(rl_state) => handle_filter(rl_state, query.version_tag).await,
        None => routing_loop_not_enabled(),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> Arc<RoutingLoopState> {
        Arc::new(RoutingLoopState::new())
    }

    fn make_routing_meta(request_id: i64, version_tag: i64, is_validate: bool) -> RoutingMeta {
        RoutingMeta {
            request_id: Some(request_id),
            prompt_id: None,
            version_tag,
            is_validate,
            rollout_instance_hint: None,
        }
    }

    /// Extract JSON body from an axum Response.
    async fn response_json(response: Response) -> serde_json::Value {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        serde_json::from_slice(&body).expect("parse JSON")
    }

    // PR 5A §5A.1a test: pause sets is_paused and returns running=false
    #[tokio::test]
    async fn test_routing_loop_pause_sets_flag() {
        let state = make_state();
        assert!(!state.is_paused.load(Ordering::Relaxed));

        let response = handle_pause(&state).await;
        assert_eq!(response.status(), StatusCode::OK);

        assert!(state.is_paused.load(Ordering::Relaxed));

        let json = response_json(response).await;
        assert_eq!(json["running"], false);
    }

    // PR 5A §5A.1b test: resume clears is_paused and returns running=true
    #[tokio::test]
    async fn test_routing_loop_resume_clears_flag() {
        let state = make_state();
        state.is_paused.store(true, Ordering::Relaxed);

        let response = handle_resume(&state).await;
        assert_eq!(response.status(), StatusCode::OK);

        assert!(!state.is_paused.load(Ordering::Relaxed));

        let json = response_json(response).await;
        assert_eq!(json["running"], true);
    }

    // PR 5A §5A.1c test: status returns current is_routing value
    #[tokio::test]
    async fn test_routing_loop_status_returns_is_routing() {
        let state = make_state();

        // Default: not routing
        let response = handle_status(&state).await;
        let json = response_json(response).await;
        assert_eq!(json["is_routing"], false);

        // Set routing active
        state.is_routing.store(true, Ordering::Relaxed);
        let response = handle_status(&state).await;
        let json = response_json(response).await;
        assert_eq!(json["is_routing"], true);
    }

    // PR 5A §5A.1d test: filter returns matching entries
    #[tokio::test]
    async fn test_routing_loop_filter_returns_matching() {
        let state = make_state();

        {
            let mut queue = state.request_queue.lock().await;
            queue.push(QueuedRequestMeta {
                routing_meta: Some(make_routing_meta(1, 3, false)),
                model_id: Some("model-a".to_string()),
            });
            queue.push(QueuedRequestMeta {
                routing_meta: Some(make_routing_meta(2, 5, true)),
                model_id: Some("model-b".to_string()),
            });
            queue.push(QueuedRequestMeta {
                routing_meta: Some(make_routing_meta(3, 7, false)),
                model_id: None,
            });
        }

        // Filter with version_tag=5: should match entries with version_tag <= 5
        let response = handle_filter(&state, 5).await;

        assert_eq!(response.status(), StatusCode::OK);
        let json = response_json(response).await;
        assert_eq!(json["version_tag_threshold"], 5);
        assert_eq!(json["count"], 2);

        let requests = json["requests"].as_array().expect("requests array");
        assert_eq!(requests.len(), 2);

        assert_eq!(requests[0]["request_id"], 1);
        assert_eq!(requests[0]["version_tag"], 3);
        assert_eq!(requests[0]["model_id"], "model-a");
        assert!(!requests[0]["is_validate"].as_bool().expect("bool"));

        assert_eq!(requests[1]["request_id"], 2);
        assert_eq!(requests[1]["version_tag"], 5);
        assert_eq!(requests[1]["model_id"], "model-b");
        assert!(requests[1]["is_validate"].as_bool().expect("bool"));
    }

    // PR 5A §5A.1d test: filter with empty queue returns empty list
    #[tokio::test]
    async fn test_routing_loop_filter_empty_queue() {
        let state = make_state();

        let response = handle_filter(&state, 10).await;

        assert_eq!(response.status(), StatusCode::OK);
        let json = response_json(response).await;
        assert_eq!(json["count"], 0);
        assert_eq!(json["requests"].as_array().expect("array").len(), 0);
    }

    // PR 5A §5A.1d test: filter with no matching version_tag
    #[tokio::test]
    async fn test_routing_loop_filter_no_match() {
        let state = make_state();

        {
            let mut queue = state.request_queue.lock().await;
            queue.push(QueuedRequestMeta {
                routing_meta: Some(make_routing_meta(1, 10, false)),
                model_id: Some("model-a".to_string()),
            });
            // Entry without routing_meta should not match
            queue.push(QueuedRequestMeta {
                routing_meta: None,
                model_id: Some("model-b".to_string()),
            });
        }

        // version_tag=5: no entries have version_tag <= 5
        let response = handle_filter(&state, 5).await;

        let json = response_json(response).await;
        assert_eq!(json["count"], 0);
        assert_eq!(json["requests"].as_array().expect("array").len(), 0);
    }
}
