//! Control-plane handlers for the request routing loop.

use std::sync::Arc;

use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use super::runtime::RoutingLoopRuntime;
use crate::{routers::error, server::AppState};

fn runtime_from_state(state: &AppState) -> Result<Arc<RoutingLoopRuntime>, Response> {
    state
        .context
        .routing_loop_runtime
        .clone()
        .ok_or_else(|| error::not_found("routing_loop_disabled", "Routing loop is not enabled"))
}

pub(crate) async fn pause_routing_loop(State(state): State<Arc<AppState>>) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => {
            runtime.pause();
            Json(json!({"status": "paused"})).into_response()
        }
        Err(response) => response,
    }
}

pub(crate) async fn resume_routing_loop(State(state): State<Arc<AppState>>) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => {
            runtime.resume();
            Json(json!({"status": "running"})).into_response()
        }
        Err(response) => response,
    }
}

pub(crate) async fn routing_loop_status(State(state): State<Arc<AppState>>) -> Response {
    match runtime_from_state(&state) {
        Ok(runtime) => Json(runtime.status().await).into_response(),
        Err(response) => response,
    }
}
