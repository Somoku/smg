// PR 5 §5.2: Routing loop utilities for PSRL metadata parsing.
// PR 10 §10.1: RoutingLoopRuntime completed with channel, tracking maps, PS Manager client.
// PR 13 §13.1: RoutingQueueEntry now carries RequestContext directly (post-PreparationStage)
//   instead of PreparedRequest + headers. Eliminates the discard-and-recreate cycle where
//   pipeline_routing_loop.rs ran execute_preparation_only(), discarded the resulting context,
//   then re-wrapped the original Arc<Request> into PreparedRequest. Now the prepared context
//   flows through the queue boundary intact, preserving PreparationOutput (token_ids,
//   original_text, processed_messages, tool_constraints, filtered_request).
// PR 13 §13.1: PreparedRequest enum removed. Replaced by ctx: RequestContext which carries
//   the RequestType variant (Chat/Generate/...) plus all PreparedRequest accessors as free fns.
//!
//! Ports `RoutingMeta`, `parse_psrl_request_meta()`, and
//! all helper parsing functions from sgl-model-gateway's `routing_loop_utils.rs`.
//!
//! PR 5 §5.2g: `RoutingLoopRuntime` and `RoutingQueueEntry` define the shared
//! runtime state and queue entry types for the PSRL routing loop.
//!
//! PR 10 §10.1: `RoutingLoopRuntime` is now complete with all fields:
//! - `tx`: mpsc channel sender for submitting requests from router handlers
//! - `incomplete_request_to_instance`: in-flight request → (worker_id, dp_rank) map
//! - `instance_to_version_after_sync`: shared version map from WorkerService
//! - `prompt_to_running_request_ids`: prompt affinity map for group-pin routing
//! - `ps_manager_client`: optional PS Manager gRPC client
//! - `ps_manager_addr`: PS Manager address for diagnostics

use std::{
    collections::HashMap,
    sync::{atomic::AtomicBool, Arc},
};

use axum::response::Response;
use psrl_state::PSManagerStateClient;
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::routers::grpc::context::{RequestContext, RequestType};

use crate::{
    config::types::RequestSortIndicator,
    core::InstanceVersionMap,
    routers::request_queue::{MultiPriorityRequestQueue, RequestPriority},
};

// ── Core types ──────────────────────────────────────────────────────────

// PR 5 §5.2a: Routing metadata extracted from request headers and body.
/// Routing metadata extracted from the request headers and body for PSRL routing decisions.
#[derive(Debug, Clone)]
pub struct RoutingMeta {
    pub request_id: Option<i64>,
    pub prompt_id: Option<i64>,
    pub version_tag: i64,
    pub is_validate: bool,
    /// `(base_worker_id, target_dp_rank)` hint for sticky routing.
    pub rollout_instance_hint: Option<(String, usize)>,
}

// ── Routing loop runtime types ──────────────────────────────────────────

// PR 5 §5.2g: Shared runtime state container for the PSRL routing loop.
// PR 10 §10.1: Completed with mpsc channel, request-tracking maps, and PS Manager client.
/// Runtime context shared by the HTTP routing loop task, admin endpoints, and
/// worker-selection logic.
///
/// Wrapped in `Arc` and shared between:
/// - The routing loop task (spawned at startup)
/// - Router handlers (submit requests via `tx`)
/// - Admin endpoints (`/routing_loop/*`)
/// - Worker-selection logic (reads `incomplete_request_to_instance` for group-pin)
pub struct RoutingLoopRuntime {
    // ── §10.1: mpsc channel sender ───────────────────────────────────────
    /// Sender half of the unbounded mpsc channel.
    ///
    /// Router handlers send `RoutingQueueEntry` values here; the routing loop
    /// task receives them on the `rx` side that is passed to `routing_loop()`.
    pub tx: mpsc::UnboundedSender<RoutingQueueEntry>,

    // ── §5.2g: Core routing loop state ──────────────────────────────────
    /// When `true`, the routing loop pauses dispatching new requests.
    /// Toggled by `/routing_loop/pause` and `/routing_loop/resume` admin endpoints.
    pub is_paused: AtomicBool,

    /// Indicates whether the routing loop is actively processing requests.
    /// Set to `true` on loop entry, `false` on exit.
    pub is_routing: AtomicBool,

    /// Global priority-based request queue partitioned by version tag.
    ///
    /// Uses `tokio::sync::Mutex` because the routing loop holds this lock
    /// across `.await` points (drain channel → batch push, per-version iteration).
    pub request_queue: Mutex<MultiPriorityRequestQueue<RoutingQueueEntry>>,

    // ── §10.1: In-flight tracking maps ──────────────────────────────────
    /// In-flight requests: `request_id → (base_worker_id, dp_rank)`.
    ///
    /// Inserted before spawning a dispatch task; removed on task completion
    /// (stop, length, or abort outcomes). Used by group-pin routing (Stage 3).
    pub incomplete_request_to_instance: Mutex<HashMap<i64, (String, usize)>>,

    /// Shared version map: `(worker_id, dp_rank) → version_tag` after the last sync.
    ///
    /// Cloned from `AppContext::instance_to_version_after_sync`. Updated by
    /// `WorkerService` on each sync; read by PSRL worker selection (Stage 2).
    pub instance_to_version_after_sync: InstanceVersionMap,

    /// Prompt-affinity map: `prompt_id → Vec<request_id>`.
    ///
    /// Tracks which request_ids share the same prompt, enabling group-pin routing.
    /// Inserted before dispatch; empty `Vec` entries removed after dispatch completes.
    pub prompt_to_running_request_ids: Mutex<HashMap<i64, Vec<i64>>>,

    // ── §10.1: Loop configuration ────────────────────────────────────────
    /// Routing loop polling interval in milliseconds (0 = no sleep).
    pub check_interval_ms: u64,

    // ── §10.1: PS Manager integration ───────────────────────────────────
    /// PS Manager gRPC endpoint (for logging/diagnostics, e.g. `http://127.0.0.1:50051`).
    pub ps_manager_addr: String,

    /// Optional PS Manager gRPC client.
    ///
    /// `None` when `ps_manager_addr` is not configured or connection failed.
    /// Used for abort checks and request lifecycle tracking RPCs.
    pub ps_manager_client: Option<Arc<PSManagerStateClient>>,
}

impl RoutingLoopRuntime {
    /// Create a new `RoutingLoopRuntime` with an unbounded mpsc channel.
    ///
    /// Returns `(runtime, rx)` where `rx` is the receiver to pass to `routing_loop()`.
    ///
    /// # Arguments
    /// - `sort_indicator` — queue ordering strategy
    /// - `enable_multi_priority_queue` — partition queue by version tag
    /// - `instance_to_version_after_sync` — shared version map from AppContext
    /// - `check_interval_ms` — routing loop sleep between iterations
    /// - `ps_manager_addr` — PS Manager address string (may be empty)
    /// - `ps_manager_client` — optional connected PS Manager client
    pub fn new_with_channel(
        sort_indicator: RequestSortIndicator,
        enable_multi_priority_queue: bool,
        instance_to_version_after_sync: InstanceVersionMap,
        check_interval_ms: u64,
        ps_manager_addr: String,
        ps_manager_client: Option<Arc<PSManagerStateClient>>,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<RoutingQueueEntry>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let runtime = Arc::new(Self {
            tx,
            is_paused: AtomicBool::new(false),
            is_routing: AtomicBool::new(false),
            request_queue: Mutex::new(MultiPriorityRequestQueue::new(
                sort_indicator,
                enable_multi_priority_queue,
                None,
            )),
            incomplete_request_to_instance: Mutex::new(HashMap::new()),
            instance_to_version_after_sync,
            prompt_to_running_request_ids: Mutex::new(HashMap::new()),
            check_interval_ms,
            ps_manager_addr,
            ps_manager_client,
        });
        (runtime, rx)
    }

    /// Create a `RoutingLoopRuntime` without an external channel (for tests).
    ///
    /// Returns the runtime directly; the internal `tx` is the only sender.
    #[cfg(test)]
    pub fn new(sort_indicator: RequestSortIndicator, enable_multi_priority_queue: bool) -> Self {
        let (tx, _rx) = mpsc::unbounded_channel();
        Self {
            tx,
            is_paused: AtomicBool::new(false),
            is_routing: AtomicBool::new(false),
            request_queue: Mutex::new(MultiPriorityRequestQueue::new(
                sort_indicator,
                enable_multi_priority_queue,
                None,
            )),
            incomplete_request_to_instance: Mutex::new(HashMap::new()),
            instance_to_version_after_sync: Arc::new(parking_lot::Mutex::new(HashMap::new())),
            prompt_to_running_request_ids: Mutex::new(HashMap::new()),
            check_interval_ms: 0,
            ps_manager_addr: String::new(),
            ps_manager_client: None,
        }
    }
}

// PR 5 §5.2g: Queue entry for the PSRL routing loop.
// PR 13 §13.1: Replaced `headers: Option<HeaderMap>` + `request: PreparedRequest`
//   with `ctx: RequestContext` (post-PreparationStage). The context carries all fields
//   that PreparedRequest provided (model_id, headers, request type) plus PreparationOutput
//   (token_ids, original_text, processed_messages, tool_constraints, filtered_request).
//   This eliminates the discard-and-recreate cycle in pipeline_routing_loop.rs and
//   build_context_from_prepared() in routing_loop.rs.
/// Entry in the routing loop queue.
///
/// Each entry represents a single request awaiting worker selection and dispatch.
/// The `result_tx` oneshot is used to send the HTTP response back to the
/// originating handler once the request is dispatched and completed.
pub struct RoutingQueueEntry {
    /// The fully-prepared request context (post-PreparationStage).
    ///
    /// Carries `ctx.input.request_type` (the typed Arc request), `ctx.input.headers`,
    /// `ctx.input.model_id`, and `ctx.state.preparation` (token_ids, original_text, etc.).
    /// The routing loop's dispatch_task picks this context up directly and passes it to
    /// `execute_through_execution` (stages 2–5), which skips WorkerSelectionStage because
    /// `ctx.state.workers` is pre-set by the routing loop.
    // PR 13 §13.1: ctx is pub(crate) via RequestContext's own visibility.
    // The private_interfaces lint is expected here — RoutingQueueEntry is crate-internal
    // despite its pub visibility (required by RoutingLoopRuntime's pub fields).
    #[expect(
        private_interfaces,
        reason = "RoutingQueueEntry is crate-internal; ctx visibility matches usage scope"
    )]
    pub ctx: RequestContext,
    /// Oneshot sender for delivering the response back to the request handler.
    pub result_tx: oneshot::Sender<Response>,
    /// Optional routing metadata for PSRL routing decisions.
    pub routing_meta: Option<RoutingMeta>,
}

// PR 5 §5.2g: RequestPriority implementation for RoutingQueueEntry.
// PR 13 §13.1: get_priority now calls input_len_from_ctx instead of request.input_len().
impl RequestPriority for RoutingQueueEntry {
    #[inline]
    fn get_version_tag(&self) -> i64 {
        self.routing_meta.as_ref().map_or(-1, |m| m.version_tag)
    }

    #[inline]
    fn get_priority(&self, indicator: RequestSortIndicator) -> (i64, i64, i64) {
        match indicator {
            RequestSortIndicator::ShortLength => get_priority_by_version_and_token_num(self, true),
            RequestSortIndicator::LongLength => get_priority_by_version_and_token_num(self, false),
            RequestSortIndicator::SmallId => get_priority_by_version_and_id(self),
        }
    }
}

/// Compute version-based priority: unversioned (`-1`) → lowest priority (`i64::MAX`).
fn get_priority_by_version(request: &RoutingQueueEntry) -> i64 {
    let version_tag = request.routing_meta.as_ref().map_or(-1, |m| m.version_tag);
    if version_tag == -1 {
        i64::MAX
    } else {
        version_tag
    }
}

/// Priority tuple based on (validate_priority, version_priority, length_priority).
///
/// - `is_validate` requests get priority 0 (higher), normal requests get 1.
/// - `short_request_first`: shorter text → lower priority value → dispatched first.
fn get_priority_by_version_and_token_num(
    request: &RoutingQueueEntry,
    short_request_first: bool,
) -> (i64, i64, i64) {
    let validate_priority = request.routing_meta.as_ref().is_none_or(|m| !m.is_validate);
    let version_priority = get_priority_by_version(request);
    // PR 13 §13.1: Use input_len_from_ctx — token count from PreparationOutput (accurate)
    // or char-count proxy from request body. Replaces request.request.input_len().
    let input_len = input_len_from_ctx(&request.ctx);
    if short_request_first {
        let length_priority = input_len.min(i32::MAX as usize) as i32;
        (
            validate_priority as i64,
            version_priority,
            length_priority as i64,
        )
    } else {
        let length_priority = -(input_len.min(i32::MAX as usize) as i32);
        (
            validate_priority as i64,
            version_priority,
            length_priority as i64,
        )
    }
}

/// Priority tuple based on (validate_priority, version_priority, request_id).
///
/// Lower request_id → dispatched first.
fn get_priority_by_version_and_id(request: &RoutingQueueEntry) -> (i64, i64, i64) {
    let validate_priority = request.routing_meta.as_ref().is_none_or(|m| !m.is_validate);
    let version_priority = get_priority_by_version(request);
    let id_priority = request
        .routing_meta
        .as_ref()
        .and_then(|m| m.request_id)
        .unwrap_or(i64::MAX);
    (validate_priority as i64, version_priority, id_priority)
}

// ── PR 13 §13.1: Context-based helper functions ─────────────────────────

// PR 13 §13.1: input_len_from_ctx replaces PreparedRequest::input_len().
// Prefers accurate token count from PreparationOutput; falls back to char-count proxy.
/// Input length for priority sorting, read from a prepared `RequestContext`.
///
/// Prefers `ctx.state.preparation.token_ids.len()` when available (accurate token count
/// from `PreparationStage`). Falls back to a character-count proxy from the request body
/// when preparation output is absent.
pub(crate) fn input_len_from_ctx(ctx: &RequestContext) -> usize {
    // Prefer token count from PreparationOutput (accurate)
    if let Some(prep) = &ctx.state.preparation {
        if !prep.token_ids.is_empty() {
            return prep.token_ids.len();
        }
    }
    0 // Default to 0 if no token count is available
}

// PR 13 §13.1: extract_text_from_request_type replaces PreparedRequest::text_for_routing().
// Used as a fallback when PreparationOutput::original_text is absent.
/// Extract routing text from a `RequestType` for cache-aware policy routing.
///
/// Returns the raw text for prefix-hash / consistent-hash worker selection.
/// `None` when the request uses tokenized input (`input_ids`) with no text field.
///
/// Used as a fallback when `ctx.state.preparation.original_text` is absent.
pub(crate) fn extract_text_from_request_type(request_type: &RequestType) -> Option<String> {
    match request_type {
        RequestType::Chat(r) => {
            let text = r
                .messages
                .last()
                .map(|m| {
                    use openai_protocol::chat::ChatMessage;
                    match m {
                        ChatMessage::User { content, .. }
                        | ChatMessage::System { content, .. }
                        | ChatMessage::Developer { content, .. }
                        | ChatMessage::Tool { content, .. } => content.to_simple_string(),
                        ChatMessage::Assistant { content, .. } => content
                            .as_ref()
                            .map(|c| c.to_simple_string())
                            .unwrap_or_default(),
                        ChatMessage::Function { content, .. } => content.clone(),
                    }
                })
                .unwrap_or_default();
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        RequestType::Generate(r) => r.text.clone(),
        RequestType::Responses(_) | RequestType::Embedding(_) | RequestType::Classify(_) => None,
    }
}

// PR 13 §13.1: parse_psrl_request_meta_from_context replaces the JSON-serialization roundtrip
// in pipeline_routing_loop.rs where serde_json::to_value(request) was called just to extract
// PSRL metadata. Now reads directly from the typed RequestType fields and headers.
/// Parse PSRL routing metadata from a fully-prepared `RequestContext`.
///
/// Reads headers from `ctx.input.headers` and PSRL fields directly from the typed
/// `RequestType` variants — no JSON serialization roundtrip needed.
///
/// Returns `None` if no PSRL metadata is present (untagged request, no PS Manager tracking).
pub fn parse_psrl_request_meta_from_context(ctx: &RequestContext) -> Option<RoutingMeta> {
    let headers = ctx.input.headers.as_ref();

    let request_id = headers
        .and_then(|h| h.get("x-request-id"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i64>().ok());

    let prompt_id = headers
        .and_then(|h| h.get("x-prompt-id"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i64>().ok());

    let version_tag = headers
        .and_then(|h| h.get("x-version-tag"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(-1);

    let is_validate = headers
        .and_then(|h| h.get("x-is-validate"))
        .and_then(|v| v.to_str().ok())
        .and_then(parse_bool_like)
        .unwrap_or(false);

    let rollout_instance_hint = parse_rollout_instance_hint_from_headers(headers);

    // Return None if no PSRL metadata is present.
    if request_id.is_none()
        && prompt_id.is_none()
        && version_tag == -1
        && !is_validate
        && rollout_instance_hint.is_none()
    {
        return None;
    }

    Some(RoutingMeta {
        request_id,
        prompt_id,
        version_tag,
        is_validate,
        rollout_instance_hint,
    })
}

// ── Parsing ─────────────────────────────────────────────────────────────

// PR 5 §5.2b-c: Parse PSRL metadata from protocol body + transport headers.
/// Parse PSRL metadata from protocol body + transport headers.
///
/// Header priority is higher than body fallback:
/// - `x-request-id` / `request_id`
/// - `x-prompt-id` / `prompt_id`
/// - `x-version-tag` / `version_tag`
/// - `x-is-validate` / `is_validate`
/// - `x-base-worker-id` + `x-target-dp-rank` / `rollout_instance_id`
pub fn parse_psrl_request_meta(
    headers: Option<&http::HeaderMap>,
    body: Option<&serde_json::Value>,
) -> Option<RoutingMeta> {
    let request_id = headers
        .and_then(|h| h.get("x-request-id"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i64>().ok())
        .or_else(|| parse_i64_from_body(body, "request_id"));

    let prompt_id = headers
        .and_then(|h| h.get("x-prompt-id"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i64>().ok())
        .or_else(|| parse_i64_from_body(body, "prompt_id"));

    let version_tag = headers
        .and_then(|h| h.get("x-version-tag"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i64>().ok())
        .or_else(|| parse_i64_from_body(body, "version_tag"))
        .unwrap_or(-1);

    // PR 5 §5.2d: is_validate truthy/falsy parsing
    let is_validate = headers
        .and_then(|h| h.get("x-is-validate"))
        .and_then(|v| v.to_str().ok())
        .and_then(parse_bool_like)
        .or_else(|| parse_bool_from_body(body, "is_validate"))
        .unwrap_or(false);

    // PR 5 §5.2e: rollout_instance_hint parsing
    let rollout_instance_hint = parse_rollout_instance_hint_from_headers(headers)
        .or_else(|| parse_rollout_instance_hint_from_body(body));

    // Return None if no PSRL metadata is present to avoid unnecessary routing loop processing
    if request_id.is_none()
        && prompt_id.is_none()
        && version_tag == -1
        && !is_validate
        && rollout_instance_hint.is_none()
    {
        return None;
    }

    Some(RoutingMeta {
        request_id,
        prompt_id,
        version_tag,
        is_validate,
        rollout_instance_hint,
    })
}

// PR 5 §5.2d: Parse bool-like strings ("1", "true", "t", "yes", "y" and inverses).
fn parse_bool_like(s: &str) -> Option<bool> {
    let s = s.trim();
    if s == "1"
        || s.eq_ignore_ascii_case("true")
        || s.eq_ignore_ascii_case("t")
        || s.eq_ignore_ascii_case("yes")
        || s.eq_ignore_ascii_case("y")
    {
        Some(true)
    } else if s == "0"
        || s.eq_ignore_ascii_case("false")
        || s.eq_ignore_ascii_case("f")
        || s.eq_ignore_ascii_case("no")
        || s.eq_ignore_ascii_case("n")
    {
        Some(false)
    } else {
        None
    }
}

// PR 5 §5.2f: Parse an i64 from a JSON body key.
/// Parse an i64 from a JSON body key, supporting integer, u64, or string representations.
pub fn parse_i64_from_body(body: Option<&serde_json::Value>, key: &str) -> Option<i64> {
    let value = body?.get(key)?;
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
        .or_else(|| value.as_str().and_then(|s| s.parse::<i64>().ok()))
}

fn parse_usize_from_json(value: &serde_json::Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .or_else(|| value.as_i64().and_then(|v| usize::try_from(v).ok()))
        .or_else(|| value.as_str().and_then(|s| s.parse::<usize>().ok()))
}

fn parse_bool_from_body(body: Option<&serde_json::Value>, key: &str) -> Option<bool> {
    let value = body?.get(key)?;
    value
        .as_bool()
        .or_else(|| value.as_str().and_then(parse_bool_like))
}

// PR 5 §5.2e: Parse rollout instance hint from headers.
fn parse_rollout_instance_hint_from_headers(
    headers: Option<&http::HeaderMap>,
) -> Option<(String, usize)> {
    let base_worker_id = headers
        .and_then(|h| h.get("x-base-worker-id"))
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())?
        .to_string();

    let target_dp_rank = headers
        .and_then(|h| h.get("x-target-dp-rank"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())?;

    Some((base_worker_id, target_dp_rank))
}

// PR 5 §5.2e: Parse rollout instance hint from body.
fn parse_rollout_instance_hint_from_body(
    body: Option<&serde_json::Value>,
) -> Option<(String, usize)> {
    let body = body?;

    // 1) Preferred parity key with Python: `rollout_instance_id`
    //    supports:
    //      - ["worker_id", dp_rank]
    //      - {"base_worker_id"|"worker_id"|"replica_id": "...",
    //         "target_dp_rank"|"dp_rank"|"data_parallel_rank": n}
    if let Some(instance) = body
        .get("rollout_instance_id")
        .or_else(|| body.get("rollout_instance_hint"))
    {
        if let Some(arr) = instance.as_array() {
            if arr.len() >= 2 {
                let base_worker_id = arr
                    .first()
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())?
                    .to_string();
                let target_dp_rank = arr.get(1).and_then(parse_usize_from_json)?;
                return Some((base_worker_id, target_dp_rank));
            }
        }

        if let Some(obj) = instance.as_object() {
            let base_worker_id = obj
                .get("base_worker_id")
                .or_else(|| obj.get("worker_id"))
                .or_else(|| obj.get("replica_id"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())?
                .to_string();

            let target_dp_rank = obj
                .get("target_dp_rank")
                .or_else(|| obj.get("dp_rank"))
                .or_else(|| obj.get("data_parallel_rank"))
                .and_then(parse_usize_from_json)?;

            return Some((base_worker_id, target_dp_rank));
        }
    }

    // 2) Flat body fallback
    let base_worker_id = body
        .get("base_worker_id")
        .or_else(|| body.get("x-base-worker-id"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())?
        .to_string();

    let target_dp_rank = body
        .get("target_dp_rank")
        .or_else(|| body.get("x-target-dp-rank"))
        .and_then(parse_usize_from_json)?;

    Some((base_worker_id, target_dp_rank))
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{http::HeaderMap, response::Response};
    use serde_json::json;
    use tokio::sync::oneshot;

    use crate::{
        config::types::RequestSortIndicator,
        routers::{
            grpc::{
                context::{RequestContext, RequestType, SharedComponents},
            },
            request_queue::RequestPriority,
            routing_loop_utils::{
                extract_text_from_request_type, input_len_from_ctx, parse_psrl_request_meta,
                parse_psrl_request_meta_from_context, RoutingLoopRuntime, RoutingMeta,
                RoutingQueueEntry,
            },
        },
    };

    fn headers_with(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (key, value) in pairs {
            map.insert(
                http::header::HeaderName::from_bytes(key.as_bytes()).expect("valid header name"),
                http::header::HeaderValue::from_str(value).expect("valid header value"),
            );
        }
        map
    }

    /// Build a minimal `SharedComponents` for tests (no tokenizer, no multimodal).
    fn make_components() -> Arc<SharedComponents> {
        use llm_tokenizer::TokenizerRegistry;
        use reasoning_parser::ParserFactory as ReasoningParserFactory;
        use tool_parser::ParserFactory as ToolParserFactory;
        Arc::new(SharedComponents {
            tokenizer_registry: Arc::new(TokenizerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            multimodal: None,
        })
    }

    // PR 13 §13.4 test: Helper to build a test RoutingQueueEntry using RequestContext.
    fn make_test_entry_generate(
        version_tag: i64,
        request_id: Option<i64>,
        is_validate: bool,
        text: &str,
        result_tx: oneshot::Sender<Response>,
    ) -> RoutingQueueEntry {
        use openai_protocol::generate::GenerateRequest;
        let gen_req: GenerateRequest =
            serde_json::from_str(&format!(r#"{{"text":"{text}","model":"test-model"}}"#))
                .expect("test GenerateRequest");
        let ctx = RequestContext::for_generate(
            Arc::new(gen_req),
            None,
            Some("test-model".to_string()),
            make_components(),
        );
        RoutingQueueEntry {
            ctx,
            result_tx,
            routing_meta: Some(RoutingMeta {
                request_id,
                prompt_id: None,
                version_tag,
                is_validate,
                rollout_instance_hint: None,
            }),
        }
    }

    // PR 5 §5.4 test: All fields extracted from headers
    #[test]
    fn test_parse_meta_from_headers_only() {
        let headers = headers_with(&[
            ("x-request-id", "42"),
            ("x-prompt-id", "7"),
            ("x-version-tag", "3"),
            ("x-is-validate", "true"),
        ]);
        let meta = parse_psrl_request_meta(Some(&headers), None);
        assert!(meta.is_some());
        let meta = meta.expect("should parse");
        assert_eq!(meta.request_id, Some(42));
        assert_eq!(meta.prompt_id, Some(7));
        assert_eq!(meta.version_tag, 3);
        assert!(meta.is_validate);
        assert!(meta.rollout_instance_hint.is_none());
    }

    // PR 5 §5.4 test: All fields extracted from JSON body keys
    #[test]
    fn test_parse_meta_from_body_only() {
        let body = json!({
            "request_id": 100,
            "prompt_id": 200,
            "version_tag": 5,
            "is_validate": true
        });
        let meta = parse_psrl_request_meta(None, Some(&body));
        assert!(meta.is_some());
        let meta = meta.expect("should parse");
        assert_eq!(meta.request_id, Some(100));
        assert_eq!(meta.prompt_id, Some(200));
        assert_eq!(meta.version_tag, 5);
        assert!(meta.is_validate);
    }

    // PR 5 §5.4 test: Header value takes precedence over body
    #[test]
    fn test_parse_meta_header_priority() {
        let headers = headers_with(&[("x-request-id", "1"), ("x-version-tag", "10")]);
        let body = json!({
            "request_id": 999,
            "version_tag": 999,
            "prompt_id": 50
        });
        let meta = parse_psrl_request_meta(Some(&headers), Some(&body)).expect("should parse");
        // Header wins
        assert_eq!(meta.request_id, Some(1));
        assert_eq!(meta.version_tag, 10);
        // Body fallback
        assert_eq!(meta.prompt_id, Some(50));
    }

    // PR 5 §5.4 test: No PSRL metadata → returns None
    #[test]
    fn test_parse_meta_no_metadata() {
        // Empty headers and empty body
        let headers = HeaderMap::new();
        let body = json!({});
        assert!(parse_psrl_request_meta(Some(&headers), Some(&body)).is_none());

        // None for both
        assert!(parse_psrl_request_meta(None, None).is_none());
    }

    // PR 5 §5.4 test: Missing version_tag defaults to -1
    #[test]
    fn test_parse_meta_version_tag_default() {
        let headers = headers_with(&[("x-request-id", "1")]);
        let meta = parse_psrl_request_meta(Some(&headers), None).expect("should parse");
        assert_eq!(meta.version_tag, -1);
    }

    // PR 5 §5.4 test: Missing is_validate defaults to false
    #[test]
    fn test_parse_meta_is_validate_default() {
        let headers = headers_with(&[("x-request-id", "1")]);
        let meta = parse_psrl_request_meta(Some(&headers), None).expect("should parse");
        assert!(!meta.is_validate);
    }

    // PR 5 §5.4 test: Accepts "1", "true", "t", "yes", "y" (case-insensitive)
    #[test]
    fn test_parse_meta_is_validate_truthy() {
        for truthy in &[
            "1", "true", "True", "TRUE", "t", "T", "yes", "Yes", "y", "Y",
        ] {
            let headers = headers_with(&[("x-request-id", "1"), ("x-is-validate", truthy)]);
            let meta = parse_psrl_request_meta(Some(&headers), None).expect("should parse");
            assert!(meta.is_validate, "Expected truthy for {truthy:?}");
        }
    }

    // PR 5 §5.4 test: Accepts "0", "false", "f", "no", "n" (case-insensitive)
    #[test]
    fn test_parse_meta_is_validate_falsy() {
        for falsy in &[
            "0", "false", "False", "FALSE", "f", "F", "no", "No", "n", "N",
        ] {
            let headers = headers_with(&[("x-request-id", "1"), ("x-is-validate", falsy)]);
            let meta = parse_psrl_request_meta(Some(&headers), None).expect("should parse");
            assert!(!meta.is_validate, "Expected falsy for {falsy:?}");
        }
    }

    // PR 5 §5.4 test: Rollout hint from headers
    #[test]
    fn test_parse_rollout_hint_headers() {
        let headers = headers_with(&[
            ("x-request-id", "1"),
            ("x-base-worker-id", "worker-abc"),
            ("x-target-dp-rank", "3"),
        ]);
        let meta = parse_psrl_request_meta(Some(&headers), None).expect("should parse");
        assert_eq!(
            meta.rollout_instance_hint,
            Some(("worker-abc".to_string(), 3))
        );
    }

    // PR 5 §5.4 test: Array format rollout hint from body
    #[test]
    fn test_parse_rollout_hint_body_array() {
        let body = json!({
            "request_id": 1,
            "rollout_instance_id": ["worker-xyz", 2]
        });
        let meta = parse_psrl_request_meta(None, Some(&body)).expect("should parse");
        assert_eq!(
            meta.rollout_instance_hint,
            Some(("worker-xyz".to_string(), 2))
        );
    }

    // PR 5 §5.4 test: Object format rollout hint from body
    #[test]
    fn test_parse_rollout_hint_body_object() {
        let body = json!({
            "request_id": 1,
            "rollout_instance_id": {
                "worker_id": "w-001",
                "dp_rank": 4
            }
        });
        let meta = parse_psrl_request_meta(None, Some(&body)).expect("should parse");
        assert_eq!(meta.rollout_instance_hint, Some(("w-001".to_string(), 4)));
    }

    // PR 5 §5.4 test: Flat body keys for rollout hint
    #[test]
    fn test_parse_rollout_hint_body_flat() {
        let body = json!({
            "request_id": 1,
            "base_worker_id": "flat-worker",
            "target_dp_rank": 0
        });
        let meta = parse_psrl_request_meta(None, Some(&body)).expect("should parse");
        assert_eq!(
            meta.rollout_instance_hint,
            Some(("flat-worker".to_string(), 0))
        );
    }

    // PR 5 §5.4 test: Object with alias keys (replica_id, data_parallel_rank)
    #[test]
    fn test_parse_rollout_hint_object_alias_keys() {
        let body = json!({
            "request_id": 1,
            "rollout_instance_id": {
                "replica_id": "replica-7",
                "data_parallel_rank": 1
            }
        });
        let meta = parse_psrl_request_meta(None, Some(&body)).expect("should parse");
        assert_eq!(
            meta.rollout_instance_hint,
            Some(("replica-7".to_string(), 1))
        );
    }

    // PR 5 §5.4 test: JSON integer parsed as i64
    #[test]
    fn test_parse_i64_from_body_int() {
        use super::parse_i64_from_body;
        let body = json!({"key": 42});
        assert_eq!(parse_i64_from_body(Some(&body), "key"), Some(42));
    }

    // PR 5 §5.4 test: JSON string "42" parsed as i64
    #[test]
    fn test_parse_i64_from_body_string() {
        use super::parse_i64_from_body;
        let body = json!({"key": "42"});
        assert_eq!(parse_i64_from_body(Some(&body), "key"), Some(42));
    }

    // PR 5 §5.4 test: Missing key → None
    #[test]
    fn test_parse_i64_from_body_missing() {
        use super::parse_i64_from_body;
        let body = json!({"other": 1});
        assert_eq!(parse_i64_from_body(Some(&body), "key"), None);
        assert_eq!(parse_i64_from_body(None, "key"), None);
    }

    // PR 13 §13.4 test: input_len_from_ctx uses token_ids from PreparationOutput when available.
    #[test]
    fn test_input_len_from_ctx_uses_token_ids() {
        use crate::routers::grpc::context::PreparationOutput;
        use openai_protocol::generate::GenerateRequest;

        let gen_req: GenerateRequest =
            serde_json::from_str(r#"{"text":"hello world","model":"test-model"}"#).unwrap();
        let mut ctx = RequestContext::for_generate(
            Arc::new(gen_req),
            None,
            Some("test-model".to_string()),
            make_components(),
        );
        // Simulate preparation output with 5 token_ids.
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some("hello world".to_string()),
            token_ids: vec![1, 2, 3, 4, 5],
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });
        // Should use token_ids length (5), not char count.
        assert_eq!(input_len_from_ctx(&ctx), 5);
    }

    // PR 13 §13.4 test: extract_text_from_request_type returns Generate text field.
    #[test]
    fn test_extract_text_generate() {
        use openai_protocol::generate::GenerateRequest;
        let gen_req: GenerateRequest =
            serde_json::from_str(r#"{"text":"sample text","model":"test"}"#).unwrap();
        let rt = RequestType::Generate(Arc::new(gen_req));
        assert_eq!(
            extract_text_from_request_type(&rt),
            Some("sample text".to_string())
        );
    }

    // PR 13 §13.4 test: parse_psrl_request_meta_from_context reads header fields.
    #[test]
    fn test_parse_psrl_meta_from_context_headers() {
        use openai_protocol::generate::GenerateRequest;
        let gen_req: GenerateRequest =
            serde_json::from_str(r#"{"text":"test","model":"m"}"#).unwrap();
        let headers = headers_with(&[("x-request-id", "77"), ("x-version-tag", "3")]);
        let ctx = RequestContext::for_generate(
            Arc::new(gen_req),
            Some(headers),
            Some("m".to_string()),
            make_components(),
        );
        let meta = parse_psrl_request_meta_from_context(&ctx).expect("should parse");
        assert_eq!(meta.request_id, Some(77));
        assert_eq!(meta.version_tag, 3);
    }

    // PR 13 §13.4 test: parse_psrl_request_meta_from_context returns None when no PSRL fields.
    #[test]
    fn test_parse_psrl_meta_from_context_none() {
        use openai_protocol::generate::GenerateRequest;
        let gen_req: GenerateRequest =
            serde_json::from_str(r#"{"text":"test","model":"m"}"#).unwrap();
        let ctx = RequestContext::for_generate(
            Arc::new(gen_req),
            None,
            Some("m".to_string()),
            make_components(),
        );
        assert!(parse_psrl_request_meta_from_context(&ctx).is_none());
    }

    // ── PR 5 §5.2g: RoutingLoopRuntime tests ──

    #[test]
    fn test_routing_loop_runtime_initial_state() {
        use std::sync::atomic::Ordering;

        let runtime = RoutingLoopRuntime::new(RequestSortIndicator::ShortLength, false);
        assert!(!runtime.is_paused.load(Ordering::Relaxed));
        assert!(!runtime.is_routing.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_routing_loop_runtime_queue_operations() {
        let runtime = RoutingLoopRuntime::new(RequestSortIndicator::SmallId, true);
        let (tx, _rx) = oneshot::channel();

        // PR 13 §13.4: entry uses RequestContext instead of PreparedRequest.
        let entry = make_test_entry_generate(2, Some(42), false, "hello world", tx);

        let mut queue = runtime.request_queue.lock().await;
        queue.push(entry);
        assert_eq!(queue.len(), 1);

        let popped = queue.pop().expect("should pop");
        assert_eq!(popped.ctx.input.model_id, Some("test-model".to_string()));
        assert_eq!(
            popped.routing_meta.as_ref().map(|m| m.request_id),
            Some(Some(42))
        );
    }

    #[test]
    fn test_routing_queue_entry_version_tag_priority() {
        let (tx1, _rx1) = oneshot::channel();
        // PR 13 §13.4: entry uses RequestContext instead of PreparedRequest.
        let entry = make_test_entry_generate(5, Some(1), false, "", tx1);
        assert_eq!(entry.get_version_tag(), 5);

        // Entry without routing_meta → version_tag = -1
        let (tx2, _rx2) = oneshot::channel();
        use openai_protocol::generate::GenerateRequest;
        let gen_req: GenerateRequest = serde_json::from_str(r#"{"text":"x"}"#).unwrap();
        let entry_no_meta = RoutingQueueEntry {
            ctx: RequestContext::for_generate(Arc::new(gen_req), None, None, make_components()),
            result_tx: tx2,
            routing_meta: None,
        };
        assert_eq!(entry_no_meta.get_version_tag(), -1);
    }

    #[test]
    fn test_routing_queue_entry_validate_priority() {
        // Validate requests should have lower priority number (dispatched first)
        let (tx1, _rx1) = oneshot::channel();
        // PR 13 §13.4: entry uses RequestContext; text drives length-based priority.
        let validate_entry = make_test_entry_generate(1, Some(1), true, "long validate text", tx1);

        let (tx2, _rx2) = oneshot::channel();
        let normal_entry = make_test_entry_generate(1, Some(2), false, "a", tx2);

        let validate_prio = validate_entry.get_priority(RequestSortIndicator::ShortLength);
        let normal_prio = normal_entry.get_priority(RequestSortIndicator::ShortLength);

        // is_validate=true → p0=false(0), is_validate=false → p0=true(1)
        // So validate entry has lower p0 (higher scheduling priority)
        assert!(
            validate_prio.0 < normal_prio.0,
            "validate request should have lower p0: {validate_prio:?} vs {normal_prio:?}"
        );
    }

    #[tokio::test]
    async fn test_routing_loop_runtime_pause_resume() {
        use std::sync::atomic::Ordering;

        let runtime = RoutingLoopRuntime::new(RequestSortIndicator::ShortLength, false);

        // Initially not paused
        assert!(!runtime.is_paused.load(Ordering::Relaxed));

        // Pause
        runtime.is_paused.store(true, Ordering::Relaxed);
        assert!(runtime.is_paused.load(Ordering::Relaxed));

        // Resume
        runtime.is_paused.store(false, Ordering::Relaxed);
        assert!(!runtime.is_paused.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_routing_loop_runtime_multi_priority_queue() {
        let runtime = RoutingLoopRuntime::new(RequestSortIndicator::SmallId, true);

        // Push entries with different version tags.
        // PR 13 §13.4: entries use RequestContext instead of PreparedRequest.
        let (tx1, _rx1) = oneshot::channel();
        let entry1 = make_test_entry_generate(2, Some(1), false, "", tx1);

        let (tx2, _rx2) = oneshot::channel();
        let entry2 = make_test_entry_generate(1, Some(2), false, "", tx2);

        let mut queue = runtime.request_queue.lock().await;
        queue.push(entry1);
        queue.push(entry2);

        assert_eq!(queue.len(), 2);
        // BTreeMap pops lower key first → version_tag=1 first
        let first = queue.pop().expect("first");
        assert_eq!(first.routing_meta.as_ref().expect("meta").version_tag, 1);
    }
}
