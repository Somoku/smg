// PR 5 §5.2: Routing loop utilities for PSRL metadata parsing.
//!
//! Ports `RoutingMeta`, `PartialRolloutState`, `parse_psrl_request_meta()`, and
//! all helper parsing functions from sgl-model-gateway's `routing_loop_utils.rs`.
//!
//! PR 5 §5.2g: `RoutingLoopRuntime` and `RoutingQueueEntry` define the shared
//! runtime state and queue entry types for the PSRL routing loop. Additional
//! fields (channel, tracking maps, PS Manager client) will be added in PR 10.

use std::sync::atomic::AtomicBool;

use axum::{http::HeaderMap, response::Response};
use tokio::sync::oneshot;

use crate::{
    config::types::RequestSortIndicator,
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

// PR 5 §5.2a: Partial-rollout state accumulated across loopback iterations.
/// Partial-rollout state accumulated across loopback iterations of a single request.
#[derive(Debug, Clone)]
pub struct PartialRolloutState {
    pub token_ids: Vec<u32>,
    pub logprobs: serde_json::Value,
}

impl PartialRolloutState {
    #[inline]
    pub fn new() -> Self {
        Self {
            token_ids: Vec::new(),
            logprobs: serde_json::Value::Null,
        }
    }
}

impl Default for PartialRolloutState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Routing loop runtime types ──────────────────────────────────────────

// PR 5 §5.2g: Shared runtime state container for the PSRL routing loop.
/// Runtime context shared by the HTTP routing loop, admin endpoints, and
/// worker-selection logic.
///
/// Core fields (`is_paused`, `is_routing`, `request_queue`) are defined here.
/// Additional fields — mpsc channel, request-tracking maps, PS Manager client —
/// will be added when the routing loop itself is implemented (PR 10 in
/// `missing.md`).
pub struct RoutingLoopRuntime {
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
    pub request_queue: tokio::sync::Mutex<MultiPriorityRequestQueue<RoutingQueueEntry>>,
}

impl RoutingLoopRuntime {
    /// Create a new `RoutingLoopRuntime` with the given sort indicator and
    /// multi-priority queue configuration.
    pub fn new(
        sort_indicator: RequestSortIndicator,
        enable_multi_priority_queue: bool,
    ) -> Self {
        Self {
            is_paused: AtomicBool::new(false),
            is_routing: AtomicBool::new(false),
            request_queue: tokio::sync::Mutex::new(MultiPriorityRequestQueue::new(
                sort_indicator,
                enable_multi_priority_queue,
                None,
            )),
        }
    }
}

// PR 5 §5.2g: Queue entry for the PSRL routing loop.
/// Entry in the HTTP routing loop queue.
///
/// Each entry represents a single request awaiting worker selection and dispatch.
/// The `result_tx` oneshot is used to send the HTTP response back to the
/// originating handler once the request is dispatched and completed.
pub struct RoutingQueueEntry {
    /// Optional HTTP headers forwarded from the original request.
    pub headers: Option<HeaderMap>,
    /// JSON body of the request (chat completion, generate, etc.).
    pub body: serde_json::Value,
    /// Static route identifier (e.g., `"/v1/chat/completions"`).
    pub route: &'static str,
    /// Model ID extracted from the request body, if present.
    pub model_id: Option<String>,
    /// Whether the request is a streaming request.
    pub is_stream: bool,
    /// Raw text content used for length-based priority sorting.
    pub text: String,
    /// Oneshot sender for delivering the response back to the request handler.
    pub result_tx: oneshot::Sender<Response>,
    /// Accumulated partial-rollout state from previous loopback iterations.
    pub partial_response: Option<PartialRolloutState>,
    /// Optional routing metadata for PSRL routing decisions.
    pub routing_meta: Option<RoutingMeta>,
}

// PR 5 §5.2g: RequestPriority implementation for RoutingQueueEntry.
impl RequestPriority for RoutingQueueEntry {
    #[inline]
    fn get_version_tag(&self) -> i64 {
        self.routing_meta.as_ref().map_or(-1, |m| m.version_tag)
    }

    #[inline]
    fn get_priority(&self, indicator: RequestSortIndicator) -> (i64, i64, i64) {
        match indicator {
            RequestSortIndicator::ShortLength => {
                get_priority_by_version_and_token_num(self, true)
            }
            RequestSortIndicator::LongLength => {
                get_priority_by_version_and_token_num(self, false)
            }
            RequestSortIndicator::SmallId => get_priority_by_version_and_id(self),
        }
    }
}

/// Compute version-based priority: unversioned (`-1`) → lowest priority (`i64::MAX`).
fn get_priority_by_version(request: &RoutingQueueEntry) -> i64 {
    let version_tag = request
        .routing_meta
        .as_ref()
        .map_or(-1, |m| m.version_tag);
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
    let validate_priority = request
        .routing_meta
        .as_ref()
        .is_none_or(|m| !m.is_validate);
    let version_priority = get_priority_by_version(request);
    if short_request_first {
        let length_priority = request.text.len().min(i32::MAX as usize) as i32;
        (
            validate_priority as i64,
            version_priority,
            length_priority as i64,
        )
    } else {
        let length_priority = -(request.text.len().min(i32::MAX as usize) as i32);
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
    let validate_priority = request
        .routing_meta
        .as_ref()
        .is_none_or(|m| !m.is_validate);
    let version_priority = get_priority_by_version(request);
    let id_priority = request
        .routing_meta
        .as_ref()
        .and_then(|m| m.request_id)
        .unwrap_or(i64::MAX);
    (validate_priority as i64, version_priority, id_priority)
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
    headers: Option<&HeaderMap>,
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
    headers: Option<&HeaderMap>,
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
    use super::*;
    use serde_json::json;

    fn headers_with(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (key, value) in pairs {
            map.insert(
                http::header::HeaderName::from_bytes(key.as_bytes())
                    .expect("valid header name"),
                http::header::HeaderValue::from_str(value).expect("valid header value"),
            );
        }
        map
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
        let meta = parse_psrl_request_meta(Some(&headers), Some(&body))
            .expect("should parse");
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
        let meta = parse_psrl_request_meta(Some(&headers), None)
            .expect("should parse");
        assert_eq!(meta.version_tag, -1);
    }

    // PR 5 §5.4 test: Missing is_validate defaults to false
    #[test]
    fn test_parse_meta_is_validate_default() {
        let headers = headers_with(&[("x-request-id", "1")]);
        let meta = parse_psrl_request_meta(Some(&headers), None)
            .expect("should parse");
        assert!(!meta.is_validate);
    }

    // PR 5 §5.4 test: Accepts "1", "true", "t", "yes", "y" (case-insensitive)
    #[test]
    fn test_parse_meta_is_validate_truthy() {
        for truthy in &["1", "true", "True", "TRUE", "t", "T", "yes", "Yes", "y", "Y"] {
            let headers = headers_with(&[
                ("x-request-id", "1"),
                ("x-is-validate", truthy),
            ]);
            let meta = parse_psrl_request_meta(Some(&headers), None)
                .expect("should parse");
            assert!(meta.is_validate, "Expected truthy for {truthy:?}");
        }
    }

    // PR 5 §5.4 test: Accepts "0", "false", "f", "no", "n" (case-insensitive)
    #[test]
    fn test_parse_meta_is_validate_falsy() {
        for falsy in &["0", "false", "False", "FALSE", "f", "F", "no", "No", "n", "N"] {
            let headers = headers_with(&[
                ("x-request-id", "1"),
                ("x-is-validate", falsy),
            ]);
            let meta = parse_psrl_request_meta(Some(&headers), None)
                .expect("should parse");
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
        let meta = parse_psrl_request_meta(Some(&headers), None)
            .expect("should parse");
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
        let meta = parse_psrl_request_meta(None, Some(&body))
            .expect("should parse");
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
        let meta = parse_psrl_request_meta(None, Some(&body))
            .expect("should parse");
        assert_eq!(
            meta.rollout_instance_hint,
            Some(("w-001".to_string(), 4))
        );
    }

    // PR 5 §5.4 test: Flat body keys for rollout hint
    #[test]
    fn test_parse_rollout_hint_body_flat() {
        let body = json!({
            "request_id": 1,
            "base_worker_id": "flat-worker",
            "target_dp_rank": 0
        });
        let meta = parse_psrl_request_meta(None, Some(&body))
            .expect("should parse");
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
        let meta = parse_psrl_request_meta(None, Some(&body))
            .expect("should parse");
        assert_eq!(
            meta.rollout_instance_hint,
            Some(("replica-7".to_string(), 1))
        );
    }

    // PR 5 §5.4 test: JSON integer parsed as i64
    #[test]
    fn test_parse_i64_from_body_int() {
        let body = json!({"key": 42});
        assert_eq!(parse_i64_from_body(Some(&body), "key"), Some(42));
    }

    // PR 5 §5.4 test: JSON string "42" parsed as i64
    #[test]
    fn test_parse_i64_from_body_string() {
        let body = json!({"key": "42"});
        assert_eq!(parse_i64_from_body(Some(&body), "key"), Some(42));
    }

    // PR 5 §5.4 test: Missing key → None
    #[test]
    fn test_parse_i64_from_body_missing() {
        let body = json!({"other": 1});
        assert_eq!(parse_i64_from_body(Some(&body), "key"), None);
        assert_eq!(parse_i64_from_body(None, "key"), None);
    }

    // PR 5 §5.4 test: PartialRolloutState default construction
    #[test]
    fn test_partial_rollout_state_new() {
        let state = PartialRolloutState::new();
        assert!(state.token_ids.is_empty());
        assert_eq!(state.logprobs, serde_json::Value::Null);

        // Default trait impl should produce the same
        let default_state = PartialRolloutState::default();
        assert!(default_state.token_ids.is_empty());
        assert_eq!(default_state.logprobs, serde_json::Value::Null);
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

        let entry = RoutingQueueEntry {
            headers: None,
            body: json!({"model": "test-model"}),
            route: "/v1/chat/completions",
            model_id: Some("test-model".to_string()),
            is_stream: false,
            text: "hello world".to_string(),
            result_tx: tx,
            partial_response: None,
            routing_meta: Some(RoutingMeta {
                request_id: Some(42),
                prompt_id: Some(1),
                version_tag: 2,
                is_validate: false,
                rollout_instance_hint: None,
            }),
        };

        let mut queue = runtime.request_queue.lock().await;
        queue.push(entry);
        assert_eq!(queue.len(), 1);

        let popped = queue.pop().expect("should pop");
        assert_eq!(popped.model_id, Some("test-model".to_string()));
        assert_eq!(popped.route, "/v1/chat/completions");
        assert_eq!(
            popped.routing_meta.as_ref().map(|m| m.request_id),
            Some(Some(42))
        );
    }

    #[test]
    fn test_routing_queue_entry_version_tag_priority() {
        let (tx1, _rx1) = oneshot::channel();
        let entry = RoutingQueueEntry {
            headers: None,
            body: json!({}),
            route: "/v1/generate",
            model_id: None,
            is_stream: false,
            text: String::new(),
            result_tx: tx1,
            partial_response: None,
            routing_meta: Some(RoutingMeta {
                request_id: Some(1),
                prompt_id: None,
                version_tag: 5,
                is_validate: false,
                rollout_instance_hint: None,
            }),
        };
        assert_eq!(entry.get_version_tag(), 5);

        // Entry without routing_meta → version_tag = -1
        let (tx2, _rx2) = oneshot::channel();
        let entry_no_meta = RoutingQueueEntry {
            headers: None,
            body: json!({}),
            route: "/v1/generate",
            model_id: None,
            is_stream: false,
            text: String::new(),
            result_tx: tx2,
            partial_response: None,
            routing_meta: None,
        };
        assert_eq!(entry_no_meta.get_version_tag(), -1);
    }

    #[test]
    fn test_routing_queue_entry_validate_priority() {
        // Validate requests should have lower priority number (dispatched first)
        let (tx1, _rx1) = oneshot::channel();
        let validate_entry = RoutingQueueEntry {
            headers: None,
            body: json!({}),
            route: "/v1/generate",
            model_id: None,
            is_stream: false,
            text: "long validate text".to_string(),
            result_tx: tx1,
            partial_response: None,
            routing_meta: Some(RoutingMeta {
                request_id: Some(1),
                prompt_id: None,
                version_tag: 1,
                is_validate: true,
                rollout_instance_hint: None,
            }),
        };

        let (tx2, _rx2) = oneshot::channel();
        let normal_entry = RoutingQueueEntry {
            headers: None,
            body: json!({}),
            route: "/v1/generate",
            model_id: None,
            is_stream: false,
            text: "a".to_string(),
            result_tx: tx2,
            partial_response: None,
            routing_meta: Some(RoutingMeta {
                request_id: Some(2),
                prompt_id: None,
                version_tag: 1,
                is_validate: false,
                rollout_instance_hint: None,
            }),
        };

        let validate_prio =
            validate_entry.get_priority(RequestSortIndicator::ShortLength);
        let normal_prio =
            normal_entry.get_priority(RequestSortIndicator::ShortLength);

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

        // Push entries with different version tags
        let (tx1, _rx1) = oneshot::channel();
        let entry1 = RoutingQueueEntry {
            headers: None,
            body: json!({}),
            route: "/v1/generate",
            model_id: None,
            is_stream: false,
            text: String::new(),
            result_tx: tx1,
            partial_response: None,
            routing_meta: Some(RoutingMeta {
                request_id: Some(1),
                prompt_id: None,
                version_tag: 2,
                is_validate: false,
                rollout_instance_hint: None,
            }),
        };

        let (tx2, _rx2) = oneshot::channel();
        let entry2 = RoutingQueueEntry {
            headers: None,
            body: json!({}),
            route: "/v1/generate",
            model_id: None,
            is_stream: false,
            text: String::new(),
            result_tx: tx2,
            partial_response: None,
            routing_meta: Some(RoutingMeta {
                request_id: Some(2),
                prompt_id: None,
                version_tag: 1,
                is_validate: false,
                rollout_instance_hint: None,
            }),
        };

        let mut queue = runtime.request_queue.lock().await;
        queue.push(entry1);
        queue.push(entry2);

        assert_eq!(queue.len(), 2);
        // BTreeMap pops lower key first → version_tag=1 first
        let first = queue.pop().expect("first");
        assert_eq!(first.routing_meta.as_ref().expect("meta").version_tag, 1);
    }
}
