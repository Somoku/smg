//! RAII guard for policy state cleanup on request completion.
//!
//! [`PolicyCompletionGuard`] mirrors the role that [`WorkerLoadGuard`] plays
//! for worker-level load tracking, but at the policy level: it holds the
//! token delta that was optimistically recorded in the policy's local state
//! when a request was routed, and subtracts it back when the request
//! completes (or when the response body is fully consumed / dropped).
//!
//! ## Lifecycle
//!
//! 1. `route_typed_request_once` selects a worker via the policy.
//! 2. The policy's `select_worker` atomically records `+req_tokens` in its
//!    local state (optimistic update).
//! 3. A `PolicyCompletionGuard` is created holding
//!    `(policy, worker_url, req_tokens)`.
//! 4. For streaming responses the guard is attached to the response body via
//!    [`AttachedBody::wrap_response`] so it lives until the body is
//!    consumed or the client disconnects.
//! 5. When the guard is dropped it calls
//!    [`LoadBalancingPolicy::on_request_complete_with_tokens`], which
//!    decrements the shadow state for the worker.
//!
//! [`WorkerLoadGuard`]: crate::worker::WorkerLoadGuard
//! [`AttachedBody::wrap_response`]: crate::worker::AttachedBody::wrap_response

use std::sync::Arc;

use super::LoadBalancingPolicy;

/// RAII guard that decrements the policy's per-worker optimistic token-delta
/// when dropped (i.e. when the request completes or the connection closes).
pub struct PolicyCompletionGuard {
    policy: Arc<dyn LoadBalancingPolicy>,
    worker_url: String,
    /// The token delta that was added to the shadow state during routing.
    /// `None` means the policy did not record a delta (e.g., stateless policy).
    token_delta: Option<i64>,
    success: bool,
}

impl PolicyCompletionGuard {
    /// Create a new guard.
    ///
    /// - `policy`: the policy that recorded the optimistic delta.
    /// - `worker_url`: URL of the worker the request was routed to.
    /// - `token_delta`: number of tokens that were added to the shadow state;
    ///   `None` for stateless policies.
    /// - `success`: whether the request was considered successful.  Passed
    ///   through to [`LoadBalancingPolicy::on_request_complete_with_tokens`].
    pub fn new(
        policy: Arc<dyn LoadBalancingPolicy>,
        worker_url: impl Into<String>,
        token_delta: Option<i64>,
        success: bool,
    ) -> Self {
        Self {
            policy,
            worker_url: worker_url.into(),
            token_delta,
            success,
        }
    }

    /// Mark the request as successful or failed before the guard is dropped.
    ///
    /// Useful when the outcome is only known after building the guard (e.g.,
    /// after reading the upstream response status).
    pub fn set_success(&mut self, success: bool) {
        self.success = success;
    }
}

impl Drop for PolicyCompletionGuard {
    fn drop(&mut self) {
        self.policy.on_request_complete_with_tokens(
            &self.worker_url,
            self.token_delta,
            self.success,
        );
    }
}

impl std::fmt::Debug for PolicyCompletionGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyCompletionGuard")
            .field("worker_url", &self.worker_url)
            .field("token_delta", &self.token_delta)
            .field("success", &self.success)
            .finish()
    }
}
