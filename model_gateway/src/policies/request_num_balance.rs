//! Request-number balance policy.
//!
//! Selects the healthy worker with the fewest in-flight requests
//! (running + waiting).  When multiple workers share the minimum, the one
//! with the lowest index in the supplied slice is chosen (stable tie-break).
//!
//! This is the simplest load-aware policy: it does not consider token counts
//! or KV-cache state, making it suitable for workloads where request latency
//! does not vary dramatically with input/output length.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Selects the worker with the minimum number of running + waiting requests,
/// including an optimistic local delta for requests routed since the last
/// engine-stats snapshot.
#[derive(Debug)]
pub struct RequestNumBalancePolicy {
    /// Per-worker optimistic request-count delta, keyed by worker URL.
    local_delta: Mutex<HashMap<String, i64>>,
}

impl RequestNumBalancePolicy {
    pub fn new() -> Self {
        Self {
            local_delta: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for RequestNumBalancePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for RequestNumBalancePolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }

        // Hold the lock for both read and increment so that two concurrent
        // callers cannot observe the same delta and pick the same worker.
        // The critical section is a HashMap lookup + i64 increment — sub-μs.
        let mut guard = self.local_delta.lock().unwrap_or_else(|e| e.into_inner());

        let idx = healthy.into_iter().min_by_key(|&idx| {
            let engine_queue = workers[idx].engine_stats().waiting_and_running_queue_size() as i64;
            let pending = guard.get(workers[idx].url()).copied().unwrap_or(0).max(0);
            engine_queue + pending
        })?;

        // Optimistic increment — the request hasn't reached the engine yet.
        *guard.entry(workers[idx].url().to_string()).or_insert(0) += 1;

        Some(idx)
    }

    fn on_request_complete_with_tokens(
        &self,
        worker_url: &str,
        _token_delta: Option<i64>,
        _success: bool,
    ) {
        let mut guard = self.local_delta.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(count) = guard.get_mut(worker_url) {
            *count = (*count - 1).max(0);
        }
    }

    fn on_engine_stats_updated(&self, worker_url: &str) {
        let mut guard = self.local_delta.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(count) = guard.get_mut(worker_url) {
            *count = 0;
        }
    }

    fn name(&self) -> &'static str {
        "request_num_balance"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{
        BasicWorkerBuilder, EngineSchedulerStats, EngineStats, EngineStatsUpdateOutcome, WorkerType,
    };

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn build_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn set_queue(worker: &Arc<dyn Worker>, running: usize, waiting: usize) {
        // Far-future timestamp so monotonicity always passes across successive calls.
        let timestamp =
            chrono::TimeZone::with_ymd_and_hms(&chrono::Utc, 2099, 1, 1, 0, 0, 0).unwrap();
        let stats = EngineStats {
            timestamp,
            scheduler_stats: EngineSchedulerStats {
                num_running_reqs: running,
                num_waiting_reqs: waiting,
                ..Default::default()
            },
        };
        let outcome = worker.update_engine_stats(stats, 0);
        assert!(
            matches!(outcome, EngineStatsUpdateOutcome::Applied),
            "set_queue failed: {outcome:?}"
        );
    }

    #[test]
    fn single_worker_always_selected() {
        let policy = RequestNumBalancePolicy::new();
        let w = build_worker("http://w1:8000");
        set_queue(&w, 0, 0);
        let selected = policy.select_worker(&[w], &SelectWorkerInfo::default());
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn empty_workers_returns_none() {
        let policy = RequestNumBalancePolicy::new();
        assert_eq!(
            policy.select_worker(&[], &SelectWorkerInfo::default()),
            None
        );
    }

    #[test]
    fn selects_worker_with_lowest_queue_depth() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");
        let w2 = build_worker("http://w2:8000");

        set_queue(&w0, 5, 0);
        set_queue(&w1, 1, 0); // lowest
        set_queue(&w2, 3, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1, w2];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn tie_broken_by_lowest_index() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        // Both have the same queue depth: first one (index 0) must win.
        set_queue(&w0, 2, 0);
        set_queue(&w1, 2, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn unhealthy_workers_excluded() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        // w0 is heavily loaded but unhealthy; w1 has more requests but is healthy.
        set_queue(&w0, 1, 0);
        set_queue(&w1, 5, 0);
        w0.set_status(WorkerStatus::NotReady);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn all_unhealthy_returns_none() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        w0.set_status(WorkerStatus::NotReady);
        w1.set_status(WorkerStatus::NotReady);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            None
        );
    }

    #[test]
    fn waiting_requests_counted_in_total() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        // w0: 2 running + 3 waiting = 5 total
        // w1: 0 running + 2 waiting = 2 total → should be selected
        set_queue(&w0, 2, 3);
        set_queue(&w1, 0, 2);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn policy_name() {
        assert_eq!(RequestNumBalancePolicy::new().name(), "request_num_balance");
    }

    #[test]
    fn uses_engine_stats_not_load_counter() {
        // Regression: policy must use engine_stats queue depth, not the load counter.
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        // Bump w1's load counter high (simulates in-flight requests without engine stats).
        for _ in 0..10 {
            w1.increment_load();
        }
        // Engine stats: w0 has a bigger queue, w1 is empty (engine stats win).
        set_queue(&w0, 5, 0);
        set_queue(&w1, 0, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(1));
    }

    // -- Optimistic local delta tests ------------------------------------------

    #[test]
    fn optimistic_counter_distributes_across_idle_workers() {
        // Three idle workers with zero engine stats.  Without the optimistic
        // counter all three calls would return index 0 (the min_by_key tie-break).
        // With the counter, each successive call picks the next worker.
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");
        let w2 = build_worker("http://w2:8000");

        set_queue(&w0, 0, 0);
        set_queue(&w1, 0, 0);
        set_queue(&w2, 0, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1, w2];
        let info = SelectWorkerInfo::default();

        let first = policy.select_worker(&workers, &info);
        let second = policy.select_worker(&workers, &info);
        let third = policy.select_worker(&workers, &info);

        assert_eq!(first, Some(0));
        assert_eq!(second, Some(1));
        assert_eq!(third, Some(2));
    }

    #[test]
    fn optimistic_counter_wraps_around() {
        // After distributing one request to each of 2 workers, the 3rd request
        // should go back to the first worker (both have delta = 1, tie-break
        // picks lowest index).
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        set_queue(&w0, 0, 0);
        set_queue(&w1, 0, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let info = SelectWorkerInfo::default();

        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        // Both at delta 1 → tie broken by lowest index.
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }

    #[test]
    fn engine_stats_reset_clears_delta() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        set_queue(&w0, 0, 0);
        set_queue(&w1, 0, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0.clone(), w1];
        let info = SelectWorkerInfo::default();

        // Route to w0, accumulating delta = 1 for w0.
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        // Next request goes to w1 (w0 has delta 1).
        assert_eq!(policy.select_worker(&workers, &info), Some(1));

        // Simulate a fresh engine-stats snapshot for w0 — reset delta to 0.
        policy.on_engine_stats_updated(w0.url());

        // Now w0 (engine 0 + delta 0 = 0) beats w1 (engine 0 + delta 1 = 1).
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }

    #[test]
    fn request_complete_decrements_delta() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        set_queue(&w0, 0, 0);
        set_queue(&w1, 0, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0.clone(), w1];
        let info = SelectWorkerInfo::default();

        // Route two requests to w0 (delta 1), then w1 (delta 1).
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));

        // Complete the request on w0 — delta drops from 1 to 0.
        policy.on_request_complete_with_tokens(w0.url(), None, true);

        // w0 (engine 0 + delta 0 = 0) beats w1 (engine 0 + delta 1 = 1).
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }

    #[test]
    fn delta_does_not_go_negative() {
        let policy = RequestNumBalancePolicy::new();
        let w = build_worker("http://w0:8000");
        set_queue(&w, 0, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w.clone()];
        let info = SelectWorkerInfo::default();

        // Route one request → delta = 1.
        assert_eq!(policy.select_worker(&workers, &info), Some(0));

        // Two completions should clamp at 0, not go to -1.
        policy.on_request_complete_with_tokens(w.url(), None, true);
        policy.on_request_complete_with_tokens(w.url(), None, true);

        // Delta is clamped at 0, so another route still picks idx 0.
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }

    #[test]
    fn optimistic_delta_combined_with_engine_stats() {
        // w0 has engine queue 3 + delta 0 = 3
        // w1 has engine queue 0 + delta 2 = 2  (routed twice before stats push)
        // → w1 should be picked (2 < 3)
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        set_queue(&w0, 3, 0);
        set_queue(&w1, 0, 0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let info = SelectWorkerInfo::default();

        // Two requests go to w1 (it's idle) → delta = 2 for w1.
        assert_eq!(policy.select_worker(&workers, &info), Some(1)); // w1: 0+0 < 3+0
        assert_eq!(policy.select_worker(&workers, &info), Some(1)); // w1: 0+1 < 3+0

        // Now w0=3+0=3, w1=0+2=2 → w1 still wins.
        assert_eq!(policy.select_worker(&workers, &info), Some(1)); // w1: 0+2 < 3+0

        // w0=3+0=3, w1=0+3=3 → tie broken by lowest index → w0.
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }
}
