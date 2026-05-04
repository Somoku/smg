//! Request-number balance policy.
//!
//! Selects the healthy worker with the fewest in-flight requests
//! (running + waiting).  When multiple workers share the minimum, the one
//! with the lowest index in the supplied slice is chosen (stable tie-break).
//!
//! This is the simplest load-aware policy: it does not consider token counts
//! or KV-cache state, making it suitable for workloads where request latency
//! does not vary dramatically with input/output length.

use std::sync::Arc;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Selects the worker with the minimum number of running + waiting requests.
#[derive(Debug, Default)]
pub struct RequestNumBalancePolicy;

impl RequestNumBalancePolicy {
    pub fn new() -> Self {
        Self
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

        healthy
            .into_iter()
            .min_by_key(|&idx| workers[idx].engine_stats().waiting_and_running_queue_size())
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
}
