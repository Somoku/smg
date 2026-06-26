//! Request-number balance policy.
//!
//! Selects the healthy worker with the fewest in-flight requests.  When
//! multiple workers share the minimum, the one with the lowest index in the
//! supplied slice is chosen (stable tie-break).
//!
//! This is the simplest load-aware policy: it does not consider token counts
//! or KV-cache state, making it suitable for workloads where request latency
//! does not vary dramatically with input/output length.
//!
//! ## In-flight signal: `worker.load()`
//!
//! The policy routes on [`Worker::load`] — the per-worker load counter
//! maintained by [`WorkerLoadGuard`]: incremented post-commit (in the
//! execution stage, *after* the routing decision passes the pause fence) and
//! decremented when the request departs (completion *or* partial-rollout
//! loopback re-route), on both the HTTP and gRPC transports.
//!
//! This is the analogue of psrl_agent's persistently-maintained
//! `instance_request_counts`: a single counter that is `+1` on admit and `-1`
//! on departure, never reset wholesale by an engine-stats snapshot. Because
//! the guard lifecycle is exact (unlike token counts, which grow during
//! decode), the load counter needs no snapshot reconciliation — it always
//! reflects exactly the requests currently assigned to the instance, including
//! those still in transit to the engine and excluding those that completed
//! before their next snapshot arrived.
//!
//! [`WorkerLoadGuard`]: crate::worker::WorkerLoadGuard
//! [`Worker::load`]: crate::worker::Worker::load

use std::sync::Arc;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Selects the worker with the minimum number of in-flight requests, read from
/// the per-worker load counter ([`Worker::load`]).
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

        // Stable tie-break: `min_by_key` keeps the first (lowest-index) worker
        // when loads are equal.
        healthy.into_iter().min_by_key(|&idx| workers[idx].load())
    }

    fn name(&self) -> &'static str {
        "request_num_balance"
    }

    fn needs_load_guard(&self) -> bool {
        true // routes on worker.load()
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
    use crate::worker::{BasicWorkerBuilder, Worker, WorkerType};

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

    /// Set a worker's in-flight load to `n` by incrementing the load counter.
    fn set_load(worker: &Arc<dyn Worker>, n: usize) {
        for _ in 0..n {
            worker.increment_load();
        }
    }

    #[test]
    fn single_worker_always_selected() {
        let policy = RequestNumBalancePolicy::new();
        let w = build_worker("http://w1:8000");
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
    fn selects_worker_with_lowest_load() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");
        let w2 = build_worker("http://w2:8000");

        set_load(&w0, 5);
        set_load(&w1, 1); // lowest
        set_load(&w2, 3);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1, w2];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn tie_broken_by_lowest_index() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        set_load(&w0, 2);
        set_load(&w1, 2);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn unhealthy_workers_excluded() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        // w0 is less loaded but unhealthy; w1 has more load but is healthy.
        set_load(&w0, 1);
        set_load(&w1, 5);
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
    fn load_decrement_shifts_selection() {
        let policy = RequestNumBalancePolicy::new();
        let w0 = build_worker("http://w0:8000");
        let w1 = build_worker("http://w1:8000");

        set_load(&w0, 2);
        set_load(&w1, 1); // w1 lower → selected

        let workers: Vec<Arc<dyn Worker>> = vec![w0.clone(), w1.clone()];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );

        // A request on w0 completes (guard drop): w0 now ties w1 → lowest index.
        w0.decrement_load();
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn policy_name() {
        assert_eq!(RequestNumBalancePolicy::new().name(), "request_num_balance");
    }
}
