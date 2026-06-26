//! Throughput-optimal routing policies.
//!
//! Both [`ThroughputOptimalPolicy`] and [`ThroughputOptimalWithBudgetPolicy`]
//! are built on a shared [`ThroughputRuntime`] that:
//!
//! 1. Parses TP/PP labels from each worker's metadata.
//! 2. Looks up per-TP/PP latency coefficients from the [`CostModel`].
//! 3. Estimates the marginal throughput gain of routing the next request to
//!    each worker and selects the worker with the largest gain.
//!
//! ## In-flight accounting
//!
//! Request and running counts come from [`Worker::load`] — the per-worker load
//! counter maintained by [`WorkerLoadGuard`] (post-commit increment, drop-time
//! decrement, partial-rollout-loopback aware, on both transports). This is the
//! exact analogue of psrl_agent's persistent `instance_to_request_num`: a
//! single counter that needs no snapshot reconciliation because the guard
//! lifecycle is exact.
//!
//! Token load is different: a request's KV footprint *grows* during decode, so
//! only the engine snapshot knows the true current token count. The runtime
//! therefore uses the engine snapshot as the token base and adds
//! [`Worker::inflight_tokens_sum`] — the sum of admit-time token estimates for
//! requests not yet reflected in a snapshot. Those estimates are cleared
//! (rebased) by the worker-stats update path once a snapshot's request count
//! agrees with the load counter, exactly mirroring psrl_agent's `is_staled`
//! count-agreement gate. There is no unconditional per-snapshot reset.
//!
//! ## Budget variant
//!
//! [`ThroughputOptimalWithBudgetPolicy`] accounts for KV-cache page
//! granularity by rounding response tokens up to the nearest multiple of
//! `request_budget` when computing the current token load of each worker
//! (via [`EngineStats::token_num_with_budget`]).
//!
//! [`Worker::load`]: crate::worker::Worker::load
//! [`Worker::inflight_tokens_sum`]: crate::worker::Worker::inflight_tokens_sum
//! [`WorkerLoadGuard`]: crate::worker::WorkerLoadGuard

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use tracing::{error, warn};

use super::{
    cost_model_utils::{CostModel, CostModelEntry},
    get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::worker::Worker;

// ---------------------------------------------------------------------------
// One-shot log guards (avoid flooding logs on every routing decision)
// ---------------------------------------------------------------------------

// One-shot guard for missing-tokens warning
static REPORTED_MISSING_TOKENS: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration shared by both throughput-optimal policies.
#[derive(Debug, Clone)]
pub struct ThroughputOptimalConfig {
    /// Path to a JSON cost-model file (keyed by `"TP{n}_PP{m}"`).
    /// Required — the policy cannot operate without a valid cost model.
    pub cost_model_path: String,
    /// Workers with this many or more queued requests are skipped.
    pub max_concurrent_seqs_per_instance: usize,
    /// Fraction of the baseline throughput that a marginal gain must exceed
    /// before the worker is accepted.  Raising this value makes the policy more
    /// conservative (prefers idle workers).
    pub delta_throughput_threshold: f64,
    /// Maximum context length used to decide whether a new request fits.
    /// Overridden per-worker by the `max_model_len` / `max_total_tokens` label.
    pub max_prompt_length: usize,
    /// KV-cache page size in tokens.  Used by [`ThroughputOptimalWithBudgetPolicy`]
    /// to round response tokens up to the nearest page boundary.
    pub request_budget: usize,
    /// Workers are assumed to have at most this many waiting requests after
    /// preemption; used in the `max_model_len` fallback calculation.
    pub max_num_waiting_reqs_after_preemption: usize,
}

// ---------------------------------------------------------------------------
// Shared runtime
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ThroughputRuntime {
    cfg: ThroughputOptimalConfig,
    cost_model: CostModel,
}

impl ThroughputRuntime {
    fn new(cfg: ThroughputOptimalConfig) -> Result<Self, String> {
        let cost_model = CostModel::load_from_file(&cfg.cost_model_path).map_err(|e| {
            format!(
                "throughput_optimal policy: failed to load cost model from '{}': {}",
                cfg.cost_model_path, e
            )
        })?;
        Ok(Self { cfg, cost_model })
    }

    // ── Label helpers ────────────────────────────────────────────────────

    #[inline]
    fn label_i64(worker: &Arc<dyn Worker>, key: &str) -> Option<i64> {
        worker
            .metadata()
            .spec
            .labels
            .get(key)
            .and_then(|v| v.parse::<i64>().ok())
    }

    #[inline]
    fn max_model_len(&self, worker: &Arc<dyn Worker>) -> i64 {
        Self::label_i64(worker, "max_model_len")
            .or_else(|| Self::label_i64(worker, "max_total_tokens"))
            .unwrap_or_else(|| {
                (self.cfg.max_prompt_length
                    + self.cfg.request_budget * self.cfg.max_num_waiting_reqs_after_preemption)
                    as i64
            })
    }

    #[inline]
    fn tp_pp_key(worker: &Arc<dyn Worker>) -> String {
        let tp = Self::label_i64(worker, "tp_size").unwrap_or(1);
        let pp = Self::label_i64(worker, "pp_size").unwrap_or(1);
        format!("TP{tp}_PP{pp}")
    }

    // ── Request token estimation ─────────────────────────────────────────

    /// Returns the number of tokens in the incoming request.
    ///
    /// When `info.tokens` is `None` (e.g., plain HTTP routing without tokenisation),
    /// falls back to `1` and emits a one-shot warning.  Callers that can supply
    /// token counts should populate [`SelectWorkerInfo::tokens`] for accurate routing.
    #[inline]
    fn request_token_num(info: &SelectWorkerInfo<'_>) -> i64 {
        if let Some(tokens) = info.tokens {
            tokens.len().max(1) as i64
        } else {
            if !REPORTED_MISSING_TOKENS.swap(true, Ordering::Relaxed) {
                warn!(
                    policy = "throughput_optimal",
                    "SelectWorkerInfo.tokens is not set; defaulting to 1 token for routing. \
                     Populate tokens in SelectWorkerInfo for accurate throughput-optimal routing."
                );
            }
            1
        }
    }

    #[inline]
    fn request_token_num_with_budget(&self, info: &SelectWorkerInfo<'_>) -> i64 {
        let prompt_tokens = Self::request_token_num(info);
        let response_tokens = info.response_token_count.unwrap_or(0) as i64;
        let budget = self.cfg.request_budget.max(1) as i64;
        // Round (response_tokens + 1) up to the nearest budget multiple.
        let budget_aligned_response = ((response_tokens + 1 + budget - 1) / budget) * budget;
        prompt_tokens + budget_aligned_response
    }

    // ── Per-worker state accessors ───────────────────────────────────────

    #[inline]
    fn current_waiting(worker: &Arc<dyn Worker>) -> i64 {
        worker.engine_stats().waiting_queue_size() as i64
    }

    /// In-flight request count: the persistent load counter, which already
    /// includes requests admitted since the last snapshot and excludes
    /// completed/looped-back ones (maintained by `WorkerLoadGuard`).
    #[inline]
    fn current_queue(worker: &Arc<dyn Worker>) -> i64 {
        worker.load() as i64
    }

    /// Running-request count used by the cost model. We use the load counter as
    /// the running estimate: it is the exact count of requests currently
    /// assigned to the instance, which is what the cost model's concurrency
    /// term needs. (Unlike token load, request count does not require an engine
    /// base because the guard lifecycle is exact.)
    #[inline]
    fn current_running(worker: &Arc<dyn Worker>) -> i64 {
        worker.load() as i64
    }

    #[inline]
    fn current_token_num(&self, worker: &Arc<dyn Worker>, use_budget: bool) -> i64 {
        let stats = worker.engine_stats();
        let has_stats = stats.waiting_and_running_queue_size() > 0 || stats.total_token_num() > 0;

        let engine_tokens = if has_stats {
            let token_num = if use_budget {
                stats.token_num_with_budget(self.cfg.request_budget)
            } else {
                stats.total_token_num()
            };

            if token_num > 0 || stats.waiting_and_running_queue_size() == 0 {
                token_num as i64
            } else {
                // KV-cache estimate when detailed token maps are not populated.
                let estimated = (stats.scheduler_stats.kv_cache_usage
                    * self.max_model_len(worker) as f64)
                    .ceil() as i64;
                if estimated > 0 {
                    estimated
                } else {
                    0
                }
            }
        } else {
            0
        };

        // Add inflight token estimates for requests admitted since the last
        // snapshot. The worker-stats update path rebases (clears) these once a
        // snapshot's request count agrees with the load counter, so this never
        // double-counts a request the engine base already reflects.
        let inflight_tokens = worker.inflight_tokens_sum() as i64;

        engine_tokens + inflight_tokens
    }

    #[inline]
    fn can_run_directly(
        &self,
        worker: &Arc<dyn Worker>,
        request_token_num: i64,
        current_token_num: i64,
    ) -> bool {
        if Self::current_waiting(worker) > 0 {
            return false;
        }
        current_token_num + request_token_num <= self.max_model_len(worker)
    }

    // ── Throughput estimation ────────────────────────────────────────────

    #[inline]
    fn cost_model_entry<'a>(&'a self, worker: &Arc<dyn Worker>) -> Option<&'a CostModelEntry> {
        let key = Self::tp_pp_key(worker);
        if let Some(entry) = self.cost_model.get(&key) {
            return Some(entry);
        }
        error!(
            policy = "throughput_optimal",
            tp_pp_key = %key,
            worker = %worker.url(),
            "cost model has no entry for TP/PP key — skipping worker"
        );
        None
    }

    #[inline]
    fn estimate_throughput(
        &self,
        worker: &Arc<dyn Worker>,
        request_num: i64,
        token_num: i64,
    ) -> Option<f64> {
        if request_num <= 0 {
            return Some(0.0);
        }
        self.cost_model_entry(worker)
            .map(|e| e.estimate_throughput(request_num, token_num))
    }

    // ── Worker selection ─────────────────────────────────────────────────

    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        use_budget: bool,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }

        let req_tokens = if use_budget {
            self.request_token_num_with_budget(info)
        } else {
            Self::request_token_num(info)
        };

        // Partition healthy workers into priority groups.
        //
        // When `info.priority_groups` is set, workers are grouped by their
        // assigned priority value (lower value = higher priority) and groups are
        // tried in ascending order.  The policy returns the best worker from the
        // first group that contains at least one eligible candidate.  If no
        // worker in the highest-priority group is eligible, it falls through to
        // the next group.
        //
        // When `info.priority_groups` is `None` all healthy workers are treated
        // as a single group, preserving the existing behaviour exactly.
        let groups: Vec<Vec<usize>> = if let Some(priority_groups) = info.priority_groups {
            let mut priority_map: std::collections::BTreeMap<i64, Vec<usize>> =
                std::collections::BTreeMap::new();
            for &idx in &healthy {
                let priority = priority_groups.get(idx).copied().unwrap_or(0);
                priority_map.entry(priority).or_default().push(idx);
            }
            priority_map.into_values().collect()
        } else {
            vec![healthy] // single group — degrades to the original behaviour
        };

        // Iterate groups in priority order.  Each group computes its own
        // baseline so that the threshold is relative to the workers in that
        // group rather than the entire pool (per-group baseline).
        for group in &groups {
            let baseline = group
                .iter()
                .find_map(|&idx| self.estimate_throughput(&workers[idx], 1, req_tokens))
                .unwrap_or(0.0);
            let threshold = baseline * self.cfg.delta_throughput_threshold;

            let mut best_idx: Option<usize> = None;
            let mut best_delta = f64::NEG_INFINITY;

            for &idx in group {
                let worker = &workers[idx];

                // Capacity check (load counter = in-flight request count).
                if Self::current_queue(worker) as usize >= self.cfg.max_concurrent_seqs_per_instance
                {
                    continue;
                }

                // KV-cache space check (engine token base + inflight estimates).
                let token_num = self.current_token_num(worker, use_budget);
                if !self.can_run_directly(worker, req_tokens, token_num) {
                    continue;
                }

                // Marginal throughput gain estimate.
                let running = Self::current_running(worker);
                let curr = match self.estimate_throughput(worker, running, token_num) {
                    Some(t) => t,
                    None => continue,
                };
                let after =
                    match self.estimate_throughput(worker, running + 1, token_num + req_tokens) {
                        Some(t) => t,
                        None => continue,
                    };
                let delta = after - curr;

                if delta > best_delta {
                    best_delta = delta;
                    best_idx = Some(idx);
                }
            }

            // If this group yielded an eligible worker, return it without
            // falling through to a lower-priority group. No optimistic delta is
            // applied here — the worker's load counter and inflight-token map
            // are updated by the `WorkerLoadGuard` minted post-commit in the
            // execution stage, so the bookkeeping lifecycle is owned by the
            // request path rather than the selection call.
            if let Some(idx) = best_idx.filter(|_| best_delta >= threshold) {
                return Some(idx);
            }

            // No eligible worker in this group → fall through to the next group.
        }

        None
    }
}

// ---------------------------------------------------------------------------
// ThroughputOptimalPolicy
// ---------------------------------------------------------------------------

/// Selects the worker that maximises marginal throughput, estimated via a
/// per-TP/PP latency cost model.
///
/// Requires [`SelectWorkerInfo::tokens`] to be populated.
#[derive(Debug)]
pub struct ThroughputOptimalPolicy {
    rt: ThroughputRuntime,
}

impl ThroughputOptimalPolicy {
    /// Create with the given configuration.
    ///
    /// Returns an error if the cost model file cannot be loaded.
    pub fn with_config(cfg: ThroughputOptimalConfig) -> Result<Self, String> {
        Ok(Self {
            rt: ThroughputRuntime::new(cfg)?,
        })
    }
}

impl LoadBalancingPolicy for ThroughputOptimalPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.rt.select_worker(workers, info, false)
    }

    fn name(&self) -> &'static str {
        "throughput_optimal"
    }

    fn needs_load_guard(&self) -> bool {
        true // routes on worker.load() + inflight_tokens_sum()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// ThroughputOptimalWithBudgetPolicy
// ---------------------------------------------------------------------------

/// Like [`ThroughputOptimalPolicy`] but accounts for KV-cache page granularity
/// when estimating each worker's current token load.
///
/// Response tokens are rounded up to the nearest multiple of `request_budget`
/// via [`EngineStats::token_num_with_budget`], reflecting the fact that a
/// partially-filled page occupies the same KV-cache space as a full page.
#[derive(Debug)]
pub struct ThroughputOptimalWithBudgetPolicy {
    rt: ThroughputRuntime,
}

impl ThroughputOptimalWithBudgetPolicy {
    /// Create with the given configuration.
    ///
    /// Returns an error if the cost model file cannot be loaded.
    pub fn with_config(cfg: ThroughputOptimalConfig) -> Result<Self, String> {
        Ok(Self {
            rt: ThroughputRuntime::new(cfg)?,
        })
    }

    /// Returns the configured KV-cache page budget.
    pub fn budget(&self) -> usize {
        self.rt.cfg.request_budget
    }
}

impl LoadBalancingPolicy for ThroughputOptimalWithBudgetPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.rt.select_worker(workers, info, true)
    }

    fn name(&self) -> &'static str {
        "throughput_optimal_with_budget"
    }

    fn needs_load_guard(&self) -> bool {
        true // routes on worker.load() + inflight_tokens_sum()
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
    use std::collections::HashMap;

    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, EngineStats, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn build_worker(url: &str, labels: HashMap<String, String>) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn default_labels() -> HashMap<String, String> {
        [
            ("tp_size", "8"),
            ("pp_size", "1"),
            ("max_total_tokens", "20000"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// Push [`EngineStats`] onto the worker via the public `update_engine_stats` API.
    ///
    /// Uses a far-future timestamp so the monotonicity check always passes across
    /// successive calls within the same test, and `snapshot_staleness_threshold_ms = 0`
    /// skips the age check entirely.
    fn set_stats(
        worker: &Arc<dyn Worker>,
        running: usize,
        waiting: usize,
        prompt: HashMap<&str, usize>,
        response: HashMap<&str, usize>,
        kv_cache_usage: f64,
    ) {
        use crate::worker::{EngineSchedulerStats, EngineStatsUpdateOutcome};

        let req_id_to_prompt_token_num: HashMap<String, usize> = prompt
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let req_id_to_response_token_num: HashMap<String, usize> = response
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let total_prompt_tokens = req_id_to_prompt_token_num.values().sum();
        let total_response_tokens = req_id_to_response_token_num.values().sum();

        // Far-future timestamp ensures monotonicity check always passes.
        let timestamp =
            chrono::TimeZone::with_ymd_and_hms(&chrono::Utc, 2099, 1, 1, 0, 0, 0).unwrap();
        let stats = EngineStats {
            timestamp,
            scheduler_stats: EngineSchedulerStats {
                req_id_to_prompt_token_num,
                req_id_to_response_token_num,
                num_running_reqs: running,
                num_waiting_reqs: waiting,
                kv_cache_usage,
                total_prompt_tokens,
                total_response_tokens,
            },
        };
        let outcome = worker.update_engine_stats(stats, 0);
        assert!(
            matches!(outcome, EngineStatsUpdateOutcome::Applied),
            "set_stats failed: {outcome:?}"
        );
    }

    fn info_with_tokens(tokens: &[u32]) -> SelectWorkerInfo<'_> {
        SelectWorkerInfo {
            tokens: Some(tokens),
            ..Default::default()
        }
    }

    /// A minimal cost-model JSON covering the `TP8_PP1` key used by `default_labels()`.
    const TEST_COST_MODEL_JSON: &str = r#"{
        "TP8_PP1": {
            "other_threshold": 0.5,
            "other_latency_b": 0.1,
            "other_latency_k": 0.02,
            "attn_latency_b": 0.05,
            "attn_latency_k": 0.001
        }
    }"#;

    /// Build a [`ThroughputOptimalConfig`] that has a valid cost model for `TP8_PP1`.
    fn config_with_cost_model() -> ThroughputOptimalConfig {
        use tempfile::NamedTempFile;
        let mut tf = NamedTempFile::new().expect("create temp file");
        std::io::Write::write_all(&mut tf, TEST_COST_MODEL_JSON.as_bytes())
            .expect("write cost model");
        let path = tf.into_temp_path().keep().expect("persist temp file");
        ThroughputOptimalConfig {
            cost_model_path: path.to_string_lossy().into_owned(),
            max_concurrent_seqs_per_instance: 1_024,
            delta_throughput_threshold: 0.5,
            max_prompt_length: 8_192,
            request_budget: 1_024,
            max_num_waiting_reqs_after_preemption: 1_000,
        }
    }

    /// Build a [`ThroughputOptimalConfig`] with a cost model and custom threshold.
    fn config_with_cost_model_and_threshold(threshold: f64) -> ThroughputOptimalConfig {
        ThroughputOptimalConfig {
            delta_throughput_threshold: threshold,
            ..config_with_cost_model()
        }
    }

    /// Build a [`ThroughputOptimalConfig`] with a cost model and custom max_concurrent_seqs.
    fn config_with_cost_model_and_max_seqs(max_seqs: usize) -> ThroughputOptimalConfig {
        ThroughputOptimalConfig {
            max_concurrent_seqs_per_instance: max_seqs,
            ..config_with_cost_model()
        }
    }

    /// Build a [`ThroughputOptimalWithBudgetPolicy`] config with a cost model.
    fn budget_config_with_cost_model(budget: usize, threshold: f64) -> ThroughputOptimalConfig {
        ThroughputOptimalConfig {
            request_budget: budget,
            delta_throughput_threshold: threshold,
            ..config_with_cost_model()
        }
    }

    // ── ThroughputRuntime helpers ────────────────────────────────────────

    #[test]
    fn tp_pp_key_from_labels() {
        let mut labels = HashMap::new();
        labels.insert("tp_size".to_string(), "8".to_string());
        labels.insert("pp_size".to_string(), "2".to_string());
        let w = build_worker("http://w1:8000", labels);
        assert_eq!(ThroughputRuntime::label_i64(&w, "tp_size"), Some(8));
        assert_eq!(ThroughputRuntime::tp_pp_key(&w), "TP8_PP2");
    }

    #[test]
    fn tp_pp_key_defaults_to_tp1_pp1() {
        let w = build_worker("http://w1:8000", HashMap::new());
        assert_eq!(ThroughputRuntime::tp_pp_key(&w), "TP1_PP1");
    }

    #[test]
    fn request_token_num_defaults_to_one_without_tokens() {
        // No tokens provided → fallback to 1, no panic.
        let count = ThroughputRuntime::request_token_num(&SelectWorkerInfo::default());
        assert_eq!(count, 1);
    }

    #[test]
    fn request_token_num_returns_exact_count() {
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let info = info_with_tokens(&tokens);
        assert_eq!(ThroughputRuntime::request_token_num(&info), 5);
    }

    #[test]
    fn request_token_num_with_budget_adds_response_tokens_to_prompt_tokens() {
        let rt = ThroughputRuntime::new(budget_config_with_cost_model(64, 0.5))
            .expect("runtime creation should succeed");
        let tokens: Vec<u32> = (0..10).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            response_token_count: Some(130),
            ..Default::default()
        };

        // 10 prompt tokens + ceil((130 response tokens + 1) / 64) * 64.
        assert_eq!(rt.request_token_num_with_budget(&info), 202);
    }

    // ── ThroughputOptimalPolicy ──────────────────────────────────────────

    #[test]
    fn empty_workers_returns_none() {
        let policy = ThroughputOptimalPolicy::with_config(config_with_cost_model())
            .expect("policy creation should succeed");
        let tokens: &[u32] = &[1, 2, 3];
        let selected = policy.select_worker(&[], &info_with_tokens(tokens));
        assert_eq!(selected, None);
    }

    #[test]
    fn all_unhealthy_returns_none() {
        let policy = ThroughputOptimalPolicy::with_config(config_with_cost_model())
            .expect("policy creation should succeed");
        let w = build_worker("http://w1:8000", default_labels());
        w.set_status(WorkerStatus::NotReady);
        let tokens: Vec<u32> = vec![1, 2];
        let selected = policy.select_worker(&[w], &info_with_tokens(&tokens));
        assert_eq!(selected, None);
    }

    #[test]
    fn selects_worker_with_lower_token_load() {
        let policy =
            ThroughputOptimalPolicy::with_config(config_with_cost_model_and_threshold(0.01))
                .expect("policy creation should succeed");

        let w1 = build_worker("http://w1:8000", default_labels());
        let w2 = build_worker("http://w2:8000", default_labels());

        // w1 has heavy load, w2 is nearly idle.
        for _ in 0..4 {
            w1.increment_load();
            w2.increment_load();
        }
        set_stats(
            &w1,
            4,
            0,
            HashMap::from([("r1", 15_000)]),
            HashMap::new(),
            0.75,
        );
        set_stats(
            &w2,
            4,
            0,
            HashMap::from([("r2", 500)]),
            HashMap::new(),
            0.03,
        );

        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let selected = policy.select_worker(&workers, &info_with_tokens(&tokens));
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn saturated_worker_skipped() {
        let policy = ThroughputOptimalPolicy::with_config(config_with_cost_model_and_max_seqs(2))
            .expect("policy creation should succeed");

        let w1 = build_worker("http://w1:8000", default_labels());
        for _ in 0..3 {
            w1.increment_load();
        }
        set_stats(&w1, 3, 0, HashMap::from([("r1", 100)]), HashMap::new(), 0.1);

        let tokens: Vec<u32> = vec![1];
        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let selected = policy.select_worker(&workers, &info_with_tokens(&tokens));
        assert_eq!(selected, None);
    }

    #[test]
    fn waiting_queue_prevents_selection() {
        let policy =
            ThroughputOptimalPolicy::with_config(config_with_cost_model_and_threshold(0.01))
                .expect("policy creation should succeed");

        let w = build_worker("http://w1:8000", default_labels());
        w.increment_load();
        set_stats(&w, 1, 5, HashMap::from([("r1", 100)]), HashMap::new(), 0.1);

        let tokens: Vec<u32> = vec![1, 2];
        let workers: Vec<Arc<dyn Worker>> = vec![w];
        let selected = policy.select_worker(&workers, &info_with_tokens(&tokens));
        assert_eq!(selected, None);
    }

    #[test]
    fn delta_below_threshold_returns_none() {
        let mut cfg = config_with_cost_model();
        cfg.delta_throughput_threshold = 1000.0; // impossibly high
        let policy =
            ThroughputOptimalPolicy::with_config(cfg).expect("policy creation should succeed");

        let w = build_worker("http://w1:8000", default_labels());
        w.increment_load();
        set_stats(&w, 1, 0, HashMap::from([("r1", 100)]), HashMap::new(), 0.1);

        let tokens: Vec<u32> = vec![1];
        let workers: Vec<Arc<dyn Worker>> = vec![w];
        let selected = policy.select_worker(&workers, &info_with_tokens(&tokens));
        assert_eq!(selected, None);
    }

    #[test]
    fn idle_single_worker_selected() {
        let policy = ThroughputOptimalPolicy::with_config(config_with_cost_model())
            .expect("policy creation should succeed");
        let w = build_worker("http://w1:8000", default_labels());
        set_stats(&w, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let tokens: Vec<u32> = vec![1, 2, 3];
        let workers: Vec<Arc<dyn Worker>> = vec![w];
        let selected = policy.select_worker(&workers, &info_with_tokens(&tokens));
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn cost_model_load_failure_propagates_error() {
        // When the cost model file cannot be loaded, the policy should
        // propagate the error from with_config() rather than silently
        // skipping workers in select_worker.
        let cfg = ThroughputOptimalConfig {
            cost_model_path: "/nonexistent/cost_model.json".to_string(),
            ..config_with_cost_model()
        };
        let result = ThroughputOptimalPolicy::with_config(cfg);
        assert!(result.is_err(), "should fail with invalid cost model path");

        let err = result.unwrap_err();
        assert!(
            err.contains("failed to load cost model"),
            "error message should mention cost model load failure, got: {err}"
        );
    }

    #[test]
    fn cost_model_missing_tp_pp_key_returns_none() {
        // A worker whose TP/PP key is absent from the cost model should be skipped.
        let cfg = config_with_cost_model();
        // The cost model only has "TP8_PP1"; this worker uses TP4_PP1.
        let mut labels = HashMap::new();
        labels.insert("tp_size".to_string(), "4".to_string());
        labels.insert("pp_size".to_string(), "1".to_string());
        labels.insert("max_total_tokens".to_string(), "20000".to_string());
        let w = build_worker("http://w1:8000", labels);
        set_stats(&w, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let policy =
            ThroughputOptimalPolicy::with_config(cfg).expect("policy creation should succeed");
        let tokens: Vec<u32> = vec![1, 2, 3];
        let workers: Vec<Arc<dyn Worker>> = vec![w];
        let selected = policy.select_worker(&workers, &info_with_tokens(&tokens));
        assert_eq!(selected, None);
    }

    // ── ThroughputOptimalWithBudgetPolicy ────────────────────────────────

    #[test]
    fn budget_policy_name() {
        let policy = ThroughputOptimalWithBudgetPolicy::with_config(config_with_cost_model())
            .expect("policy creation should succeed");
        assert_eq!(policy.name(), "throughput_optimal_with_budget");
    }

    #[test]
    fn budget_accessor() {
        let p = ThroughputOptimalWithBudgetPolicy::with_config(ThroughputOptimalConfig {
            request_budget: 256,
            cost_model_path: config_with_cost_model().cost_model_path,
            ..config_with_cost_model()
        })
        .expect("policy creation should succeed");
        assert_eq!(p.budget(), 256);
    }

    #[test]
    fn budget_policy_selects_idle_worker() {
        let policy = ThroughputOptimalWithBudgetPolicy::with_config(config_with_cost_model())
            .expect("policy creation should succeed");
        let w = build_worker("http://w1:8000", default_labels());
        set_stats(&w, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let workers: Vec<Arc<dyn Worker>> = vec![w];
        let selected = policy.select_worker(&workers, &info_with_tokens(&tokens));
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn budget_policy_prefers_lower_load() {
        let policy =
            ThroughputOptimalWithBudgetPolicy::with_config(budget_config_with_cost_model(16, 0.01))
                .expect("policy creation should succeed");

        let w1 = build_worker("http://w1:8000", default_labels());
        let w2 = build_worker("http://w2:8000", default_labels());

        for _ in 0..4 {
            w1.increment_load();
            w2.increment_load();
        }
        // w1 has many response tokens → more budget-rounded load
        set_stats(
            &w1,
            4,
            0,
            HashMap::from([("r1", 8000)]),
            HashMap::from([("r1", 200)]),
            0.5,
        );
        // w2 is nearly idle
        set_stats(
            &w2,
            4,
            0,
            HashMap::from([("r2", 100)]),
            HashMap::new(),
            0.01,
        );

        let tokens: Vec<u32> = vec![1, 2, 3];
        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let selected = policy.select_worker(&workers, &info_with_tokens(&tokens));
        assert_eq!(selected, Some(1));
    }

    // ── Priority groups ──────────────────────────────────────────────────

    /// When `priority_groups` is set, the policy should prefer the highest-
    /// priority group (lowest numeric value) even if a lower-priority group has
    /// a less-loaded worker.
    #[test]
    fn priority_groups_prefers_higher_priority_group() {
        let policy =
            ThroughputOptimalPolicy::with_config(config_with_cost_model_and_threshold(0.01))
                .expect("policy creation should succeed");

        let w0 = build_worker("http://w0:8000", default_labels()); // priority 0 (high)
        let w1 = build_worker("http://w1:8000", default_labels()); // priority 1 (low)

        // w0 has moderate load; w1 is completely idle.
        set_stats(
            &w0,
            2,
            0,
            HashMap::from([("r0a", 1000), ("r0b", 1000)]),
            HashMap::new(),
            0.1,
        );
        set_stats(&w1, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let tokens: Vec<u32> = vec![1, 2, 3];
        // priority_groups[0] = 0  → w0 is high-priority
        // priority_groups[1] = 1  → w1 is low-priority
        let priority_groups = [0i64, 1i64];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            priority_groups: Some(&priority_groups),
            ..Default::default()
        };

        // w0 is eligible (low enough load, no waiting queue) so the policy
        // must not fall through to w1 even though w1 is idle.
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(selected, Some(0), "should pick the high-priority worker");
    }

    /// When the high-priority group is saturated, the policy falls through to
    /// the next group.
    #[test]
    fn priority_groups_falls_through_when_high_priority_saturated() {
        // Set max_concurrent_seqs to 2 so w0 (3 running) is over-capacity.
        let policy = ThroughputOptimalPolicy::with_config(config_with_cost_model_and_max_seqs(2))
            .expect("policy creation should succeed");

        let w0 = build_worker("http://w0:8000", default_labels()); // priority 0 (high, saturated)
        let w1 = build_worker("http://w1:8000", default_labels()); // priority 1 (low, idle)

        set_stats(
            &w0,
            3,
            0,
            HashMap::from([("r0a", 1000), ("r0b", 1000), ("r0c", 500)]),
            HashMap::new(),
            0.2,
        );
        set_stats(&w1, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let tokens: Vec<u32> = vec![1, 2, 3];
        let priority_groups = [0i64, 1i64];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            priority_groups: Some(&priority_groups),
            ..Default::default()
        };

        // w0 is over the max_concurrent_seqs limit → must fall through to w1.
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(
            selected,
            Some(1),
            "should fall through to low-priority worker"
        );
    }

    /// Without priority_groups the policy behaves identically to the original
    /// (single-group) behaviour.
    #[test]
    fn no_priority_groups_single_group_behaviour() {
        let policy =
            ThroughputOptimalPolicy::with_config(config_with_cost_model_and_threshold(0.01))
                .expect("policy creation should succeed");

        let w0 = build_worker("http://w0:8000", default_labels());
        let w1 = build_worker("http://w1:8000", default_labels());

        set_stats(
            &w0,
            4,
            0,
            HashMap::from([("r0", 15_000)]),
            HashMap::new(),
            0.75,
        );
        set_stats(&w1, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let workers: Vec<Arc<dyn Worker>> = vec![w0, w1];
        let tokens: Vec<u32> = vec![1, 2, 3];
        // No priority_groups — all workers treated equally.
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(selected, Some(1), "should pick the less-loaded worker");
    }
}
