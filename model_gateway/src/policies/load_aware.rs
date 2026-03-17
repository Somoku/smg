// PR 3 §3.2: Load-aware routing policies for throughput-optimal worker selection.
//! Ported from sgl-model-gateway `engine_aware.rs` with adaptations:
//! - Synchronous `select_worker` (SMG trait is not async)
//! - Uses `EngineStats` (PR 1) instead of `WorkerStats`
//! - Module name `load_aware` (not `engine_aware`)

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use tracing::{error, warn};

use super::{
    cost_model_utils::{CostModel, CostModelEntry},
    get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::Worker;

// PR 3 §3.2a: One-shot log guards for noisy fallback paths
static REPORTED_MISSING_TOKEN_STATS: AtomicBool = AtomicBool::new(false);
static REPORTED_GET_LOAD_IGNORED: AtomicBool = AtomicBool::new(false);
static REPORTED_MISSING_COST_MODEL: AtomicBool = AtomicBool::new(false);

// PR 3 §3.2a: Sensible fallback when cost model is loaded but missing a specific TP/PP key.
// Only used for per-key misses, NOT as a substitute for a missing cost model file.
const DEFAULT_FALLBACK_COST_MODEL_ENTRY: CostModelEntry = CostModelEntry {
    other_threshold: 0.0,
    other_latency_b: 1.0,
    other_latency_k: 0.02,
    attn_latency_b: 0.0,
    attn_latency_k: 0.001,
};

// ── Config types ────────────────────────────────────────────────────────

// PR 3 §3.2b: Configuration for RequestNumBalancePolicy
#[derive(Debug, Clone)]
pub struct RequestNumBalanceConfig {
    pub balanced_concurrent_seqs_per_instance: usize,
    pub max_concurrent_seqs_per_instance: usize,
}

impl Default for RequestNumBalanceConfig {
    fn default() -> Self {
        Self {
            balanced_concurrent_seqs_per_instance: 512,
            max_concurrent_seqs_per_instance: 1024,
        }
    }
}

// PR 3 §3.2c-d: Configuration for ThroughputOptimal and ThroughputOptimalWithBudget
#[derive(Debug, Clone)]
pub struct ThroughputOptimalConfig {
    pub cost_model_path: Option<String>,
    pub max_num_waiting_reqs_after_preemption: usize,
    pub balanced_concurrent_seqs_per_instance: usize,
    pub max_concurrent_seqs_per_instance: usize,
    pub delta_throughput_threshold: f64,
    pub max_prompt_length: usize,
    pub request_budget: usize,
}

impl Default for ThroughputOptimalConfig {
    fn default() -> Self {
        Self {
            cost_model_path: None,
            max_num_waiting_reqs_after_preemption: 1_000,
            balanced_concurrent_seqs_per_instance: 512,
            max_concurrent_seqs_per_instance: 1_024,
            delta_throughput_threshold: 0.5,
            max_prompt_length: 8_192,
            request_budget: 1_024,
        }
    }
}

// ── Runtime ─────────────────────────────────────────────────────────────

// PR 3 §3.2a: Shared throughput estimation runtime
#[derive(Debug)]
struct ThroughputRuntime {
    cfg: ThroughputOptimalConfig,
    cost_model: Option<CostModel>,
}

impl ThroughputRuntime {
    fn new(cfg: ThroughputOptimalConfig) -> Self {
        let cost_model = match cfg.cost_model_path.as_deref() {
            Some(path) => match CostModel::load_from_file(path) {
                Ok(model) => Some(model),
                Err(e) => {
                    error!(
                        policy = "throughput_optimal",
                        path = %path,
                        error = %e,
                        "Failed to load cost model file — throughput estimation will use default coefficients. \
                         Provide a valid cost model file via --cost-model-path for accurate routing."
                    );
                    None
                }
            },
            None => {
                warn!(
                    policy = "throughput_optimal",
                    "No cost model file configured (--cost-model-path not set). \
                     Throughput estimation will use default coefficients, which may produce \
                     suboptimal routing decisions. Provide a cost model file for production use."
                );
                None
            }
        };

        Self { cfg, cost_model }
    }

    // PR 3 §3.2a: TP/PP key parsing from worker metadata labels
    #[inline]
    fn parse_label_i64(worker: &Arc<dyn Worker>, key: &str) -> Option<i64> {
        worker
            .metadata()
            .spec
            .labels
            .get(key)
            .and_then(|v: &String| v.parse::<i64>().ok())
    }

    #[inline]
    fn max_model_len(&self, worker: &Arc<dyn Worker>) -> i64 {
        Self::parse_label_i64(worker, "max_model_len")
            .or_else(|| Self::parse_label_i64(worker, "max_total_tokens"))
            .unwrap_or_else(|| {
                (self.cfg.max_prompt_length
                    + self.cfg.request_budget * self.cfg.max_num_waiting_reqs_after_preemption)
                    as i64
            })
    }

    #[inline]
    fn tp_pp_key(worker: &Arc<dyn Worker>) -> String {
        let tp = Self::parse_label_i64(worker, "tp_size").unwrap_or(1);
        let pp = Self::parse_label_i64(worker, "pp_size").unwrap_or(1);
        format!("TP{tp}_PP{pp}")
    }

    #[inline]
    fn candidate_indicator(worker: &Arc<dyn Worker>) -> i64 {
        // Align with PSRL version-priority behavior when no explicit list is passed.
        Self::parse_label_i64(worker, "version_tag").unwrap_or(0)
    }

    // PR 3 §3.2a: Token estimation helpers
    // PR 13 Gap 2: Removed the text.len()/4 character-count heuristic — it was a wrong
    // approximation that could skew ThroughputOptimalPolicy decisions. When tokens are not
    // available, return 1 as a neutral estimate. The routing loop now passes
    // PreparationOutput::token_ids via SelectWorkerInfo::tokens, so the accurate count
    // is available on all paths where it matters.
    #[inline]
    #[expect(
        clippy::panic,
        reason = "tokens is contract-required for ThroughputOptimalPolicy callers; \
                  callers must populate SelectWorkerInfo::tokens before invoking"
    )]
    fn request_token_num(info: &SelectWorkerInfo<'_>) -> i64 {
        if let Some(tokens) = info.tokens {
            tokens.len().max(1) as i64
        } else {
            panic!(
                "SelectWorkerInfo.tokens is required for ThroughputOptimalPolicy but was not provided. Please ensure that the token count is passed in SelectWorkerInfo for accurate routing decisions."
            )
        }
    }

    #[inline]
    fn request_token_num_with_budget(&self, info: &SelectWorkerInfo<'_>) -> i64 {
        let prompt = Self::request_token_num(info);
        let budget = self.cfg.request_budget.max(1) as i64;
        prompt + budget
    }

    // PR 3 §3.2a: Queue and token accessors using EngineStats (not WorkerStats)
    #[inline]
    fn current_waiting_request_num(worker: &Arc<dyn Worker>) -> i64 {
        let stats = worker.engine_stats();
        stats.waiting_queue_size() as i64
    }

    #[inline]
    fn current_queue_request_num(worker: &Arc<dyn Worker>) -> i64 {
        let stats = worker.engine_stats();
        let queue = stats.waiting_and_running_queue_size() as i64;
        if queue > 0 || stats.running_queue_size() > 0 {
            queue.max(worker.load() as i64)
        } else {
            worker.load() as i64
        }
    }

    #[inline]
    fn current_running_request_num(worker: &Arc<dyn Worker>) -> i64 {
        let stats = worker.engine_stats();
        let running = stats.running_queue_size() as i64;
        if running > 0 {
            running
        } else {
            worker.load() as i64
        }
    }

    #[inline]
    fn current_token_num(&self, worker: &Arc<dyn Worker>, use_budget: bool) -> i64 {
        let stats = worker.engine_stats();
        let has_stats = stats.waiting_and_running_queue_size() > 0 || stats.total_token_num() > 0;

        if has_stats {
            tracing::trace!(
                policy = "throughput_optimal",
                worker = %worker.url(),
                waiting = stats.waiting_queue_size(),
                running = stats.running_queue_size(),
                "token stats"
            );

            let token_num = if use_budget {
                stats.token_num_with_budget(self.cfg.request_budget)
            } else {
                stats.total_token_num()
            };

            if token_num > 0 || stats.waiting_and_running_queue_size() == 0 {
                return token_num as i64;
            }

            // Fallback to KV-cache estimate when detailed token maps are not populated.
            let estimated = (stats.snapshot.scheduler_stats.kv_cache_usage
                * self.max_model_len(worker) as f64)
                .ceil() as i64;
            if estimated > 0 {
                return estimated;
            }
        } else if !REPORTED_MISSING_TOKEN_STATS.swap(true, Ordering::Relaxed) {
            error!(
                policy = "throughput_optimal",
                worker = %worker.url(),
                "Engine stats unavailable; please push stats via /workers/stats"
            );
        }

        0
    }

    #[inline]
    fn can_run_directly(
        &self,
        worker: &Arc<dyn Worker>,
        request_token_num: i64,
        current_token_num: i64,
    ) -> bool {
        if Self::current_waiting_request_num(worker) > 0 {
            return false;
        }

        let max_model_len = self.max_model_len(worker);
        current_token_num + request_token_num <= max_model_len
    }

    // PR 3 §3.2a: Throughput estimation using cost model
    #[inline]
    fn estimate_throughput(
        &self,
        worker: &Arc<dyn Worker>,
        request_num: i64,
        token_num: i64,
    ) -> f64 {
        if request_num <= 0 {
            return 0.0;
        }

        let tp_pp_key = Self::tp_pp_key(worker);

        if let Some(ref model) = self.cost_model {
            if let Some(entry) = model.get(&tp_pp_key) {
                return entry.estimate_throughput(request_num, token_num);
            }
            // Cost model loaded but missing this specific TP/PP key — use default coefficients.
            // This is expected when workers have heterogeneous TP/PP configurations.
            warn!(
                policy = "throughput_optimal",
                tp_pp_key = %tp_pp_key,
                worker = %worker.url(),
                "Cost model has no entry for TP/PP key, using default coefficients"
            );
        } else if !REPORTED_MISSING_COST_MODEL.swap(true, Ordering::Relaxed) {
            error!(
                policy = "throughput_optimal",
                "No cost model loaded — using default coefficients for throughput estimation. \
                 Provide --cost-model-path for accurate routing."
            );
        }

        DEFAULT_FALLBACK_COST_MODEL_ENTRY.estimate_throughput(request_num, token_num)
    }

    #[inline]
    fn estimate_curr_throughput(
        &self,
        worker: &Arc<dyn Worker>,
        request_num: i64,
        token_num: i64,
    ) -> f64 {
        self.estimate_throughput(worker, request_num, token_num)
    }

    #[inline]
    fn estimate_curr_throughput_after_route_request(
        &self,
        worker: &Arc<dyn Worker>,
        running_request_num: i64,
        token_num: i64,
        request_token_num: i64,
    ) -> f64 {
        let new_request_num = running_request_num + 1;
        let new_token_num = token_num + request_token_num;
        self.estimate_throughput(worker, new_request_num, new_token_num)
    }

    #[inline]
    fn estimate_baseline_delta_throughput(
        &self,
        worker: &Arc<dyn Worker>,
        request_token_num: i64,
    ) -> f64 {
        self.estimate_throughput(worker, 1, request_token_num)
    }
}

// ── RequestNumBalancePolicy ─────────────────────────────────────────────

// PR 3 §3.2b: Load balancing by request count
#[derive(Debug)]
pub struct RequestNumBalancePolicy {
    cfg: RequestNumBalanceConfig,
}

impl RequestNumBalancePolicy {
    pub fn new() -> Self {
        Self::with_config(RequestNumBalanceConfig::default())
    }

    pub fn with_config(cfg: RequestNumBalanceConfig) -> Self {
        Self { cfg }
    }
}

impl LoadBalancingPolicy for RequestNumBalancePolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        let (best_idx, best_load) = healthy_indices
            .into_iter()
            .map(|idx| (idx, workers[idx].load()))
            .min_by_key(|(_, load)| *load)?;

        // Reject when saturated
        if best_load >= self.cfg.max_concurrent_seqs_per_instance {
            return None;
        }

        Some(best_idx)
    }

    fn name(&self) -> &'static str {
        "request_num_balance"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for RequestNumBalancePolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ── ThroughputOptimalPolicy ─────────────────────────────────────────────

// PR 3 §3.2c: Throughput-optimal routing using cost model
#[derive(Debug)]
pub struct ThroughputOptimalPolicy {
    rt: ThroughputRuntime,
}

impl ThroughputOptimalPolicy {
    pub fn new() -> Self {
        Self::with_config(ThroughputOptimalConfig::default())
    }

    pub fn with_config(cfg: ThroughputOptimalConfig) -> Self {
        Self {
            rt: ThroughputRuntime::new(cfg),
        }
    }

    fn select_worker_internal(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        use_budget: bool,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let request_token_num = if use_budget {
            self.rt.request_token_num_with_budget(info)
        } else {
            ThroughputRuntime::request_token_num(info)
        };

        // PR 3 §3.2c: Group workers by candidate indicator (version tag or explicit IDs)
        let mut grouped: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        if let Some(candidate_group_ids) = info
            .candidate_group_ids
            .filter(|ids| ids.len() == workers.len())
        {
            for idx in healthy_indices {
                grouped
                    .entry(candidate_group_ids[idx] as i64)
                    .or_default()
                    .push(idx);
            }
        } else {
            for idx in healthy_indices {
                let indicator = ThroughputRuntime::candidate_indicator(&workers[idx]);
                grouped.entry(indicator).or_default().push(idx);
            }
        }

        // Iterate groups in BTreeMap order (lowest indicator first)
        for (_, group) in grouped {
            let baseline = self
                .rt
                .estimate_baseline_delta_throughput(&workers[group[0]], request_token_num);
            let threshold = baseline * self.rt.cfg.delta_throughput_threshold;

            let mut best_idx: Option<usize> = None;
            let mut best_delta = f64::NEG_INFINITY;

            for idx in group {
                let worker = &workers[idx];
                let queue_request_num = ThroughputRuntime::current_queue_request_num(worker);
                let running_request_num = ThroughputRuntime::current_running_request_num(worker);
                let token_num = self.rt.current_token_num(worker, use_budget);

                if queue_request_num as usize >= self.rt.cfg.max_concurrent_seqs_per_instance {
                    continue;
                }

                if !self
                    .rt
                    .can_run_directly(worker, request_token_num, token_num)
                {
                    continue;
                }

                let curr = self
                    .rt
                    .estimate_curr_throughput(worker, running_request_num, token_num);
                let after = self.rt.estimate_curr_throughput_after_route_request(
                    worker,
                    running_request_num,
                    token_num,
                    request_token_num,
                );
                let delta = after - curr;
                if delta > best_delta {
                    best_delta = delta;
                    best_idx = Some(idx);
                }
            }

            if let Some(idx) = best_idx {
                if best_delta >= threshold {
                    return Some(idx);
                }
            }
        }

        None
    }
}

impl LoadBalancingPolicy for ThroughputOptimalPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.select_worker_internal(workers, info, false)
    }

    fn name(&self) -> &'static str {
        "throughput_optimal"
    }

    fn update_loads(
        &self,
        _loads: &std::collections::HashMap<String, openai_protocol::worker::WorkerLoadResponse>,
    ) {
        if !REPORTED_GET_LOAD_IGNORED.swap(true, Ordering::Relaxed) {
            warn!(
                policy = "throughput_optimal",
                "Received /get_load token loads, but cached_token_loads has been removed. This update is ignored"
            );
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for ThroughputOptimalPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ── ThroughputOptimalWithBudgetPolicy ───────────────────────────────────

// PR 3 §3.2d: Throughput-optimal with per-request token budget
#[derive(Debug)]
pub struct ThroughputOptimalWithBudgetPolicy {
    inner: ThroughputOptimalPolicy,
}

impl ThroughputOptimalWithBudgetPolicy {
    pub fn new() -> Self {
        Self::with_config(ThroughputOptimalConfig::default())
    }

    pub fn with_config(cfg: ThroughputOptimalConfig) -> Self {
        Self {
            inner: ThroughputOptimalPolicy::with_config(cfg),
        }
    }
}

impl LoadBalancingPolicy for ThroughputOptimalWithBudgetPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.inner.select_worker_internal(workers, info, true)
    }

    fn name(&self) -> &'static str {
        "throughput_optimal_with_budget"
    }

    fn update_loads(
        &self,
        loads: &std::collections::HashMap<String, openai_protocol::worker::WorkerLoadResponse>,
    ) {
        self.inner.update_loads(loads);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for ThroughputOptimalWithBudgetPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{
        worker::{EngineSchedulerStats, EngineStats, EngineStatsSnapshot},
        BasicWorkerBuilder, WorkerType,
    };

    // Helper to build a worker with given labels
    fn build_worker(url: &str, labels: HashMap<String, String>) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .build(),
        )
    }

    // Helper to build a default-label worker
    fn build_simple_worker(url: &str) -> Arc<dyn Worker> {
        build_worker(url, HashMap::new())
    }

    // Helper to push engine stats onto a worker
    fn push_stats(
        worker: &Arc<dyn Worker>,
        num_running: usize,
        num_waiting: usize,
        prompt_tokens: HashMap<String, usize>,
        response_tokens: HashMap<String, usize>,
        kv_cache_usage: f64,
    ) {
        use crate::core::EngineStatsUpdateOutcome;
        // Use a far-future timestamp to ensure it always passes monotonicity checks.
        // The default snapshot timestamp is Utc::now(), so we need something newer.
        let stats = EngineStats {
            snapshot: EngineStatsSnapshot {
                timestamp: "2099-01-01T00:00:00Z".to_string(),
                scheduler_stats: EngineSchedulerStats {
                    req_id_to_prompt_token_num: prompt_tokens,
                    req_id_to_response_token_num: response_tokens,
                    num_running_reqs: num_running,
                    num_waiting_reqs: num_waiting,
                    kv_cache_usage,
                },
            },
        };
        let outcome = worker.update_engine_stats(stats, 0);
        assert!(
            matches!(outcome, EngineStatsUpdateOutcome::Applied),
            "push_stats failed: {outcome:?}"
        );
    }

    // PR 13 Gap 2 test helper: throughput policies now require explicit tokens in
    // SelectWorkerInfo, so the tests share a stable tokenized request fixture.
    const TEST_REQUEST_TOKENS: &[u32] = &[11, 22, 33, 44];

    fn throughput_info<'a>(request_text: &'a str) -> SelectWorkerInfo<'a> {
        SelectWorkerInfo {
            request_text: Some(request_text),
            tokens: Some(TEST_REQUEST_TOKENS),
            ..Default::default()
        }
    }

    // ── RequestNumBalance tests ─────────────────────────────────────────

    // PR 3 §3.6 test: Picks worker with lowest load()
    #[test]
    fn test_request_num_balance_selects_lowest_load() {
        let policy = RequestNumBalancePolicy::with_config(RequestNumBalanceConfig {
            balanced_concurrent_seqs_per_instance: 2,
            max_concurrent_seqs_per_instance: 10,
        });

        let w1 = build_simple_worker("http://w1:8000");
        let w2 = build_simple_worker("http://w2:8000");

        for _ in 0..5 {
            w1.increment_load();
        }
        for _ in 0..2 {
            w2.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(1));
    }

    // PR 3 §3.6 test: All at max → None
    #[test]
    fn test_request_num_balance_all_saturated() {
        let policy = RequestNumBalancePolicy::with_config(RequestNumBalanceConfig {
            balanced_concurrent_seqs_per_instance: 2,
            max_concurrent_seqs_per_instance: 3,
        });

        let w1 = build_simple_worker("http://w1:8000");
        let w2 = build_simple_worker("http://w2:8000");

        for _ in 0..3 {
            w1.increment_load();
            w2.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, None);
    }

    // PR 3 §3.6 test: Single worker
    #[test]
    fn test_request_num_balance_single_worker() {
        let policy = RequestNumBalancePolicy::new();
        let w1 = build_simple_worker("http://w1:8000");
        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(0));
    }

    // PR 3 §3.6 test: Equal load → first wins (min_by_key stable)
    #[test]
    fn test_request_num_balance_equal_load() {
        let policy = RequestNumBalancePolicy::new();
        let w1 = build_simple_worker("http://w1:8000");
        let w2 = build_simple_worker("http://w2:8000");
        w1.increment_load();
        w2.increment_load();
        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(0));
    }

    // PR 3 §3.6 test: All unhealthy → None
    #[test]
    fn test_request_num_balance_no_healthy() {
        let policy = RequestNumBalancePolicy::new();
        let w1 = build_simple_worker("http://w1:8000");
        let w2 = build_simple_worker("http://w2:8000");
        w1.set_healthy(false);
        w2.set_healthy(false);
        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, None);
    }

    // PR 3 §3.6 test: Empty workers → None
    #[test]
    fn test_request_num_balance_empty_workers() {
        let policy = RequestNumBalancePolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, None);
    }

    // PR 3 §3.6 test: Mix of healthy/unhealthy
    #[test]
    fn test_request_num_balance_one_unhealthy() {
        let policy = RequestNumBalancePolicy::new();
        let w1 = build_simple_worker("http://w1:8000");
        let w2 = build_simple_worker("http://w2:8000");

        // w1 has lower load but is unhealthy
        for _ in 0..1 {
            w1.increment_load();
        }
        for _ in 0..5 {
            w2.increment_load();
        }
        w1.set_healthy(false);

        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, Some(1));
    }

    // ── ThroughputOptimal tests ─────────────────────────────────────────

    fn default_throughput_labels() -> HashMap<String, String> {
        let mut labels = HashMap::new();
        labels.insert("tp_size".to_string(), "8".to_string());
        labels.insert("pp_size".to_string(), "1".to_string());
        labels.insert("version_tag".to_string(), "0".to_string());
        labels.insert("max_total_tokens".to_string(), "20000".to_string());
        labels
    }

    // PR 3 §3.6 test: Picks worker with lower token load
    #[test]
    fn test_throughput_optimal_lower_token_load() {
        // Lower threshold to ensure marginal gain passes the check
        let policy = ThroughputOptimalPolicy::with_config(ThroughputOptimalConfig {
            delta_throughput_threshold: 0.1,
            ..Default::default()
        });

        let labels1 = default_throughput_labels();
        let labels2 = labels1.clone();

        let w1 = build_worker("http://w1:8000", labels1);
        let w2 = build_worker("http://w2:8000", labels2);

        for _ in 0..4 {
            w1.increment_load();
            w2.increment_load();
        }

        // w1 has high token load
        push_stats(
            &w1,
            4,
            0,
            HashMap::from([("r1".to_string(), 15_000)]),
            HashMap::new(),
            0.75,
        );

        // w2 has low token load → higher marginal throughput
        push_stats(
            &w2,
            4,
            0,
            HashMap::from([("r2".to_string(), 2_000)]),
            HashMap::new(),
            0.1,
        );

        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let info = throughput_info("hello world");
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(selected, Some(1));
    }

    // PR 3 §3.6 test: No healthy workers → None
    #[test]
    fn test_throughput_optimal_no_healthy() {
        let policy = ThroughputOptimalPolicy::new();
        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        w1.set_healthy(false);
        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let selected = policy.select_worker(&workers, &SelectWorkerInfo::default());
        assert_eq!(selected, None);
    }

    // PR 3 §3.6 test: All saturated → None
    #[test]
    fn test_throughput_optimal_all_saturated() {
        let policy = ThroughputOptimalPolicy::with_config(ThroughputOptimalConfig {
            max_concurrent_seqs_per_instance: 2,
            ..Default::default()
        });

        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        // Push load to saturate
        for _ in 0..3 {
            w1.increment_load();
        }
        push_stats(
            &w1,
            3,
            0,
            HashMap::from([("r1".to_string(), 100)]),
            HashMap::new(),
            0.1,
        );

        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let info = throughput_info("test");
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(selected, None);
    }

    // PR 3 §3.6 test: Workers with waiting reqs → can_run_directly false → skipped
    #[test]
    fn test_throughput_optimal_waiting_queue() {
        let policy = ThroughputOptimalPolicy::with_config(ThroughputOptimalConfig::default());

        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        w1.increment_load();
        push_stats(
            &w1,
            1,
            5, // waiting > 0
            HashMap::from([("r1".to_string(), 100)]),
            HashMap::new(),
            0.1,
        );

        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let info = throughput_info("test");
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(selected, None);
    }

    // PR 3 §3.6 test: Delta below threshold → None
    #[test]
    fn test_throughput_optimal_delta_below_threshold() {
        // Very high threshold to force rejection
        let policy = ThroughputOptimalPolicy::with_config(ThroughputOptimalConfig {
            delta_throughput_threshold: 100.0,
            ..Default::default()
        });

        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        w1.increment_load();
        push_stats(
            &w1,
            1,
            0,
            HashMap::from([("r1".to_string(), 100)]),
            HashMap::new(),
            0.1,
        );

        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let info = throughput_info("test");
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(selected, None);
    }

    // PR 3 §3.6 test: No cost model file → logs warning, uses default coefficients
    #[test]
    fn test_throughput_optimal_with_cost_model() {
        // No cost model file → warns at construction + errors once at runtime, but still works
        let policy = ThroughputOptimalPolicy::with_config(ThroughputOptimalConfig {
            cost_model_path: None,
            ..Default::default()
        });

        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        push_stats(&w1, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let info = throughput_info("test query");
        let selected = policy.select_worker(&workers, &info);
        // Should select the idle worker using fallback cost model
        assert_eq!(selected, Some(0));
    }

    // PR 3 §3.6 test: Multiple candidate groups → BTreeMap iteration
    #[test]
    fn test_throughput_optimal_candidate_group_ids() {
        let policy = ThroughputOptimalPolicy::with_config(ThroughputOptimalConfig::default());

        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        let w2 = build_worker("http://w2:8000", default_throughput_labels());

        // w1 in group 1 (higher priority), w2 in group 0 (lower number = first checked)
        let candidate_ids: Vec<u64> = vec![1, 0];

        // Make w1 busy so group 1 fails, w2 idle
        for _ in 0..2 {
            w1.increment_load();
        }
        push_stats(
            &w1,
            2,
            0,
            HashMap::from([("r1".to_string(), 5000)]),
            HashMap::new(),
            0.5,
        );

        // w2 is idle
        push_stats(&w2, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let workers: Vec<Arc<dyn Worker>> = vec![w1, w2];
        let info = SelectWorkerInfo {
            candidate_group_ids: Some(&candidate_ids),
            ..throughput_info("test")
        };
        let selected = policy.select_worker(&workers, &info);
        // group 0 (w2 at index 1) is checked first
        assert_eq!(selected, Some(1));
    }

    // PR 3 §3.6 test: KV cache fallback when total_token_num=0 but queue_size > 0
    #[test]
    fn test_throughput_optimal_kv_cache_fallback() {
        // Use a low threshold so the marginal throughput from KV-cache fallback passes.
        let policy = ThroughputOptimalPolicy::with_config(ThroughputOptimalConfig {
            delta_throughput_threshold: 0.05,
            ..Default::default()
        });

        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        w1.increment_load();

        // Running reqs > 0 but no token map entries → should use kv_cache_usage fallback.
        // kv_cache_usage=0.5, max_total_tokens=20000 → estimated tokens = ceil(0.5*20000) = 10000
        push_stats(&w1, 1, 0, HashMap::new(), HashMap::new(), 0.5);

        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let info = throughput_info("short");
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(selected, Some(0));
    }

    // PR 13 Gap 2: request_token_num no longer uses text.len()/4 heuristic.
    // When tokens are available, the exact count is returned.
    #[test]
    fn test_throughput_optimal_request_text_estimation() {
        // With exact tokens: exact count takes precedence.
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let token_estimate = ThroughputRuntime::request_token_num(&SelectWorkerInfo {
            request_text: Some("hello world"),
            tokens: Some(&tokens),
            ..Default::default()
        });
        assert_eq!(token_estimate, 5);
    }

    // ── ThroughputOptimalWithBudget tests ───────────────────────────────

    // PR 3 §3.6 test: Budget-aligned token count
    #[test]
    fn test_throughput_budget_adds_budget() {
        let policy = ThroughputOptimalWithBudgetPolicy::with_config(ThroughputOptimalConfig {
            request_budget: 512,
            ..Default::default()
        });

        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        push_stats(&w1, 0, 0, HashMap::new(), HashMap::new(), 0.0);

        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let info = throughput_info("test");
        let selected = policy.select_worker(&workers, &info);
        // Should select idle worker with budget calculation
        assert_eq!(selected, Some(0));
    }

    // PR 3 §3.6 test: Budget policy calls token_num_with_budget
    #[test]
    fn test_throughput_budget_current_tokens() {
        let policy = ThroughputOptimalWithBudgetPolicy::with_config(ThroughputOptimalConfig {
            request_budget: 256,
            ..Default::default()
        });

        let w1 = build_worker("http://w1:8000", default_throughput_labels());
        w1.increment_load();

        // Push stats with response tokens → token_num_with_budget will round up
        push_stats(
            &w1,
            1,
            0,
            HashMap::from([("r1".to_string(), 100)]),
            HashMap::from([("r1".to_string(), 50)]),
            0.05,
        );

        let workers: Vec<Arc<dyn Worker>> = vec![w1];
        let info = throughput_info("test query");
        let selected = policy.select_worker(&workers, &info);
        assert_eq!(selected, Some(0));
    }

    // ── Runtime tests ───────────────────────────────────────────────────

    // PR 3 §3.6 test: DEFAULT_FALLBACK_COST_MODEL_ENTRY produces sensible values
    #[test]
    fn test_fallback_cost_model_entry() {
        let throughput = DEFAULT_FALLBACK_COST_MODEL_ENTRY.estimate_throughput(4, 1000);
        assert!(
            throughput > 0.0,
            "Fallback cost model should produce positive throughput"
        );
        assert!(throughput.is_finite(), "Throughput should be finite");

        // Zero requests → zero throughput
        let zero = DEFAULT_FALLBACK_COST_MODEL_ENTRY.estimate_throughput(0, 0);
        // estimate_throughput with request_num=0 is handled by the caller (returns 0.0 from runtime)
        // But the entry itself should still produce a finite value
        assert!(zero.is_finite());
    }

    // PR 3 §3.6 test: TP/PP key parsing from worker labels
    #[test]
    fn test_tp_pp_key_parsing() {
        // With labels
        let mut labels = HashMap::new();
        labels.insert("tp_size".to_string(), "8".to_string());
        labels.insert("pp_size".to_string(), "2".to_string());
        let w = build_worker("http://w1:8000", labels);
        assert_eq!(ThroughputRuntime::tp_pp_key(&w), "TP8_PP2");

        // Without labels → defaults to TP1_PP1
        let w_default = build_simple_worker("http://w2:8000");
        assert_eq!(ThroughputRuntime::tp_pp_key(&w_default), "TP1_PP1");
    }

    // PR 13 Gap 2: ThroughputOptimal callers must now pass explicit tokens.
    #[test]
    #[should_panic(expected = "SelectWorkerInfo.tokens is required for ThroughputOptimalPolicy")]
    fn test_request_token_num_without_tokens_panics() {
        let _ = ThroughputRuntime::request_token_num(&SelectWorkerInfo::default());
    }

    // PR 3 §3.6 test: Policy name correctness
    #[test]
    fn test_policy_names() {
        assert_eq!(RequestNumBalancePolicy::new().name(), "request_num_balance");
        assert_eq!(ThroughputOptimalPolicy::new().name(), "throughput_optimal");
        assert_eq!(
            ThroughputOptimalWithBudgetPolicy::new().name(),
            "throughput_optimal_with_budget"
        );
    }
}
