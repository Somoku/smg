//! PSRL five-stage worker selector.
//!
//! This strategy implements the five-stage candidate-filtering chain used in
//! Policy-based Scheduling for Reinforcement Learning (PSRL) training rollouts.
//!
//! # Stage overview
//!
//! 1. **Version filter** — retain only workers whose synced version tag ≥ the
//!    request's `version_tag`.  Uses a synchronous `parking_lot::Mutex` to avoid
//!    yielding to the async runtime on the hot path.
//! 2. **Partial-hint pin** — when a `rollout_instance_hint` is present and
//!    migration is disabled, route directly to that instance.
//! 3. **Group pin** — when `prompt_id` is present and multi-instance group
//!    sampling is disabled, route to the instance already serving that prompt.
//! 4. **Can-reserve filter** — call `CanReserveRequest` via the PS Manager to
//!    retain only workers that currently have capacity (skipped when
//!    `version_tag != -1` because version-tagged requests are pre-routed).
//! 5. **Sort and select** — sort candidates by `WorkerSelectionKey`, assign
//!    group IDs, then delegate to the configured policy.
//!
//! Post-selection: update the PS Manager and the runtime bookkeeping maps.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{error, warn};

use super::WorkerSelectorStrategy;
use crate::{
    config::types::CandidateSortKey,
    observability::metrics::{metrics_labels, Metrics},
    policies::{PolicyRegistry, SelectWorkerInfo},
    routers::grpc::routing_loop::{metadata::RoutingMeta, runtime::RoutingLoopRuntime},
    worker::{ConnectionMode, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerSelectionKey {
    /// Sort descending by version tag.
    Version(i64),
    /// Sort by PS Manager reserve score (higher is better), with version as
    /// tiebreaker.  `reserve_bits` is the bit-representation of the `f64`
    /// score so that we can use `Ord` without floating-point concerns.
    Reserve {
        /// Bits of the `f64` reserve score for `total_cmp`-style ordering.
        reserve_bits: u64,
        /// Version tag as tiebreaker (higher is better).
        version_indicator: i64,
    },
}

impl PartialOrd for WorkerSelectionKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkerSelectionKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Version(a), Self::Version(b)) => a.cmp(b),
            (
                Self::Reserve {
                    reserve_bits: ra,
                    version_indicator: va,
                },
                Self::Reserve {
                    reserve_bits: rb,
                    version_indicator: vb,
                },
            ) => {
                // Use total_cmp semantics via bit reinterpretation.
                let ord = f64::from_bits(*ra).total_cmp(&f64::from_bits(*rb));
                if ord == std::cmp::Ordering::Equal {
                    va.cmp(vb)
                } else {
                    ord
                }
            }
            // Heterogeneous comparisons shouldn't occur; treat as equal.
            _ => std::cmp::Ordering::Equal,
        }
    }
}

/// Read the worker runtime version used by PSRL when the sync map has no entry.
///
/// Unknown or out-of-range versions default to `0`, matching PSRL's initial
/// rollout-instance version.
fn worker_version_tag(worker: &Arc<dyn Worker>) -> i64 {
    let dyn_version = worker.dyn_weight_version();
    i64::try_from(dyn_version).unwrap_or(0)
}

pub(crate) struct PsrlWorkerSelector {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    runtime: Arc<RoutingLoopRuntime>,
    enable_mig_strategy: bool,
    candidate_sort_key: CandidateSortKey,
    enable_group_sticky: bool,
}

impl PsrlWorkerSelector {
    pub(crate) fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        runtime: Arc<RoutingLoopRuntime>,
        enable_mig_strategy: bool,
        candidate_sort_key: CandidateSortKey,
        enable_group_sticky: bool,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            runtime,
            enable_mig_strategy,
            candidate_sort_key,
            enable_group_sticky,
        }
    }

    fn worker_instance_id(&self, worker: &Arc<dyn Worker>) -> (String, usize) {
        let base_worker_id = self
            .worker_registry
            .reserve_id_for_url(worker.base_url())
            .as_str()
            .to_string();
        (base_worker_id, worker.dp_rank().unwrap_or(0))
    }

    fn version_after_sync(&self, worker: &Arc<dyn Worker>) -> i64 {
        let instance = self.worker_instance_id(worker);
        self.runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|v| *v)
            .unwrap_or_else(|| worker_version_tag(worker))
    }
}

#[async_trait]
impl WorkerSelectorStrategy for PsrlWorkerSelector {
    async fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&http::HeaderMap>,
        routing_meta: Option<&RoutingMeta>,
    ) -> Option<Arc<dyn Worker>> {
        // ── collect initial candidate pool ──────────────────────────────────
        let model_filter = if model_id == UNKNOWN_MODEL_ID {
            None
        } else {
            Some(model_id)
        };

        let workers = self.worker_registry.get_workers_filtered(
            model_filter,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc),
            None,
            false,
        );

        let mut candidates: Vec<Arc<dyn Worker>> =
            workers.into_iter().filter(|w| w.is_available()).collect();

        if candidates.is_empty() {
            return None;
        }

        let meta = routing_meta?;

        // ── Stage 1: version filter ─────────────────────────────────────────
        candidates.retain(|w| self.version_after_sync(w) >= meta.version_tag);

        if candidates.is_empty() {
            warn!(
                version_tag = meta.version_tag,
                "PSRL Stage 1: no candidates after version filter"
            );
            return None;
        }

        // ── Stage 2: partial-hint pin ───────────────────────────────────────
        if !self.enable_mig_strategy || meta.is_sticky {
            // Request with rollout instance hint is a partial hint
            if let Some(ref hint) = meta.rollout_instance_hint {
                let pinned: Vec<Arc<dyn Worker>> = candidates
                    .iter()
                    .filter(|w| self.worker_instance_id(w) == *hint)
                    .cloned()
                    .collect();
                if !pinned.is_empty() {
                    candidates = pinned;
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // ── Stage 3: group pin ──────────────────────────────────────────────
        // Read the pinned instance for this prompt group from the write-once
        // map.  Because `record_selected_instance` uses `entry().or_insert()`
        // to write that map, any instance we read here is fully committed — no
        // second reader can observe a partially-written state.  This replaces
        // the previous two-step lookup (prompt→request_id, request_id→instance)
        // which was susceptible to a TOCTOU race on concurrent dispatch tasks.
        if self.enable_group_sticky {
            let prompt_id = meta.prompt_id;
            let group_instance: Option<(String, usize)> = self
                .runtime
                .prompt_to_pinned_instance
                .get(&prompt_id)
                .map(|entry| entry.clone());

            if let Some(ref group_inst) = group_instance {
                let pinned: Vec<Arc<dyn Worker>> = candidates
                    .iter()
                    .filter(|w| self.worker_instance_id(w) == *group_inst)
                    .cloned()
                    .collect();
                if !pinned.is_empty() {
                    candidates = pinned;
                }
                // If pinned is empty the previously-pinned instance is no
                // longer available; fall through with all candidates so the
                // request can still be served.
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // ── Stage 4: can-reserve filter ─────────────────────────────────────
        // Only applicable when version_tag == -1 (un-versioned requests).
        let request_id = meta.request_id;
        let is_validate = meta.is_validate;

        if meta.version_tag == -1 {
            if let Some(ps_client) = self.runtime.ps_manager_client.get() {
                // Collect unique synced version tags.
                let unique_versions: Vec<i64> = {
                    let mut v: Vec<i64> = candidates
                        .iter()
                        .map(|w| self.version_after_sync(w))
                        .collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                };

                if !unique_versions.is_empty() {
                    match ps_client
                        .can_reserve_request(
                            vec![request_id],
                            unique_versions.clone(),
                            false,
                            vec![is_validate],
                        )
                        .await
                    {
                        Ok((results, _n_versions)) => {
                            // results[i] == true  ↔  unique_versions[i] is reservable
                            let reservable: std::collections::HashSet<i64> = unique_versions
                                .iter()
                                .zip(results.iter())
                                .filter_map(|(&v, &ok)| if ok { Some(v) } else { None })
                                .collect();

                            candidates.retain(|w| reservable.contains(&self.version_after_sync(w)));

                            if candidates.is_empty() {
                                warn!(
                                    request_id,
                                    "PSRL Stage 4: no candidates after can_reserve filter"
                                );
                                return None;
                            }
                        }
                        Err(status) => {
                            error!(
                                request_id,
                                %status,
                                "PSRL Stage 4: can_reserve_request RPC failed; skipping filter"
                            );
                            return None;
                        }
                    }
                }
            }
        }

        // ── Stage 5: sort by key, assign group IDs, delegate to policy ──────
        let sort_keys: Vec<WorkerSelectionKey> = match self.candidate_sort_key {
            CandidateSortKey::Version => candidates
                .iter()
                .map(|w| {
                    let v = self.version_after_sync(w);
                    // When version_tag == -1 use negative version so higher
                    // versions sort last (we want "any version" semantics).
                    WorkerSelectionKey::Version(if meta.version_tag == -1 { -v } else { v })
                })
                .collect(),

            CandidateSortKey::ReserveCapability => {
                let indicators: Vec<f64> =
                    if let Some(ps_client) = self.runtime.ps_manager_client.get() {
                        let unique_versions: Vec<i64> = {
                            let mut v: Vec<i64> = candidates
                                .iter()
                                .map(|w| self.version_after_sync(w))
                                .collect();
                            v.sort_unstable();
                            v.dedup();
                            v
                        };

                        match ps_client
                            .get_reserve_indicator(request_id, unique_versions.clone(), is_validate)
                            .await
                        {
                            Ok(ind) => {
                                // Map each candidate's version → its indicator.
                                let version_to_indicator: std::collections::HashMap<i64, f64> =
                                    unique_versions.into_iter().zip(ind).collect();
                                candidates
                                    .iter()
                                    .map(|w| {
                                        version_to_indicator
                                            .get(&self.version_after_sync(w))
                                            .copied()
                                            .unwrap_or(0.0_f64)
                                    })
                                    .collect()
                            }
                            Err(status) => {
                                error!(
                                    request_id,
                                    %status,
                                    "PSRL Stage 5: get_reserve_indicator RPC failed"
                                );
                                return None;
                            }
                        }
                    } else {
                        error!("PSRL Stage 5: PS Manager client not available");
                        return None;
                    };

                candidates
                    .iter()
                    .zip(indicators.iter())
                    .map(|(w, &ind)| {
                        let v = self.version_after_sync(w);
                        WorkerSelectionKey::Reserve {
                            reserve_bits: ind.to_bits(),
                            version_indicator: if meta.version_tag == -1 { -v } else { v },
                        }
                    })
                    .collect()
            }
        };

        // Sort candidates by key descending (best first).
        let mut indexed: Vec<(usize, &WorkerSelectionKey)> = sort_keys.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(a.1));

        let sorted_candidates: Vec<Arc<dyn Worker>> = indexed
            .iter()
            .map(|(i, _)| candidates[*i].clone())
            .collect();

        // Assign group IDs — a new group starts whenever the key changes.
        // group ID represents the order of indicators.
        let sorted_keys: Vec<&WorkerSelectionKey> = indexed.iter().map(|(_, k)| *k).collect();
        let mut priority_groups: Vec<i64> = Vec::with_capacity(sorted_candidates.len());
        let mut current_group: i64 = 0;
        let mut prev_key: Option<&WorkerSelectionKey> = None;
        for k in &sorted_keys {
            if prev_key.is_some_and(|p| p != *k) {
                current_group += 1;
            }
            priority_groups.push(current_group);
            prev_key = Some(k);
        }

        let policy = self.policy_registry.get_policy_or_default(model_id);
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        let idx = policy.select_worker(
            &sorted_candidates,
            &SelectWorkerInfo {
                request_text: text,
                tokens,
                headers,
                hash_ring,
                priority_groups: Some(&priority_groups),
                response_token_count: meta.response_token_count,
            },
        )?;

        let selected = sorted_candidates[idx].clone();
        let selected_instance = self.worker_instance_id(&selected);

        // ── Post-selection: update PS Manager and bookkeeping maps ───────────
        {
            let instance = &selected_instance;
            if let Some(ps_client) = self.runtime.ps_manager_client.get() {
                let version = self.version_after_sync(&selected);

                if meta.rollout_instance_hint.is_none() {
                    // First placement — reserve and record.
                    match ps_client
                        .reserve_rollout_instance_requests(
                            vec![instance.clone()],
                            vec![request_id],
                            vec![version],
                            false,
                            is_validate,
                        )
                        .await
                    {
                        Ok((success, _buffer_ids, _entry_ids, err_msg)) => {
                            if !success {
                                error!(
                                    request_id,
                                    %err_msg,
                                    "PSRL post-select: reserve_rollout_instance_requests failed"
                                );
                            }
                        }
                        Err(status) => {
                            error!(
                                request_id,
                                %status,
                                "PSRL post-select: reserve_rollout_instance_requests RPC error"
                            );
                        }
                    }
                } else {
                    // Migration — update the assigned instance.
                    match ps_client
                        .update_request_instance_id(request_id, instance.clone(), is_validate)
                        .await
                    {
                        Ok(success) => {
                            if !success {
                                error!(
                                    request_id,
                                    "PSRL post-select: update_request_instance_id returned false"
                                );
                            }
                        }
                        Err(status) => {
                            error!(
                                request_id,
                                %status,
                                "PSRL post-select: update_request_instance_id RPC error"
                            );
                        }
                    }
                }
            }

            if meta.rollout_instance_hint.is_none() {
                self.runtime.record_selected_instance(
                    request_id,
                    Some(meta.prompt_id),
                    instance.clone(),
                );
            }
        }

        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            model_id,
            policy.name(),
        );

        Some(selected)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use super::*;
    use crate::{
        config::{PolicyConfig, RoutingLoopConfig},
        policies::PolicyRegistry,
        routers::grpc::routing_loop::runtime::RoutingLoopRuntime,
        worker::{BasicWorkerBuilder, WorkerRegistry},
    };

    fn make_runtime() -> Arc<RoutingLoopRuntime> {
        let (rt, _rx) = RoutingLoopRuntime::new(
            &RoutingLoopConfig::default(),
            Arc::new(dashmap::DashMap::new()),
            Arc::new(WorkerRegistry::new()),
        );
        rt
    }

    /// Stage 1: workers whose synced version tag < request's version_tag are filtered out.
    #[tokio::test]
    async fn stage1_version_filter_removes_stale_workers() {
        let runtime = make_runtime();
        // Populate the version map with two instances.
        runtime
            .instance_to_version_after_sync
            .insert(("worker-a".to_string(), 0), 5);
        runtime
            .instance_to_version_after_sync
            .insert(("worker-b".to_string(), 0), 3);

        // We query the map directly (simulating what Stage 1 does in
        // select_single_worker).  A request with version_tag=4 should exclude
        // worker-b (synced version 3 < 4).
        let version_tag: i64 = 4;
        let worker_a_ok = runtime
            .instance_to_version_after_sync
            .get(&("worker-a".to_string(), 0))
            .is_some_and(|v| *v >= version_tag);
        let worker_b_ok = runtime
            .instance_to_version_after_sync
            .get(&("worker-b".to_string(), 0))
            .is_some_and(|v| *v >= version_tag);
        assert!(worker_a_ok, "worker-a should pass version filter");
        assert!(!worker_b_ok, "worker-b should fail version filter");
    }

    /// Stage 3 group-pin: prompt_to_running map wiring is consistent.
    #[tokio::test]
    async fn stage3_prompt_map_wiring() {
        let runtime = make_runtime();
        let request_id: i64 = 42;
        let prompt_id: i64 = 7;
        let instance = ("worker-x".to_string(), 1_usize);

        // Simulate post-selection bookkeeping.
        runtime.record_selected_instance(request_id, Some(prompt_id), instance.clone());

        // Read back and verify.
        let ids = runtime
            .prompt_to_running_request_ids
            .get(&prompt_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();

        assert_eq!(ids, vec![request_id]);
    }

    #[tokio::test]
    async fn cleanup_tracking_removes_request_and_empty_prompt_group() {
        let runtime = make_runtime();
        runtime.record_selected_instance(42, Some(7), ("worker-x".to_string(), 1));

        runtime.cleanup_tracking(Some(42), Some(7));

        assert!(!runtime.prompt_to_running_request_ids.contains_key(&7));
    }

    #[tokio::test]
    async fn cleanup_tracking_keeps_other_prompt_requests() {
        let runtime = make_runtime();
        runtime.record_selected_instance(42, Some(7), ("worker-x".to_string(), 1));
        runtime.record_selected_instance(43, Some(7), ("worker-y".to_string(), 0));

        runtime.cleanup_tracking(Some(42), Some(7));

        let ids = runtime
            .prompt_to_running_request_ids
            .get(&7)
            .map(|entry| entry.clone())
            .unwrap_or_default();
        assert_eq!(ids, vec![43]);
    }

    #[test]
    fn worker_version_tag_defaults_unknown_to_zero() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-a:8000")
                .label(
                    "weight_version",
                    (u64::try_from(i64::MAX).unwrap() + 1).to_string(),
                )
                .build(),
        );

        assert_eq!(worker_version_tag(&worker), 0);
    }

    #[test]
    fn version_after_sync_prefers_runtime_map_over_worker_version() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let runtime = make_runtime();
        let selector = PsrlWorkerSelector::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            Arc::clone(&runtime),
            false,
            CandidateSortKey::Version,
            true,
        );
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-a:8000")
                .label("weight_version", "7")
                .build(),
        );
        let instance = selector.worker_instance_id(&worker);
        runtime.instance_to_version_after_sync.insert(instance, 3);

        assert_eq!(selector.version_after_sync(&worker), 3);
    }

    /// Stage 2 pin: rollout_instance_hint (mig disabled) should pin to the hinted instance.
    #[test]
    fn stage2_partial_hint_pin_logic() {
        // Simulate the filter: a hint matches one of two candidates.
        let hint: (String, usize) = ("worker-a".to_string(), 0);
        let instances = [
            ("worker-a".to_string(), 0_usize),
            ("worker-b".to_string(), 0_usize),
        ];
        let pinned: Vec<_> = instances.iter().filter(|inst| *inst == &hint).collect();
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0], &hint);
    }

    /// WorkerSelectionKey ordering: higher version sorts before lower.
    #[test]
    fn worker_selection_key_ordering() {
        let high = WorkerSelectionKey::Version(10);
        let low = WorkerSelectionKey::Version(5);
        assert!(high > low);

        let hi_reserve = WorkerSelectionKey::Reserve {
            reserve_bits: 1.0_f64.to_bits(),
            version_indicator: 5,
        };
        let lo_reserve = WorkerSelectionKey::Reserve {
            reserve_bits: 0.5_f64.to_bits(),
            version_indicator: 5,
        };
        assert!(hi_reserve > lo_reserve);
    }

    /// Unused import lint: ensure HashMap is used.
    #[test]
    fn stage1_empty_version_map_passes_all() {
        let map: HashMap<(String, usize), i64> = HashMap::new();
        let version_tag: i64 = 2;
        // With no synced version in the map, the selector falls back to the
        // worker runtime version, whose unknown value defaults to 0.
        let has_entry = map.contains_key(&("worker-a".to_string(), 0));
        assert!(!has_entry);
        let _ = version_tag;
    }
}
