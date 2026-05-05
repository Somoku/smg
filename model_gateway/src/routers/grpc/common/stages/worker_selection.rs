//! Worker selection stage: Select appropriate worker(s) based on routing mode

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::{error, warn};

use super::{worker_selector::WorkerSelectorStrategy, PipelineStage};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    policies::{PolicyRegistry, SelectWorkerInfo},
    routers::{
        error,
        grpc::{
            context::{RequestContext, WorkerSelection},
            routing_loop::metadata::parse_routing_request_meta_from_context,
        },
    },
    worker::{ConnectionMode, RuntimeType, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
};

/// Result type for PD worker pair selection: (prefill, decode, runtime_type)
type PdWorkerPair = (Arc<dyn Worker>, Arc<dyn Worker>, RuntimeType);

/// Internal representation of the two selection modes.
enum WorkerSelectionMode {
    /// Regular (single-worker) mode: delegates to a pluggable strategy.
    Regular {
        strategy: Arc<dyn WorkerSelectorStrategy>,
    },
    /// PD (prefill-decode) mode: always uses policy-based naive selection.
    PrefillDecode {
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    },
}

/// Worker selection stage: selects appropriate worker(s) for the current routing mode.
pub(crate) struct WorkerSelectionStage {
    inner: WorkerSelectionMode,
}

impl WorkerSelectionStage {
    /// Construct a Regular-mode stage that delegates to the given strategy.
    pub fn new_regular(strategy: Arc<dyn WorkerSelectorStrategy>) -> Self {
        Self {
            inner: WorkerSelectionMode::Regular { strategy },
        }
    }

    /// Construct a PD-mode stage using policy-based naive selection.
    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self {
            inner: WorkerSelectionMode::PrefillDecode {
                worker_registry,
                policy_registry,
            },
        }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        let text = prep.routing_text();

        // Get tokens for PrefixHash policy support
        let ids = prep.token_ids();
        let tokens = if ids.is_empty() { None } else { Some(ids) };

        let headers = ctx.input.headers.as_ref();

        let model_id = ctx.input.model_id.as_str();
        let workers = match &self.inner {
            WorkerSelectionMode::Regular { strategy } => {
                let routing_meta = parse_routing_request_meta_from_context(ctx);

                match strategy
                    .select_single_worker(model_id, text, tokens, headers, routing_meta.as_ref())
                    .await
                {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "Regular",
                            model_id = %model_id,
                            "No available workers for model"
                        );
                        return Err(error::model_not_found(model_id));
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode {
                worker_registry,
                policy_registry,
            } => {
                match select_pd_pair(
                    model_id,
                    text,
                    tokens,
                    headers,
                    worker_registry,
                    policy_registry,
                ) {
                    Some((prefill, decode, runtime_type)) => WorkerSelection::Dual {
                        prefill,
                        decode,
                        runtime_type,
                    },
                    None => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            mode = "PrefillDecode",
                            model_id = %model_id,
                            "No available PD worker pairs for model"
                        );
                        return Err(error::model_not_found(model_id));
                    }
                }
            }
        };

        ctx.state.workers = Some(workers);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }
}

fn select_pd_pair(
    model_id: &str,
    text: Option<&str>,
    tokens: Option<&[u32]>,
    headers: Option<&http::HeaderMap>,
    worker_registry: &Arc<WorkerRegistry>,
    policy_registry: &Arc<PolicyRegistry>,
) -> Option<PdWorkerPair> {
    // Treat "unknown" model as wildcard (match any worker)
    let model_filter = if model_id == UNKNOWN_MODEL_ID {
        None
    } else {
        Some(model_id)
    };

    let all_workers = worker_registry.get_workers_filtered(
        model_filter,
        None,
        Some(ConnectionMode::Grpc), // Match any gRPC worker
        None,                       // any runtime type
        false,
    );

    let (all_prefill, all_decode): (Vec<_>, Vec<_>) =
        all_workers
            .into_iter()
            .fold((Vec::new(), Vec::new()), |mut acc, w| {
                if w.is_available() {
                    match w.metadata().spec.worker_type {
                        WorkerType::Prefill => acc.0.push(w),
                        WorkerType::Decode => acc.1.push(w),
                        WorkerType::Regular => {}
                    }
                }
                acc
            });

    if all_prefill.is_empty() {
        warn!("No available prefill workers");
        return None;
    }

    if all_decode.is_empty() {
        warn!("No available decode workers");
        return None;
    }

    // Determine the runtime type from prefill workers.
    // All workers in a PD pair must use the same runtime.
    let first_runtime = all_prefill.first()?.metadata().spec.runtime_type;

    // Check for mixed runtimes in both prefill and decode pools
    let prefill_mixed = all_prefill
        .iter()
        .skip(1)
        .any(|w| w.metadata().spec.runtime_type != first_runtime);
    let decode_mixed = all_decode
        .iter()
        .any(|w| w.metadata().spec.runtime_type != first_runtime);

    if prefill_mixed || decode_mixed {
        warn!(
            "Mixed runtime types in PD workers (prefill_mixed={}, decode_mixed={}). Using {:?}.",
            prefill_mixed, decode_mixed, first_runtime
        );
    }

    let target_runtime = first_runtime;

    // Filter both pools to the target runtime
    let available_prefill: Vec<_> = all_prefill
        .into_iter()
        .filter(|w| w.metadata().spec.runtime_type == target_runtime)
        .collect();
    let available_decode: Vec<_> = all_decode
        .into_iter()
        .filter(|w| w.metadata().spec.runtime_type == target_runtime)
        .collect();

    if available_prefill.is_empty() || available_decode.is_empty() {
        warn!("No available PD pair for runtime {:?}", target_runtime);
        return None;
    }

    // Select using policies (PD mode always uses naive policy-based selection)
    let policy = policy_registry.get_policy_or_default(model_id);

    // Get cached hash ring for consistent hashing (O(log n) lookup)
    let hash_ring = worker_registry.get_hash_ring(model_id);

    let info = SelectWorkerInfo {
        request_text: text,
        tokens,
        headers,
        hash_ring,
        response_token_count: None,
        priority_groups: None,
    };
    let prefill_idx = policy.select_worker(&available_prefill, &info)?;
    let decode_idx = policy.select_worker(&available_decode, &info)?;

    let policy_name = policy.name();

    // Record worker selection metrics for both prefill and decode
    Metrics::record_worker_selection(
        metrics_labels::WORKER_PREFILL,
        metrics_labels::CONNECTION_GRPC,
        model_id,
        policy_name,
    );
    Metrics::record_worker_selection(
        metrics_labels::WORKER_DECODE,
        metrics_labels::CONNECTION_GRPC,
        model_id,
        policy_name,
    );

    Some((
        available_prefill[prefill_idx].clone(),
        available_decode[decode_idx].clone(),
        target_runtime,
    ))
}
