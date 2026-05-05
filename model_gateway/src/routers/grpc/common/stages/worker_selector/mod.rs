//! Pluggable worker selection strategies.
//!
//! # Registration
//!
//! Strategies are identified by [`WorkerSelectionStrategy`] and constructed via
//! [`build_strategy`].  Two strategies are built in:
//!
//! * [`WorkerSelectionStrategy::Naive`] (default) — the existing policy-based
//!   single-worker selection.
//! * [`WorkerSelectionStrategy::Psrl`] — the five-stage PSRL candidate filter chain.
//!
//! # Extension
//!
//! Add a new strategy by:
//! 1. Implementing [`WorkerSelectorStrategy`] in a new sub-module.
//! 2. Adding a match arm to [`build_strategy`] (optionally gated by a feature).

use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    config::types::{PsrlConfig, WorkerSelectionStrategy},
    policies::PolicyRegistry,
    routers::grpc::routing_loop::{metadata::RoutingMeta, runtime::RoutingLoopRuntime},
    worker::{Worker, WorkerRegistry},
};

pub(crate) mod naive;
pub(crate) mod psrl;

pub(crate) use naive::NaiveWorkerSelector;
pub(crate) use psrl::PsrlWorkerSelector;

/// A pluggable worker selection strategy for Regular-mode requests.
///
/// Implementations must be `Send + Sync` so they can be shared across async tasks.
#[async_trait]
pub(crate) trait WorkerSelectorStrategy: Send + Sync {
    /// Select a single worker for a Regular-mode request.
    ///
    /// Returns `None` when no worker is available or all candidates were
    /// filtered out.
    async fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&http::HeaderMap>,
        routing_meta: Option<&RoutingMeta>,
    ) -> Option<Arc<dyn Worker>>;
}

pub(crate) fn build_strategy(
    strategy: WorkerSelectionStrategy,
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    runtime: Option<Arc<RoutingLoopRuntime>>,
    config: &PsrlConfig,
) -> Result<Arc<dyn WorkerSelectorStrategy>, String> {
    match strategy {
        WorkerSelectionStrategy::Naive => Ok(Arc::new(NaiveWorkerSelector::new(
            worker_registry,
            policy_registry,
        ))),

        WorkerSelectionStrategy::Psrl => {
            let rt = runtime.ok_or_else(|| {
                "psrl strategy requires the routing loop to be enabled".to_string()
            })?;
            Ok(Arc::new(PsrlWorkerSelector::new(
                worker_registry,
                policy_registry,
                rt,
                config.enable_mig_strategy,
                config.candidate_sort_key,
                config.enable_group_sticky_routing,
            )))
        }
    }
}
