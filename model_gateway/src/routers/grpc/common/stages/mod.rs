//! Common pipeline stages shared across all endpoints and model types
//!
//! These stages are endpoint-agnostic and model-agnostic:
//! - Worker selection
//! - Client acquisition
//! - Dispatch metadata generation
//! - Request execution

use async_trait::async_trait;
use axum::response::Response;

use crate::routers::grpc::context::RequestContext;

/// Phase a pipeline stage belongs to.
///
/// Used by `execute_through_execution` / `execute_remaining_stages` to split
/// the pipeline at the execution boundary for partial-rollout loopback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StagePhase {
    /// Stage 0 — tokenisation, prompt pre-processing.
    Preparation,
    /// Stages 1-5 — worker selection, client acquisition, request building,
    /// dispatch metadata, request execution (the "hot path").
    Execution,
    /// Stage 6 — response serialisation, streaming, metrics emission.
    PostExecution,
}

/// Trait for pipeline stages that process requests
#[async_trait]
pub trait PipelineStage: Send + Sync {
    /// Execute this stage, mutating the context
    ///
    /// Returns:
    /// - `Ok(None)` - Continue to next stage
    /// - `Ok(Some(response))` - Pipeline complete, return this response (e.g., streaming)
    /// - `Err(response)` - Error occurred, return this error response
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response>;

    /// Stage name for logging
    fn name(&self) -> &'static str;

    /// Phase this stage belongs to.
    ///
    /// The default implementation returns [`StagePhase::Execution`], which is
    /// correct for the common middle stages (worker selection through request
    /// execution).  Only preparation and response-processing stages need to
    /// override this.
    fn phase(&self) -> StagePhase {
        StagePhase::Execution
    }
}

mod client_acquisition;
mod dispatch_metadata;
pub(crate) mod helpers;
mod request_execution;
mod worker_selection;
pub(crate) mod worker_selector;

// Export stage implementations
pub(crate) use client_acquisition::ClientAcquisitionStage;
pub(crate) use dispatch_metadata::DispatchMetadataStage;
pub(crate) use request_execution::{ExecutionMode, RequestExecutionStage};
pub(crate) use worker_selection::WorkerSelectionStage;
pub(crate) use worker_selector::{build_strategy, WorkerSelectorStrategy};
