//! gRPC router implementations

use openai_protocol::common::StringOrArray;

pub mod client; // Used by core/
pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod harmony;
pub(crate) mod multimodal;
pub(crate) mod pd_router; // Used by routers/factory
pub(crate) mod pipeline;
// Refactor Notes 1+2+3: RoutingLoopPipeline — two-stage routing-loop dispatch.
pub(crate) mod pipeline_routing_loop;
pub(crate) mod proto_wrapper;
pub(crate) mod regular;
// PR 10 §10.2: Routing loop task body.
pub(crate) mod router;
pub(crate) mod routing_loop; // Used by routers/factory
                             // PR 11 §11.1: PSRL worker selection logic.
pub(crate) mod worker_selection;
// PR 12 §12.2: Partial rollout protocol (drain stream, extract state, loopback mutation).
pub(crate) mod partial_rollout;
pub mod utils; // Used by routers/http and bindings/golang

// Re-export for convenience
pub use proto_wrapper::{MultimodalData, TensorBytes};

/// Processed chat messages ready for gRPC generation
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    /// Preprocessed multimodal intermediate (deferred assembly).
    /// Populated during preparation when multimodal content is detected.
    /// Assembled into backend-specific `MultimodalData` in request_building.
    pub(crate) multimodal_intermediate: Option<multimodal::MultimodalIntermediate>,
    pub stop_sequences: Option<StringOrArray>,
}
