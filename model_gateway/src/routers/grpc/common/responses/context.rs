//! Shared context for /v1/responses endpoint handlers
//!
//! This context is used by both regular and harmony response implementations.

use std::sync::Arc;

use smg_data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};
use smg_mcp::McpOrchestrator;

use crate::routers::grpc::{
    context::SharedComponents, pipeline::RequestPipeline,
    pipeline_routing_loop::RoutingLoopPipeline,
};

/// Context for /v1/responses endpoint
///
/// Used by both regular and harmony implementations.
/// All fields are Arc/shared references, so cloning this context is cheap.
#[derive(Clone)]
pub(crate) struct ResponsesContext {
    /// Chat pipeline for executing requests
    pub pipeline: Arc<RequestPipeline>,

    /// Shared components (tokenizer, parsers)
    pub components: Arc<SharedComponents>,

    /// Response storage backend
    pub response_storage: Arc<dyn ResponseStorage>,

    /// Conversation storage backend
    pub conversation_storage: Arc<dyn ConversationStorage>,

    /// Conversation item storage backend
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,

    /// MCP orchestrator for tool support
    pub mcp_orchestrator: Arc<McpOrchestrator>,

    // PR 17 (Gap 4): When Some, regular /v1/responses chat executions are dispatched
    // through the routing loop for PSRL worker selection and PS Manager tracking.
    // Harmony responses use their own pipeline and bypass the routing loop for now
    // (see TODO in router.rs::route_responses_impl).
    /// Routing-loop pipeline for PSRL-aware chat dispatch (regular responses only).
    /// `None` when `enable_routing_loop = false`.
    // PR-A (Notes 1+2): store as Arc so responses context and router share one pipeline instance.
    pub routing_loop_pipeline: Option<Arc<RoutingLoopPipeline>>,
}

impl ResponsesContext {
    /// Create a new responses context
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        response_storage: Arc<dyn ResponseStorage>,
        conversation_storage: Arc<dyn ConversationStorage>,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
        mcp_orchestrator: Arc<McpOrchestrator>,
    ) -> Self {
        Self {
            pipeline,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_orchestrator,
            routing_loop_pipeline: None,
        }
    }

    /// Set the routing-loop pipeline (builder-style).
    ///
    /// Called by `GrpcRouter::new()` when `enable_routing_loop = true` to wire
    /// PSRL-aware dispatch into regular (non-Harmony) /v1/responses handling.
    pub fn with_routing_loop(mut self, rl_pipeline: Arc<RoutingLoopPipeline>) -> Self {
        self.routing_loop_pipeline = Some(rl_pipeline);
        self
    }
}
