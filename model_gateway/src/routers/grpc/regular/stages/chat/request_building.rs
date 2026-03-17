//! Chat request building stage: Build proto GenerateRequest for chat requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext},
        multimodal::assemble_multimodal_data,
        proto_wrapper::ProtoRequest,
    },
};

/// Chat request building stage
///
/// Extracts chat-specific request building logic from the old unified RequestBuildingStage.
pub(crate) struct ChatRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl ChatRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for ChatRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        // Get client for building request (use prefill client if PD mode)
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Take multimodal_intermediate from preparation on the first call (if present).
        // Subsequent loopback iterations see None here, which is correct: the multimodal
        // content is part of the initial prompt and does not change across loopback iterations.
        // MultimodalIntermediate does not implement Clone (it holds preprocessed image tensors),
        // so we take it out of preparation once. This exclusive borrow must complete before
        // the shared borrows of `prep` below begin.
        let multimodal_data = ctx
            .state
            .preparation
            .as_mut()
            .and_then(|p| p.processed_messages.as_mut())
            .and_then(|m| m.multimodal_intermediate.take())
            .map(|intermediate| assemble_multimodal_data(intermediate, builder_client));

        // Issue 2: Use as_ref() instead of take() so that preparation survives
        // partial-rollout loopback iterations. On the first abort the preparation was
        // previously consumed by take(), causing a hard failure on the second loopback.
        // Now preparation persists across all loopback iterations; clonable fields are cloned.
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let chat_request = ctx.chat_request_arc();

        // Build chat request
        let request_id = format!("chatcmpl-{}", Uuid::now_v7());
        let body_ref = prep.filtered_request.as_ref().unwrap_or(&chat_request);

        let processed_messages = prep.processed_messages.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "processed_messages not set in preparation state"
            );
            error::internal_error(
                "processed_messages_missing",
                "processed_messages not set - this is a bug in the pipeline",
            )
        })?;

        // Issue 2: Clone clonable preparation fields to leave preparation intact for future
        // loopback iterations. processed_text and token_ids must be cloned; tool_constraints
        // can be cloned as it is Option<(String, String)>.
        let mut proto_request = builder_client
            .build_chat_request(
                request_id,
                body_ref,
                processed_messages.text.clone(),
                prep.token_ids.clone(),
                multimodal_data,
                prep.tool_constraints.clone(),
            )
            .map_err(|e| {
                error!(function = "ChatRequestBuildingStage::execute", error = %e, "Failed to build generate request");
                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {e}"))
            })?;

        // PR 18 (Gap 5): apply accumulated loopback token_ids/max-token budget
        // at proto-request level so Chat requests resume after abort.
        // Each call rebuilds the proto from scratch, so all accumulated token_ids are
        // injected unconditionally into the freshly built proto.
        helpers::maybe_apply_partial_rollout_loopback(
            &mut proto_request,
            ctx.state.partial_rollout_state.as_ref(),
        );

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        ctx.state.proto_request = Some(ProtoRequest::Generate(proto_request));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}
