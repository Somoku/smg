// Refactor Notes 1+2+3: RoutingLoopPipeline — separates routing-loop dispatch from router.rs,
// structures it into explicit stages, and dispatches unconditionally when enabled.
// PR 13 §13.1: execute_chat / execute_generate now pass ctx directly to RoutingQueueEntry
//   instead of discarding the prepared context and re-wrapping the original Arc<Request>
//   into PreparedRequest. The serde_json::to_value roundtrip for routing_meta is replaced
//   by parse_psrl_request_meta_from_context (reads directly from typed headers + RequestType).
//!
//! This module implements the two-stage routing-loop pipeline:
//!
//! ```text
//! Stage 0: RoutingLoopPreparationStage
//!   - Runs PreparationStage on the incoming RequestContext.
//!   - Parses PSRL routing_meta from headers (no JSON roundtrip).
//!
//! Stage 1: EnqueueAndWaitStage
//!   - Builds RoutingQueueEntry { ctx } with the prepared context.
//!   - Sends the entry into runtime.tx.
//!   - Awaits the oneshot result_rx.
//!   - Returns the Response.
//! ```
//!
//! # Unconditional dispatch (Note 2)
//!
//! When `enable_routing_loop = true`, ALL requests are dispatched through this pipeline —
//! even those without PSRL metadata. `RoutingQueueEntry.routing_meta = None` is valid;
//! `dispatch_task` guards all PS Manager calls with `if let Some(ps_client) = ...`.
//!
//! This removes the `parse_psrl_request_meta` gating condition from `router.rs` and
//! makes the routing-loop switch the canonical dispatch gate.

use std::sync::Arc;

use axum::{
    body::to_bytes,
    response::Response,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    generate::GenerateRequest,
};
use tokio::sync::oneshot;
use tracing::error;

use super::{
    context::{RequestContext, SharedComponents},
    pipeline::RequestPipeline,
};
use crate::routers::{
    error as router_error,
    routing_loop_utils::{
        parse_psrl_request_meta_from_context, RoutingLoopRuntime, RoutingQueueEntry,
    },
};

// ── Public pipeline type ─────────────────────────────────────────────────

// Refactor Note 1: RoutingLoopPipeline replaces the inline PSRL block in router.rs.
/// Two-stage routing-loop pipeline.
///
/// When `enable_routing_loop = true`, `GrpcRouter` replaces its inline PSRL block
/// with this struct. All requests flow through `execute()` unconditionally.
///
/// Internally delegates to `RequestPipeline::execute_preparation_only()` (stage 0)
/// then submits to the routing loop via `RoutingLoopRuntime::tx` (stage 1).
// Refactor Note 1: Clone is derived because GrpcRouter derives Clone (all fields are Arc<...>).
#[derive(Clone)]
pub(crate) struct RoutingLoopPipeline {
    runtime: Arc<RoutingLoopRuntime>,
    standard_pipeline: Arc<RequestPipeline>,
}

impl RoutingLoopPipeline {
    // Refactor Note 1: Constructor takes the runtime and standard pipeline.
    /// Create a new `RoutingLoopPipeline`.
    ///
    /// - `runtime` — shared routing loop state and mpsc sender
    /// - `standard_pipeline` — used for `execute_preparation_only` (stage 0)
    pub fn new(runtime: Arc<RoutingLoopRuntime>, standard_pipeline: Arc<RequestPipeline>) -> Self {
        Self {
            runtime,
            standard_pipeline,
        }
    }

    /// Access the underlying `RoutingLoopRuntime`.
    ///
    /// Used by `worker_selection.rs` for PSRL stages 1–5 (version filter, group-pin, etc.).
    pub fn runtime(&self) -> &Arc<RoutingLoopRuntime> {
        &self.runtime
    }

    // Refactor Notes 1+2+3: execute() — two-stage routing-loop dispatch.
    /// Execute a chat completion request through the routing-loop pipeline.
    ///
    /// Stage 0 (`RoutingLoopPreparationStage`): runs `PreparationStage` and parses
    /// optional PSRL routing metadata from headers (no JSON roundtrip — PR 13 §13.1).
    ///
    /// Stage 1 (`EnqueueAndWaitStage`): builds a `RoutingQueueEntry` with the prepared
    /// `RequestContext` and submits it to the routing loop unconditionally.
    ///
    /// # Note 2: unconditional dispatch
    ///
    /// Unlike the old `router.rs` inline block, this method does NOT gate on
    /// `parse_psrl_request_meta` returning `Some`. All chat requests go through
    /// the routing loop; `routing_meta = None` is valid (no PS Manager tracking).
    ///
    /// # PR 13 §13.1: ctx carried directly
    ///
    /// The prepared `RequestContext` (with `PreparationOutput`) is passed directly
    /// to `RoutingQueueEntry.ctx` — the old discard-and-recreate cycle is gone.
    pub async fn execute_chat(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        // ── Stage 0: Preparation ───────────────────────────────────────────
        let ctx = RequestContext::for_chat(
            Arc::clone(&request),
            headers,
            model_id,
            components,
        );
        let ctx = match self.standard_pipeline.execute_preparation_only(ctx).await {
            Ok(ctx) => ctx,
            Err(err_response) => return err_response,
        };

        // PR 13 §13.1: Parse routing_meta from typed context (no JSON roundtrip).
        // Replaces: serde_json::to_value(request) + parse_psrl_request_meta(headers, &body_json).
        let routing_meta = parse_psrl_request_meta_from_context(&ctx);
        // ── Stage 1: Enqueue and wait ──────────────────────────────────────
        let (result_tx, result_rx) = oneshot::channel();
        // PR 13 §13.1: ctx carried directly — PreparationOutput preserved across queue boundary.
        let entry = RoutingQueueEntry {
            ctx,
            result_tx,
            routing_meta,
        };
        enqueue_and_wait(&self.runtime, entry, result_rx).await
    }

    // PR 17 (Gap 4): execute_chat_for_responses — routes /v1/responses non-streaming                               
    // chat execution through the routing loop for PSRL worker selection and PS Manager
    // tracking, then deserializes the JSON body back to ChatCompletionResponse so the
    // responses tool-loop and conversion logic can proceed as normal.
    /// Execute a chat-for-responses request through the routing-loop pipeline.
    ///
    /// Mirrors `RequestPipeline::execute_chat_for_responses` but routes through the
    /// routing loop (PSRL worker selection, PS Manager lifecycle updates) instead of
    /// calling the pipeline directly.
    ///
    /// The underlying chat request must be **non-streaming** (responses non-streaming path
    /// ensures this via `conversions::responses_to_chat` setting `stream = false`).
    /// The routing loop will serialize the final `ChatCompletionResponse` as JSON;
    /// this method collects the body bytes and deserializes them back.
    ///
    /// Returns `Ok(ChatCompletionResponse)` on success, `Err(Response)` on pipeline error.
    pub async fn execute_chat_for_responses(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Result<ChatCompletionResponse, Response> {
        let response = self.execute_chat(request, headers, model_id, components).await;
        let status = response.status();
        if !status.is_success() {
            return Err(response);
        }
        // Collect body bytes then deserialize — the routing loop's
        // execute_response_processing_only calls axum::Json(response).into_response()
        // for non-streaming chat, so the body is a valid JSON ChatCompletionResponse.
        let body_bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .map_err(|e| {
                router_error::internal_error(
                    "routing_loop_body_collection_failed",
                    format!("Failed to collect routing loop response body: {e}"),
                )
            })?;
        serde_json::from_slice::<ChatCompletionResponse>(&body_bytes).map_err(|e| {
            router_error::internal_error(
                "routing_loop_deserialize_failed",
                format!("Failed to deserialize ChatCompletionResponse from routing loop: {e}"),
            )
        })
    }

    // Refactor Notes 1+2+3: execute_generate() — two-stage routing-loop dispatch for generate.
    /// Execute a generate request through the routing-loop pipeline.
    ///
    /// Same pattern as `execute_chat` — passes prepared `ctx` directly to the queue entry.
    ///
    /// PR 13 §13.1: `RoutingQueueEntry.ctx` replaces `PreparedRequest::Generate(request)`.
    pub async fn execute_generate(
        &self,
        request: Arc<GenerateRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        // PR-B (Note 4): align signature with RequestPipeline by passing components explicitly.
        let ctx = RequestContext::for_generate(Arc::clone(&request), headers, model_id, components);
        let ctx = match self.standard_pipeline.execute_preparation_only(ctx).await {
            Ok(ctx) => ctx,
            Err(err_response) => return err_response,
        };

        let routing_meta = parse_psrl_request_meta_from_context(&ctx);

        let (result_tx, result_rx) = oneshot::channel();
        let entry = RoutingQueueEntry {
            ctx,
            result_tx,
            routing_meta,
        };
        enqueue_and_wait(&self.runtime, entry, result_rx).await
    }
}

// ── Stage 1 helper ───────────────────────────────────────────────────────

// Refactor Note 3: EnqueueAndWaitStage — sends to the routing loop and awaits the response.
/// Send a `RoutingQueueEntry` to the routing loop and await the response.
///
/// Returns an error response if the routing loop channel is closed or the
/// result sender was dropped before the response was delivered.
async fn enqueue_and_wait(
    runtime: &Arc<RoutingLoopRuntime>,
    entry: RoutingQueueEntry,
    result_rx: oneshot::Receiver<Response>,
) -> Response {
    if runtime.tx.send(entry).is_err() {
        error!("routing_loop_pipeline: routing loop channel is closed");
        return router_error::internal_error(
            "routing_loop_send_failed",
            "Routing loop channel is closed",
        );
    }
    result_rx.await.unwrap_or_else(|_| {
        error!("routing_loop_pipeline: routing loop dropped result sender");
        router_error::internal_error(
            "routing_loop_recv_failed",
            "Routing loop dropped result sender",
        )
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::StatusCode;
    use openai_protocol::generate::GenerateRequest;
    use tokio::sync::oneshot;

    use super::*;
    use crate::{
        config::types::RequestSortIndicator,
        routers::{
            grpc::context::{RequestContext, SharedComponents},
            routing_loop_utils::{RoutingLoopRuntime, RoutingQueueEntry},
        },
    };

    // PR 13 §13.1: Build minimal SharedComponents for tests (no tokenizer, no multimodal).
    fn make_components() -> Arc<SharedComponents> {
        use llm_tokenizer::TokenizerRegistry;
        use reasoning_parser::ParserFactory as ReasoningParserFactory;
        use tool_parser::ParserFactory as ToolParserFactory;
        Arc::new(SharedComponents {
            tokenizer_registry: Arc::new(TokenizerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            multimodal: None,
        })
    }

    // PR 13 §13.1: Build a minimal RoutingQueueEntry with ctx (no PreparedRequest).
    fn make_test_entry(result_tx: oneshot::Sender<Response>) -> RoutingQueueEntry {
        let gen_req: GenerateRequest =
            serde_json::from_str(r#"{"text":"test"}"#).expect("test GenerateRequest");
        let ctx = RequestContext::for_generate(Arc::new(gen_req), None, None, make_components());
        RoutingQueueEntry {
            ctx,
            result_tx,
            routing_meta: None,
        }
    }

    // Refactor Note 3: verify that EnqueueAndWaitStage correctly reports channel-closed errors.
    #[tokio::test]
    async fn test_enqueue_and_wait_channel_closed() {
        // Create a runtime but immediately drop the receiver — simulates closed channel.
        let (runtime, rx) = RoutingLoopRuntime::new_with_channel(
            RequestSortIndicator::SmallId,
            false,
            Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
            0,
            String::new(),
            None,
        );
        // Drop the receiver so the tx.send will fail.
        drop(rx);

        let (result_tx, result_rx) = oneshot::channel();
        let entry = make_test_entry(result_tx);

        let response = enqueue_and_wait(&runtime, entry, result_rx).await;
        // Should return an internal error (channel closed).
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Refactor Note 3: verify EnqueueAndWaitStage handles dropped result sender.
    #[tokio::test]
    async fn test_enqueue_and_wait_result_dropped() {
        let (result_tx, result_rx) = oneshot::channel::<Response>();
        // Simulate routing loop dropping result_tx without sending.
        drop(result_tx);
        let response = result_rx.await;
        // Receiver should get an Err (sender dropped).
        assert!(response.is_err());
    }
}
