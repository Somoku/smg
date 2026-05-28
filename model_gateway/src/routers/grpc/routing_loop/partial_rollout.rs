//! Partial-rollout state management and stream drain utilities.
//!
//! When a PS weight-sync interrupts generation (`finish_reason == "abort"`), the
//! accumulated tokens from the aborted stream are preserved here so the
//! re-routed request can continue generation from the right offset.
//!
//! # Life-cycle
//!
//! 1. `dispatch_entry_with_partial_rollout` detects `is_psrl = true` and enters
//!    the loopback loop.
//! 2. After each execution, `drain_stream_for_partial_rollout` consumes the
//!    raw gRPC stream into a `DrainedStreamResult`.
//! 3. If `finish_reason == "abort"`:
//!    - `merge_into_partial_state` appends the new tokens to
//!      `PartialRolloutState`.
//!    - `reset_ctx_for_loopback` rebuilds `ctx` for the next iteration.
//! 4. Otherwise the loop exits and the caller forwards the result to the client.

use tracing::warn;

use crate::routers::grpc::{
    context::RequestContext,
    proto_wrapper::{ProtoGenerateComplete, ProtoResponseVariant, ProtoStream},
};

/// Accumulated token state carried across partial-rollout loopback iterations.
#[derive(Debug, Clone, Default)]
pub(crate) struct PartialRolloutState {
    /// All output token IDs accumulated so far across every loopback iteration.
    pub token_ids: Vec<u32>,
    /// Per-token log-probabilities, parallel to `token_ids`.
    /// `None` if the backend did not return log-prob data in any iteration.
    pub logprobs: Option<Vec<f32>>,
    /// Number of loopback iterations completed so far (excluding the final one).
    ///
    /// Incremented by `merge_into_partial_state` on each `"abort"` cycle.
    /// At completion this equals the number of PS weight-sync interruptions.
    pub iteration_count: u32,
}

impl PartialRolloutState {
    #[inline]
    pub fn response_token_count(&self) -> usize {
        self.token_ids.len()
    }
}

/// The result of draining a gRPC stream until either a `Complete` frame or
/// end-of-stream is reached.
pub(crate) struct DrainedStreamResult {
    /// New token IDs emitted by *this* stream segment (not yet accumulated).
    pub new_token_ids: Vec<u32>,
    /// New per-token log-probabilities for this segment (if available).
    pub new_logprobs: Option<Vec<f32>>,
    /// The finish reason from the `Complete` frame.
    ///
    /// `"abort"` → PS weight-sync interrupted generation (loopback needed).
    /// `"stop"` / `"length"` → generation finished normally.
    pub finish_reason: String,
    /// The raw `Complete` proto frame, forwarded to response-processing stages.
    pub complete: ProtoGenerateComplete,
}

/// Consume `stream` frame-by-frame until a `Complete` frame is received and
/// return the accumulated result.
pub(crate) async fn drain_stream_for_partial_rollout(
    stream: &mut ProtoStream,
) -> Result<DrainedStreamResult, String> {
    let mut new_token_ids: Vec<u32> = Vec::new();
    let mut new_logprobs: Option<Vec<f32>> = None;

    loop {
        let Some(response_result) = stream.next().await else {
            return Err("stream ended without Complete frame".to_owned());
        };

        let response = match response_result {
            Ok(r) => r,
            Err(status) => return Err(format!("gRPC stream error: {status}")),
        };

        match response.into_response() {
            ProtoResponseVariant::Chunk(chunk) => {
                new_token_ids.extend_from_slice(chunk.token_ids());

                if let Some(lp) = chunk.output_logprobs() {
                    new_logprobs
                        .get_or_insert_with(Vec::new)
                        .extend_from_slice(&lp.token_logprobs);
                }
            }
            ProtoResponseVariant::Complete(complete) => {
                if new_token_ids.is_empty() {
                    new_token_ids = complete.output_ids().to_vec();
                }
                let finish_reason = complete.finish_reason().to_owned();
                return Ok(DrainedStreamResult {
                    new_token_ids,
                    new_logprobs,
                    finish_reason,
                    complete,
                });
            }
            ProtoResponseVariant::None => {
                warn!("partial rollout drain: received empty proto frame, skipping");
            }
        }
    }
}

/// Append the tokens from `drained` into `state`.
///
/// If `state` already contains log-probs and the new segment also has them,
/// they are appended.  If only one side has log-probs, the mismatch is logged
/// and the log-prob field is dropped to keep `token_ids` and `logprobs`
/// always in sync.
pub(crate) fn merge_into_partial_state(
    state: &mut PartialRolloutState,
    drained: &DrainedStreamResult,
) {
    state.token_ids.extend_from_slice(&drained.new_token_ids);
    state.iteration_count += 1;

    match (&mut state.logprobs, &drained.new_logprobs) {
        (Some(acc), Some(new)) => acc.extend_from_slice(new),
        (None, Some(_new)) if state.iteration_count == 1 => {
            // First iteration produced log-probs; initialise the field.
            state.logprobs.clone_from(&drained.new_logprobs);
        }
        (Some(_), None) => {
            // Previous iterations had log-probs but this one didn't — drop them
            // to keep the vecs in sync.
            warn!("partial rollout merge: log-prob mismatch (had logprobs, new segment missing); dropping logprobs");
            state.logprobs = None;
        }
        _ => {}
    }
}

/// Reset `ctx` so it is ready for the next loopback iteration.
///
/// Specifically this clears the stage outputs that must be re-computed
/// (worker selection, client, request, dispatch, load-guards, execution
/// result) while preserving the preparation output and partial-rollout
/// state (which are needed by downstream stages).
pub(crate) fn reset_ctx_for_loopback(ctx: &mut RequestContext) {
    ctx.state.workers = None;
    ctx.state.clients = None;
    ctx.state.proto_request = None;
    ctx.state.dispatch = None;
    ctx.state.load_guards = None;
    // Clear only per-iteration response fields; keep preparation-stage
    // products (`stop_decoder`, `skip_special_tokens`) intact so the final
    // post-execution stage can use them after the last loopback iteration.
    ctx.state.response.execution_result = None;
    ctx.state.response.final_response = None;
    ctx.state.response.responses_iteration_result = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PartialRolloutState ────────────────────────────────────────────

    #[test]
    fn response_token_count_empty() {
        let state = PartialRolloutState::default();
        assert_eq!(state.response_token_count(), 0);
    }

    #[test]
    fn response_token_count_with_tokens() {
        let state = PartialRolloutState {
            token_ids: vec![1, 2, 3],
            logprobs: None,
            iteration_count: 0,
        };
        assert_eq!(state.response_token_count(), 3);
    }

    // ─── merge_into_partial_state ────────────────────────────────────────────

    fn make_drained(tokens: &[u32], logprobs: Option<Vec<f32>>) -> DrainedStreamResult {
        use smg_grpc_client::sglang_proto::GenerateComplete as SglangGenerateComplete;

        use crate::routers::grpc::proto_wrapper::ProtoGenerateComplete;

        // We need a concrete ProtoGenerateComplete for DrainedStreamResult.
        // Use the Sglang variant with a "stop" finish_reason.
        let complete = ProtoGenerateComplete::Sglang(SglangGenerateComplete {
            finish_reason: "stop".to_owned(),
            ..Default::default()
        });
        DrainedStreamResult {
            new_token_ids: tokens.to_vec(),
            new_logprobs: logprobs,
            finish_reason: "stop".to_owned(),
            complete,
        }
    }

    #[test]
    fn merge_tokens_no_logprobs() {
        let mut state = PartialRolloutState::default();
        let drained = make_drained(&[10, 20, 30], None);
        merge_into_partial_state(&mut state, &drained);
        assert_eq!(state.token_ids, vec![10, 20, 30]);
        assert!(state.logprobs.is_none());
    }

    #[test]
    fn merge_tokens_with_logprobs_two_iterations() {
        let mut state = PartialRolloutState::default();

        // Iteration 1 — with logprobs
        let d1 = make_drained(&[1, 2], Some(vec![0.1, 0.2]));
        merge_into_partial_state(&mut state, &d1);
        assert_eq!(state.token_ids, vec![1, 2]);
        assert_eq!(state.logprobs, Some(vec![0.1, 0.2]));

        // Iteration 2 — append
        let d2 = make_drained(&[3, 4], Some(vec![0.3, 0.4]));
        merge_into_partial_state(&mut state, &d2);
        assert_eq!(state.token_ids, vec![1, 2, 3, 4]);
        assert_eq!(state.logprobs, Some(vec![0.1, 0.2, 0.3, 0.4]));
    }

    #[test]
    fn merge_logprob_mismatch_drops_logprobs() {
        let mut state = PartialRolloutState {
            token_ids: vec![1],
            logprobs: Some(vec![0.5]),
            iteration_count: 1,
        };
        // Second segment has no logprobs → drop the accumulated ones
        let drained = make_drained(&[2], None);
        merge_into_partial_state(&mut state, &drained);
        assert_eq!(state.token_ids, vec![1, 2]);
        assert!(state.logprobs.is_none());
    }

    /// `merge_into_partial_state` increments `iteration_count` on every call.
    #[test]
    fn merge_increments_iteration_count() {
        let mut state = PartialRolloutState::default();
        assert_eq!(state.iteration_count, 0);

        let d1 = make_drained(&[1], None);
        merge_into_partial_state(&mut state, &d1);
        assert_eq!(state.iteration_count, 1);

        let d2 = make_drained(&[2], None);
        merge_into_partial_state(&mut state, &d2);
        assert_eq!(state.iteration_count, 2);

        // Token accumulation is also correct
        assert_eq!(state.token_ids, vec![1, 2]);
    }

    /// `response_token_count` reflects tokens accumulated across iterations.
    #[test]
    fn response_token_count_accumulates_across_merges() {
        let mut state = PartialRolloutState::default();
        assert_eq!(state.response_token_count(), 0);

        merge_into_partial_state(&mut state, &make_drained(&[10, 20], None));
        assert_eq!(state.response_token_count(), 2);

        merge_into_partial_state(&mut state, &make_drained(&[30, 40, 50], None));
        assert_eq!(state.response_token_count(), 5);
    }

    /// Merging when `state` is empty and new segment has log-probs initialises the field.
    #[test]
    fn merge_initialises_logprobs_from_first_segment() {
        let mut state = PartialRolloutState::default();
        let d = make_drained(&[1, 2], Some(vec![0.1, 0.2]));
        merge_into_partial_state(&mut state, &d);
        assert_eq!(state.logprobs, Some(vec![0.1, 0.2]));
    }

    /// When both sides have no log-probs, the field stays `None`.
    #[test]
    fn merge_no_logprobs_stays_none() {
        let mut state = PartialRolloutState::default();
        merge_into_partial_state(&mut state, &make_drained(&[1], None));
        merge_into_partial_state(&mut state, &make_drained(&[2], None));
        assert!(state.logprobs.is_none());
    }

    // ─── reset_ctx_for_loopback ──────────────────────────────────────────────
    #[test]
    fn reset_preserves_preparation_products_in_response_state() {
        use std::sync::Arc;

        use llm_tokenizer::{stop::StopSequenceDecoderBuilder, MockTokenizer};
        use openai_protocol::chat::ChatCompletionRequest;
        use reasoning_parser::ParserFactory as ReasoningParserFactory;
        use tool_parser::ParserFactory as ToolParserFactory;

        use crate::routers::grpc::context::{
            FinalResponse, ProcessingState, RequestContext, RequestInput, RequestType,
            SharedComponents,
        };

        // Minimal RequestContext built directly so the test stays inside the
        // crate (no dependency on the integration test helpers).  We use a
        // Chat request because `ChatCompletionRequest` derives `Default`
        // whereas `GenerateRequest` does not — the field-clearing logic under
        // test does not depend on the request variant.
        let components = Arc::new(SharedComponents {
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            multimodal: None,
        });
        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Chat(Arc::new(ChatCompletionRequest::default())),
                headers: None,
                model_id: "test-model".to_string(),
                tenant_request_meta: None,
            },
            components,
            state: ProcessingState::default(),
        };

        // Simulate Preparation stage outputs that live in `ResponseState`.
        let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = StopSequenceDecoderBuilder::new(tokenizer)
            .skip_special_tokens(true)
            .build();
        ctx.state.response.stop_decoder = Some(stop_decoder);
        ctx.state.response.skip_special_tokens = Some(true);

        // Simulate a per-iteration response field being populated; reset must
        // clear it so the next iteration starts clean.
        ctx.state.response.final_response = Some(FinalResponse::Generate(vec![]));

        reset_ctx_for_loopback(&mut ctx);

        // Preparation-stage products preserved.
        assert!(
            ctx.state.response.stop_decoder.is_some(),
            "stop_decoder must survive loopback reset (set once during preparation)"
        );
        assert_eq!(
            ctx.state.response.skip_special_tokens,
            Some(true),
            "skip_special_tokens must survive loopback reset"
        );

        // Per-iteration response fields cleared.
        assert!(
            ctx.state.response.execution_result.is_none(),
            "execution_result must be cleared between iterations"
        );
        assert!(
            ctx.state.response.final_response.is_none(),
            "final_response must be cleared between iterations"
        );
        assert!(
            ctx.state.response.responses_iteration_result.is_none(),
            "responses_iteration_result must be cleared between iterations"
        );
    }
}
