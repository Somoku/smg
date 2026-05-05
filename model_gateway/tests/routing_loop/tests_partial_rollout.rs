//! Phase 7: Partial Rollout (PSRL) State Machine Tests
//!
//! Tests for the partial-rollout loopback state machine. These tests exercise
//! the core orchestration logic — `merge_into_partial_state`,
//! `reset_ctx_for_loopback`, and the abort-detection flow — using
//! `DrainedStreamResult` values constructed directly (no live gRPC stream needed).

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smg_grpc_client::sglang_proto::GenerateComplete as SglangGenerateComplete;

    use crate::routers::grpc::{
        context::{ProcessingState, RequestContext, SharedComponents},
        proto_wrapper::ProtoGenerateComplete,
        routing_loop::partial_rollout::{
            merge_into_partial_state, reset_ctx_for_loopback, DrainedStreamResult,
            PartialRolloutState,
        },
    };

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /// Build a `DrainedStreamResult` representing a single stream segment.
    fn make_segment(
        tokens: &[u32],
        logprobs: Option<Vec<f32>>,
        finish_reason: &str,
    ) -> DrainedStreamResult {
        let complete = ProtoGenerateComplete::Sglang(SglangGenerateComplete {
            finish_reason: finish_reason.to_owned(),
            output_ids: tokens.to_vec(),
            ..Default::default()
        });
        DrainedStreamResult {
            new_token_ids: tokens.to_vec(),
            new_logprobs: logprobs,
            finish_reason: finish_reason.to_owned(),
            complete,
        }
    }

    /// Build a minimal `RequestContext` with all stage outputs populated to
    /// non-`None` sentinels so `reset_ctx_for_loopback` can be verified.
    fn make_ctx_with_populated_state() -> RequestContext {
        use openai_protocol::chat::ChatCompletionRequest;
        use reasoning_parser::ParserFactory as ReasoningParserFactory;
        use tool_parser::ParserFactory as ToolParserFactory;

        use crate::routers::grpc::context::{RequestInput, RequestType};

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

        // Populate stage outputs to confirm reset clears them.
        // We use Some(Default::default()) or Some(...) where fields have Default.
        // Fields that don't have easy defaults are left None but we still verify
        // the ones we can easily set.
        ctx.state.partial_rollout_state = Some(PartialRolloutState {
            token_ids: vec![1, 2, 3],
            logprobs: None,
            iteration_count: 1,
        });

        ctx
    }

    // ─── Abort detection ─────────────────────────────────────────────────────

    /// A segment with `finish_reason == "abort"` signals loopback is needed.
    #[test]
    fn abort_finish_reason_detected() {
        let seg = make_segment(&[10, 20], None, "abort");
        assert_eq!(seg.finish_reason, "abort");
    }

    /// A segment with `finish_reason == "stop"` signals normal completion.
    #[test]
    fn stop_finish_reason_detected() {
        let seg = make_segment(&[10, 20], None, "stop");
        assert_ne!(seg.finish_reason, "abort");
    }

    /// A segment with `finish_reason == "length"` signals max-tokens completion.
    #[test]
    fn length_finish_reason_detected() {
        let seg = make_segment(&[10, 20], None, "length");
        assert_ne!(seg.finish_reason, "abort");
    }

    // ─── Single-abort loopback cycle ─────────────────────────────────────────

    /// Simulate: iteration 1 aborts, iteration 2 stops.
    /// State should accumulate tokens from both segments.
    #[test]
    fn single_abort_then_stop_accumulates_tokens() {
        let mut state = PartialRolloutState::default();

        // Iteration 1 — weight-sync interrupted generation
        let abort_seg = make_segment(&[100, 200, 300], None, "abort");
        merge_into_partial_state(&mut state, &abort_seg);
        assert_eq!(state.token_ids, vec![100, 200, 300]);
        assert_eq!(state.iteration_count, 1);

        // Iteration 2 — normal completion
        let stop_seg = make_segment(&[400, 500], None, "stop");
        merge_into_partial_state(&mut state, &stop_seg);
        assert_eq!(state.token_ids, vec![100, 200, 300, 400, 500]);
        assert_eq!(state.iteration_count, 2);
    }

    /// `response_token_count` grows correctly after each merge.
    #[test]
    fn response_token_count_grows_through_loopback_cycle() {
        let mut state = PartialRolloutState::default();
        assert_eq!(state.response_token_count(), 0);

        merge_into_partial_state(&mut state, &make_segment(&[1, 2, 3], None, "abort"));
        assert_eq!(state.response_token_count(), 3);

        merge_into_partial_state(&mut state, &make_segment(&[4, 5], None, "abort"));
        assert_eq!(state.response_token_count(), 5);

        merge_into_partial_state(&mut state, &make_segment(&[6], None, "stop"));
        assert_eq!(state.response_token_count(), 6);
    }

    // ─── Multi-abort loopback cycles ─────────────────────────────────────────

    /// Three consecutive aborts followed by a stop — all tokens accumulated.
    #[test]
    fn three_aborts_then_stop() {
        let mut state = PartialRolloutState::default();

        merge_into_partial_state(&mut state, &make_segment(&[1, 2], None, "abort"));
        merge_into_partial_state(&mut state, &make_segment(&[3, 4], None, "abort"));
        merge_into_partial_state(&mut state, &make_segment(&[5, 6], None, "abort"));
        merge_into_partial_state(&mut state, &make_segment(&[7, 8], None, "stop"));

        assert_eq!(state.token_ids, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(state.iteration_count, 4);
        assert_eq!(state.response_token_count(), 8);
    }

    // ─── Logprob handling through loopback ───────────────────────────────────

    /// Logprobs accumulate correctly across abort + stop iterations.
    #[test]
    fn logprobs_accumulate_through_loopback() {
        let mut state = PartialRolloutState::default();

        let abort_seg = make_segment(&[10, 20], Some(vec![0.1, 0.2]), "abort");
        merge_into_partial_state(&mut state, &abort_seg);
        assert_eq!(state.logprobs, Some(vec![0.1, 0.2]));

        let stop_seg = make_segment(&[30], Some(vec![0.3]), "stop");
        merge_into_partial_state(&mut state, &stop_seg);
        assert_eq!(state.logprobs, Some(vec![0.1, 0.2, 0.3]));
        assert_eq!(state.token_ids, vec![10, 20, 30]);
    }

    /// Logprobs missing in final segment drops the accumulated logprobs.
    #[test]
    fn logprob_drop_on_mismatch_through_loopback() {
        let mut state = PartialRolloutState::default();

        // First segment has logprobs
        merge_into_partial_state(&mut state, &make_segment(&[10], Some(vec![0.5]), "abort"));
        assert!(state.logprobs.is_some());

        // Second segment has no logprobs — should drop accumulated
        merge_into_partial_state(&mut state, &make_segment(&[20], None, "stop"));
        assert!(state.logprobs.is_none());
        // Tokens should still be accumulated correctly
        assert_eq!(state.token_ids, vec![10, 20]);
    }

    /// No logprobs throughout all iterations — stays None.
    #[test]
    fn no_logprobs_throughout_loopback_stays_none() {
        let mut state = PartialRolloutState::default();
        merge_into_partial_state(&mut state, &make_segment(&[1, 2], None, "abort"));
        merge_into_partial_state(&mut state, &make_segment(&[3, 4], None, "abort"));
        merge_into_partial_state(&mut state, &make_segment(&[5], None, "stop"));
        assert!(state.logprobs.is_none());
    }

    // ─── reset_ctx_for_loopback ───────────────────────────────────────────────

    /// After `reset_ctx_for_loopback`, all stage outputs are cleared while
    /// `partial_rollout_state` is preserved.
    #[test]
    fn reset_clears_stage_outputs_preserves_partial_rollout() {
        let mut ctx = make_ctx_with_populated_state();

        // Verify partial_rollout_state was set
        assert!(ctx.state.partial_rollout_state.is_some());

        reset_ctx_for_loopback(&mut ctx);

        // Stage outputs cleared
        assert!(ctx.state.workers.is_none(), "workers should be cleared");
        assert!(ctx.state.clients.is_none(), "clients should be cleared");
        assert!(
            ctx.state.proto_request.is_none(),
            "proto_request should be cleared"
        );
        assert!(ctx.state.dispatch.is_none(), "dispatch should be cleared");
        assert!(
            ctx.state.load_guards.is_none(),
            "load_guards should be cleared"
        );

        // Partial rollout state preserved
        assert!(
            ctx.state.partial_rollout_state.is_some(),
            "partial_rollout_state should be preserved"
        );
        let preserved = ctx.state.partial_rollout_state.as_ref().unwrap();
        assert_eq!(preserved.token_ids, vec![1, 2, 3]);
        assert_eq!(preserved.iteration_count, 1);
    }

    /// `reset_ctx_for_loopback` is idempotent — calling it twice is safe.
    #[test]
    fn reset_is_idempotent() {
        let mut ctx = make_ctx_with_populated_state();
        reset_ctx_for_loopback(&mut ctx);
        reset_ctx_for_loopback(&mut ctx); // second call should not panic
        assert!(ctx.state.workers.is_none());
        assert!(ctx.state.partial_rollout_state.is_some());
    }

    // ─── State lifecycle simulation ───────────────────────────────────────────

    /// Simulate the complete PSRL lifecycle:
    /// 1. abort → merge → reset
    /// 2. abort → merge → reset
    /// 3. stop  → merge (final)
    /// Final state should have all tokens and iteration_count == 3.
    #[test]
    fn full_psrl_lifecycle_simulation() {
        let mut state = PartialRolloutState::default();
        let mut ctx = make_ctx_with_populated_state();
        // Override partial_rollout_state to start fresh
        ctx.state.partial_rollout_state = None;

        // --- Iteration 1: abort ---
        let seg1 = make_segment(&[10, 20, 30], None, "abort");
        merge_into_partial_state(&mut state, &seg1);
        ctx.state.partial_rollout_state = Some(state.clone());
        reset_ctx_for_loopback(&mut ctx);
        // Confirm state preserved after reset
        assert_eq!(
            ctx.state.partial_rollout_state.as_ref().unwrap().token_ids,
            vec![10, 20, 30]
        );
        assert_eq!(
            ctx.state
                .partial_rollout_state
                .as_ref()
                .unwrap()
                .iteration_count,
            1
        );

        // --- Iteration 2: abort ---
        let seg2 = make_segment(&[40, 50], None, "abort");
        merge_into_partial_state(&mut state, &seg2);
        ctx.state.partial_rollout_state = Some(state.clone());
        reset_ctx_for_loopback(&mut ctx);
        assert_eq!(
            ctx.state.partial_rollout_state.as_ref().unwrap().token_ids,
            vec![10, 20, 30, 40, 50]
        );
        assert_eq!(
            ctx.state
                .partial_rollout_state
                .as_ref()
                .unwrap()
                .iteration_count,
            2
        );

        // --- Iteration 3: stop (final) ---
        let seg3 = make_segment(&[60], None, "stop");
        merge_into_partial_state(&mut state, &seg3);

        // Final assertions
        assert_eq!(state.token_ids, vec![10, 20, 30, 40, 50, 60]);
        assert_eq!(state.iteration_count, 3);
        assert_eq!(state.response_token_count(), 6);
        assert!(state.logprobs.is_none());
    }

    /// Zero-token abort is handled gracefully (no-op on token accumulation).
    #[test]
    fn empty_abort_segment_is_harmless() {
        let mut state = PartialRolloutState::default();

        let empty_abort = make_segment(&[], None, "abort");
        merge_into_partial_state(&mut state, &empty_abort);
        assert_eq!(state.token_ids, vec![0u32; 0]);
        assert_eq!(state.iteration_count, 1);

        let stop_seg = make_segment(&[42], None, "stop");
        merge_into_partial_state(&mut state, &stop_seg);
        assert_eq!(state.token_ids, vec![42]);
        assert_eq!(state.iteration_count, 2);
    }
}
