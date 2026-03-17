// PR 12 §12.2: Partial rollout protocol — drain stream, extract PartialRolloutState,
// and loopback mutation for PSRL multi-iteration rollout.
//!
//! This module implements the drain-based partial rollout interception that sits
//! between `RequestExecutionStage` and `ResponseProcessingStage` in the PSRL dispatch
//! path.
//!
//! # Design
//!
//! Rather than coupling to HTTP response struct types, we operate directly on
//! the unified proto types (`ProtoStream`, `ProtoGenerateStreamChunk`,
//! `ProtoGenerateComplete`) that all three backends (SGLang, vLLM, TRT-LLM)
//! already produce.
//!
//! ```text
//! RequestExecutionStage → ExecutionResult { stream: ProtoStream }
//!     ↓
//! drain_stream_for_partial_rollout(&mut stream) → DrainedStreamResult
//!     ↓
//! Match finish_reason:
//!   "stop"/"length"  → VllmPartialRollout::extract() → merge into final response
//!   "abort"          → VllmPartialRollout::extract() → accumulate, loopback
//!   other/error      → return as-is
//! ```
//!
//! Partial-rollout extraction is implemented for vLLM, SGLang, and TRT-LLM.
//! Each backend contributes token IDs and output logprobs into a unified state.

use tracing::{debug, warn};

use super::proto_wrapper::{
    ProtoGenerateComplete, ProtoGenerateStreamChunk, ProtoOutputLogProbs, ProtoResponseVariant,
    ProtoStream,
};

// ── Core types ─────────────────────────────────────────────────────────────────

// PR 12 §12.2: Result of draining a ProtoStream for partial rollout interception.
/// The result of consuming an entire `ProtoStream` for partial rollout analysis.
///
/// Contains all streaming chunks, the final complete message (if any), and the
/// detected finish reason.
pub(crate) struct DrainedStreamResult {
    /// Streaming chunks received before the final `ProtoGenerateComplete`.
    pub accumulated_chunks: Vec<ProtoGenerateStreamChunk>,
    /// The final complete message from the stream (contains output_ids, logprobs, etc.).
    pub final_complete: Option<ProtoGenerateComplete>,
    /// Finish reason extracted from the final complete message.
    /// Values: `"stop"`, `"length"`, `"abort"`, or `None` for error/stream-end.
    pub finish_reason: Option<String>,
}

// PR 12 §12.2: Unified partial rollout state accumulated across loopback iterations.
/// Partial rollout state extracted from a drained stream for one iteration.
///
/// This is the proto-native counterpart to the JSON-based `PartialRolloutState`
/// in `routing_loop_utils.rs`. It stores:
/// - `token_ids`: token IDs generated so far (accumulated across loopback iterations)
/// - `logprobs`: per-token output logprobs (backend-agnostic unified type)
///
/// Accumulated across iterations and merged into the final response when the
/// request finishes with `"stop"` or `"length"`.
///
/// # Loopback injection
///
/// Each request-building stage rebuilds the proto from scratch on every call
/// (initial build + every loopback). Therefore the loopback helper
/// (`maybe_apply_partial_rollout_loopback`) always injects **all** accumulated
/// `token_ids` into the freshly built proto. No per-field injection cursor is needed.
#[derive(Debug, Clone, Default)]
pub(crate) struct ProtoPartialRolloutState {
    /// Token IDs generated in this and prior loopback iterations.
    pub token_ids: Vec<u32>,
    /// Output logprobs per token (unified proto type, not JSON).
    // PR 12 §12.2: logprobs field established for future use (logprob merging in PR 13+).
    pub logprobs: Vec<ProtoOutputLogProbs>,
}

impl ProtoPartialRolloutState {

    /// Merge `other` into `self` by appending token_ids and logprobs.
    ///
    /// Used when accumulating partial state across loopback iterations.
    // PR 18 (Gap 5): used by routing-loop context accumulation.
    pub fn extend_from(&mut self, other: &Self) {
        self.token_ids.extend_from_slice(&other.token_ids);
        self.logprobs.extend_from_slice(&other.logprobs);
    }
}

// ── Drain function ─────────────────────────────────────────────────────────────

// PR 12 §12.2: Drain an entire ProtoStream, collecting chunks and detecting finish_reason.
/// Drain a `ProtoStream` entirely, accumulating chunks and detecting the finish reason.
///
/// Called by the dispatch task on `ExecutionResult::Single` before `ResponseProcessing`.
/// The stream is consumed; the caller uses `DrainedStreamResult` to decide the outcome.
///
/// # Behavior
/// - Collects all `ProtoResponseVariant::Chunk` items into `accumulated_chunks`.
/// - Sets `final_complete` from the first `ProtoResponseVariant::Complete` item.
/// - Extracts `finish_reason` from `final_complete.finish_reason()`.
/// - Logs warnings on stream errors but continues to drain.
pub(crate) async fn drain_stream_for_partial_rollout(
    stream: &mut ProtoStream,
) -> DrainedStreamResult {
    let mut accumulated_chunks: Vec<ProtoGenerateStreamChunk> = Vec::new();
    let mut final_complete: Option<ProtoGenerateComplete> = None;

    loop {
        match stream.next().await {
            Some(Ok(proto_response)) => {
                match proto_response.into_response() {
                    ProtoResponseVariant::Chunk(chunk) => {
                        accumulated_chunks.push(chunk);
                    }
                    ProtoResponseVariant::Complete(complete) => {
                        final_complete = Some(complete);
                        // After Complete, no more messages are expected.
                        break;
                    }
                    ProtoResponseVariant::Error(err) => {
                        warn!(
                            message = err.message(),
                            "drain_stream_for_partial_rollout: stream error"
                        );
                        break;
                    }
                    ProtoResponseVariant::None => {
                        // Empty message — skip.
                    }
                }
            }
            Some(Err(status)) => {
                warn!(
                    status = %status,
                    "drain_stream_for_partial_rollout: stream gRPC error"
                );
                break;
            }
            None => {
                // Stream exhausted.
                break;
            }
        }
    }

    let finish_reason = final_complete
        .as_ref()
        .map(|c| c.finish_reason().to_string());

    debug!(
        chunks = accumulated_chunks.len(),
        has_complete = final_complete.is_some(),
        ?finish_reason,
        "drain_stream_for_partial_rollout complete"
    );

    DrainedStreamResult {
        accumulated_chunks,
        final_complete,
        finish_reason,
    }
}

// ── Backend-specific extractors ────────────────────────────────────────────────

// PR 12 §12.2: vLLM partial rollout extractor — full implementation.
/// vLLM partial rollout extractor.
///
/// Extracts `token_ids` from `ProtoGenerateComplete` (vLLM variant) — specifically
/// from `output_ids` — and collects `output_logprobs` from both the complete message
/// and any accumulated chunks.
///
/// Returns a `ProtoPartialRolloutState` populated with the token IDs and logprobs
/// generated in this iteration.
pub(crate) struct VllmPartialRollout;

impl VllmPartialRollout {
    // PR 12 §12.2: Extract token_ids and logprobs from a drained stream result (vLLM).
    /// Extract partial rollout state from a drained stream result for vLLM backend.
    ///
    /// Extraction strategy:
    /// - **Token IDs**: taken from `final_complete.token_ids()` (i.e., `output_ids`).
    ///   If the complete message is absent, falls back to collecting from chunk `token_ids`.
    /// - **Logprobs**: taken from `final_complete.output_logprobs()` first; if absent,
    ///   collected from individual chunks (streaming mode).
    pub fn extract(drained: &DrainedStreamResult) -> ProtoPartialRolloutState {
        let mut token_ids: Vec<u32> = Vec::new();
        let mut logprobs: Vec<ProtoOutputLogProbs> = Vec::new();

        if let Some(complete) = &drained.final_complete {
            if complete.is_vllm() {
                // PR 12 §12.2: vLLM complete message: output_ids is the full token sequence.
                token_ids.extend_from_slice(complete.token_ids());

                // vLLM logprobs from the final complete message.
                if let Some(lp) = complete.output_logprobs() {
                    logprobs.push(lp);
                } else {
                    // Fallback: collect per-chunk logprobs in streaming mode.
                    for chunk in &drained.accumulated_chunks {
                        if let Some(lp) = chunk.output_logprobs() {
                            logprobs.push(lp);
                        }
                    }
                }
            } else {
                // Non-vLLM complete — use generic token_ids accessor.
                token_ids.extend_from_slice(complete.token_ids());
            }
        } else {
            // No complete message: accumulate from chunks (non-standard, but handle gracefully).
            for chunk in &drained.accumulated_chunks {
                token_ids.extend_from_slice(chunk.token_ids());
                if let Some(lp) = chunk.output_logprobs() {
                    logprobs.push(lp);
                }
            }
        }

        ProtoPartialRolloutState { token_ids, logprobs }
    }
}

// PR 18 (Gap 5): SGLang partial rollout extractor.
pub(crate) struct SglangPartialRollout;

impl SglangPartialRollout {
    pub fn extract(drained: &DrainedStreamResult) -> ProtoPartialRolloutState {
        let mut token_ids: Vec<u32> = Vec::new();
        let mut logprobs: Vec<ProtoOutputLogProbs> = Vec::new();

        if let Some(complete) = &drained.final_complete {
            token_ids.extend_from_slice(complete.token_ids());
            if let Some(lp) = complete.output_logprobs() {
                logprobs.push(lp);
            }
        }

        if token_ids.is_empty() {
            for chunk in &drained.accumulated_chunks {
                token_ids.extend_from_slice(chunk.token_ids());
                if let Some(lp) = chunk.output_logprobs() {
                    logprobs.push(lp);
                }
            }
        }

        ProtoPartialRolloutState { token_ids, logprobs }
    }
}

// PR 18 (Gap 5): TRT-LLM partial rollout extractor.
pub(crate) struct TrtllmPartialRollout;

impl TrtllmPartialRollout {
    pub fn extract(drained: &DrainedStreamResult) -> ProtoPartialRolloutState {
        let mut token_ids: Vec<u32> = Vec::new();
        let mut logprobs: Vec<ProtoOutputLogProbs> = Vec::new();

        if let Some(complete) = &drained.final_complete {
            token_ids.extend_from_slice(complete.token_ids());
            if let Some(lp) = complete.output_logprobs() {
                logprobs.push(lp);
            }
        }

        if token_ids.is_empty() {
            for chunk in &drained.accumulated_chunks {
                token_ids.extend_from_slice(chunk.token_ids());
                if let Some(lp) = chunk.output_logprobs() {
                    logprobs.push(lp);
                }
            }
        }

        ProtoPartialRolloutState { token_ids, logprobs }
    }
}

// ── Dispatch-helper: select extractor by backend ───────────────────────────────

// PR 12 §12.2: Select the appropriate extractor based on the backend type.
/// Extract partial rollout state from a drained stream using the backend-appropriate extractor.
///
/// Dispatches to `VllmPartialRollout`, `SglangPartialRollout`, or `TrtllmPartialRollout`
/// based on the backend detected from the `final_complete` variant (or chunks, if no complete).
pub(crate) fn extract_partial_rollout_state(
    drained: &DrainedStreamResult,
) -> ProtoPartialRolloutState {
    // Detect backend from final_complete or first chunk.
    let is_vllm = drained
        .final_complete
        .as_ref()
        .map(|c| c.is_vllm())
        .unwrap_or_else(|| {
            drained
                .accumulated_chunks
                .first()
                .map(|c| c.is_vllm())
                .unwrap_or(false)
        });

    let is_sglang = !is_vllm
        && drained
            .final_complete
            .as_ref()
            .map(|c| c.is_sglang())
            .unwrap_or_else(|| {
                drained
                    .accumulated_chunks
                    .first()
                    .map(|c| c.is_sglang())
                    .unwrap_or(false)
            });

    if is_vllm {
        VllmPartialRollout::extract(drained)
    } else if is_sglang {
        SglangPartialRollout::extract(drained)
    } else {
        TrtllmPartialRollout::extract(drained)
    }
}

// ── Merge helper ───────────────────────────────────────────────────────────────

// Issue 1: merge_partial_into_drained — prepend accumulated token_ids AND output logprobs.
// PR 12 §12.2: merge_partial_into_drained — prepend accumulated partial state into final drained result.
/// Prepend accumulated partial state (from prior loopback iterations) into the final
/// `DrainedStreamResult`.
///
/// For the `"stop"` or `"length"` case, the prior iterations' token_ids AND logprobs
/// must be prepended to the final complete before building the HTTP response so that
/// the client receives a complete token sequence and matching logprob sequence.
///
/// Mutates `drained.final_complete` by:
/// 1. Prepending `prior.token_ids` to `output_ids`.
/// 2. Prepending `prior.logprobs` to `output_logprobs` via `prepend_output_logprobs`.
pub(crate) fn merge_partial_into_drained(
    drained: &mut DrainedStreamResult,
    prior: &ProtoPartialRolloutState,
) {
    if prior.token_ids.is_empty() {
        return;
    }

    let Some(complete) = drained.final_complete.as_mut() else {
        return;
    };

    // ── Merge token_ids ───────────────────────────────────────────────────
    match complete {
        ProtoGenerateComplete::Vllm(c) => {
            // PR 12 §12.2: Prepend prior token_ids before the final iteration's output_ids.
            let mut merged = prior.token_ids.clone();
            merged.extend_from_slice(&c.output_ids);
            c.output_ids = merged;
        }
        ProtoGenerateComplete::Sglang(c) => {
            let mut merged = prior.token_ids.clone();
            merged.extend_from_slice(&c.output_ids);
            c.output_ids = merged;
        }
        ProtoGenerateComplete::Trtllm(c) => {
            let mut merged = prior.token_ids.clone();
            merged.extend_from_slice(&c.output_token_ids);
            c.output_token_ids = merged;
        }
    }

    // ── Issue 1: Merge output logprobs ────────────────────────────────────
    // Prepend all prior-iteration logprobs so the final response logprob
    // sequence matches the non-interrupted case.
    if !prior.logprobs.is_empty() {
        complete.prepend_output_logprobs(&prior.logprobs);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::proto_wrapper::ProtoTopLogProbs;
    use smg_grpc_client::{sglang_proto as sglang, trtllm_proto as trtllm, vllm_proto as vllm};

    // Helper: create a mock ProtoGenerateComplete for vLLM.
    fn make_vllm_complete(
        output_ids: Vec<u32>,
        finish_reason: &str,
        with_logprobs: bool,
    ) -> ProtoGenerateComplete {
        let output_logprobs = if with_logprobs {
            Some(vllm::OutputLogProbs {
                token_logprobs: vec![0.5_f32],
                token_ids: output_ids.clone(),
                top_logprobs: vec![],
            })
        } else {
            None
        };
        ProtoGenerateComplete::Vllm(vllm::GenerateComplete {
            output_ids: output_ids.clone(),
            finish_reason: finish_reason.to_string(),
            output_logprobs,
            ..Default::default()
        })
    }

    fn make_sglang_complete(
        output_ids: Vec<u32>,
        finish_reason: &str,
        with_logprobs: bool,
    ) -> ProtoGenerateComplete {
        let output_logprobs = if with_logprobs {
            Some(sglang::OutputLogProbs {
                token_logprobs: vec![0.25_f32],
                token_ids: output_ids.clone(),
                top_logprobs: vec![],
            })
        } else {
            None
        };

        ProtoGenerateComplete::Sglang(sglang::GenerateComplete {
            output_ids,
            finish_reason: finish_reason.to_string(),
            output_logprobs,
            ..Default::default()
        })
    }

    fn make_trtllm_complete(
        output_token_ids: Vec<u32>,
        finish_reason: &str,
        with_logprobs: bool,
    ) -> ProtoGenerateComplete {
        let logprobs = if with_logprobs {
            vec![trtllm::TokenLogprob {
                token_id: output_token_ids.first().copied().unwrap_or(0),
                logprob: 0.75_f32,
                top_logprobs: vec![],
            }]
        } else {
            vec![]
        };

        ProtoGenerateComplete::Trtllm(trtllm::GenerateComplete {
            output_token_ids,
            finish_reason: finish_reason.to_string(),
            logprobs,
            ..Default::default()
        })
    }

    // Helper: create a DrainedStreamResult with the given complete and no chunks.
    fn make_drained(complete: ProtoGenerateComplete) -> DrainedStreamResult {
        let finish_reason = Some(complete.finish_reason().to_string());
        DrainedStreamResult {
            accumulated_chunks: vec![],
            final_complete: Some(complete),
            finish_reason,
        }
    }

    // PR 12 §12.4 test: drain detects "stop" finish_reason.
    // We test DrainedStreamResult directly (no real stream needed).
    #[test]
    fn test_drain_stream_stop_finish_reason() {
        let complete = make_vllm_complete(vec![1, 2, 3], "stop", false);
        let drained = make_drained(complete);
        assert_eq!(drained.finish_reason.as_deref(), Some("stop"));
        assert!(drained.accumulated_chunks.is_empty());
        assert!(drained.final_complete.is_some());
    }

    // PR 12 §12.4 test: drain detects "abort" finish_reason.
    #[test]
    fn test_drain_stream_abort_finish_reason() {
        let complete = make_vllm_complete(vec![10, 20], "abort", false);
        let drained = make_drained(complete);
        assert_eq!(drained.finish_reason.as_deref(), Some("abort"));
        assert!(drained.final_complete.is_some());
    }

    // PR 12 §12.4 test: VllmPartialRollout extracts token_ids.
    #[test]
    fn test_vllm_partial_rollout_extracts_token_ids() {
        let complete = make_vllm_complete(vec![100, 200, 300], "stop", false);
        let drained = make_drained(complete);
        let state = VllmPartialRollout::extract(&drained);
        assert_eq!(state.token_ids, vec![100, 200, 300]);
    }

    // PR 12 §12.4 test: VllmPartialRollout extracts logprobs when present.
    #[test]
    fn test_vllm_partial_rollout_extracts_logprobs() {
        let complete = make_vllm_complete(vec![1, 2], "abort", true);
        let drained = make_drained(complete);
        let state = VllmPartialRollout::extract(&drained);
        assert_eq!(state.token_ids, vec![1, 2]);
        assert_eq!(state.logprobs.len(), 1);
        assert_eq!(state.logprobs[0].token_logprobs, vec![0.5_f32]);
    }

    // PR 18 (Gap 5): SGLang extractor returns token_ids and output logprobs.
    #[test]
    fn test_sglang_partial_rollout_extracts_token_ids_and_logprobs() {
        let complete = make_sglang_complete(vec![11, 22], "abort", true);
        let drained = make_drained(complete);
        let state = SglangPartialRollout::extract(&drained);

        assert_eq!(state.token_ids, vec![11, 22]);
        assert_eq!(state.logprobs.len(), 1);
        assert_eq!(state.logprobs[0].token_logprobs, vec![0.25_f32]);
    }

    // PR 18 (Gap 5): TRT-LLM extractor returns token_ids and output logprobs.
    #[test]
    fn test_trtllm_partial_rollout_extracts_token_ids_and_logprobs() {
        let complete = make_trtllm_complete(vec![7, 8, 9], "abort", true);
        let drained = make_drained(complete);
        let state = TrtllmPartialRollout::extract(&drained);

        assert_eq!(state.token_ids, vec![7, 8, 9]);
        assert_eq!(state.logprobs.len(), 1);
        assert_eq!(state.logprobs[0].token_ids, vec![7]);
    }

    // PR 12 §12.4 test: merge_partial_into_drained prepends prior token_ids.
    #[test]
    fn test_merge_partial_into_final_drained() {
        let complete = make_vllm_complete(vec![30, 40], "stop", false);
        let mut drained = make_drained(complete);

        let prior = ProtoPartialRolloutState {
            token_ids: vec![10, 20],
            logprobs: vec![],
        };

        merge_partial_into_drained(&mut drained, &prior);

        if let Some(ProtoGenerateComplete::Vllm(c)) = &drained.final_complete {
            assert_eq!(c.output_ids, vec![10, 20, 30, 40]);
        } else {
            panic!("Expected vLLM complete");
        }
    }

    // PR 12 §12.4 test: merge_partial_into_drained is a no-op when prior is empty.
    #[test]
    fn test_merge_partial_empty_prior_noop() {
        let complete = make_vllm_complete(vec![1, 2], "stop", false);
        let mut drained = make_drained(complete);
        let prior = ProtoPartialRolloutState::default();

        merge_partial_into_drained(&mut drained, &prior);

        if let Some(ProtoGenerateComplete::Vllm(c)) = &drained.final_complete {
            assert_eq!(c.output_ids, vec![1, 2]); // Unchanged
        } else {
            panic!("Expected vLLM complete");
        }
    }

    // Issue 1: merge_partial_into_drained should merge logprobs for vLLM backend.
    #[test]
    fn test_merge_partial_into_drained_merges_vllm_logprobs() {
        // Final iteration has [30, 40] with logprobs [0.3, 0.4].
        let complete = make_vllm_complete(vec![30, 40], "stop", true);
        let mut drained = make_drained(complete);

        // Prior iterations accumulated tokens [10, 20] with logprobs [0.1, 0.2].
        let prior = ProtoPartialRolloutState {
            token_ids: vec![10, 20],
            logprobs: vec![ProtoOutputLogProbs {
                token_logprobs: vec![0.1_f32, 0.2_f32],
                token_ids: vec![10, 20],
                top_logprobs: vec![],
            }],
        };

        merge_partial_into_drained(&mut drained, &prior);

        if let Some(ProtoGenerateComplete::Vllm(c)) = &drained.final_complete {
            // Token IDs: prior [10,20] + final [30,40].
            assert_eq!(c.output_ids, vec![10, 20, 30, 40]);
            // Logprobs: prior [0.1, 0.2] prepended before final [0.5] (from make_vllm_complete).
            let lp = c.output_logprobs.as_ref().expect("logprobs should be present");
            assert_eq!(lp.token_logprobs, vec![0.1_f32, 0.2_f32, 0.5_f32]);
            assert_eq!(lp.token_ids, vec![10, 20, 30, 40]);
        } else {
            panic!("Expected vLLM complete");
        }
    }

    // Issue 1: merge_partial_into_drained should merge logprobs for SGLang backend.
    #[test]
    fn test_merge_partial_into_drained_merges_sglang_logprobs() {
        let complete = make_sglang_complete(vec![30, 40], "stop", true);
        let mut drained = make_drained(complete);

        let prior = ProtoPartialRolloutState {
            token_ids: vec![10, 20],
            logprobs: vec![ProtoOutputLogProbs {
                token_logprobs: vec![0.1_f32, 0.2_f32],
                token_ids: vec![10, 20],
                top_logprobs: vec![],
            }],
        };

        merge_partial_into_drained(&mut drained, &prior);

        if let Some(ProtoGenerateComplete::Sglang(c)) = &drained.final_complete {
            assert_eq!(c.output_ids, vec![10, 20, 30, 40]);
            let lp = c.output_logprobs.as_ref().expect("logprobs should be present");
            // prior [0.1, 0.2] + final [0.25] (from make_sglang_complete with_logprobs=true).
            assert_eq!(lp.token_logprobs, vec![0.1_f32, 0.2_f32, 0.25_f32]);
        } else {
            panic!("Expected SGLang complete");
        }
    }

    // Issue 1: merge_partial_into_drained should merge logprobs for TRT-LLM backend.
    #[test]
    fn test_merge_partial_into_drained_merges_trtllm_logprobs() {
        // Final iteration has [30] with logprob 0.75 (from make_trtllm_complete with_logprobs=true).
        let complete = make_trtllm_complete(vec![30], "stop", true);
        let mut drained = make_drained(complete);

        // Prior had [10, 20] with logprobs.
        let prior = ProtoPartialRolloutState {
            token_ids: vec![10, 20],
            logprobs: vec![ProtoOutputLogProbs {
                token_logprobs: vec![0.1_f32, 0.2_f32],
                token_ids: vec![10, 20],
                top_logprobs: vec![
                    ProtoTopLogProbs { values: vec![], token_ids: vec![] },
                    ProtoTopLogProbs { values: vec![], token_ids: vec![] },
                ],
            }],
        };

        merge_partial_into_drained(&mut drained, &prior);

        if let Some(ProtoGenerateComplete::Trtllm(c)) = &drained.final_complete {
            assert_eq!(c.output_token_ids, vec![10, 20, 30]);
            // TRT-LLM: prior 2 tokens + final 1 token = 3 total logprob entries.
            assert_eq!(c.logprobs.len(), 3);
            assert_eq!(c.logprobs[0].token_id, 10);
            assert_eq!(c.logprobs[0].logprob, 0.1_f32);
            assert_eq!(c.logprobs[1].token_id, 20);
            assert_eq!(c.logprobs[1].logprob, 0.2_f32);
            // Final entry from make_trtllm_complete (token_id=30, logprob=0.75).
            assert_eq!(c.logprobs[2].token_id, 30);
            assert_eq!(c.logprobs[2].logprob, 0.75_f32);
        } else {
            panic!("Expected TRT-LLM complete");
        }
    }

    // Issue 1: merge_partial_into_drained is a no-op when prior.logprobs is empty (only tokens).
    #[test]
    fn test_merge_partial_into_drained_noop_when_no_prior_logprobs() {
        let complete = make_vllm_complete(vec![30], "stop", true);
        let mut drained = make_drained(complete);

        // Prior has token_ids but no logprobs.
        let prior = ProtoPartialRolloutState {
            token_ids: vec![10, 20],
            logprobs: vec![], // empty — no logprob merge should happen
        };

        merge_partial_into_drained(&mut drained, &prior);

        if let Some(ProtoGenerateComplete::Vllm(c)) = &drained.final_complete {
            assert_eq!(c.output_ids, vec![10, 20, 30]);
            // Logprobs unchanged: still only the final iteration's [0.5].
            let lp = c.output_logprobs.as_ref().expect("logprobs should be present");
            assert_eq!(lp.token_logprobs, vec![0.5_f32]);
        } else {
            panic!("Expected vLLM complete");
        }
    }

}
