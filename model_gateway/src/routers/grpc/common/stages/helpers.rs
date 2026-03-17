//! Common helper functions shared across stages

use std::sync::Arc;

use rand::Rng;
use smg_grpc_client::sglang_proto::DisaggregatedParams;
use tracing::debug;

use crate::{
    core::{RuntimeType, Worker, DEFAULT_BOOTSTRAP_PORT},
    routers::grpc::{
        context::WorkerSelection, partial_rollout::ProtoPartialRolloutState,
        proto_wrapper::ProtoGenerateRequest,
    },
};

/// Inject PD bootstrap metadata for SGLang if needed.
///
/// SGLang uses DisaggregatedParams with bootstrap host/port/room.
/// vLLM uses different mechanisms: NIXL (automatic prefix matching) or
/// Mooncake (kv_transfer_params injected in request_execution stage).
pub(crate) fn maybe_inject_pd_metadata(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) {
    if let WorkerSelection::Dual {
        prefill,
        runtime_type,
        ..
    } = workers
    {
        if *runtime_type == RuntimeType::Sglang {
            inject_sglang_bootstrap_metadata(request, prefill);
        }
    }
}

// PR 18 (Gap 5): Apply accumulated partial-rollout state to the proto request
// for the next routing-loop iteration, regardless of original request type.
/// Apply accumulated partial-rollout tokens to the proto request for the next loopback iteration.
///
/// Each request-building stage rebuilds the proto from scratch on every call, so all
/// accumulated `token_ids` must be appended to the freshly built request each time.
/// This function is a no-op when `partial_state` is `None` or its `token_ids` is empty.
pub(crate) fn maybe_apply_partial_rollout_loopback(
    request: &mut ProtoGenerateRequest,
    partial_state: Option<&ProtoPartialRolloutState>,
) {
    let Some(state) = partial_state else { return };
    if state.token_ids.is_empty() {
        return;
    }
    request.apply_partial_rollout(&state.token_ids);
}

/// Inject bootstrap metadata into a SGLang gRPC request.
fn inject_sglang_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
) {
    let hostname = prefill_worker.bootstrap_host();
    let bootstrap_port = prefill_worker
        .bootstrap_port()
        .unwrap_or(DEFAULT_BOOTSTRAP_PORT);
    let room_id = rand::rng().random_range(0..i32::MAX);

    let disagg_params = DisaggregatedParams {
        bootstrap_host: hostname.to_string(),
        bootstrap_port: bootstrap_port as i32,
        bootstrap_room: room_id,
    };

    let sglang_request = request.as_sglang_mut();
    sglang_request.disaggregated_params = Some(disagg_params);

    debug!(
        "Injected bootstrap metadata: host={}, port={}, room={}",
        hostname, bootstrap_port, room_id
    );
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::{sglang_proto as sglang, trtllm_proto as trtllm, vllm_proto as vllm};

    use super::*;

    // PR 18 (Gap 5): proto-level loopback applies to SGLang requests.
    #[test]
    fn test_maybe_apply_partial_rollout_loopback_sglang() {
        let mut request = ProtoGenerateRequest::Sglang(Box::new(sglang::GenerateRequest {
            tokenized: Some(sglang::TokenizedInput {
                input_ids: vec![1, 2],
                ..Default::default()
            }),
            sampling_params: Some(sglang::SamplingParams {
                max_new_tokens: Some(10),
                ..Default::default()
            }),
            ..Default::default()
        }));

        let partial_state = ProtoPartialRolloutState {
            token_ids: vec![3, 4],
            logprobs: vec![],
        };

        maybe_apply_partial_rollout_loopback(&mut request, Some(&partial_state));

        match request {
            ProtoGenerateRequest::Sglang(req) => {
                assert_eq!(
                    req.tokenized
                        .as_ref()
                        .expect("tokenized input should exist")
                        .input_ids,
                    vec![1, 2, 3, 4]
                );
                assert_eq!(
                    req.sampling_params
                        .as_ref()
                        .expect("sampling params should exist")
                        .max_new_tokens,
                    Some(8)
                );
            }
            _ => panic!("expected sglang request"),
        }
    }

    // PR 18 (Gap 5): proto-level loopback applies to vLLM requests.
    #[test]
    fn test_maybe_apply_partial_rollout_loopback_vllm() {
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    input_ids: vec![9],
                    ..Default::default()
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(6),
                ..Default::default()
            }),
            ..Default::default()
        }));

        let partial_state = ProtoPartialRolloutState {
            token_ids: vec![10, 11],
            logprobs: vec![],
        };

        maybe_apply_partial_rollout_loopback(&mut request, Some(&partial_state));

        match request {
            ProtoGenerateRequest::Vllm(req) => {
                let input_ids = match req.input.as_ref() {
                    Some(vllm::generate_request::Input::Tokenized(tokenized)) => {
                        tokenized.input_ids.clone()
                    }
                    _ => panic!("expected vllm tokenized input"),
                };
                assert_eq!(input_ids, vec![9, 10, 11]);
                assert_eq!(
                    req.sampling_params
                        .as_ref()
                        .expect("sampling params should exist")
                        .max_tokens,
                    Some(4)
                );
            }
            _ => panic!("expected vllm request"),
        }
    }

    // PR 18 (Gap 5): proto-level loopback applies to TensorRT-LLM requests.
    #[test]
    fn test_maybe_apply_partial_rollout_loopback_trtllm() {
        let mut request = ProtoGenerateRequest::Trtllm(Box::new(trtllm::GenerateRequest {
            tokenized: Some(trtllm::TokenizedInput {
                input_token_ids: vec![7],
                ..Default::default()
            }),
            max_tokens: 5,
            ..Default::default()
        }));

        let partial_state = ProtoPartialRolloutState {
            token_ids: vec![8, 9],
            logprobs: vec![],
        };

        maybe_apply_partial_rollout_loopback(&mut request, Some(&partial_state));

        match request {
            ProtoGenerateRequest::Trtllm(req) => {
                assert_eq!(
                    req.tokenized
                        .as_ref()
                        .expect("tokenized input should exist")
                        .input_token_ids,
                    vec![7, 8, 9]
                );
                assert_eq!(req.max_tokens, 3);
            }
            _ => panic!("expected trtllm request"),
        }
    }

    // PR 18 (Gap 5): no partial state means no-op mutation.
    #[test]
    fn test_maybe_apply_partial_rollout_loopback_none_noop() {
        let mut request = ProtoGenerateRequest::Sglang(Box::new(sglang::GenerateRequest {
            tokenized: Some(sglang::TokenizedInput {
                input_ids: vec![1],
                ..Default::default()
            }),
            sampling_params: Some(sglang::SamplingParams {
                max_new_tokens: Some(4),
                ..Default::default()
            }),
            ..Default::default()
        }));

        maybe_apply_partial_rollout_loopback(&mut request, None);

        match request {
            ProtoGenerateRequest::Sglang(req) => {
                assert_eq!(
                    req.tokenized
                        .as_ref()
                        .expect("tokenized input should exist")
                        .input_ids,
                    vec![1]
                );
                assert_eq!(
                    req.sampling_params
                        .as_ref()
                        .expect("sampling params should exist")
                        .max_new_tokens,
                    Some(4)
                );
            }
            _ => panic!("expected sglang request"),
        }
    }
}
