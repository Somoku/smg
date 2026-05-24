//! Chat response processing stage: Handles both streaming and non-streaming responses
//!
//! - For streaming: Spawns background task and returns SSE response (early exit)
//! - For non-streaming: Collects all responses and builds final ChatCompletionResponse

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use openai_protocol::chat::{ChatCompletionMessage, ChatMessage, MessageContent};

use smg_tito::{RenderContext, TitoStore};
use tracing::{debug, error, warn};

use crate::{
    routers::{
        error,
        grpc::{
            common::{
                response_collection,
                stages::{PipelineStage, StagePhase},
            },
            context::{FinalResponse, RequestContext, TitoRequestContext},
            proto_wrapper::ProtoGenerateComplete,
            regular::{processor, streaming},
            utils,
        },
    },
    worker::AttachedBody,
};

/// Chat response processing stage
pub(crate) struct ChatResponseProcessingStage {
    processor: processor::ResponseProcessor,
    streaming_processor: Arc<streaming::StreamingProcessor>,
    tito_store: Option<Arc<TitoStore>>,
}

impl ChatResponseProcessingStage {
    pub fn new(
        processor: processor::ResponseProcessor,
        streaming_processor: Arc<streaming::StreamingProcessor>,
        tito_store: Option<Arc<TitoStore>>,
    ) -> Self {
        Self {
            processor,
            streaming_processor,
            tito_store,
        }
    }
}

#[async_trait]
impl PipelineStage for ChatResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        self.process_chat_response(ctx).await
    }

    fn name(&self) -> &'static str {
        "ChatResponseProcessing"
    }

    fn phase(&self) -> StagePhase {
        StagePhase::PostExecution
    }
}

impl ChatResponseProcessingStage {
    async fn process_chat_response(
        &self,
        ctx: &mut RequestContext,
    ) -> Result<Option<Response>, Response> {
        let is_streaming = ctx.is_streaming();

        // Extract execution result
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::execute",
                "No execution result"
            );
            error::internal_error("no_execution_result", "No execution result")
        })?;

        // Get dispatch metadata (needed by both streaming and non-streaming)
        let dispatch = ctx
            .state
            .dispatch
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "ChatResponseProcessingStage::execute",
                    "Dispatch metadata not set"
                );
                error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
            })?
            .clone();

        // Take tito_context once — consumed by one of the two paths below.
        let tito_ctx = ctx.state.tito_context.take();

        // Get cached tokenizer (resolved once in preparation stage)
        let tokenizer = ctx.tokenizer_arc().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::process_chat_response",
                "Tokenizer not cached in context"
            );
            error::internal_error(
                "tokenizer_not_cached",
                "Tokenizer not cached in context - preparation stage may have been skipped",
            )
        })?;

        if is_streaming {
            // Read derived skip_special_tokens (set in preparation, survives request_building .take())
            let skip_special_tokens = ctx
                .state
                .response
                .skip_special_tokens
                .unwrap_or_else(|| ctx.chat_request().skip_special_tokens);

            // Streaming: Use StreamingProcessor and return SSE response
            let response = self.streaming_processor.clone().process_streaming_response(
                execution_result,
                ctx.chat_request_arc(), // Cheap Arc clone (8 bytes)
                dispatch,
                tokenizer,
                skip_special_tokens,
            );

            // Attach load guards to response body for proper RAII lifecycle
            let response = match ctx.state.load_guards.take() {
                Some(guards) => AttachedBody::wrap_response(response, guards),
                None => response,
            };

            return Ok(Some(response));
        }

        // Non-streaming: Delegate to ResponseProcessor
        let request_logprobs = ctx.chat_request().logprobs;

        let chat_request = ctx.chat_request_arc();

        // Single drain — both TITO and response formatting share this Vec.
        let all_completes =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        // TITO non-streaming capture: uses already-parsed ChatCompletionMessage from first choice.
        if let (Some(tc), Some(store)) = (tito_ctx, self.tito_store.as_ref()) {
            self.do_tito_capture_non_streaming(
                ctx,
                Arc::clone(store),
                tc,
                &tokenizer,
                &all_completes,
            )
            .await?;
        }

        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::execute",
                "Stop decoder not initialized"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        let response = self
            .processor
            .process_chat_choices_from_completes(
                &all_completes,
                chat_request,
                dispatch,
                tokenizer,
                stop_decoder,
                request_logprobs,
            )
            .await?;

        // Store the final response
        ctx.state.response.final_response = Some(FinalResponse::Chat(response));

        Ok(None)
    }

    /// TITO capture for the non-streaming path.
    ///
    /// Uses the already-collected `all_completes` (no second drain) and processes
    /// the first complete into a `ChatCompletionMessage` to get tool_calls and reasoning.
    async fn do_tito_capture_non_streaming(
        &self,
        ctx: &mut RequestContext,
        store: Arc<TitoStore>,
        tito_ctx: TitoRequestContext,
        tokenizer: &Arc<dyn llm_tokenizer::traits::Tokenizer>,
        all_completes: &[ProtoGenerateComplete],
    ) -> Result<(), Response> {
        let first_complete = all_completes.first().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::do_tito_capture_non_streaming",
                session_id = %tito_ctx.session_id,
                "No completion available for TITO assistant capture"
            );
            error::internal_error(
                "tito_empty_completion",
                "No completion available for TITO assistant capture",
            )
        })?;

        if all_completes.len() > 1 {
            debug!(
                session_id = %tito_ctx.session_id,
                choices = all_completes.len(),
                "TITO capture received multiple choices; using the first choice"
            );
        }

        let assistant_message = self
            .parse_assistant_message(ctx, first_complete, tokenizer)
            .await?;

        let request_messages: &[ChatMessage] = tito_ctx.request.messages.as_slice();
        // render_context was built once in try_tito — no repeated JSON serialization.
        let render_context = &tito_ctx.render_context;

        let model_id = tito_ctx.request.model.as_str();
        let adapter = smg_tito::model_adapter::select_adapter(model_id);
        let max_trim = adapter.max_trim_tokens();
        tracing::debug!(
            session_id = %tito_ctx.session_id,
            model_id = %model_id,
            is_tito_hit = tito_ctx.is_tito_hit,
            adapter_type = std::any::type_name_of_val(&*adapter),
            max_trim_tokens = max_trim,
            "do_tito_capture_non_streaming: processing final response with adapter"
        );
        store.set_session_max_trim_tokens(&tito_ctx.session_id, max_trim);

        let output_ids = first_complete.output_ids();
        let prompt_ids = &tito_ctx.prompt_token_ids;
        let mut full_ids = Vec::with_capacity(prompt_ids.len() + output_ids.len());
        full_ids.extend_from_slice(prompt_ids);
        full_ids.extend_from_slice(output_ids);
        let all_messages =
            concat_messages_with_assistant_message(request_messages, &assistant_message);
        let mismatch_report = build_mismatch_report(
            tito_ctx.is_tito_hit,
            &full_ids,
            &all_messages,
            tokenizer,
            render_context,
            &store,
            model_id,
        );
        tracing::debug!(
            session_id = %tito_ctx.session_id,
            is_tito_hit = tito_ctx.is_tito_hit,
            prompt_len = prompt_ids.len(),
            output_len = output_ids.len(),
            full_ids_len = full_ids.len(),
            mismatch_count = mismatch_report.len(),
            "do_tito_capture_non_streaming: storing tokens"
        );
        let turn_record = extract_turn_record(first_complete, prompt_ids.len(), mismatch_report);

        match store.store(
            &tito_ctx.session_id,
            &all_messages,
            full_ids,
            turn_record,
            render_context,
            tito_ctx.trajectory_id,
        ) {
            Ok(()) => debug!(session_id = %tito_ctx.session_id, "TITO stored generation result"),
            Err(e) => {
                warn!(session_id = %tito_ctx.session_id, error = %e, "TITO store failed (non-fatal)");
            }
        };

        Ok(())
    }

    /// Process the first ProtoGenerateComplete into a ChatCompletionMessage for TITO storage.
    ///
    /// Runs the full stop_decoder + reasoning + tool parser pipeline once, producing a message
    /// that includes tool_calls and reasoning_content (same quality as the final response choice).
    async fn parse_assistant_message(
        &self,
        ctx: &mut RequestContext,
        complete: &ProtoGenerateComplete,
        tokenizer: &Arc<dyn llm_tokenizer::traits::Tokenizer>,
    ) -> Result<ChatCompletionMessage, Response> {
        let request = ctx.chat_request_arc();
        let history_tool_calls_count = utils::get_history_tool_calls_count(&request);
        let reasoning_parser_available = request.separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.processor.reasoning_parser_factory,
                self.processor.configured_reasoning_parser.as_deref(),
                &request.model,
            );
        let tool_choice_enabled = !matches!(
            &request.tool_choice,
            Some(openai_protocol::common::ToolChoice::Value(
                openai_protocol::common::ToolChoiceValue::None
            ))
        );
        let tool_parser_available = tool_choice_enabled
            && request.tools.is_some()
            && utils::check_tool_parser_availability(
                &self.processor.tool_parser_factory,
                self.processor.configured_tool_parser.as_deref(),
                &request.model,
            );
        let stop_decoder = ctx.state.response.stop_decoder.as_mut().ok_or_else(|| {
            error!(
                function = "ChatResponseProcessingStage::parse_assistant_message",
                "Stop decoder not initialized for TITO assistant capture"
            );
            error::internal_error(
                "stop_decoder_not_initialized",
                "Stop decoder not initialized",
            )
        })?;

        self.processor
            .process_single_choice(
                complete,
                0,
                &request,
                tokenizer,
                stop_decoder,
                history_tool_calls_count,
                reasoning_parser_available,
                tool_parser_available,
            )
            .await
            .map(|choice| choice.message)
            .map_err(|e| {
                error!(
                    function = "ChatResponseProcessingStage::parse_assistant_message",
                    error = %e,
                    "Failed to process assistant message for TITO capture"
                );
                error::internal_error(
                    "tito_process_assistant_failed",
                    format!("Failed to process final assistant message for TITO capture: {e}"),
                )
            })
    }
}

fn build_mismatch_report(
    is_tito_hit: bool,
    full_ids: &[u32],
    messages: &[ChatMessage],
    tokenizer: &Arc<dyn llm_tokenizer::traits::Tokenizer>,
    render_context: &RenderContext,
    store: &TitoStore,
    model_id: &str,
) -> Vec<smg_tito::MismatchEntry> {
    if !store.is_debug() || !is_tito_hit {
        return vec![];
    }

    let adapter = smg_tito::model_adapter::select_adapter(model_id);
    let assistant_start_str = adapter.assistant_start_str().map(String::from);
    let trim_trailing_ids: std::collections::HashSet<u32> =
        adapter.trailing_token_ids().iter().copied().collect();
    let validator = smg_tito::validator::TokenSeqValidator::new(
        Arc::clone(tokenizer),
        assistant_start_str,
        trim_trailing_ids,
    );
    // full_ids = prompt_ids + output_ids (complete accumulated sequence).
    // Validate against canonical retokenization with add_generation_prompt=false
    // since the assistant turn content is already in `messages`.
    validator.validate(full_ids, messages, false, render_context)
}

/// Build a `TurnRecord` from the selected completed generation.
///
/// Extracts logprobs and finish_reason once, avoiding per-site boilerplate.
fn extract_turn_record(
    complete: &ProtoGenerateComplete,
    prompt_token_count: usize,
    mismatch_report: Vec<smg_tito::MismatchEntry>,
) -> smg_tito::TurnRecord {
    let output_logprobs: Vec<(f32, u32)> = complete
        .output_logprobs()
        .map(|lp| {
            lp.token_logprobs
                .iter()
                .zip(lp.token_ids.iter())
                .map(|(prob, id)| (*prob, *id))
                .collect()
        })
        .unwrap_or_default();
    let finish_reason = complete.finish_reason().to_string();
    smg_tito::TurnRecord {
        prompt_token_count,
        output_logprobs: if output_logprobs.is_empty() {
            None
        } else {
            Some(output_logprobs)
        },
        finish_reason,
        mismatch_report,
    }
}

/// Concatenate the request messages with the assistant message.
fn concat_messages_with_assistant_message(
    request_messages: &[ChatMessage],
    assistant_message: &ChatCompletionMessage,
) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(request_messages.len() + 1);
    messages.extend_from_slice(request_messages);
    messages.push(ChatMessage::Assistant {
        content: assistant_message
            .content
            .as_ref()
            .map(|content| MessageContent::Text(content.clone())),
        name: None,
        tool_calls: assistant_message.tool_calls.clone(),
        reasoning_content: assistant_message.reasoning_content.clone(),
    });
    messages
}
