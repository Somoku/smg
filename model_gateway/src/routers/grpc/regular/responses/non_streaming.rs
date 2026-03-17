//! Non-streaming execution for Regular Responses API
//!
//! This module handles non-streaming request execution:
//! - `route_responses_internal` - Core execution orchestration
//! - `execute_tool_loop` - MCP tool loop execution
//! - `execute_without_mcp` - Simple pipeline execution without MCP

use std::sync::Arc;

use axum::{http::HeaderMap, response::Response};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    responses::{ResponseStatus, ResponsesRequest, ResponsesResponse},
};
use serde_json::json;
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput};
use tracing::{debug, error, trace, warn};

use super::{
    common::{
        build_next_request, convert_mcp_tools_to_chat_tools, extract_all_tool_calls_from_chat,
        load_conversation_history, prepare_chat_tools_and_choice, ExtractedToolCall,
        ResponsesCallContext, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        error,
        grpc::common::responses::{
            ensure_mcp_connection, persist_response_if_needed, ResponsesContext,
        },
        mcp_utils::DEFAULT_MAX_ITERATIONS,
    },
};

/// Internal implementation for non-streaming responses
///
/// This is the core execution path that:
/// 1. Loads conversation history / response chain
/// 2. Checks for MCP tools
/// 3. Executes with or without MCP tool loop
/// 4. Persists to storage
pub(super) async fn route_responses_internal(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    params: ResponsesCallContext,
) -> Result<ResponsesResponse, Response> {
    // 1. Load conversation history and build modified request
    let modified_request = load_conversation_history(ctx, &request).await?;

    // 2. Check MCP connection and get whether MCP tools are present
    let (has_mcp_tools, mcp_servers) =
        ensure_mcp_connection(&ctx.mcp_orchestrator, request.tools.as_deref()).await?;

    let responses_response = if has_mcp_tools {
        debug!("MCP tools detected, using tool loop");

        // Execute with MCP tool loop
        execute_tool_loop(ctx, modified_request, &request, &params, mcp_servers).await?
    } else {
        // No MCP tools - execute without MCP (may have function tools or no tools)
        execute_without_mcp(ctx, &modified_request, &request, params).await?
    };

    // 5. Persist response to storage if store=true
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &responses_response,
        &request,
    )
    .await;

    Ok(responses_response)
}

/// Execute request without MCP tool loop (simple pipeline execution)
pub(super) async fn execute_without_mcp(
    ctx: &ResponsesContext,
    modified_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
) -> Result<ResponsesResponse, Response> {
    // Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(modified_request).map_err(|e| {
        error!(
            function = "execute_without_mcp",
            error = %e,
            "Failed to convert ResponsesRequest to ChatCompletionRequest"
        );
        error::bad_request(
            "convert_request_failed",
            format!("Failed to convert request: {e}"),
        )
    })?;

    // PR 17 (Gap 4): route regular responses chat execution through the routing-loop
    // pipeline when enabled; fall back to direct RequestPipeline execution otherwise.
    let chat_response = execute_chat_for_responses_with_routing_loop(
        ctx,
        Arc::new(chat_request),
        params.headers,
        params.model_id,
    )
    .await?; // Preserve the Response error as-is

    // Convert ChatCompletionResponse → ResponsesResponse
    conversions::chat_to_responses(&chat_response, original_request, params.response_id).map_err(
        |e| {
            error!(
                function = "execute_without_mcp",
                error = %e,
                "Failed to convert ChatCompletionResponse to ResponsesResponse"
            );
            error::internal_error(
                "convert_to_responses_format_failed",
                format!("Failed to convert to responses format: {e}"),
            )
        },
    )
}

// PR 17 (Gap 4): central helper for regular responses execution.
// Both single-shot and MCP loop paths call this so routing-loop dispatch is consistent.
async fn execute_chat_for_responses_with_routing_loop(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
) -> Result<ChatCompletionResponse, Response> {
    if let Some(routing_loop_pipeline) = ctx.routing_loop_pipeline.as_ref() {
        routing_loop_pipeline
            .execute_chat_for_responses(chat_request, headers, model_id, ctx.components.clone())
            .await
    } else {
        ctx.pipeline
            .execute_chat_for_responses(chat_request, headers, model_id, ctx.components.clone())
            .await
    }
}

/// Execute the MCP tool calling loop
///
/// This wraps pipeline.execute_chat_for_responses() in a loop that:
/// 1. Executes the chat pipeline
/// 2. Checks if response has tool calls
/// 3. If yes, executes MCP tools and builds resume request
/// 4. Repeats until no more tool calls or limit reached
pub(super) async fn execute_tool_loop(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: &ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
) -> Result<ResponsesResponse, Response> {
    let mut state = ToolLoopState::new(original_request.input.clone());

    // Configuration: max iterations as safety limit
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    trace!(
        "Starting MCP tool loop: max_tool_calls={:?}, max_iterations={}",
        max_tool_calls,
        DEFAULT_MAX_ITERATIONS
    );

    // Create session once — bundles orchestrator, request_ctx, server_keys, mcp_tools
    let session_request_id = params
        .response_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", uuid::Uuid::now_v7()));

    let session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &session_request_id);

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&session);
    trace!(
        "Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    loop {
        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request).map_err(|e| {
            error!(
                function = "tool_loop",
                iteration = state.iteration,
                error = %e,
                "Failed to convert ResponsesRequest to ChatCompletionRequest in tool loop"
            );
            error::bad_request(
                "convert_request_failed",
                format!("Failed to convert request: {e}"),
            )
        })?;

        // Prepare tools and tool_choice for this iteration
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

        // PR 17 (Gap 4): keep MCP-loop iterations on the same routing-loop execution path.
        let chat_response = execute_chat_for_responses_with_routing_loop(
            ctx,
            Arc::new(chat_request),
            params.headers.clone(),
            params.model_id.clone(),
        )
        .await?;

        // Check for function calls (extract all for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&chat_response);

        if tool_calls.is_empty() {
            // No more tool calls, we're done
            trace!(
                "Tool loop completed: {} iterations, {} total calls",
                state.iteration,
                state.total_calls
            );

            // Convert final chat response to responses format
            let mut responses_response = conversions::chat_to_responses(
                &chat_response,
                original_request,
                params.response_id.clone(),
            )
            .map_err(|e| {
                error!(
                    function = "tool_loop",
                    iteration = state.iteration,
                    error = %e,
                    context = "final_response",
                    "Failed to convert ChatCompletionResponse to ResponsesResponse"
                );
                error::internal_error(
                    "convert_to_responses_format_failed",
                    format!("Failed to convert to responses format: {e}"),
                )
            })?;

            // Inject MCP metadata into output
            if state.total_calls > 0 {
                session
                    .inject_mcp_output_items(&mut responses_response.output, state.mcp_call_items);

                trace!(
                    "Injected MCP metadata: {} mcp_list_tools + {} mcp_call items",
                    session.mcp_servers().len(),
                    state.total_calls
                );
            }

            return Ok(responses_response);
        } else {
            state.iteration += 1;

            // Record tool loop iteration metric
            Metrics::record_mcp_tool_iteration(&current_request.model);

            trace!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls using session-exposed names.
            let (mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) =
                tool_calls
                    .into_iter()
                    .partition(|tc| session.has_exposed_tool(tc.name.as_str()));

            trace!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

            // If ANY tool call is a function tool, return to caller immediately
            if !function_tool_calls.is_empty() {
                // Convert chat response to responses format (includes all tool calls)
                let responses_response = conversions::chat_to_responses(
                    &chat_response,
                    original_request,
                    params.response_id.clone(),
                )
                .map_err(|e| {
                    error!(
                        function = "tool_loop",
                        iteration = state.iteration,
                        error = %e,
                        context = "function_tool_calls",
                        "Failed to convert ChatCompletionResponse to ResponsesResponse"
                    );
                    error::internal_error(
                        "convert_to_responses_format_failed",
                        format!("Failed to convert to responses format: {e}"),
                    )
                })?;

                // Return response with function tool calls to caller
                return Ok(responses_response);
            }

            // All MCP tools - check combined limit BEFORE executing
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    DEFAULT_MAX_ITERATIONS
                );

                // Convert chat response to responses format and mark as incomplete
                let mut responses_response = conversions::chat_to_responses(
                    &chat_response,
                    original_request,
                    params.response_id.clone(),
                )
                .map_err(|e| {
                    error!(
                        function = "tool_loop",
                        iteration = state.iteration,
                        error = %e,
                        context = "max_tool_calls_limit",
                        "Failed to convert ChatCompletionResponse to ResponsesResponse"
                    );
                    error::internal_error(
                        "convert_to_responses_format_failed",
                        format!("Failed to convert to responses format: {e}"),
                    )
                })?;

                // Mark as completed but with incomplete details
                responses_response.status = ResponseStatus::Completed;
                responses_response.incomplete_details = Some(json!({ "reason": "max_tool_calls" }));

                return Ok(responses_response);
            }

            // Convert tool calls to execution inputs
            let inputs: Vec<ToolExecutionInput> = mcp_tool_calls
                .into_iter()
                .map(|tc| ToolExecutionInput {
                    call_id: tc.call_id,
                    tool_name: tc.name,
                    arguments: serde_json::from_str(&tc.arguments).unwrap_or_else(|_| json!({})),
                })
                .collect();

            // Execute all MCP tools via session
            let results = session.execute_tools(inputs).await;

            // Process results: record metrics and state
            for result in results {
                trace!(
                    "Tool '{}' (call_id: {}) completed in {:?}, success={}",
                    result.tool_name,
                    result.call_id,
                    result.duration,
                    !result.is_error
                );

                // Record MCP tool metrics
                Metrics::record_mcp_tool_duration(
                    &current_request.model,
                    &result.tool_name,
                    result.duration,
                );
                Metrics::record_mcp_tool_call(
                    &current_request.model,
                    &result.tool_name,
                    if result.is_error {
                        metrics_labels::RESULT_ERROR
                    } else {
                        metrics_labels::RESULT_SUCCESS
                    },
                );

                // Record the call in state with transformed output item
                let output_item = result.to_response_item();
                let output_str = result.output.to_string();
                state.record_call(
                    result.call_id,
                    result.tool_name,
                    result.arguments_str,
                    output_str,
                    output_item,
                    !result.is_error,
                );

                // Increment total calls counter
                state.total_calls += 1;
            }

            // Build resume request with conversation history
            current_request = build_next_request(&state, current_request);

            // Continue to next iteration
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use axum::response::Response;
    use llm_tokenizer::{
        chat_template::ChatTemplateParams,
        traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
        MockTokenizer, TokenizerRegistry,
    };
    use openai_protocol::chat::ChatCompletionRequest;
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use smg_mcp::{McpConfig, McpOrchestrator};
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::execute_chat_for_responses_with_routing_loop;
    use crate::{
        config::{PolicyConfig, RequestSortIndicator},
        core::WorkerRegistry,
        policies::PolicyRegistry,
        routers::{
            error::HEADER_X_SMG_ERROR_CODE,
            grpc::{
                common::responses::ResponsesContext, context::SharedComponents,
                pipeline::RequestPipeline, pipeline_routing_loop::RoutingLoopPipeline,
            },
            routing_loop_utils::RoutingLoopRuntime,
        },
    };

    #[derive(Default)]
    struct TemplateMockTokenizer {
        inner: MockTokenizer,
    }

    impl Encoder for TemplateMockTokenizer {
        fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
            self.inner.encode(input, add_special_tokens)
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            self.inner.encode_batch(inputs, add_special_tokens)
        }
    }

    impl Decoder for TemplateMockTokenizer {
        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
            self.inner.decode(token_ids, skip_special_tokens)
        }
    }

    impl Tokenizer for TemplateMockTokenizer {
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            self.inner.get_special_tokens()
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.inner.token_to_id(token)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.inner.id_to_token(id)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn apply_chat_template(
            &self,
            messages: &[serde_json::Value],
            _params: ChatTemplateParams,
        ) -> anyhow::Result<String> {
            Ok(messages
                .iter()
                .filter_map(|msg| msg.get("content").and_then(|value| value.as_str()))
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    fn make_shared_components() -> Arc<SharedComponents> {
        Arc::new(SharedComponents {
            tokenizer_registry: Arc::new(TokenizerRegistry::new()),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            multimodal: None,
        })
    }

    fn make_pipeline() -> Arc<RequestPipeline> {
        Arc::new(RequestPipeline::new_regular(
            Arc::new(WorkerRegistry::new()),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            ToolParserFactory::default(),
            ReasoningParserFactory::default(),
            None,
            None,
        ))
    }

    async fn make_mcp_orchestrator() -> Arc<McpOrchestrator> {
        let empty_config = McpConfig {
            servers: vec![],
            pool: Default::default(),
            proxy: None,
            warmup: vec![],
            inventory: Default::default(),
            policy: Default::default(),
        };
        Arc::new(
            McpOrchestrator::new(empty_config)
                .await
                .expect("test MCP orchestrator should initialize"),
        )
    }

    fn make_closed_routing_loop_pipeline(
        standard_pipeline: Arc<RequestPipeline>,
    ) -> Arc<RoutingLoopPipeline> {
        let (runtime, rx) = RoutingLoopRuntime::new_with_channel(
            RequestSortIndicator::SmallId,
            false,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            0,
            String::new(),
            None,
        );
        // PR-A (Note 1): test helper returns Arc<RoutingLoopPipeline> to match ResponsesContext.
        // Force routing-loop send failure so this test can verify the helper used RL path.
        drop(rx);
        Arc::new(RoutingLoopPipeline::new(runtime, standard_pipeline))
    }

    async fn make_responses_context(with_routing_loop: bool) -> ResponsesContext {
        let components = make_shared_components();
        let pipeline = make_pipeline();

        let tokenizer_id = TokenizerRegistry::generate_id();
        components
            .tokenizer_registry
            .load(&tokenizer_id, "mock-model", "mock://tokenizer", || async {
                Ok(Arc::new(TemplateMockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .expect("mock tokenizer should register");

        let base = ResponsesContext::new(
            Arc::clone(&pipeline),
            Arc::clone(&components),
            Arc::new(MemoryResponseStorage::new()),
            Arc::new(MemoryConversationStorage::new()),
            Arc::new(MemoryConversationItemStorage::new()),
            make_mcp_orchestrator().await,
        );

        if with_routing_loop {
            base.with_routing_loop(make_closed_routing_loop_pipeline(pipeline))
        } else {
            base
        }
    }

    fn make_chat_request() -> Arc<ChatCompletionRequest> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            }))
            .expect("test chat request should deserialize"),
        )
    }

    fn error_code(response: &Response) -> Option<&str> {
        response
            .headers()
            .get(HEADER_X_SMG_ERROR_CODE)
            .and_then(|v| v.to_str().ok())
    }

    #[tokio::test]
    async fn test_execute_chat_for_responses_prefers_routing_loop_when_available() {
        let ctx = make_responses_context(true).await;

        let err = execute_chat_for_responses_with_routing_loop(
            &ctx,
            make_chat_request(),
            None,
            Some("mock-model".to_string()),
        )
        .await
        .expect_err("closed routing-loop channel should return error");

        assert_eq!(error_code(&err), Some("routing_loop_send_failed"));
    }

    #[tokio::test]
    async fn test_execute_chat_for_responses_falls_back_without_routing_loop() {
        let ctx = make_responses_context(false).await;

        let err = execute_chat_for_responses_with_routing_loop(
            &ctx,
            make_chat_request(),
            None,
            Some("mock-model".to_string()),
        )
        .await
        .expect_err("no-worker fallback path should error in test setup");

        assert_ne!(error_code(&err), Some("routing_loop_send_failed"));
    }

    #[tokio::test]
    async fn test_with_routing_loop_stores_shared_arc_instance() {
        // PR-A (Note 2): ensure context wiring keeps the same Arc instance.
        let components = make_shared_components();
        let pipeline = make_pipeline();

        let base = ResponsesContext::new(
            Arc::clone(&pipeline),
            Arc::clone(&components),
            Arc::new(MemoryResponseStorage::new()),
            Arc::new(MemoryConversationStorage::new()),
            Arc::new(MemoryConversationItemStorage::new()),
            make_mcp_orchestrator().await,
        );

        let rl_pipeline = make_closed_routing_loop_pipeline(Arc::clone(&pipeline));
        let ctx = base.with_routing_loop(Arc::clone(&rl_pipeline));

        let stored = ctx
            .routing_loop_pipeline
            .as_ref()
            .expect("routing loop pipeline should be present");
        assert!(Arc::ptr_eq(stored, &rl_pipeline));
    }
}
