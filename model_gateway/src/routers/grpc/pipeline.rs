//! Pipeline orchestrator for gRPC router request processing
//!
//! This module defines the RequestPipeline orchestrator that coordinates
//! the execution of pipeline stages from request preparation to response delivery.

use std::{sync::Arc, time::Instant};

use axum::response::{IntoResponse, Response};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::ClassifyRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{debug, error};

// Import embedding-specific and classify-specific stages
use super::regular::stages::classify::ClassifyResponseProcessingStage;
use super::{
    common::{responses::ResponsesContext, stages::*},
    context::*,
    harmony,
    regular::{
        processor,
        stages::{
            embedding::{
                preparation::EmbeddingPreparationStage,
                request_building::EmbeddingRequestBuildingStage,
                response_processing::EmbeddingResponseProcessingStage,
            },
            *,
        },
        streaming,
    },
    utils::error_type_from_status,
};
use crate::{
    core::{WorkerRegistry, UNKNOWN_MODEL_ID},
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    policies::PolicyRegistry,
    routers::error,
};

/// Generic request pipeline for all request types
///
/// Orchestrates all stages from request preparation to response delivery.
/// Configured differently for regular vs PD mode.
#[derive(Clone)]
pub(crate) struct RequestPipeline {
    stages: Arc<Vec<Box<dyn PipelineStage>>>,
    /// Backend type for metrics labeling
    backend_type: &'static str,
}

impl RequestPipeline {
    /// Create a regular (single-worker) pipeline
    pub fn new_regular(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        let processor = processor::ResponseProcessor::new(
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            configured_tool_parser.clone(),
            configured_reasoning_parser.clone(),
        );

        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
            metrics_labels::BACKEND_REGULAR,
        ));

        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(PreparationStage::new()),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::Regular,
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(RequestBuildingStage::new(false)), // No PD metadata
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::Single)),
            Box::new(ResponseProcessingStage::new(processor, streaming_processor)),
        ];

        Self {
            stages: Arc::new(stages),
            backend_type: metrics_labels::BACKEND_REGULAR,
        }
    }

    /// Create a Harmony (single-worker) pipeline for Harmony-capable models
    pub fn new_harmony(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        _tool_parser_factory: ToolParserFactory,
        _reasoning_parser_factory: ReasoningParserFactory,
        _configured_tool_parser: Option<String>,
        _configured_reasoning_parser: Option<String>,
    ) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(harmony::stages::HarmonyPreparationStage::new()),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::Regular,
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(harmony::stages::HarmonyRequestBuildingStage::new(false)),
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::Single)),
            Box::new(harmony::stages::HarmonyResponseProcessingStage::new()),
        ];

        Self {
            stages: Arc::new(stages),
            backend_type: metrics_labels::BACKEND_REGULAR,
        }
    }

    /// Create a Harmony PD (prefill-decode) pipeline
    #[expect(dead_code)]
    pub fn new_harmony_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        _tool_parser_factory: ToolParserFactory,
        _reasoning_parser_factory: ReasoningParserFactory,
        _configured_tool_parser: Option<String>,
        _configured_reasoning_parser: Option<String>,
    ) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(harmony::stages::HarmonyPreparationStage::new()),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::PrefillDecode,
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(harmony::stages::HarmonyRequestBuildingStage::new(true)),
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::DualDispatch)),
            Box::new(harmony::stages::HarmonyResponseProcessingStage::new()),
        ];

        Self {
            stages: Arc::new(stages),
            backend_type: metrics_labels::BACKEND_PD,
        }
    }

    /// Create a PD (prefill-decode) pipeline
    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        let processor = processor::ResponseProcessor::new(
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            configured_tool_parser.clone(),
            configured_reasoning_parser.clone(),
        );

        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
            metrics_labels::BACKEND_PD,
        ));

        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(PreparationStage::new()),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::PrefillDecode,
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(RequestBuildingStage::new(true)), // Inject PD metadata
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::DualDispatch)),
            Box::new(ResponseProcessingStage::new(processor, streaming_processor)),
        ];

        Self {
            stages: Arc::new(stages),
            backend_type: metrics_labels::BACKEND_PD,
        }
    }

    /// Create an embeddings pipeline
    pub fn new_embeddings(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(EmbeddingPreparationStage::new()),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::Regular, // Embeddings are always single
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(EmbeddingRequestBuildingStage::new()),
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::Single)),
            Box::new(EmbeddingResponseProcessingStage::new()),
        ];

        Self {
            stages: Arc::new(stages),
            backend_type: metrics_labels::BACKEND_REGULAR, // Embeddings are regular for now
        }
    }

    /// Create a classify pipeline
    ///
    /// Classify reuses embedding stages for preparation and request building,
    /// but uses its own response processing for softmax + label mapping.
    pub fn new_classify(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(EmbeddingPreparationStage::new()),
            Box::new(WorkerSelectionStage::new(
                worker_registry,
                policy_registry,
                WorkerSelectionMode::Regular, // Classify is always single worker
            )),
            Box::new(ClientAcquisitionStage),
            Box::new(EmbeddingRequestBuildingStage::new()),
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::Single)),
            Box::new(ClassifyResponseProcessingStage::new()),
        ];

        Self {
            stages: Arc::new(stages),
            backend_type: metrics_labels::BACKEND_REGULAR,
        }
    }

    /// Execute the complete pipeline for a chat request
    pub async fn execute_chat(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        let start = Instant::now();
        // Clone Arc for metrics (cheap atomic increment) to avoid borrow issues
        let request_for_metrics = Arc::clone(&request);
        let streaming = request.stream;

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &request_for_metrics.model,
            metrics_labels::ENDPOINT_CHAT,
            bool_to_static_str(streaming),
        );

        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage completed with streaming response - record success and return
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &request_for_metrics.model,
                        metrics_labels::ENDPOINT_CHAT,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &request_for_metrics.model,
                        metrics_labels::ENDPOINT_CHAT,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &request_for_metrics.model,
                    metrics_labels::ENDPOINT_CHAT,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(FinalResponse::Generate(_))
            | Some(FinalResponse::Embedding(_))
            | Some(FinalResponse::Classify(_)) => {
                error!(
                    function = "execute_chat",
                    "Wrong response type: expected Chat, got Generate/Embedding/Classify"
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &request_for_metrics.model,
                    metrics_labels::ENDPOINT_CHAT,
                    metrics_labels::ERROR_INTERNAL,
                );
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_chat",
                    "No response produced by pipeline"
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &request_for_metrics.model,
                    metrics_labels::ENDPOINT_CHAT,
                    metrics_labels::ERROR_INTERNAL,
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute the complete pipeline for a generate request
    pub async fn execute_generate(
        &self,
        request: Arc<GenerateRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        let start = Instant::now();
        let streaming = request.stream;

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
            metrics_labels::ENDPOINT_GENERATE,
            bool_to_static_str(streaming),
        );

        let mut ctx = RequestContext::for_generate(request, headers, model_id.clone(), components);

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        metrics_labels::ENDPOINT_GENERATE,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        metrics_labels::ENDPOINT_GENERATE,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Generate(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                    metrics_labels::ENDPOINT_GENERATE,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(FinalResponse::Chat(_))
            | Some(FinalResponse::Embedding(_))
            | Some(FinalResponse::Classify(_)) => {
                error!(
                    function = "execute_generate",
                    "Wrong response type: expected Generate, got Chat/Embedding/Classify"
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                    metrics_labels::ENDPOINT_GENERATE,
                    metrics_labels::ERROR_INTERNAL,
                );
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_generate",
                    "No response produced by pipeline"
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                    metrics_labels::ENDPOINT_GENERATE,
                    metrics_labels::ERROR_INTERNAL,
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute the complete pipeline for an embedding request
    pub async fn execute_embeddings(
        &self,
        request: Arc<EmbeddingRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        debug!(
            "execute_embeddings: Starting execution for model: {}",
            model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID)
        );
        let start = Instant::now();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
            metrics_labels::ENDPOINT_EMBEDDINGS,
            bool_to_static_str(false),
        );

        let mut ctx = RequestContext::for_embedding(request, headers, model_id.clone(), components);

        for stage in self.stages.iter() {
            debug!("execute_embeddings: Executing stage: {}", stage.name());
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    debug!(
                        "execute_embeddings: Stage {} returned final response.",
                        stage.name()
                    );
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        metrics_labels::ENDPOINT_EMBEDDINGS,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => {
                    debug!(
                        "execute_embeddings: Stage {} completed, continuing to next stage.",
                        stage.name()
                    );
                    continue;
                }
                Err(response) => {
                    error!(
                        "execute_embeddings: Stage {} failed with status {:?}, returning error response.",
                        stage.name(),
                        response.status()
                    );
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        metrics_labels::ENDPOINT_EMBEDDINGS,
                        error_type_from_status(response.status()),
                    );
                    return response;
                }
            }
        }

        debug!(
            "execute_embeddings: Pipeline finished, processing final_response. Current state: {:?}",
            ctx.state.response.final_response
        );
        match ctx.state.response.final_response {
            Some(FinalResponse::Embedding(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                    metrics_labels::ENDPOINT_EMBEDDINGS,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(_) => {
                error!(function = "execute_embeddings", "Wrong response type");
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_embeddings",
                    "No final response produced by pipeline."
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute the complete pipeline for a classify request
    pub async fn execute_classify(
        &self,
        request: Arc<ClassifyRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        debug!(
            "execute_classify: Starting execution for model: {}",
            model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID)
        );
        let start = Instant::now();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
            metrics_labels::ENDPOINT_CLASSIFY,
            bool_to_static_str(false), // Classify is never streaming
        );

        let mut ctx = RequestContext::for_classify(request, headers, model_id.clone(), components);

        for stage in self.stages.iter() {
            debug!("execute_classify: Executing stage: {}", stage.name());
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    debug!(
                        "execute_classify: Stage {} returned final response.",
                        stage.name()
                    );
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        metrics_labels::ENDPOINT_CLASSIFY,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => {
                    debug!(
                        "execute_classify: Stage {} completed, continuing to next stage.",
                        stage.name()
                    );
                    continue;
                }
                Err(response) => {
                    error!(
                        "execute_classify: Stage {} failed with status {:?}, returning error response.",
                        stage.name(),
                        response.status()
                    );
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        metrics_labels::ENDPOINT_CLASSIFY,
                        error_type_from_status(response.status()),
                    );
                    return response;
                }
            }
        }

        debug!(
            "execute_classify: Pipeline finished, processing final_response. Current state: {:?}",
            ctx.state.response.final_response
        );
        match ctx.state.response.final_response {
            Some(FinalResponse::Classify(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                    metrics_labels::ENDPOINT_CLASSIFY,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(_) => {
                error!(function = "execute_classify", "Wrong response type");
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_classify",
                    "No final response produced by pipeline."
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute chat pipeline for responses endpoint
    ///
    /// Used by ALL non-streaming /v1/responses requests.
    /// Uses the same 7 pipeline stages as execute_chat(), with two differences:
    /// 1. Returns Result<ChatCompletionResponse, Response> for tool_loop composition
    /// 2. Disallows streaming (responses endpoint uses different SSE format)
    pub async fn execute_chat_for_responses(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Result<ChatCompletionResponse, Response> {
        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(_response)) => {
                    // Streaming not supported for responses sync mode
                    error!(
                        function = "execute_chat_for_responses",
                        "Streaming attempted in responses context"
                    );
                    return Err(error::bad_request(
                        "streaming_not_supported",
                        "Streaming is not supported in this context".to_string(),
                    ));
                }
                Ok(None) => {
                    continue;
                }
                Err(response) => {
                    // Error occurred - return the response as-is to preserve HTTP status codes
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => Ok(response),
            Some(FinalResponse::Generate(_))
            | Some(FinalResponse::Embedding(_))
            | Some(FinalResponse::Classify(_)) => {
                error!(
                    function = "execute_chat_for_responses",
                    "Wrong response type: expected Chat, got Generate/Embedding/Classify"
                );
                Err(error::internal_error(
                    "wrong_response_type",
                    "Internal error: wrong response type",
                ))
            }
            None => {
                error!(
                    function = "execute_chat_for_responses",
                    "No response produced by pipeline"
                );
                Err(error::internal_error(
                    "no_response_produced",
                    "No response produced",
                ))
            }
        }
    }

    /// Execute Harmony Responses API request through all pipeline stages
    ///
    /// This method runs a single iteration of the Responses API request,
    /// returning either ToolCallsFound (continue serving) or Completed (final response).
    ///
    /// Called by harmony::responses::serve_harmony_responses() for each iteration.
    ///
    /// # Arguments
    ///
    /// * `request` - Responses API request
    /// * `ctx` - Harmony Responses context with MCP manager and components
    ///
    /// # Returns
    ///
    /// ResponsesIterationResult indicating whether to continue iteration or return
    pub async fn execute_harmony_responses(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
    ) -> Result<harmony::ResponsesIterationResult, Response> {
        // Create RequestContext for this Responses request
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None, // No headers needed for internal pipeline execution
            None, // Model ID already set in request
            harmony_ctx.components.clone(),
        );

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage returned early response (e.g., streaming) - not expected for Responses iteration
                    error!(
                        "Stage {} ({}) returned unexpected response during Responses iteration",
                        idx + 1,
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => {
                    continue;
                }
                Err(response) => {
                    // Stage failed
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract ResponsesIterationResult from context
        // This should have been set by HarmonyResponseProcessingStage
        ctx.state
            .response
            .responses_iteration_result
            .take()
            .ok_or_else(|| {
                error!(
                    function = "execute_harmony_responses",
                    "No ResponsesIterationResult produced by pipeline"
                );
                error::internal_error(
                    "no_responses_iteration_result",
                    "No ResponsesIterationResult produced by pipeline",
                )
            })
    }

    /// Execute Harmony Responses pipeline iteration with streaming support
    ///
    /// This version executes the pipeline up to the dispatch stage and returns
    /// the raw ExecutionResult (with stream) and LoadGuards for token-level streaming processing.
    /// The caller is responsible for keeping load_guards alive until stream processing completes.
    pub async fn execute_harmony_responses_streaming(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
    ) -> Result<(ExecutionResult, Option<LoadGuards>), Response> {
        // Create RequestContext for this Responses request
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None,
            None,
            harmony_ctx.components.clone(),
        );

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    error!(
                        "Stage {} ({}) returned unexpected response during streaming Responses",
                        idx + 1,
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => continue,
                Err(response) => {
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract execution_result (the raw stream from workers) and load_guards
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "execute_harmony_responses_streaming",
                "No ExecutionResult produced by pipeline"
            );
            error::internal_error(
                "no_execution_result_produced",
                "No ExecutionResult produced by pipeline",
            )
        })?;

        let load_guards = ctx.state.load_guards.take();

        Ok((execution_result, load_guards))
    }

    // PR 10 §10.4: execute_preparation_only — run only the preparation stage.
    /// Run only Stage 0 (Preparation) and return the prepared `RequestContext`.
    ///
    /// Used by router handlers that need to run preparation before deciding
    /// whether to route the request through the PSRL routing loop or the
    /// normal pipeline. The returned context is submitted to the routing loop
    /// as a `RoutingQueueEntry` (PSRL path) or the full pipeline resumes
    /// from `execute_chat`/`execute_generate` (non-PSRL path).
    ///
    /// Returns `Ok(ctx)` on success, `Err(response)` if preparation fails.
    pub async fn execute_preparation_only(
        &self,
        ctx: RequestContext,
    ) -> Result<RequestContext, Response> {
        // Stage 0 is always PreparationStage (index 0 in self.stages)
        let prep_stage = self
            .stages
            .first()
            .ok_or_else(|| error::internal_error("empty_pipeline", "Pipeline has no stages"))?;

        let mut ctx = ctx;
        match prep_stage.execute(&mut ctx).await {
            Ok(None) => Ok(ctx),
            Ok(Some(response)) => {
                // Preparation should never return an early response
                error!(
                    function = "execute_preparation_only",
                    stage = prep_stage.name(),
                    "Preparation stage returned unexpected early response"
                );
                Err(response)
            }
            Err(response) => Err(response),
        }
    }

    // PR 12 §12.1: execute_response_processing_only — run only the last stage (ResponseProcessing)
    // on a RequestContext that already has execution_result set (used by PSRL dispatch after drain).
    /// Execute only the last pipeline stage (ResponseProcessing) on a `RequestContext`
    /// whose `execution_result` has already been populated (e.g., `ExecutionResult::PreDrained`).
    ///
    /// Used by the PSRL dispatch task after `drain_stream_for_partial_rollout()` converts
    /// the raw stream into a `PreDrained` execution result. The caller is responsible for
    /// setting `ctx.state.response.execution_result` before calling this method.
    ///
    /// Returns the final `axum::Response`.
    pub async fn execute_response_processing_only(&self, mut ctx: RequestContext) -> Response {
        // The last stage (index stages.len()-1) is always ResponseProcessing.
        let last_stage = match self.stages.last() {
            Some(s) => s,
            None => {
                error!(
                    function = "execute_response_processing_only",
                    "Pipeline has no stages"
                );
                return error::internal_error("empty_pipeline", "Pipeline has no stages");
            }
        };

        match last_stage.execute(&mut ctx).await {
            Ok(Some(response)) => response,
            Ok(None) => {
                // ResponseProcessing set final_response in ctx; extract it.
                match ctx.state.response.final_response {
                    Some(response) => match response {
                        FinalResponse::Chat(r) => axum::Json(r).into_response(),
                        FinalResponse::Generate(r) => axum::Json(r).into_response(),
                        FinalResponse::Embedding(r) => axum::Json(r).into_response(),
                        FinalResponse::Classify(r) => axum::Json(r).into_response(),
                    },
                    None => {
                        error!(
                            function = "execute_response_processing_only",
                            "No response produced by ResponseProcessing stage"
                        );
                        error::internal_error("no_response_produced", "No response produced")
                    }
                }
            }
            Err(response) => {
                error!(
                    "ResponseProcessing stage failed with status {}",
                    response.status()
                );
                response
            }
        }
    }

    // PR 12 §12.1: execute_through_execution — run stages 2-5 (ClientAcquisition through
    // RequestExecution) and return the raw ExecutionResult before ResponseProcessing.
    /// Execute stages 2–5 (ClientAcquisition, RequestBuilding, DispatchMetadata, Execution)
    /// on a `RequestContext` where preparation (stage 0) and worker selection (stage 1)
    /// have already been completed by the routing loop.
    ///
    /// Used by the PSRL dispatch task for partial rollout interception.
    /// The caller drains the returned `ExecutionResult` stream and decides whether to
    /// continue (stop/length) or loopback (abort) before running ResponseProcessing.
    ///
    /// Returns `Ok((ctx, execution_result))` on success, `Err(response)` if any stage fails.
    pub async fn execute_through_execution(
        &self,
        mut ctx: RequestContext,
    ) -> Result<(RequestContext, ExecutionResult), Response> {
        // Stages 2-5: skip stage 0 (Preparation) and stage 1 (WorkerSelection),
        // and stop before stage 6 (ResponseProcessing — last stage).
        // Stage indices: 0=Prep, 1=WorkerSelection, 2=ClientAcq, 3=ReqBuilding,
        //                4=DispatchMeta, 5=RequestExecution, 6=ResponseProcessing
        let stages_to_run = self.stages.iter().skip(2).take(self.stages.len() - 3);

        for stage in stages_to_run {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Unexpected early response from a pre-execution stage.
                    error!(
                        "Stage {} returned unexpected early response in execute_through_execution",
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => continue,
                Err(response) => {
                    error!(
                        "Stage {} failed with status {} in execute_through_execution",
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract ExecutionResult from the context (set by RequestExecutionStage).
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "execute_through_execution",
                "No ExecutionResult produced by pipeline (RequestExecutionStage did not run?)"
            );
            error::internal_error(
                "no_execution_result",
                "No ExecutionResult produced by pipeline",
            )
        })?;

        Ok((ctx, execution_result))
    }
}
