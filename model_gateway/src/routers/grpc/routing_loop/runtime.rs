//! Runtime for request routing-loop dispatch.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use axum::{
    http::HeaderValue,
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use metrics::{counter, gauge, histogram};
use openai_protocol::chat::ChatCompletionResponse;
use serde::Serialize;
use tokio::{
    sync::{mpsc, oneshot, Mutex},
    task::{yield_now, JoinSet},
    time::sleep,
};
use tracing::error;

use super::{
    metadata::RoutingMeta,
    partial_rollout::{
        drain_stream_for_partial_rollout, merge_into_partial_state, reset_ctx_for_loopback,
    },
    queue::{MultiPriorityRequestQueue, RequestPriority},
};
use crate::{
    config::RoutingLoopConfig,
    routers::{
        error as router_error,
        grpc::{
            context::{
                ExecutionResult, FinalResponse, LoadGuards, RequestContext, WorkerSelection,
            },
            harmony::ResponsesIterationResult,
            pipeline::RequestPipeline,
        },
    },
    worker::{Worker, WorkerRegistry},
};

pub(crate) type InstanceVersionMap = Arc<DashMap<(String, usize), i64>>;

pub(crate) struct RoutingQueueEntry {
    pub(crate) ctx: RequestContext,
    pub(crate) pipeline: RequestPipeline,
    pub(crate) completion: RoutingLoopCompletion,
    pub(crate) routing_meta: Option<RoutingMeta>,
}

pub(crate) enum RoutingLoopCompletion {
    Http(oneshot::Sender<Response>),
    ChatForResponses(oneshot::Sender<Result<ChatCompletionResponse, Response>>),
    HarmonyResponses(oneshot::Sender<Result<ResponsesIterationResult, Response>>),
    HarmonyResponsesStreaming(
        oneshot::Sender<Result<(ExecutionResult, Option<LoadGuards>), Response>>,
    ),
}

impl RequestPriority for RoutingQueueEntry {
    fn version_tag(&self) -> i64 {
        self.routing_meta
            .as_ref()
            .map_or(-1, |meta| meta.version_tag)
    }

    fn is_validation(&self) -> bool {
        self.routing_meta
            .as_ref()
            .is_some_and(|meta| meta.is_validate)
    }

    fn input_len(&self) -> usize {
        self.ctx
            .state
            .preparation
            .as_ref()
            .map_or(0, |preparation| preparation.token_ids().len())
    }

    fn request_id(&self) -> Option<i64> {
        self.routing_meta.as_ref().map(|meta| meta.request_id)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RoutingLoopStatus {
    pub(crate) enabled: bool,
    pub(crate) paused: bool,
    pub(crate) selecting: bool,
    pub(crate) queue_len: usize,
    pub(crate) pending_request_num: usize,
    pub(crate) running_tasks: usize,
    pub(crate) running_request_num: usize,
    pub(crate) max_running_dispatch_tasks: usize,
    pub(crate) queue_keys: Vec<i32>,
    pub(crate) partition_queue_lens: HashMap<i32, usize>,
}

/// Lightweight metadata snapshot for a single queued request.
///
/// Returned by `RoutingLoopRuntime::filter_queue_by_version_tag` and serialised
/// as JSON by the `GET /routing_loop/filter` endpoint.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct QueuedRequestMeta {
    pub(crate) request_id: i64,
    pub(crate) version_tag: i64,
    pub(crate) prompt_id: i64,
}

pub struct RoutingLoopRuntime {
    queue: Mutex<MultiPriorityRequestQueue<RoutingQueueEntry>>,
    tx: mpsc::UnboundedSender<RoutingQueueEntry>,
    paused: AtomicBool,
    running_tasks: AtomicUsize,
    running_selections: AtomicUsize,
    running_requests: AtomicUsize,
    check_interval_ms: u64,
    receive_batch_size: usize,
    dispatch_batch_size: usize,
    max_running_dispatch_tasks: usize,
    /// `(base_worker_id, dp_rank)` → last synced version tag.
    ///
    /// Uses a sharded concurrent map because Stage 1 is on the hot dispatch
    /// path and should not serialize all readers on a global lock.
    pub(crate) instance_to_version_after_sync: InstanceVersionMap,

    /// `prompt_id` → running `request_id`s sharing that prompt.
    pub(crate) prompt_to_running_request_ids: DashMap<i64, Vec<i64>>,

    /// `prompt_id` → the instance pinned to that prompt group.
    ///
    /// Written once (first placement) via `entry().or_insert()` so that
    /// concurrent Stage 3 reads see either *no entry* (free selection) or a
    /// committed instance — eliminating the TOCTOU window between Stage 3
    /// read and post-selection write.
    pub(crate) prompt_to_pinned_instance: DashMap<i64, (String, usize)>,

    /// PS Manager gRPC client.
    ///
    /// `None` when the strategy is `"naive"` or no address was configured.
    /// Initialised once during router start-up via `connect_ps_manager()`.
    /// `OnceLock` avoids a Mutex on the read hot path.
    pub(crate) ps_manager_client: OnceLock<Arc<psrl_state::PSManagerStateClient>>,

    /// Worker registry used to resolve stable worker IDs during partial-rollout
    /// loopback — specifically to call `reserve_id_for_url` for header injection.
    pub(crate) worker_registry: Arc<WorkerRegistry>,
}

impl RoutingLoopRuntime {
    pub(crate) fn new(
        config: &RoutingLoopConfig,
        instance_to_version_after_sync: InstanceVersionMap,
        worker_registry: Arc<WorkerRegistry>,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<RoutingQueueEntry>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let runtime = Arc::new(Self {
            queue: Mutex::new(MultiPriorityRequestQueue::new(
                config.request_sort_key,
                config.enable_multi_priority_queue,
            )),
            tx,
            paused: AtomicBool::new(false),
            running_tasks: AtomicUsize::new(0),
            running_selections: AtomicUsize::new(0),
            running_requests: AtomicUsize::new(0),
            check_interval_ms: config.check_interval_ms,
            receive_batch_size: config.receive_batch_size.max(1),
            dispatch_batch_size: config.dispatch_batch_size.max(1),
            max_running_dispatch_tasks: config.max_running_dispatch_tasks.max(1),
            instance_to_version_after_sync,
            prompt_to_running_request_ids: DashMap::new(),
            prompt_to_pinned_instance: DashMap::new(),
            ps_manager_client: OnceLock::new(),
            worker_registry,
        });
        (runtime, rx)
    }

    pub(crate) async fn connect_ps_manager(
        &self,
        addr: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.ps_manager_client.get().is_some() {
            return Ok(());
        }
        let client = psrl_state::PSManagerStateClient::connect(addr).await?;
        // If another task raced us here the value is already set; that's fine.
        let _ = self.ps_manager_client.set(Arc::new(client));
        Ok(())
    }

    pub(crate) fn record_selected_instance(
        &self,
        request_id: i64,
        prompt_id: Option<i64>,
        instance: (String, usize),
    ) {
        if let Some(prompt_id) = prompt_id {
            // Atomically claim the group pin for this prompt if not yet set.
            // `entry().or_insert()` holds the DashMap shard write-lock during
            // the check-and-insert, so at most one caller ever writes the value.
            self.prompt_to_pinned_instance
                .entry(prompt_id)
                .or_insert(instance);

            let mut entry = self
                .prompt_to_running_request_ids
                .entry(prompt_id)
                .or_default();
            if !entry.contains(&request_id) {
                entry.push(request_id);
                self.running_requests.fetch_add(1, Ordering::AcqRel);
            }
        }
    }

    pub(crate) fn cleanup_tracking(&self, request_id: Option<i64>, prompt_id: Option<i64>) {
        let Some(request_id) = request_id else {
            return;
        };

        if let Some(prompt_id) = prompt_id {
            let remove_prompt = self
                .prompt_to_running_request_ids
                .get_mut(&prompt_id)
                .map(|mut ids| {
                    let before = ids.len();
                    ids.retain(|rid| *rid != request_id);
                    if ids.len() < before {
                        self.running_requests.fetch_sub(1, Ordering::AcqRel);
                    }
                    ids.is_empty()
                })
                .unwrap_or(false);
            if remove_prompt {
                self.prompt_to_running_request_ids.remove(&prompt_id);
                // All requests in the group have completed; release the pin so
                // a future group for the same prompt_id can be freely placed.
                self.prompt_to_pinned_instance.remove(&prompt_id);
            }
        }
    }

    pub(crate) fn instance_id_for_worker(&self, worker: &Arc<dyn Worker>) -> (String, usize) {
        let base_worker_id = self
            .worker_registry
            .reserve_id_for_url(worker.base_url())
            .as_str()
            .to_string();
        (base_worker_id, worker.dp_rank().unwrap_or(0))
    }

    pub(crate) fn enqueue(&self, entry: RoutingQueueEntry) -> Result<(), Box<RoutingQueueEntry>> {
        self.tx.send(entry).map_err(|err| Box::new(err.0))
    }

    pub(crate) fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }

    pub(crate) fn resume(&self) {
        self.paused.store(false, Ordering::Release);
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// Returns `true` while at least one dispatch task is still inside the
    /// worker-selection stage.
    pub(crate) fn is_selecting(&self) -> bool {
        self.running_selections.load(Ordering::Acquire) > 0
    }

    /// Acquire a guard that increments `running_selections` and decrements it
    /// when dropped (including on panic / `.await` cancellation, because
    /// `Drop::drop` runs unconditionally).
    pub(crate) fn selection_guard(self: &Arc<Self>) -> SelectionGuard {
        self.running_selections.fetch_add(1, Ordering::AcqRel);
        gauge!("smg_routing_loop_running_selections")
            .set(self.running_selections.load(Ordering::Acquire) as f64);
        SelectionGuard {
            runtime: Arc::clone(self),
        }
    }

    pub(crate) async fn status(&self) -> RoutingLoopStatus {
        let queue = self.queue.lock().await;
        let queue_len = queue.len();
        RoutingLoopStatus {
            enabled: true,
            paused: self.is_paused(),
            selecting: self.is_selecting(),
            queue_len,
            pending_request_num: queue_len,
            running_tasks: self.running_tasks.load(Ordering::Acquire),
            running_request_num: self.running_requests.load(Ordering::Acquire),
            max_running_dispatch_tasks: self.max_running_dispatch_tasks,
            queue_keys: queue.queue_keys(),
            partition_queue_lens: queue.per_partition_sizes(),
        }
    }

    /// Return metadata for all queued entries whose `version_tag ≤ max_version_tag`.
    ///
    /// Used by `GET /routing_loop/filter` to let the sync coordinator check
    /// whether old-version requests have fully drained from the queue.
    pub(crate) async fn filter_queue_by_version_tag(
        &self,
        max_version_tag: i64,
    ) -> Vec<QueuedRequestMeta> {
        let queue = self.queue.lock().await;
        queue
            .iter_requests()
            .filter_map(|entry| {
                entry.routing_meta.as_ref().and_then(|meta| {
                    (meta.version_tag <= max_version_tag).then_some(QueuedRequestMeta {
                        request_id: meta.request_id,
                        version_tag: meta.version_tag,
                        prompt_id: meta.prompt_id,
                    })
                })
            })
            .collect()
    }

    async fn push_entries(&self, entries: Vec<RoutingQueueEntry>) {
        let mut queue = self.queue.lock().await;
        for entry in entries {
            queue.push(entry);
        }
        let queue_len = queue.len() as f64;
        // Update per-version-partition gauges.
        for (key, size) in queue.per_partition_sizes() {
            gauge!(
                "smg_routing_loop_partition_queue_length",
                "version" => key.to_string(),
            )
            .set(size as f64);
        }
        gauge!("smg_routing_loop_queue_length").set(queue_len);
    }

    async fn pop_entries(&self, max_entries: usize) -> Vec<RoutingQueueEntry> {
        let mut queue = self.queue.lock().await;
        let mut entries = Vec::with_capacity(max_entries);
        for _ in 0..max_entries {
            let Some(entry) = queue.pop() else { break };
            entries.push(entry);
        }
        queue.remove_empty_partitions();
        gauge!("smg_routing_loop_queue_length").set(queue.len() as f64);
        entries
    }

    fn idle_interval(&self) -> Duration {
        Duration::from_millis(self.check_interval_ms.max(1))
    }

    fn task_started(&self) {
        let running = self.running_tasks.fetch_add(1, Ordering::AcqRel) + 1;
        gauge!("smg_routing_loop_running_tasks").set(running as f64);
    }

    fn task_finished(&self) {
        let previous = self.running_tasks.fetch_sub(1, Ordering::AcqRel);
        let running = if previous > 0 { previous - 1 } else { 0 };
        gauge!("smg_routing_loop_running_tasks").set(running as f64);
    }
}

pub(crate) async fn run_routing_loop(
    runtime: Arc<RoutingLoopRuntime>,
    mut rx: mpsc::UnboundedReceiver<RoutingQueueEntry>,
) {
    let mut dispatch_tasks = JoinSet::new();

    loop {
        while let Some(join_result) = dispatch_tasks.try_join_next() {
            if let Err(err) = join_result {
                error!(error = %err, "routing loop dispatch task failed");
            }
        }

        let entries = drain_receiver(&mut rx, runtime.receive_batch_size);
        if !entries.is_empty() {
            runtime.push_entries(entries).await;
        }

        if runtime.is_paused() {
            sleep(runtime.idle_interval()).await;
            continue;
        }

        let running_tasks = runtime.running_tasks.load(Ordering::Acquire);
        if running_tasks >= runtime.max_running_dispatch_tasks {
            tokio::select! {
                maybe_entry = rx.recv() => {
                    match maybe_entry {
                        Some(entry) => {
                            let mut entries = Vec::with_capacity(runtime.receive_batch_size);
                            entries.push(entry);
                            entries.extend(drain_receiver(&mut rx, runtime.receive_batch_size - 1));
                            runtime.push_entries(entries).await;
                        }
                        None => break,
                    }
                }
                join_result = dispatch_tasks.join_next(), if !dispatch_tasks.is_empty() => {
                    if let Some(Err(err)) = join_result {
                        error!(error = %err, "routing loop dispatch task failed");
                    }
                }
                () = sleep(runtime.idle_interval()) => {}
            }
            continue;
        }

        let dispatch_limit = runtime
            .dispatch_batch_size
            .min(runtime.max_running_dispatch_tasks - running_tasks);
        let entries = runtime.pop_entries(dispatch_limit).await;
        let dispatched = !entries.is_empty();
        let aborted_ids = check_aborted_requests(&runtime, &entries).await;
        for entry in entries {
            let entry_request_id = entry.routing_meta.as_ref().map(|meta| meta.request_id);
            if entry_request_id.is_some_and(|request_id| aborted_ids.contains(&request_id)) {
                let prompt_id = entry.routing_meta.as_ref().map(|meta| meta.prompt_id);
                send_aborted(entry);
                runtime.cleanup_tracking(entry_request_id, prompt_id);
                continue;
            }

            let runtime_for_task = Arc::clone(&runtime);
            runtime_for_task.task_started();
            dispatch_tasks.spawn(async move {
                let dispatch_start = std::time::Instant::now();
                dispatch_entry(Arc::clone(&runtime_for_task), entry).await;
                histogram!("smg_routing_loop_dispatch_duration_seconds")
                    .record(dispatch_start.elapsed().as_secs_f64());
                runtime_for_task.task_finished();
            });
        }
        if dispatched {
            yield_now().await;
            continue;
        }

        tokio::select! {
            maybe_entry = rx.recv() => {
                match maybe_entry {
                    Some(entry) => {
                        let mut entries = Vec::with_capacity(runtime.receive_batch_size);
                        entries.push(entry);
                        entries.extend(drain_receiver(&mut rx, runtime.receive_batch_size - 1));
                        runtime.push_entries(entries).await;
                    }
                    None => break,
                }
            }
            join_result = dispatch_tasks.join_next(), if !dispatch_tasks.is_empty() => {
                if let Some(Err(err)) = join_result {
                    error!(error = %err, "routing loop dispatch task failed");
                }
            }
            () = sleep(runtime.idle_interval()) => {}
        }
    }

    while let Some(join_result) = dispatch_tasks.join_next().await {
        if let Err(err) = join_result {
            error!(error = %err, "routing loop dispatch task failed during shutdown");
        }
    }
}

async fn check_aborted_requests(
    runtime: &Arc<RoutingLoopRuntime>,
    entries: &[RoutingQueueEntry],
) -> HashSet<i64> {
    let Some(ps_client) = runtime.ps_manager_client.get() else {
        return HashSet::new();
    };

    let request_ids: Vec<i64> = entries
        .iter()
        .filter_map(|entry| entry.routing_meta.as_ref().map(|meta| meta.request_id))
        .collect();
    if request_ids.is_empty() {
        return HashSet::new();
    }

    match ps_client
        .check_aborted_requests(request_ids.clone(), true)
        .await
    {
        Ok(flags) => request_ids
            .into_iter()
            .zip(flags)
            .filter_map(|(request_id, aborted)| aborted.then_some(request_id))
            .collect(),
        Err(err) => {
            error!(error = %err, "check_aborted_requests RPC failed; continuing dispatch");
            HashSet::new()
        }
    }
}

fn send_aborted(entry: RoutingQueueEntry) {
    let response = router_error::bad_request("request_aborted", "Request aborted by PS Manager");
    match entry.completion {
        RoutingLoopCompletion::Http(tx) => {
            let _ = tx.send(response);
        }
        RoutingLoopCompletion::ChatForResponses(tx) => {
            let _ = tx.send(Err(response));
        }
        RoutingLoopCompletion::HarmonyResponses(tx) => {
            let _ = tx.send(Err(response));
        }
        RoutingLoopCompletion::HarmonyResponsesStreaming(tx) => {
            let _ = tx.send(Err(response));
        }
    }
}

async fn dispatch_entry(runtime: Arc<RoutingLoopRuntime>, mut entry: RoutingQueueEntry) {
    let request_id = entry.routing_meta.as_ref().map(|meta| meta.request_id);
    let prompt_id = entry.routing_meta.as_ref().map(|meta| meta.prompt_id);

    if entry.routing_meta.is_none() {
        // ── Worker selection (drain barrier for `pause?wait=true`) ───────────
        // The guard increments `running_selections` until the future
        // completes; PSRL's post-select reserve happens inside this call.
        let select_result = {
            let _guard = runtime.selection_guard();
            entry
                .pipeline
                .execute_worker_selection(&mut entry.ctx)
                .await
        };
        if select_result.is_err() {
            // Worker selection found no available worker — re-enqueue.
            runtime.push_entries(vec![entry]).await;
            return;
        }

        // ── Run remaining execution stages (client acquisition → dispatch) ──
        if let Err(response) = entry
            .pipeline
            .execute_post_selection_execution(&mut entry.ctx)
            .await
        {
            // A post-selection execution stage failed — propagate to caller.
            match entry.completion {
                RoutingLoopCompletion::Http(tx) => {
                    let _ = tx.send(response);
                }
                RoutingLoopCompletion::ChatForResponses(tx) => {
                    let _ = tx.send(Err(response));
                }
                RoutingLoopCompletion::HarmonyResponses(tx) => {
                    let _ = tx.send(Err(response));
                }
                RoutingLoopCompletion::HarmonyResponsesStreaming(tx) => {
                    let _ = tx.send(Err(response));
                }
            }
            runtime.cleanup_tracking(request_id, prompt_id);
            return;
        }

        // ── Execution succeeded — run post-execution stages and send result ──
        match entry.completion {
            RoutingLoopCompletion::Http(tx) => {
                let response = match entry.pipeline.execute_remaining_stages(&mut entry.ctx).await {
                    Ok(Some(r)) => r,
                    Ok(None) => extract_final_response(&mut entry.ctx),
                    Err(r) => r,
                };
                let _ = tx.send(response);
            }
            RoutingLoopCompletion::ChatForResponses(tx) => {
                let result = match entry.pipeline.execute_remaining_stages(&mut entry.ctx).await {
                    Ok(_) => match entry.ctx.state.response.final_response.take() {
                        Some(FinalResponse::Chat(r)) => Ok(r),
                        Some(_) => Err(router_error::internal_error(
                            "wrong_response_type",
                            "Wrong response type for ChatForResponses",
                        )),
                        None => Err(router_error::internal_error(
                            "no_response_produced",
                            "No response produced",
                        )),
                    },
                    Err(r) => Err(r),
                };
                let _ = tx.send(result);
            }
            RoutingLoopCompletion::HarmonyResponses(tx) => {
                let result = match entry.pipeline.execute_remaining_stages(&mut entry.ctx).await {
                    Ok(_) => entry
                        .ctx
                        .state
                        .response
                        .responses_iteration_result
                        .take()
                        .ok_or_else(|| {
                            router_error::internal_error(
                                "no_responses_iteration_result",
                                "No ResponsesIterationResult produced",
                            )
                        }),
                    Err(r) => Err(r),
                };
                let _ = tx.send(result);
            }
            RoutingLoopCompletion::HarmonyResponsesStreaming(tx) => {
                let result = match entry.pipeline.execute_remaining_stages(&mut entry.ctx).await {
                    Ok(_) => match entry.ctx.state.response.execution_result.take() {
                        Some(execution_result) => {
                            let load_guards = entry.ctx.state.load_guards.take();
                            Ok((execution_result, load_guards))
                        }
                        None => Err(router_error::internal_error(
                            "no_execution_result",
                            "No execution result produced",
                        )),
                    },
                    Err(r) => Err(r),
                };
                let _ = tx.send(result);
            }
        }
        runtime.cleanup_tracking(request_id, prompt_id);
    } else {
        dispatch_entry_with_partial_rollout(runtime, entry, request_id, prompt_id).await;
    }
}

/// Extract a `Response` from `ctx.state.response.final_response`
fn extract_final_response(ctx: &mut RequestContext) -> Response {
    match ctx.state.response.final_response.take() {
        Some(FinalResponse::Chat(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Generate(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Completion(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Embedding(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Classify(r)) => axum::Json(r).into_response(),
        Some(FinalResponse::Messages(r)) => axum::Json(r).into_response(),
        None => router_error::internal_error("no_response_produced", "No response produced"),
    }
}

/// Send the final `Response` through the appropriate completion channel.
fn send_http_completion(completion: RoutingLoopCompletion, response: Response) {
    match completion {
        RoutingLoopCompletion::Http(tx) => {
            let _ = tx.send(response);
        }
        // PSRL dispatch only uses Http completion; other variants are not
        // reachable here, but we log an error rather than panic.
        other => {
            error!(
                "dispatch_entry_with_partial_rollout: unexpected completion type; dropping response"
            );
            drop(other);
        }
    }
}

/// Full partial-rollout loopback loop for PSRL requests.
///
/// Runs pipeline stages through execution on each iteration.  If the stream
/// finishes with `"abort"` (PS weight-sync interrupted generation), the
/// accumulated tokens are preserved and the request is re-routed to the
/// newly-synced instance.  The loop terminates on `"stop"` / `"length"`.
async fn dispatch_entry_with_partial_rollout(
    runtime: Arc<RoutingLoopRuntime>,
    entry: RoutingQueueEntry,
    request_id: Option<i64>,
    prompt_id: Option<i64>,
) {
    let RoutingQueueEntry {
        mut ctx,
        pipeline,
        completion,
        routing_meta,
    } = entry;

    let mut partial_state = ctx.state.partial_rollout_state.take().unwrap_or_default();

    // Snapshot `preparation` before the loop starts
    ctx.state.preparation_snapshot = ctx.state.preparation.clone();

    loop {
        // ── Step 1: publish accumulated response-token count for selection ───
        let headers = ctx.input.headers.get_or_insert_with(Default::default);
        if let Ok(v) = HeaderValue::from_str(&partial_state.response_token_count().to_string()) {
            headers.insert("x-response-token-count", v);
        }

        // ── Step 2: run execution-phase stages (worker select → dispatch) ────
        let select_result = {
            let _guard = runtime.selection_guard();
            pipeline.execute_worker_selection(&mut ctx).await
        };
        if select_result.is_err() {
            // Worker selection found no available worker — restore accumulated
            // partial-rollout state and re-enqueue for the next tick.
            ctx.state.partial_rollout_state = Some(partial_state);
            runtime
                .push_entries(vec![RoutingQueueEntry {
                    ctx,
                    pipeline,
                    completion,
                    routing_meta,
                }])
                .await;
            return;
        }

        if let Err(response) = pipeline.execute_post_selection_execution(&mut ctx).await {
            send_http_completion(completion, response);
            runtime.cleanup_tracking(request_id, prompt_id);
            return;
        }

        // ── Step 3: extract the raw gRPC stream ──────────────────────────────
        let stream = match ctx.state.response.execution_result.take() {
            Some(ExecutionResult::Single { stream }) => stream,
            Some(ExecutionResult::Dual { decode, .. }) => *decode,
            other => {
                // Non-generate result (embedding, etc.) — put it back and run
                // post-execution stages normally (no loopback needed).
                ctx.state.response.execution_result = other;
                let response = match pipeline.execute_remaining_stages(&mut ctx).await {
                    Ok(Some(r)) => r,
                    Ok(None) => extract_final_response(&mut ctx),
                    Err(r) => r,
                };
                send_http_completion(completion, response);
                runtime.cleanup_tracking(request_id, prompt_id);
                return;
            }
        };

        // ── Step 4: drain the stream, collecting tokens and finish_reason ────
        let mut stream = stream;
        let drained = match drain_stream_for_partial_rollout(&mut stream).await {
            Ok(d) => d,
            Err(msg) => {
                let response = router_error::internal_error("stream_drain_failed", msg.as_str());
                send_http_completion(completion, response);
                runtime.cleanup_tracking(request_id, prompt_id);
                return;
            }
        };

        // ── Step 5: branch on finish_reason ─────────────────────────────────
        match drained.finish_reason.as_str() {
            "stop" | "length" => {
                // ── ROLLOUT_COMPLETED ────────────────────────────────────────
                merge_into_partial_state(&mut partial_state, &drained);

                // Emit completion metrics.
                histogram!("smg_partial_rollout_loopback_count")
                    .record(partial_state.iteration_count as f64);
                let accumulated_tokens = partial_state.token_ids.len();
                histogram!("smg_partial_rollout_accumulated_tokens")
                    .record(accumulated_tokens as f64);
                counter!("smg_partial_rollout_completed_total").increment(1);

                // Override the output_ids in the final complete frame with the
                // full accumulated token sequence from all loopback iterations.
                //
                // Multi-iteration: also rewrite `output_logprobs` so the frame
                // is internally consistent — without this, downstream TITO
                // turn-record extraction sees `output_ids = merged` but
                // `output_logprobs = last_iteration_only`, which surfaces in
                // PSRL training-array construction as a "trailing trim
                // overflow" because the per-token logprobs no longer line up
                // with the accumulated token sequence.
                //
                // Single-iteration: the complete frame already carries
                // aligned outputs from its sole iteration, so we keep the
                // (cheaper) `set_output_ids` path. `iteration_count == 1` is
                // guaranteed by `merge_into_partial_state` incrementing
                // exactly once per drained segment (including the final
                // non-abort one).
                let mut complete = drained.complete;
                let merged_token_ids = std::mem::take(&mut partial_state.token_ids);
                if partial_state.iteration_count > 1 {
                    complete.set_partial_rollout_outputs(
                        merged_token_ids,
                        partial_state.logprobs.take(),
                    );
                } else {
                    complete.set_output_ids(merged_token_ids);
                }

                // Place the assembled complete frame back for PostExecution stages.
                ctx.state.response.execution_result = Some(ExecutionResult::Complete(complete));

                let response = match pipeline.execute_remaining_stages(&mut ctx).await {
                    Ok(Some(r)) => r,
                    Ok(None) => extract_final_response(&mut ctx),
                    Err(r) => r,
                };
                send_http_completion(completion, response);
                runtime.cleanup_tracking(request_id, prompt_id);
                return;
            }
            "abort" => {
                // ── ROLLOUT_INTERRUPTED — loopback to newly-synced instance ──
                merge_into_partial_state(&mut partial_state, &drained);
                counter!("smg_partial_rollout_abort_total").increment(1);

                // Determine the instance that was just used so we can pin the
                // loopback request to the same worker (which now has the fresh
                // weights after the sync that triggered the abort).
                let worker = match ctx.state.workers.as_ref() {
                    Some(WorkerSelection::Single { worker }) => worker.clone(),
                    Some(WorkerSelection::Dual { decode, .. }) => decode.clone(),
                    None => {
                        let response = router_error::internal_error(
                            "loopback_no_instance",
                            "no worker selected",
                        );
                        send_http_completion(completion, response);
                        runtime.cleanup_tracking(request_id, prompt_id);
                        return;
                    }
                };
                let (base_id, dp_rank) = runtime.instance_id_for_worker(&worker);

                // Reset ctx for the next iteration (clears workers, clients,
                // proto_request, dispatch, load_guards, and execution_result).
                reset_ctx_for_loopback(&mut ctx);

                // Inject loopback routing headers so that on the next iteration
                // `parse_routing_request_meta_from_context` picks up the hint.
                let headers = ctx.input.headers.get_or_insert_with(Default::default);
                if let Ok(v) = HeaderValue::from_str(&base_id) {
                    headers.insert("x-base-worker-id", v);
                }
                if let Ok(v) = HeaderValue::from_str(&dp_rank.to_string()) {
                    headers.insert("x-target-dp-rank", v);
                }

                // continue loop
            }
            other => {
                error!(
                    finish_reason = other,
                    request_id = ?request_id,
                    "unexpected finish_reason in partial rollout; terminating"
                );
                let response = router_error::internal_error("unexpected_finish_reason", other);
                send_http_completion(completion, response);
                runtime.cleanup_tracking(request_id, prompt_id);
                return;
            }
        }
    }
}

fn drain_receiver(
    rx: &mut mpsc::UnboundedReceiver<RoutingQueueEntry>,
    max_entries: usize,
) -> Vec<RoutingQueueEntry> {
    let mut entries = Vec::with_capacity(max_entries);
    for _ in 0..max_entries {
        let Ok(entry) = rx.try_recv() else {
            break;
        };
        entries.push(entry);
    }
    entries
}

pub(crate) struct SelectionGuard {
    runtime: Arc<RoutingLoopRuntime>,
}

impl Drop for SelectionGuard {
    fn drop(&mut self) {
        self.runtime
            .running_selections
            .fetch_sub(1, Ordering::AcqRel);
        gauge!("smg_routing_loop_running_selections")
            .set(self.runtime.running_selections.load(Ordering::Acquire) as f64);
    }
}
