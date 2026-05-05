//! Runtime for request routing-loop dispatch.

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use axum::response::Response;
use dashmap::DashMap;
use metrics::{gauge, histogram};
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
    queue::{MultiPriorityRequestQueue, RequestPriority},
};
use crate::{
    config::RoutingLoopConfig,
    routers::{
        error as router_error,
        grpc::{
            context::{ExecutionResult, LoadGuards, RequestContext},
            harmony::ResponsesIterationResult,
            pipeline::RequestPipeline,
        },
    },
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
    pub(crate) routing: bool,
    pub(crate) queue_len: usize,
    pub(crate) running_tasks: usize,
    pub(crate) queue_keys: Vec<i32>,
}

pub struct RoutingLoopRuntime {
    queue: Mutex<MultiPriorityRequestQueue<RoutingQueueEntry>>,
    tx: mpsc::UnboundedSender<RoutingQueueEntry>,
    paused: AtomicBool,
    routing: AtomicBool,
    running_tasks: AtomicUsize,
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
}

impl RoutingLoopRuntime {
    pub(crate) fn new(
        config: &RoutingLoopConfig,
        instance_to_version_after_sync: InstanceVersionMap,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<RoutingQueueEntry>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let runtime = Arc::new(Self {
            queue: Mutex::new(MultiPriorityRequestQueue::new(
                config.request_sort_key,
                config.enable_multi_priority_queue,
            )),
            tx,
            paused: AtomicBool::new(false),
            routing: AtomicBool::new(false),
            running_tasks: AtomicUsize::new(0),
            check_interval_ms: config.check_interval_ms,
            receive_batch_size: config.receive_batch_size.max(1),
            dispatch_batch_size: config.dispatch_batch_size.max(1),
            max_running_dispatch_tasks: config.max_running_dispatch_tasks.max(1),
            instance_to_version_after_sync,
            prompt_to_running_request_ids: DashMap::new(),
            prompt_to_pinned_instance: DashMap::new(),
            ps_manager_client: OnceLock::new(),
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
                    ids.retain(|rid| *rid != request_id);
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

    pub(crate) async fn status(&self) -> RoutingLoopStatus {
        let queue = self.queue.lock().await;
        RoutingLoopStatus {
            enabled: true,
            paused: self.is_paused(),
            routing: self.routing.load(Ordering::Acquire),
            queue_len: queue.len(),
            running_tasks: self.running_tasks.load(Ordering::Acquire),
            queue_keys: queue.queue_keys(),
        }
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
            let Some(entry) = queue.pop() else {
                break;
            };
            entries.push(entry);
        }
        queue.remove_empty_partitions();
        gauge!("smg_routing_loop_queue_length").set(queue.len() as f64);
        entries
    }

    fn idle_interval(&self) -> Duration {
        Duration::from_millis(self.check_interval_ms.max(1))
    }

    fn set_routing(&self, routing: bool) {
        self.routing.store(routing, Ordering::Release);
    }

    fn task_started(&self) {
        let running = self.running_tasks.fetch_add(1, Ordering::AcqRel) + 1;
        self.set_routing(true);
        gauge!("smg_routing_loop_running_tasks").set(running as f64);
    }

    fn task_finished(&self) {
        let previous = self.running_tasks.fetch_sub(1, Ordering::AcqRel);
        if previous <= 1 {
            self.set_routing(false);
        }
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
            let entry_request_id = entry
                .routing_meta
                .as_ref()
                .map(|meta| meta.request_id);
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

async fn dispatch_entry(runtime: Arc<RoutingLoopRuntime>, entry: RoutingQueueEntry) {
    let request_id = entry.routing_meta.as_ref().map(|meta| meta.request_id);
    let prompt_id = entry.routing_meta.as_ref().map(|meta| meta.prompt_id);

    match entry.completion {
        RoutingLoopCompletion::Http(tx) => {
            let response = entry.pipeline.execute_after_preparation(entry.ctx).await;
            let _ = tx.send(response);
        }
        RoutingLoopCompletion::ChatForResponses(tx) => {
            let result = entry
                .pipeline
                .execute_chat_for_responses_after_preparation(entry.ctx)
                .await;
            let _ = tx.send(result);
        }
        RoutingLoopCompletion::HarmonyResponses(tx) => {
            let result = entry
                .pipeline
                .execute_harmony_responses_after_preparation(entry.ctx)
                .await;
            let _ = tx.send(result);
        }
        RoutingLoopCompletion::HarmonyResponsesStreaming(tx) => {
            let result = entry
                .pipeline
                .execute_harmony_responses_streaming_after_preparation(entry.ctx)
                .await;
            let _ = tx.send(result);
        }
    }

    runtime.cleanup_tracking(request_id, prompt_id);
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
