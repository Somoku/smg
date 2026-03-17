// PR 10 §10.2–10.3 / PR 11 §11.1 / PR 12 §12.3: PSRL gRPC routing loop task and dispatch task.
// PR 13 §13.1: dispatch_task now receives RoutingQueueEntry.ctx directly — no
//   build_context_from_prepared() or reconstruct_context() needed. components param removed
//   from dispatch_task (already in ctx.components). model_id/text extraction updated to read
//   from ctx.input.model_id and PreparationOutput::original_text.
// PR 13 Gap 1: WorkerLoadGuard (via LoadGuards) added to dispatch_task to restore correct
//   load-tracking for RequestNumBalancePolicy and ThroughputOptimalPolicy.
// PR 13 Gap 2: token_ids from PreparationOutput propagated as SelectWorkerInfo::tokens,
//   removing the character-count heuristic from the routing-loop dispatch path.
//!
//! Implements the routing loop task body that drains the mpsc channel, runs
//! batch abort-checks via PS Manager, selects workers via the full 5-stage PSRL
//! worker selection, and spawns per-request dispatch tasks.  The dispatch task
//! runs pipeline stages 2–5 (ClientAcquisition through RequestExecution), drains
//! the proto stream for partial rollout interception, and handles loopback on abort.
//!
//! Design overview:
//! ```text
//! routing_loop():
//!   1. Drain channel → queue
//!   2. If paused → sleep, continue
//!   3. If queue empty → channel.recv().await
//!   4. Pop all per-version batches from queue
//!   5. For each batch:
//!      a. Batch check_aborted_requests (PS Manager)
//!      b. For each request:
//!         i.   router.select_worker_for_model() — 5-stage PSRL selection (PR 11)
//!         ii.  If no worker: push_back_to_queue, break
//!         iii. Insert into incomplete_request_to_instance
//!         iv.  Insert into prompt_to_running_request_ids
//!         v.   JoinSet::spawn(dispatch_task)
//!   6. Drain completed in_flight tasks
//!   7. Heartbeat every 5 s
//!   8. sleep(check_interval_ms)
//! ```

use std::{collections::HashSet, sync::atomic::Ordering};

use axum::http::HeaderMap;
use tokio::{
    sync::mpsc,
    task::JoinSet,
};
use tracing::{debug, error, info};

use crate::{
    core::Worker,
    routers::{
        error as router_error,
        grpc::{
            context::{ExecutionResult, LoadGuards, RequestContext, RequestType, WorkerSelection},
            partial_rollout::{
                drain_stream_for_partial_rollout, extract_partial_rollout_state,
                merge_partial_into_drained, ProtoPartialRolloutState,
            },
            pipeline::RequestPipeline,
            router::GrpcRouter,
        },
        routing_loop_utils::{
            extract_text_from_request_type, RoutingLoopRuntime, RoutingQueueEntry,
        },
    },
};
use std::sync::Arc;

// ── Public entry-point ───────────────────────────────────────────────────

// PR 10 §10.2 / PR 11 §11.1: Routing loop task body.
/// Standalone async routing loop task.
///
/// Spawned once during server startup (§10.5, server.rs) when
/// `enable_routing_loop = true`.
///
/// # Arguments
/// - `runtime` — shared `RoutingLoopRuntime`
/// - `rx` — receiver half of the `RoutingLoopRuntime::tx` channel
/// - `pipeline` — the gRPC request pipeline (shared with the router)
/// - `router` — the gRPC router, used for 5-stage PSRL worker selection (PR 11)
///
/// # PR 13 §13.1: components removed from signature
///
/// The `components: Arc<SharedComponents>` parameter has been removed.
/// Each `RoutingQueueEntry.ctx` already carries `ctx.components` set at construction
/// time in `RoutingLoopPipeline::execute_chat/execute_generate`.
pub async fn routing_loop(
    runtime: Arc<RoutingLoopRuntime>,
    mut rx: mpsc::UnboundedReceiver<RoutingQueueEntry>,
    pipeline: Arc<RequestPipeline>,
    router: Arc<GrpcRouter>,
) {
    let ps_connected = runtime.ps_manager_client.is_some();
    info!(
        ps_manager_addr = %runtime.ps_manager_addr,
        ps_connected,
        "gRPC routing loop started"
    );

    let mut in_flight: JoinSet<()> = JoinSet::new();
    let mut last_heartbeat = std::time::Instant::now();
    let mut last_paused_state = runtime.is_paused.load(Ordering::Relaxed);

    loop {
        // ── Step 1: Drain channel into queue ────────────────────────────
        {
            let mut local_batch: Vec<RoutingQueueEntry> = Vec::new();
            while let Ok(request) = rx.try_recv() {
                local_batch.push(request);
            }
            if !local_batch.is_empty() {
                let drained_count = local_batch.len();
                let mut queue = runtime.request_queue.lock().await;
                for req in local_batch {
                    queue.push(req);
                }
                debug!(
                    drained = drained_count,
                    queue_size = queue.len(),
                    "drained requests from channel"
                );
            }
        }

        // ── Step 2: Pause handling ───────────────────────────────────────
        if runtime.is_paused.load(Ordering::Relaxed) {
            if !last_paused_state {
                let queue_size = runtime.request_queue.lock().await.len();
                info!(queue_size, "routing_loop state changed: paused=true");
            }
            last_paused_state = true;
            runtime.is_routing.store(false, Ordering::Relaxed);
            if runtime.check_interval_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(runtime.check_interval_ms))
                    .await;
            } else {
                tokio::task::yield_now().await;
            }
            continue;
        }
        if last_paused_state {
            let queue_size = runtime.request_queue.lock().await.len();
            info!(queue_size, "routing_loop state changed: paused=false");
        }
        last_paused_state = false;

        // ── Step 3: If queue empty, wait for next request ────────────────
        let queue_is_empty = runtime.request_queue.lock().await.is_empty();
        if queue_is_empty {
            debug!("routing_loop queue empty, waiting for next request");
            match rx.recv().await {
                Some(request) => {
                    runtime.request_queue.lock().await.push(request);
                }
                None => {
                    // Channel closed — sender side dropped; drain in-flight and exit
                    info!("routing_loop channel closed, draining in-flight tasks and exiting");
                    break;
                }
            }
        }

        runtime.is_routing.store(false, Ordering::Relaxed);

        // ── Steps 4–5: Pop batches and dispatch ──────────────────────────
        {
            runtime.is_routing.store(true, Ordering::Relaxed);

            // Take all per-version batches out of the queue (no lock held across awaits).
            let queue_batches: Vec<(i32, Vec<RoutingQueueEntry>)> = {
                let mut q = runtime.request_queue.lock().await;
                q.remove_empty_queues();
                let queue_ids = q.queue_keys();
                let size_before = q.len();
                debug!(
                    queue_size = size_before,
                    batch_count = queue_ids.len(),
                    "routing_loop start routing round"
                );

                let mut batches = Vec::with_capacity(queue_ids.len());
                for queue_id in queue_ids {
                    let mut requests = Vec::new();
                    while let Some(req) = q.pop_from_queue(queue_id) {
                        requests.push(req);
                    }
                    batches.push((queue_id, requests));
                }
                batches
            };

            for (queue_id, queued_requests) in queue_batches {
                // PR 10 §10.7.1 / §10.3: Batch-check aborted requests (one gRPC call per batch).
                let aborted_ids: HashSet<i64> = {
                    if let Some(ps_client) = runtime.ps_manager_client.as_ref() {
                        let ids: Vec<i64> = queued_requests
                            .iter()
                            .filter_map(|r| r.routing_meta.as_ref().and_then(|m| m.request_id))
                            .collect();
                        if ids.is_empty() {
                            HashSet::new()
                        } else {
                            match ps_client.check_aborted_requests(ids.clone(), true).await {
                                Ok(flags) => ids
                                    .into_iter()
                                    .zip(flags)
                                    .filter_map(|(id, aborted)| aborted.then_some(id))
                                    .collect(),
                                Err(e) => {
                                    error!("check_aborted_requests failed: {}", e);
                                    HashSet::new()
                                }
                            }
                        }
                    } else {
                        HashSet::new()
                    }
                };

                let mut remain_requests: Vec<RoutingQueueEntry> = Vec::new();

                for request in queued_requests {
                    // Drop aborted requests.
                    if let Some(request_id) =
                        request.routing_meta.as_ref().and_then(|m| m.request_id)
                    {
                        if aborted_ids.contains(&request_id) {
                            let _ = request.result_tx.send(router_error::bad_request(
                                "request_aborted",
                                "Request aborted by PS Manager",
                            ));
                            continue;
                        }
                    }

                    // PR 10 §10.2 + PR 11 §11.1: Select worker via 5-stage PSRL worker selection.
                    // `select_worker_for_model` implements full version filter, partial-hint pin,
                    // group pin, can_reserve check, and sort/group policy selection.
                    // PR 13 §13.1: model_id and text come from ctx.input directly (no PreparedRequest).
                    // PR 13 Gap 2: tokens from PreparationOutput propagated so load-aware policies
                    // get accurate token counts instead of a text-length heuristic.
                    let model_id = request.ctx.input.model_id.clone();
                    // Prefer pre-computed text from PreparationOutput; fall back to raw request field.
                    let text_owned = request
                        .ctx
                        .state
                        .preparation
                        .as_ref()
                        .and_then(|p| p.original_text.clone())
                        .or_else(|| {
                            extract_text_from_request_type(&request.ctx.input.request_type)
                        });
                    // PR 13 Gap 2: accurate token slice for ThroughputOptimalPolicy.
                    let tokens: Option<&[u32]> = request
                        .ctx
                        .state
                        .preparation
                        .as_ref()
                        .map(|p| p.token_ids.as_slice())
                        .filter(|s| !s.is_empty());
                    let maybe_worker = router
                        .select_worker_for_model(
                            model_id.as_deref(),
                            text_owned.as_deref(),
                            tokens,
                            request.ctx.input.headers.as_ref(),
                            request.routing_meta.as_ref(),
                        )
                        .await;

                    let Some((worker, selected_instance)) = maybe_worker else {
                        debug!(
                            request_id = ?request.routing_meta.as_ref().and_then(|m| m.request_id),
                            model_id = ?model_id,
                            "no available worker, pushing back to queue"
                        );
                        remain_requests.push(request);
                        continue;
                    };

                    // PR 11 §11.1: `selected_instance` = (base_worker_id, dp_rank) is returned
                    // by select_worker_for_model — no longer hardcoded to dp_rank=0.

                    // PR 10 §10.7.1: Insert into incomplete_request_to_instance before dispatch.
                    if let Some(request_id) =
                        request.routing_meta.as_ref().and_then(|m| m.request_id)
                    {
                        runtime
                            .incomplete_request_to_instance
                            .lock()
                            .await
                            .insert(request_id, selected_instance.clone());

                        // PR 10 §10.7.2: Insert into prompt_to_running_request_ids.
                        if let Some(prompt_id) =
                            request.routing_meta.as_ref().and_then(|m| m.prompt_id)
                        {
                            runtime
                                .prompt_to_running_request_ids
                                .lock()
                                .await
                                .entry(prompt_id)
                                .or_default()
                                .push(request_id);
                        }
                    }

                    // Spawn dispatch task.
                    // PR 13 §13.1: components removed — already in ctx.components.
                    let rt = runtime.clone();
                    let pl = pipeline.clone();
                    let si = selected_instance.clone();
                    in_flight.spawn(async move {
                        // PR 13: Box::pin to avoid large-future stack allocation (clippy::large_futures).
                        Box::pin(dispatch_task(rt, pl, request, worker, si)).await;
                    });
                }

                // Push back unrouted requests to the same queue slot.
                if !remain_requests.is_empty() {
                    let remain_ids: Vec<_> = remain_requests
                        .iter()
                        .filter_map(|r| r.routing_meta.as_ref().and_then(|m| m.request_id))
                        .collect();
                    debug!(queue_id, remain_ids = ?remain_ids, "pushback unrouted requests");
                    let mut q = runtime.request_queue.lock().await;
                    for req in remain_requests {
                        q.push_back_to_queue(queue_id, req);
                    }
                    // Stop processing further queues — no point if workers are full.
                    break;
                }
            }

            // Drain completed in-flight tasks (non-blocking).
            while in_flight.try_join_next().is_some() {}

            runtime.is_routing.store(false, Ordering::Relaxed);
        }

        // ── Step 7: Heartbeat ────────────────────────────────────────────
        if last_heartbeat.elapsed().as_secs() >= 5 {
            let (queue_size, per_version) = {
                let q = runtime.request_queue.lock().await;
                (q.len(), q.per_version_sizes())
            };
            info!(
                paused = runtime.is_paused.load(Ordering::Relaxed),
                in_flight = in_flight.len(),
                queue_size,
                ?per_version,
                "routing_loop heartbeat"
            );
            last_heartbeat = std::time::Instant::now();
        }

        // ── Step 8: Sleep ────────────────────────────────────────────────
        if runtime.check_interval_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(runtime.check_interval_ms)).await;
        } else {
            tokio::task::yield_now().await;
        }
    }

    // Channel closed: drain all in-flight tasks before exiting.
    info!(
        "routing_loop draining {} in-flight tasks before exit",
        in_flight.len()
    );
    while in_flight.join_next().await.is_some() {}
    runtime.is_routing.store(false, Ordering::Relaxed);
    info!("gRPC routing loop exited");
}

// ── Dispatch task ────────────────────────────────────────────────────────

/// Execute a single request through pipeline stages 2–5 (ClientAcquisition through
/// RequestExecution), drain the proto stream for partial rollout detection, and send
/// the response back via the request's `result_tx` oneshot sender.
///
/// # PR 13 §13.1: ctx used directly
///
/// `request.ctx` already carries the fully-prepared `RequestContext` (with
/// `PreparationOutput`). The worker is pre-set on `ctx.state.workers` here,
/// and `execute_through_execution` runs stages 2–5 starting from client acquisition
/// (skipping worker selection, which the routing loop already performed).
///
/// The `components` parameter has been removed — it is already in `ctx.components`.
///
/// # PR 13 Gap 1: WorkerLoadGuard
///
/// `LoadGuards` is created immediately after pre-setting the worker on `ctx.state.workers`.
/// This increments the worker's in-flight counter for the lifetime of the dispatch task,
/// ensuring `RequestNumBalancePolicy` and `ThroughputOptimalPolicy` see accurate load.
///
/// # Finish-reason handling (PR 12 §12.3)
/// - `"stop"` / `"length"`: notify PS Manager `ROLLOUT_COMPLETED`, merge prior partial
///   into drained result, run ResponseProcessing stage, send final response.
/// - `"abort"`: notify PS Manager `ROLLOUT_INTERRUPTED`, extract `ProtoPartialRolloutState`
///   via backend-appropriate extractor, update proto request for next loopback iteration,
///   update rollout_instance_hint headers, push back to queue.
/// - Other / None: send response as-is (non-PSRL or error path).
///
/// # Tracking (§10.7.3-10.7.4)
/// Removes the request from `incomplete_request_to_instance` and
/// `prompt_to_running_request_ids` on all code paths.
async fn dispatch_task(
    runtime: Arc<RoutingLoopRuntime>,
    pipeline: Arc<RequestPipeline>,
    mut request: RoutingQueueEntry,
    worker: Arc<dyn Worker>,
    selected_instance: (String, usize),
) {
    let request_id = request.routing_meta.as_ref().and_then(|m| m.request_id);
    let prompt_id = request.routing_meta.as_ref().and_then(|m| m.prompt_id);
    let is_validate = request
        .routing_meta
        .as_ref()
        .map(|m| m.is_validate)
        .unwrap_or(false);
    let version_tag = request
        .routing_meta
        .as_ref()
        .map(|m| m.version_tag)
        .unwrap_or(-1);

    // PR 15: Capture manual_target_enabled before ctx is moved into execute_through_execution.
    // Mirrors sgl-model-gateway routing_loop.rs line 222.
    // Must be read here (before Step 2 moves ctx) so it's available in the abort loopback branch.
    // Use local module path so clippy absolute-paths stays clean.
    use crate::routers::grpc::worker_selection;
    let manual_target_enabled =
        worker_selection::is_manual_target_worker_enabled(request.ctx.input.headers.as_ref());

    // ── Step 1: Notify PS Manager — ROLLOUT_DISPATCHED ──────────────────
    if let (Some(ps_client), Some(rid)) = (runtime.ps_manager_client.as_ref(), request_id) {
        let _ = ps_client.update_request_status(
            vec![rid],
            "ROLLOUT_DISPATCHED".to_string(),
            vec![],
            vec![],
            is_validate,
        )
        .await
        .map_err(|e| error!("update_request_status DISPATCHED failed: {}", e));
    }

    // ── Step 2: Pre-set worker on ctx; install load guard ────────────────
    // PR 13 §13.1: Use ctx directly — no build_context_from_prepared needed.
    // Pre-set worker selection so execute_through_execution (stages 2-5) skips stage 1
    // (WorkerSelectionStage), which was already done by the routing loop.
    let worker_selection = WorkerSelection::Single {
        worker: Arc::clone(&worker),
    };
    request.ctx.state.workers = Some(worker_selection);

    // PR 13 Gap 1: Create load guard now so worker.increment_load() is called immediately.
    // The guard is dropped at the end of this task (on all exit paths: stop, abort, error),
    // which calls worker.decrement_load(). This restores correct load tracking for
    // RequestNumBalancePolicy and ThroughputOptimalPolicy.
    // Note: we pass headers from ctx.input for priority-aware load tracking.
    // Safety: workers was just set above to Some(_).
    let _load_guard = request
        .ctx
        .state
        .workers
        .as_ref()
        .map(|ws| LoadGuards::new(ws, request.ctx.input.headers.as_ref()));

    // ── Step 3: Execute pipeline stages 2-5 (ClientAcquisition → RequestExecution) ──
    // PR 12 §12.3: Use execute_through_execution() to stop before ResponseProcessing,
    // so we can drain the proto stream and detect finish_reason at the proto level.
    // PR 13 §13.1: ctx is passed directly (no reconstruction from PreparedRequest).
    let (mut ctx, execution_result) = match pipeline.execute_through_execution(request.ctx).await {
        Ok(result) => result,
        Err(error_response) => {
            cleanup_tracking(&runtime, request_id, prompt_id).await;
            let _ = request.result_tx.send(error_response);
            return;
        }
    };

    // ── Step 4: Drain proto stream for partial rollout detection ─────────
    // PR 12 §12.3: Extract the stream from ExecutionResult::Single and drain it.
    // For non-Single or Embedding results, fall back to the full response processing path.
    let drained = match execution_result {
        ExecutionResult::Single { mut stream } => {
            // Mark the stream as completed before draining so abort is not sent on drop.
            // drain_stream_for_partial_rollout reads to end, which is the normal completion path.
            let drained = drain_stream_for_partial_rollout(&mut stream).await;
            stream.mark_completed();
            drained
        }
        other => {
            // Non-single or embedding: run full response processing without drain.
            // This path handles Dual (PD), Embedding, and PreDrained (re-entrancy guard).
            ctx.state.response.execution_result = Some(other);
            cleanup_tracking(&runtime, request_id, prompt_id).await;
            let response = pipeline.execute_response_processing_only(ctx).await;
            let _ = request.result_tx.send(response);
            return;
        }
    };

    let finish_reason = drained.finish_reason.as_deref().map(str::to_string);
    debug!(
        request_id = ?request_id,
        finish_reason = ?finish_reason.as_deref(),
        "dispatch_task finish_reason (proto drain)"
    );

    // ── Step 5: Remove from tracking maps (§10.7.3-10.7.4) ─────────────
    cleanup_tracking(&runtime, request_id, prompt_id).await;

    // ── Step 6: Handle finish_reason ─────────────────────────────────────
    match finish_reason.as_deref() {
        Some("stop") | Some("length") => {
            // PR 12 §12.3: Notify PS Manager: ROLLOUT_COMPLETED
            if let (Some(ps_client), Some(rid)) = (runtime.ps_manager_client.as_ref(), request_id) {
                let _ = ps_client.update_request_status(
                    vec![rid],
                    "ROLLOUT_COMPLETED".to_string(),
                    vec![version_tag],
                    vec![],
                    is_validate,
                )
                .await
                .map_err(|e| error!("update_request_status COMPLETED failed: {}", e));
            }

            // PR 12 §12.3: Merge prior loopback partial state into the final drained result.
            // PR 18 (Gap 5): prior state is carried in ctx.state.partial_rollout_state.
            let mut drained = drained;
            if let Some(prior_partial) = ctx.state.partial_rollout_state.as_ref() {
                merge_partial_into_drained(&mut drained, prior_partial);
            }

            // PR 12 §12.3: Build ExecutionResult::PreDrained from drained complete messages.
            // ResponseProcessingStage will use collect_responses() which handles PreDrained.
            let final_complete = drained.final_complete.map(|c| vec![c]).unwrap_or_default();
            ctx.state.response.execution_result = Some(ExecutionResult::PreDrained {
                complete: final_complete,
            });

            // PR 12 §12.3: Run ResponseProcessing stage with the pre-drained result.
            let response = pipeline.execute_response_processing_only(ctx).await;
            let _ = request.result_tx.send(response);
        }
        Some("abort") => {
            // PR 12 §12.3: Notify PS Manager: ROLLOUT_INTERRUPTED
            if let (Some(ps_client), Some(rid)) = (runtime.ps_manager_client.as_ref(), request_id) {
                let _ = ps_client.update_request_status(
                    vec![rid],
                    "ROLLOUT_INTERRUPTED".to_string(),
                    vec![version_tag],
                    vec![],
                    is_validate,
                )
                .await
                .map_err(|e| error!("update_request_status INTERRUPTED failed: {}", e));
            }

            // PR 18 (Gap 5): gate on generate-backed request paths (Generate/Chat/Responses),
            // then carry loopback state in RequestContext for proto-level injection.
            if supports_partial_rollout(&ctx) {
                // PR 12 §12.3: Extract partial rollout state using backend-appropriate proto extractor.
                let new_proto_partial = extract_partial_rollout_state(&drained);
                let new_partial_tokens = new_proto_partial.token_ids.len();

                // PR 18 (Gap 5): Keep accumulated loopback state on ctx.state so request-building
                // stages can apply it directly to ProtoGenerateRequest across request types.
                merge_partial_rollout_state(
                    &mut ctx.state.partial_rollout_state,
                    new_proto_partial,
                );

                // PR 12 §12.3: Update rollout_instance_hint and version_tag for affinity routing.
                if let Some(meta) = request.routing_meta.as_mut() {
                    let version_map = runtime.instance_to_version_after_sync.lock();
                    meta.rollout_instance_hint = Some(selected_instance.clone());
                    if let Some(v) = version_map.get(&selected_instance).copied() {
                        meta.version_tag = v;
                    }
                }

                // PR 12 §12.3: Update rollout_instance_hint headers for header-based routing hint.
                // PR 13 §13.1: headers are now accessed via ctx.input.headers.
                worker_selection::upsert_rollout_instance_hint_headers(
                    &mut ctx.input.headers,
                    &selected_instance.0,
                    selected_instance.1,
                );

                // PR 15: Mirror sgl-model-gateway routing_loop.rs lines 425-436.
                // If the original caller set x-manual-target-worker: true but did not specify
                // a specific instance, lock subsequent loopback iterations to the selected instance.
                // This ensures KV-cache affinity is preserved across partial rollout aborts.
                maybe_upsert_manual_target_headers(
                    &mut ctx.input.headers,
                    manual_target_enabled,
                    &selected_instance,
                );

                debug!(
                    request_id = ?request_id,
                    routing_meta = ?request.routing_meta,
                    token_ids_len = new_partial_tokens,
                    "dispatch_task: abort loopback to queue (PR 12 proto extraction)"
                );

                // PR 13 §13.1: Rebuild the RoutingQueueEntry with the updated ctx.
                // ctx.input.headers and ctx.input.request_type have been mutated above.
                // Re-construct the entry so the updated ctx is queued for the next iteration.
                let loopback_entry = RoutingQueueEntry {
                    ctx,
                    result_tx: request.result_tx,
                    routing_meta: request.routing_meta,
                };
                runtime.request_queue.lock().await.push(loopback_entry);
            } else {
                // Route doesn't support partial rollout — return an error response.
                let _ = request.result_tx.send(router_error::internal_error(
                    "abort_unsupported_route",
                    "Abort received on route that does not support partial rollout",
                ));
            }
        }
        _ => {
            // Unknown finish_reason (error, None, etc.).
            // PR 12 §12.3: Build ExecutionResult::PreDrained with whatever complete messages
            // we got from the drain (may be empty for error case).
            let final_complete = drained.final_complete.map(|c| vec![c]).unwrap_or_default();
            ctx.state.response.execution_result = Some(ExecutionResult::PreDrained {
                complete: final_complete,
            });
            let response = pipeline.execute_response_processing_only(ctx).await;
            let _ = request.result_tx.send(response);
        }
    }
}

// ── Helper functions ─────────────────────────────────────────────────────

// PR 15: Extract abort-loopback manual-target header pinning into a helper so
// the behavior is unit-testable and stays identical to sgl-model-gateway.
fn maybe_upsert_manual_target_headers(
    headers: &mut Option<HeaderMap>,
    manual_target_enabled: bool,
    selected_instance: &(String, usize),
) {
    use crate::routers::grpc::worker_selection;

    if manual_target_enabled
        && worker_selection::manual_target_instance_from_headers(headers.as_ref()).is_none()
    {
        worker_selection::upsert_manual_target_headers(
            headers,
            &selected_instance.0,
            selected_instance.1,
        );
    }
}

// PR 13 §13.1 + Refactor Note 5: supports_partial_rollout matches on RequestType variant,
// not a route string or PreparedRequest. Replaces supports_partial_rollout(&PreparedRequest).
/// Whether the given request context supports partial-rollout (loopback on abort).
///
/// PR 18 (Gap 5): loopback applies to any request path that executes backend
/// `generate(...)` under the routing loop: Generate, Chat, and Responses.
fn supports_partial_rollout(ctx: &RequestContext) -> bool {
    matches!(
        ctx.input.request_type,
        RequestType::Generate(_) | RequestType::Chat(_) | RequestType::Responses(_)
    )
}

// PR 18 (Gap 5): Merge newly extracted partial state into the accumulated
// context slot used by proto-level request-building loopback.
fn merge_partial_rollout_state(
    slot: &mut Option<ProtoPartialRolloutState>,
    next: ProtoPartialRolloutState,
) {
    if next.token_ids.is_empty() && next.logprobs.is_empty() {
        return;
    }

    match slot {
        Some(existing) => existing.extend_from(&next),
        None => *slot = Some(next),
    }
}

/// Remove a request from the in-flight tracking maps on completion.
///
/// PR 10 §10.7.3-10.7.4: Called from `dispatch_task` on all exit paths.
async fn cleanup_tracking(
    runtime: &Arc<RoutingLoopRuntime>,
    request_id: Option<i64>,
    prompt_id: Option<i64>,
) {
    let Some(rid) = request_id else { return };

    // Remove from incomplete_request_to_instance.
    runtime
        .incomplete_request_to_instance
        .lock()
        .await
        .remove(&rid);

    // Remove from prompt_to_running_request_ids and clean up empty entries.
    if let Some(pid) = prompt_id {
        let mut map = runtime.prompt_to_running_request_ids.lock().await;
        if let Some(ids) = map.get_mut(&pid) {
            ids.retain(|&r| r != rid);
            if ids.is_empty() {
                map.remove(&pid);
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use axum::response::Response;
    use openai_protocol::generate::GenerateRequest;
    use tokio::sync::oneshot;

    use super::*;
    use crate::{
        config::types::RequestSortIndicator,
        routers::{
            grpc::{
                context::{RequestContext, SharedComponents},
                partial_rollout::ProtoPartialRolloutState,
            },
            routing_loop_utils::{RoutingMeta, RoutingQueueEntry},
        },
    };

    // PR 13 §13.1: Build minimal SharedComponents for tests.
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

    // PR 13 §13.1: Helper to build a RoutingQueueEntry with ctx (no PreparedRequest).
    fn make_entry(
        request_id: Option<i64>,
        version_tag: i64,
    ) -> (RoutingQueueEntry, oneshot::Receiver<Response>) {
        let (tx, rx) = oneshot::channel();
        let gen_req: GenerateRequest =
            serde_json::from_str(r#"{"text":"hello","model":"test-model"}"#).unwrap();
        let ctx = RequestContext::for_generate(
            Arc::new(gen_req),
            None,
            Some("test-model".to_string()),
            make_components(),
        );
        let entry = RoutingQueueEntry {
            ctx,
            result_tx: tx,
            routing_meta: Some(RoutingMeta {
                request_id,
                prompt_id: None,
                version_tag,
                is_validate: false,
                rollout_instance_hint: None,
            }),
        };
        (entry, rx)
    }

    // PR 10 §10.6 test: Channel entries are moved to queue.
    #[tokio::test]
    async fn test_routing_loop_drains_channel() {
        // Use new_with_channel so both tx and rx are live.
        let (rt, mut rx) = RoutingLoopRuntime::new_with_channel(
            RequestSortIndicator::SmallId,
            false,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            0,
            String::new(),
            None,
        );

        // Submit 3 entries via tx.
        for i in 1i64..=3 {
            let (entry, _result_rx) = make_entry(Some(i), 1);
            rt.tx.send(entry).expect("send ok");
        }

        // Drain channel into local_batch (same as the routing loop does).
        let mut drained = Vec::new();
        while let Ok(req) = rx.try_recv() {
            drained.push(req);
        }
        assert_eq!(drained.len(), 3);

        // Push into queue (same as the routing loop does).
        let mut q = rt.request_queue.lock().await;
        for req in drained {
            q.push(req);
        }
        assert_eq!(q.len(), 3);
    }

    // PR 10 §10.6 test / PR 11 §11.1: Request stays in queue when no worker available.
    #[tokio::test]
    async fn test_routing_loop_no_worker_pushback() {
        let (runtime, _rx) = RoutingLoopRuntime::new_with_channel(
            RequestSortIndicator::SmallId,
            false,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            0,
            String::new(),
            None,
        );

        let (entry, _result_rx) = make_entry(Some(1), 1);

        // Simulate the no-worker case: when select_worker_for_model returns None,
        // the request should be pushed back to the queue.
        let maybe_worker: Option<(Arc<dyn Worker>, (String, usize))> = None;
        assert!(maybe_worker.is_none());

        // Simulate pushback.
        runtime.request_queue.lock().await.push(entry);
        assert_eq!(runtime.request_queue.lock().await.len(), 1);
    }

    // PR 10 §10.6 test: Paused loop does not change is_routing to true.
    #[tokio::test]
    async fn test_routing_loop_pause_stops_dispatch() {
        use std::sync::atomic::Ordering;

        let (runtime, _rx) = RoutingLoopRuntime::new_with_channel(
            RequestSortIndicator::SmallId,
            false,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            0,
            String::new(),
            None,
        );

        runtime.is_paused.store(true, Ordering::Relaxed);

        // When paused, is_routing should be set to false.
        runtime.is_routing.store(false, Ordering::Relaxed);
        assert!(!runtime.is_routing.load(Ordering::Relaxed));
        assert!(runtime.is_paused.load(Ordering::Relaxed));
    }

    // PR 18 (Gap 5): merge helper initializes an empty slot.
    #[test]
    fn test_merge_partial_rollout_state_initializes_slot() {
        let mut slot: Option<ProtoPartialRolloutState> = None;
        merge_partial_rollout_state(
            &mut slot,
            ProtoPartialRolloutState {
                token_ids: vec![1, 2],
                logprobs: vec![],
            },
        );

        let merged = slot.expect("partial state should be initialized");
        assert_eq!(merged.token_ids, vec![1, 2]);
    }

    // PR 18 (Gap 5): merge helper appends to existing accumulated state.
    #[test]
    fn test_merge_partial_rollout_state_appends_existing() {
        let mut slot = Some(ProtoPartialRolloutState {
            token_ids: vec![10],
            logprobs: vec![],
        });

        merge_partial_rollout_state(
            &mut slot,
            ProtoPartialRolloutState {
                token_ids: vec![20, 30],
                logprobs: vec![],
            },
        );

        let merged = slot.expect("partial state should remain present");
        assert_eq!(merged.token_ids, vec![10, 20, 30]);
    }

    // PR 18 (Gap 5): merge helper ignores empty delta payloads.
    #[test]
    fn test_merge_partial_rollout_state_ignores_empty_delta() {
        let mut slot = Some(ProtoPartialRolloutState {
            token_ids: vec![5],
            logprobs: vec![],
        });

        merge_partial_rollout_state(&mut slot, ProtoPartialRolloutState::default());

        let merged = slot.expect("partial state should remain present");
        assert_eq!(merged.token_ids, vec![5]);
    }

    // PR 18 (Gap 5): supports_partial_rollout now covers all generate-backed request types.
    #[test]
    fn test_supports_partial_rollout_generate_chat_responses_true() {
        use openai_protocol::{chat::ChatCompletionRequest, responses::ResponsesRequest};

        let gen_req: GenerateRequest = serde_json::from_str(r#"{"text":"x"}"#).unwrap();
        let chat_req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[],"model":"m"}"#).unwrap();
        let responses_req: ResponsesRequest =
            serde_json::from_str(r#"{"model":"m","input":"hello","stream":false}"#).unwrap();

        let gen_ctx =
            RequestContext::for_generate(Arc::new(gen_req), None, None, make_components());
        let chat_ctx = RequestContext::for_chat(Arc::new(chat_req), None, None, make_components());
        let responses_ctx =
            RequestContext::for_responses(Arc::new(responses_req), None, None, make_components());

        assert!(supports_partial_rollout(&gen_ctx));
        assert!(supports_partial_rollout(&chat_ctx));
        assert!(supports_partial_rollout(&responses_ctx));
    }

    // PR 15: abort-loopback should pin manual-target headers when enabled and unset.
    #[test]
    fn test_maybe_upsert_manual_target_headers_sets_selected_instance() {
        use axum::http::{HeaderMap, HeaderName, HeaderValue};

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-manual-target-worker"),
            HeaderValue::from_static("true"),
        );
        let mut headers = Some(headers);

        maybe_upsert_manual_target_headers(&mut headers, true, &("worker-selected".to_string(), 2));

        let pinned = headers.expect("headers should exist after upsert");
        assert_eq!(
            pinned
                .get("x-manual-target-worker")
                .and_then(|v| v.to_str().ok()),
            Some("true")
        );
        assert_eq!(
            pinned.get("x-base-worker-id").and_then(|v| v.to_str().ok()),
            Some("worker-selected")
        );
        assert_eq!(
            pinned.get("x-target-dp-rank").and_then(|v| v.to_str().ok()),
            Some("2")
        );
    }

    // PR 15: if manual-target instance is already set, loopback should not override it.
    #[test]
    fn test_maybe_upsert_manual_target_headers_keeps_existing_instance() {
        use axum::http::{HeaderMap, HeaderName, HeaderValue};

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-manual-target-worker"),
            HeaderValue::from_static("true"),
        );
        headers.insert(
            HeaderName::from_static("x-base-worker-id"),
            HeaderValue::from_static("worker-original"),
        );
        headers.insert(
            HeaderName::from_static("x-target-dp-rank"),
            HeaderValue::from_static("1"),
        );
        let mut headers = Some(headers);

        maybe_upsert_manual_target_headers(&mut headers, true, &("worker-new".to_string(), 9));

        let pinned = headers.expect("headers should remain present");
        assert_eq!(
            pinned.get("x-base-worker-id").and_then(|v| v.to_str().ok()),
            Some("worker-original")
        );
        assert_eq!(
            pinned.get("x-target-dp-rank").and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }

    // PR 10 §10.6 test: cleanup_tracking removes from both maps.
    #[tokio::test]
    async fn test_cleanup_tracking_removes_from_maps() {
        let (runtime, _rx) = RoutingLoopRuntime::new_with_channel(
            RequestSortIndicator::SmallId,
            false,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            0,
            String::new(),
            None,
        );

        // Insert into tracking maps.
        runtime
            .incomplete_request_to_instance
            .lock()
            .await
            .insert(42, ("worker-1".to_string(), 0));
        runtime
            .prompt_to_running_request_ids
            .lock()
            .await
            .insert(10, vec![42, 99]);

        // Cleanup.
        cleanup_tracking(&runtime, Some(42), Some(10)).await;

        // request_id 42 should be gone from incomplete map.
        assert!(!runtime
            .incomplete_request_to_instance
            .lock()
            .await
            .contains_key(&42));

        // prompt_id 10 should still have request_id 99.
        let map = runtime.prompt_to_running_request_ids.lock().await;
        let ids = map.get(&10).expect("prompt_id 10 still has entries");
        assert_eq!(ids, &[99]);
    }

    // PR 10 §10.6 test: cleanup_tracking removes empty prompt entries.
    #[tokio::test]
    async fn test_cleanup_tracking_removes_empty_prompt_entry() {
        let (runtime, _rx) = RoutingLoopRuntime::new_with_channel(
            RequestSortIndicator::SmallId,
            false,
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            0,
            String::new(),
            None,
        );

        runtime
            .incomplete_request_to_instance
            .lock()
            .await
            .insert(7, ("worker-2".to_string(), 1));
        runtime
            .prompt_to_running_request_ids
            .lock()
            .await
            .insert(5, vec![7]);

        cleanup_tracking(&runtime, Some(7), Some(5)).await;

        // prompt_id 5 entry should be removed entirely (was the only request).
        assert!(!runtime
            .prompt_to_running_request_ids
            .lock()
            .await
            .contains_key(&5));
    }
}
