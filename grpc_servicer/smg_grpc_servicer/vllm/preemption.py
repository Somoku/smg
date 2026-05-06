"""Preemption event bridge between vLLM scheduler stats and the gRPC stream.

``PreemptionStatLogger`` is a ``StatLoggerBase`` implementation that bridges
vLLM's scheduler stats hook to the ``SubscribePreemptionEvents`` gRPC stream.

It is registered at engine construction via the ``stat_loggers`` parameter of
``AsyncLLM``.  For PSRL deployments, use ``PSRLPreemptionStatLogger`` (defined
in the PSRL package) which additionally maintains the ``scheduler_abort_requests``
set required to distinguish scheduler-preempted from coordinator-explicit aborts.

Queue sizing note
-----------------
``preemption_queue`` should be bounded (e.g. ``asyncio.Queue(maxsize=256)``).
When the queue is full — meaning the Rust consumer has been disconnected for an
extended period — the *oldest* batch is dropped to make room for the newest one.
This caps memory usage while preserving the most recent preemption signal.
Stale events that survive until reconnection are filtered by TTL on the Rust side.
"""

import asyncio

from vllm.v1.metrics.loggers import StatLoggerBase

# Default bound for the preemption queue.  At ~10 scheduler steps/sec a
# 256-entry queue holds ~25 s of preemption history before shedding begins.
DEFAULT_QUEUE_MAXSIZE: int = 256


class PreemptionStatLogger(StatLoggerBase):
    """Forwards threshold-crossing preempted request IDs to the preemption queue.

    The queue is drained by the ``SubscribePreemptionEvents`` gRPC handler and
    streamed to the SMG Rust gateway, which issues a single batch ``Abort`` RPC
    and relies on the existing ``dispatch_entry_with_partial_rollout`` loopback
    for global re-queuing.
    """

    def __init__(
        self,
        vllm_config,
        engine_index: int = 0,
        *,
        preemption_queue: asyncio.Queue[list[str]],
    ) -> None:
        self._queue = preemption_queue

    # StatLoggerBase interface -------------------------------------------------

    def record(
        self,
        scheduler_stats,
        iteration_stats,
        mm_cache_stats=None,
        engine_idx: int = 0,
    ) -> None:
        """Called by vLLM after every scheduler step.

        Only enqueues when ``preemption_req_ids`` is non-empty, i.e. when
        ``preemption_notification_threshold`` is configured and crossed.
        ``put_nowait`` is O(1) and never suspends — safe on the hot path.

        When the queue is full (SMG disconnected for an extended period), the
        oldest batch is dropped to make room for the new one, preventing
        unbounded memory growth.  Dropped events were stale enough that the
        Rust-side TTL filter would have discarded them anyway.
        """
        if not (scheduler_stats is not None and scheduler_stats.preemption_req_ids):
            return

        if self._queue.full():
            # Drop the oldest entry to bound memory; preserves most-recent signal.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(scheduler_stats.preemption_req_ids)

    def log_engine_initialized(self) -> None:
        pass


async def drain_preemption_queue(
    queue: asyncio.Queue[list[str]],
    abort_event: asyncio.Event,
) -> list[str] | None:
    """Await the next batch(es) from the queue, coalescing concurrent entries.

    Blocks until at least one batch is available *or* ``abort_event`` is set.
    Returns a flat list of all coalesced request IDs, or ``None`` if
    ``abort_event`` fired before any items arrived (signals caller to stop).

    Natural batching: after the first ``await queue.get()`` returns, any
    additional items that arrived concurrently are drained non-blocking, so a
    single yield to the gRPC stream carries the full burst without extra RTTs.
    """
    get_task = asyncio.ensure_future(queue.get())
    abort_task = asyncio.ensure_future(abort_event.wait())
    try:
        done, pending = await asyncio.wait(
            [get_task, abort_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

        if abort_task in done:
            return None

        req_ids: list[str] = get_task.result()
        while not queue.empty():
            req_ids.extend(queue.get_nowait())
        return req_ids
    except asyncio.CancelledError:
        get_task.cancel()
        abort_task.cancel()
        raise
