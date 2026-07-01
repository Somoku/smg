"""KV-cache event replay hub for the vLLM gRPC servicer.

A long-lived ZMQ ingester per ``dp_rank`` buffers recent proto ``KvEventBatch``
messages so ``SubscribeKvEvents(start_sequence_number=…)`` can replay missed
batches before switching to live delivery — matching the semantics of
``mock_worker``'s ``subscribe_kv``.

The ingester runs on its **own thread** (with a synchronous ZMQ socket), NOT on
the vLLM engine's asyncio event loop. This is deliberate: at hundreds of KV
events/sec, decoding + proto conversion are non-trivial synchronous CPU work,
and running them on the engine loop preempted token generation (observed ~3x
slowdown). Off-loading to a dedicated thread keeps the engine loop free; fan-out
to gRPC subscribers hops back onto the main loop via ``call_soon_threadsafe``.

Replay only covers batches the ingester successfully received. If ZMQ drops a
frame before ingest, that sequence is unrecoverable here; SMG's gateway force-
resync handles that case.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REPLAY_CAPACITY = 4096

# ZMQ SUB receive high-water mark. Generous so a brief ingester hiccup does not
# make the PUB side drop frames (which would surface as gateway sequence gaps).
_SUB_RCVHWM = 100_000


def replay_capacity() -> int:
    raw = os.environ.get("SMG_KV_EVENTS_REPLAY_CAPACITY", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_REPLAY_CAPACITY


def _offset_endpoint(endpoint: str, dp_rank: int, offset_fn: Callable[[str, int], str]) -> str:
    resolved = endpoint.replace("*", "127.0.0.1").replace("0.0.0.0", "127.0.0.1")
    if dp_rank:
        return offset_fn(resolved, dp_rank)
    return resolved


class _PendingBatch:
    """A buffered entry whose proto conversion is deferred to the consumer.

    The ingester thread stores only the cheaply-extracted ``sequence_number``
    plus the raw msgpack ``payload`` bytes — it does NOT decode/convert, so it
    barely holds the GIL. The heavy ``decode``+``convert`` runs lazily in the
    ``subscribe`` coroutine on the main loop (once per subscription, matching the
    proven-fine inline path). Tests inject an already-built ``batch`` directly.
    """

    __slots__ = ("sequence_number", "_payload", "_decode", "_convert", "_batch")

    def __init__(
        self,
        seq: int,
        *,
        payload: bytes | None = None,
        decode: Callable[[bytes], object] | None = None,
        convert: Callable[[object, int], Any] | None = None,
        batch: Any | None = None,
    ) -> None:
        self.sequence_number = seq
        self._payload = payload
        self._decode = decode
        self._convert = convert
        self._batch = batch

    def materialize(self) -> Any:
        """Produce the proto batch, decoding/converting on first access."""
        if self._batch is None:
            raw = self._decode(self._payload)  # type: ignore[misc]
            self._batch = self._convert(raw, self.sequence_number)  # type: ignore[misc]
            # Drop the raw payload reference once converted to free memory.
            self._payload = None
        return self._batch


class KvEventRankState:
    """Per-``dp_rank`` ingest buffer and live subscriber fan-out.

    The buffer and subscriber list are shared between the ingester **thread**
    and ``subscribe`` coroutines running on the main loop, so they are guarded
    by a plain ``threading.Lock``. Critical sections are short and CPU-only (no
    awaits, no I/O), so blocking the main loop on the lock is sub-millisecond.
    """

    def __init__(
        self,
        *,
        dp_rank: int,
        endpoint: str,
        topic: str,
        capacity: int,
        decode: Callable[[bytes], object],
        convert: Callable[[object, int], Any],
        offset_endpoint_port: Callable[[str, int], str],
    ) -> None:
        self.dp_rank = dp_rank
        self.endpoint = endpoint
        self.topic = topic
        self.capacity = capacity
        self.decode = decode
        self.convert = convert
        self.offset_endpoint_port = offset_endpoint_port
        self.buffer: deque[Any] = deque()
        self.subscribers: list[asyncio.Queue[Any]] = []
        self.lock = threading.Lock()

        # Main asyncio loop, captured at start(); fan-out from the ingester
        # thread hops onto it via call_soon_threadsafe.
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._ingest_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Capture the main loop and launch the ingester thread."""
        if self._ingest_thread is not None:
            return
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None
        self._stop_event.clear()
        self._ingest_thread = threading.Thread(
            target=self._ingest_loop,
            name=f"kv-event-ingest-dp{self.dp_rank}",
            daemon=True,
        )
        self._ingest_thread.start()

    async def stop(self) -> None:
        if self._ingest_thread is None:
            return
        self._stop_event.set()
        thread = self._ingest_thread
        self._ingest_thread = None
        # Join off the event loop; the thread exits within one poll timeout (1s).
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, thread.join, 3.0)

    def _ingest_loop(self) -> None:
        """Synchronous ZMQ receive loop — runs on a dedicated thread."""
        import zmq

        pub_endpoint = _offset_endpoint(self.endpoint, self.dp_rank, self.offset_endpoint_port)
        zmq_ctx = zmq.Context()
        sub_socket = zmq_ctx.socket(zmq.SUB)
        sub_socket.setsockopt(zmq.RCVHWM, _SUB_RCVHWM)
        sub_socket.subscribe(self.topic.encode("utf-8"))
        sub_socket.connect(pub_endpoint)
        poller = zmq.Poller()
        poller.register(sub_socket, zmq.POLLIN)
        logger.info(
            "KV event ingester thread started endpoint=%s dp_rank=%d capacity=%d",
            pub_endpoint,
            self.dp_rank,
            self.capacity,
        )
        try:
            while not self._stop_event.is_set():
                if not dict(poller.poll(timeout=1000)):
                    continue
                frames = sub_socket.recv_multipart()
                if len(frames) < 3:
                    continue
                zmq_seq = int.from_bytes(frames[1], "big")
                # Defer decode+convert to the consumer: store only the raw
                # payload here so the ingester thread holds the GIL for just the
                # 8-byte seq extraction, not the full msgpack/proto build.
                pending = _PendingBatch(
                    zmq_seq,
                    payload=frames[2],
                    decode=self.decode,
                    convert=self.convert,
                )
                self._store_and_fanout(pending)
        except Exception:
            logger.exception("KV event ingester thread failed dp_rank=%d", self.dp_rank)
        finally:
            poller.unregister(sub_socket)
            sub_socket.close(linger=0)
            zmq_ctx.term()
            logger.info("KV event ingester thread stopped dp_rank=%d", self.dp_rank)

    def _store_and_fanout(self, pending: Any) -> None:
        """Append a ``_PendingBatch`` to the ring buffer and fan out to queues.

        Called from the ingester thread (and, in tests, from the main loop via
        ``inject_batch``). Only does a short locked buffer update plus a thread-
        safe enqueue; the heavy decode/convert is deferred to the consumer via
        ``_PendingBatch.materialize()`` so the ingester thread barely holds the GIL.
        """
        with self.lock:
            self.buffer.append(pending)
            while len(self.buffer) > self.capacity:
                self.buffer.popleft()
            queues = list(self.subscribers)

        loop = self._main_loop
        for queue in queues:
            if loop is not None:
                # Ingester thread (or any non-loop caller): hop onto the main loop.
                loop.call_soon_threadsafe(queue.put_nowait, pending)
            else:
                # No main loop captured (e.g. unit tests calling on the loop directly).
                queue.put_nowait(pending)

    async def subscribe(
        self,
        start_seq: int,
        *,
        send_initial_metadata: Callable[[], Awaitable[None]] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AsyncIterator[Any]:
        """Replay buffered batches with ``seq > start_seq``, then live events."""
        # Tests call subscribe() without start(); capture the loop here so
        # inject_batch fan-out can target this subscriber's queue.
        if self._main_loop is None:
            try:
                self._main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        queue: asyncio.Queue[Any] = asyncio.Queue()
        with self.lock:
            self.subscribers.append(queue)
            replay = [b for b in self.buffer if b.sequence_number > start_seq]

        if send_initial_metadata is not None:
            await send_initial_metadata()

        last_seq = start_seq
        try:
            for pending in replay:
                if pending.sequence_number > last_seq:
                    last_seq = pending.sequence_number
                    yield pending.materialize()

            while is_cancelled is None or not is_cancelled():
                try:
                    pending = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                if pending.sequence_number > last_seq:
                    last_seq = pending.sequence_number
                    yield pending.materialize()
        finally:
            with self.lock:
                if queue in self.subscribers:
                    self.subscribers.remove(queue)


class KvEventReplayHub:
    """Manages per-rank KV event ingest + replay for one vLLM replica."""

    def __init__(
        self,
        *,
        endpoint: str,
        topic: str,
        dp_size: int,
        decode: Callable[[bytes], object],
        convert: Callable[[object, int], Any],
        offset_endpoint_port: Callable[[str, int], str],
        capacity: int | None = None,
    ) -> None:
        cap = capacity if capacity is not None else replay_capacity()
        self._ranks: dict[int, KvEventRankState] = {
            rank: KvEventRankState(
                dp_rank=rank,
                endpoint=endpoint,
                topic=topic,
                capacity=cap,
                decode=decode,
                convert=convert,
                offset_endpoint_port=offset_endpoint_port,
            )
            for rank in range(max(1, dp_size))
        }

    def start(self) -> None:
        for state in self._ranks.values():
            state.start()

    async def stop(self) -> None:
        await asyncio.gather(*(state.stop() for state in self._ranks.values()))

    def subscribe(
        self,
        dp_rank: int,
        start_seq: int,
        *,
        send_initial_metadata: Callable[[], Awaitable[None]] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AsyncIterator[Any]:
        if dp_rank not in self._ranks:
            raise KeyError(f"unknown dp_rank {dp_rank}")
        return self._ranks[dp_rank].subscribe(
            start_seq,
            send_initial_metadata=send_initial_metadata,
            is_cancelled=is_cancelled,
        )

    # Test hook: inject an already-built batch without ZMQ.
    async def inject_batch(self, dp_rank: int, batch: Any) -> None:
        pending = _PendingBatch(batch.sequence_number, batch=batch)
        self._ranks[dp_rank]._store_and_fanout(pending)

    @property
    def rank_states(self) -> Sequence[KvEventRankState]:
        return list(self._ranks.values())
