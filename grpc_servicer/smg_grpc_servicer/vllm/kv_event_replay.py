"""KV-cache event replay hub for the vLLM gRPC servicer.

A long-lived ZMQ ingester per ``dp_rank`` buffers recent proto ``KvEventBatch``
messages so ``SubscribeKvEvents(start_sequence_number=…)`` can replay missed
batches before switching to live delivery — matching the semantics of
``mock_worker``'s ``subscribe_kv``.

Replay only covers batches the ingester successfully received. If ZMQ drops a
frame before ingest, that sequence is unrecoverable here; SMG's gateway force-
resync handles that case.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REPLAY_CAPACITY = 4096


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


class KvEventRankState:
    """Per-``dp_rank`` ingest buffer and live subscriber fan-out."""

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
        self.lock = asyncio.Lock()
        self._ingest_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._ingest_task is None:
            self._ingest_task = asyncio.create_task(
                self._ingest_loop(),
                name=f"kv-event-ingest-dp{self.dp_rank}",
            )

    async def stop(self) -> None:
        if self._ingest_task is not None:
            self._ingest_task.cancel()
            try:
                await self._ingest_task
            except asyncio.CancelledError:
                pass
            self._ingest_task = None

    async def _ingest_loop(self) -> None:
        import zmq
        import zmq.asyncio

        pub_endpoint = _offset_endpoint(self.endpoint, self.dp_rank, self.offset_endpoint_port)
        zmq_ctx = zmq.asyncio.Context.instance()
        sub_socket = zmq_ctx.socket(zmq.SUB)
        sub_socket.subscribe(self.topic.encode("utf-8"))
        sub_socket.connect(pub_endpoint)
        logger.info(
            "KV event ingester started endpoint=%s dp_rank=%d capacity=%d",
            pub_endpoint,
            self.dp_rank,
            self.capacity,
        )
        try:
            while True:
                if not await sub_socket.poll(timeout=1000):
                    continue
                frames = await sub_socket.recv_multipart()
                if len(frames) < 3:
                    continue
                zmq_seq = int.from_bytes(frames[1], "big")
                try:
                    raw_batch = self.decode(frames[2])
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "KV event ingester decode failed dp_rank=%d: %s",
                        self.dp_rank,
                        exc,
                    )
                    continue
                proto_batch = self.convert(raw_batch, zmq_seq)
                await self._store_and_fanout(proto_batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("KV event ingester failed dp_rank=%d", self.dp_rank)
            raise
        finally:
            sub_socket.close(linger=0)
            logger.info("KV event ingester stopped dp_rank=%d", self.dp_rank)

    async def _store_and_fanout(self, batch: Any) -> None:
        async with self.lock:
            self.buffer.append(batch)
            while len(self.buffer) > self.capacity:
                self.buffer.popleft()
            queues = list(self.subscribers)
        for queue in queues:
            queue.put_nowait(batch)

    async def subscribe(
        self,
        start_seq: int,
        *,
        send_initial_metadata: Callable[[], Awaitable[None]] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AsyncIterator[Any]:
        """Replay buffered batches with ``seq > start_seq``, then live events."""
        queue: asyncio.Queue[Any] = asyncio.Queue()
        async with self.lock:
            self.subscribers.append(queue)
            replay = [b for b in self.buffer if b.sequence_number > start_seq]

        if send_initial_metadata is not None:
            await send_initial_metadata()

        last_seq = start_seq
        for batch in replay:
            if batch.sequence_number > last_seq:
                yield batch
                last_seq = batch.sequence_number

        try:
            while is_cancelled is None or not is_cancelled():
                try:
                    batch = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                if batch.sequence_number > last_seq:
                    yield batch
                    last_seq = batch.sequence_number
        finally:
            async with self.lock:
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

    # Test hook: inject a batch without ZMQ.
    async def inject_batch(self, dp_rank: int, batch: Any) -> None:
        await self._ranks[dp_rank]._store_and_fanout(batch)

    @property
    def rank_states(self) -> Sequence[KvEventRankState]:
        return list(self._ranks.values())
