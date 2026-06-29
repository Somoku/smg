"""Tests for vLLM KV event replay hub (no vLLM / smg_grpc_proto required)."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "kv_event_replay.py"
_spec = importlib.util.spec_from_file_location("vllm_kv_event_replay", _MODULE_PATH)
kv_event_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kv_event_replay)

KvEventReplayHub = kv_event_replay.KvEventReplayHub


def _batch(seq: int):
    return SimpleNamespace(sequence_number=seq)


def _identity_offset(endpoint: str, dp_rank: int) -> str:
    return endpoint


def test_replay_after_late_subscribe():
    """Buffer has 1..3; subscriber with start_seq=1 gets 2,3 then live 4."""

    async def _run():
        hub = KvEventReplayHub(
            endpoint="tcp://unused:1",
            topic="kv",
            dp_size=1,
            decode=lambda payload: payload,
            convert=lambda _raw, seq: _batch(seq),
            offset_endpoint_port=_identity_offset,
        )

        for seq in (1, 2, 3):
            await hub.inject_batch(0, _batch(seq))

        collected: list[int] = []

        async def consume():
            async for batch in hub.subscribe(0, 1, is_cancelled=lambda: len(collected) >= 3):
                collected.append(batch.sequence_number)

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        await hub.inject_batch(0, _batch(4))
        await asyncio.wait_for(consumer, timeout=2)
        assert collected == [2, 3, 4]

    asyncio.run(_run())


def test_replay_on_reconnect():
    """After first subscriber leaves, a new one replays from buffer."""

    async def _run():
        hub = KvEventReplayHub(
            endpoint="tcp://unused:1",
            topic="kv",
            dp_size=1,
            decode=lambda payload: payload,
            convert=lambda _raw, seq: _batch(seq),
            offset_endpoint_port=_identity_offset,
            capacity=16,
        )

        for seq in (1, 2, 3):
            await hub.inject_batch(0, _batch(seq))

        first: list[int] = []
        async for batch in hub.subscribe(0, 0):
            first.append(batch.sequence_number)
            if len(first) >= 2:
                break

        assert first == [1, 2]

        await hub.inject_batch(0, _batch(4))

        second: list[int] = []
        async for batch in hub.subscribe(0, 2):
            second.append(batch.sequence_number)
            if len(second) >= 2:
                break

        assert second == [3, 4]

    asyncio.run(_run())


def test_buffer_capacity_evicts_oldest():
    async def _run():
        hub = KvEventReplayHub(
            endpoint="tcp://unused:1",
            topic="kv",
            dp_size=1,
            decode=lambda payload: payload,
            convert=lambda _raw, seq: _batch(seq),
            offset_endpoint_port=_identity_offset,
            capacity=2,
        )

        for seq in (1, 2, 3):
            await hub.inject_batch(0, _batch(seq))

        replay: list[int] = []
        async for batch in hub.subscribe(0, 0):
            replay.append(batch.sequence_number)
            if len(replay) >= 2:
                break

        assert replay == [2, 3]

    asyncio.run(_run())
