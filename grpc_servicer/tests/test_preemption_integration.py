"""Integration smoke test: PreemptionStatLogger → SubscribePreemptionEvents."""
import asyncio
import time
from unittest.mock import MagicMock

import pytest
from smg_grpc_proto import vllm_engine_pb2

from smg_grpc_servicer.vllm.preemption import PreemptionStatLogger
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer


def _make_stats(req_ids):
    s = MagicMock()
    s.gateway_preemption_req_ids = req_ids
    return s


@pytest.mark.asyncio
async def test_stat_logger_feeds_subscribe_stream():
    """Events put by PreemptionStatLogger appear in SubscribePreemptionEvents."""
    preemption_queue: asyncio.Queue = asyncio.Queue()
    logger = PreemptionStatLogger(MagicMock(), 0, preemption_queue=preemption_queue)

    # Simulate two scheduler steps with preempted requests
    logger.record(_make_stats(["req-1"]), iteration_stats=None)
    logger.record(_make_stats(["req-2", "req-3"]), iteration_stats=None)

    # Build minimal servicer
    mock_llm = MagicMock()
    servicer = VllmEngineServicer(
        mock_llm, start_time=time.time(), preemption_queue=preemption_queue
    )
    mock_context = MagicMock()
    # SubscribePreemptionEvents uses `while not context.cancelled()`.
    # MagicMock().cancelled() returns a truthy MagicMock by default, which
    # would skip the loop body entirely. Return False so the loop body runs.
    mock_context.cancelled.return_value = False
    mock_request = MagicMock()

    # Collect one yielded event then stop
    gen = servicer.SubscribePreemptionEvents(mock_request, mock_context)
    event = await gen.__anext__()

    assert isinstance(event, vllm_engine_pb2.PreemptionEvent)
    assert set(event.request_ids) == {"req-1", "req-2", "req-3"}
    assert event.timestamp_ns > 0
    await gen.aclose()


@pytest.mark.asyncio
async def test_subscribe_terminates_on_cancellation():
    """SubscribePreemptionEvents terminates cleanly when context is cancelled."""
    preemption_queue: asyncio.Queue = asyncio.Queue()
    # Pre-load the queue so the generator yields one event before cancellation
    preemption_queue.put_nowait(["req-x"])

    mock_llm = MagicMock()
    servicer = VllmEngineServicer(
        mock_llm, start_time=time.time(), preemption_queue=preemption_queue
    )
    mock_context = MagicMock()
    mock_request = MagicMock()

    # First call: not cancelled → yields one event
    mock_context.cancelled.return_value = False
    gen = servicer.SubscribePreemptionEvents(mock_request, mock_context)
    event = await gen.__anext__()
    assert event.request_ids == ["req-x"]

    # Second call: cancelled → generator should stop
    mock_context.cancelled.return_value = True
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    await gen.aclose()


@pytest.mark.asyncio
async def test_generate_admission_close_waits_for_registered_requests():
    servicer = VllmEngineServicer(MagicMock(), start_time=time.time())
    assert servicer._begin_generate_admission()

    close_task = asyncio.create_task(servicer.close_generate_admission())
    await asyncio.sleep(0)
    assert not close_task.done()
    assert not servicer._begin_generate_admission()

    servicer._finish_generate_admission()
    await close_task

    await servicer.open_generate_admission()
    assert servicer._begin_generate_admission()
    servicer._finish_generate_admission()
