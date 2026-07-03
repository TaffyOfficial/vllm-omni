"""Unit tests for OmniSchedulerMixin streaming session replacement.

These tests pin the behavior of `_replace_session_with_streaming_update` against
current vLLM `Request` / `StreamingUpdate` (and Omni patches). When upgrading
vLLM, failures here should highlight incompatible changes to request state or
update payloads early.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniNewRequestData

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SchedulerStub(OmniSchedulerMixin):
    """Minimal scheduler surface required by OmniSchedulerMixin."""

    def __init__(self, *, log_stats: bool = False) -> None:
        self.num_waiting_for_streaming_input = 0
        self.log_stats = log_stats


def _make_request(**kwargs) -> Request:
    sp = SamplingParams(max_tokens=8)
    defaults = dict(
        request_id="req-mixin-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    defaults.update(kwargs)
    return Request(**defaults)


def _make_update(**kwargs) -> StreamingUpdate:
    sp_new = SamplingParams(max_tokens=16)
    defaults = dict(
        mm_features=None,
        prompt_token_ids=[10, 20],
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=sp_new,
    )
    defaults.update(kwargs)
    return StreamingUpdate(**defaults)


class TestReplaceSessionWithStreamingUpdate:
    def test_resets_tokens_and_prompt_from_update(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.append_output_token_ids([7, 8])
        session.num_computed_tokens = 99
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

        update = _make_update(prompt_token_ids=[40, 41, 42])
        sched.num_waiting_for_streaming_input = 3
        sched._replace_session_with_streaming_update(session, update)

        assert session._output_token_ids == []
        assert list(session._all_token_ids) == [40, 41, 42]
        assert session.prompt_token_ids == [40, 41, 42]
        assert session.num_computed_tokens == 0
        assert session.num_prompt_tokens == 3
        assert session.arrival_time == 200.0
        assert session.sampling_params is update.sampling_params
        assert session.status == RequestStatus.WAITING
        assert sched.num_waiting_for_streaming_input == 2

    def test_none_prompt_token_ids_becomes_empty(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.status = RequestStatus.RUNNING
        update = _make_update(prompt_token_ids=None)
        sched._replace_session_with_streaming_update(session, update)

        assert session.prompt_token_ids == ()
        assert list(session._all_token_ids) == []
        assert session.num_prompt_tokens == 0
        assert sched.num_waiting_for_streaming_input == 0

    def test_additional_information_cleared_when_update_omits_it(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        if not hasattr(session, "additional_information"):
            pytest.skip("Request has no additional_information (Omni patch inactive?)")
        session.additional_information = {"keep": True}
        session.status = RequestStatus.RUNNING

        base = _make_update()
        if not hasattr(base, "additional_information"):
            pytest.skip("StreamingUpdate has no additional_information (Omni patch inactive?)")
        update = replace(base, additional_information=None)

        sched._replace_session_with_streaming_update(session, update)
        assert session.additional_information is None

    def test_does_not_decrement_waiting_when_not_streaming_status(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.status = RequestStatus.RUNNING
        sched.num_waiting_for_streaming_input = 5
        sched._replace_session_with_streaming_update(session, _make_update())
        assert sched.num_waiting_for_streaming_input == 5

    def test_records_queued_event_when_log_stats_enabled(self) -> None:
        sched = _SchedulerStub(log_stats=True)
        session = _make_request()
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        sched._replace_session_with_streaming_update(session, _make_update())

        assert session.events
        assert session.events[-1].type == EngineCoreEventType.QUEUED


class _MockQueue:
    def __init__(self, requests):
        self._requests = list(requests)
        self.prepend_requests_called = False

    def __iter__(self):
        return iter(self._requests)

    def add_request(self, request):
        self._requests.append(request)

    def remove_requests(self, requests):
        remove_ids = {id(request) for request in requests}
        self._requests = [request for request in self._requests if id(request) not in remove_ids]

    def prepend_requests(self, requests):
        self.prepend_requests_called = True
        self._requests = list(requests) + self._requests


class _OrderingSchedulerStub(OmniSchedulerMixin):
    def __init__(self, requests):
        self.waiting = _MockQueue(requests)
        self.max_num_running_reqs = 8
        self.policy = SchedulingPolicy.PRIORITY


def _priority_request(request_id: str, priority: int):
    return SimpleNamespace(request_id=request_id, priority=priority)


def test_waiting_order_is_unchanged_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    scheduler = _OrderingSchedulerStub(
        [
            _priority_request("b", 20),
            _priority_request("a", 10),
            _priority_request("c", 10),
        ]
    )

    scheduler._order_waiting_for_batch_invariance()

    assert [request.request_id for request in scheduler.waiting] == ["b", "a", "c"]


def test_waiting_order_uses_priority_then_request_id_in_batch_invariant_mode(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    scheduler = _OrderingSchedulerStub(
        [
            _priority_request("b", 20),
            _priority_request("c", 10),
            _priority_request("a", 10),
        ]
    )

    scheduler._order_waiting_for_batch_invariance()

    assert [request.request_id for request in scheduler.waiting] == ["a", "c", "b"]
    assert scheduler.waiting.prepend_requests_called is False


def test_batch_invariant_limits_running_requests(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    scheduler = _OrderingSchedulerStub([])

    scheduler._apply_batch_invariant_limits()

    assert scheduler.max_num_running_reqs == 1


def test_batch_invariant_limits_use_fcfs_waiting_queue(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    scheduler = _OrderingSchedulerStub(
        [
            _priority_request("b", 20),
            _priority_request("a", 10),
        ]
    )

    scheduler._apply_batch_invariant_limits()

    assert scheduler.policy == SchedulingPolicy.FCFS
    assert scheduler.waiting.__class__.__name__ == "FCFSRequestQueue"
    assert [request.request_id for request in scheduler.waiting] == ["b", "a"]


def test_running_request_limit_is_unchanged_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    scheduler = _OrderingSchedulerStub([])

    scheduler._apply_batch_invariant_limits()

    assert scheduler.max_num_running_reqs == 8


def test_omni_new_request_data_preserves_sampling_seed():
    sampling_params = SamplingParams(max_tokens=4, seed=123)
    request = SimpleNamespace(
        request_id="seeded-req",
        external_req_id="seeded-req",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=True,
        additional_information=None,
    )

    data = OmniNewRequestData.from_request(request, block_ids=([0],))

    assert data.sampling_params is sampling_params
    assert data.sampling_params.seed == 123
