"""Unit tests for batch-invariant scheduler mixin behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy

# Import vllm_omni first so its vLLM request patches are applied before vLLM
# classes are bound in this module.
import vllm_omni  # noqa: F401
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniNewRequestData

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _MockQueue:
    def __init__(self, requests):
        self._requests = list(requests)
        self.add_requests_called = False

    def __iter__(self):
        return iter(self._requests)

    def __len__(self):
        return len(self._requests)

    def add_request(self, request):
        self._requests.append(request)

    def add_requests(self, requests):
        self.add_requests_called = True
        self._requests.extend(requests)

    def remove_requests(self, requests):
        remove_ids = {id(request) for request in requests}
        self._requests = [request for request in self._requests if id(request) not in remove_ids]


class _OrderingSchedulerStub(OmniSchedulerMixin):
    def __init__(self, requests):
        self.waiting = _MockQueue(requests)
        self.max_num_running_reqs = 8
        self.policy = SchedulingPolicy.PRIORITY


def _priority_request(request_id: str, priority: int, arrival_time: float = 0.0):
    return SimpleNamespace(request_id=request_id, priority=priority, arrival_time=arrival_time)


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
    assert scheduler.waiting.add_requests_called is False


def test_waiting_order_matches_vllm_priority_arrival_request_id(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    scheduler = _OrderingSchedulerStub(
        [
            _priority_request("b", 20, 0.0),
            _priority_request("c", 10, 2.0),
            _priority_request("a", 10, 1.0),
        ]
    )

    scheduler._order_waiting_for_batch_invariance()

    assert [request.request_id for request in scheduler.waiting] == ["a", "c", "b"]
    assert scheduler.waiting.add_requests_called is True


def test_waiting_order_uses_arrival_time_before_numeric_request_id(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    scheduler = _OrderingSchedulerStub(
        [
            _priority_request("0", 0, 0.0),
            _priority_request("1", 0, 1.0),
            _priority_request("10", 0, 2.0),
            _priority_request("2", 0, 3.0),
        ]
    )

    scheduler._order_waiting_for_batch_invariance()

    assert [request.request_id for request in scheduler.waiting] == ["0", "1", "10", "2"]


def test_waiting_order_uses_request_id_when_arrival_time_ties(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    scheduler = _OrderingSchedulerStub(
        [
            _priority_request("2", 0, 0.0),
            _priority_request("10", 0, 0.0),
        ]
    )

    scheduler._order_waiting_for_batch_invariance()

    assert [request.request_id for request in scheduler.waiting] == ["10", "2"]


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
