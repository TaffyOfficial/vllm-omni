from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner, HiddenStatesCPUStaging

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_runner(engine_output_type: str | None, downstream_req_ids: set[str]) -> GPUARModelRunner:
    runner = object.__new__(GPUARModelRunner)
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(engine_output_type=engine_output_type),
    )
    runner._request_needs_downstream_stage_payload = lambda rid: rid in downstream_req_ids
    return runner


class _FakeEvent:
    def __init__(self, ready: bool):
        self.ready = ready
        self.synchronized = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.synchronized = True
        self.ready = True


def _make_staging_runner() -> GPUARModelRunner:
    runner = object.__new__(GPUARModelRunner)
    runner._prefix_cache_staging_buffers = {}
    runner._prefix_cache_staging_next_index = {}
    runner._prefix_cache_staging_buffer_count = 2
    return runner


def test_resolve_pooler_payload_req_ids_audio_terminal_stage_keeps_payload():
    runner = _make_runner(engine_output_type="audio", downstream_req_ids=set())

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2"])

    assert engine_output_type == "audio"
    assert payload_req_ids == ["r1", "r2"]


def test_resolve_pooler_payload_req_ids_text_terminal_stage_drops_payload():
    runner = _make_runner(engine_output_type="text", downstream_req_ids=set())

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2"])

    assert engine_output_type == "text"
    assert payload_req_ids == []


def test_resolve_pooler_payload_req_ids_downstream_stage_uses_filtered_requests():
    runner = _make_runner(engine_output_type="latent", downstream_req_ids={"r2"})

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2", "r3"])

    assert engine_output_type == "latent"
    assert payload_req_ids == ["r2"]


def test_claim_hidden_states_cpu_staging_buffer_reuses_pinned_buffers(monkeypatch):
    runner = _make_staging_runner()
    allocations = []

    def fake_empty(shape, *, dtype, device, pin_memory):
        assert device == "cpu"
        assert pin_memory is True
        tensor = torch.zeros(tuple(shape), dtype=dtype)
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "empty", fake_empty)

    src = torch.ones((3, 4), dtype=torch.float32)
    first = GPUARModelRunner._claim_hidden_states_cpu_staging_buffer(runner, src)
    runner._prefix_cache_staging_buffers[first.key][first.index] = HiddenStatesCPUStaging(
        first.tensor,
        _FakeEvent(ready=False),
    )

    second = GPUARModelRunner._claim_hidden_states_cpu_staging_buffer(runner, src)

    assert len(allocations) == 2
    assert first.index == 0
    assert second.index == 1
    assert second.tensor is allocations[1]


def test_claim_hidden_states_cpu_staging_buffer_waits_when_all_busy(monkeypatch):
    runner = _make_staging_runner()

    def fake_empty(shape, *, dtype, device, pin_memory):
        return torch.zeros(tuple(shape), dtype=dtype)

    monkeypatch.setattr(torch, "empty", fake_empty)

    src = torch.ones((3, 4), dtype=torch.float32)
    first = GPUARModelRunner._claim_hidden_states_cpu_staging_buffer(runner, src)
    second = GPUARModelRunner._claim_hidden_states_cpu_staging_buffer(runner, src)
    first_event = _FakeEvent(ready=False)
    second_event = _FakeEvent(ready=False)
    runner._prefix_cache_staging_buffers[first.key][first.index] = HiddenStatesCPUStaging(
        first.tensor,
        first_event,
    )
    runner._prefix_cache_staging_buffers[second.key][second.index] = HiddenStatesCPUStaging(
        second.tensor,
        second_event,
    )
    runner._prefix_cache_staging_next_index[first.key] = first.index

    claimed = GPUARModelRunner._claim_hidden_states_cpu_staging_buffer(runner, src)

    assert claimed.index == first.index
    assert first_event.synchronized is True
    assert second_event.synchronized is False


def test_resolve_hidden_states_cpu_staging_waits_for_ready_event():
    tensor = torch.ones((2, 3), dtype=torch.float32)
    event = _FakeEvent(ready=False)

    resolved = GPUARModelRunner._resolve_hidden_states_cpu_staging(
        HiddenStatesCPUStaging(tensor, event),
    )

    assert resolved is tensor
    assert event.synchronized is True
