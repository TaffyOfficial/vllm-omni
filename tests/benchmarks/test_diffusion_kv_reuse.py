import pytest
import torch

from benchmarks.diffusion.backends import RequestFuncInput
from benchmarks.diffusion.diffusion_benchmark_serving import attach_synthetic_ar_kv_request_context
from benchmarks.diffusion.kv_reuse import SyntheticARKVConfig, SyntheticARKVProducer
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import KVCacheTransferData

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class _RecordingConnector:
    def __init__(self):
        self.puts = []

    def put(self, from_stage, to_stage, put_key, data):
        self.puts.append(
            {
                "from_stage": from_stage,
                "to_stage": to_stage,
                "put_key": put_key,
                "data": data,
            }
        )
        return True, 123, None


def test_synthetic_ar_kv_producer_writes_rank_aware_payloads():
    connector = _RecordingConnector()
    producer = SyntheticARKVProducer(
        SyntheticARKVConfig(
            num_layers=2,
            seq_len=4,
            num_heads=3,
            head_dim=5,
            from_stage="0",
            to_stage="0",
            from_tp=1,
            to_tp=2,
        ),
        connector=connector,
    )

    result = producer.prepare("req-1")

    assert result.request_id == "chatcmpl-req-1"
    assert result.connector_keys == [
        "chatcmpl-req-1_0_0_0_0",
        "chatcmpl-req-1_0_0_0_1",
    ]
    assert result.bytes_sent == 246
    assert [p["put_key"] for p in connector.puts] == result.connector_keys
    payload = connector.puts[0]["data"]
    assert isinstance(payload, KVCacheTransferData)
    assert payload.metadata["synthetic_ar_kv"] is True
    assert len(payload.layer_blocks["key_cache"]) == 2
    assert tuple(payload.layer_blocks["key_cache"][0].shape) == (4, 3, 5)
    assert payload.layer_blocks["key_cache"][0].dtype == torch.float16


def test_synthetic_ar_kv_config_rejects_non_divisible_tp():
    with pytest.raises(ValueError, match="evenly divisible"):
        SyntheticARKVConfig(
            num_layers=1,
            seq_len=1,
            num_heads=1,
            head_dim=1,
            from_tp=3,
            to_tp=2,
        ).validate()


def test_synthetic_ar_kv_request_context_preserves_existing_trace_value():
    req = RequestFuncInput(
        prompt="draw a dog",
        api_url="http://test.local/v1/chat/completions",
        model="test-model",
        extra_body={"ar_generated_text": "<think>trace</think><recaption>caption</recaption>"},
    )

    attach_synthetic_ar_kv_request_context(req, "<think>synthetic</think><recaption>fallback</recaption>")

    assert req.extra_body["ar_generated_text"] == "<think>trace</think><recaption>caption</recaption>"


def test_synthetic_ar_kv_request_context_adds_default_ar_text():
    req = RequestFuncInput(
        prompt="draw a dog",
        api_url="http://test.local/v1/chat/completions",
        model="test-model",
    )

    attach_synthetic_ar_kv_request_context(req, "<think>synthetic</think><recaption>caption</recaption>")

    assert req.extra_body["ar_generated_text"] == "<think>synthetic</think><recaption>caption</recaption>"
