"""Tests for diffusion benchmark warmup request shaping."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.diffusion.backends import RequestFuncInput

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "diffusion"))
from benchmarks.diffusion.diffusion_benchmark_serving import (  # noqa: E402
    CustomDataset,
    _make_warmup_request,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request() -> RequestFuncInput:
    return RequestFuncInput(
        prompt="test",
        api_url="http://127.0.0.1:8000/v1/videos",
        model="test-model",
        num_frames=25,
        num_inference_steps=20,
    )


def test_t2v_warmup_defaults_to_lightweight_single_frame():
    args = SimpleNamespace(
        task="t2v",
        warmup_num_frames=None,
        warmup_num_inference_steps=2,
    )

    warm_req = _make_warmup_request([_request()], 0, args)

    assert warm_req.num_frames == 1
    assert warm_req.num_inference_steps == 2


def test_t2v_warmup_can_match_measured_frame_shape():
    args = SimpleNamespace(
        task="t2v",
        warmup_num_frames=25,
        warmup_num_inference_steps=20,
    )

    warm_req = _make_warmup_request([_request()], 0, args)

    assert warm_req.num_frames == 25
    assert warm_req.num_inference_steps == 20


def test_custom_dataset_inherits_t2v_shape_from_args(tmp_path):
    dataset_path = tmp_path / "prompts.jsonl"
    dataset_path.write_text('{"prompt": "test prompt"}\n', encoding="utf-8")
    args = SimpleNamespace(
        dataset_path=str(dataset_path),
        width=512,
        height=384,
        num_frames=25,
        num_inference_steps=20,
        fps=24,
        seed=123,
        num_prompts=1,
    )

    req = CustomDataset(args, "http://127.0.0.1:8000/v1/videos", "test-model")[0]

    assert req.width == 512
    assert req.height == 384
    assert req.num_frames == 25
    assert req.num_inference_steps == 20
    assert req.fps == 24
    assert req.extra_body == {}
