"""Tests for diffusion benchmark warmup request shaping."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.diffusion.backends import RequestFuncInput

sys.path.insert(0, str(Path(__file__).parents[4] / "benchmarks" / "diffusion"))
from benchmarks.diffusion.diffusion_benchmark_serving import _make_warmup_request  # noqa: E402

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
