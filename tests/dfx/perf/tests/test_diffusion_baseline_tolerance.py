"""Tests for diffusion benchmark baseline tolerance handling."""

import pytest

from tests.dfx.perf.scripts import run_diffusion_benchmark as runner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_baseline_tolerance_relaxes_throughput_lower_bound():
    params = {
        "baseline-tolerance": 0.1,
        "baseline": {"throughput_qps": 100.0},
    }

    runner.assert_result(
        {"completed_requests": 1, "throughput_qps": 90.0},
        params,
        num_prompts=1,
        assert_baseline=True,
    )

    with pytest.raises(AssertionError, match=r"throughput_qps: 89\.0000 < baseline 90\.0"):
        runner.assert_result(
            {"completed_requests": 1, "throughput_qps": 89.0},
            params,
            num_prompts=1,
            assert_baseline=True,
        )


def test_baseline_tolerance_relaxes_upper_bound_metrics():
    params = {
        "baseline-tolerance": 0.1,
        "baseline": {
            "latency_mean": 10.0,
            "peak_memory_mb_mean": 100.0,
        },
    }

    runner.assert_result(
        {
            "completed_requests": 1,
            "latency_mean": 11.0,
            "peak_memory_mb_mean": 110.0,
        },
        params,
        num_prompts=1,
        assert_baseline=True,
    )

    with pytest.raises(AssertionError, match=r"latency_mean: 11\.1000 > baseline 11\.0"):
        runner.assert_result(
            {
                "completed_requests": 1,
                "latency_mean": 11.1,
                "peak_memory_mb_mean": 100.0,
            },
            params,
            num_prompts=1,
            assert_baseline=True,
        )


def test_baseline_tolerance_defaults_to_strict_threshold():
    params = {"baseline": {"throughput_qps": 100.0}}

    with pytest.raises(AssertionError, match=r"throughput_qps: 99\.9000 < baseline 100\.0"):
        runner.assert_result(
            {"completed_requests": 1, "throughput_qps": 99.9},
            params,
            num_prompts=1,
            assert_baseline=True,
        )
