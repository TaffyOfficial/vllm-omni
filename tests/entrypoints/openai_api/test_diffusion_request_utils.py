# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.openai.diffusion_request_utils import (
    apply_normalized_diffusion_request_extra_args,
    normalize_diffusion_request_extra_args,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SOURCE_KWARGS = (
    "root_extra_args",
    "extra_args",
    "extra_params",
    "nested_root_extra_args",
    "nested_extra_args",
    "nested_extra_params",
)


@pytest.mark.parametrize("source", SOURCE_KWARGS)
def test_normalizer_preserves_pipeline_owned_values(source: str) -> None:
    normalized = normalize_diffusion_request_extra_args(**{source: {"solver": "euler"}})

    assert normalized == {"solver": "euler"}


@pytest.mark.parametrize("source", SOURCE_KWARGS)
def test_explicit_none_preserves_stage_default(source: str) -> None:
    normalized = normalize_diffusion_request_extra_args(**{source: {"solver": None}})
    sampling_params = OmniDiffusionSamplingParams(extra_args={"solver": "euler"})
    apply_normalized_diffusion_request_extra_args(sampling_params, normalized)

    assert sampling_params.extra_args["solver"] == "euler"


def test_request_extras_overlay_matching_defaults() -> None:
    sampling_params = OmniDiffusionSamplingParams(
        extra_args={"solver": "euler", "stage_default": True},
    )

    apply_normalized_diffusion_request_extra_args(sampling_params, {"solver": "ddim"})

    assert sampling_params.extra_args == {"solver": "ddim", "stage_default": True}


def test_non_overlapping_legacy_and_canonical_values_are_merged(mocker) -> None:
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.diffusion_request_utils.logger.warning_once")

    normalized = normalize_diffusion_request_extra_args(
        extra_args={"cfg_text_scale": 7.0},
        nested_extra_params={"sample_solver": "euler"},
    )

    assert normalized == {"cfg_text_scale": 7.0, "sample_solver": "euler"}
    warning_once.assert_called_once()


@pytest.mark.parametrize(
    "kwargs, expected_sources",
    [
        (
            {"provided_root_fields": {"seed"}, "extra_args": {"seed": 2}},
            "request.seed, request.extra_args.seed",
        ),
        (
            {
                "provided_root_fields": {"cfg_scale"},
                "nested_provided_root_fields": {"true_cfg_scale"},
                "root_field_aliases": {"cfg_scale": "true_cfg_scale"},
            },
            "request.cfg_scale, request.extra_body.true_cfg_scale",
        ),
        (
            {"extra_args": {"solver": "euler"}, "extra_params": {"solver": "ddim"}},
            "request.extra_args.solver, request.extra_params.solver",
        ),
        (
            {
                "root_extra_args": {"cfg_text_scale": 7.0},
                "nested_extra_args": {"cfg_text_scale": 8.0},
            },
            "request.cfg_text_scale, request.extra_body.extra_args.cfg_text_scale",
        ),
    ],
)
def test_duplicate_sources_are_rejected(
    kwargs: dict[str, object],
    expected_sources: str,
) -> None:
    with pytest.raises(ValueError, match=expected_sources):
        normalize_diffusion_request_extra_args(**kwargs)


def test_duplicate_validation_precedes_deprecation_warning(mocker) -> None:
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.diffusion_request_utils.logger.warning_once")

    with pytest.raises(ValueError, match="provided more than once"):
        normalize_diffusion_request_extra_args(
            extra_args={"solver": "euler"},
            extra_params={"solver": "ddim"},
        )

    warning_once.assert_not_called()


@pytest.mark.parametrize(
    "name",
    ("extra_args", "extra_params", "nested_extra_args", "nested_extra_params"),
)
def test_public_containers_must_be_objects(name: str) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        normalize_diffusion_request_extra_args(**{name: ["not", "an", "object"]})


def test_container_keys_must_be_strings() -> None:
    with pytest.raises(ValueError, match="must be a JSON object with string keys"):
        normalize_diffusion_request_extra_args(extra_args={1: "invalid"})


def test_multiple_conflicts_are_reported_deterministically() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_diffusion_request_extra_args(
            provided_root_fields={"flow_shift", "seed"},
            extra_args={"seed": 1, "flow_shift": 3.0},
            extra_params={"seed": 2},
        )

    assert str(exc_info.value) == (
        'Diffusion request parameters were provided more than once: "flow_shift": '
        'request.flow_shift, request.extra_args.flow_shift; "seed": request.seed, '
        "request.extra_args.seed, request.extra_params.seed."
    )
