# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Collection, Mapping

from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)


def _request_mapping(name: str, value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys.")
    return dict(value)


def normalize_diffusion_request_extra_args(
    *,
    provided_root_fields: Collection[str] = (),
    root_extra_args: object | None = None,
    extra_args: object | None = None,
    extra_params: object | None = None,
    nested_provided_root_fields: Collection[str] = (),
    nested_root_extra_args: object | None = None,
    nested_extra_args: object | None = None,
    nested_extra_params: object | None = None,
    root_field_aliases: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Normalize model-specific request extras without implicit precedence."""
    root = _request_mapping("root_extra_args", root_extra_args)
    nested_root = _request_mapping("nested_root_extra_args", nested_root_extra_args)
    canonical = _request_mapping("extra_args", extra_args)
    legacy = _request_mapping("extra_params", extra_params)
    nested_canonical = _request_mapping("extra_body.extra_args", nested_extra_args)
    nested_legacy = _request_mapping("extra_body.extra_params", nested_extra_params)

    aliases = root_field_aliases or {}
    sources_by_key: dict[str, list[str]] = {}
    sources = (
        ((set(provided_root_fields) | root.keys()), "request"),
        ((set(nested_provided_root_fields) | nested_root.keys()), "request.extra_body"),
        (canonical, "request.extra_args"),
        (legacy, "request.extra_params"),
        (nested_canonical, "request.extra_body.extra_args"),
        (nested_legacy, "request.extra_body.extra_params"),
    )
    for keys, prefix in sources:
        for key in sorted(keys):
            sources_by_key.setdefault(aliases.get(key, key), []).append(f"{prefix}.{key}")

    conflicts = {key: list(dict.fromkeys(paths)) for key, paths in sources_by_key.items() if len(set(paths)) > 1}
    if conflicts:
        if len(conflicts) == 1:
            key, paths = next(iter(conflicts.items()))
            raise ValueError(f'Parameter "{key}" was provided more than once: {", ".join(paths)}.')
        details = "; ".join(f'"{key}": {", ".join(conflicts[key])}' for key in sorted(conflicts))
        raise ValueError(f"Diffusion request parameters were provided more than once: {details}.")

    if extra_params is not None or nested_extra_params is not None:
        logger.warning_once(
            "extra_params is deprecated; use extra_args for model-specific diffusion request parameters."
        )

    # Duplicate request keys were rejected above, so ordering is not precedence.
    normalized = {
        **nested_root,
        **root,
        **legacy,
        **nested_legacy,
        **canonical,
        **nested_canonical,
    }
    return {key: value for key, value in normalized.items() if value is not None}


def apply_normalized_diffusion_request_extra_args(
    sampling_params: OmniDiffusionSamplingParams,
    normalized_extra_args: Mapping[str, object],
) -> None:
    """Overlay request extras without discarding stage defaults."""
    if normalized_extra_args:
        sampling_params.extra_args = {
            **(sampling_params.extra_args or {}),
            **normalized_extra_args,
        }
