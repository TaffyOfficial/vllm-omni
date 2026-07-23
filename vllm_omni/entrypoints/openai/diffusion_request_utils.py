# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class CompiledDiffusionRequestOverrides:
    """Request-owned diffusion values after provenance and ownership checks."""

    request_values: Mapping[str, object]
    sampling_overrides: Mapping[str, object]
    declared_extra_args: Mapping[str, object]
    extra_args: Mapping[str, object]
    control_overrides: Mapping[str, object]


def _copy_request_mapping(name: str, value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys.")
    return dict(value)


def _raise_duplicate_sources(sources_by_key: Mapping[str, list[str]]) -> None:
    conflicts = {key: sources for key, sources in sources_by_key.items() if len(sources) > 1}
    if not conflicts:
        return
    if len(conflicts) == 1:
        key, sources = next(iter(conflicts.items()))
        raise ValueError(f'Parameter "{key}" was provided more than once: {", ".join(sources)}.')
    details = "; ".join(f'"{key}": {", ".join(conflicts[key])}' for key in sorted(conflicts))
    raise ValueError(f"Diffusion request parameters were provided more than once: {details}.")


def normalize_diffusion_request_extra_args(
    *,
    provided_root_fields: Collection[str] = (),
    extra_args: object | None = None,
    extra_params: object | None = None,
    nested_provided_root_fields: Collection[str] = (),
    nested_extra_args: object | None = None,
    nested_extra_params: object | None = None,
    root_field_aliases: Mapping[str, str] | None = None,
    stage_provided_root_fields: Mapping[int, Collection[str]] | None = None,
    stage_extra_args: Mapping[int, object | None] | None = None,
) -> dict[str, object]:
    """Normalize model-specific diffusion request arguments without overwrites.

    ``provided_root_fields`` must contain only fields explicitly supplied by the
    caller, not fields populated by request-model defaults. Unknown keys inside
    ``extra_args`` remain valid because their schema is owned by each pipeline.

    Raises:
        ValueError: If an extra-argument container is not an object or if the
            same key is present in more than one request source.
    """
    canonical = _copy_request_mapping("extra_args", extra_args)
    legacy = _copy_request_mapping("extra_params", extra_params)
    nested_canonical = _copy_request_mapping("extra_body.extra_args", nested_extra_args)
    nested_legacy = _copy_request_mapping("extra_body.extra_params", nested_extra_params)

    aliases = root_field_aliases or {}
    sources_by_key: dict[str, list[str]] = {}
    root_fields = set(provided_root_fields) - {"extra_args", "extra_params"}
    for key in sorted(root_fields):
        sources_by_key.setdefault(aliases.get(key, key), []).append(f"request.{key}")
    nested_root_fields = set(nested_provided_root_fields) - {"extra_args", "extra_params"}
    for key in sorted(nested_root_fields):
        sources_by_key.setdefault(aliases.get(key, key), []).append(f"request.extra_body.{key}")
    for key in canonical:
        sources_by_key.setdefault(key, []).append(f"request.extra_args.{key}")
    for key in legacy:
        sources_by_key.setdefault(key, []).append(f"request.extra_params.{key}")
    for key in nested_canonical:
        sources_by_key.setdefault(key, []).append(f"request.extra_body.extra_args.{key}")
    for key in nested_legacy:
        sources_by_key.setdefault(key, []).append(f"request.extra_body.extra_params.{key}")

    _raise_duplicate_sources(sources_by_key)

    stage_root_fields = stage_provided_root_fields or {}
    stage_canonical_args = stage_extra_args or {}
    for stage_index in sorted(set(stage_root_fields) | set(stage_canonical_args)):
        stage_sources_by_key = {key: list(sources) for key, sources in sources_by_key.items()}
        provided_stage_fields = set(stage_root_fields.get(stage_index, ())) - {
            "extra_args",
            "extra_params",
        }
        for key in sorted(provided_stage_fields):
            stage_sources_by_key.setdefault(aliases.get(key, key), []).append(
                f"request.sampling_params_list[{stage_index}].{key}"
            )
        provided_stage_extra_args = _copy_request_mapping(
            f"sampling_params_list[{stage_index}].extra_args",
            stage_canonical_args.get(stage_index),
        )
        for key in provided_stage_extra_args:
            stage_sources_by_key.setdefault(key, []).append(
                f"request.sampling_params_list[{stage_index}].extra_args.{key}"
            )
        _raise_duplicate_sources(stage_sources_by_key)

    if extra_params is not None or nested_extra_params is not None:
        logger.warning_once(
            "extra_params is deprecated; use extra_args for model-specific diffusion request parameters."
        )

    # Duplicate request keys were rejected above, so these mappings are disjoint.
    return {**legacy, **nested_legacy, **canonical, **nested_canonical}


def compile_diffusion_request_overrides(
    *,
    root_values: Mapping[str, object],
    nested_root_values: Mapping[str, object],
    sampling_root_fields: Mapping[str, str],
    declared_extra_fields: Collection[str],
    control_root_fields: Collection[str],
    extra_args: object | None = None,
    extra_params: object | None = None,
    nested_extra_args: object | None = None,
    nested_extra_params: object | None = None,
    stage_provided_root_fields: Mapping[int, Collection[str]] | None = None,
    stage_extra_args: Mapping[int, object | None] | None = None,
) -> CompiledDiffusionRequestOverrides:
    """Compile all request roots into their model-aware runtime consumers.

    A model registry declaration owns that root field and routes it to
    ``extra_args``. Otherwise a serving sampling field routes to its mapped
    sampling attribute. Control fields remain available to dispatchers even
    when a model declaration also consumes them. Server defaults are absent
    because they are not request provenance.
    """
    declared_extra_fields = frozenset(declared_extra_fields)
    effective_sampling_fields = {
        request_field: sampling_field
        for request_field, sampling_field in sampling_root_fields.items()
        if request_field not in declared_extra_fields
    }
    control_root_fields = frozenset(control_root_fields)
    consumed_root_fields = declared_extra_fields | effective_sampling_fields.keys() | control_root_fields
    filtered_root_values = {key: value for key, value in root_values.items() if key in consumed_root_fields}
    filtered_nested_root_values = {
        key: value for key, value in nested_root_values.items() if key in consumed_root_fields
    }

    normalized_extra_args = normalize_diffusion_request_extra_args(
        provided_root_fields=filtered_root_values,
        extra_args=extra_args,
        extra_params=extra_params,
        nested_provided_root_fields=filtered_nested_root_values,
        nested_extra_args=nested_extra_args,
        nested_extra_params=nested_extra_params,
        root_field_aliases=effective_sampling_fields,
        stage_provided_root_fields=stage_provided_root_fields,
        stage_extra_args=stage_extra_args,
    )

    request_values: dict[str, object] = {}
    sampling_overrides: dict[str, object] = {}
    declared_root_extra_args: dict[str, object] = {}
    control_overrides: dict[str, object] = {}
    for source in (filtered_root_values, filtered_nested_root_values):
        for request_field, value in source.items():
            if value is None:
                continue
            request_values[request_field] = value
            if request_field in declared_extra_fields:
                declared_root_extra_args[request_field] = value
            elif request_field in effective_sampling_fields:
                sampling_overrides[effective_sampling_fields[request_field]] = value
            if request_field in control_root_fields:
                control_overrides[request_field] = value

    # Root declarations and argument containers were proven disjoint above.
    # The expansion is therefore a set union, not a precedence rule.
    compiled_extra_args = {
        **declared_root_extra_args,
        **normalized_extra_args,
    }
    return CompiledDiffusionRequestOverrides(
        request_values=MappingProxyType(request_values),
        sampling_overrides=MappingProxyType(sampling_overrides),
        declared_extra_args=MappingProxyType(declared_root_extra_args),
        extra_args=MappingProxyType(compiled_extra_args),
        control_overrides=MappingProxyType(control_overrides),
    )
