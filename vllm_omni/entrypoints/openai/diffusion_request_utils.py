# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any

from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.image_api_utils import validate_layered_layers
from vllm_omni.entrypoints.openai.utils import parse_lora_request
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)


def _clone(params: Any, source: str) -> Any:
    try:
        cloned = copy.deepcopy(params)
    except Exception as exc:
        raise RuntimeError(f"Unable to isolate {source}: {exc}") from exc
    if cloned is params:
        raise RuntimeError(f"Unable to isolate {source}: deepcopy returned the original object.")
    return cloned


@dataclass(frozen=True)
class DiffusionChatRequestPlan:
    """Final controls and private per-stage parameter prototypes."""

    controls: Mapping[str, object]
    request_sources: Mapping[str, object]
    _stage_params: tuple[Any, ...]

    def clone_sampling_params_list(self) -> list[Any]:
        """Return isolated parameters for one dispatcher invocation."""
        return [_clone(params, "sampling parameters") for params in self._stage_params]


def _mapping(
    name: str,
    value: object | None,
    *,
    required: bool = False,
) -> dict[str, object]:
    if value is None:
        if required:
            raise ValueError(f"{name} must be a JSON object.")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys.")
    return dict(value)


def _reject_duplicates(sources_by_key: Mapping[str, list[str]]) -> None:
    # Registry fan-out may route one source to two stage consumers; only
    # distinct request source paths are duplicates.
    conflicts = {key: list(dict.fromkeys(sources)) for key, sources in sources_by_key.items() if len(set(sources)) > 1}
    if not conflicts:
        return
    if len(conflicts) == 1:
        key, sources = next(iter(conflicts.items()))
        raise ValueError(f'Parameter "{key}" was provided more than once: {", ".join(sources)}.')
    details = "; ".join(f'"{key}": {", ".join(conflicts[key])}' for key in sorted(conflicts))
    raise ValueError(f"Diffusion request parameters were provided more than once: {details}.")


def _parse(key: str, value: object) -> object:
    if key == "num_inference_steps" and value is not None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("num_inference_steps must be an integer.") from exc
    if key in {"height", "width", "target_h", "target_w"} and value is not None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be an integer.") from exc
    if key == "layers" and value is not None:
        return validate_layered_layers(value)
    return value


def _stage_params(stage_type: str, default: object | None) -> Any:
    cls = OmniDiffusionSamplingParams if stage_type == "diffusion" else SamplingParams
    if default is None:
        return cls()
    if isinstance(default, Mapping):
        try:
            return cls(**copy.deepcopy(dict(default)))
        except Exception as exc:
            raise RuntimeError(f"Invalid default sampling parameters: {exc}") from exc
    if isinstance(default, cls):
        return _clone(default, "default sampling parameters")
    raise RuntimeError(
        f"Default sampling parameters for {stage_type} stage have incompatible type {type(default).__name__}."
    )


def _init_values(params: Any) -> dict[str, object]:
    if is_dataclass(params):
        names = [field.name for field in fields(params) if field.init]
    else:
        names = list(getattr(type(params), "__struct_fields__", ()))
        if not names:
            names = list(getattr(params, "__dict__", ()))
        names = [name for name in names if not name.startswith("_") and name != "output_text_buffer_length"]
    return {name: getattr(params, name) for name in names}


class _Assignments:
    """Collect provenance, then build final typed stages from cloned defaults."""

    def __init__(self, params: list[Any]) -> None:
        self.params = params
        self.items: list[tuple[int, str, str, object, str, bool, bool]] = []

    def add(
        self,
        stage: int,
        key: str,
        value: object,
        source: str,
        *,
        target: str | None = None,
        keep_none: bool = False,
        parse_value: bool = True,
    ) -> None:
        self.items.append((stage, key, target or key, value, source, keep_none, parse_value))

    def reject_duplicates(self) -> None:
        for stage in range(len(self.params)):
            claims: dict[str, list[str]] = {}
            for claimed_stage, key, _target, _value, source, _keep_none, _parse_value in self.items:
                if claimed_stage == stage:
                    claims.setdefault(key, []).append(source)
            _reject_duplicates(claims)

    def apply(self) -> None:
        self.reject_duplicates()
        for stage, params in enumerate(self.params):
            overrides: dict[str, object] = {}
            extra_args: dict[str, object] = {}
            for item_stage, key, target, value, source, keep_none, parse_value in self.items:
                if item_stage != stage or (value is None and not keep_none):
                    continue
                if target == "extra_args":
                    if not hasattr(params, "extra_args"):
                        raise ValueError(f"{source} is not supported by stage {stage}.")
                    destination = extra_args
                elif hasattr(params, target):
                    destination = overrides
                else:
                    raise ValueError(f'{source} targets unsupported parameter "{target}" for stage {stage}.')
                destination[key if target == "extra_args" else target] = _parse(key, value) if parse_value else value
            if not overrides and not extra_args:
                continue
            if extra_args:
                overrides["extra_args"] = {
                    **(getattr(params, "extra_args", None) or {}),
                    **extra_args,
                }
            try:
                self.params[stage] = type(params)(**{**_init_values(params), **overrides})
            except Exception as exc:
                raise ValueError(f"Invalid sampling parameters for stage {stage}: {exc}") from exc


def compile_diffusion_chat_request_plan(
    *,
    request: object,
    stage_types: Sequence[str],
    default_sampling_params_list: Sequence[object],
    comprehension_stage_index: int | None,
    sampling_root_fields: Mapping[str, str],
    standard_sampling_fields: Collection[str],
    control_root_fields: Collection[str],
    declared_extra_fields: Collection[str],
    apply_declared_to_non_diffusion: bool,
) -> DiffusionChatRequestPlan:
    """Compile all Chat request sources into final typed stage parameters."""
    declared = frozenset(declared_extra_fields)
    controls_fields = frozenset(control_root_fields)
    sampling_fields = {
        request_key: target_key
        for request_key, target_key in sampling_root_fields.items()
        if request_key not in declared
    }
    consumed = declared | sampling_fields.keys() | controls_fields

    model_extra = getattr(request, "model_extra", None)
    flat = dict(model_extra) if isinstance(model_extra, Mapping) else {}
    nested_value = flat.pop("extra_body", None)
    if nested_value is None:
        nested_value = getattr(request, "extra_body", None)
    nested = _mapping("extra_body", nested_value)

    explicit = getattr(request, "model_fields_set", None)
    if explicit is None:
        explicit = getattr(request, "__fields_set__", ())
    explicit = set(explicit) if isinstance(explicit, Collection) else set()
    for key in explicit:
        if key not in flat and (key in consumed or key in standard_sampling_fields or key == "sampling_params_list"):
            flat[key] = getattr(request, key, None)

    roots: list[tuple[str, object, str]] = []
    for values, prefix in ((flat, "request"), (nested, "request.extra_body")):
        roots.extend((key, values[key], f"{prefix}.{key}") for key in sorted(values) if key in consumed)

    raw_containers = (
        ("request.extra_args", flat.get("extra_args")),
        ("request.extra_params", flat.get("extra_params")),
        ("request.extra_body.extra_args", nested.get("extra_args")),
        ("request.extra_body.extra_params", nested.get("extra_params")),
    )
    containers = [(name, _mapping(name.removeprefix("request."), value)) for name, value in raw_containers]
    global_values = [(key, value, source, False) for key, value, source in roots]
    global_values.extend(
        (key, values[key], f"{name}.{key}", True) for name, values in containers for key in sorted(values)
    )
    global_claims: dict[str, list[str]] = {}
    for key, _value, source, _is_container in global_values:
        conflict_key = key if key in declared else sampling_root_fields.get(key, key)
        global_claims.setdefault(conflict_key, []).append(source)
    _reject_duplicates(global_claims)

    types = list(stage_types) or ["diffusion"]
    params = [
        _stage_params(
            stage_type,
            default_sampling_params_list[index] if index < len(default_sampling_params_list) else None,
        )
        for index, stage_type in enumerate(types)
    ]
    diffusion_stages = [index for index, stage_type in enumerate(types) if stage_type == "diffusion"]
    assignments = _Assignments(params)
    serving_by_key = {
        key: (value, source)
        for key, value, source, is_container in global_values
        if not is_container or key in declared
    }

    controls = {
        key: value
        for key, (value, _source) in serving_by_key.items()
        if key in controls_fields and key not in {"size", "lora"} and value is not None
    }
    modalities = controls.get("modalities")
    if modalities is not None and (
        not isinstance(modalities, list) or not all(isinstance(modality, str) for modality in modalities)
    ):
        raise ValueError("'modalities' must be a list of strings.")
    height = serving_by_key.get("height", (None, ""))[0]
    width = serving_by_key.get("width", (None, ""))[0]
    invalid_size: object | None = None
    if "size" in serving_by_key and (height is None or width is None):
        size, source = serving_by_key["size"]
        try:
            if isinstance(size, str) and "x" in size.lower():
                parsed_width, parsed_height = (int(part) for part in size.lower().split("x"))
                if height is None:
                    height = parsed_height
                    serving_by_key["height"] = (height, source)
                if width is None:
                    width = parsed_width
                    serving_by_key["width"] = (width, source)
        except ValueError:
            invalid_size = size

    for request_key, value, source, is_container in global_values:
        if request_key in declared:
            targets = list(range(len(types))) if apply_declared_to_non_diffusion else diffusion_stages
            for stage in targets:
                assignments.add(
                    stage,
                    request_key,
                    value,
                    source,
                    target="extra_args",
                )
        elif is_container:
            for stage in diffusion_stages:
                assignments.add(
                    stage,
                    request_key,
                    value,
                    source,
                    target="extra_args",
                    keep_none=True,
                    parse_value=False,
                )
        elif request_key in sampling_fields:
            target = sampling_fields[request_key]
            for stage in diffusion_stages:
                assignments.add(stage, target, value, source)

    if comprehension_stage_index is not None:
        if apply_declared_to_non_diffusion and types[comprehension_stage_index] != "diffusion":
            for key, value, source, _is_container in global_values:
                if key in declared and key in standard_sampling_fields:
                    assignments.add(comprehension_stage_index, key, value, source)
        for key in standard_sampling_fields:
            if key not in explicit:
                continue
            if key in declared:
                continue
            if types[comprehension_stage_index] == "diffusion" and key in sampling_fields:
                continue
            value = getattr(request, key, None)
            if not isinstance(value, list) or value:
                assignments.add(comprehension_stage_index, key, value, f"request.{key}")
        for key, value in (("target_h", height), ("target_w", width)):
            if value is not None:
                source_key = "height" if key == "target_h" else "width"
                assignments.add(
                    comprehension_stage_index,
                    key,
                    value,
                    serving_by_key[source_key][1],
                    target="extra_args",
                )

    stage_sources = [
        (f"{prefix}.sampling_params_list", values["sampling_params_list"])
        for values, prefix in ((flat, "request"), (nested, "request.extra_body"))
        if "sampling_params_list" in values
    ]
    _reject_duplicates({"sampling_params_list": [source for source, _value in stage_sources]})
    stage_source, raw_stages = (
        stage_sources[0]
        if stage_sources
        else ("request.sampling_params_list", getattr(request, "sampling_params_list", None))
    )
    if raw_stages is not None:
        if not isinstance(raw_stages, list):
            raise ValueError(f"{stage_source} must be a JSON array.")
        if len(raw_stages) > len(params):
            raise ValueError(
                f"sampling_params_list has {len(raw_stages)} entries, but the pipeline has {len(params)} stages."
            )
        for stage, raw_params in enumerate(raw_stages):
            stage_values = _mapping(
                f"{stage_source}[{stage}]",
                raw_params,
                required=True,
            )
            stage_extras = _mapping(
                f"{stage_source}[{stage}].extra_args",
                stage_values.pop("extra_args", None),
            )
            is_diffusion = types[stage] == "diffusion"
            for key in sorted(stage_values):
                value = stage_values[key]
                source = f"{stage_source}[{stage}].{key}"
                if is_diffusion and key in declared:
                    assignments.add(stage, key, value, source, target="extra_args", keep_none=True)
                else:
                    target = sampling_fields.get(key, key) if is_diffusion else key
                    assignments.add(stage, target, value, source, keep_none=True)
            for key in sorted(stage_extras):
                value = stage_extras[key]
                assignments.add(
                    stage,
                    key,
                    value,
                    f"{stage_source}[{stage}].extra_args.{key}",
                    target="extra_args",
                    keep_none=True,
                    parse_value=is_diffusion and key in declared,
                )

    assignments.reject_duplicates()
    controls.update(
        {key: _parse(key, value) for key, value in (("height", height), ("width", width)) if value is not None}
    )
    if "lora" in serving_by_key and serving_by_key["lora"][0] is not None:
        value, source = serving_by_key["lora"]
        try:
            lora_request, lora_scale = parse_lora_request(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(str(exc)) from exc
        for stage in diffusion_stages:
            if lora_request is not None:
                assignments.add(stage, "lora_request", lora_request, source)
            if lora_scale is not None:
                assignments.add(stage, "lora_scale", lora_scale, source)

    assignments.apply()
    # Source-qualified keys are disjoint; this expansion has no precedence.
    request_sources = {
        **{f"request.extra_body.{key}": value for key, value in nested.items()},
        **{f"request.{key}": value for key, value in flat.items()},
    }
    plan = DiffusionChatRequestPlan(
        controls=MappingProxyType(copy.deepcopy(controls)),
        request_sources=MappingProxyType(copy.deepcopy(request_sources)),
        _stage_params=tuple(params),
    )
    if invalid_size is not None:
        logger.warning("Invalid size format: %s", invalid_size)
    if flat.get("extra_params") is not None or nested.get("extra_params") is not None:
        logger.warning_once(
            "extra_params is deprecated; use extra_args for model-specific diffusion request parameters."
        )
    return plan
