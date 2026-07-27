# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from dataclasses import dataclass, is_dataclass
from dataclasses import replace as dataclass_replace
from functools import partial
from types import MappingProxyType
from typing import Any

from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.image_api_utils import validate_layered_layers
from vllm_omni.entrypoints.openai.stage_params import get_default_sampling_params_list
from vllm_omni.entrypoints.openai.utils import (
    get_stage_type,
    is_single_stage_diffusion,
    parse_lora_request,
    resolve_diffusion_od_config,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    get_extra_body_params,
    should_init_extra_args_for_non_diffusion_stages,
)

logger = init_logger(__name__)

_SAMPLING_ROOT_FIELDS = {
    "height": "height",
    "width": "width",
    "num_outputs_per_prompt": "num_outputs_per_prompt",
    "seed": "seed",
    "num_inference_steps": "num_inference_steps",
    "guidance_scale": "guidance_scale",
    "true_cfg_scale": "true_cfg_scale",
    "cfg_scale": "true_cfg_scale",
    "num_frames": "num_frames",
    "guidance_scale_2": "guidance_scale_2",
    "layers": "layers",
    "resolution": "resolution",
}
_CONTROL_ROOT_FIELDS = frozenset({"size", "negative_prompt", "lora", "modalities"})


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
    _stage_params: tuple[Any, ...]

    def clone_sampling_params_list(self) -> list[Any]:
        """Return isolated parameters for one dispatcher invocation."""
        return [_clone(params, "sampling parameters") for params in self._stage_params]


@dataclass(frozen=True)
class DiffusionChatRequestContext:
    """Model topology and request-field policy consumed by the compiler."""

    stage_types: tuple[str, ...]
    default_sampling_params_list: tuple[object, ...]
    comprehension_stage_index: int | None
    standard_sampling_fields: frozenset[str]
    declared_extra_fields: frozenset[str]
    apply_declared_to_non_diffusion: bool
    supported_modalities: frozenset[str]


def resolve_diffusion_chat_request_context(
    *,
    engine_client: object,
    diffusion_engine: object | None,
    diffusion_mode: bool,
    standard_sampling_fields: Collection[str],
    declared_extra_fields: Collection[str] | None,
) -> DiffusionChatRequestContext:
    """Resolve all model-owned compiler policy from the active topology."""
    stage_owner = diffusion_engine if diffusion_mode else engine_client
    stage_configs = list(getattr(stage_owner, "stage_configs", ()) or ())
    declared = frozenset(declared_extra_fields) if declared_extra_fields is not None else None
    fan_out_declared = False
    try:
        od_config = resolve_diffusion_od_config(engine_client, diffusion_engine)
        model_class_name = getattr(od_config, "model_class_name", None)
        if isinstance(model_class_name, str):
            if declared is None:
                declared = get_extra_body_params(model_class_name)
            fan_out_declared = should_init_extra_args_for_non_diffusion_stages(model_class_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to resolve diffusion request model extras: {exc}") from exc

    supported_modalities = {
        modality for modality in getattr(stage_owner, "output_modalities", ()) if modality is not None
    }
    if is_single_stage_diffusion(stage_owner):
        supported_modalities.add("text")
    return DiffusionChatRequestContext(
        stage_types=tuple(get_stage_type(stage) for stage in stage_configs),
        default_sampling_params_list=tuple(get_default_sampling_params_list(stage_owner)),
        comprehension_stage_index=next(
            (index for index, stage in enumerate(stage_configs) if getattr(stage, "is_comprehension", False)),
            None,
        ),
        standard_sampling_fields=frozenset(standard_sampling_fields),
        declared_extra_fields=declared or frozenset(),
        apply_declared_to_non_diffusion=fan_out_declared,
        supported_modalities=frozenset(supported_modalities),
    )


@dataclass(frozen=True)
class _Assignment:
    key: str
    target: str
    value: object
    source: str
    parse_value: bool


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
    if key in {"num_inference_steps", "height", "width", "target_h", "target_w"} and value is not None:
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


def _replace_params(params: Any, overrides: Mapping[str, object]) -> Any:
    if is_dataclass(params):
        return dataclass_replace(params, **overrides)
    values = {
        name: getattr(params, name)
        for name in type(params).__struct_fields__
        if not name.startswith("_") and name != "output_text_buffer_length"
    }
    return type(params)(**(values | overrides))


class _PlanBuilder:
    def __init__(
        self,
        request: object,
        context: DiffusionChatRequestContext,
    ) -> None:
        self.request = request
        self.declared = context.declared_extra_fields
        self.standard_fields = context.standard_sampling_fields
        self.sampling_fields = {
            key: target for key, target in _SAMPLING_ROOT_FIELDS.items() if key not in self.declared
        }
        self.fan_out_declared = context.apply_declared_to_non_diffusion
        self.supported_modalities = context.supported_modalities
        self.comprehension_stage = context.comprehension_stage_index
        self.types = list(context.stage_types) or ["diffusion"]
        self.params = [
            _stage_params(
                stage_type,
                context.default_sampling_params_list[index]
                if index < len(context.default_sampling_params_list)
                else None,
            )
            for index, stage_type in enumerate(self.types)
        ]
        self.diffusion_stages = [index for index, stage_type in enumerate(self.types) if stage_type == "diffusion"]
        self.items: list[list[_Assignment]] = [[] for _ in self.params]
        self.claims: list[dict[str, list[str]]] = [{} for _ in self.params]

    def collect_sources(self) -> None:
        consumed = self.declared | self.sampling_fields.keys() | _CONTROL_ROOT_FIELDS
        model_extra = getattr(self.request, "model_extra", None)
        self.flat = dict(model_extra) if isinstance(model_extra, Mapping) else {}
        nested_value = self.flat.pop("extra_body", None)
        if nested_value is None:
            nested_value = getattr(self.request, "extra_body", None)
        self.nested = _mapping("extra_body", nested_value)
        explicit = getattr(self.request, "model_fields_set", None)
        if explicit is None:
            explicit = getattr(self.request, "__fields_set__", ())
        self.explicit = set(explicit) if isinstance(explicit, Collection) else set()
        for key in self.explicit:
            if key not in self.flat and (
                key in consumed or key in self.standard_fields or key == "sampling_params_list"
            ):
                self.flat[key] = getattr(self.request, key, None)

        self.values: list[tuple[str, object, str, bool]] = [
            (key, values[key], f"{prefix}.{key}", False)
            for values, prefix in ((self.flat, "request"), (self.nested, "request.extra_body"))
            for key in sorted(values)
            if key in consumed
        ]
        for name, value in (
            ("request.extra_args", self.flat.get("extra_args")),
            ("request.extra_params", self.flat.get("extra_params")),
            ("request.extra_body.extra_args", self.nested.get("extra_args")),
            ("request.extra_body.extra_params", self.nested.get("extra_params")),
        ):
            values = _mapping(name.removeprefix("request."), value)
            self.values.extend((key, values[key], f"{name}.{key}", True) for key in sorted(values))

        claims: dict[str, list[str]] = {}
        self.by_key: dict[str, tuple[object, str]] = {}
        for key, value, source, is_container in self.values:
            conflict_key = key if key in self.declared else _SAMPLING_ROOT_FIELDS.get(key, key)
            claims.setdefault(conflict_key, []).append(source)
            if not is_container or key in self.declared:
                self.by_key[key] = (value, source)
        _reject_duplicates(claims)

    def validate_sources(self) -> None:
        self.controls = {
            key: value
            for key, (value, _source) in self.by_key.items()
            if key in _CONTROL_ROOT_FIELDS and key not in {"size", "lora"} and value is not None
        }
        modalities = self.controls.get("modalities")
        if modalities is not None:
            if not isinstance(modalities, list) or not all(isinstance(modality, str) for modality in modalities):
                raise ValueError("'modalities' must be a list of strings.")
            unsupported = set(modalities) - self.supported_modalities
            if unsupported:
                raise ValueError(
                    f"Unsupported output modalities {', '.join(sorted(unsupported))} for this model. "
                    f"Supported modalities: {', '.join(sorted(self.supported_modalities))}"
                )

        self.height = self.by_key.get("height", (None, ""))[0]
        self.width = self.by_key.get("width", (None, ""))[0]
        self.invalid_size: object | None = None
        if "size" in self.by_key and (self.height is None or self.width is None):
            size, source = self.by_key["size"]
            try:
                if isinstance(size, str) and "x" in size.lower():
                    width, height = (int(part) for part in size.lower().split("x"))
                    for key, value in (("height", height), ("width", width)):
                        if getattr(self, key) is None:
                            setattr(self, key, value)
                            self.by_key[key] = (value, source)
                            self.values.append((key, value, source, False))
            except ValueError:
                self.invalid_size = size
        self.controls.update(
            {
                key: _parse(key, value)
                for key, value in (("height", self.height), ("width", self.width))
                if value is not None
            }
        )

    def _assign(
        self,
        stages: int | Collection[int],
        key: str,
        value: object,
        source: str,
        *,
        target: str | None = None,
        parse_value: bool = True,
    ) -> None:
        for stage in (stages,) if isinstance(stages, int) else stages:
            self.claims[stage].setdefault(key, []).append(source)
            self.items[stage].append(
                _Assignment(
                    key=key,
                    target=target or key,
                    value=value,
                    source=source,
                    parse_value=parse_value,
                )
            )

    def build_stage_assignments(self) -> None:
        assign = self._assign
        extra = partial(assign, target="extra_args")
        diffusion = self.diffusion_stages
        declared_stages: Collection[int] = range(len(self.types)) if self.fan_out_declared else diffusion
        for key, value, source, is_container in self.values:
            if value is None:
                continue
            if key in self.declared:
                extra(declared_stages, key, value, source)
            elif is_container:
                extra(diffusion, key, value, source, parse_value=False)
            elif key in self.sampling_fields:
                assign(diffusion, self.sampling_fields[key], value, source)

        stage = self.comprehension_stage
        if stage is not None:
            if self.fan_out_declared and self.types[stage] != "diffusion":
                for key, value, source, _is_container in self.values:
                    if value is not None and key in self.declared and key in self.standard_fields:
                        assign(stage, key, value, source)
            for key in self.standard_fields & self.explicit - self.declared:
                if self.types[stage] == "diffusion" and key in self.sampling_fields:
                    continue
                value = getattr(self.request, key, None)
                if value is not None and (not isinstance(value, list) or value):
                    assign(stage, key, value, f"request.{key}")
            for key, value, source_key in (
                ("target_h", self.height, "height"),
                ("target_w", self.width, "width"),
            ):
                if value is not None:
                    extra(stage, key, value, self.by_key[source_key][1])

        stage_sources = [
            (f"{prefix}.sampling_params_list", values["sampling_params_list"])
            for values, prefix in ((self.flat, "request"), (self.nested, "request.extra_body"))
            if "sampling_params_list" in values
        ]
        _reject_duplicates({"sampling_params_list": [source for source, _value in stage_sources]})
        stage_source, raw_stages = (
            stage_sources[0]
            if stage_sources
            else ("request.sampling_params_list", getattr(self.request, "sampling_params_list", None))
        )
        if raw_stages is not None:
            if not isinstance(raw_stages, list):
                raise ValueError(f"{stage_source} must be a JSON array.")
            if len(raw_stages) > len(self.params):
                raise ValueError(
                    f"sampling_params_list has {len(raw_stages)} entries, "
                    f"but the pipeline has {len(self.params)} stages."
                )
            for stage, raw_params in enumerate(raw_stages):
                values = _mapping(f"{stage_source}[{stage}]", raw_params, required=True)
                extras = _mapping(f"{stage_source}[{stage}].extra_args", values.pop("extra_args", None))
                is_diffusion = self.types[stage] == "diffusion"
                for key in sorted(values):
                    target = "extra_args" if is_diffusion and key in self.declared else None
                    claim = self.sampling_fields.get(key, key) if is_diffusion and target is None else key
                    assign(stage, claim, values[key], f"{stage_source}[{stage}].{key}", target=target)
                for key in sorted(extras):
                    extra(
                        stage,
                        key,
                        extras[key],
                        f"{stage_source}[{stage}].extra_args.{key}",
                        parse_value=is_diffusion and key in self.declared,
                    )

        for claims in self.claims:
            _reject_duplicates(claims)
        if "lora" in self.by_key and self.by_key["lora"][0] is not None:
            value, source = self.by_key["lora"]
            try:
                lora_request, lora_scale = parse_lora_request(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(str(exc)) from exc
            if lora_request is not None:
                assign(diffusion, "lora_request", lora_request, source)
            if lora_scale is not None:
                assign(diffusion, "lora_scale", lora_scale, source)

    def apply_assignments(self) -> None:
        for claims in self.claims:
            _reject_duplicates(claims)
        for stage, (params, items) in enumerate(zip(self.params, self.items)):
            overrides: dict[str, object] = {}
            extra_args: dict[str, object] = {}
            for assignment in items:
                if assignment.target == "extra_args":
                    if not hasattr(params, "extra_args"):
                        raise ValueError(f"{assignment.source} is not supported by stage {stage}.")
                    destination = extra_args
                elif hasattr(params, assignment.target):
                    destination = overrides
                else:
                    raise ValueError(
                        f'{assignment.source} targets unsupported parameter "{assignment.target}" for stage {stage}.'
                    )
                destination[assignment.key if assignment.target == "extra_args" else assignment.target] = (
                    _parse(assignment.key, assignment.value) if assignment.parse_value else assignment.value
                )
            if not overrides and not extra_args:
                continue
            if extra_args:
                overrides["extra_args"] = {
                    **(getattr(params, "extra_args", None) or {}),
                    **extra_args,
                }
            try:
                self.params[stage] = _replace_params(params, overrides)
            except Exception as exc:
                raise ValueError(f"Invalid sampling parameters for stage {stage}: {exc}") from exc


def compile_diffusion_chat_request_plan(
    *,
    request: object,
    context: DiffusionChatRequestContext,
) -> DiffusionChatRequestPlan:
    """Compile all Chat request sources into final typed stage parameters."""
    builder = _PlanBuilder(request, context)
    builder.collect_sources()
    builder.validate_sources()
    builder.build_stage_assignments()
    builder.apply_assignments()
    if builder.invalid_size is not None:
        logger.warning("Invalid size format: %s", builder.invalid_size)
    if builder.flat.get("extra_params") is not None or builder.nested.get("extra_params") is not None:
        logger.warning_once(
            "extra_params is deprecated; use extra_args for model-specific diffusion request parameters."
        )
    return DiffusionChatRequestPlan(
        controls=MappingProxyType(copy.deepcopy(builder.controls)),
        _stage_params=tuple(builder.params),
    )
