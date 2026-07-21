# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import warnings
from collections.abc import Collection, Mapping


def _copy_request_mapping(name: str, value: object | None) -> dict[str, object]:
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
    extra_args: object | None = None,
    extra_params: object | None = None,
    nested_provided_root_fields: Collection[str] = (),
    nested_extra_args: object | None = None,
    nested_extra_params: object | None = None,
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

    sources_by_key: dict[str, list[str]] = {}
    root_fields = set(provided_root_fields) - {"extra_args", "extra_params"}
    for key in root_fields:
        sources_by_key.setdefault(key, []).append(f"request.{key}")
    nested_root_fields = set(nested_provided_root_fields) - {"extra_args", "extra_params"}
    for key in nested_root_fields:
        sources_by_key.setdefault(key, []).append(f"request.extra_body.{key}")
    for key in canonical:
        sources_by_key.setdefault(key, []).append(f"request.extra_args.{key}")
    for key in legacy:
        sources_by_key.setdefault(key, []).append(f"request.extra_params.{key}")
    for key in nested_canonical:
        sources_by_key.setdefault(key, []).append(f"request.extra_body.extra_args.{key}")
    for key in nested_legacy:
        sources_by_key.setdefault(key, []).append(f"request.extra_body.extra_params.{key}")

    conflicts = {key: sources for key, sources in sources_by_key.items() if len(sources) > 1}
    if conflicts:
        if len(conflicts) == 1:
            key, sources = next(iter(conflicts.items()))
            raise ValueError(f'Parameter "{key}" was provided more than once: {", ".join(sources)}.')
        details = "; ".join(f'"{key}": {", ".join(conflicts[key])}' for key in sorted(conflicts))
        raise ValueError(f"Diffusion request parameters were provided more than once: {details}.")

    if extra_params is not None or nested_extra_params is not None:
        warnings.warn(
            "extra_params is deprecated; use extra_args for model-specific diffusion request parameters.",
            FutureWarning,
            stacklevel=2,
        )

    return {**legacy, **nested_legacy, **canonical, **nested_canonical}
