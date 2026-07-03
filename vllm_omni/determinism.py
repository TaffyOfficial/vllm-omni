# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
from typing import Any

from vllm import envs


def is_batch_invariant_enabled() -> bool:
    """Return vLLM's batch-invariant mode flag."""
    return bool(envs.VLLM_BATCH_INVARIANT)


def deterministic_request_key(request: Any, *, arrival_time: float | None = None) -> tuple[int, float, str]:
    """Match vLLM priority ordering: priority, arrival time, then request id."""
    request_id = getattr(request, "request_id", None)
    if request_id is None:
        request_ids = getattr(request, "request_ids", None)
        if request_ids:
            request_id = request_ids[0]
    if arrival_time is None:
        arrival_time = getattr(request, "arrival_time", 0.0)
    return (
        int(getattr(request, "priority", 0) or 0),
        float(arrival_time or 0.0),
        str(request_id or ""),
    )


def deterministic_sample_seed(base_seed: int, sample_id: str) -> int:
    """Derive a stable per-sample seed from a request seed and sample id."""
    payload = f"{int(base_seed)}:{sample_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**63)
