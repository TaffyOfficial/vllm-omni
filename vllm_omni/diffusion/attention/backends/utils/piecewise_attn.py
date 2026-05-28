# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Piecewise attention for mixed causal / full (bidirectional) masks.

Dispatches each segment as a separate attention call whose causal flag
follows FlashAttention's bottom-right convention (``K[:e]`` is attended by
``Q[s:e]``, with causal alignment anchored at the bottom-right corner).

Per segment:
  - causal segment ``[s, e)``: ``attn(Q[:, s:e], K[:, :e], V[:, :e], causal=True)``
  - full-attn span ``[a, e)``: ``attn(Q[:, a:e], K[:, :e], V[:, :e], causal=False)``
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch


class Segment(NamedTuple):
    start: int
    end: int
    mode: Literal["causal", "full"]


def build_segments(full_attn_spans, query_offset, query_len):
    """
    full_attn_spans: list of (start, end) half-open spans in global coordinates
    query_offset: starting position of query in the global sequence
    query_len: length of the query

    return:
        List[Segment] in global coordinates, clipped to [query_offset, query_offset + query_len)
    """
    q_start = query_offset
    q_end = query_offset + query_len

    segs: list[Segment] = []
    cur = q_start

    for a, e in full_attn_spans:
        # clip span to query range
        a_clipped = max(a, q_start)
        e_clipped = min(e, q_end)
        if a_clipped >= e_clipped:
            continue

        if cur < a_clipped:
            segs.append(Segment(cur, a_clipped, "causal"))
        segs.append(Segment(a_clipped, e_clipped, "full"))
        cur = e_clipped

    if cur < q_end:
        segs.append(Segment(cur, q_end, "causal"))

    return segs


def _piecewise_attn_homogeneous(
    query,  # (B, Sq, H, D)
    key,
    value,
    spans: list[tuple[int, int]],
    softmax_scale: float,
    attn_func,
):
    B, Sq, H, D = query.shape

    query_offset = key.shape[1] - Sq
    out = query.new_zeros(B, Sq, H, D)

    for s, e, mode in build_segments(spans, query_offset, Sq):
        q_s = s - query_offset
        q_e = e - query_offset
        out_seg = attn_func(
            query[:, q_s:q_e],
            key[:, :e],
            value[:, :e],
            causal=(mode == "causal"),
            softmax_scale=softmax_scale,
        )
        out[:, q_s:q_e] = out_seg
    return out


def piecewise_attn(
    query,  # (B, Sq, H, D)
    key,
    value,
    full_attn_spans: list[list[tuple[int, int]]],
    softmax_scale: float,
    attn_func,
):
    B = query.shape[0]
    if len(full_attn_spans) != B:
        raise ValueError(
            f"piecewise_attn expects one full_attn_spans entry per batch row, got {len(full_attn_spans)} for batch {B}"
        )

    normalized_spans = [tuple((int(start), int(end)) for start, end in spans) for spans in full_attn_spans]
    span_groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for row_idx, spans in enumerate(normalized_spans):
        span_groups.setdefault(spans, []).append(row_idx)

    if len(span_groups) == 1:
        return _piecewise_attn_homogeneous(
            query,
            key,
            value,
            list(normalized_spans[0]),
            softmax_scale,
            attn_func,
        )

    out = query.new_empty(query.shape)
    for span_key, row_indices in span_groups.items():
        index = query.new_tensor(row_indices, dtype=torch.long)
        group_out = _piecewise_attn_homogeneous(
            query.index_select(0, index),
            key.index_select(0, index),
            value.index_select(0, index),
            list(span_key),
            softmax_scale,
            attn_func,
        )
        out.index_copy_(0, index, group_out)
    return out
