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


def _segments_key(segments: list[Segment]) -> tuple[Segment, ...]:
    return tuple(segments)


def _compact_spans(spans: list[tuple[int, int]], kept_positions: list[int]) -> list[tuple[int, int]]:
    compact_spans: list[tuple[int, int]] = []
    for start, end in spans:
        span_positions = [compact_pos for compact_pos, old_pos in enumerate(kept_positions) if start <= old_pos < end]
        if not span_positions:
            continue
        span_start = span_positions[0]
        previous = span_start
        for compact_pos in span_positions[1:]:
            if compact_pos != previous + 1:
                compact_spans.append((span_start, previous + 1))
                span_start = compact_pos
            previous = compact_pos
        compact_spans.append((span_start, previous + 1))
    return compact_spans


def _normalize_attention_mask(attn_mask: torch.Tensor, batch: int, query_len: int, key_len: int) -> torch.Tensor:
    if attn_mask.dtype != torch.bool:
        raise ValueError(f"piecewise attention requires a boolean attention mask, got {attn_mask.dtype}.")
    if attn_mask.ndim == 4:
        if attn_mask.shape[0] != batch or attn_mask.shape[-2:] != (query_len, key_len):
            raise ValueError(
                "4D attention mask shape must be broadcastable to "
                f"({batch}, *, {query_len}, {key_len}), got {tuple(attn_mask.shape)}."
            )
        if attn_mask.shape[1] != 1:
            head_mask = attn_mask[:, :1]
            if not torch.equal(attn_mask, head_mask.expand_as(attn_mask)):
                raise ValueError("piecewise attention does not support different masks per attention head.")
        return attn_mask[:, 0]
    if attn_mask.ndim == 3:
        if tuple(attn_mask.shape) != (batch, query_len, key_len):
            raise ValueError(
                f"3D attention mask shape must be ({batch}, {query_len}, {key_len}), got {tuple(attn_mask.shape)}."
            )
        return attn_mask
    if attn_mask.ndim == 2:
        if tuple(attn_mask.shape) != (batch, key_len):
            raise ValueError(f"2D attention mask shape must be ({batch}, {key_len}), got {tuple(attn_mask.shape)}.")
        return attn_mask.unsqueeze(1).expand(batch, query_len, key_len)
    raise ValueError(f"Unsupported attention mask rank for piecewise attention: {attn_mask.ndim}.")


def _piecewise_mask(
    spans: list[tuple[int, int]],
    query_offset: int,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(query_offset, query_offset + query_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    for start, end in spans:
        query_in_span = (query_positions >= start) & (query_positions < end)
        if torch.any(query_in_span):
            mask[query_in_span] |= key_positions < end
    return mask


def _padding_keep_masks(row_mask: torch.Tensor, baseline_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.any(row_mask & ~baseline_mask):
        raise ValueError("piecewise attention mask cannot allow tokens outside the causal/full-attention layout.")
    query_keep = row_mask.any(dim=-1)
    key_keep = row_mask.any(dim=-2)
    expected_mask = baseline_mask & query_keep[:, None] & key_keep[None, :]
    if not torch.equal(row_mask, expected_mask):
        raise ValueError(
            "piecewise attention only supports padding-style attention masks "
            "that remove whole query rows or key columns; arbitrary query-key "
            "mask holes are not supported."
        )
    return query_keep, key_keep


def _piecewise_attn_grouped(
    query,  # (B, Sq, H, D)
    key,
    value,
    full_attn_spans: list[list[tuple[int, int]]],
    softmax_scale: float,
    attn_func,
    query_offset: int | None = None,
) -> torch.Tensor:
    B, Sq, H, D = query.shape
    if len(full_attn_spans) != B:
        raise ValueError(f"Expected {B} full-attention span entries, got {len(full_attn_spans)}.")

    if query_offset is None:
        query_offset = key.shape[1] - Sq
    out = query.new_zeros(B, Sq, H, D)
    grouped_rows: dict[tuple[Segment, ...], list[int]] = {}
    for row, spans in enumerate(full_attn_spans):
        segments = build_segments(spans, query_offset, Sq)
        grouped_rows.setdefault(_segments_key(segments), []).append(row)

    for segments, rows in grouped_rows.items():
        row_index = query.new_tensor(rows, dtype=torch.long).to(device=query.device)
        query_rows = query.index_select(0, row_index)
        key_rows = key.index_select(0, row_index)
        value_rows = value.index_select(0, row_index)
        out_rows = out.index_select(0, row_index)
        for s, e, mode in segments:
            q_s = s - query_offset
            q_e = e - query_offset
            out_seg = attn_func(
                query_rows[:, q_s:q_e],
                key_rows[:, :e],
                value_rows[:, :e],
                causal=(mode == "causal"),
                softmax_scale=softmax_scale,
            )
            out_rows[:, q_s:q_e] = out_seg
        out.index_copy_(0, row_index, out_rows)
    return out


def piecewise_attn(
    query,  # (B, Sq, H, D)
    key,
    value,
    full_attn_spans: list[list[tuple[int, int]]],
    softmax_scale: float,
    attn_func,
    attn_mask: torch.Tensor | None = None,
):
    if attn_mask is None:
        return _piecewise_attn_grouped(query, key, value, full_attn_spans, softmax_scale, attn_func)

    B, Sq, H, D = query.shape
    if len(full_attn_spans) != B:
        raise ValueError(f"Expected {B} full-attention span entries, got {len(full_attn_spans)}.")
    key_len = key.shape[1]
    original_query_offset = key_len - Sq
    mask = _normalize_attention_mask(attn_mask, B, Sq, key_len)

    out = query.new_zeros(B, Sq, H, D)
    for row, spans in enumerate(full_attn_spans):
        row_mask = mask[row]
        baseline_mask = _piecewise_mask(spans, original_query_offset, Sq, key_len, row_mask.device)
        if attn_mask.ndim == 2:
            row_mask = baseline_mask & row_mask
        query_keep, key_keep = _padding_keep_masks(row_mask, baseline_mask)
        if not torch.any(query_keep):
            continue
        if not torch.any(key_keep):
            raise ValueError("Piecewise attention row has query tokens but no visible key tokens.")

        query_index = torch.nonzero(query_keep, as_tuple=False).flatten()
        key_index = torch.nonzero(key_keep, as_tuple=False).flatten()
        kept_positions = key_index.tolist()
        old_query_positions = (query_index + original_query_offset).tolist()
        compact_query_positions: list[int] = []
        for old_query_position in old_query_positions:
            try:
                compact_query_positions.append(kept_positions.index(old_query_position))
            except ValueError as e:
                raise ValueError(
                    "piecewise attention requires valid query positions to be present in the key mask."
                ) from e
        compact_query_start = compact_query_positions[0]
        if compact_query_positions != list(
            range(compact_query_start, compact_query_start + len(compact_query_positions))
        ):
            raise ValueError("piecewise attention requires contiguous valid query positions after mask compaction.")
        compact_spans = _compact_spans(spans, kept_positions)
        compact_out = _piecewise_attn_grouped(
            query[row : row + 1].index_select(1, query_index),
            key[row : row + 1].index_select(1, key_index),
            value[row : row + 1].index_select(1, key_index),
            [compact_spans],
            softmax_scale,
            attn_func,
            query_offset=compact_query_start,
        )
        out[row : row + 1].index_copy_(1, query_index, compact_out)
    return out
