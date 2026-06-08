# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for ``piecewise_attn`` (CPU).

Verify that running attention in segments (causal outside full-attn spans,
bidirectional inside full-attn spans) matches running a single full SDPA call
with the equivalent 2D attention mask.

Covers:
  * batch size = 1 and batch size > 1 (homogeneous CFG-like batch)
  * query length == key length   (full prefill)
  * query length <  key length   (decode-like tail slice)
  * various full-attn-span layouts (none / start / middle / end / multi)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    piecewise_attn,
    piecewise_attn_with_plan,
)

DEVICE = torch.device("cpu")


def _sdpa_attn_func(q, k, v, causal, softmax_scale):
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    attn_mask = None
    if causal:
        Sq, Sk = q_.shape[-2], k_.shape[-2]
        i = torch.arange(Sq, device=q.device).unsqueeze(1)
        j = torch.arange(Sk, device=q.device).unsqueeze(0)
        attn_mask = j <= (i + (Sk - Sq))
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def _full_reference(query, key, value, global_spans, q_start, q_end, softmax_scale):
    """Build a full 2D mask with global spans and compute reference output."""
    Sk = key.shape[1]
    mask = torch.tril(torch.ones(Sk, Sk, dtype=torch.bool, device=key.device))
    for a, e in global_spans:
        mask[a:e, :e] = True
    mask_q = mask[q_start:q_end, :]
    q_ = query.transpose(1, 2)
    k_ = key.transpose(1, 2)
    v_ = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask_q, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def _masked_reference(query, key, value, attn_mask, softmax_scale):
    q_ = query.transpose(1, 2)
    k_ = key.transpose(1, 2)
    v_ = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def _piecewise_allowed_mask(global_spans, q_start, q_end, key_len):
    mask = torch.tril(torch.ones(key_len, key_len, dtype=torch.bool, device=DEVICE))
    for a, e in global_spans:
        mask[a:e, :e] = True
    return mask[q_start:q_end]


SPAN_CASES = [
    pytest.param([], id="no-spans"),
    pytest.param([(0, 10)], id="span-at-start"),
    pytest.param([(10, 30), (54, 64)], id="multi-spans"),
]

Q_RANGE_CASES = [
    pytest.param((0, 64), id="q_eq_k"),  # Sq == Sk (prefill)
    pytest.param((53, 64), id="q_lt_k"),  # Sq < Sk (decode-like)
]

BATCH_CASES = [
    pytest.param(1, id="B1"),
    pytest.param(2, id="B2"),
]


@pytest.mark.parametrize("global_spans", SPAN_CASES)
@pytest.mark.parametrize("q_range", Q_RANGE_CASES)
@pytest.mark.parametrize("batch_size", BATCH_CASES)
def test_piecewise_matches_full(global_spans, q_range, batch_size):
    torch.manual_seed(0)
    H, D, Sk = 2, 16, 64
    q_start, q_end = q_range
    Sq = q_end - q_start

    key = torch.randn(batch_size, Sk, H, D, device=DEVICE)
    value = torch.randn(batch_size, Sk, H, D, device=DEVICE)
    query = torch.randn(batch_size, Sq, H, D, device=DEVICE)

    full_attn_spans = [list(global_spans) for _ in range(batch_size)]
    softmax_scale = 1.0 / (D**0.5)

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
    )
    expected = _full_reference(query, key, value, global_spans, q_start, q_end, softmax_scale)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_span_fully_before_qstart():
    """Spans entirely before query region produce pure causal attention."""
    torch.manual_seed(0)
    B, H, D, Sk = 1, 2, 16, 30
    q_start, q_end = 15, 30
    Sq = q_end - q_start

    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)

    global_spans = [(5, 10)]
    full_attn_spans = [list(global_spans) for _ in range(B)]
    softmax_scale = 1.0 / (D**0.5)

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
    )
    expected = _full_reference(query, key, value, global_spans, q_start, q_end, softmax_scale)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_matches_full_with_heterogeneous_spans():
    torch.manual_seed(0)
    B, H, D, Sk = 3, 2, 16, 64
    q_start, q_end = 0, 64
    Sq = q_end - q_start

    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    full_attn_spans = [
        [(8, 20)],
        [(12, 36), (48, 56)],
        [],
    ]
    softmax_scale = 1.0 / (D**0.5)

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
    )
    expected = torch.cat(
        [
            _full_reference(
                query[row : row + 1],
                key[row : row + 1],
                value[row : row + 1],
                spans,
                q_start,
                q_end,
                softmax_scale,
            )
            for row, spans in enumerate(full_attn_spans)
        ],
        dim=0,
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_plan_matches_grouped_reference_with_mixed_signatures():
    torch.manual_seed(0)
    B, H, D, Sk = 4, 2, 16, 64
    q_start, q_end = 0, 64
    Sq = q_end - q_start

    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    full_attn_spans = [
        [],
        [(0, 64)],
        [],
        [(0, 64)],
    ]
    plan = [
        {
            "mask_kind": "baseline",
            "query_range": (q_start, q_end),
            "key_ranges": [(0, Sk)],
            "compact_query_offset": q_start,
            "full_attn_spans": spans,
            "signature": ("baseline", Sq, Sk, q_start, tuple(spans)),
        }
        for spans in full_attn_spans
    ]
    softmax_scale = 1.0 / (D**0.5)
    calls = 0

    def tracked_attn_func(q, k, v, causal, softmax_scale):
        nonlocal calls
        calls += 1
        return _sdpa_attn_func(q, k, v, causal, softmax_scale)

    got = piecewise_attn_with_plan(
        query,
        key,
        value,
        piecewise_mask_plan=plan,
        softmax_scale=softmax_scale,
        attn_func=tracked_attn_func,
    )
    expected = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
    )
    assert calls == 2
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_plan_rejects_complex_mask_kind():
    torch.manual_seed(0)
    B, H, D, Sq, Sk = 1, 2, 16, 8, 8
    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    plan = [
        {
            "mask_kind": "complex",
            "query_range": (0, Sq),
            "key_ranges": [(0, Sk)],
            "compact_query_offset": 0,
            "full_attn_spans": [],
            "signature": ("complex", Sq, Sk, 0, ()),
        }
    ]

    with pytest.raises(ValueError, match="Unsupported piecewise mask plan kind"):
        piecewise_attn_with_plan(
            query,
            key,
            value,
            piecewise_mask_plan=plan,
            softmax_scale=1.0 / (D**0.5),
            attn_func=_sdpa_attn_func,
        )


def test_piecewise_matches_full_with_padding_gaps():
    torch.manual_seed(0)
    B, H, D, Sq, Sk = 2, 2, 16, 3, 7
    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    full_attn_spans = [[(4, 7)], [(4, 7)]]
    softmax_scale = 1.0 / (D**0.5)

    mask = torch.ones(B, Sq, Sk, dtype=torch.bool, device=DEVICE)
    mask[0, :, 2:4] = False

    calls = 0

    def tracked_attn_func(q, k, v, causal, softmax_scale):
        nonlocal calls
        calls += 1
        return _sdpa_attn_func(q, k, v, causal, softmax_scale)

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=tracked_attn_func,
        attn_mask=mask,
    )
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask.unsqueeze(1),
        scale=softmax_scale,
    ).transpose(1, 2)

    assert calls > 0
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_matches_masked_reference_with_padding_hole():
    torch.manual_seed(0)
    B, H, D, Sq, Sk = 2, 2, 16, 4, 6
    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    full_attn_spans = [[(2, 6)], [(2, 6)]]
    softmax_scale = 1.0 / (D**0.5)

    attn_mask = torch.ones(B, 1, Sq, Sk, dtype=torch.bool, device=DEVICE)
    attn_mask[0, :, :, 1] = False

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
        attn_mask=attn_mask,
    )
    expected = _masked_reference(query, key, value, attn_mask, softmax_scale)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_matches_causal_full_mask_with_padding_gap():
    torch.manual_seed(0)
    B, H, D, Sk = 2, 2, 16, 8
    q_start, q_end = 3, 8
    Sq = q_end - q_start
    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    full_attn_spans = [[(5, 8)], [(4, 6)]]
    softmax_scale = 1.0 / (D**0.5)
    attn_mask = torch.stack(
        [_piecewise_allowed_mask(spans, q_start, q_end, Sk) for spans in full_attn_spans],
        dim=0,
    )
    attn_mask[0, :, 1] = False

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
        attn_mask=attn_mask,
    )
    expected = _masked_reference(query, key, value, attn_mask.unsqueeze(1), softmax_scale)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_rejects_masked_span_count_mismatch():
    torch.manual_seed(0)
    B, H, D, Sq, Sk = 2, 2, 16, 3, 5
    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    attn_mask = torch.stack(
        [_piecewise_allowed_mask([], Sk - Sq, Sk, Sk) for _ in range(B)],
        dim=0,
    )

    with pytest.raises(ValueError, match="Expected 2 full-attention span entries, got 1"):
        piecewise_attn(
            query,
            key,
            value,
            full_attn_spans=[[]],
            softmax_scale=1.0 / (D**0.5),
            attn_func=_sdpa_attn_func,
            attn_mask=attn_mask,
        )


def test_piecewise_rejects_pairwise_mask_hole():
    torch.manual_seed(0)
    B, H, D, Sq, Sk = 1, 2, 16, 4, 6
    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    full_attn_spans = [[(2, 6)]]
    softmax_scale = 1.0 / (D**0.5)

    attn_mask = torch.ones(B, Sq, Sk, dtype=torch.bool, device=DEVICE)
    attn_mask[0, 1, 2] = False

    with pytest.raises(ValueError, match="arbitrary query-key mask holes"):
        piecewise_attn(
            query,
            key,
            value,
            full_attn_spans=full_attn_spans,
            softmax_scale=softmax_scale,
            attn_func=_sdpa_attn_func,
            attn_mask=attn_mask,
        )


def test_piecewise_rejects_all_true_pairwise_mask_when_baseline_is_causal():
    torch.manual_seed(0)
    B, H, D, Sq, Sk = 1, 2, 16, 4, 6
    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)
    attn_mask = torch.ones(B, 1, Sq, Sk, dtype=torch.bool, device=DEVICE)

    with pytest.raises(ValueError, match="causal/full-attention layout"):
        piecewise_attn(
            query,
            key,
            value,
            full_attn_spans=[[]],
            softmax_scale=1.0 / (D**0.5),
            attn_func=_sdpa_attn_func,
            attn_mask=attn_mask,
        )
