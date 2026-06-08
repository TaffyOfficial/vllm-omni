# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _make_piecewise_mask_plan,
    _shift_piecewise_mask_plan,
)


def test_make_piecewise_mask_plan_aligns_rows_with_batch():
    spans = [[(4, 8)], [(2, 6), (10, 12)]]

    plan = _make_piecewise_mask_plan(spans, query_len=16, key_len=16)

    assert len(plan) == 2
    assert plan[0]["mask_kind"] == "baseline"
    assert plan[0]["query_range"] == (0, 16)
    assert plan[0]["key_ranges"] == [(0, 16)]
    assert plan[0]["compact_query_offset"] == 0
    assert plan[0]["full_attn_spans"] == [(4, 8)]
    assert plan[1]["full_attn_spans"] == [(2, 6), (10, 12)]
    assert plan[0]["signature"] != plan[1]["signature"]


def test_shift_piecewise_mask_plan_moves_current_region_after_prefix_padding():
    plan = _make_piecewise_mask_plan([[(6, 10)]], query_len=4, key_len=10)

    shifted = _shift_piecewise_mask_plan(plan, prefix_len=6, max_prefix_len=8)

    assert shifted[0]["mask_kind"] == "complex"
    assert shifted[0]["query_range"] == (8, 12)
    assert shifted[0]["key_ranges"] == [(0, 12)]
    assert shifted[0]["compact_query_offset"] == 8
    assert shifted[0]["full_attn_spans"] == [(8, 12)]
    assert shifted[0]["signature"] == ("complex", 4, 12, 8, ((8, 12),))


def test_shift_piecewise_mask_plan_preserves_prompt_spans_before_prefix():
    plan = _make_piecewise_mask_plan([[(2, 4), (6, 10)]], query_len=4, key_len=10)

    shifted = _shift_piecewise_mask_plan(plan, prefix_len=6, max_prefix_len=8)

    assert shifted[0]["full_attn_spans"] == [(2, 4), (8, 12)]


def test_shift_piecewise_mask_plan_keeps_baseline_without_prefix_padding():
    plan = _make_piecewise_mask_plan([[(6, 10)]], query_len=4, key_len=10)

    shifted = _shift_piecewise_mask_plan(plan, prefix_len=6, max_prefix_len=6)

    assert shifted[0]["mask_kind"] == "baseline"
    assert shifted[0]["query_range"] == (6, 10)
    assert shifted[0]["key_ranges"] == [(0, 10)]
    assert shifted[0]["full_attn_spans"] == [(6, 10)]
