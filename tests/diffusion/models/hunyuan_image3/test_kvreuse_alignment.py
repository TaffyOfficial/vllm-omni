# SPDX-License-Identifier: Apache-2.0
"""Regression tests for AR-token-IDs preservation through DiT prompt building.

Pins the KV-reuse alignment contract: when the AR-side stage input
processor (`ar2diffusion`) forwards `ar_token_ids` to the diffusion
stage, `apply_chat_template` must consume those IDs verbatim (no
re-encode of the decoded cot text via `tokenizer.encode`) so that the
DiT-side prompt tokenization matches AR's actually-sampled token
sequence byte-for-byte.

Why this matters: tokenize-detokenize-tokenize over the cot text is not
lossless (BPE re-merges on multi-byte UTF-8 / punctuation boundaries),
and the resulting length drift breaks AR KV position alignment --
DiT's `positive_reuse_len` (computed from `tokenizer.encode(cot_text)`)
ends up larger than the actual cached AR KV length, and
`inject_ar_kv_into_layers` then silently truncates via Python slice,
leaving `_cache_prompt_kv`'s `q_len + ar_kv_len == seq_len` assert off
by N (hard 500 on KV-reuse-enabled requests; see
`pipeline_hunyuan_image3.py:_cache_prompt_kv`).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.core_model]


def _hf_cached(model_id: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots")
    return os.path.isdir(snap_dir) and any(os.scandir(snap_dir))


_HUNYUAN_MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"


@pytest.mark.skipif(
    not _hf_cached(_HUNYUAN_MODEL_ID),
    reason=f"{_HUNYUAN_MODEL_ID} tokenizer not in HF cache",
)
def test_get_cot_sections_from_token_ids_round_trips_ar_ids():
    """`get_cot_sections_from_token_ids` must split AR-sampled IDs at the
    `<think>` / `</think>` token-id positions and emit sections whose
    concatenated tokens equal the input (no re-encode).

    Catches the failure mode where DiT re-encodes the decoded cot text
    and the BPE merges differ from AR's sampled tokens (length drift).
    """
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import (
        TokenizerWrapper,
    )

    tkw = TokenizerWrapper(_HUNYUAN_MODEL_ID)

    think_id = tkw.tokenizer.convert_tokens_to_ids("<think>")
    end_think_id = tkw.end_think_token_id

    # Fabricate an AR-style id sequence: arbitrary "thought" payload tokens
    # surrounded by <think>/</think> markers, plus some leading + trailing
    # tokens (e.g. <answer>/<boi> tail that gets truncated upstream).
    thought_payload = [1000, 1001, 1002, 1003, 1004]
    leading = [2000, 2001]
    trailing = [3000]
    ar_token_ids = leading + [think_id] + thought_payload + [end_think_id] + trailing

    sections = tkw.get_cot_sections_from_token_ids(
        ar_token_ids,
        uncond_kwargs={},
        drop_think=False,
    )

    # Sections concatenated must equal the input verbatim.
    out: list[int] = []
    for sec in sections:
        assert sec["type"] == "text", f"unexpected section type: {sec}"
        toks = sec.get("tokens")
        assert toks is not None, f"section missing 'tokens' field: {sec}"
        out.extend(toks)
    assert out == ar_token_ids, (
        f"split-by-token-id must be lossless; got {len(out)} ids vs {len(ar_token_ids)} input; "
        f"diff at first mismatch index = {next((i for i, (a, b) in enumerate(zip(out, ar_token_ids)) if a != b), None)}"
    )


@pytest.mark.skipif(
    not _hf_cached(_HUNYUAN_MODEL_ID),
    reason=f"{_HUNYUAN_MODEL_ID} tokenizer not in HF cache",
)
def test_apply_chat_template_batch_cot_token_ids_preserves_ar_ids():
    """When `batch_cot_token_ids` is passed, the assistant section in the
    final encoded token sequence must contain the AR-sampled token ids
    verbatim -- no `tokenizer.encode(cot_text)` round-trip.

    Pins the end-to-end contract that KV-reuse alignment relies on.
    """
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import (
        TokenizerWrapper,
    )

    tkw = TokenizerWrapper(_HUNYUAN_MODEL_ID)
    think_id = tkw.tokenizer.convert_tokens_to_ids("<think>")
    end_think_id = tkw.end_think_token_id

    # Construct a synthetic AR cot id sequence. Use mid-range vocab ids
    # that are very unlikely to collide with any chat-template specials.
    payload = [55001, 55002, 55003]
    ar_token_ids = [think_id] + payload + [end_think_id]

    out_with_ids = tkw.apply_chat_template(
        batch_prompt=["draw a robot"],
        batch_system_prompt=[None],
        batch_cot_token_ids=[ar_token_ids],
        mode="gen_text",
        sequence_template="instruct",
    )
    tokens_with_ids = out_with_ids["output"].tokens.tolist()[0]  # batched output: take batch 0

    # The exact AR payload must appear as a contiguous subsequence in the
    # encoded output, sandwiched by the think markers we forwarded.
    def _find_subseq(haystack: list[int], needle: list[int]) -> int:
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                return i
        return -1

    full_cot = [think_id] + payload + [end_think_id]
    idx = _find_subseq(tokens_with_ids, full_cot)
    assert idx >= 0, (
        f"AR cot ids {full_cot} not found as contiguous subseq in encoded output; "
        f"means apply_chat_template did NOT respect batch_cot_token_ids and re-encoded cot text instead"
    )
