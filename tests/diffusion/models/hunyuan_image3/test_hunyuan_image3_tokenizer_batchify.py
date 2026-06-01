# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

from vllm.model_executor.models.utils import PPMissingLayer

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import (
    TokenizerWrapper,
)
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    ImageKVCacheManager,
)
from vllm_omni.diffusion.models.hunyuan_image3.mixfusion import MixFusionImageLayout
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)


def test_apply_general_template_batchify_preserves_all_rows(monkeypatch):
    wrapper = object.__new__(TokenizerWrapper)
    messages = [
        [{"role": "user", "type": "text", "content": "first", "context_type": "str"}],
        [{"role": "user", "type": "text", "content": "second", "context_type": "str"}],
        [{"role": "user", "type": "text", "content": "third", "context_type": "str"}],
    ]
    captured = {}

    def fake_batch_gen_infer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(wrapper, "batch_gen_infer", fake_batch_gen_infer)

    wrapper.apply_general_template(message_list=messages, batchify=True)

    assert captured["prompt_list"] == [[], [], []]
    assert [row["message_list"] for row in captured["infer_fn_kwargs_list"]] == messages


def test_mixfusion_repeats_scalar_system_prompt_for_all_chunks():
    layouts = (
        MixFusionImageLayout(index=0, token_height=64, token_width=64, seq_len=4096, chunk_start=0, chunk_count=2),
        MixFusionImageLayout(index=1, token_height=52, token_width=76, seq_len=3952, chunk_start=2, chunk_count=3),
    )

    assert HunyuanImage3Pipeline._repeat_optional_by_mixfusion_layout(["a", "b"], layouts) == [
        "a",
        "a",
        "b",
        "b",
        "b",
    ]
    assert HunyuanImage3Pipeline._repeat_optional_by_mixfusion_layout("sys", layouts) == [
        "sys",
        "sys",
        "sys",
        "sys",
        "sys",
    ]


class _FakeBackend:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


def test_mixfusion_skips_attention_mask_buckets_for_flash_attention():
    manager = object.__new__(ImageKVCacheManager)
    manager.attn = SimpleNamespace(attn_backend=_FakeBackend("FLASH_ATTN"))

    assert not manager._mixfusion_requires_attention_mask()


def test_mixfusion_requires_attention_mask_buckets_for_sdpa_pipeline():
    manager = object.__new__(ImageKVCacheManager)
    manager.attn = SimpleNamespace(attn_backend=_FakeBackend("SDPA"))
    layer = SimpleNamespace(self_attn=SimpleNamespace(image_attn=manager))
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline.model = SimpleNamespace(layers=[PPMissingLayer(), layer])

    assert pipeline._mixfusion_requires_attention_mask_buckets()


def test_mixfusion_can_skip_attention_mask_values_only_without_mask_buckets():
    assert HunyuanImage3Pipeline._mixfusion_can_skip_attention_mask_values({"mixfusion_sequence_plan": object()})
    assert not HunyuanImage3Pipeline._mixfusion_can_skip_attention_mask_values(
        {"mixfusion_sequence_plan": object(), "mixfusion_attention_masks": {"first": {}}}
    )
    assert not HunyuanImage3Pipeline._mixfusion_can_skip_attention_mask_values({})
