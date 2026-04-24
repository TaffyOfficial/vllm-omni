from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# Match unprefixed HF id even when MODEL_PREFIX is set (omni_runner resolves full path).
_QWEN_IMAGE_RANDOM_ID = "riverclouds/qwen_image_random"


def _is_qwen_image_random(model_path: str) -> bool:
    return model_path.rstrip("/").endswith(_QWEN_IMAGE_RANDOM_ID)


models = ["Tongyi-MAI/Z-Image-Turbo", "riverclouds/qwen_image_random"]

# Modelscope can't find riverclouds/qwen_image_random
# TODO: When NPU support is ready, remove this branch.
if current_omni_platform.is_npu():
    models = ["Tongyi-MAI/Z-Image-Turbo", "Qwen/Qwen-Image"]

# omni_runner expects (model, stage_configs_path); single-stage diffusion has no YAML.
test_params = [(m, None) for m in models]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 1, "rocm": 1, "xpu": 2})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_diffusion_model(omni_runner, run_level):
    resolved = omni_runner.model_name
    if run_level == "core_model" and not _is_qwen_image_random(resolved):
        pytest.skip()

    if run_level == "advanced_model" and _is_qwen_image_random(resolved):
        pytest.skip()

    # high resolution may cause OOM on L4
    height = 256
    width = 256
    sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=2,
        guidance_scale=0.0,
        generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
        num_outputs_per_prompt=2,
    )

    # OmniRunner.generate() is typed for list[TextPrompt]; diffusion uses Omni.generate(str, ...).
    outputs = omni_runner.omni.generate(
        "a photo of a cat sitting on a laptop keyboard",
        sampling,
    )

    # Extract images from request_output['images']
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images

    assert len(images) == 2
    # check image size
    assert images[0].width == width
    assert images[0].height == height
    images[0].save("image_output.png")


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_hunyuan_image3_instruct_t2i_dummy_forward(monkeypatch):
    from vllm_omni.diffusion.models.hunyuan_image3 import (
        pipeline_hunyuan_image3 as hunyuan_pipe,
    )

    prompt = "a tiny brass robot watering a bonsai tree"
    generator = torch.Generator("cpu").manual_seed(1234)
    image = Image.new("RGB", (96, 64), color=(11, 22, 33))
    captured = {}

    class DummyImageProcessor:
        def build_image_info(self, image_size):
            captured["image_size"] = image_size
            return SimpleNamespace(
                image_height=image_size[0],
                image_width=image_size[1],
                image_token_length=2,
            )

    class DummyTokenizerWrapper:
        boi_token_id = 10
        eos_token_id = 11
        end_recaption_token_id = 12
        end_answer_token_id = 13
        special_token_map = {f"<img_ratio_{i}>": 100 + i for i in range(33)}

        def apply_chat_template(self, **kwargs):
            captured["chat_template"] = kwargs
            gen_image_mask = torch.tensor(
                [
                    [False, False, True, True],
                    [False, False, True, True],
                ],
            )
            output = SimpleNamespace(
                tokens=torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
                gen_image_mask=gen_image_mask,
                gen_timestep_scatter_index=torch.tensor([[1], [1]]),
                cond_vae_image_mask=None,
                cond_vit_image_mask=None,
                cond_timestep_scatter_index=None,
                all_image_slices=[[slice(2, 4)]],
            )
            sections = [
                [
                    {
                        "type": "gen_image",
                        "token_height": 1,
                        "token_width": 2,
                    },
                ],
            ]
            return {"output": output, "sections": sections}

    monkeypatch.setattr(
        hunyuan_pipe.HunyuanImage3Pipeline,
        "device",
        property(lambda self: torch.device("cpu")),
        raising=False,
    )

    pipeline = object.__new__(hunyuan_pipe.HunyuanImage3Pipeline)
    pipeline.stage_durations = {"dummy_generate": 0.0}
    pipeline.image_processor = DummyImageProcessor()
    pipeline._tkwrapper = DummyTokenizerWrapper()
    pipeline.config = SimpleNamespace(
        attention_head_dim=4,
        image_base_size=1024,
        rope_theta=10000,
    )
    pipeline.generation_config = SimpleNamespace(drop_think=False, max_length=4)

    sampling = OmniDiffusionSamplingParams(
        height=64,
        width=96,
        num_inference_steps=3,
        guidance_scale=3.5,
        generator=generator,
        extra_args={"use_system_prompt": "en_vanilla"},
    )
    request = OmniDiffusionRequest(
        prompts=[{"prompt": prompt}],
        sampling_params=sampling,
    )

    def fake_get_system_prompt(name, task, override=None):
        captured["system_prompt_args"] = (name, task, override)
        return "dummy system prompt\n"

    def fake_build_batch_2d_rope(**kwargs):
        captured["rope"] = kwargs
        return torch.ones(1, 1), torch.zeros(1, 1)

    def fake_generate(self, **kwargs):
        captured["generate"] = kwargs
        return [image]

    monkeypatch.setattr(hunyuan_pipe, "get_system_prompt", fake_get_system_prompt)
    monkeypatch.setattr(
        hunyuan_pipe,
        "build_batch_2d_rope",
        fake_build_batch_2d_rope,
    )
    monkeypatch.setattr(
        hunyuan_pipe.HunyuanImage3Pipeline,
        "_generate",
        fake_generate,
    )

    output = hunyuan_pipe.HunyuanImage3Pipeline.forward(pipeline, request)

    assert output.output is image
    assert output.stage_durations == {"dummy_generate": 0.0}
    assert captured["system_prompt_args"] == ("en_vanilla", "image", None)
    assert captured["image_size"] == (64, 96)
    assert captured["chat_template"] == {
        "batch_prompt": [prompt],
        "batch_message_list": None,
        "mode": "gen_image",
        "batch_gen_image_info": [captured["chat_template"]["batch_gen_image_info"][0]],
        "batch_cond_image_info": None,
        "batch_system_prompt": ["dummy system prompt"],
        "batch_cot_text": None,
        "max_length": None,
        "bot_task": "auto",
        "image_base_size": 1024,
        "sequence_template": "pretrain",
        "cfg_factor": 2,
        "drop_think": False,
    }
    assert captured["chat_template"]["batch_gen_image_info"][0].image_height == 64
    assert captured["chat_template"]["batch_gen_image_info"][0].image_width == 96
    assert captured["rope"]["image_infos"] == [[(slice(2, 4), (1, 2))]]
    assert captured["rope"]["seq_len"] == 4
    assert captured["rope"]["n_elem"] == 4
    assert captured["rope"]["device"] == torch.device("cpu")
    assert captured["generate"]["input_ids"].tolist() == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert captured["generate"]["position_ids"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert captured["generate"]["custom_pos_emb"][0].tolist() == [[1.0]]
    assert captured["generate"]["custom_pos_emb"][1].tolist() == [[0.0]]
    assert captured["generate"]["mode"] == "gen_image"
    assert captured["generate"]["num_inference_steps"] == 3
    assert captured["generate"]["guidance_scale"] == 3.5
    assert captured["generate"]["image_mask"].tolist() == [
        [False, False, True, True],
        [False, False, True, True],
    ]
    assert captured["generate"]["gen_timestep_scatter_index"].tolist() == [[1], [1]]
    assert captured["generate"]["generator"] is generator
    assert captured["generate"]["eos_token_id"] == [10]
    assert captured["generate"]["max_new_tokens"] is None
    assert captured["generate"]["batch_gen_image_info"] == captured["chat_template"]["batch_gen_image_info"]
    assert captured["generate"]["tokenizer_output"].tokens.tolist() == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert captured["generate"]["cond_vae_images"] is None
    assert captured["generate"]["cond_timestep"] is None
    assert captured["generate"]["cond_vae_image_mask"] is None
    assert captured["generate"]["cond_vit_images"] is None
    assert captured["generate"]["cond_vit_image_mask"] is None
    assert captured["generate"]["vit_kwargs"] is None
    assert captured["generate"]["cond_timestep_scatter_index"] is None
    assert captured["generate"]["past_key_values"] is None
