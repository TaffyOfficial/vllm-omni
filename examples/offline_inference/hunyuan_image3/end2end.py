"""
HunyuanImage-3.0-Instruct unified end-to-end inference script.

Supports all modalities through a single entry point:
  - text2img:  Text → AR → DiT → Image
  - img2img:   Text+Image → AR → DiT → Edited Image (IT2I)
  - img2text:  Image+Text → AR → Text description (I2T)
  - text2text: Text → AR → Text (comprehension, no image)

Usage:
    python end2end.py --modality text2img --prompts "A cute cat"
    python end2end.py --modality img2img --image-path input.png --prompts "Make it snowy"
    python end2end.py --modality img2img --image-path img1.png,img2.png --prompts "Combine"
    python end2end.py --modality img2text --image-path input.png --prompts "Describe this image"
"""

import argparse
import os

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    build_prompt_tokens,
    resolve_sys_type,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniPromptType

# Modality → (task, default bot_task) mapping. `task` selects only whether
# `<img>` placeholders are emitted; `bot_task` (None | think | recaption |
# think_recaption | vanilla) selects the system prompt + trigger tag.
#
# Both verbose (`text2img`) and short (`t2i`) forms are accepted; the short
# forms match the internal task names (see prompt_utils._TASK_PRESETS) so
# users who think in those terms don't have to translate.
_MODALITY_TASK_MAP: dict[str, tuple[str, str | None]] = {
    "text2img": ("t2i", "think"),
    "t2i": ("t2i", "think"),
    "img2img": ("it2i", "think"),
    "it2i": ("it2i", "think"),
    "img2text": ("i2t", None),
    "i2t": ("i2t", None),
    "text2text": ("t2t", None),
    "t2t": ("t2t", None),
}


# Modality → default stage config
_MODALITY_DEFAULT_CONFIG = {
    "text2img": "hunyuan_image3_t2i.yaml",
    "t2i": "hunyuan_image3_t2i.yaml",
    "img2img": "hunyuan_image3_it2i.yaml",
    "it2i": "hunyuan_image3_it2i.yaml",
    "img2text": "hunyuan_image3_i2t.yaml",
    "i2t": "hunyuan_image3_i2t.yaml",
    "text2text": "hunyuan_image3_t2t.yaml",
    "t2t": "hunyuan_image3_t2t.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0-Instruct end-to-end inference.")
    parser.add_argument(
        "--model",
        default="tencent/HunyuanImage-3.0-Instruct",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--modality",
        default="text2img",
        choices=["text2img", "t2i", "img2img", "it2i", "img2text", "i2t", "text2text", "t2t"],
        help=(
            "Modality mode to control stage execution. Verbose names "
            "(text2img/img2img/img2text/text2text) and the matching internal "
            "task short names (t2i/it2i/i2t/t2t) are both accepted."
        ),
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Input image path(s) for img2img/img2text. Comma-separated for multi-image (up to 3).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory to save results.",
    )

    # Generation parameters
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.",
    )

    # Prompt configuration
    parser.add_argument(
        "--bot-task",
        type=str,
        default=None,
        choices=["none", "think", "recaption", "think_recaption", "vanilla"],
        help=(
            "Override prompt mode. `none` selects the unified system prompt with no "
            "trigger tag (escape hatch for image modalities whose default is `think`). "
            "Default: auto from --modality (think for image-output, none for text-output)."
        ),
    )
    parser.add_argument(
        "--sys-type",
        type=str,
        default=None,
        help="Override system prompt type (e.g. en_unified, en_vanilla).",
    )

    # Omni init args
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Custom stage config YAML path.")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--init-timeout", type=int, default=300, help="Initialization timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")

    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Canonicalize short-form modality aliases to verbose names so the
    # downstream branches (e.g. `if args.modality == "img2img"`) keep
    # working without listing every alias variant.
    _MODALITY_CANONICAL = {
        "t2i": "text2img",
        "it2i": "img2img",
        "i2t": "img2text",
        "t2t": "text2text",
    }
    args.modality = _MODALITY_CANONICAL.get(args.modality, args.modality)

    # Determine (task, bot_task) for prompt formatting. `--bot-task none` is
    # the explicit way to request bot_task=None on a modality whose default is
    # not None (e.g. text2img defaults to "think"); leaving --bot-task unset
    # falls back to the modality default.
    task, default_bot_task = _MODALITY_TASK_MAP[args.modality]
    if args.bot_task is None:
        bot_task: str | None = default_bot_task
    elif args.bot_task == "none":
        bot_task = None
    else:
        bot_task = args.bot_task

    # Determine stage config
    stage_configs_path = args.stage_configs_path or _MODALITY_DEFAULT_CONFIG[args.modality]

    # Build Omni
    omni_kwargs = {
        "model": args.model,
        "vae_use_tiling": args.vae_use_tiling,
        "stage_configs_path": stage_configs_path,
        "log_stats": args.log_stats,
        "init_timeout": args.init_timeout,
        "enforce_eager": args.enforce_eager,
    }
    if args.modality in ("text2img", "img2img"):
        omni_kwargs["mode"] = "text-to-image"

    omni = Omni(**omni_kwargs)

    # Prepare prompts
    prompts = args.prompts or ["A cute cat"]
    if not prompts:
        print("[Info] No prompts provided, using default.")
        prompts = ["A cute cat"]

    input_images: list = []
    if args.modality in ("img2img", "img2text"):
        if not args.image_path:
            raise ValueError(f"--image-path required for {args.modality}, got: {args.image_path}")
        from PIL import Image

        image_paths = [p.strip() for p in args.image_path.split(",") if p.strip()]
        for p in image_paths:
            if not os.path.exists(p):
                raise ValueError(f"Image path does not exist: {p}")
            input_images.append(Image.open(p).convert("RGB"))
        if not input_images:
            raise ValueError(f"--image-path produced no usable paths: {args.image_path!r}")

    # Load tokenizer for segment-wise prompt tokenization (matches HF
    # apply_chat_template byte-for-byte; see build_prompt_tokens docstring).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    mm_image_payload = (input_images[0] if len(input_images) == 1 else input_images) if input_images else None

    # Format prompts
    formatted_prompts: list[OmniPromptType] = []
    for p in prompts:
        # Only pass `num_images` for modalities that actually consume images;
        # text-only paths (t2i / t2t) ignore the parameter, but threading
        # it unconditionally reads as if t2i needed at least one image.
        build_kwargs: dict = {"task": task, "bot_task": bot_task, "sys_type": args.sys_type}
        if input_images:
            build_kwargs["num_images"] = len(input_images)
        token_ids = build_prompt_tokens(p, tokenizer, **build_kwargs)
        effective_sys_type = args.sys_type or resolve_sys_type(bot_task)

        # `prompt_token_ids` drives the AR stage (matches HF byte-for-byte).
        # `prompt` and `use_system_prompt` are forwarded by ar2diffusion to
        # the DiT stage so the diffusion pipeline can rebuild the same
        # system prefix when constructing its model inputs.
        prompt_dict: dict = {
            "prompt_token_ids": token_ids,
            "prompt": p,
            "use_system_prompt": effective_sys_type,
        }

        if args.modality == "text2img":
            prompt_dict["modalities"] = ["image"]
        elif args.modality == "img2img":
            prompt_dict["modalities"] = ["image"]
            prompt_dict["multi_modal_data"] = {"image": mm_image_payload}
            prompt_dict["height"] = input_images[0].height
            prompt_dict["width"] = input_images[0].width
        elif args.modality == "img2text":
            prompt_dict["modalities"] = ["text"]
            prompt_dict["multi_modal_data"] = {"image": mm_image_payload}
        elif args.modality == "text2text":
            prompt_dict["modalities"] = ["text"]

        formatted_prompts.append(prompt_dict)

    # Build sampling params from defaults
    params_list = list(omni.default_sampling_params_list)

    # Override diffusion params if applicable
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    for i, sp in enumerate(params_list):
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.num_inference_steps = args.steps
            sp.guidance_scale = args.guidance_scale
            sp.guidance_scale_provided = True
            if args.seed is not None:
                sp.seed = args.seed
            if args.modality in ("text2img",):
                sp.height = args.height
                sp.width = args.width

    # Print configuration
    print(f"\n{'=' * 60}")
    print("HunyuanImage-3.0 Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Modality: {args.modality}")
    print(f"  Stage config: {stage_configs_path}")
    print(f"  Num stages: {omni.num_stages}")
    if args.modality in ("text2img", "img2img"):
        print(f"  Inference steps: {args.steps}")
        print(f"  Guidance scale: {args.guidance_scale}")
        print(f"  Seed: {args.seed}")
    if args.modality == "text2img":
        print(f"  Output size: {args.width}x{args.height}")
    if args.image_path:
        print(f"  Input image: {args.image_path}")
    print(f"  Prompts: {prompts}")
    print(f"{'=' * 60}\n")

    # Generate
    omni_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))

    # Process outputs
    img_idx = 0
    for req_output in omni_outputs:
        # Text output (AR stage or text-only)
        ro = getattr(req_output, "request_output", None)
        if ro and getattr(ro, "outputs", None):
            txt = "".join(getattr(o, "text", "") or "" for o in ro.outputs)
            if txt:
                print(f"[Output] Text:\n{txt}")

        # Image output (DiT stage)
        images = getattr(req_output, "images", None)
        if not images and ro and hasattr(ro, "images"):
            images = ro.images

        if images:
            for j, img in enumerate(images):
                save_path = os.path.join(args.output, f"output_{img_idx}_{j}.png")
                img.save(save_path)
                print(f"[Output] Saved image to {save_path}")
            img_idx += 1


if __name__ == "__main__":
    main()
