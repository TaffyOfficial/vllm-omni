# Image-To-Video

This example demonstrates how to generate videos from images using Wan2.2,
LTX-2/LTX-2.3, and HunyuanVideo-1.5 Image-to-Video models with vLLM-Omni's
offline inference API.

## Local CLI Usage

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

### Wan2.2-I2V-A14B-Diffusers (MoE)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 5.0 \
  --guidance-scale-high 6.0 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Wan2.2-TI2V-5B-Diffusers (Unified)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 4.0 \
  --num-inference-steps 40 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### LTX-2.3

```bash
python image_to_video.py \
  --model dg845/LTX-2.3-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze with synchronized ambient sound" \
  --negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 384 \
  --width 512 \
  --num-frames 25 \
  --guidance-scale 4.0 \
  --num-inference-steps 20 \
  --frame-rate 24 \
  --fps 24 \
  --output ltx23_i2v_output.mp4
```

LTX-2.3 Diffusers-format checkpoints such as `dg845/LTX-2.3-Diffusers` are
routed to `LTX23ImageToVideoPipeline` automatically. The upstream
`Lightricks/LTX-2.3` repository contains raw safetensors and is not directly
loadable by this Diffusers-subfolder pipeline. The script uses the audio sample
rate returned by the pipeline when muxing synchronized audio.

Key arguments:

- `--model`: Model ID (I2V-A14B for MoE, TI2V-5B for unified T2V+I2V, or
  LTX-2/LTX-2.3).
- `--image`: Path to input image (required).
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set).
  Wan dimensions should be multiples of 16; LTX dimensions should be multiples
  of 32.
- `--num-frames`: Number of frames (model-specific default; LTX-style models
  work best with `8k + 1`).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative-prompt`: Optional list of artifacts to suppress.
- `--boundary-ratio`: Boundary split ratio for two-stage MoE models.
- `--flow-shift`: Scheduler flow shift (5.0 for 720p, 12.0 for 480p).
- `--sample-solver`: Wan2.2 sampling solver. Use `unipc` for the default multistep solver, or `euler` for Lightning/Distill checkpoints.
- `--num-inference-steps`: Number of denoising steps (default 50).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--audio-sample-rate`: fallback audio sample rate for embedded audio.
  Pipelines such as LTX-2.3 may return their own sample rate, which takes
  precedence.
- `--output`: Path to save the generated video.
- `--vae-use-slicing`: Enable VAE slicing for memory optimization.
- `--vae-use-tiling`: Enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism/cfg_parallel.md).
- `--tensor-parallel-size`: tensor parallel size (effective for models that support TP, e.g. LTX2).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--use-hsdp`: Enable Hybrid Sharded Data Parallel to shard model weights across GPUs.
- `--hsdp-shard-size`: Number of GPUs to shard model weights across within each replica group. -1 (default) auto-calculates as world_size / replicate_size.
- `--hsdp-replicate-size`: Number of replica groups for HSDP. Each replica holds a full sharded copy. Default 1 means pure sharding (no replication).



> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

For Wan2.2 LightX2V-converted local Diffusers directories and related LoRA
assets, see the [LoRA guide](../../../docs/user_guide/diffusion/lora.md#wan22-lightx2v-offline-assembly).
