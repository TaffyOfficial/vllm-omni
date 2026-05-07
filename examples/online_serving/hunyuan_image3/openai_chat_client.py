#!/usr/bin/env python3
"""
HunyuanImage-3.0-Instruct OpenAI-compatible chat client.

Targets the t2i-with-`<think>` path served by `run_server.sh` + `chat_template.jinja`.
For higher-quality runs, pass the official `unified_system_prompt_en` (from
`vllm_omni.diffusion.models.hunyuan_image3.system_prompt`) via --system-prompt-file.

Usage:
    python openai_chat_client.py --prompt "A cute cat sitting on a windowsill" \
        --output cat.png --steps 50 --height 1024 --width 1024

    python openai_chat_client.py --prompt "Make the petals neon pink" \
        --image-url input.png --output edited.png --steps 50
"""

import argparse
import base64
from pathlib import Path

import requests


def generate(
    prompt: str,
    server_url: str = "http://localhost:8091",
    image_url: str | None = None,
    height: int | None = None,
    width: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    guidance_scale: float | None = None,
    negative_prompt: str | None = None,
    system_prompt: str | None = None,
    modality: str = "text2img",
    timeout: int = 600,
) -> bytes | str | None:
    """Generate an image (or text) via /v1/chat/completions.

    Note: vllm-omni's chat endpoint reads generation params from the JSON
    payload top-level — NOT from `extra_body`. The OpenAI Python SDK packs
    unknown kwargs into `extra_body` by default, which would be silently
    dropped. This client posts raw JSON with params at top-level to avoid
    that footgun.
    """
    content: list[dict] = []
    if image_url:
        if Path(image_url).exists():
            with open(image_url, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            url = f"data:image/jpeg;base64,{b64}"
        else:
            url = image_url
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": prompt})

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    payload: dict = {"messages": messages}
    if modality in ("text2img", "img2img"):
        payload["modalities"] = ["image"]
    elif modality in ("img2text", "text2text"):
        payload["modalities"] = ["text"]
    if height is not None:
        payload["height"] = height
    if width is not None:
        payload["width"] = width
    if steps is not None:
        payload["num_inference_steps"] = steps
    if seed is not None:
        payload["seed"] = seed
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    print(f"Sending request to {server_url} (modality={modality})...")
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])

    for choice in choices:
        c = choice.get("message", {}).get("content")
        if isinstance(c, list) and c and isinstance(c[0], dict) and "image_url" in c[0]:
            url = c[0]["image_url"].get("url", "")
            if url.startswith("data:image"):
                return base64.b64decode(url.split(",", 1)[1])
    for choice in choices:
        c = choice.get("message", {}).get("content")
        if isinstance(c, str) and c:
            return c

    print(f"Unexpected response: {data}")
    return None


def main():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0-Instruct chat client")
    parser.add_argument("--prompt", "-p", default="A cute cat sitting on a windowsill")
    parser.add_argument("--output", "-o", default="hyimage3_output.png")
    parser.add_argument("--server", "-s", default="http://localhost:8091")
    parser.add_argument("--image-url", "-i", help="Input image URL or local path")
    parser.add_argument(
        "--modality",
        "-m",
        default="text2img",
        choices=["text2img", "img2img", "img2text", "text2text"],
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative")
    parser.add_argument(
        "--system-prompt-file",
        help="Optional path to a system prompt file (e.g. unified_system_prompt_en.txt). "
        "Without this, image quality is degraded vs the offline end2end.py path.",
    )
    args = parser.parse_args()

    system_prompt = None
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

    result = generate(
        prompt=args.prompt,
        server_url=args.server,
        image_url=args.image_url,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative,
        system_prompt=system_prompt,
        modality=args.modality,
    )
    if result is None:
        print("Generation failed")
        raise SystemExit(1)
    if isinstance(result, bytes):
        Path(args.output).write_bytes(result)
        print(f"Image saved to {args.output} ({len(result) / 1024:.1f} KB)")
    else:
        print("Response:")
        print(result)


if __name__ == "__main__":
    main()
