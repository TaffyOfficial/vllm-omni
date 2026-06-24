# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synthetic AR KV reuse pressure benchmark for HunyuanImage3 DiT serving.

This is a preset wrapper around diffusion_benchmark_serving.py. It keeps the
same CLI surface but defaults to the DiT KV reuse path: chat completions,
random t2i requests, synthetic AR KV payloads, and stage metrics.
"""

from __future__ import annotations

import asyncio

if __package__:
    from benchmarks.diffusion.diffusion_benchmark_serving import (
        benchmark,
    )
    from benchmarks.diffusion.diffusion_benchmark_serving import (
        create_arg_parser as create_diffusion_arg_parser,
    )
else:
    from diffusion_benchmark_serving import benchmark
    from diffusion_benchmark_serving import create_arg_parser as create_diffusion_arg_parser


def create_arg_parser():
    parser = create_diffusion_arg_parser(
        description="Benchmark HunyuanImage3 DiT serving with synthetic AR KV reuse payloads."
    )
    parser.set_defaults(
        endpoint="/v1/chat/completions",
        dataset="random",
        task="t2i",
        synthetic_ar_kv=True,
        return_stage_metrics=True,
        synthetic_ar_kv_layers=32,
        synthetic_ar_kv_seq_len=128,
        synthetic_ar_kv_num_heads=2,
        synthetic_ar_kv_head_dim=128,
        synthetic_ar_kv_dtype="bfloat16",
        synthetic_ar_kv_from_stage="-1",
        synthetic_ar_kv_to_stage="0",
        synthetic_ar_kv_from_tp=4,
        synthetic_ar_kv_to_tp=4,
        synthetic_ar_kv_request_id_prefix="chatcmpl-",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = create_arg_parser()
    args = parser.parse_args(argv)
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
