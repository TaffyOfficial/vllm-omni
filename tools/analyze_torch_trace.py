#!/usr/bin/env python3
"""Analyze torch profiler traces to extract top-N kernels and confirm fp8 GEMM path.

Usage:
    python3 tools/analyze_torch_trace.py <trace_dir> [top_n]
"""

import json
import sys
from pathlib import Path


def load_trace(trace_dir: str) -> dict:
    trace_path = Path(trace_dir)
    json_files = sorted(trace_path.glob("*.json")) + sorted(trace_path.glob("*.json.gz"))
    if not json_files:
        print(f"No trace files found in {trace_dir}")
        sys.exit(1)

    trace_file = json_files[0]
    print(f"Loading trace: {trace_file}")

    if str(trace_file).endswith(".gz"):
        import gzip

        with gzip.open(trace_file, "rt") as f:
            return json.load(f)
    else:
        with open(trace_file) as f:
            return json.load(f)


def extract_cuda_kernels(trace: dict, top_n: int = 30):
    """Extract CUDA kernel events sorted by duration."""
    events = trace.get("traceEvents", [])
    kernels = []
    for ev in events:
        cat = ev.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "cuda_runtime"):
            name = ev.get("name", "")
            dur = ev.get("dur", 0)  # microseconds
            kernels.append({"name": name, "dur_us": dur})

    kernels.sort(key=lambda x: -x["dur_us"])
    return kernels[:top_n]


def check_fp8_gemm(kernels: list[dict]) -> list[dict]:
    """Find kernels that indicate fp8 GEMM path is active."""
    fp8_indicators = [
        "fp8",
        "e4m3",
        "e5m2",
        "f8",
        "cutlass_fp8",
        "sm90_xmma",
        "cublas_fp8",
        "cublasLt",
    ]
    fp8_kernels = []
    for k in kernels:
        name_lower = k["name"].lower()
        if any(ind in name_lower for ind in fp8_indicators):
            fp8_kernels.append(k)
    return fp8_kernels


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_torch_trace.py <trace_dir> [top_n]")
        sys.exit(1)

    trace_dir = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    trace = load_trace(trace_dir)

    print(f"\n{'=' * 70}")
    print(f"  Top-{top_n} CUDA Kernels by Duration")
    print(f"{'=' * 70}")

    kernels = extract_cuda_kernels(trace, top_n=top_n)
    if not kernels:
        print("  No CUDA kernel events found in trace.")
        print("  Check that profiler_config was set correctly.")
        return

    for i, k in enumerate(kernels, 1):
        dur_ms = k["dur_us"] / 1000
        print(f"  {i:3d}. {dur_ms:10.3f} ms  {k['name'][:100]}")

    print(f"\n{'=' * 70}")
    print("  FP8 GEMM Path Analysis")
    print(f"{'=' * 70}")

    # Check all kernels (not just top-N) for fp8
    all_kernels = extract_cuda_kernels(trace, top_n=10000)
    fp8_kernels = check_fp8_gemm(all_kernels)

    if fp8_kernels:
        total_fp8_us = sum(k["dur_us"] for k in fp8_kernels)
        total_all_us = sum(k["dur_us"] for k in all_kernels) or 1
        print(f"  FP8 kernels found: {len(fp8_kernels)}")
        print(f"  FP8 total time: {total_fp8_us / 1000:.3f} ms ({100 * total_fp8_us / total_all_us:.1f}% of GPU time)")
        print("\n  Top FP8 kernels:")
        for i, k in enumerate(sorted(fp8_kernels, key=lambda x: -x["dur_us"])[:15], 1):
            print(f"    {i:3d}. {k['dur_us'] / 1000:10.3f} ms  {k['name'][:100]}")
        print("\n  Conclusion: FP8 GEMM path is ACTIVE")
    else:
        # Check for regular GEMM kernels
        gemm_keywords = ["gemm", "cublas", "cutlass", "xmma", "matmul"]
        gemm_kernels = [k for k in all_kernels if any(g in k["name"].lower() for g in gemm_keywords)]
        if gemm_kernels:
            print(f"  WARNING: No FP8-specific kernels found, but {len(gemm_kernels)} GEMM kernels detected.")
            print("  These may be running in FP16/BF16 instead of FP8:")
            for i, k in enumerate(sorted(gemm_kernels, key=lambda x: -x["dur_us"])[:10], 1):
                print(f"    {i:3d}. {k['dur_us'] / 1000:10.3f} ms  {k['name'][:100]}")
            print("\n  Conclusion: FP8 GEMM path may NOT be active. Check quantization config.")
        else:
            print("  No GEMM kernels found at all. Trace may be incomplete.")

    # Summary for cross-config comparison
    print(f"\n{'=' * 70}")
    print("  Kernel Category Breakdown")
    print(f"{'=' * 70}")

    categories = {
        "GEMM/MatMul": ["gemm", "cublas", "cutlass", "xmma", "matmul"],
        "Attention": ["attention", "flash", "fmha", "sdpa"],
        "Elementwise": ["elementwise", "vectorized", "unrolled"],
        "Reduction": ["reduce", "softmax", "layernorm", "rmsnorm"],
        "Memory": ["memcpy", "memset"],
    }

    for cat_name, keywords in categories.items():
        cat_kernels = [k for k in all_kernels if any(kw in k["name"].lower() for kw in keywords)]
        if cat_kernels:
            total_ms = sum(k["dur_us"] for k in cat_kernels) / 1000
            pct = 100 * sum(k["dur_us"] for k in cat_kernels) / (sum(k["dur_us"] for k in all_kernels) or 1)
            print(f"  {cat_name:20s}: {total_ms:10.1f} ms ({pct:5.1f}%)  [{len(cat_kernels)} kernels]")


if __name__ == "__main__":
    main()
