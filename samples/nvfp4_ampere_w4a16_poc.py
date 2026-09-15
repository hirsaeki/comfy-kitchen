# SPDX-License-Identifier: Apache-2.0
"""Standalone NVFP4 W4A16 experiment for Ampere/SM8x.

This builds ``nvfp4_ampere_w4a16_poc.cu`` as a local PyTorch extension and
compares it with comfy-kitchen's existing NVFP4 dequantization + BF16 linear.
It does not modify comfy-kitchen dispatch; the point is to answer one question
first: can packed NVFP4 weights stay resident at 4-bit while Ampere tensor
cores consume BF16 fragments fast enough to be useful?

Example (RTX 3090):

    python samples/nvfp4_ampere_w4a16_poc.py --shape 1024,3072,3072

Repeat ``--shape M,N,K`` to test several DiT-like linear shapes.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import comfy_kitchen as ck


HERE = Path(__file__).resolve().parent
CUDA_SOURCE = HERE / "nvfp4_ampere_w4a16_poc.cu"


def _parse_shape(text: str) -> tuple[int, int, int]:
    try:
        m, n, k = (int(x.strip()) for x in text.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError("shape must be M,N,K") from exc
    if min(m, n, k) <= 0:
        raise argparse.ArgumentTypeError("M, N and K must be positive")
    if k % 64:
        raise argparse.ArgumentTypeError("K must be divisible by 64 for this POC")
    if n % 16:
        raise argparse.ArgumentTypeError("N must be divisible by 16 for this POC")
    return m, n, k


def _build(verbose: bool):
    return load(
        name="comfy_kitchen_nvfp4_ampere_w4a16_poc",
        sources=[str(CUDA_SOURCE)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
        with_cuda=True,
    )


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_delta_bytes(fn) -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    y = fn()
    # Keep the result alive through the measurement.
    _ = y.shape
    torch.cuda.synchronize()
    return max(0, torch.cuda.max_memory_allocated() - before)


def _fmt_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _quantize_weight(weight: torch.Tensor):
    # Same global scale used by TensorCoreNVFP4Layout.
    f8_e4m3_max = 448.0
    f4_e2m1_max = 6.0
    global_scale = (weight.abs().amax() / (f8_e4m3_max * f4_e2m1_max)).float()
    qweight, block_scales = ck.quantize_nvfp4(
        weight, global_scale, pad_16x=True, hi_first=True
    )
    return qweight, block_scales, global_scale


def _run_shape(ext, shape: tuple[int, int, int], warmup: int, iters: int):
    m, n, k = shape
    device = torch.device("cuda")

    # Keep magnitudes tame so error metrics are easy to interpret while still
    # exercising normal NVFP4 dynamic range.
    x = (torch.randn(m, k, device=device, dtype=torch.bfloat16) / math.sqrt(k)).contiguous()
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16).contiguous()

    qweight, block_scales, global_scale = _quantize_weight(weight)
    del weight
    torch.cuda.empty_cache()

    # Reference dequant is intentionally performed once here. It gives the
    # kernel a fair numerical target and supplies a best-case resident-BF16
    # baseline separate from the current dequant+linear fallback cost.
    w_deq_full = ck.dequantize_nvfp4(
        qweight, global_scale, block_scales, torch.bfloat16, hi_first=True
    )
    w_deq = w_deq_full[:n, :k].contiguous()

    def poc():
        return ext.nvfp4_w4a16(x, qweight, block_scales, global_scale, n)

    def bf16_resident():
        return F.linear(x, w_deq)

    def current_fallback_like():
        w = ck.dequantize_nvfp4(
            qweight, global_scale, block_scales, torch.bfloat16, hi_first=True
        )[:n, :k]
        return F.linear(x, w)

    y_ref = bf16_resident()
    y_poc = poc()
    torch.cuda.synchronize()

    diff = (y_poc.float() - y_ref.float()).abs()
    denom = y_ref.float().abs().clamp_min(1e-5)
    rel = diff / denom
    rmse = torch.sqrt(torch.mean((y_poc.float() - y_ref.float()) ** 2))
    cosine = F.cosine_similarity(
        y_poc.float().reshape(1, -1), y_ref.float().reshape(1, -1)
    ).item()

    poc_ms = _time_ms(poc, warmup, iters)
    resident_ms = _time_ms(bf16_resident, warmup, iters)
    fallback_ms = _time_ms(current_fallback_like, warmup, iters)

    poc_peak = _peak_delta_bytes(poc)
    fallback_peak = _peak_delta_bytes(current_fallback_like)

    packed_bytes = qweight.numel() * qweight.element_size()
    scales_bytes = block_scales.numel() * block_scales.element_size() + global_scale.numel() * 4
    bf16_bytes = n * k * 2

    print(f"\nshape M={m} N={n} K={k}")
    print(f"  packed NVFP4 weight : {_fmt_bytes(packed_bytes + scales_bytes)}")
    print(f"  BF16 weight         : {_fmt_bytes(bf16_bytes)}")
    print(f"  max abs error       : {diff.max().item():.6g}")
    print(f"  mean abs error      : {diff.mean().item():.6g}")
    print(f"  RMSE                : {rmse.item():.6g}")
    print(f"  mean relative error : {rel.mean().item():.6g}")
    print(f"  cosine              : {cosine:.9f}")
    print(f"  POC W4A16           : {poc_ms:.3f} ms")
    print(f"  resident BF16       : {resident_ms:.3f} ms  ({resident_ms / poc_ms:.3f}x vs POC)")
    print(f"  dequant + BF16      : {fallback_ms:.3f} ms  ({fallback_ms / poc_ms:.3f}x vs POC)")
    print(f"  POC peak delta      : {_fmt_bytes(poc_peak)}")
    print(f"  fallback peak delta : {_fmt_bytes(fallback_peak)}")

    return {
        "shape": shape,
        "poc_ms": poc_ms,
        "resident_ms": resident_ms,
        "fallback_ms": fallback_ms,
        "cosine": cosine,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        action="append",
        default=None,
        help="M,N,K; may be repeated (K %% 64 == 0, N %% 16 == 0)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--build-verbose", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise SystemExit(f"SM80+ is required, found sm{major}{minor}")
    if torch.version.cuda is None:
        raise SystemExit("PyTorch CUDA build is required")

    print(f"GPU        : {torch.cuda.get_device_name()}")
    print(f"capability : sm{major}{minor}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA       : {torch.version.cuda}")

    ext = _build(args.build_verbose)
    shapes = args.shape or [(1024, 3072, 3072)]
    for shape in shapes:
        _run_shape(ext, shape, args.warmup, args.iters)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
