#!/usr/bin/env python3
"""
BF16/FP16 correctness and speed test for the ARM llamafile path.

Strategy: build synthetic expert weights, feed the SAME data through the
llamafile MoE as F32, BF16, and FP16, then compare each low-precision output
against the F32 baseline and report per-call latency.

Usage:
  python examples/test_llamafile_bf16.py

Optional env:
  E=32        experts
  H=2048      hidden size
  INTER=256   moe intermediate size (must be % 256 == 0)
  BATCH=4     tokens per call
  TOPK=8
  N_ITERS=200
  CPU_THREADS=8
  SEED=1
  BF16_TOL=0.02
  FP16_TOL=0.03
"""

from __future__ import annotations

import os
import sys
import time

import torch

import kt_kernel.utils.llamafile as llamafile_mod
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
from kt_kernel_ext.kvcache import ggml_type
from kt_kernel_ext.moe import MOE, MOEConfig


class _DummyGGUFLoader:
    """Weights are built synthetically; gguf loading is bypassed."""


llamafile_mod.GGUFLoader = lambda path: _DummyGGUFLoader()


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def build_wrapper(weights_f32, run_type, hidden, inter, experts, topk, cpu_threads):
    torch_dtype = {
        ggml_type.FP32: torch.float32,
        ggml_type.BF16: torch.bfloat16,
        ggml_type.FP16: torch.float16,
    }[run_type]
    weights = tuple(t.to(torch_dtype).contiguous() for t in weights_f32)
    gate, up, down = weights
    keep = (gate, up, down)

    wrapper = LlamafileMoEWrapper(
        layer_idx=0,
        num_experts=experts,
        num_experts_per_tok=topk,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        gpu_experts_mask=None,
        cpuinfer_threads=cpu_threads,
        threadpool_count=1,
        weight_path=__file__,  # only exists-check; we build MOE manually below
        chunked_prefill_size=512,
        method="LLAMAFILE",
    )

    cfg = MOEConfig(experts, topk, hidden, inter, wrapper.gpu_experts_mask.data_ptr())
    cfg.layer_idx = 0
    cfg.pool = wrapper.cpu_infer.backend_
    cfg.m_block = 32
    cfg.group_min_len = 10
    cfg.max_len = 512
    cfg.group_max_len = 512
    cfg.gate_proj = gate.data_ptr()
    cfg.up_proj = up.data_ptr()
    cfg.down_proj = down.data_ptr()
    cfg.gate_type = run_type
    cfg.up_type = run_type
    cfg.down_type = run_type
    cfg.hidden_type = ggml_type.BF16

    wrapper.moe = MOE(cfg)
    id_map = torch.arange(experts, dtype=torch.int32)
    wrapper.cpu_infer.submit(wrapper.moe.load_weights_task(id_map.data_ptr()))
    wrapper.cpu_infer.sync()
    return wrapper, keep, id_map


def run(wrapper, hidden_states, topk_ids, topk_weights, stream, n_iters, device):
    out = wrapper.forward(hidden_states, topk_ids, topk_weights, stream)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        wrapper.submit_forward(hidden_states, topk_ids, topk_weights, stream)
        out = wrapper.sync_forward(hidden_states, stream)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) / n_iters * 1000
    return out.float().cpu(), ms


def main() -> int:
    experts = getenv_int("E", 32)
    hidden = getenv_int("H", 2048)
    inter = getenv_int("INTER", 256)
    batch = getenv_int("BATCH", 4)
    topk = getenv_int("TOPK", 8)
    n_iters = getenv_int("N_ITERS", 200)
    cpu_threads = getenv_int("CPU_THREADS", 8)
    seed = getenv_int("SEED", 1)
    tolerances = {
        "BF16": getenv_float("BF16_TOL", 0.02),
        "FP16": getenv_float("FP16_TOL", 0.03),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if inter % 256 != 0:
        print("INTER must be divisible by 256")
        return 2
    if hidden % 4 != 0 or inter % 4 != 0:
        print("H and INTER must both be divisible by 4")
        return 2

    print(
        f"device={device}, experts={experts}, hidden={hidden}, inter={inter}, "
        f"batch={batch}, topk={topk}, threads={cpu_threads}, iterations={n_iters}"
    )

    torch.manual_seed(seed)
    shape = (experts, inter, hidden)
    gate = torch.randn(shape, dtype=torch.float32) * 0.02
    up = torch.randn(shape, dtype=torch.float32) * 0.02
    down = torch.randn(experts, hidden, inter, dtype=torch.float32) * 0.02
    weights_f32 = (gate.contiguous(), up.contiguous(), down.contiguous())

    stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else 0
    hidden_states = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, experts, (batch, topk), dtype=torch.long, device=device)
    topk_weights = torch.rand(batch, topk, dtype=torch.float32, device=device)
    topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

    results = {}
    run_types = (
        ("F32", ggml_type.FP32),
        ("BF16", ggml_type.BF16),
        ("FP16", ggml_type.FP16),
    )
    for name, run_type in run_types:
        wrapper, keep, id_map = build_wrapper(
            weights_f32, run_type, hidden, inter, experts, topk, cpu_threads
        )
        out, ms = run(wrapper, hidden_states, topk_ids, topk_weights, stream, n_iters, device)
        if not torch.isfinite(out).all():
            print(f"FAIL: {name} output contains NaN or Inf")
            return 1
        results[name] = (out, ms)
        print(f"{name}: {ms:.3f} ms/call, out[0,:4]={out[0,:4].tolist()}")
        del wrapper, keep, id_map, out

    out_f32, ms_f32 = results["F32"]
    denom = out_f32.abs().max().item()
    failed = False
    for name in ("BF16", "FP16"):
        out, ms = results[name]
        max_abs_diff = (out - out_f32).abs().max().item()
        rel = (max_abs_diff / denom) if denom > 0 else 0.0
        threshold = tolerances[name]
        print(
            f"{name} vs F32: max_abs_diff={max_abs_diff:.6f}, "
            f"scale={denom:.6f}, rel={rel:.4%}, speedup={ms_f32 / ms:.2f}x"
        )
        if rel > threshold:
            print(f"FAIL: {name} relative error {rel:.4%} > {threshold:.2%}")
            failed = True

    if failed:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
