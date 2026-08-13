#!/usr/bin/env python
# coding=utf-8
"""Performance benchmark for ARM NEON BF16 MoE (Ampere One class CPUs).

Answers the question: is the CPU-side MoE bottleneck memory bandwidth or
compute? It measures:

  1. Raw memory read bandwidth ceiling (large-buffer streaming read).
  2. Decode path (qlen=1): tokens/s and effective GB/s vs. the ceiling, swept
     over thread counts to show where bandwidth saturates.
  3. Prefill path (qlen sweep): GFLOP/s, showing the compute-bound crossover.

Usage (on the ARM host, after building kt-kernel):
    python kt-kernel/test/per_commit/bench_moe_neon_perf.py \
        --threads 1 8 32 64 96 128 160 192
"""

import argparse
import platform
import sys
import time

_HOST_ARCH = platform.machine().lower()
if _HOST_ARCH not in ("aarch64", "arm64"):
    print("This benchmark requires an aarch64 host (NEON BF16 kernel).")
    sys.exit(1)

import torch

from kt_kernel import kt_kernel_ext

NEON_MOE_CLS = getattr(kt_kernel_ext.moe, "NEONBF16_MOE", None)
if NEON_MOE_CLS is None:
    raise RuntimeError("NEONBF16_MOE missing; build kt-kernel on this ARM host first.")


def bench_bandwidth(size_gb=4, iters=5):
    """Streaming read bandwidth ceiling using torch sum over a big buffer."""
    numel = int(size_gb * (1 << 30) // 2)  # bf16 elements
    buf = torch.ones(numel, dtype=torch.bfloat16)
    _ = buf.sum()  # touch memory
    best = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = buf.sum()
        dt = time.perf_counter() - t0
        best = max(best, size_gb / dt)
    del buf
    return best


def fast_weights(shape):
    """Allocate bf16 weights without randn: bf16 randn on aarch64 is
    unusably slow (144s per 7.5GB tensor, single-threaded). Values do not
    matter for timing, but write some non-zero data so the pages are real
    and DZ optimization tricks cannot kick in."""
    w = torch.empty(shape, dtype=torch.bfloat16)
    flat = w.view(-1)
    flat[::4097].fill_(0.01)  # touch ~every 2nd page worth of elements, sparse
    flat[0::50176].fill_(0.02)
    return w


def build_moe(expert_num, hidden, inter, topk, threads):
    cpu = kt_kernel_ext.CPUInfer(threads)
    t0 = time.time()
    gate = fast_weights((expert_num, inter, hidden))
    up = fast_weights((expert_num, inter, hidden))
    down = fast_weights((expert_num, hidden, inter))
    print(f"  [weights allocated in {time.time()-t0:.1f}s]", flush=True)

    config = kt_kernel_ext.moe.MOEConfig(expert_num, topk, hidden, inter, 0)
    config.max_len = 1024
    config.gate_proj = gate.data_ptr()
    config.up_proj = up.data_ptr()
    config.down_proj = down.data_ptr()
    config.gate_scale = 0
    config.up_scale = 0
    config.down_scale = 0
    config.pool = cpu.backend_

    moe = NEON_MOE_CLS(config)
    mapping = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpu.submit(moe.load_weights_task(mapping.data_ptr()))
    cpu.sync()
    return cpu, moe, (gate, up, down)


def run_once(cpu, moe, qlen, expert_num, hidden, topk):
    expert_ids = torch.stack(
        [torch.randperm(expert_num)[:topk] for _ in range(qlen)]
    ).contiguous()
    weights = torch.rand((qlen, topk), dtype=torch.float32).contiguous()
    x = (torch.randn((qlen, hidden), dtype=torch.float32) / 100).to(torch.bfloat16).contiguous()
    out = torch.empty((qlen, hidden), dtype=torch.bfloat16).contiguous()
    bsz = torch.tensor([qlen], dtype=torch.int32)

    t0 = time.perf_counter()
    cpu.submit(
        moe.forward_task(
            bsz.data_ptr(), topk,
            expert_ids.data_ptr(), weights.data_ptr(),
            x.data_ptr(), out.data_ptr(), False,
        )
    )
    cpu.sync()
    return time.perf_counter() - t0



def compute_probe(threads, iters=10):
    """Cache-resident probe: weights tiny enough to stay in cache, so DRAM
    bandwidth is (almost) irrelevant and the measured GFLOP/s approximates
    the pure BFDOT compute rate at this thread count."""
    hidden, inter, topk = 1024, 512, 2
    cpu, moe, wts = build_moe(2, hidden, inter, topk, threads)
    run_once(cpu, moe, 1, 2, hidden, topk)  # warm up, populate cache
    best_dt = min(run_once(cpu, moe, 1, 2, hidden, topk) for _ in range(iters))
    flops = topk * 3 * 2 * hidden * inter * 2
    del cpu, moe, wts
    return flops / best_dt / 1e9  # GFLOP/s


def verdict(args, bw_ceiling, compute_gflops, threads):
    """Roofline verdict for the decode path."""
    per_expert_bytes = 3 * args.inter * args.hidden * 2
    bytes_per_tok = per_expert_bytes * args.topk
    flops_per_tok = args.topk * 3 * 2 * args.hidden * args.inter * 2
    t_mem = bytes_per_tok / (bw_ceiling * 1e9)
    t_cmp = flops_per_tok / (compute_gflops * 1e9)
    print(f"\n--- roofline verdict (decode, threads={threads}) ---")
    print(f"  bytes/token       : {bytes_per_tok/1e6:.1f} MB")
    print(f"  flops/token       : {flops_per_tok/1e9:.3f} GFLOP")
    print(f"  time if MEM-bound : {t_mem*1e3:.3f} ms/token  (bw={bw_ceiling:.1f} GB/s)")
    print(f"  time if COMPUTE-  : {t_cmp*1e3:.3f} ms/token  (BFDOT rate={compute_gflops:.1f} GFLOP/s)")
    if t_mem > t_cmp * 1.5:
        print(f"  => MEMORY (bandwidth) BOUND: transfer takes {t_mem/t_cmp:.1f}x longer than compute.")
        print("     Quantizing experts (GGUF Q4 ~ 4x fewer bytes) is the top lever.")
    elif t_cmp > t_mem * 1.5:
        print(f"  => COMPUTE BOUND: BFDOT math takes {t_cmp/t_mem:.1f}x longer than transfer.")
        print("     Optimizing neon_bf16_gemm.hpp (tiling, prefetch, unroll) is the lever,")
        print("     or move more work to the GPU.")
    else:
        print(f"  => BALANCED: memory and compute within 1.5x; both matter.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=2048)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    p.add_argument("--qlens", type=int, nargs="+", default=[1, 2, 8, 32, 128])
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()

    per_expert_bytes = 3 * args.inter * args.hidden * 2
    print("=" * 70)
    print(f"NEON BF16 MoE perf: experts={args.experts} hidden={args.hidden} "
          f"inter={args.inter} topk={args.topk}")
    print(f"  expert size: {per_expert_bytes/1e6:.1f} MB, "
          f"bytes/token(qlen=1): {per_expert_bytes*args.topk/1e6:.1f} MB")

    bw_ceiling = bench_bandwidth()
    print(f"  raw memory read bandwidth ceiling: {bw_ceiling:.1f} GB/s")
    print("=" * 70)

    prev = None  # only one MOE alive at a time: load_weights duplicates weights
    for threads in args.threads:
        if prev is not None:
            del prev
            import gc; gc.collect()
            print("  [released previous MOE weights]", flush=True)
        print(f"building MOE for threads={threads} ...", flush=True)
        prev = build_moe(args.experts, args.hidden, args.inter, args.topk, threads)
        cpu, moe, wts = prev
        print(f"\n--- threads = {threads} ---", flush=True)
        for qlen in args.qlens:
            run_once(cpu, moe, qlen, args.experts, args.hidden, args.topk)  # warm up
            best_dt = min(
                run_once(cpu, moe, qlen, args.experts, args.hidden, args.topk)
                for _ in range(args.iters)
            )
            tok_s = qlen / best_dt
            # decode: each token activates topk experts; prefill batches tokens
            # sharing an expert, dedup by min(qlen*topk, expert_num)
            n_touched = min(qlen * args.topk, args.experts)
            bytes_read = n_touched * per_expert_bytes
            gbs = bytes_read / best_dt / 1e9
            flops = qlen * args.topk * 3 * 2 * args.hidden * args.inter * 2
            gflops = flops / best_dt / 1e9
            bw_pct = 100.0 * gbs / bw_ceiling
            label = "decode" if qlen == 1 else "prefill"
            print(f"  qlen={qlen:4d} ({label:7s}): {best_dt*1e3:8.2f} ms  "
                  f"{tok_s:8.1f} tok/s  {gbs:7.1f} GB/s ({bw_pct:4.1f}% bw)  "
                  f"{gflops:8.1f} GFLOP/s", flush=True)

    print("\n" + "=" * 70)
    print("How to read this:")
    print("  - decode (qlen=1): if GB/s is close to the ceiling you are")
    print("    bandwidth-bound -> quantize experts (GGUF); extra threads stop")
    print("    helping once the bandwidth curve flattens.")
    print("  - prefill: GFLOP/s flattening while GB/s drops -> compute bound;")
    print("    push those batches to the GPU instead.")

    # --- controlled experiment: isolate compute vs bandwidth ---
    th = args.threads[-1]
    gflops = compute_probe(th)
    print(f"  cache-resident BFDOT rate @ {th} threads: {gflops:.1f} GFLOP/s")
    verdict(args, bw_ceiling, gflops, th)
    print("=" * 70)


if __name__ == "__main__":
    main()