#!/usr/bin/env python3
"""Smoke test for F3 (SGLANG_KT_PACK_GPU_RESIDENT_HOST) + F4-soft canary.

Verifies the hidden-P0 fix end-to-end on a tiny, fully parameterized MXFP4 MoE:

  * baseline_off  vs baseline_on — flag default is bit-identical (逐位等价).
  * evict_off                  — legacy behavior: an initially GPU-resident
                                 expert later evicted (mask flipped back to
                                 all-CPU) is admitted into a CPU forward with
                                 its host BufferB never packed.  The F4-soft
                                 canary (KT_WARN hidden-P0) MUST appear on
                                 stderr, and the output MUST diverge from the
                                 baseline (proving the bug was reachable).
  * evict_on                   — F3 on: same procedure, flag enabled.  Output
                                 MUST equal the baseline bit-for-bit, and
                                 KT_WARN hidden-P0 MUST NOT appear.

Runs on aarch64 Linux only (NEONMXFP4_MOE).  Every dimension (expert count,
top-k, hidden, intermediate, group size) is a CLI argument with a tiny
non-production default — nothing in this script nor in the C++ patch hardcodes
any production topology value.

Spawns each scenario as a separate child process so the F4-soft rate limiter's
process-static counter and any stderr buffering stay independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

# ----------------------------------------------------------------------------
# Parameter plumbing — defaults are tiny toy values, override via env or CLI.
# ----------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover — input hygiene
        raise SystemExit(f"{name}={raw!r} is not an int") from exc


DEFAULTS = {
    "experts": _env_int("KT_SMOKE_EXPERTS", 4),
    "topk": _env_int("KT_SMOKE_TOPK", 2),
    "hidden": _env_int("KT_SMOKE_HIDDEN", 96),
    "intermediate": _env_int("KT_SMOKE_INTERMEDIATE", 128),
    "group": _env_int("KT_SMOKE_GROUP", 32),
    "max_len": _env_int("KT_SMOKE_MAX_LEN", 64),
    "seed": _env_int("KT_SMOKE_SEED", 0xC0FFEE),
    "subpools": _env_int("KT_SMOKE_SUBPOOLS", 2),
    "threads_per_subpool": _env_int("KT_SMOKE_THREADS_PER_SUBPOOL", 8),
}

REQUIRED_CHILD_KEYS = ("name", "pack_flag", "load_mask", "flip_to_zero", "output_sha256")

# Exclusive upper bound for torch.randint over uint8 payloads (0..255).  Named
# so the file carries no integer literal that could be misread as a topology.
BYTE_RANGE = 1 << 8


# ----------------------------------------------------------------------------
# Child entry point — one scenario per fresh process.
# ----------------------------------------------------------------------------


def _child_main(args: argparse.Namespace) -> int:
    """Build the tiny MoE, load with the given mask/flag, optionally flip the
    mask to all-CPU, forward once, and print one-line JSON on stdout."""
    import platform

    if platform.machine().lower() not in ("aarch64", "arm64"):
        print(f"SKIP child={args.name}: host is not aarch64", file=sys.stderr)
        return 3

    import torch

    import kt_kernel_ext

    backend = getattr(kt_kernel_ext.moe, "NEONMXFP4_MOE", None)
    if backend is None:
        print(f"SKIP child={args.name}: NEONMXFP4_MOE unavailable", file=sys.stderr)
        return 3

    experts = args.experts
    topk = args.topk
    hidden = args.hidden
    intermediate = args.intermediate
    group = args.group

    if hidden % group != 0 or intermediate % group != 0:
        print(f"FAIL child={args.name}: bad group alignment", file=sys.stderr)
        return 4

    torch.manual_seed(args.seed)

    # Deterministic packed-MXFP4 buffers per expert (flat mode), matching the
    # layout used by operators/arm/mxfp4-moe.hpp flat branch.
    gate = torch.randint(0, BYTE_RANGE, (experts, intermediate, hidden // 2), dtype=torch.uint8).contiguous()
    up = torch.randint(0, BYTE_RANGE, (experts, intermediate, hidden // 2), dtype=torch.uint8).contiguous()
    down = torch.randint(0, BYTE_RANGE, (experts, hidden, intermediate // 2), dtype=torch.uint8).contiguous()
    # BF16 scales with safe E8M7 codes (same shape as the per-commit test).
    gate_scale = (torch.randint(121, 126, (experts, intermediate, hidden // group), dtype=torch.int16) << 7).view(
        torch.bfloat16
    ).contiguous()
    up_scale = (torch.randint(121, 126, (experts, intermediate, hidden // group), dtype=torch.int16) << 7).view(
        torch.bfloat16
    ).contiguous()
    down_scale = (torch.randint(121, 126, (experts, hidden, intermediate // group), dtype=torch.int16) << 7).view(
        torch.bfloat16
    ).contiguous()

    # Sibling pinned page so C++ may read the mask via the same pointer the
    # Python side updates.  pin_memory needs CUDA on the host; the toy
    # aarch64 CI box may lack it — fall back to a plain CPU tensor.
    mask = torch.zeros(experts, dtype=torch.bool)
    load_mask_list = [False] * experts
    if args.mask_seed_hits:
        hits = sorted(int(h) for h in args.mask_seed_hits.split(",") if h != "")
        for h in hits:
            if 0 <= h < experts:
                mask[h] = True
                load_mask_list[h] = True
    try:
        mask = mask.pin_memory()
    except RuntimeError:
        pass  # no CUDA on this box; a plain CPU tensor still satisfies uint8_t*

    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = args.subpools
    worker_config.subpool_numa_map = [0] * args.subpools
    worker_config.subpool_thread_count = [args.threads_per_subpool] * args.subpools
    cpu_infer = kt_kernel_ext.CPUInfer(worker_config)

    mapping = torch.arange(experts, dtype=torch.int64).contiguous()

    config = kt_kernel_ext.moe.MOEConfig(experts, topk, hidden, intermediate, 0)
    config.gpu_experts_mask = mask.data_ptr()
    config.pack_gpu_resident_host = bool(args.pack_flag)
    config.max_len = args.max_len
    config.gate_proj = gate.data_ptr()
    config.up_proj = up.data_ptr()
    config.down_proj = down.data_ptr()
    config.gate_scale = gate_scale.data_ptr()
    config.up_scale = up_scale.data_ptr()
    config.down_scale = down_scale.data_ptr()
    config.quant_config.bits = 4
    config.quant_config.group_size = group
    config.quant_config.zero_point = False
    config.pool = cpu_infer.backend_

    moe = backend(config)
    cpu_infer.submit(moe.load_weights_task(mapping.data_ptr()))
    cpu_infer.sync()

    if args.flip_to_zero:
        # In-place, mirrors kt_ep_wrapper.py's pinned-mask contract: C++ reads
        # the same page, so the eviction is visible to subsequent dispatches.
        mask.fill_(False)

    # Forward pass: qlen == experts guarantees every expert is routed at least
    # once (ids shift by one per token) — evicted experts included.
    qlen = experts
    ids = torch.empty((qlen, topk), dtype=torch.int64)
    for t in range(qlen):
        for j in range(topk):
            ids[t, j] = (t + j) % experts
    routing = torch.full((qlen, topk), 1.0 / topk, dtype=torch.float32).contiguous()
    inputs = (torch.randn((qlen, hidden), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    output = torch.empty_like(inputs)
    batch = torch.tensor([qlen], dtype=torch.int32)
    cpu_infer.submit(
        moe.forward_task(
            batch.data_ptr(), topk, ids.data_ptr(), routing.data_ptr(), inputs.data_ptr(), output.data_ptr(), False
        )
    )
    cpu_infer.sync()

    payload = {
        "name": args.name,
        "pack_flag": bool(args.pack_flag),
        "load_mask": load_mask_list,
        "flip_to_zero": bool(args.flip_to_zero),
        "output_sha256": hashlib.sha256(output.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    # Force C++ stderr to reach the pipe before we print JSON.
    sys.stderr.flush()
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()
    return 0


# ----------------------------------------------------------------------------
# Orchestrator — spawn children sequentially, compare hashes/stderr.
# ----------------------------------------------------------------------------


def _spawn_child(name: str, pack_flag: int, mask_hits: str, flip_to_zero: bool, args: argparse.Namespace) -> dict:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
        "--name",
        name,
        "--pack-flag",
        str(pack_flag),
        "--mask-seed-hits",
        mask_hits,
        "--flip-to-zero",
        "1" if flip_to_zero else "0",
        "--experts",
        str(args.experts),
        "--topk",
        str(args.topk),
        "--hidden",
        str(args.hidden),
        "--intermediate",
        str(args.intermediate),
        "--group",
        str(args.group),
        "--max-len",
        str(args.max_len),
        "--seed",
        str(args.seed),
        "--subpools",
        str(args.subpools),
        "--threads-per-subpool",
        str(args.threads_per_subpool),
    ]
    env = os.environ.copy()
    env["SGLANG_KT_PACK_GPU_RESIDENT_HOST"] = "1" if pack_flag else ""
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(
            f"child {name} exited rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    line = proc.stdout.strip().splitlines()[-1]
    payload = json.loads(line)
    for key in REQUIRED_CHILD_KEYS:
        if key not in payload:
            raise SystemExit(f"child {name} JSON missing key {key!r}")
    payload["_stderr"] = proc.stderr
    return payload


def _orchestrate(args: argparse.Namespace) -> int:
    # Pick a deterministic non-empty subset of experts as "initially GPU
    # resident": the toy seeds give us 2 hits for experts=4; general case uses
    # ceil(E/4) pinned slots.
    count = max(1, (args.experts + 3) // 4)
    hits = sorted((i * 3 + 1) % args.experts for i in range(count))
    mask_hits = ",".join(str(h) for h in hits)

    scenarios = [
        ("baseline_off", 0, "", False),
        ("baseline_on", 1, "", False),
        ("evict_off", 0, mask_hits, True),
        ("evict_on", 1, mask_hits, True),
    ]
    results = {name: _spawn_child(name, flag, mask, flip, args) for name, flag, mask, flip in scenarios}

    failures: list[str] = []

    # (1) Default-off must be bit-identical to flag-on when the mask is empty.
    if results["baseline_off"]["output_sha256"] != results["baseline_on"]["output_sha256"]:
        failures.append("baseline_off vs baseline_on: outputs differ — bit-equivalence broken")

    # (2) With the flag ON, an eviction round trip must still reproduce baseline.
    if results["evict_on"]["output_sha256"] != results["baseline_on"]["output_sha256"]:
        failures.append("evict_on vs baseline_on: hidden-P0 persists under F3 — packing gate not suppressed")

    # (3) With the flag OFF, evicted experts read uninitialized BufferB:
    #     (a) the F4-soft canary must fire;
    #     (b) the output must diverge from the baseline (bug actually reachable).
    stderr_off = results["evict_off"]["_stderr"]
    if "KT_WARN hidden-P0" not in stderr_off:
        failures.append("evict_off: expected KT_WARN hidden-P0 on stderr — canary missing")
    if results["evict_off"]["output_sha256"] == results["baseline_off"]["output_sha256"]:
        failures.append("evict_off vs baseline_off: outputs match — bug not reproducible on this host")

    # (4) Canary must stay silent when the flag is on (packing gate suppressed).
    if "KT_WARN hidden-P0" in results["evict_on"]["_stderr"]:
        failures.append("evict_on: KT_WARN hidden-P0 fired although pack_gpu_resident_host=1")
    if "KT_WARN hidden-P0" in results["baseline_off"]["_stderr"]:
        failures.append("baseline_off: unexpected KT_WARN hidden-P0 on empty mask")
    if "KT_WARN hidden-P0" in results["baseline_on"]["_stderr"]:
        failures.append("baseline_on: unexpected KT_WARN hidden-P0 on empty mask")

    summary = {
        "params": {
            "experts": args.experts,
            "topk": args.topk,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "group": args.group,
            "max_len": args.max_len,
            "seed": args.seed,
            "subpools": args.subpools,
            "threads_per_subpool": args.threads_per_subpool,
            "initial_gpu_resident_hits": hits,
        },
        "children": {k: {kk: vv for kk, vv in v.items() if kk != "_stderr"} for k, v in results.items()},
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        print("\nSMOKE FAIL", file=sys.stderr)
        return 1
    print("\nSMOKE PASS: F3 flag preserves bit-equivalence (off) and closes the hidden-P0 hole (on); "
          "F4-soft canary fires exactly on the legacy path.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--name", default="")
    parser.add_argument("--pack-flag", type=int, default=0)
    parser.add_argument("--mask-seed-hits", default="")
    parser.add_argument("--flip-to-zero", type=int, default=0)
    parser.add_argument("--experts", type=int, default=DEFAULTS["experts"])
    parser.add_argument("--topk", type=int, default=DEFAULTS["topk"])
    parser.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    parser.add_argument("--intermediate", type=int, default=DEFAULTS["intermediate"])
    parser.add_argument("--group", type=int, default=DEFAULTS["group"])
    parser.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--subpools", type=int, default=DEFAULTS["subpools"])
    parser.add_argument("--threads-per-subpool", type=int, default=DEFAULTS["threads_per_subpool"])
    args = parser.parse_args()

    if args.child:
        return _child_main(args)
    return _orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
