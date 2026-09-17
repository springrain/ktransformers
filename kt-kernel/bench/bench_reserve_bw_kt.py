#!/usr/bin/env python
# coding=utf-8
"""bench_reserve_bw_kt.py -- reserve_cores_per_numa 0/1/2 three-tier sweep probe.

Answers the tier-3 open question (doc/ft-kt-phase3-plan-decode-side.md §9-5):
does reserving 1-2 low cores per NUMA node (keeping OS IRQ / coordinator / CUDA
callback threads off the pinned GEMV pool range) measurably change the worker
pool's aggregate GEMV bandwidth or iteration-time jitter?

Form: one process == one reserve tier (CPUInfer is a process-wide singleton and
cannot be re-pinned inside a live process). The default master mode spawns one
child process per selected tier, each writing its own JSON, then merges a
comparison JSON. Measurements follow the kt-kernel/bench/bench_moe_kernel.py
driver shape (WorkerPoolConfig + moe.Int4/Int8_KERNEL_MOE + submit/sync loop).

Pool construction mirrors experts_base.py:_get_cpu_infer semantics exactly:
subpool_thread_count split, field written only for a truthy reserve, and the
torch intra-op clamp gated on reserve (bit-exact legacy path at reserve=0).

Production untouched: imports neither sglang nor kt-kernel's python wrappers,
changes zero files, no CUDA, no torch.distributed. CPU-only DRAM load — still
schedule inside the maintenance window per doc/ft-kt-xysa10-runbook.md.

Environment gates (all read once at start, echoed into the JSON config block):
  SGLANG_KT_RESERVEBW_RESERVE   unset = master sweep over TIERS; child: 0/1/2/3
  SGLANG_KT_RESERVEBW_TIERS     default 0,1,2          master sweep set
  SGLANG_KT_RESERVEBW_CHILD     default 0 (internal; set only by the master)
  SGLANG_KT_RESERVEBW_QUANT     default int4 (int8 fallback if the build lacks
                                the int4 moe kernel on this platform)
  SGLANG_KT_RESERVEBW_EXPERTS   default 64    work-set geometry (env-tunable,
  SGLANG_KT_RESERVEBW_HIDDEN    default 2048  NOT production pins; resize via
  SGLANG_KT_RESERVEBW_INTER     default 768   env to match any checkpoint)
  SGLANG_KT_RESERVEBW_TOPK      default 8
  SGLANG_KT_RESERVEBW_QLENS     default 8,256,2048   batch shapes to measure
  SGLANG_KT_RESERVEBW_ITERS     default 30    timed iterations per qlen
  SGLANG_KT_RESERVEBW_WARMUP    default 5
  SGLANG_KT_RESERVEBW_SUBPOOLS  default: NUMA node count (sysfs; fallback 1)
  SGLANG_KT_RESERVEBW_THREADS   default: affinity CPU count
  SGLANG_KT_RESERVEBW_OUT_DIR   default .    JSON directory
  SGLANG_KT_RESERVEBW_QUICK     default 0    1 = iters halved (dev loop)

Exit codes: 0 ok | 2 refused (platform/geometry/env) | 3 harness/binding fault.

Usage:
  python kt-kernel/bench/bench_reserve_bw_kt.py            # master: sweep 0/1/2
  SGLANG_KT_RESERVEBW_QUICK=1 python kt-kernel/bench/bench_reserve_bw_kt.py
"""

from __future__ import annotations

import glob
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

SCHEMA = "bench_reserve_bw_kt/1"
RNG_SEED = 20260917


class HarnessFault(RuntimeError):
    pass


def _die(code, msg):
    print(f"[bench_reserve_bw_kt] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _log(msg):
    print(f"[bench_reserve_bw_kt] {msg}", flush=True)


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        _die(2, f"{name} must be an int, got {raw!r}")


def _env_csv_ints(name, default_csv):
    raw = os.environ.get(name, default_csv)
    try:
        vals = [int(s) for s in raw.split(",") if s.strip() != ""]
    except ValueError:
        _die(2, f"{name} must be comma-separated ints, got {raw!r}")
    if not vals:
        _die(2, f"{name} must not be empty")
    return vals


def read_config():
    reserve_raw = os.environ.get("SGLANG_KT_RESERVEBW_RESERVE", "").strip()
    reserve = None if reserve_raw == "" else _env_int("SGLANG_KT_RESERVEBW_RESERVE", 0)
    if reserve is not None and reserve < 0:
        _die(2, "SGLANG_KT_RESERVEBW_RESERVE must be >= 0")
    quant = os.environ.get("SGLANG_KT_RESERVEBW_QUANT", "int4").strip().lower()
    if quant not in ("int4", "int8"):
        _die(2, f"SGLANG_KT_RESERVEBW_QUANT must be int4|int8, got {quant!r}")
    cfg = {
        "reserve": reserve,
        "tiers": _env_csv_ints("SGLANG_KT_RESERVEBW_TIERS", "0,1,2"),
        "child": _env_int("SGLANG_KT_RESERVEBW_CHILD", 0),
        "quant": quant,
        "experts": _env_int("SGLANG_KT_RESERVEBW_EXPERTS", 64),
        "hidden": _env_int("SGLANG_KT_RESERVEBW_HIDDEN", 2048),
        "inter": _env_int("SGLANG_KT_RESERVEBW_INTER", 768),
        "topk": _env_int("SGLANG_KT_RESERVEBW_TOPK", 8),
        "qlens": _env_csv_ints("SGLANG_KT_RESERVEBW_QLENS", "8,256,2048"),
        "iters": _env_int("SGLANG_KT_RESERVEBW_ITERS", 30),
        "warmup": _env_int("SGLANG_KT_RESERVEBW_WARMUP", 5),
        "subpools": _env_int("SGLANG_KT_RESERVEBW_SUBPOOLS", 0),  # 0 = detect
        "threads": _env_int("SGLANG_KT_RESERVEBW_THREADS", 0),  # 0 = detect
        "out_dir": os.environ.get("SGLANG_KT_RESERVEBW_OUT_DIR", "."),
        "quick": _env_int("SGLANG_KT_RESERVEBW_QUICK", 0),
    }
    if cfg["quick"]:
        cfg["iters"] = max(2, cfg["iters"] // 2)
        cfg["warmup"] = max(1, cfg["warmup"] // 2)
    if cfg["experts"] <= 0 or cfg["hidden"] <= 0 or cfg["inter"] <= 0:
        _die(2, "work-set geometry must be positive")
    if cfg["topk"] <= 0 or cfg["topk"] > cfg["experts"]:
        _die(2, f"topk {cfg['topk']} invalid for experts={cfg['experts']}")
    for q in cfg["qlens"]:
        if q <= 0:
            _die(2, "qlens must be positive")
    return cfg


def discover_topology(cfg):
    try:
        aff = sorted(os.sched_getaffinity(0))
    except AttributeError:
        _die(2, "sched_getaffinity unavailable: this probe targets Linux (xysa10)")
    threads = cfg["threads"] if cfg["threads"] > 0 else len(aff)
    nodes = glob.glob("/sys/devices/system/node/node[0-9]*")
    subpools = cfg["subpools"] if cfg["subpools"] > 0 else max(1, len(nodes))
    if threads < subpools:
        _die(2, f"threads {threads} < subpools {subpools}")
    counts = []
    base, rem = divmod(threads, subpools)
    for i in range(subpools):
        counts.append(base + (1 if i < rem else 0))
    return {
        "affinity_cpus": len(aff),
        "numa_nodes_seen": len(nodes),
        "threads": threads,
        "subpools": subpools,
        "subpool_thread_count": counts,
        "subpool_numa_map": list(range(subpools)),
    }


def sys_meta():
    meta = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    for cmd, key in (("lscpu", "lscpu"),):
        try:
            out = subprocess.check_output(cmd, timeout=10).decode()
            meta[key + "_summary"] = {
                k.strip(): v.strip()
                for k, v in (line.split(":", 1) for line in out.splitlines() if ":" in line)
                if k.strip() in ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "NUMA node(s)")
            }
        except Exception:
            pass
    try:
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], timeout=10).decode().strip()
        meta["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], timeout=10).decode().strip())
    except Exception:
        meta["git_commit"] = None
        meta["git_dirty"] = None
    return meta


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def expert_bytes_estimate(cfg):
    # gate+up+down scale-free byte basis for the GB/s estimate; scale bytes
    # excluded (reference-only, never part of any pass/fail).
    per_matrix = cfg["hidden"] * cfg["inter"]
    factor = 0.5 if cfg["quant"] == "int4" else 1.0
    return int(3 * per_matrix * factor)


def build_pool(cfg, topo, ext):
    if not hasattr(ext, "WorkerPoolConfig"):
        _die(3, "kt_kernel_ext lacks WorkerPoolConfig (pre-tier-2 build)")
    wc = ext.WorkerPoolConfig()
    wc.subpool_count = topo["subpools"]
    wc.subpool_numa_map = topo["subpool_numa_map"]
    wc.subpool_thread_count = topo["subpool_thread_count"]
    if cfg["reserve"]:
        if not hasattr(wc, "reserve_cores_per_numa"):
            _die(
                3,
                "kt_kernel_ext.WorkerPoolConfig lacks reserve_cores_per_numa: "
                "rebuild kt_kernel_ext with the tier-3 worker_pool changes "
                "(plan §9-1) before sweeping tiers 1/2",
            )
        wc.reserve_cores_per_numa = cfg["reserve"]
    try:
        pool = ext.CPUInfer(wc)
    except Exception as e:
        _die(
            3,
            f"CPUInfer({topo['subpools']} subpools, counts={topo['subpool_thread_count']}, "
            f"reserve={cfg['reserve']}) construction failed (config-first validation): {e}",
        )
    if cfg["reserve"]:
        # Same gate as experts_base._get_cpu_infer: clamp only when reserve is
        # truthy, clamp down only. Keeps the torch intra-op threads inside a
        # subpool's reduced budget.
        import torch

        torch.set_num_threads(min(torch.get_num_threads(), min(topo["subpool_thread_count"])))
    return pool


def make_moe(cfg, pool, ext):
    import torch

    gate = torch.randn((cfg["experts"], cfg["inter"], cfg["hidden"]), dtype=torch.float32).contiguous()
    up = torch.randn((cfg["experts"], cfg["inter"], cfg["hidden"]), dtype=torch.float32).contiguous()
    down = torch.randn((cfg["experts"], cfg["hidden"], cfg["inter"]), dtype=torch.float32).contiguous()
    config = ext.moe.MOEConfig(cfg["experts"], cfg["topk"], cfg["hidden"], cfg["inter"], 0)
    config.max_len = max(cfg["qlens"])
    config.gate_proj = gate.data_ptr()
    config.up_proj = up.data_ptr()
    config.down_proj = down.data_ptr()
    config.pool = pool.backend_
    if cfg["quant"] == "int4":
        moe = ext.moe.Int4_KERNEL_MOE(config)
    else:
        moe = ext.moe.Int8_KERNEL_MOE(config)
    phys_map = torch.arange(cfg["experts"], dtype=torch.int64).contiguous()
    pool.submit(moe.load_weights_task(phys_map.data_ptr()))
    pool.sync()
    return moe, (gate, up, down)


def measure_qlen(cfg, pool, moe, qlen):
    import torch

    g = torch.Generator().manual_seed(RNG_SEED)
    n = cfg["iters"] + cfg["warmup"]
    ids_full = torch.rand(n * qlen, cfg["experts"], generator=g).argsort(dim=-1)[:, : cfg["topk"]].contiguous()
    ids_full = ids_full.reshape(n, qlen * cfg["topk"]).contiguous()
    weights = torch.rand((n, qlen, cfg["topk"]), dtype=torch.float32, generator=g).contiguous()
    inp = torch.randn((qlen, cfg["hidden"]), dtype=torch.float32, generator=g).to(torch.bfloat16).contiguous()
    out = torch.empty((qlen, cfg["hidden"]), dtype=torch.bfloat16).contiguous()
    bsz = torch.tensor([qlen], dtype=torch.int32).contiguous()
    lat = []
    for i in range(n):
        t0 = time.perf_counter_ns()
        pool.submit(
            moe.forward_task(
                bsz.data_ptr(),
                cfg["topk"],
                ids_full[i].data_ptr(),
                weights[i].data_ptr(),
                inp.data_ptr(),
                out.data_ptr(),
                False,
            )
        )
        pool.sync()
        if i >= cfg["warmup"]:
            lat.append(time.perf_counter_ns() - t0)
    lat_ms = [x / 1e6 for x in lat]
    p50 = statistics.median(lat_ms)
    mean = statistics.fmean(lat_ms)
    stdev = statistics.pstdev(lat_ms) if len(lat_ms) > 1 else 0.0
    unique_experts = int(torch.unique(ids_full).numel())
    visited_bytes = unique_experts * expert_bytes_estimate(cfg)
    return {
        "qlen": qlen,
        "iters": len(lat_ms),
        "warmup": cfg["warmup"],
        "p50_ms": round(p50, 4),
        "p95_ms": round(sorted(lat_ms)[max(0, int(0.95 * len(lat_ms)) - 1)], 4),
        "min_ms": round(min(lat_ms), 4),
        "mean_ms": round(mean, 4),
        "cv_pct": round(100.0 * stdev / mean, 3) if mean > 0 else None,
        "unique_experts_per_iter": unique_experts,
        "visited_bytes_per_iter_est": visited_bytes,
        "gbps_est": round(visited_bytes / (p50 / 1e3) / 1e9, 3) if p50 > 0 else None,
    }


def child_main(cfg):
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"))
    try:
        from kt_kernel import kt_kernel_ext as ext
    except Exception as e:
        _die(2, f"import kt_kernel_ext failed (build kt-kernel first): {e}")
    topo = discover_topology(cfg)
    _log(
        f"child reserve={cfg['reserve']}: subpools={topo['subpools']} "
        f"counts={topo['subpool_thread_count']} quant={cfg['quant']}"
    )
    pool = build_pool(cfg, topo, ext)
    moe, _keep = make_moe(cfg, pool, ext)
    rows = [measure_qlen(cfg, pool, moe, q) for q in cfg["qlens"]]
    report = {
        "schema": SCHEMA,
        "kind": "tier",
        "reserve": cfg["reserve"],
        "meta": sys_meta(),
        "config": {k: v for k, v in cfg.items()},
        "topology": topo,
        "rows": rows,
    }
    name = f"bench_reserve_bw_kt_r{cfg['reserve']}_{socket.gethostname()}_{int(time.time())}.json"
    path = os.path.join(cfg["out_dir"], name)
    atomic_write_json(path, report)
    _log(f"wrote {path}")
    for r in rows:
        _log(f"  qlen={r['qlen']:<5} p50={r['p50_ms']:.3f}ms cv={r['cv_pct']}% gbps~{r['gbps_est']}")
    print(json.dumps({"reserve": cfg["reserve"], "json": path, "rows": rows}))
    return path


def master_main(cfg):
    tiers = cfg["tiers"]
    for t in tiers:
        if t < 0:
            _die(2, f"tier {t} invalid")
    out_paths = []
    for t in tiers:
        env = dict(os.environ)
        env["SGLANG_KT_RESERVEBW_RESERVE"] = str(t)
        env["SGLANG_KT_RESERVEBW_CHILD"] = "1"
        _log(f"=== tier reserve={t} ===")
        proc = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env)
        if proc.returncode != 0:
            _die(3, f"child reserve={t} exited {proc.returncode}")
        # child printed its path; re-glob the newest tier file in out_dir
        cand = sorted(
            glob.glob(os.path.join(cfg["out_dir"], f"bench_reserve_bw_kt_r{t}_*.json")),
            key=os.path.getmtime,
        )
        if not cand:
            _die(3, f"child reserve={t} produced no JSON in {cfg['out_dir']}")
        out_paths.append(cand[-1])
    per_tier = []
    for p in out_paths:
        with open(p, encoding="utf-8") as f:
            per_tier.append(json.load(f))
    baseline = per_tier[0]
    comparison = []
    for rep in per_tier:
        for row, base_row in zip(rep["rows"], baseline["rows"]):
            if row["qlen"] != base_row["qlen"]:
                _die(3, "qlen row mismatch across tiers: sweep must share one env geometry")
            if base_row["p50_ms"] and row["p50_ms"]:
                delta = 100.0 * (base_row["p50_ms"] - row["p50_ms"]) / base_row["p50_ms"]
            else:
                delta = None
            comparison.append(
                {
                    "reserve": rep["reserve"],
                    "qlen": row["qlen"],
                    "p50_ms": row["p50_ms"],
                    "cv_pct": row["cv_pct"],
                    "gbps_est": row["gbps_est"],
                    "p50_delta_pct_vs_tier0": round(delta, 3) if delta is not None else None,
                }
            )
    noise_floor = max(
        (r["cv_pct"] or 0.0) for rep in per_tier for r in rep["rows"]
    )
    merged = {
        "schema": SCHEMA,
        "kind": "comparison",
        "meta": sys_meta(),
        "config": {k: v for k, v in cfg.items()},
        "tier_jsons": out_paths,
        "baseline_reserve": baseline["reserve"],
        "rows": comparison,
        "noise_floor_cv_pct_max": round(noise_floor, 3),
        "reading_rule": (
            "treat p50_delta_pct_vs_tier0 > 2x noise_floor_cv_pct_max as a real "
            "signal at that qlen; anything inside the noise floor is not a gain "
            "(doc/ft-kt-xysa10-runbook.md)"
        ),
    }
    name = f"bench_reserve_bw_kt_compare_{socket.gethostname()}_{int(time.time())}.json"
    path = os.path.join(cfg["out_dir"], name)
    atomic_write_json(path, merged)
    _log(f"wrote {path}")
    for c in comparison:
        _log(
            f"  reserve={c['reserve']} qlen={c['qlen']:<5} p50={c['p50_ms']:.3f}ms "
            f"cv={c['cv_pct']}% delta_vs_t0={c['p50_delta_pct_vs_tier0']}%"
        )
    return path


def main():
    cfg = read_config()
    if platform.system() != "Linux":
        _die(2, "Linux-only probe (sched affinity, sysfs NUMA); run on xysa10")
    if cfg["reserve"] is None and not cfg["child"]:
        master_main(cfg)
    else:
        reserve = cfg["reserve"] if cfg["reserve"] is not None else 0
        cfg["reserve"] = reserve
        child_main(cfg)


if __name__ == "__main__":
    main()
