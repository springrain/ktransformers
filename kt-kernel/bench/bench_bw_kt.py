#!/usr/bin/env python
# coding=utf-8
"""bench_bw_kt.py -- KT PCIe/DRAM bandwidth attribution probe (plan: doc/ft-kt-phase1-plan-benchbw.md).

Standalone torch-only probe. Imports neither sglang nor kt_kernel_ext, changes zero
production files, never initializes torch.distributed. Run it on the target host
(xysa10-class aarch64, dual GPU, no NVLink) ONLY in a maintenance window; startup
guard rails refuse to run alongside a live server unless SGLANG_KT_BENCHBW_FORCE=1.

The question: production sustains ~11 GB/s per card H2D for the KT expert staging
path while the PCIe link can do ~22 GB/s per card -- who owns the gap?
  (a) host write supply   (rank0 writes BOTH ranks' shm; cross-NUMA write is the
                          production shape, kt_ep_wrapper.py :2240-2241/:2259-2266)
  (b) PCIe link / RC      (two DMA engines sharing an upstream resource)
  (c) rig invalid         (solo calibration outside the LnkSta-relative band)
  (d) CPU/DRAM contention (168-core NEON GEMV vs dual DMA on the same DRAM)

Production anchors reproduced here (verified against kt_ep_wrapper.py @ e26c065):
  - 4-bank per-expert copy  RAW_NAMES :1832-1836, copy shape :2281-2284 (side stream,
    non_blocking); expert bytes 19,161,088 (doc/ft-kt-migration-audit.md:5).
  - bf16 scale staging (:867-873) -> default BANK_SPLIT 32:4:16:2 (approximate split,
    NOT production-exact; sum asserted == EXPERT_BYTES, never enters verdict math).
  - shm lifecycle: owner create + first-touch + cudaHostRegister (:879-898) -> peer
    open (:931-960) -> owner unlink (:923-929). The P1 remote leg replays exactly
    this order via mp.Event A/B; unlink-before-open would FileNotFoundError on POSIX.

Methodology ported from FreeToken (read in full): physical_core_cpus /
thread-sibling dedup (cpu_executor.py:92-141), measure_cpu_mem_bw (:184-253),
measure_pcie_bw (:256-279), torch.cuda._sleep queue-priming (:433-434),
measure_overlap_bw contended regime rationale (:538-595), atomic JSON (:682-688).
House meta style follows kt-kernel/bench/bench_write_buffer.py:50-84.

Environment gates (12, all read at main() start, echoed into the JSON config block):
  SGLANG_KT_BENCHBW_EXPERT_BYTES   default 19161088   byte-accounting invariant base
  SGLANG_KT_BENCHBW_BANK_SPLIT     default 32:4:16:2  bf16-scale geometry (approx)
  SGLANG_KT_BENCHBW_EXPERT_COUNTS  default 1,8,24,64,232
  SGLANG_KT_BENCHBW_GPUS           default 0,1
  SGLANG_KT_BENCHBW_THREADS_SWEEP  default 1,2,4,8,16 (P1)
  SGLANG_KT_BENCHBW_OUT            default ./bench_bw_kt_<host>_<epoch>.json (atomic)
  SGLANG_KT_BENCHBW_QUICK          default 0  (1 = iters/buffers halved; verdict is
                                   branded quick-dev and confidence forced to low)
  SGLANG_KT_BENCHBW_ONLY           default all; subset of p1,p2,p3; dev-only values
                                   inject-half-shm / inject-kill run the built-in
                                   negative injections (plan §4) and nothing else —
                                   mixing injections with productive legs is refused
  SGLANG_KT_BENCHBW_PIN            default shm-register; alloc = cudaHostAlloc;
                                   off = negative injection (pageable, P2 must
                                   collapse below 5 GB/s or the rig is lying)
  SGLANG_KT_BENCHBW_CONTENDED      default 0  (1 = DRAM background load during P2 dual)
  SGLANG_KT_BENCHBW_FORCE          default 0  (1 = bypass startup guard rails)
  SGLANG_KT_BENCHBW_PROD_MS_PER_EXPERT  optional float; production ms/expert from the
                                   SGLANG_KT_PREFILL_STATS pure-copy ledger entry —
                                   attaches the plan §4 verdict-(b) control-plane
                                   cross-check (probe pure-copy vs production)

Exit codes: 0 ok (legs may be loudly skipped) | 2 refused (platform/env/guard)
  3 harness fault (child death / barrier timeout / byte-accounting assert /
     any negative-injection leg reporting pass=False)
  5 rig-invalid (calibration outside band; verdict='c', JSON still written)

Usage:
  python kt-kernel/bench/bench_bw_kt.py                       # full matrix
  SGLANG_KT_BENCHBW_QUICK=1 python kt-kernel/bench/bench_bw_kt.py
  SGLANG_KT_BENCHBW_ONLY=inject-half-shm python kt-kernel/bench/bench_bw_kt.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import queue
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing import shared_memory

import torch

# torch-only probe by construction; the final assert in main() proves no
# torch.distributed process group was ever initialized.

SCHEMA = "bench_bw_kt/1"

# Bank labels mirror RAW_NAMES order (kt_ep_wrapper.py:1832-1836); split weights
# follow the bf16-scale staging geometry 32:4:16:2 (plan verify P6).
RAW_BANK_NAMES = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")
DEFAULT_EXPERT_BYTES = 19161088
DEFAULT_BANK_SPLIT = "32:4:16:2"
DEFAULT_EXPERT_COUNTS = "1,8,24,64,232"
DEFAULT_GPUS = "0,1"
DEFAULT_THREADS_SWEEP = "1,2,4,8,16"

# xysa10 production observation feeding judgment (a)'s demand side (audit doc §5).
PROD_OBSERVED_PER_CARD_GBS = 11.0
# PCIe per-lane RAW signalling GB/s by link rate (GT/s); Gen6 PAM4/FLIT included so
# the band is LnkSta-relative, never Gen4-hardcoded (plan verify P3). The verdict
# band is computed on PAYLOAD rates: raw 128b/130b signalling overstates what DMA
# copies actually deliver (TLP headers/ACK/ordering ≈ 15-20% overhead) — a healthy
# Gen4-x16 link calibrates at ~24-26 GB/s, i.e. ~0.82 x its 31.5 GB/s raw figure;
# a raw-rate band floor would sit at/above a healthy link and scream rig-invalid.
LINK_GBS_PER_LANE_RAW = {2.5: 0.25, 5.0: 0.5, 8.0: 0.985, 16.0: 1.969, 32.0: 3.938, 64.0: 7.563}
PAYLOAD_FACTOR = 0.82
FALLBACK_BAND = (20.0, 25.0)  # unknown-gen fallback ≈ a healthy Gen4-x16 payload link; flagged
BAND_LO_FACTOR = 0.75
BAND_HI_FACTOR = 1.10

# P3 knee fractions of physical cores -> {40,80,120,168} on a 168-core host
# (truncation int(len(phys)*f), NOT rounding: 0.72*168 = 120.96 -> 120).
P3_KNEE_FRACTIONS = (0.24, 0.48, 0.72, 1.00)

BARRIER_TIMEOUT_S = 120          # plan §4 negative injection (c): never hang past this
GRANULARITY_SIZES = (64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024)  # + expert_bytes
MEM_GUARD_MIB = 512              # refuse if a target GPU already uses more than this
PIN_OFF_COLLAPSE_GBS = 5.0       # negative injection (a): PIN=off H2D must land below this
CV_LIMIT = 0.05                  # self-check: 3 repeats CV < 5%
SKEW_LIMIT = 0.10                # self-check: event vs perf_counter skew > 10% flagged

ENV_NAMES = (
    "SGLANG_KT_BENCHBW_EXPERT_BYTES",
    "SGLANG_KT_BENCHBW_BANK_SPLIT",
    "SGLANG_KT_BENCHBW_EXPERT_COUNTS",
    "SGLANG_KT_BENCHBW_GPUS",
    "SGLANG_KT_BENCHBW_THREADS_SWEEP",
    "SGLANG_KT_BENCHBW_OUT",
    "SGLANG_KT_BENCHBW_QUICK",
    "SGLANG_KT_BENCHBW_ONLY",
    "SGLANG_KT_BENCHBW_PIN",
    "SGLANG_KT_BENCHBW_CONTENDED",
    "SGLANG_KT_BENCHBW_FORCE",
    "SGLANG_KT_BENCHBW_PROD_MS_PER_EXPERT",
)

ONLY_LEGIT = ("p1", "p2", "p3")
ONLY_INJECT = ("inject-half-shm", "inject-kill")


class HarnessFault(RuntimeError):
    """Measurement rig failure (child death, barrier timeout, byte-accounting
    violation). Top-level maps this to exit code 3 with the partial JSON written."""


def _die(code, msg):
    print(f"[bench_bw_kt] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def _log(msg):
    print(f"[bench_bw_kt] {msg}", flush=True)


# --------------------------------------------------------------------------
# env / config
# --------------------------------------------------------------------------

def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        _die(2, f"env {name}={raw!r} is not an int")


def _env_flag(name):
    return _env_int(name, 0) == 1


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        _die(2, f"env {name}={raw!r} is not a float")


def _env_csv_ints(name, default_csv):
    raw = os.environ.get(name, default_csv)
    try:
        vals = [int(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        _die(2, f"env {name}={raw!r} must be a comma-separated list of ints")
    if not vals:
        _die(2, f"env {name} produced an empty list")
    return vals


def compute_banks(expert_bytes, split):
    """Proportional 4-bank split; remainder goes to the largest bank (w13).
    The byte identity sum(banks) == expert_bytes is asserted HERE, at config time,
    i.e. strictly before any timing loop (plan §4 negative injection b)."""
    parts = split.split(":")
    if len(parts) != len(RAW_BANK_NAMES):
        _die(2, f"BANK_SPLIT {split!r} must have {len(RAW_BANK_NAMES)} components")
    try:
        weights = [int(p) for p in parts]
    except ValueError:
        _die(2, f"BANK_SPLIT {split!r} must be colon-separated ints")
    if any(w <= 0 for w in weights):
        _die(2, f"BANK_SPLIT {split!r} must be strictly positive")
    total_w = sum(weights)
    sizes = [expert_bytes * w // total_w for w in weights]
    sizes[0] += expert_bytes - sum(sizes)  # largest bank absorbs the remainder
    banks = dict(zip(RAW_BANK_NAMES, sizes))
    assert sum(banks.values()) == expert_bytes, "bank identity broken pre-timing"
    return banks


def read_config():
    expert_bytes = _env_int("SGLANG_KT_BENCHBW_EXPERT_BYTES", DEFAULT_EXPERT_BYTES)
    if expert_bytes <= 0:
        _die(2, "EXPERT_BYTES must be positive")
    pin = os.environ.get("SGLANG_KT_BENCHBW_PIN", "shm-register")
    if pin not in ("shm-register", "alloc", "off"):
        _die(2, f"PIN={pin!r} must be one of shm-register|alloc|off")
    only_raw = os.environ.get("SGLANG_KT_BENCHBW_ONLY", "").strip().lower()
    only = tuple(x for x in only_raw.split(",") if x) if only_raw else ONLY_LEGIT
    unknown = [x for x in only if x not in ONLY_LEGIT + ONLY_INJECT]
    if unknown:
        _die(2, f"ONLY contains unknown leg(s) {unknown}; valid: {ONLY_LEGIT + ONLY_INJECT}")
    inject_legs = [x for x in only if x in ONLY_INJECT]
    prod_legs = [x for x in only if x in ONLY_LEGIT]
    if inject_legs and prod_legs:
        _die(2, f"ONLY mixes negative injections {inject_legs} with productive legs "
                f"{prod_legs}; injections run alone (their exit-code contract differs)")
    expert_counts = _env_csv_ints("SGLANG_KT_BENCHBW_EXPERT_COUNTS", DEFAULT_EXPERT_COUNTS)
    bad_counts = [n for n in expert_counts if not (1 <= n <= 4096)]
    if bad_counts:
        _die(2, f"EXPERT_COUNTS entries outside [1, 4096]: {bad_counts}")
    gpus = _env_csv_ints("SGLANG_KT_BENCHBW_GPUS", DEFAULT_GPUS)
    if any(g < 0 for g in gpus):
        _die(2, f"GPUS entries must be >= 0: {gpus}")
    if len(set(gpus)) != len(gpus):
        _die(2, f"GPUS contains duplicate indices: {gpus}")
    threads_sweep = _env_csv_ints("SGLANG_KT_BENCHBW_THREADS_SWEEP", DEFAULT_THREADS_SWEEP)
    bad_sweep = [n for n in threads_sweep if n <= 0]
    if bad_sweep:
        _die(2, f"THREADS_SWEEP entries must be positive: {bad_sweep}")
    prod_ms = _env_float("SGLANG_KT_BENCHBW_PROD_MS_PER_EXPERT", None)
    if prod_ms is not None and prod_ms <= 0:
        _die(2, "PROD_MS_PER_EXPERT must be positive when set")
    quick = _env_flag("SGLANG_KT_BENCHBW_QUICK")
    out = os.environ.get("SGLANG_KT_BENCHBW_OUT", "")
    if not out:
        out = os.path.abspath(
            f"./bench_bw_kt_{socket.gethostname()}_{int(time.time())}.json"
        )
    cfg = {
        "expert_bytes": expert_bytes,
        "banks": compute_banks(expert_bytes, os.environ.get("SGLANG_KT_BENCHBW_BANK_SPLIT", DEFAULT_BANK_SPLIT)),
        "banks_note": "approximate production granularity, NOT production-exact split (verify P6)",
        "expert_counts": expert_counts,
        "gpus": gpus,
        "threads_sweep": threads_sweep,
        "out": out,
        "quick": quick,
        "only": only,
        "pin": pin,
        "contended": _env_flag("SGLANG_KT_BENCHBW_CONTENDED"),
        "force": _env_flag("SGLANG_KT_BENCHBW_FORCE"),
        "prod_ms_per_expert": prod_ms,
    }
    return cfg


# --------------------------------------------------------------------------
# small sys utils
# --------------------------------------------------------------------------

def _read_text(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def _run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def parse_cpulist(s):
    cpus = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(tok))
    return cpus


def affinity_cpus():
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_git_commit():
    """House meta style (bench_write_buffer.py:50-63)."""
    head = _run(["git", "-C", repo_root(), "rev-parse", "HEAD"])
    dirty_raw = _run(["git", "-C", repo_root(), "status", "--porcelain"])
    return {
        "commit": head.strip() if head else "unknown",
        "dirty": bool(dirty_raw and dirty_raw.strip()),
        "dirty_files": dirty_raw.splitlines()[:50] if dirty_raw else [],
    }


def get_system_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "logical_cpus": os.cpu_count(),
        "cpu_model": None,
        "mem_total_gib": None,
    }
    cpuinfo = _read_text("/proc/cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    meminfo = _read_text("/proc/meminfo")
    if meminfo:
        m = re.search(r"MemTotal:\s+(\d+) kB", meminfo)
        if m:
            info["mem_total_gib"] = round(int(m.group(1)) / 2 ** 20, 1)
    return info


def mem_available_bytes():
    """MemAvailable in bytes, or None when unreadable (callers loud-skip their
    guard and record a warning; the old hardcoded 8GiB fallback could both
    green-light an OOM on a small host and veto a healthy big one)."""
    meminfo = _read_text("/proc/meminfo")
    if meminfo:
        m = re.search(r"MemAvailable:\s+(\d+) kB", meminfo)
        if m:
            return int(m.group(1)) * 1024
    return None


def atomic_write_json(path, obj):
    """FT benchbw.py:682-688 port: tmp file in same dir + os.replace."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# P0 machine discovery
# --------------------------------------------------------------------------

def physical_core_cpus():
    """FT cpu_executor.py:92-116 port: one representative per SMT sibling set,
    intersected with sched_getaffinity. sysfs unreadable -> degenerate safe."""
    allowed = set(affinity_cpus())
    reps = set()
    for cpu in sorted(allowed):
        rep = cpu
        sib = _read_text(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        if sib:
            try:
                sibs = sorted(set(parse_cpulist(sib)) & allowed)
                if sibs:
                    rep = sibs[0]
            except ValueError:
                rep = cpu
        reps.add(rep)
    return sorted(reps)


def discover_numa(phys):
    """node id -> [physical cores on that node]; sysfs cpulist, single synthetic
    node fallback. Container / missing sysfs is loud in meta, never a crash."""
    nodes = {}
    base = "/sys/devices/system/node"
    try:
        names = sorted(n for n in os.listdir(base) if re.fullmatch(r"node\d+", n))
    except OSError:
        names = []
    for name in names:
        cl = _read_text(os.path.join(base, name, "cpulist"))
        if not cl:
            continue
        try:
            cores = sorted(set(parse_cpulist(cl)) & set(phys))
        except ValueError:
            continue
        if cores:
            nodes[int(name[4:])] = cores
    if not nodes:
        nodes = {0: list(phys)}
    return nodes


def discover_gpus(requested):
    """nvidia-smi index topology + sysfs LnkSta/NUMA per BDF. The nvidia-smi index
    order is assumed to match CUDA device order (true on xysa10, no
    CUDA_VISIBLE_DEVICES reorder); the assumption is recorded in meta."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,pci.bus_id,driver_version,memory.used",
        "--format=csv,noheader,nounits",
    ])
    if out is None:
        _die(2, "nvidia-smi unavailable; cannot map GPUs to PCI topology")
    gpus = {}
    for line in out.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 5:
            continue
        idx = int(cols[0])
        bus = cols[2].lower()  # e.g. 00000000:c1:00.0
        domain, rest = bus.split(":", 1)
        bdf = f"{domain[-4:]}:{rest}"  # sysfs form 0000:c1:00.0
        sysdir = f"/sys/bus/pci/devices/{bdf}"
        speed_raw = _read_text(sysdir + "/current_link_speed")   # "16.0 GT/s PCIe"
        width_raw = _read_text(sysdir + "/current_link_width")   # "16"
        numa_raw = _read_text(sysdir + "/numa_node")
        gts, per_lane = None, None
        if speed_raw:
            m = re.match(r"([\d.]+)", speed_raw)
            if m:
                gts = float(m.group(1))
                per_lane = LINK_GBS_PER_LANE_RAW.get(gts)
        width = None
        if width_raw is not None:
            try:
                width = int(width_raw)
            except ValueError:
                width = None
        theo = per_lane * width * PAYLOAD_FACTOR if (per_lane and width) else None
        lo, hi = (BAND_LO_FACTOR * theo, BAND_HI_FACTOR * theo) if theo else FALLBACK_BAND
        numa_node = None
        if numa_raw is not None:
            try:
                n = int(numa_raw)
                numa_node = n if n >= 0 else None
            except ValueError:
                numa_node = None
        gpus[idx] = {
            "index": idx,
            "name": cols[1],
            "bus_id": bus,
            "bdf": bdf,
            "driver": cols[3],
            "mem_used_mib": float(cols[4]),
            "numa_node": numa_node,
            "lnk_gts": gts,
            "lnk_width": width,
            "theoretical_gbs": theo,
            "band_lo": lo,
            "band_hi": hi,
            "band_source": "lnksta" if theo else "unknown-gen",
        }
    missing = [g for g in requested if g not in gpus]
    if missing:
        _die(2, f"requested GPU(s) {missing} not visible to nvidia-smi")
    return [gpus[g] for g in requested]


# --------------------------------------------------------------------------
# startup guard rails (plan verify P5)
# --------------------------------------------------------------------------

def _scan_sglang_processes():
    hits = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return hits
    me = os.getpid()
    for pid in pids:
        if int(pid) == me:
            continue
        cmd = _read_text(f"/proc/{pid}/cmdline")
        if not cmd:
            continue
        flat = cmd.replace("\x00", " ")
        if "sglang" in flat and "bench_bw_kt" not in flat:
            hits.append(int(pid))
    return hits


def guard_rails(cfg, gpus):
    """Refuse to run beside a production server: it would steal DRAM/PCIe and its
    cudaHostRegister page-locking would perturb a capturing graph (plan verify P5)."""
    if cfg["force"]:
        note = "FORCE=1: startup guard rails bypassed by operator"
        _log(note)
        return [note]
    reasons = []
    busy = [g for g in gpus if g["mem_used_mib"] > MEM_GUARD_MIB]
    if busy:
        reasons.append(
            "GPU memory already in use: "
            + "; ".join(f"gpu{g['index']}={g['mem_used_mib']:.0f}MiB" for g in busy)
        )
    procs = _scan_sglang_processes()
    if procs:
        reasons.append(f"sglang-like process(es) alive, pids={procs[:8]}")
    if reasons:
        _die(
            2,
            "guard rails refuse to run alongside a production server "
            "(set SGLANG_KT_BENCHBW_FORCE=1 to override):\n  - " + "\n  - ".join(reasons),
        )
    return []


# --------------------------------------------------------------------------
# host measurement primitives
# --------------------------------------------------------------------------

_AFFINITY_FAILURES = []
_AFFINITY_LOCK = threading.Lock()


def _try_set_affinity(cores):
    """Best-effort sched_setaffinity. Failures are RECORDED (module global, capped
    64) — a silently unpinned leg still produces numbers but they must be branded.
    The P1 remote owner hard-fails on its own pin (see p1_remote_owner_child)."""
    try:
        os.sched_setaffinity(0, set(cores))
        return True
    except (AttributeError, OSError) as e:
        with _AFFINITY_LOCK:
            if len(_AFFINITY_FAILURES) < 64:
                _AFFINITY_FAILURES.append(
                    {"pid": os.getpid(), "cores": list(cores)[:8], "error": repr(e)}
                )
        return False


def cuda_host_register(t, nbytes):
    """Production shape (kt_ep_wrapper.py:891): raise on nonzero rc."""
    rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), int(nbytes), 0)
    if rc != 0:
        raise HarnessFault(f"cudaHostRegister failed rc={rc} for {nbytes} bytes")


def _touch_worker(view, core, errors, idx):
    try:
        _try_set_affinity([core])
        torch.set_num_threads(1)
        view.fill_(0)
    except BaseException as e:  # noqa: BLE001 - worker must never wedge silently
        errors[idx] = repr(e)


def first_touch(view, cores, max_workers=16):
    """Parallel zero-fill first-touch: under the default localalloc policy the
    touching thread's NUMA node owns the page (kt_ep_wrapper.py:838-845 semantics)."""
    cores = list(cores)[:max_workers]
    if not cores:
        cores = [0]
    n = len(cores)
    numel = view.numel()
    base = numel // n
    errors, threads = [None] * n, []
    off = 0
    for i in range(n):
        cnt = base if i < n - 1 else numel - off
        t = threading.Thread(target=_touch_worker, args=(view[off:off + cnt], cores[i], errors, i), daemon=True)
        off += cnt
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    err = next((e for e in errors if e), None)
    if err:
        raise HarnessFault(f"first_touch failed: {err}")


def _bw_worker(barrier, view, core, op, iters, sink, errors, idx):
    try:
        _try_set_affinity([core])
        torch.set_num_threads(1)
        barrier.wait()
        if op == "read":
            acc = 0.0
            for _ in range(iters):
                acc += float(view.sum())
            sink[idx] = acc
        else:
            for _ in range(iters):
                view.fill_(1.0)
            sink[idx] = 1.0
    except BaseException as e:  # noqa: BLE001 - never leave the group barrier short
        errors[idx] = repr(e)
    finally:
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass


def measure_host_bw(view_f32, cores, op, iters, repeats):
    """FT measure_cpu_mem_bw (benchbw.py:184-253) skeleton with a pinned-fill write
    leg added (plan §1.P1: production is a WRITE, verdict (a) reads the write leg).
    view_f32: 1-D float32 over the whole region; workers take equal contiguous
    slices, one physical core each, barrier-released, wall-timed in the caller."""
    n = len(cores)
    numel = view_f32.numel()
    base = numel // n
    chunks, off = [], 0
    for i in range(n):
        cnt = base if i < n - 1 else numel - off
        chunks.append(view_f32[off:off + cnt])
        off += cnt
    nbytes = numel * 4
    rates = []
    for _rep in range(repeats):
        barrier = threading.Barrier(n + 1)
        sink, errors = [0.0] * n, [None] * n
        threads = [
            threading.Thread(
                target=_bw_worker,
                args=(barrier, chunks[i], cores[i], op, iters, sink, errors, i),
                daemon=True,
            )
            for i in range(n)
        ]
        for t in threads:
            t.start()
        barrier.wait()           # release the workers together
        t0 = time.perf_counter()
        barrier.wait()           # all workers finished their loops
        wall = time.perf_counter() - t0
        for t in threads:
            t.join(timeout=5)
        err = next((e for e in errors if e), None)
        if err:
            raise HarnessFault(f"{op} leg worker failed: {err}")
        rates.append(nbytes * iters / wall / 1e9)
    med = statistics.median(rates)
    cv = statistics.pstdev(rates) / med if len(rates) > 1 and med > 0 else 0.0
    return {"gbs_median": med, "gbs_runs": rates, "cv": cv, "bytes": nbytes, "iters": iters, "op": op}


# --------------------------------------------------------------------------
# backing allocation (pin modes)
# --------------------------------------------------------------------------

def _alloc_raw(nbytes, pin):
    """Returns (uint8 flat tensor, shm-or-None). 'shm-register': POSIX shm like
    production :878-887; 'alloc': cudaHostAlloc via pin_memory; 'off': pageable
    (negative-injection leg only, must collapse in P2)."""
    shm = None
    if pin == "shm-register":
        shm = shared_memory.SharedMemory(create=True, size=int(nbytes))
        t = torch.frombuffer(shm.buf, dtype=torch.uint8, count=int(nbytes))
    elif pin == "alloc":
        t = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True)
    elif pin == "off":
        t = torch.empty(int(nbytes), dtype=torch.uint8)
    else:
        raise HarnessFault(f"unknown pin mode {pin!r}")
    return t, shm


def _finish_local_backing(t, shm, nbytes, pin):
    """After first-touch: register, then unlink immediately (no peers in a local
    leg; POSIX keeps the segment alive via our mapping, production :923-929)."""
    if pin == "shm-register":
        cuda_host_register(t, nbytes)
        if shm is not None and hasattr(shm, "unlink"):
            shm.unlink()
    return t, shm


def alloc_backing(nbytes, pin, touch_cores):
    t, shm = _alloc_raw(nbytes, pin)
    first_touch(t, touch_cores)
    return _finish_local_backing(t, shm, nbytes, pin)


# --------------------------------------------------------------------------
# P1: host pinned bandwidth rig, {read,write} x {local,remote} x threads
# --------------------------------------------------------------------------

def p1_remote_owner_child(shm_name, nbytes, owner_cores, register, ev_a, ev_b, ev_done, q, size_divisor=1):
    """Owner side of the production shm handshake (kt_ep_wrapper.py :879-898 ->
    :931-960 -> :923-929), pinned on the owner NUMA node:
        create + first-touch + cudaHostRegister -> Event A
        -> (peer opens) Event B -> unlink -> Event Done -> close/exit.
    size_divisor>1 exists only for the inject-half-shm negative leg."""
    try:
        if owner_cores and not _try_set_affinity(list(owner_cores)[:1]):
            q.put({"ok": False,
                   "error": "owner-node sched_setaffinity failed; the remote-write "
                            "supply leg (verdict-(a) input) would run unpinned — refusing"})
            return
        real_nbytes = int(nbytes) // int(size_divisor)
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=real_nbytes)
        t = None
        try:
            t = torch.frombuffer(shm.buf, dtype=torch.uint8, count=real_nbytes)
            first_touch(t, owner_cores)
            if register:
                cuda_host_register(t, real_nbytes)
            # CPython BufferError guard: the exported torch buffer must die BEFORE
            # any shm.close() below (production precedent kt_ep_wrapper.py:783-785).
            del t
            t = None
        except BaseException:
            del t
            shm.close()
            if hasattr(shm, "unlink"):
                shm.unlink()
            raise
        ev_a.set()
        if not ev_b.wait(timeout=BARRIER_TIMEOUT_S):
            shm.close()
            if hasattr(shm, "unlink"):
                shm.unlink()
            q.put({"ok": False, "error": "timeout waiting for peer open (Event B)"})
            return
        if hasattr(shm, "unlink"):
            shm.unlink()  # production unlink point; peer mapping stays alive
        if not ev_done.wait(timeout=10 * BARRIER_TIMEOUT_S):
            shm.close()
            q.put({"ok": False, "error": "timeout waiting for peer done"})
            return
        shm.close()
        q.put({"ok": True})
    except BaseException as e:  # noqa: BLE001
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]})


def _assert_peer_size(shm, expected):
    """Byte/stride accounting assert, strictly pre-timing (plan §4 injection b:
    a halved peer segment must blow up here, not after minutes of measurement)."""
    if shm.size != expected:
        raise HarnessFault(
            f"byte accounting assert (pre-timing): peer shm size {shm.size} != expected {expected}"
        )


def _q_nowait(q):
    try:
        return q.get_nowait()
    except queue.Empty:
        return None


def run_p1_remote_session(cfg, measurer_cores, owner_cores, sweep, size_divisor=1):
    """One remote (rank0-writes-peer-shm) session: child owns the segment on its
    NUMA node, parent (=rank0 writer) measures read/write sweeps into it."""
    quick = cfg["quick"]
    nbytes = (1 if quick else 2) * 2 ** 30
    ctx = mp.get_context("spawn")
    ev_a, ev_b, ev_done = ctx.Event(), ctx.Event(), ctx.Event()
    q = ctx.Queue()
    shm_name = f"kt_benchbw_p1_{os.getpid()}_{int(time.time() * 1000) % 10 ** 9}"
    child = ctx.Process(
        target=p1_remote_owner_child,
        args=(shm_name, nbytes, owner_cores, cfg["pin"] == "shm-register",
              ev_a, ev_b, ev_done, q, size_divisor),
        daemon=True,
    )
    child.start()
    try:
        # Poll Event A so a child that DIES before posting it (e.g. owner-pin
        # refusal, shm create failure) surfaces its own error instead of the
        # parent stalling the full barrier timeout on a bare "no Event A".
        deadline = time.monotonic() + BARRIER_TIMEOUT_S
        while not ev_a.wait(timeout=0.2):
            msg = _q_nowait(q)
            if isinstance(msg, dict) and msg.get("ok") is False:
                raise HarnessFault(
                    f"P1 remote owner failed before Event A: {msg.get('error')}\n{msg.get('tb', '')}"
                )
            if child.exitcode not in (None, 0):
                raise HarnessFault(
                    f"P1 remote owner died exitcode={child.exitcode} before Event A; "
                    f"child said: {_q_nowait(q)}"
                )
            if time.monotonic() > deadline:
                raise HarnessFault(
                    f"P1 remote owner did not post Event A within {BARRIER_TIMEOUT_S}s; "
                    f"child said: {_q_nowait(q)}"
                )
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        t, view = None, None
        try:
            # Protocol-exact expectation: ALWAYS the full size, never divided —
            # size_divisor is the child's lie; the injection must fire HERE.
            _assert_peer_size(shm, nbytes)
            ev_b.set()
            t = torch.frombuffer(shm.buf, dtype=torch.uint8, count=shm.size)
            view = t.view(torch.float32)
            out = {}
            for op in ("read", "write"):
                leg = {}
                for nth in sweep:
                    leg[str(nth)] = measure_host_bw(
                        view, measurer_cores[:nth], op,
                        iters=3 if quick else 5, repeats=2 if quick else 5,
                    )
                out[op] = leg
            return out
        finally:
            ev_b.set()     # no-op on the success path; releases the child on assert paths
            ev_done.set()
            del view, t    # BufferError guard: exported buffers die before the map closes
            shm.close()
    finally:
        child.join(timeout=BARRIER_TIMEOUT_S)
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
        if child.exitcode not in (0, None):
            msg = _q_nowait(q)
            raise HarnessFault(f"P1 remote owner died exitcode={child.exitcode}; child said: {msg}")


def run_p1(cfg, sysinfo):
    """{read,write} x {local,remote} x {threads} (plan §1.P1). The write leg is the
    verdict-(a) input: production is rank0 cross-NUMA write, not read."""
    quick = cfg["quick"]
    nodes = sysinfo["numa_nodes"]
    mnode = sysinfo["measurer_node"]
    local_cores = nodes[mnode]
    sweep = [n for n in cfg["threads_sweep"] if 0 < n <= len(local_cores)]
    if not sweep:
        sweep = [1]
    dropped = sorted(set(cfg["threads_sweep"]) - set(sweep))
    out = {
        "measurer_node": mnode,
        "threads_sweep": sweep,
        "note": "verdict (a) is judged on the WRITE leg numbers; read kept as DRAM ceiling reference",
        "local": {},
        "remote": {},
        "warnings": ([f"threads_sweep entries dropped (no such cores on node {mnode}): {dropped}"]
                     if dropped else []),
    }
    # ---- local legs: same-process create + first-touch + measure on one node
    nbytes = (1 if quick else 2) * 2 ** 30
    view = None
    t, shm = alloc_backing(nbytes, cfg["pin"], local_cores)
    try:
        view = t.view(torch.float32)
        for op in ("read", "write"):
            leg = {}
            for nth in sweep:
                _log(f"P1 local {op} threads={nth}")
                leg[str(nth)] = measure_host_bw(
                    view, local_cores[:nth], op,
                    iters=3 if quick else 5, repeats=2 if quick else 5,
                )
            out["local"][op] = leg
    finally:
        # BufferError guard: exported buffers (t AND its float32 view) die BEFORE
        # shm.close() (production precedent kt_ep_wrapper.py:783-785).
        del view, t
        if shm is not None:
            shm.close()
    # ---- remote legs: child owns shm on the OTHER node; parent measures into it
    others = [n for n in sorted(nodes) if n != mnode]
    if not others:
        out["remote"] = {"skipped": "single NUMA node visible; remote legs loud-skip"}
        _log("P1 remote loud-skip: single NUMA node")
        return out
    rnode = others[0]
    out["remote_owner_node"] = rnode
    _log(f"P1 remote session (owner node {rnode} -> measurer node {mnode})")
    out["remote"] = run_p1_remote_session(cfg, local_cores, nodes[rnode], sweep)
    return out


# --------------------------------------------------------------------------
# P2: per-expert-shaped H2D gather probe (production copy shape)
# --------------------------------------------------------------------------
# Production shape (kt_ep_wrapper.py:2279-2284): per expert, per RAW bank
# (:1832-1836),   dev_view.copy_(shm_pinned_view, non_blocking=True)
# enqueued back-to-back on the rank's stream from the CanonicalBufferTable.
# P2 recreates exactly that access shape with bank geometry from compute_banks()
# (Σ == expert_bytes asserted at config time) and times the window with CUDA
# events; a perf_counter span cross-checks event/wall skew (self-check >10%).

def _variant_list(nodes):
    vs = [f"node{k}" for k in sorted(nodes)]
    if len(nodes) >= 2:
        vs.append("interleaved")
    return vs


def _alloc_bank_matrix(nrows, row_bytes, variant, nodes, pin):
    """(nrows, row_bytes) uint8 matrix, first-touched into the requested NUMA
    placement (blocks of 8 rows in 'interleaved'), then finished per pin mode.
    Returns (matrix, shm-or-None)."""
    nbytes = nrows * row_bytes
    t, shm = _alloc_raw(nbytes, pin)
    mat = t.view(nrows, row_bytes)
    if pin == "off":
        mat.fill_(0)  # pageable negative-injection leg; placement meaningless
        return mat, shm
    if variant == "interleaved" and len(nodes) >= 2:
        order = sorted(nodes)
        block = 8
        for bi, lo in enumerate(range(0, nrows, block)):
            hi = min(nrows, lo + block)
            first_touch(mat[lo:hi].reshape(-1), nodes[order[bi % len(order)]])
    elif variant.startswith("node"):
        k = int(variant[len("node"):])
        if k not in nodes:
            raise HarnessFault(f"variant {variant!r} names unknown node {k}")
        first_touch(mat.reshape(-1), nodes[k])
    else:
        raise HarnessFault(f"unknown placement variant {variant!r}")
    _finish_local_backing(t, shm, nbytes, pin)
    return mat, shm


def _alloc_gather_buffers(cfg, nmax, variant, nodes, gpu_index, pin):
    """Per-RAW-bank host staging (nmax, bank_bytes) + device mirror.
    host bytes == nmax * expert_bytes exactly (banks asserted at config time)."""
    torch.cuda.set_device(gpu_index)
    host, dev, shms = {}, {}, []
    for name in RAW_BANK_NAMES:
        bs = int(cfg["banks"][name])
        mat, shm = _alloc_bank_matrix(nmax, bs, variant, nodes, pin)
        host[name] = mat
        if shm is not None:
            shms.append(shm)
        dev[name] = torch.empty((nmax, bs), dtype=torch.uint8, device=f"cuda:{gpu_index}")
    return host, dev, shms


def _release_gather_buffers(host, dev, shms):
    dev.clear()
    host.clear()
    for shm in shms:
        try:
            shm.close()
        except Exception:  # noqa: BLE001 - cleanup must never mask the real error
            pass
    shms.clear()
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _window_bytes(cfg, host, n):
    """Byte-identity anchor (plan §1.P2): the gather window moves n x
    expert_bytes exactly, computed from the actual tensors — never assumed."""
    total = sum(int(host[name][:n].numel()) for name in RAW_BANK_NAMES)
    expected = n * int(cfg["expert_bytes"])
    if total != expected:
        raise HarnessFault(
            f"byte accounting assert (pre-timing): window bytes {total} != "
            f"{n} experts x expert_bytes = {expected}"
        )
    return total


def _gather_window(host, dev, n, gpu_index):
    """One window: n experts x 4 RAW banks H2D, enqueued on a side stream.
    Returns (event_ms, wall_ms)."""
    stream = torch.cuda.Stream(device=gpu_index)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        if hasattr(torch.cuda, "_sleep"):
            # Prime on the MEASURED stream (FT benchbw.py:433-434): a long
            # busy-wait kernel keeps this stream draining while the copies
            # below enqueue, so the window times the hardware, not the CPU.
            torch.cuda._sleep(10 ** 7)
        t0 = time.perf_counter()
        start.record()
        for e in range(n):
            for name in RAW_BANK_NAMES:
                dev[name][e].copy_(host[name][e], non_blocking=True)
        end.record()
        end.synchronize()
        wall = time.perf_counter() - t0
    return start.elapsed_time(end), wall * 1e3


def _summarize_runs(runs):
    rates = [r["gbs"] for r in runs]
    med = statistics.median(rates)
    cv = statistics.pstdev(rates) / med if len(rates) > 1 and med > 0 else 0.0
    return {
        "gbs_median": med,
        "gbs_runs": rates,
        "cv": cv,
        "skew_flag": any(r["skew_flag"] for r in runs),
        "runs": runs,
    }


def _measure_gather(cfg, host, dev, n, repeats, warmup, gpu_index):
    nbytes = _window_bytes(cfg, host, n)
    for _ in range(warmup):
        _gather_window(host, dev, n, gpu_index)
    runs = []
    for _ in range(repeats):
        ev_ms, wall_ms = _gather_window(host, dev, n, gpu_index)
        if ev_ms <= 0:
            raise HarnessFault(
                f"CUDA event timing returned {ev_ms} ms for a {nbytes}-byte window "
                f"(n={n} experts) — the rig cannot time gathers (mirrors the "
                "calibration drop-then-fault rule)"
            )
        skew = abs(wall_ms - ev_ms) / ev_ms
        runs.append({
            "bytes": nbytes,
            "event_ms": ev_ms,
            "wall_ms": wall_ms,
            "gbs": nbytes / (ev_ms / 1e3) / 1e9,
            "skew": skew,
            "skew_flag": skew > SKEW_LIMIT,
        })
    out = _summarize_runs(runs)
    out["n_experts"] = n
    out["bytes"] = nbytes
    return out


def _calibrate_pcie(gpu, quick):
    """FT measure_pcie_bw (benchbw.py:256-279) port: pinned buffer, one big
    contiguous H2D copy per iteration, CUDA events, warmup 3, median of iters.
    NOT expert-shaped — the link-reference ceiling for the solo calibration
    band; verdict (c) asserts the gather legs sit inside it."""
    idx = gpu["index"]
    torch.cuda.set_device(idx)
    nbytes = (128 if quick else 256) * 2 ** 20
    iters = 12 if quick else 30
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{idx}")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize(idx)
    rates = []
    for _ in range(iters):
        start.record()
        dev.copy_(host, non_blocking=True)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        if ms > 0:
            rates.append(nbytes / (ms / 1e3) / 1e9)
    if not rates:
        raise HarnessFault(f"PCIe calibration produced no timed runs on gpu{idx}")
    med = statistics.median(rates)
    cv = statistics.pstdev(rates) / med if len(rates) > 1 and med > 0 else 0.0
    del host, dev
    torch.cuda.empty_cache()
    return {"gbs_median": med, "gbs_runs": rates, "cv": cv, "bytes": nbytes, "iters": len(rates)}


def _solo_gather(cfg, gpu, variants, nodes, counts, repeats, warmup):
    """One GPU, all placement variants x expert counts (production shape)."""
    out = {}
    nmax = max(counts)
    for variant in variants:
        host, dev, shms = _alloc_gather_buffers(cfg, nmax, variant, nodes, gpu["index"], cfg["pin"])
        try:
            for n in counts:
                _log(f"P2 solo gpu{gpu['index']} variant={variant} experts={n}")
                out[f"{variant}:{n}"] = _measure_gather(cfg, host, dev, n, repeats, warmup, gpu["index"])
        finally:
            _release_gather_buffers(host, dev, shms)
    return out


def _bwait(barrier, what):
    """Coordination barrier with the plan §4 negative-injection (c) timeout:
    a broken/expired barrier is a HarnessFault, never a silent hang."""
    try:
        barrier.wait(BARRIER_TIMEOUT_S)
    except threading.BrokenBarrierError as e:
        raise HarnessFault(f"coordination barrier broken during {what}: {e}") from e


def _collect_children(q, procs, expect, deadline_s, label):
    """Drain <expect> result dicts off <q> with a hard deadline; a child that
    dies nonzero or wedges past the deadline is a HarnessFault (shared by the
    dual window and the inject-kill negative leg, plan §4)."""
    got = []
    deadline = time.monotonic() + deadline_s
    while len(got) < expect and time.monotonic() < deadline:
        for p in procs:
            if p.exitcode not in (None, 0):
                raise HarnessFault(f"{label}: child pid={p.pid} died exitcode={p.exitcode}")
        item = _q_nowait(q)
        if item is None:
            time.sleep(0.05)
            continue
        if isinstance(item, dict) and item.get("ok") is False:
            raise HarnessFault(
                f"{label}: child reported failure: {item.get('err')}\n{item.get('tb', '')}"
            )
        got.append(item)
    if len(got) < expect:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=10)
        raise HarnessFault(
            f"{label}: collected {len(got)}/{expect} child results within {deadline_s:.0f}s"
        )
    for p in procs:
        p.join(timeout=BARRIER_TIMEOUT_S)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)
            raise HarnessFault(f"{label}: child pid={p.pid} wedged after reporting")
        if p.exitcode != 0:
            raise HarnessFault(f"{label}: child pid={p.pid} exitcode={p.exitcode} after reporting")
    return got


def p2_dual_child(cfg, ginfo, variants, counts, repeats, warmup, barrier, q):
    """One rank of the dual window: own GPU, own staging, all variants x counts.
    Barrier(2) rendezvous BEFORE and AFTER every window, outside the timed
    region, so the two cards' measurement spans coincide (plan §1.P2 dual)."""
    try:
        idx = ginfo["index"]
        torch.cuda.set_device(idx)
        nodes = discover_numa(physical_core_cpus())
        nmax = max(counts)
        out = {}
        for variant in variants:
            host, dev, shms = _alloc_gather_buffers(cfg, nmax, variant, nodes, idx, cfg["pin"])
            try:
                for n in counts:
                    nbytes = _window_bytes(cfg, host, n)
                    runs = []
                    for rep in range(warmup + repeats):
                        _bwait(barrier, f"p2-dual {variant}:{n} rep{rep} pre-window")
                        ev_ms, wall_ms = _gather_window(host, dev, n, idx)
                        _bwait(barrier, f"p2-dual {variant}:{n} rep{rep} post-window")
                        if rep < warmup:
                            continue
                        if ev_ms <= 0:
                            raise HarnessFault(
                                f"CUDA event timing returned {ev_ms} ms on gpu{idx} "
                                f"for a {nbytes}-byte dual window (n={n}) — the rig "
                                "cannot time gathers (mirrors the calibration "
                                "drop-then-fault rule)"
                            )
                        skew = abs(wall_ms - ev_ms) / ev_ms
                        runs.append({
                            "bytes": nbytes,
                            "event_ms": ev_ms,
                            "wall_ms": wall_ms,
                            "gbs": nbytes / (ev_ms / 1e3) / 1e9,
                            "skew": skew,
                            "skew_flag": skew > SKEW_LIMIT,
                        })
                    res = _summarize_runs(runs)
                    res["n_experts"] = n
                    res["bytes"] = nbytes
                    out[f"{variant}:{n}"] = res
            finally:
                _release_gather_buffers(host, dev, shms)
        q.put({"ok": True, "gpu": idx, "results": out,
               "affinity_failures": list(_AFFINITY_FAILURES)})
    except BaseException as e:  # noqa: BLE001 - child reports, parent judges
        q.put({"ok": False, "gpu": ginfo.get("index"), "err": repr(e), "tb": traceback.format_exc()})


def _dram_loader(cores, gib, started, stop, q):
    """CONTENDED=1 leg: pin ALL physical cores and stream pinned DRAM reads for
    the whole dual window. FT measure_overlap_bw (benchbw.py:538-595) lesson:
    standalone numbers cannot predict the contended split — measure it."""
    try:
        pin_ok = _try_set_affinity(cores)
        torch.set_num_threads(max(1, len(cores)))
        buf = torch.ones(gib * 2 ** 28, dtype=torch.float32, pin_memory=True)
        started.set()
        acc, iters = 0.0, 0
        t0 = time.perf_counter()
        while not stop.is_set():
            acc += float(buf.sum())
            iters += 1
        wall = time.perf_counter() - t0
        nbytes = buf.numel() * 4 * iters
        q.put({"ok": True, "gbs": nbytes / wall / 1e9 if wall > 0 else 0.0,
               "iters": iters, "seconds": wall, "sink": acc,
               "affinity_ok": pin_ok})
    except BaseException as e:  # noqa: BLE001
        q.put({"ok": False, "err": repr(e), "tb": traceback.format_exc()})


def _run_dual(cfg, gpus, variants, counts, repeats, warmup, phys, contended=False):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    barrier = ctx.Barrier(2)
    procs = []
    for g in gpus[:2]:
        p = ctx.Process(
            target=p2_dual_child,
            args=(cfg, g, variants, counts, repeats, warmup, barrier, q),
            daemon=True,
        )
        p.start()
        procs.append(p)
    loader, lq, started, stop = None, None, None, None

    def _cleanup_all():
        """Fault-path teardown: never mask the original exception."""
        if stop is not None:
            stop.set()
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            if p.exitcode is None:
                p.join(timeout=10)
        if loader is not None and loader.is_alive():
            loader.terminate()
            loader.join(timeout=10)

    warns = []
    if contended:
        avail = mem_available_bytes()
        loader_gib = 16
        if avail is not None:
            loader_gib = max(4, min(16, (avail // 2 ** 30) // 8))
        else:
            warns.append("MemAvailable unreadable; DRAM loader sized by the 16GiB default")
        lq = ctx.Queue()
        started, stop = ctx.Event(), ctx.Event()
        loader = ctx.Process(
            target=_dram_loader, args=(phys, loader_gib, started, stop, lq), daemon=True
        )
        loader.start()
        # Poll (not a bare wait) so a loader that DIES during startup surfaces
        # its own error instead of stalling the full barrier timeout.
        deadline_l = time.monotonic() + BARRIER_TIMEOUT_S
        while not started.wait(timeout=0.5):
            lmsg = _q_nowait(lq)
            if isinstance(lmsg, dict) and lmsg.get("ok") is False:
                _cleanup_all()
                raise HarnessFault(
                    f"contended DRAM loader failed during startup: {lmsg.get('err')}\n"
                    f"{lmsg.get('tb', '')}"
                )
            if loader.exitcode not in (None, 0):
                _cleanup_all()
                raise HarnessFault(
                    f"contended DRAM loader died exitcode={loader.exitcode} during "
                    f"startup; loader said: {_q_nowait(lq)}"
                )
            if time.monotonic() > deadline_l:
                _cleanup_all()
                raise HarnessFault(
                    f"contended DRAM loader failed to start within {BARRIER_TIMEOUT_S}s; "
                    f"loader said: {_q_nowait(lq)}"
                )
    deadline = max(600.0, 120 + 120.0 * len(variants) * len(counts) * (warmup + repeats))
    try:
        got = _collect_children(q, procs, expect=2, deadline_s=deadline,
                                label="p2-dual-contended" if contended else "p2-dual")
    except BaseException:
        _cleanup_all()
        raise
    lres = None
    if loader is not None:
        stop.set()
        dl = time.monotonic() + 60
        while time.monotonic() < dl:
            lres = _q_nowait(lq)
            if lres is not None:
                break
            time.sleep(0.1)
        loader.join(timeout=15)
        if loader.is_alive():
            loader.terminate()
            loader.join(timeout=10)
            raise HarnessFault("contended DRAM loader wedged on shutdown")
        if loader.exitcode != 0:
            raise HarnessFault(f"contended DRAM loader exitcode={loader.exitcode}")
        if not isinstance(lres, dict) or not lres.get("ok"):
            raise HarnessFault(f"contended DRAM loader did not report: {lres}")
    # normalize to str keys so solo[str(idx)] and by_gpu[str(idx)] line up
    by_gpu = {str(r["gpu"]): r["results"] for r in got}
    aff = [f for r in got for f in (r.get("affinity_failures") or [])]
    if aff:
        warns.append(f"dual children reported {len(aff)} sched_setaffinity failure(s) "
                     f"(unpinned measurement threads): {aff[:4]}")
    if isinstance(lres, dict) and lres.get("affinity_ok") is False:
        warns.append("contended DRAM loader ran UNPINNED (sched_setaffinity failed) — "
                     "its load placement is not the designed full-core spread")
    out = {"by_gpu": by_gpu,
           "note": "per-window Barrier(2) rendezvous; times are event-timed per card",
           "contended": contended}
    if lres is not None:
        out["dram_loader"] = {k: v for k, v in lres.items() if k != "ok"}
    if warns:
        out["warnings"] = warns
    return out


def _granularity_sweep(cfg, gpu, quick):
    """Is per-copy size the limiter? Same total bytes (4 x expert_bytes), copy
    chunk sizes 64KiB..expert_bytes, event-timed, gpu0 only (plan §1.P2-P2g)."""
    idx = gpu["index"]
    torch.cuda.set_device(idx)
    total = 4 * int(cfg["expert_bytes"])
    sizes = [s for s in GRANULARITY_SIZES if s <= total] + [int(cfg["expert_bytes"])]
    sizes = sorted(set(sizes))
    host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(total, dtype=torch.uint8, device=f"cuda:{idx}")
    repeats = 2 if quick else 3
    out = {"total_bytes": total, "note": "link-shape reference; not part of the verdict ratio"}
    try:
        for sz in sizes:
            offs = list(range(0, total, sz))
            n_copies = len(offs)
            rates, enq_walls = [], []
            for rep in range(1 + repeats):  # rep 0 = warmup
                if hasattr(torch.cuda, "_sleep"):
                    # Priming must cover the copy-ENQUEUE span: at tiny chunk
                    # sizes thousands of copy_ calls enqueue slower than the GPU
                    # drains them, and an unprimed window then measures the CPU.
                    torch.cuda._sleep(max(10 ** 7, n_copies * 5 * 10 ** 4))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                te0 = time.perf_counter()
                for lo in offs:
                    hi = min(total, lo + sz)
                    dev[lo:hi].copy_(host[lo:hi], non_blocking=True)
                enq_wall = (time.perf_counter() - te0) * 1e3
                end.record()
                end.synchronize()
                ms = start.elapsed_time(end)
                if rep > 0 and ms > 0:
                    rates.append(total / (ms / 1e3) / 1e9)
                    enq_walls.append(enq_wall)
            med = statistics.median(rates) if rates else 0.0
            cv = statistics.pstdev(rates) / med if len(rates) > 1 and med > 0 else 0.0
            event_med_ms = (total / (med * 1e9)) * 1e3 if med > 0 else 0.0
            enq_med = statistics.median(enq_walls) if enq_walls else 0.0
            out[str(sz)] = {"gbs_median": med, "gbs_runs": rates, "cv": cv,
                            "copies_per_window": n_copies,
                            "enqueue_wall_ms_median": round(enq_med, 3),
                            "enqueue_bound": bool(med > 0 and enq_med > 0.5 * event_med_ms),
                            "timed_runs": len(rates)}
    finally:
        del host, dev
        torch.cuda.empty_cache()
    return out


def run_p2(cfg, gpus, nodes, phys):
    out = {
        "note": "production per-expert H2D shape (kt_ep_wrapper.py:2279-2284); "
                "byte identity asserted pre-timing on every window",
        "pin_mode": cfg["pin"],
        "calibration": {},
        "granularity_gpu0": {},
        "solo": {},
        "dual": {"skipped": "not reached"},
        "dual_contended": {"skipped": "CONTENDED=1 not set"},
        "warnings": [],
    }
    counts = sorted(set(int(n) for n in cfg["expert_counts"]))  # range-validated in read_config
    if not counts:
        _die(2, "EXPERT_COUNTS resolved to an empty sweep")
    repeats = 2 if cfg["quick"] else 5
    warmup = 1
    variants = _variant_list(nodes)
    # staging guards (refuse early, before any alloc). The dual window stages PER
    # CARD in its own child process, and the CONTENDED loader reserves on top —
    # the old single-copy guard green-lit OOMs on exactly those legs.
    need = max(counts) * int(cfg["expert_bytes"])
    dual_factor = 2 if len(gpus) >= 2 else 1
    reserved = dual_factor * need + 2 * 2 ** 30
    if cfg["contended"] and len(gpus) >= 2:
        reserved += 16 * 2 ** 30
    avail = mem_available_bytes()
    if avail is None:
        out["warnings"].append(
            f"MemAvailable unreadable; host RAM guard loud-skipped "
            f"({reserved / 2 ** 30:.1f}GiB incl. dual/loader reservations would be needed)"
        )
    elif avail < reserved:
        _die(2, f"host RAM too small for P2 staging "
                f"({reserved / 2 ** 30:.1f}GiB needed = {dual_factor}x "
                f"{need / 2 ** 30:.1f}GiB + headroom; available {avail / 2 ** 30:.1f}GiB)")
    if cfg["pin"] == "shm-register":
        try:
            st = os.statvfs("/dev/shm")
            shm_avail = st.f_bavail * st.f_bsize
        except OSError:
            shm_avail = None
        shm_need = dual_factor * need + 2 * 2 ** 30
        if shm_avail is None:
            out["warnings"].append("/dev/shm not stat-able; shm capacity guard loud-skipped")
        elif shm_avail < shm_need:
            _die(2, f"/dev/shm too small for shm-register staging "
                    f"({shm_need / 2 ** 30:.1f}GiB needed, {shm_avail / 2 ** 30:.1f}GiB "
                    "available); enlarge the tmpfs (docker --shm-size) or use PIN=alloc")
    for g in gpus[:2]:
        free, _total = torch.cuda.mem_get_info(g["index"])
        if free < need + 2 ** 30:
            _die(2, f"gpu{g['index']} free VRAM too small for P2 mirror "
                    f"({need / 2 ** 30:.1f}GiB + 1GiB headroom needed)")
    for g in gpus:
        _log(f"P2 link calibration gpu{g['index']}")
        out["calibration"][str(g["index"])] = _calibrate_pcie(g, cfg["quick"])
    out["granularity_gpu0"] = _granularity_sweep(cfg, gpus[0], cfg["quick"])
    for g in gpus[:2]:
        _log(f"P2 solo gpu{g['index']}")
        out["solo"][str(g["index"])] = _solo_gather(cfg, g, variants, nodes, counts, repeats, warmup)
    if len(gpus) >= 2:
        _log("P2 dual window (both cards)")
        out["dual"] = _run_dual(cfg, gpus, variants, counts, repeats, warmup, phys, contended=False)
        if cfg["contended"]:
            _log("P2 dual + full-core DRAM contention (CONTENDED=1)")
            try:
                out["dual_contended"] = _run_dual(cfg, gpus, variants, counts, repeats, warmup,
                                                  phys, contended=True)
            except HarnessFault as e:
                # The contended leg is supporting evidence only: its failure must
                # not nuke the base dual window that verdict (a)/(b) reads.
                out["dual_contended"] = {
                    "ok": False, "error": f"HarnessFault: {e}",
                    "note": "contended leg failed; the base dual verdict is unaffected",
                }
    else:
        out["dual"] = {"skipped": "fewer than 2 GPUs requested; dual window unavailable",
                       "loud": True}
    return out


# --------------------------------------------------------------------------
# P3: aggregate host DRAM read knee (feeds verdict note d, not the ratio)
# --------------------------------------------------------------------------

def run_p3(cfg, phys, nodes):
    """Threads = {24,48,72,100}% of physical cores (xysa10: 40/80/120/168);
    buffer = max(2GiB, threads x 128MiB) capped at half of MemAvailable (FT
    measure_cpu_mem_bw sizing, benchbw.py:184-253). Read leg only."""
    counts = sorted(set(max(1, int(len(phys) * f)) for f in P3_KNEE_FRACTIONS))  # truncation, never rounding
    cap = None
    avail = mem_available_bytes()
    if avail is not None:
        cap = max(0, avail // 2)
    out = {
        "physical_cores": len(phys),
        "note": "read-only knee sweep; buffers first-touched spread across nodes",
        "knees": {},
        "warnings": ([] if cap is not None else
                     ["MemAvailable unreadable; P3 buffers uncapped (formula sizing only)"]),
    }
    results = []
    for k in counts:
        need = max(2 * 2 ** 30, k * 128 * 2 ** 20)
        if cfg["quick"]:
            need = max(1 * 2 ** 30, k * 64 * 2 ** 20)
        if cap is not None:
            need = min(need, cap)
        need -= need % (4 * 2 ** 20)  # 4MiB alignment keeps float32 view exact
        if need < 2 ** 28:
            _die(2, "host RAM too small for a meaningful P3 knee sweep")
        t, shm = None, None
        try:
            t, shm = _alloc_raw(need, cfg["pin"])
            if cfg["pin"] == "off":
                t.fill_(0)
            else:
                order = sorted(nodes)
                chunk = need // len(order)
                off = 0
                for i, nk in enumerate(order):
                    cnt = chunk if i < len(order) - 1 else need - off
                    first_touch(t[off:off + cnt], nodes[nk])
                    off += cnt
                _finish_local_backing(t, shm, need, cfg["pin"])
            _log(f"P3 knee threads={k} buffer={need / 2 ** 30:.1f}GiB")
            res = measure_host_bw(t.view(torch.float32), phys[:k], "read",
                                  iters=2 if cfg["quick"] else 3,
                                  repeats=2 if cfg["quick"] else 5)
            res["buffer_bytes"] = need
            out["knees"][str(k)] = res
            results.append((k, res["gbs_median"]))
        finally:
            # BufferError guard: the exported buffer dies BEFORE shm.close().
            del t
            if shm is not None:
                shm.close()
    peak = max((r for _, r in results), default=0.0)
    out["peak_gbs"] = peak
    out["knee_threads"] = next((k for k, r in sorted(results) if r >= 0.95 * peak), None)
    # Reference-only cross-check (verify P7): the in-repo NEON bench measures
    # UNPINNED bf16 torch.ones().sum() best-of-5 — a different style. Record the
    # pointer; a mismatch is expected and is NOT rig-invalid.
    out["cross_check"] = {
        "reference": "kt-kernel/test/per_commit/bench_moe_neon_perf.py bench_bandwidth()",
        "reference_style": "unpinned bf16 4GiB torch.ones().sum(), best-of-5, no thread pinning",
        "note": "style mismatch vs this probe (pinned, physical-core-pinned) is expected; "
                "never gate on the two agreeing",
    }
    return out


# --------------------------------------------------------------------------
# negative-injection legs (plan §4): prove the harness fails loudly
# --------------------------------------------------------------------------

def run_inject_half_shm(cfg, sysinfo):
    """(b) halved peer segment: the byte-accounting assert MUST fire pre-timing
    (peer announces half the protocol size). PASS = HarnessFault with '!='."""
    nodes = sysinfo["numa_nodes"]
    mnode = sysinfo["measurer_node"]
    local_cores = nodes[mnode]
    others = [n for n in sorted(nodes) if n != mnode]
    owner_cores = nodes[others[0]] if others else local_cores
    t0 = time.perf_counter()
    try:
        run_p1_remote_session(cfg, local_cores[:1], owner_cores, [1], size_divisor=2)
    except HarnessFault as e:
        # PASS only when the fault is the DESIGNED one: the pre-timing
        # byte-accounting assert (its message carries "!="). Any other
        # HarnessFault means the wrong failure mode tripped — that is a FAIL.
        ok = "byte accounting assert" in str(e) and "!=" in str(e)
        out = {
            "pass": ok,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "raised": f"{type(e).__name__}: {e}"[:400],
            "expect": "HarnessFault with the pre-timing byte-accounting assert marker ('!=')",
        }
        if not ok:
            out["note"] = ("a HarnessFault fired but NOT the byte-accounting assert — "
                           "the wrong failure mode tripped (injection fidelity broken)")
        return out
    return {
        "pass": False,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "raised": None,
        "expect": "HarnessFault with the pre-timing byte-accounting assert marker ('!=')",
        "note": "assert did NOT fire — byte accounting in the handshake is broken",
    }


def _suicide_child():
    """Dies with exitcode 7 without ever reporting (plan §4 injection (c): a
    rank dying mid-window must trip the exit-code watch, never hang the run)."""
    os._exit(7)


def run_inject_kill(cfg):
    del cfg  # nothing configurable; exits fast by construction
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_suicide_child, daemon=True)
    t0 = time.perf_counter()
    p.start()
    try:
        _collect_children(q, [p], expect=1, deadline_s=60, label="inject-kill")
    except HarnessFault as e:
        return {
            "pass": True,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "raised": f"{type(e).__name__}: {e}"[:400],
            "expect": "HarnessFault from the exit-code watch",
        }
    finally:
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)
    return {
        "pass": False,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "raised": None,
        "expect": "HarnessFault from the exit-code watch",
        "note": "dead child was silently tolerated — HarnessFault discipline broken",
    }


# --------------------------------------------------------------------------
# self-checks + verdict tree
# --------------------------------------------------------------------------

def _collect_leg_warnings(results):
    """Post-walk: surface per-leg "warnings" lists and loud-skip "skipped" dicts
    into one top-level report["warnings"] list (dispatch stays signature-free)."""
    out = []

    def walk(node, path):
        if isinstance(node, dict):
            skipped = node.get("skipped")
            if isinstance(skipped, str):
                out.append(f"{path}: skipped — {skipped}")
            for k, v in node.items():
                if k == "warnings":
                    if isinstance(v, list):
                        out.extend(f"{path}: {w}" for w in v)
                    continue
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(results, "results")
    return out


def collect_self_checks(report, cfg):
    """Post-hoc invariants over the whole results tree (plan §1 self-checks)."""
    cv_bad, skew_bad, byte_bad = [], [], []

    def walk(node, path):
        if isinstance(node, dict):
            if isinstance(node.get("n_experts"), int) and isinstance(node.get("bytes"), int):
                if node["bytes"] != node["n_experts"] * int(cfg["expert_bytes"]):
                    byte_bad.append(path)
            if isinstance(node.get("cv"), (int, float)) and node["cv"] > CV_LIMIT:
                cv_bad.append(path)
            if node.get("skew_flag") is True:
                skew_bad.append(path)
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(report.get("results", {}), "results")
    return {
        "torch_distributed_untouched": not torch.distributed.is_initialized(),
        "byte_identity_violations": byte_bad,   # must be [] (asserts run pre-timing too)
        "cv_violations": cv_bad,                # cv > 5%
        "skew_flags": skew_bad,                 # |wall - event|/event > 10%
    }


def _row_sort_key(rowkey):
    variant, sn = rowkey.rsplit(":", 1)
    return variant, int(sn)


def _dual_ratio_rows(solo, dual_by_gpu):
    """Per (variant, n): dual_sum / mean(card solos). n==1 rows excluded per
    plan §1 verdict tree (1-expert windows are latency, not bandwidth)."""
    cards = [c for c in solo if c in (dual_by_gpu or {})]
    if len(cards) < 2:
        return []
    c0, c1 = cards[0], cards[1]
    keys = set(solo[c0]) & set(solo[c1]) & set(dual_by_gpu[c0]) & set(dual_by_gpu[c1])
    rows = []
    for key in sorted(keys, key=_row_sort_key):
        variant, sn = key.rsplit(":", 1)
        n = int(sn)
        if n == 1:
            continue
        s0 = solo[c0][key]["gbs_median"]
        s1 = solo[c1][key]["gbs_median"]
        d0 = dual_by_gpu[c0][key]["gbs_median"]
        d1 = dual_by_gpu[c1][key]["gbs_median"]
        solo_mean = (s0 + s1) / 2.0
        if solo_mean <= 0:
            continue
        rows.append({
            "variant": variant, "n_experts": n,
            "card_solo_gbs": [s0, s1], "card_dual_gbs": [d0, d1],
            "solo_mean_gbs": solo_mean, "dual_sum_gbs": d0 + d1,
            "ratio": (d0 + d1) / solo_mean,
        })
    return rows


def compute_verdict(report, cfg):
    """Plan §1 judgment tree. Reads ONLY the results tree; never re-measures."""
    res = report.get("results", {})
    p1, p2, p3 = res.get("p1", {}) or {}, res.get("p2", {}) or {}, res.get("p3", {}) or {}
    gpus = ((report.get("meta", {}).get("sysinfo") or {}).get("gpus")) or []
    checks = report.get("self_checks", {}) or {}

    # PIN=off is a negative-injection run, not a verdict run (plan §4 injection a)
    if cfg["pin"] == "off":
        best, measured = 0.0, 0
        for gres in (p2.get("solo") or {}).values():
            for leg in gres.values():
                if isinstance(leg, dict) and leg.get("gbs_median"):
                    best = max(best, leg["gbs_median"])
                    measured += 1
        if measured == 0:
            return {
                "code": "injection-pin-off-no-data",
                "headline": "PIN=off injection run produced NO measured solo gather "
                            "legs — vacuous, cannot judge the collapse",
                "max_gather_gbs": None,
                "collapse_threshold_gbs": PIN_OFF_COLLAPSE_GBS,
                "pass": None,
                "confidence": "n/a",
            }
        return {
            "code": "injection-pin-off",
            "headline": "PIN=off negative-injection run: pageable staging must collapse H2D",
            "max_gather_gbs": round(best, 3),
            "collapse_threshold_gbs": PIN_OFF_COLLAPSE_GBS,
            "pass": best < PIN_OFF_COLLAPSE_GBS,
            "confidence": "n/a",
            "note": "production-rig verdicts require PIN=shm-register (default) or alloc",
        }

    # (c) rig-invalid (plan line 40 / G1 line 82): BOTH the plain large-copy
    # calibration AND each card's largest-n solo gather per variant must sit
    # inside the link band derived from sysfs LnkSta (or the 20-25 fallback,
    # flagged unknown-gen). A solo leg below the band means the rig itself
    # cannot move production-shape loads at link speed (allocation path /
    # IOMMU / NUMA mislabel) — fix the rig first, no verdict.
    solo = p2.get("solo") or {}
    dual = p2.get("dual") or {}
    dual_by_gpu = dual.get("by_gpu") if isinstance(dual, dict) else None
    band_bad = []
    for g in gpus:
        idx = str(g["index"])
        band = [round(g["band_lo"], 3), round(g["band_hi"], 3)]
        cres = (p2.get("calibration") or {}).get(idx)
        if cres and not (g["band_lo"] <= cres["gbs_median"] <= g["band_hi"]):
            band_bad.append({
                "kind": "calibration",
                "gpu": g["index"], "observed_gbs": round(cres["gbs_median"], 3),
                "band": band,
                "band_source": g["band_source"],
            })
        best_per_variant = {}
        for rowkey, leg in (solo.get(idx) or {}).items():
            if not (isinstance(leg, dict) and leg.get("gbs_median")
                    and leg.get("n_experts")):
                continue
            variant = rowkey.rsplit(":", 1)[0]
            cur = best_per_variant.get(variant)
            if cur is None or leg["n_experts"] > cur[0]:
                best_per_variant[variant] = (leg["n_experts"], leg["gbs_median"], rowkey)
        for variant, (n, gbs, rowkey) in sorted(best_per_variant.items()):
            if not (g["band_lo"] <= gbs <= g["band_hi"]):
                band_bad.append({
                    "kind": "solo-gather",
                    "gpu": g["index"], "variant": variant, "n_experts": n,
                    "row": rowkey,
                    "observed_gbs": round(gbs, 3),
                    "band": band,
                    "band_source": g["band_source"],
                })
    if band_bad:
        return {
            "code": "c-rig-invalid",
            "headline": "calibration or a largest-n solo gather leg sits outside the "
                        "LnkSta-derived band — fix the rig (allocation path / IOMMU / "
                        "NUMA mislabel) before any verdict (plan §1 (c))",
            "band_violations": band_bad,
            "confidence": "n/a",
        }

    rows = _dual_ratio_rows(solo, dual_by_gpu)
    sc_conf_low = bool(checks.get("cv_violations")) or bool(checks.get("skew_flags")) \
        or any(g.get("band_source") == "unknown-gen" for g in gpus)
    confidence = "low" if (sc_conf_low or cfg["quick"]) else "normal"
    p3_note = {"knee_threads": p3.get("knee_threads"), "peak_gbs": p3.get("peak_gbs")} if p3 else None

    if not rows:
        skipped = dual.get("skipped") if isinstance(dual, dict) else None
        return {
            "code": "skipped-dual" if skipped else "insufficient-legs",
            "headline": skipped or "no overlapping (variant, n) rows between solo and dual",
            "ratio_table": [], "p3_note": p3_note, "confidence": confidence,
        }

    med_ratio = statistics.median(r["ratio"] for r in rows)
    verdict = {
        "ratio_table": rows,
        "median_ratio": round(med_ratio, 3),
        "p3_note": p3_note,
        "confidence": confidence,
        "production_reference_per_card_gbs": PROD_OBSERVED_PER_CARD_GBS,
    }
    if cfg["quick"]:
        verdict["quick_dev"] = ("QUICK=1: repeats/iters halved — a dev-loop run, "
                                "confidence forced to low")

    # CONTENDED=1 support/deny for an upstream-shared reading (FT lesson:
    # standalone numbers cannot predict the contended split).
    dc = p2.get("dual_contended") or {}
    if isinstance(dc, dict) and dc.get("by_gpu"):
        rows_c = _dual_ratio_rows(solo, dc["by_gpu"])
        if rows_c:
            med_c = statistics.median(r["ratio"] for r in rows_c)
            sum_u = statistics.median(r["dual_sum_gbs"] for r in rows)
            sum_c = statistics.median(r["dual_sum_gbs"] for r in rows_c)
            verdict["contended"] = {
                "median_ratio": round(med_c, 3),
                "dual_sum_median_uncontended_gbs": round(sum_u, 3),
                "dual_sum_median_contended_gbs": round(sum_c, 3),
                "loader": dc.get("dram_loader"),
                "note": "large dual-sum collapse under full-core DRAM load supports a "
                        "shared-upstream (host) reading for verdict (a)",
            }

    if med_ratio <= 1.15:
        # (a) dual cards share an upstream bottleneck. Sub-judgment: can the
        # host cross-NUMA WRITE supply feed 2 x 11 GB/s? Production is rank0
        # writing BOTH ranks' shm (kt_ep_wrapper.py:2240-2241, :2259-2266).
        demand = 2.0 * PROD_OBSERVED_PER_CARD_GBS
        write_legs = ((p1.get("remote") or {}).get("write") or {})
        supply = max((v["gbs_median"] for v in write_legs.values() if v.get("gbs_median")), default=0.0)
        verdict["a_sub"] = {
            "demand_gbs": demand,
            "supply_gbs_remote_write_best": round(supply, 3),
            "remote_write_legs": write_legs or "skipped",
        }
        if supply <= 0:
            verdict["code"] = "a-inconclusive-supply"
            verdict["headline"] = "dual/shared confirmed but P1 remote write legs missing"
        elif supply <= 1.3 * demand:
            verdict["code"] = "a-host-supply"
            verdict["headline"] = (
                f"host cross-NUMA write supply (~{supply:.1f} GB/s) ≈ the 2x11 GB/s "
                "production demand — the gap is HOST SUPPLY into the canonical staging"
            )
        elif supply >= 2.0 * demand:
            verdict["code"] = "a-pcie-link-rc"
            verdict["headline"] = (
                f"host write supply (~{supply:.1f} GB/s) comfortably feeds 2x11 GB/s — "
                "the shared upstream is the PCIe link/root-complex side of TP=2"
            )
        else:
            verdict["code"] = "a-margin"
            verdict["headline"] = (
                f"host write supply (~{supply:.1f} GB/s) sits in the gray band "
                f"[1.3, 2.0] x {demand:.1f} GB/s — suspect both, re-run CONTENDED=1"
            )
    elif med_ratio >= 1.7:
        verdict["code"] = "b-control-plane"
        verdict["headline"] = (
            "dual cards scale near-linearly (median ratio "
            f"{med_ratio:.2f} >= 1.7) — hardware path is fine; suspect the "
            "protocol control plane (per-expert device consensus kt_ep_wrapper.py"
            ":2253/:2273, all_reduce+.item() :2042-2050, gloo :2331 family)"
        )
        prod_ms = cfg.get("prod_ms_per_expert")
        if prod_ms:
            pure = []
            for g in gpus:
                idx = str(g["index"])
                best_per_variant = {}
                for rowkey, leg in (solo.get(idx) or {}).items():
                    if not (isinstance(leg, dict) and leg.get("gbs_median")
                            and leg.get("n_experts")):
                        continue
                    variant = rowkey.rsplit(":", 1)[0]
                    cur = best_per_variant.get(variant)
                    if cur is None or leg["n_experts"] > cur[0]:
                        best_per_variant[variant] = leg
                for variant, leg in sorted(best_per_variant.items()):
                    pure.append({
                        "gpu": g["index"], "variant": variant,
                        "n_experts": leg["n_experts"],
                        "pure_copy_ms_per_expert": round(
                            int(cfg["expert_bytes"]) / (leg["gbs_median"] * 1e9) * 1e3, 4),
                    })
            if pure:
                best = min(p["pure_copy_ms_per_expert"] for p in pure)
                verdict["control_plane_estimate"] = {
                    "legs": pure,
                    "probe_pure_copy_ms_per_expert_best": round(best, 4),
                    "prod_ms_per_expert": prod_ms,
                    "control_plane_overhead_ms_per_expert": round(prod_ms - best, 4),
                    "note": "overhead = prod (SGLANG_KT_BENCHBW_PROD_MS_PER_EXPERT) minus "
                            "the fastest probe pure-copy leg; a positive overhead supports "
                            "(b) — anchors kt_ep_wrapper.py :2253/:2273, :2042-2050, gloo :2331",
                }
        else:
            verdict["control_plane_note"] = (
                "set SGLANG_KT_BENCHBW_PROD_MS_PER_EXPERT (production ms/expert) to "
                "attach the plan §4 control-plane cross-check here"
            )
    else:
        verdict["code"] = "inconclusive-ratio"
        verdict["headline"] = (
            f"median dual/solo ratio {med_ratio:.2f} sits between the (a) and (b) "
            "domains — re-run with more repeats or CONTENDED=1 before judging"
        )
    return verdict


# --------------------------------------------------------------------------
# report + main
# --------------------------------------------------------------------------

def print_report(report, out_path):
    v = report.get("verdict") or {}
    print("\n================ bench_bw_kt verdict ================")
    print(f"json: {out_path}")
    print(f"code: {v.get('code')}   confidence: {v.get('confidence')}")
    print(f"headline: {v.get('headline')}")
    res = report.get("results", {})
    p2 = res.get("p2", {}) or {}
    cal = p2.get("calibration") or {}
    if cal:
        print("calibration (link-reference, FT big-copy): " +
              ", ".join(f"gpu{k}={r['gbs_median']:.1f}GB/s" for k, r in cal.items()))
    p1 = res.get("p1", {}) or {}
    rw = (p1.get("remote") or {}).get("write") if isinstance(p1.get("remote"), dict) else None
    if rw:
        best = max(r["gbs_median"] for r in rw.values())
        print(f"P1 remote write best: {best:.1f} GB/s (production supply leg)")
    rows = v.get("ratio_table") or []
    if rows:
        print("ratio table (dual_sum / mean(solo); n=1 excluded):")
        for r in rows:
            print(f"  {r['variant']:<12} n={r['n_experts']:<4} "
                  f"solo_mean={r['solo_mean_gbs']:6.1f}  dual_sum={r['dual_sum_gbs']:6.1f}  "
                  f"ratio={r['ratio']:.2f}")
        print(f"median ratio: {v.get('median_ratio')}")
    p3n = v.get("p3_note") or {}
    if p3n.get("knee_threads"):
        print(f"P3 knee: threads={p3n['knee_threads']}  peak={p3n.get('peak_gbs'):.1f} GB/s")
    checks = report.get("self_checks") or {}
    bad = {k: val for k, val in checks.items()
           if val not in (True, [], None) and k != "torch_distributed_untouched"}
    if bad:
        print(f"self-check WARNINGS: {bad}")
    for key in ("inject_half_shm", "inject_kill"):
        leg = (report.get("results", {}) or {}).get(key)
        if isinstance(leg, dict) and "pass" in leg:
            status = "PASS" if leg["pass"] is True else (
                "NO-DATA" if leg["pass"] is None else "FAIL")
            line = f"negative-injection {key}: {status} (raised={leg.get('raised')})"
            if leg.get("note"):
                line += f" — {leg['note']}"
            print(line)
    if (v.get("code") or "").startswith("injection-pin-off"):
        print(f"negative-injection pin-off verdict: pass={v.get('pass')} "
              f"(max gather {v.get('max_gather_gbs')} vs collapse threshold "
              f"{v.get('collapse_threshold_gbs')} GB/s)")
    wl = report.get("warnings") or []
    if wl:
        print(f"warnings ({len(wl)}):")
        for w in wl[:12]:
            print(f"  - {w}")
        if len(wl) > 12:
            print(f"  ... +{len(wl) - 12} more (see json)")
    print("=====================================================\n")


def main():
    ap = argparse.ArgumentParser(
        description="bench_bw_kt: host/PCIe verdict probe for the KT per-expert "
                    "gather path. All config via SGLANG_KT_BENCHBW_* env vars.")
    ap.parse_args()  # no options; --help works; unknown args fail loudly
    if platform.system() != "Linux":
        _die(2, "bench_bw_kt is Linux-only (POSIX shm, /sys PCI topology, sched affinity)")
    if not torch.cuda.is_available():
        _die(2, "CUDA unavailable on this host")
    cfg = read_config()
    torch.cuda.init()
    started = time.time()
    report = {
        "schema": SCHEMA,
        "meta": {
            "host": socket.gethostname(),
            "started_epoch": started,
            "argv": sys.argv,
            "sysinfo": None,
            "elapsed_s": None,
        },
        "config": dict(cfg),
        "env_raw": {name: os.environ.get(name) for name in ENV_NAMES},
        "guard_notes": [],
        "warnings": [],
        "results": {},
    }
    try:
        gpus = discover_gpus(cfg["gpus"])
        phys = physical_core_cpus()
        nodes = discover_numa(phys)
        mnode = gpus[0]["numa_node"] if gpus and gpus[0]["numa_node"] in nodes else min(nodes)
        sysinfo = {
            "system": get_system_info(),
            "git": get_git_commit(),
            "gpus": gpus,
            "numa_nodes": nodes,
            "measurer_node": mnode,
            "physical_core_cpus_count": len(phys),
        }
        # CUDA_VISIBLE_DEVICES identity check: the nvidia-smi-index ==
        # CUDA-index assumption (needed to map sysfs LnkSta topology onto the
        # right torch device) holds only when CVD is unset or an identity
        # prefix covering every requested GPU. A filtering/reordering CVD
        # silently mismaps link topology to the wrong card.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        sysinfo["cuda_visible_devices_raw"] = cvd
        assumption = True
        if cvd is not None and cvd.strip():
            vals = [v.strip() for v in cvd.split(",") if v.strip()]
            max_req = max(cfg["gpus"])
            assumption = vals[: max_req + 1] == [str(i) for i in range(max_req + 1)]
            if not assumption:
                _die(2, f"CUDA_VISIBLE_DEVICES={cvd!r} filters/reorders GPUs; the "
                        "nvidia-smi-index == CUDA-index assumption is broken — "
                        "unset it and retry (or request GPUs inside the CVD identity prefix)")
        sysinfo["nvidia_smi_index_equals_cuda_index_assumption"] = assumption
        report["meta"]["sysinfo"] = sysinfo
        report["guard_notes"] = guard_rails(cfg, gpus)
        dispatch = {
            "p1": lambda: run_p1(cfg, sysinfo),
            "p2": lambda: run_p2(cfg, gpus, nodes, phys),
            "p3": lambda: run_p3(cfg, phys, nodes),
            "inject-half-shm": lambda: run_inject_half_shm(cfg, sysinfo),
            "inject-kill": lambda: run_inject_kill(cfg),
        }
        for leg in cfg["only"]:
            _log(f"===== leg {leg} =====")
            report["results"][leg.replace("-", "_")] = dispatch[leg]()
        assert not torch.distributed.is_initialized(), \
            "probe must never join a process group"
        report["warnings"] = _collect_leg_warnings(report["results"])
        if _AFFINITY_FAILURES:
            report["warnings"].append(
                f"main process recorded {len(_AFFINITY_FAILURES)} sched_setaffinity "
                f"failure(s), first: {_AFFINITY_FAILURES[:4]}"
            )
        report["self_checks"] = collect_self_checks(report, cfg)
        report["verdict"] = compute_verdict(report, cfg)
        report["meta"]["elapsed_s"] = round(time.time() - started, 1)
        atomic_write_json(cfg["out"], report)
        print_report(report, cfg["out"])
        v = report.get("verdict") or {}
        if v.get("code") == "c-rig-invalid":
            return 5
        # Negative-injection fidelity: a leg whose pass is strictly False
        # (None = no-data, not counted) fails the whole run as a harness fault.
        failed = [
            f"results.{key}.pass=False"
            for key in ("inject_half_shm", "inject_kill")
            if isinstance(report["results"].get(key), dict)
            and report["results"][key].get("pass") is False
        ]
        if (v.get("code") or "").startswith("injection-pin-off") and v.get("pass") is False:
            failed.append("verdict(injection-pin-off).pass=False")
        if failed:
            _log(f"negative-injection fidelity FAILED -> exit 3: {failed}")
            return 3
        return 0
    except HarnessFault as e:
        report["meta"]["elapsed_s"] = round(time.time() - started, 1)
        report["harness_fault"] = {"error": str(e), "traceback": traceback.format_exc()[-3000:]}
        try:
            atomic_write_json(cfg["out"], report)
        except OSError:
            pass
        print(f"\nHarnessFault: {e}\npartial json: {cfg['out']}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001 - unexpected: still write the partial json
        report["meta"]["elapsed_s"] = round(time.time() - started, 1)
        report["unexpected_fault"] = {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-3000:],
        }
        try:
            atomic_write_json(cfg["out"], report)
        except OSError:
            pass
        traceback.print_exc()
        print(f"unexpected error (treated as harness fault): {e}\n"
              f"partial json: {cfg['out']}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    assert not torch.distributed.is_initialized(), "probe must never join a process group"
    sys.exit(main())
