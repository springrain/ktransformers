# -*- coding: utf-8 -*-
"""Architecture-aware build helpers for kt-kernel packaging.

This module is import-safe (no setuptools side effects) so both ``setup.py``
and the pytest suite can use it to decide which CMake flags / wheel variant
layout make sense for the build host.

Supported layouts:
  * x86-64  : multi-variant wheels (amx / avx512_* / avx2) driven by the
              existing CPU_FEATURE_MAP entries.
  * AArch64 : a single native variant named ``arm`` (e.g. Ampere Altra /
              Ampere One servers). One binary, tuned with ``-mcpu=native``
              (default), with ``-DKT_ARM_CPU=<cpu>`` for a per-microarch
              portable build, or with GENERIC for a baseline portable build.
"""

from __future__ import annotations

import os
import platform

ARM_ARCH_NAMES = ("aarch64", "arm64")
ARM_VARIANT_NAME = "arm"

# -mcpu values commonly used for Linux AArch64 servers. Keys are lower-cased
# aliases accepted via the CPUINFER_ARM_CPU environment variable.
ARM_MCPU_CHOICES = {
    "native": "native",
    "neoverse-n1": "neoverse-n1",      # Ampere Altra / Altra Max
    "ampere1": "ampere1",              # Ampere One
    "ampere1a": "ampere1a",            # Ampere One (later stepping)
    "ampere1b": "ampere1b",            # Ampere One M / MX family
    "neoverse-n2": "neoverse-n2",
    "neoverse-v1": "neoverse-v1",
    "neoverse-v2": "neoverse-v2",
}


def is_arm_machine(machine: str | None = None) -> bool:
    """Return True when the (build) host is a 64-bit ARM machine."""
    machine = (machine or platform.machine()).lower()
    return machine in ARM_ARCH_NAMES


def arm_mcpu_from_env(env: dict | None = None) -> str | None:
    """Resolve the requested -mcpu value for AArch64 builds.

    Honours CPUINFER_ARM_CPU (e.g. ``ampere1``) and returns the canonical
    -mcpu value, or None when unset. Unknown values are rejected so a typo
    cannot silently produce a host-native, non-portable binary.
    """
    env = os.environ if env is None else env
    raw = (env.get("CPUINFER_ARM_CPU") or "").strip().lower()
    if not raw:
        return None
    try:
        return ARM_MCPU_CHOICES[raw]
    except KeyError as exc:
        choices = ", ".join(sorted(ARM_MCPU_CHOICES))
        raise ValueError(f"Unsupported CPUINFER_ARM_CPU={raw!r}; expected one of: {choices}") from exc


def arm_cmake_args(env: dict | None = None, machine: str | None = None) -> list[str]:
    """Extra CMake args for AArch64 builds (currently just KT_ARM_CPU).

    KT_ARM_CPU is only consulted by CMake in the non-native branch, and the
    build defaults to LLAMA_NATIVE=ON. An explicit (non-native) -mcpu request
    therefore also turns LLAMA_NATIVE off so the choice is not silently
    ignored. Ignored entirely on non-ARM hosts.
    """
    if not is_arm_machine(machine):
        return []
    mcpu = arm_mcpu_from_env(env)
    if not mcpu or mcpu == "native":
        return []
    return [f"-DKT_ARM_CPU={mcpu}", "-DLLAMA_NATIVE=OFF"]


CPU_FEATURE_MAP = {
    "FANCY": "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_AVX512=ON -DLLAMA_AVX512_FANCY_SIMD=ON",
    "AVX512": "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_AVX512=ON",
    "AVX2": "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON",
    "NATIVE": "-DLLAMA_NATIVE=ON",
    # AArch64 aliases: NATIVE remains the default which CMake maps to
    # -mcpu=native; GENERIC produces a portable Armv8-A binary.
    "GENERIC": "-DLLAMA_NATIVE=OFF",
}


def cpu_feature_flags(env: dict | None = None, machine: str | None = None) -> list[str]:
    """Return the -D flags selecting the instruction-set baseline.

    On x86-64 this honours CPUINFER_CPU_INSTRUCT (NATIVE/FANCY/AVX512/AVX2).
    On AArch64 only NATIVE (Ampere-optimised) or GENERIC are meaningful;
    x86-only modes are ignored with a warning printed by the caller.
    """
    env = os.environ if env is None else env
    mode = (env.get("CPUINFER_CPU_INSTRUCT") or "NATIVE").upper()
    if is_arm_machine(machine):
        if mode not in ("NATIVE", "GENERIC"):
            # x86-only modes make no sense here; fall back to native tuning.
            mode = "GENERIC" if mode == "PORTABLE" else "NATIVE"
        flags = CPU_FEATURE_MAP[mode]
    else:
        flags = CPU_FEATURE_MAP.get(mode, CPU_FEATURE_MAP["NATIVE"])
    return [tok for tok in flags.split() if tok]


def should_build_all_variants(env: dict | None = None, machine: str | None = None) -> bool:
    """Multi-variant wheels only make sense for x86-64 hosts."""
    env = os.environ if env is None else env
    if is_arm_machine(machine):
        return False
    val = (env.get("CPUINFER_BUILD_ALL_VARIANTS") or "").strip().lower()
    return val in ("1", "on", "true", "yes")


def runtime_variant_names(machine: str | None = None) -> list[str]:
    """Names of the runtime-loadable extension variants for this arch."""
    if is_arm_machine(machine):
        return [ARM_VARIANT_NAME]
    return ["amx", "avx512_bf16", "avx512_vbmi", "avx512_vnni", "avx512_base", "avx2"]
