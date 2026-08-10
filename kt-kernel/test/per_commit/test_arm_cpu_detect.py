#!/usr/bin/env python
# coding=utf-8
"""Unit tests for kt_kernel._cpu_detect variant selection (x86 + ARM64).

These tests are pure Python: they exercise the cpuinfo-parsing helpers and
arch dispatch with monkeypatched inputs, so they run on any host without the
native extension. The NEON kernel itself is covered by
test_moe_neon_accuracy_bf16.py (aarch64 only).
"""

import glob
import os
import sys

# Add test/ to path (for ci registration) and python/ (for _cpu_detect)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import pytest

from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

pytestmark = pytest.mark.cpu

import _cpu_detect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpuinfo(flags: str) -> str:
    return (
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        f"flags\t\t: fpu vme de pse tsc {flags}\n"
        "bogomips\t: 5000\n"
    )


FULL_AVX512 = "avx512f avx512bw avx512_vnni avx512_vbmi avx512_bf16"


# ---------------------------------------------------------------------------
# x86 pure-function detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flags,expected",
    [
        (f"{FULL_AVX512} amx_tile amx_int8 amx_bf16", "amx"),
        (FULL_AVX512, "avx512_bf16"),
        ("avx512f avx512bw avx512_vnni avx512_vbmi", "avx512_vbmi"),
        ("avx512f avx512bw avx512_vnni", "avx512_vnni"),
        ("avx512f avx512bw", "avx512_base"),
        ("avx2 f16c fma", "avx2"),
        ("", "avx2"),  # nothing usable -> terminal fallback
        ("sse4_2", "avx2"),
        # Underscore-free flag spellings must be accepted too
        ("avx512f avx512bw avx512vnni avx512vbmi avx512bf16", "avx512_bf16"),
    ],
)
def test_detect_x86_variant_from_cpuinfo(flags, expected):
    assert _cpu_detect._detect_x86_variant_from_cpuinfo(_cpuinfo(flags)) == expected


def test_detect_x86_variant_is_case_insensitive():
    assert _cpu_detect._detect_x86_variant_from_cpuinfo(_cpuinfo("AVX2")) == "avx2"


def test_arm_features_line_is_parsed():
    cpuinfo = (
        "processor\t: 0\n"
        "Features\t: fp asimd evtstrm sve sve2 i8mm bf16 asimddp\n"
    )
    flags = _cpu_detect._cpu_flags_from_cpuinfo(cpuinfo)
    assert {"sve", "sve2", "i8mm", "bf16", "asimddp"} <= flags


# ---------------------------------------------------------------------------
# Arch dispatch: ARM64 hosts always get 'arm'
# ---------------------------------------------------------------------------

def test_arm64_host_returns_arm_variant(monkeypatch):
    monkeypatch.setattr(_cpu_detect, "_host_arch", lambda: "aarch64")
    monkeypatch.delenv("KT_KERNEL_CPU_VARIANT", raising=False)
    assert _cpu_detect.detect_cpu_features() == "arm"


def test_arm64_alias_arm64_returns_arm_variant(monkeypatch):
    monkeypatch.setattr(_cpu_detect, "_host_arch", lambda: "arm64")
    monkeypatch.delenv("KT_KERNEL_CPU_VARIANT", raising=False)
    assert _cpu_detect.detect_cpu_features() == "arm"


def test_arm64_env_override_arm_accepted(monkeypatch):
    monkeypatch.setattr(_cpu_detect, "_host_arch", lambda: "aarch64")
    monkeypatch.setenv("KT_KERNEL_CPU_VARIANT", "arm")
    assert _cpu_detect.detect_cpu_features() == "arm"


def test_arm64_rejects_x86_env_override(monkeypatch):
    monkeypatch.setattr(_cpu_detect, "_host_arch", lambda: "aarch64")
    monkeypatch.setenv("KT_KERNEL_CPU_VARIANT", "avx2")
    assert _cpu_detect.detect_cpu_features() == "arm"


def test_x86_env_override_valid(monkeypatch):
    monkeypatch.setattr(_cpu_detect, "_host_arch", lambda: "x86_64")
    monkeypatch.setenv("KT_KERNEL_CPU_VARIANT", "amx")
    assert _cpu_detect.detect_cpu_features() == "amx"


# ---------------------------------------------------------------------------
# Fallback chain: 'arm' is terminal
# ---------------------------------------------------------------------------

def test_arm_variant_is_terminal(monkeypatch, tmp_path):
    # Pretend no .so files exist anywhere so loading must fail
    monkeypatch.setattr(glob, "glob", lambda pattern: [])
    with pytest.raises(ImportError) as excinfo:
        _cpu_detect.load_extension("arm")
    assert "arm" in str(excinfo.value)


def test_x86_variant_falls_back_then_raises(monkeypatch):
    monkeypatch.setattr(glob, "glob", lambda pattern: [])
    with pytest.raises(ImportError):
        _cpu_detect.load_extension("avx2")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
