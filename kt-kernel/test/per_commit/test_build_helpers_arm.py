#!/usr/bin/env python
# coding=utf-8
"""Unit tests for build_helpers.py (arch-aware CMake flag selection).

Pure Python tests - they exercise the AArch64 code paths with an injected
machine name / env dict, so they run on any host.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# kt-kernel/ contains build_helpers.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

pytestmark = pytest.mark.cpu

import build_helpers as bh


class TestArchDetection:
    def test_is_arm_machine_aarch64(self):
        assert bh.is_arm_machine("aarch64") is True

    def test_is_arm_machine_arm64(self):
        assert bh.is_arm_machine("arm64") is True

    def test_is_arm_machine_x86(self):
        assert bh.is_arm_machine("x86_64") is False


class TestCpuFeatureFlags:
    def test_arm_native_default(self):
        assert bh.cpu_feature_flags(env={}, machine="aarch64") == ["-DLLAMA_NATIVE=ON"]

    def test_arm_generic(self):
        flags = bh.cpu_feature_flags(env={"CPUINFER_CPU_INSTRUCT": "GENERIC"}, machine="aarch64")
        assert flags == ["-DLLAMA_NATIVE=OFF"]

    def test_arm_rejects_x86_modes(self):
        # x86-only modes must collapse to NATIVE on AArch64
        for mode in ("AVX2", "AVX512", "FANCY"):
            flags = bh.cpu_feature_flags(env={"CPUINFER_CPU_INSTRUCT": mode}, machine="aarch64")
            assert flags == ["-DLLAMA_NATIVE=ON"], f"mode={mode}"

    def test_x86_modes_pass_through(self):
        flags = bh.cpu_feature_flags(env={"CPUINFER_CPU_INSTRUCT": "AVX2"}, machine="x86_64")
        assert "-DLLAMA_AVX2=ON" in flags
        assert "-DLLAMA_NATIVE=OFF" in flags


class TestArmMcpu:
    def test_no_env_gives_no_args(self):
        assert bh.arm_cmake_args(env={}, machine="aarch64") == []

    @pytest.mark.parametrize("alias,expected", [
        ("ampere1a", "ampere1a"),
        ("ampere1b", "ampere1b"),
        ("neoverse-v2", "neoverse-v2"),
    ])
    def test_known_aliases(self, alias, expected):
        # A non-native -mcpu implies a non-native build, otherwise CMake
        # (LLAMA_NATIVE=ON by default) would silently ignore KT_ARM_CPU.
        assert bh.arm_cmake_args(env={"CPUINFER_ARM_CPU": alias}, machine="aarch64") == [
            f"-DKT_ARM_CPU={expected}",
            "-DLLAMA_NATIVE=OFF",
        ]

    def test_native_alias_keeps_native_build(self):
        assert bh.arm_cmake_args(env={"CPUINFER_ARM_CPU": "native"}, machine="aarch64") == []

    def test_unknown_alias_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported CPUINFER_ARM_CPU"):
            bh.arm_cmake_args(env={"CPUINFER_ARM_CPU": "skylake"}, machine="aarch64")

    def test_ignored_on_x86(self):
        assert bh.arm_cmake_args(env={"CPUINFER_ARM_CPU": "ampere1a"}, machine="x86_64") == []


class TestVariantSelection:
    def test_multi_variant_disabled_on_arm(self):
        assert bh.should_build_all_variants(env={"CPUINFER_BUILD_ALL_VARIANTS": "1"}, machine="aarch64") is False

    def test_multi_variant_enabled_on_x86(self):
        assert bh.should_build_all_variants(env={"CPUINFER_BUILD_ALL_VARIANTS": "1"}, machine="x86_64") is True

    def test_runtime_variants_arm(self):
        assert bh.runtime_variant_names("aarch64") == ["arm"]

    def test_runtime_variants_x86(self):
        assert "amx" in bh.runtime_variant_names("x86_64")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
