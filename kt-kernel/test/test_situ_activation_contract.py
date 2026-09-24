"""Contract and SIMD accuracy tests for the native-MoE SiTU activation."""

from __future__ import annotations

import ast
import copy
import math
import platform
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import torch


KT_KERNEL_ROOT = Path(__file__).resolve().parents[1]
EXPERTS_PATH = KT_KERNEL_ROOT / "python/experts.py"
AMX_WRAPPER_PATH = KT_KERNEL_ROOT / "python/utils/amx.py"
NATIVE_SITU_METHODS = frozenset(
    {
        "RAWINT4",
        "FP8",
        "BF16",
        "FP8_PERCHANNEL",
        "GPTQ_INT4",
        "MXFP4",
        "NVFP4",
        "MXFP8",
    }
)
NATIVE_SWIGLU_METHODS = NATIVE_SITU_METHODS | {"SYCL_GPTQ_INT4"}


class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _AMXRecorder(_Recorder):
    pass


class _NativeRecorder(_Recorder):
    pass


class _LlamafileRecorder(_Recorder):
    pass


class _GeneralRecorder(_Recorder):
    pass


def _compile_factory():
    tree = ast.parse(EXPERTS_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_inference_wrapper"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                copy.deepcopy(function),
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "AMXMoEWrapper": _AMXRecorder,
        "NativeMoEWrapper": _NativeRecorder,
        "LlamafileMoEWrapper": _LlamafileRecorder,
        "GeneralMoEWrapper": _GeneralRecorder,
        "NATIVE_SITU_METHODS": NATIVE_SITU_METHODS,
        "NATIVE_SWIGLU_METHODS": NATIVE_SWIGLU_METHODS,
        "math": math,
    }
    exec(compile(module, str(EXPERTS_PATH), "exec"), namespace)
    return namespace["_create_inference_wrapper"]


def _factory_kwargs(method: str = "MXFP4"):
    return dict(
        layer_idx=0,
        num_experts=1,
        num_experts_per_tok=1,
        hidden_size=128,
        moe_intermediate_size=256,
        gpu_experts_mask=None,
        cpuinfer_threads=1,
        threadpool_count=1,
        weight_path="unused",
        chunked_prefill_size=1,
        cpu_save=False,
        max_deferred_experts_per_token=None,
        method=method,
        pack_all_experts_on_load=False,
        numa_nodes=None,
        swiglu_limit=0.0,
        swiglu_alpha=0.0,
        activation=None,
        situ_beta=None,
        situ_linear_beta=None,
    )


def test_explicit_situ_contract_is_forwarded_without_swiglu_aliases():
    kwargs = _factory_kwargs()
    kwargs.update(activation="situ", situ_beta=4.0, situ_linear_beta=25.0)
    wrapper = _compile_factory()(**kwargs)

    assert wrapper.kwargs["activation"] == "situ"
    assert wrapper.kwargs["situ_beta"] == 4.0
    assert wrapper.kwargs["situ_linear_beta"] == 25.0
    assert wrapper.kwargs["swiglu_alpha"] == 0.0
    assert wrapper.kwargs["swiglu_limit"] == 0.0


def test_zero_situ_linear_beta_disables_the_up_softcap():
    kwargs = _factory_kwargs()
    kwargs.update(activation="situ", situ_beta=4.0, situ_linear_beta=0.0)

    wrapper = _compile_factory()(**kwargs)

    assert wrapper.kwargs["situ_linear_beta"] == 0.0


def test_legacy_k3_tuple_is_translated_to_the_explicit_situ_contract():
    kwargs = _factory_kwargs()
    kwargs.update(swiglu_alpha=4.0, swiglu_limit=25.0)
    wrapper = _compile_factory()(**kwargs)

    assert wrapper.kwargs["activation"] == "situ"
    assert wrapper.kwargs["situ_beta"] == 4.0
    assert wrapper.kwargs["situ_linear_beta"] == 25.0
    assert wrapper.kwargs["swiglu_alpha"] == 0.0
    assert wrapper.kwargs["swiglu_limit"] == 0.0


def test_swiglu_oai_and_v4_silu_clamp_remain_distinct_from_situ():
    factory = _compile_factory()

    oai = _factory_kwargs("MXFP8")
    oai.update(swiglu_alpha=1.702, swiglu_limit=7.0)
    oai_wrapper = factory(**oai)
    assert oai_wrapper.kwargs["activation"] == "swiglu_oai"
    assert oai_wrapper.kwargs["swiglu_alpha"] == pytest.approx(1.702)
    assert oai_wrapper.kwargs["swiglu_limit"] == 7.0
    assert oai_wrapper.kwargs["situ_beta"] is None

    explicit_oai = _factory_kwargs()
    explicit_oai.update(activation="swiglu_oai", swiglu_alpha=4.0, swiglu_limit=25.0)
    explicit_oai_wrapper = factory(**explicit_oai)
    assert explicit_oai_wrapper.kwargs["activation"] == "swiglu_oai"
    assert explicit_oai_wrapper.kwargs["situ_beta"] is None

    v4 = _factory_kwargs("FP8")
    v4.update(swiglu_limit=10.0)
    v4_wrapper = factory(**v4)
    assert v4_wrapper.kwargs["activation"] == "silu"
    assert v4_wrapper.kwargs["swiglu_limit"] == 10.0
    assert v4_wrapper.kwargs["situ_beta"] is None


def test_swiglu_oai_is_forwarded_for_all_shared_native_activation_methods():
    kwargs = _factory_kwargs("BF16")
    kwargs.update(activation="swiglu_oai", swiglu_alpha=1.702)

    wrapper = _compile_factory()(**kwargs)

    assert wrapper.kwargs["activation"] == "swiglu_oai"
    assert wrapper.kwargs["swiglu_alpha"] == pytest.approx(1.702)


def test_llamafile_swiglu_oai_contract_is_forwarded_explicitly():
    kwargs = _factory_kwargs("LLAMAFILE")
    kwargs.update(
        activation="swiglu_oai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )

    wrapper = _compile_factory()(**kwargs)

    assert wrapper.kwargs["activation"] == "swiglu_oai"
    assert wrapper.kwargs["swiglu_alpha"] == pytest.approx(1.702)
    assert wrapper.kwargs["swiglu_limit"] == 7.0


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"activation": "situ"}, "positive situ_beta"),
        ({"activation": "situ", "situ_beta": 0.0}, "positive situ_beta"),
        ({"activation": "situ", "situ_beta": float("nan")}, "positive situ_beta"),
        (
            {"activation": "situ", "situ_beta": 4.0, "situ_linear_beta": -1.0},
            "situ_linear_beta must be finite and non-negative",
        ),
        (
            {"activation": "situ", "situ_beta": 4.0, "swiglu_limit": 25.0},
            "must not be mixed",
        ),
        ({"activation": "silu", "situ_beta": 4.0}, "require activation='situ'"),
        ({"activation": "unknown"}, "Unsupported MoE activation"),
        ({"activation": 2}, "Unsupported MoE activation"),
    ],
)
def test_invalid_situ_contracts_fail_early(updates, error):
    kwargs = _factory_kwargs()
    kwargs.update(updates)
    with pytest.raises(ValueError, match=error):
        _compile_factory()(**kwargs)


@pytest.mark.parametrize("method", sorted(NATIVE_SITU_METHODS))
def test_situ_is_supported_by_shared_native_cpu_activation_bases(method):
    kwargs = _factory_kwargs(method)
    kwargs.update(activation="situ", situ_beta=4.0, situ_linear_beta=25.0)
    wrapper = _compile_factory()(**kwargs)
    assert wrapper.kwargs["method"] == method
    assert wrapper.kwargs["activation"] == "situ"


def test_situ_rejects_backends_that_bypass_the_shared_cpu_activation_bases():
    factory = _compile_factory()
    for method in ("SYCL_GPTQ_INT4", "AMXINT4", "LLAMAFILE", "MOE_INT4"):
        kwargs = _factory_kwargs(method)
        kwargs.update(activation="situ", situ_beta=4.0, situ_linear_beta=25.0)
        with pytest.raises(
            ValueError,
            match="unsupported for method|unsupported by backend|does not support SiTU",
        ):
            factory(**kwargs)


def test_native_wrapper_writes_an_explicit_cpp_activation_contract():
    source = AMX_WRAPPER_PATH.read_text(encoding="utf-8")
    assert (
        'if method == "MXFP4" and swiglu_alpha == 4.0 and swiglu_limit == 25.0:'
        in source
    )
    assert 'activation_types = {"silu": 0, "swiglu_oai": 1, "situ": 2}' in source
    assert (
        "moe_config.activation_type = activation_types[self._activation_type]" in source
    )
    assert "moe_config.situ_beta = self._situ_beta or 0.0" in source
    assert "moe_config.situ_linear_beta = self._situ_linear_beta or 0.0" in source


def test_cpp_activation_contract_has_legacy_auto_and_explicit_validation():
    source = (KT_KERNEL_ROOT / "operators/common.hpp").read_text(encoding="utf-8")
    assert "MOE_ACTIVATION_AUTO = -1" in source
    assert "int activation_type = MOE_ACTIVATION_AUTO" in source
    assert "SwiGLU-OAI requires a finite positive swiglu_alpha" in source
    assert "SiTU requires a finite positive situ_beta" in source
    assert "SiTU situ_linear_beta must be finite and non-negative" in source
    assert "activation_type == MOE_ACTIVATION_SILU ? 0.0f : swiglu_alpha" in source


def test_sft_cpp_entrypoints_fail_fast_for_non_silu_activation():
    common = (KT_KERNEL_ROOT / "operators/common.hpp").read_text(encoding="utf-8")
    assert "void validate_sft_activation() const" in common
    assert "currently supports only plain SiLU activation" in common

    callsites = {
        "amx_sft": KT_KERNEL_ROOT / "operators/amx/sft_moe.hpp",
        "rawint4_sft": KT_KERNEL_ROOT / "operators/amx/sft-k2-moe.hpp",
        "tp_sft": KT_KERNEL_ROOT / "operators/moe-sft-tp.hpp",
    }
    for name, path in callsites.items():
        source = path.read_text(encoding="utf-8")
        assert (
            "make_validated_sft_base_config(config)" in source
            or "config.validate_sft_activation();" in source
        ), name


def test_legacy_cpp_backends_do_not_silently_ignore_activation_contracts():
    llamafile = (KT_KERNEL_ROOT / "operators/llamafile/moe.hpp").read_text(
        encoding="utf-8"
    )
    assert "config_.validate_activation();" in llamafile
    assert "does not support SiTU" in llamafile
    assert "config_.effective_swiglu_alpha()" in llamafile
    assert "gate = fminf(gate, swiglu_limit);" in llamafile

    generic = (KT_KERNEL_ROOT / "operators/moe_kernel/moe.hpp").read_text(
        encoding="utf-8"
    )
    assert "config.validate_activation();" in generic
    assert "Generic MoE kernel supports only plain SiLU activation" in generic


def test_native_situ_method_whitelist_matches_backend_geometry():
    source = AMX_WRAPPER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "NATIVE_SITU_METHODS"
            for target in node.targets
        )
    )
    declared = frozenset(ast.literal_eval(assignment.value.args[0]))
    assert declared == NATIVE_SITU_METHODS
    assert 'elif self.method == "SYCL_GPTQ_INT4"' in source
    sycl_source = (KT_KERNEL_ROOT / "operators/sycl/gptq_int4_sycl-moe.hpp").read_text(
        encoding="utf-8"
    )
    assert "submit_gate_up_decode(" in sycl_source
    assert "does not support SiTU in its fused device activation kernels" in sycl_source
    assert ": Base(validated_sycl_config(config), tp_part_idx_)" in sycl_source
    assert sycl_source.count("config_.effective_swiglu_alpha()") >= 3


def test_all_mxfp4_cpu_callsites_dispatch_situ():
    sources = {
        "amx_base": KT_KERNEL_ROOT / "operators/amx/moe_base.hpp",
        "amx_fp4_direct": KT_KERNEL_ROOT / "operators/amx/fp4-moe.hpp",
        "avx2_base": KT_KERNEL_ROOT / "operators/avx2/moe_base.hpp",
        "neon_base": KT_KERNEL_ROOT / "operators/arm/moe_base.hpp",
    }
    for name, path in sources.items():
        source = path.read_text(encoding="utf-8")
        assert "MOE_ACTIVATION_SITU" in source, name
        assert "situ_fn" in source, name
        if name != "amx_fp4_direct":
            assert "config_.validate_activation();" in source, name

    # ARM's MXFP4 qlen=1 fused path delegates to the same activation block.
    neon_mxfp4 = (KT_KERNEL_ROOT / "operators/arm/mxfp4-moe.hpp").read_text(
        encoding="utf-8"
    )
    assert "Base::apply_activation_block(expert_idx, ith, nth)" in neon_mxfp4


def test_native_situ_whitelist_backends_inherit_a_situ_enabled_base():
    lineages = {
        "RAWINT4_AMX": ("operators/amx/k2-moe.hpp", "public AMX_MOE_BASE"),
        "RAWINT4_AVX2": ("operators/avx2/rawint4-moe.hpp", "public AVX2_MOE_BASE"),
        "FP8_AMX": ("operators/amx/fp8-moe.hpp", "public AMX_MOE_BASE"),
        "FP8_AVX2": ("operators/avx2/fp8-moe.hpp", "public AVX2_MOE_BASE"),
        "FP8_NEON": ("operators/arm/fp8-moe.hpp", "public NEON_MOE_BASE"),
        "BF16_AMX": ("operators/amx/bf16-moe.hpp", "public AMX_MOE_BASE"),
        "BF16_AVX2": ("operators/avx2/bf16-moe.hpp", "public AVX2_MOE_BASE"),
        "BF16_NEON": ("operators/arm/bf16-moe.hpp", "public NEON_MOE_BASE"),
        "FP8_PERCHANNEL": (
            "operators/amx/fp8-perchannel-moe.hpp",
            "public AMX_MOE_BASE",
        ),
        "GPTQ_INT4": ("operators/avx2/gptq_int4-moe.hpp", "public AVX2_MOE_BASE"),
        "MXFP4_AVX2_NVFP4": ("operators/avx2/mxfp4-moe.hpp", "public AVX2_MOE_BASE"),
        "MXFP8_AVX2": ("operators/avx2/mxfp8-moe.hpp", "public AVX2_MOE_BASE"),
        "MXFP8_NEON": ("operators/arm/mxfp8-moe.hpp", "public NEON_MOE_BASE"),
    }
    for name, (relative_path, base) in lineages.items():
        source = (KT_KERNEL_ROOT / relative_path).read_text(encoding="utf-8")
        assert base in source, name


@pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64"), reason="AVX2 host test"
)
def test_avx2_situ_simd_matches_torch_formula(tmp_path):
    capability = torch.backends.cpu.get_cpu_capability().upper()
    if not any(feature in capability for feature in ("AVX2", "AVX512", "AMX")):
        pytest.skip("host CPU does not advertise AVX2")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("an AVX2-capable C++ compiler is not installed")

    # The utility header needs only the BF16 storage type from ggml.h. Keeping
    # this compile oracle self-contained avoids requiring the llama.cpp submodule.
    (tmp_path / "ggml.h").write_text(
        "#pragma once\n#include <cstdint>\nstruct ggml_bf16_t { std::uint16_t bits; };\n",
        encoding="utf-8",
    )
    source = tmp_path / "situ_simd.cpp"
    source.write_text(
        textwrap.dedent(
            """
            #include <iomanip>
            #include <iostream>
            #include <limits>
            #include "avx2_bf16_utils.hpp"

            int main() {
              alignas(32) float gate[8] = {-100.0f, -8.0f, -4.0f, -1.0f, 0.0f, 1.0f, 4.0f, 100.0f};
              alignas(32) float up[8] = {-100.0f, -25.0f, -5.0f, -1.0f, 1.0f, 5.0f, 25.0f, 100.0f};
              alignas(32) float out[8];
              const __m256 g = _mm256_load_ps(gate);
              const __m256 u = _mm256_load_ps(up);
              std::cout << std::setprecision(9);
              for (float linear_beta : {25.0f, 0.0f}) {
                _mm256_storeu_ps(out, avx2::situ_fn(g, u, 4.0f, linear_beta));
                for (float value : out) std::cout << value << ' ';
                std::cout << std::endl;
              }
              _mm256_storeu_ps(out, avx2::act_fn(g, u, 7.0f, 1.702f));
              for (float value : out) std::cout << value << ' ';
              std::cout << std::endl;
              gate[0] = std::numeric_limits<float>::quiet_NaN();
              const __m256 nan_g = _mm256_load_ps(gate);
              _mm256_storeu_ps(out, avx2::situ_fn(nan_g, u, 4.0f, 25.0f));
              for (float value : out) std::cout << value << ' ';
              std::cout << std::endl;
              return 0;
            }
            """
        ),
        encoding="utf-8",
    )
    executable = tmp_path / (
        "situ_simd.exe" if platform.system() == "Windows" else "situ_simd"
    )
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-mavx2",
            "-mfma",
            "-I",
            str(tmp_path),
            "-I",
            str(KT_KERNEL_ROOT / "operators/avx2"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    output_path = tmp_path / "situ_simd.out"
    run_command = (
        ["cmd", "/c", str(executable)]
        if platform.system() == "Windows"
        else [str(executable)]
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        run_result = subprocess.run(
            run_command,
            stdout=output_file,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    assert run_result.returncode == 0, run_result.stderr
    actual = torch.tensor(
        [
            [float(value) for value in line.split()]
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ],
        dtype=torch.float32,
    )

    gate = torch.tensor([-100.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 100.0])
    up = torch.tensor([-100.0, -25.0, -5.0, -1.0, 1.0, 5.0, 25.0, 100.0])
    gate_term = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected_situ = torch.stack(
        (
            gate_term * (25.0 * torch.tanh(up / 25.0)),
            gate_term * up,
        )
    )
    torch.testing.assert_close(actual[:2], expected_situ, rtol=3e-4, atol=3e-4)

    oai_gate = gate.clamp(max=7.0)
    oai_up = up.clamp(min=-7.0, max=7.0)
    expected_oai = oai_gate * torch.sigmoid(oai_gate * 1.702) * (oai_up + 1.0)
    torch.testing.assert_close(actual[2], expected_oai, rtol=5e-4, atol=1e-7)
    assert torch.isnan(actual[3, 0])
