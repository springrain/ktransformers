import ctypes
import gc
import glob
import logging
import math
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# Use relative imports for package structure
import kt_kernel_ext.moe as _moe_mod
from kt_kernel_ext.moe import MOEConfig

from ..experts_base import BaseMoEWrapper, _temporary_all_cpu_experts_mask
from .loader import (
    BF16SafeTensorLoader,
    CompressedSafeTensorLoader,
    FP8SafeTensorLoader,
    GPTQSafeTensorLoader,
    MXFP4SafeTensorLoader,
    MXFP8SafeTensorLoader,
    NVFP4SafeTensorLoader,
    SafeTensorLoader,
)

AMXInt4_MOE = getattr(_moe_mod, "AMXInt4_MOE", None)
AMXInt8_MOE = getattr(_moe_mod, "AMXInt8_MOE", None)
AMXInt4_KGroup_MOE = getattr(_moe_mod, "AMXInt4_KGroup_MOE", None)
AMXInt4_KGroupBlocked_MOE = getattr(_moe_mod, "AMXInt4_KGroupBlocked_MOE", None)
AMXFP4_KGroup_MOE = getattr(_moe_mod, "AMXFP4_KGroup_MOE", None)
AMXMXFP8_KGroup_MOE = getattr(_moe_mod, "AMXMXFP8_KGroup_MOE", None)
AMXFP8_MOE = getattr(_moe_mod, "AMXFP8_MOE", None)
AMXBF16_MOE = getattr(_moe_mod, "AMXBF16_MOE", None)
AMXFP8PerChannel_MOE = getattr(_moe_mod, "AMXFP8PerChannel_MOE", None)
AVX2BF16_MOE = getattr(_moe_mod, "AVX2BF16_MOE", None)
AVX2FP8_MOE = getattr(_moe_mod, "AVX2FP8_MOE", None)
AVX2GPTQInt4_MOE = getattr(_moe_mod, "AVX2GPTQInt4_MOE", None)
AVX2RawInt4_MOE = getattr(_moe_mod, "AVX2RawInt4_MOE", None)
AVX2MXFP4_MOE = getattr(_moe_mod, "AVX2MXFP4_MOE", None)
AVX2MXFP8_MOE = getattr(_moe_mod, "AVX2MXFP8_MOE", None)
NEONMXFP4_MOE = getattr(_moe_mod, "NEONMXFP4_MOE", None)
AVXVNNI256GPTQInt4_MOE = getattr(_moe_mod, "AVXVNNI256GPTQInt4_MOE", None)
AVXVNNI256GPTQInt4Packed_MOE = getattr(_moe_mod, "AVXVNNI256GPTQInt4Packed_MOE", None)
AVXVNNI256RawInt4_MOE = getattr(_moe_mod, "AVXVNNI256RawInt4_MOE", None)
NEONFP8_MOE = getattr(_moe_mod, "NEONFP8_MOE", None)
NEONMXFP8_MOE = getattr(_moe_mod, "NEONMXFP8_MOE", None)
NEONBF16_MOE = getattr(_moe_mod, "NEONBF16_MOE", None)
SYCLGPTQInt4_MOE = getattr(_moe_mod, "SYCLGPTQInt4_MOE", None)

_HAS_AMXINT4_SUPPORT = AMXInt4_MOE is not None
_HAS_AMXINT8_SUPPORT = AMXInt8_MOE is not None
_HAS_RAWINT4_SUPPORT = AMXInt4_KGroup_MOE is not None
_HAS_RAWINT4_BLOCKED_SUPPORT = AMXInt4_KGroupBlocked_MOE is not None
_HAS_MXFP4_SUPPORT = AMXFP4_KGroup_MOE is not None
_HAS_MXFP8_SUPPORT = AMXMXFP8_KGroup_MOE is not None
_HAS_FP8_SUPPORT = AMXFP8_MOE is not None
_HAS_BF16_SUPPORT = AMXBF16_MOE is not None
_HAS_FP8_PERCHANNEL_SUPPORT = AMXFP8PerChannel_MOE is not None
_HAS_AVX2_BF16_SUPPORT = AVX2BF16_MOE is not None
_HAS_AVX2_FP8_SUPPORT = AVX2FP8_MOE is not None
_HAS_AVX2_GPTQ_INT4_SUPPORT = AVX2GPTQInt4_MOE is not None
_HAS_AVX2_RAWINT4_SUPPORT = AVX2RawInt4_MOE is not None
_HAS_AVX2_MXFP4_SUPPORT = AVX2MXFP4_MOE is not None
_HAS_AVX2_MXFP8_SUPPORT = AVX2MXFP8_MOE is not None
_HAS_NEON_MXFP4_SUPPORT = NEONMXFP4_MOE is not None
_HAS_AVXVNNI256_GPTQ_INT4_SUPPORT = AVXVNNI256GPTQInt4_MOE is not None
_HAS_AVXVNNI256_PACKED_GPTQ_INT4_SUPPORT = AVXVNNI256GPTQInt4Packed_MOE is not None
_HAS_AVXVNNI256_RAW_INT4_SUPPORT = AVXVNNI256RawInt4_MOE is not None
_HAS_NEON_FP8_SUPPORT = NEONFP8_MOE is not None
_HAS_NEON_MXFP8_SUPPORT = NEONMXFP8_MOE is not None
_HAS_NEON_BF16_SUPPORT = NEONBF16_MOE is not None
_HAS_SYCL_GPTQ_INT4_SUPPORT = SYCLGPTQInt4_MOE is not None
_AVXVNNI256_GPTQ_INT4_MAX_GROUP_SIZE = 256
_AVXVNNI256_PACKED_GPTQ_INT4_MAX_GROUP_SIZE = 2048
_AVXVNNI256_RAW_INT4_MAX_GROUP_SIZE = 256

# Native CPU formats whose gate/up outputs all flow through the shared
# AMX_MOE_BASE, AVX2_MOE_BASE, or NEON_MOE_BASE activation contract. SYCL is
# deliberately excluded because its fused device kernels apply activation
# internally and currently accept only the legacy swiglu fields.
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
# SYCL has its own fused activation kernels: they implement the legacy
# SiLU/SwiGLU-OAI alpha+clamp contract, but not SiTU.
NATIVE_SWIGLU_METHODS = NATIVE_SITU_METHODS | {"SYCL_GPTQ_INT4"}


def _validate_block_fp8_layout(
    weights,
    scales,
    projection: str,
    expected_weight_shape,
    block_size: int = 128,
) -> None:
    """Reject layouts that the block-FP8 kernels would otherwise misinterpret."""
    if len(weights) != len(scales):
        raise ValueError(
            f"FP8 {projection}: weight/scale expert counts differ "
            f"({len(weights)} != {len(scales)})"
        )
    for expert_id, (weight, scale) in enumerate(zip(weights, scales)):
        if (
            weight.ndim != 2
            or weight.element_size() != 1
            or weight.dtype not in (torch.uint8, torch.float8_e4m3fn)
        ):
            raise ValueError(
                f"FP8 {projection} expert {expert_id}: expected a 2-D, one-byte E4M3FN "
                f"weight tensor, got shape={tuple(weight.shape)}, dtype={weight.dtype}"
            )
        if tuple(weight.shape) != tuple(expected_weight_shape):
            raise ValueError(
                f"FP8 {projection} expert {expert_id}: expected weight shape "
                f"{tuple(expected_weight_shape)}, got {tuple(weight.shape)}"
            )
        expected = (
            (int(weight.shape[0]) + block_size - 1) // block_size,
            (int(weight.shape[1]) + block_size - 1) // block_size,
        )
        if scale.dtype != torch.float32 or scale.ndim != 2 or tuple(scale.shape) != expected:
            raise ValueError(
                f"FP8 {projection} expert {expert_id}: NEON/AVX block FP8 requires "
                f"a float32 {block_size}x{block_size} scale tensor with shape {expected}, "
                f"got shape={tuple(scale.shape)}, dtype={scale.dtype}. "
                "Per-channel FP8 is a different format."
            )


def _validate_mxfp8_layout(
    weights,
    scales,
    projection: str,
    expected_weight_shape,
    group_size: int = 32,
) -> None:
    """Validate native MXFP8 E4M3FN + row-wise UE8M0 group scales."""
    if len(weights) != len(scales):
        raise ValueError(
            f"MXFP8 {projection}: weight/scale expert counts differ "
            f"({len(weights)} != {len(scales)})"
        )
    for expert_id, (weight, scale) in enumerate(zip(weights, scales)):
        if weight.dtype != torch.uint8 or weight.ndim != 2 or weight.element_size() != 1:
            raise ValueError(
                f"MXFP8 {projection} expert {expert_id}: expected a 2-D, one-byte E4M3FN "
                f"weight tensor, got shape={tuple(weight.shape)}, dtype={weight.dtype}"
            )
        if tuple(weight.shape) != tuple(expected_weight_shape):
            raise ValueError(
                f"MXFP8 {projection} expert {expert_id}: expected weight shape "
                f"{tuple(expected_weight_shape)}, got {tuple(weight.shape)}"
            )
        n, k = (int(weight.shape[0]), int(weight.shape[1]))
        if k % group_size != 0:
            raise ValueError(
                f"MXFP8 {projection} expert {expert_id}: K={k} must be divisible by "
                f"group_size={group_size}"
            )
        expected = (n, k // group_size)
        if scale.dtype != torch.uint8 or scale.ndim != 2 or tuple(scale.shape) != expected:
            raise ValueError(
                f"MXFP8 {projection} expert {expert_id}: expected uint8 UE8M0 scales "
                f"with shape {expected}, got shape={tuple(scale.shape)}, dtype={scale.dtype}"
            )


def _validate_mxfp4_layout(
    weights,
    scales,
    projection: str,
    expected_weight_shape,
    group_size: int = 32,
) -> None:
    """Validate native nibble-packed MXFP4 weights and BF16 UE8M0 scales."""
    if len(weights) != len(scales):
        raise ValueError(
            f"MXFP4 {projection}: weight/scale expert counts differ "
            f"({len(weights)} != {len(scales)})"
        )
    for expert_id, (weight, scale) in enumerate(zip(weights, scales)):
        if weight.dtype != torch.uint8 or weight.ndim != 2 or weight.element_size() != 1:
            raise ValueError(
                f"MXFP4 {projection} expert {expert_id}: expected a 2-D packed uint8 "
                f"weight tensor, got shape={tuple(weight.shape)}, dtype={weight.dtype}"
            )
        expected = (int(expected_weight_shape[0]), int(expected_weight_shape[1]) // 2)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"MXFP4 {projection} expert {expert_id}: expected packed weight shape "
                f"{expected} for logical shape {tuple(expected_weight_shape)}, "
                f"got {tuple(weight.shape)}"
            )
        n, packed_k = (int(weight.shape[0]), int(weight.shape[1]))
        if (packed_k * 2) % group_size != 0:
            raise ValueError(
                f"MXFP4 {projection} expert {expert_id}: logical K={packed_k * 2} "
                f"must be divisible by group_size={group_size}"
            )
        expected_scale = (n, packed_k * 2 // group_size)
        if scale.dtype != torch.bfloat16 or scale.ndim != 2 or tuple(scale.shape) != expected_scale:
            raise ValueError(
                f"MXFP4 {projection} expert {expert_id}: expected BF16 UE8M0 scales "
                f"with shape {expected_scale}, got shape={tuple(scale.shape)}, dtype={scale.dtype}"
            )


def _validate_expert_mapping(
    physical_to_logical_map_cpu: torch.Tensor,
    expected_physical_experts: int,
    projection_weights,
) -> int:
    """Validate EPLB mapping without conflating physical and logical experts."""
    if not isinstance(physical_to_logical_map_cpu, torch.Tensor):
        raise TypeError("physical_to_logical_map_cpu must be a torch.Tensor")
    if physical_to_logical_map_cpu.device.type != "cpu":
        raise ValueError("physical_to_logical_map_cpu must reside on CPU")
    if physical_to_logical_map_cpu.dtype != torch.int64:
        raise ValueError(
            "physical_to_logical_map_cpu must use torch.int64 because the native "
            f"backend reads uint64 IDs, got {physical_to_logical_map_cpu.dtype}"
        )
    if physical_to_logical_map_cpu.ndim != 1:
        raise ValueError(
            "physical_to_logical_map_cpu must be one-dimensional, got shape="
            f"{tuple(physical_to_logical_map_cpu.shape)}"
        )
    if not physical_to_logical_map_cpu.is_contiguous():
        raise ValueError("physical_to_logical_map_cpu must be contiguous")
    if physical_to_logical_map_cpu.numel() != expected_physical_experts:
        raise ValueError(
            "physical_to_logical_map_cpu length must match the native physical "
            f"expert count ({physical_to_logical_map_cpu.numel()} != "
            f"{expected_physical_experts})"
        )

    logical_counts = {name: len(weights) for name, weights in projection_weights.items()}
    if not logical_counts or len(set(logical_counts.values())) != 1:
        raise ValueError(
            "FP8/MXFP4/MXFP8 expert projections contain different logical expert counts: "
            f"{logical_counts}"
        )
    logical_experts = next(iter(logical_counts.values()))
    if logical_experts <= 0:
        raise ValueError("FP8/MXFP8 checkpoint contains no logical experts")

    if physical_to_logical_map_cpu.numel():
        min_id = int(physical_to_logical_map_cpu.min().item())
        max_id = int(physical_to_logical_map_cpu.max().item())
        if min_id < 0 or max_id >= logical_experts:
            raise ValueError(
                "physical_to_logical_map_cpu contains an out-of-range logical "
                f"expert ID: range=[{min_id}, {max_id}], checkpoint experts="
                f"{logical_experts}"
            )
    return logical_experts


def _host_has_cpu_flag(*flag_names: str) -> bool:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].strip().split())
                    return any(name in flags for name in flag_names)
    except OSError:
        return False
    return False


_HOST_HAS_AVX_VNNI = _host_has_cpu_flag("avx_vnni", "avxvnni")


def _preflight_sycl_device() -> None:
    """Report the common Linux render-node permission error before C++ initialization."""
    if os.getenv("ONEAPI_DEVICE_SELECTOR", "").strip():
        return

    render_nodes = sorted(glob.glob("/dev/dri/renderD*"))
    if render_nodes and not any(os.access(path, os.R_OK | os.W_OK) for path in render_nodes):
        raise RuntimeError(
            "SYCL_GPTQ_INT4 selects a GPU by default, but the current user cannot access "
            "/dev/dri/renderD*. Add the user to the render group and re-login, or set "
            "ONEAPI_DEVICE_SELECTOR to an accessible SYCL GPU."
        )


def _supports_avxvnni256_gptq_int4_group_size(group_size: Optional[int]) -> bool:
    if group_size is None:
        return True
    return group_size > 0 and group_size % 32 == 0 and group_size <= _AVXVNNI256_GPTQ_INT4_MAX_GROUP_SIZE


def _supports_avxvnni256_packed_gptq_int4_group_size(group_size: Optional[int]) -> bool:
    if group_size is None:
        return True
    return group_size > 0 and group_size % 32 == 0 and group_size <= _AVXVNNI256_PACKED_GPTQ_INT4_MAX_GROUP_SIZE


def _supports_avxvnni256_rawint4_group_size(group_size: Optional[int]) -> bool:
    if group_size is None:
        return True
    return group_size > 0 and group_size % 32 == 0 and group_size <= _AVXVNNI256_RAW_INT4_MAX_GROUP_SIZE


def _select_gptq_int4_backend(group_size: Optional[int] = None):
    forced = os.getenv("KT_GPTQ_INT4_BACKEND", "").strip().lower()
    avxvnni_group_supported = _supports_avxvnni256_gptq_int4_group_size(group_size)
    packed_group_supported = _supports_avxvnni256_packed_gptq_int4_group_size(group_size)

    # The packed backend keeps the int4 weights resident (~0.56 B/weight incl.
    # scales + c2) instead of pre-unpacking them into an int8 [N, K] copy
    # (~1.05 B/weight), so it is preferred whenever it is available. Shape
    # limits beyond group_size (the k % group_size / k % 8 alignment) are
    # enforced by the C++ BufferA/BufferB init with the real config; there is
    # no hidden/intermediate size cap (the kernel staging auto-sizes).
    if forced in {"packed", "avxvnni-packed", "avxvnni256-packed"}:
        if not _HAS_AVXVNNI256_PACKED_GPTQ_INT4_SUPPORT:
            raise RuntimeError(
                "KT_GPTQ_INT4_BACKEND=packed requested, but AVXVNNI256GPTQInt4Packed_MOE is not compiled in."
            )
        if not _HOST_HAS_AVX_VNNI:
            raise RuntimeError("KT_GPTQ_INT4_BACKEND=packed requested, but the current CPU does not support avx_vnni.")
        if not packed_group_supported:
            raise RuntimeError(
                "KT_GPTQ_INT4_BACKEND=packed requested, but "
                f"group_size={group_size} is unsupported. The packed AVX-VNNI-256 GPTQ_INT4 backend supports "
                f"positive multiples of 32 up to {_AVXVNNI256_PACKED_GPTQ_INT4_MAX_GROUP_SIZE}."
            )
        return AVXVNNI256GPTQInt4Packed_MOE

    if forced in {"avxvnni", "avxvnni256"}:
        # Escape hatch: keep the pre-unpacked (int8-resident) AVX-VNNI backend.
        if not _HAS_AVXVNNI256_GPTQ_INT4_SUPPORT:
            raise RuntimeError("KT_GPTQ_INT4_BACKEND=avxvnni requested, but AVXVNNI256GPTQInt4_MOE is not compiled in.")
        if not _HOST_HAS_AVX_VNNI:
            raise RuntimeError("KT_GPTQ_INT4_BACKEND=avxvnni requested, but the current CPU does not support avx_vnni.")
        if not avxvnni_group_supported:
            raise RuntimeError(
                "KT_GPTQ_INT4_BACKEND=avxvnni requested, but "
                f"group_size={group_size} is unsupported. AVX-VNNI-256 GPTQ_INT4 only supports "
                f"positive multiples of 32 up to {_AVXVNNI256_GPTQ_INT4_MAX_GROUP_SIZE}."
            )
        return AVXVNNI256GPTQInt4_MOE

    if forced == "avx2":
        if not _HAS_AVX2_GPTQ_INT4_SUPPORT:
            raise RuntimeError("KT_GPTQ_INT4_BACKEND=avx2 requested, but AVX2GPTQInt4_MOE is not compiled in.")
        return AVX2GPTQInt4_MOE

    if _HAS_AVXVNNI256_PACKED_GPTQ_INT4_SUPPORT and _HOST_HAS_AVX_VNNI and packed_group_supported:
        return AVXVNNI256GPTQInt4Packed_MOE
    if _HAS_AVXVNNI256_GPTQ_INT4_SUPPORT and _HOST_HAS_AVX_VNNI and avxvnni_group_supported:
        return AVXVNNI256GPTQInt4_MOE
    if _HAS_AVX2_GPTQ_INT4_SUPPORT:
        return AVX2GPTQInt4_MOE
    return None


def _select_rawint4_backend(group_size: Optional[int] = None):
    forced = os.getenv("KT_RAWINT4_BACKEND", "").strip().lower()
    avxvnni_group_supported = _supports_avxvnni256_rawint4_group_size(group_size)

    if forced == "amx":
        if not _HAS_RAWINT4_SUPPORT:
            raise RuntimeError("KT_RAWINT4_BACKEND=amx requested, but AMXInt4_KGroup_MOE is not compiled in.")
        return AMXInt4_KGroup_MOE

    if forced in {"amx_blocked", "blocked"}:
        if not _HAS_RAWINT4_BLOCKED_SUPPORT:
            raise RuntimeError(
                "KT_RAWINT4_BACKEND=amx_blocked requested, but AMXInt4_KGroupBlocked_MOE is not compiled in."
            )
        return AMXInt4_KGroupBlocked_MOE

    if forced in {"avxvnni", "avxvnni256"}:
        if not _HAS_AVXVNNI256_RAW_INT4_SUPPORT:
            raise RuntimeError("KT_RAWINT4_BACKEND=avxvnni requested, but AVXVNNI256RawInt4_MOE is not compiled in.")
        if not _HOST_HAS_AVX_VNNI:
            raise RuntimeError("KT_RAWINT4_BACKEND=avxvnni requested, but the current CPU does not support avx_vnni.")
        if not avxvnni_group_supported:
            raise RuntimeError(
                "KT_RAWINT4_BACKEND=avxvnni requested, but "
                f"group_size={group_size} is unsupported. AVX-VNNI-256 RAWINT4 only supports "
                f"positive multiples of 32 up to {_AVXVNNI256_RAW_INT4_MAX_GROUP_SIZE}."
            )
        return AVXVNNI256RawInt4_MOE

    if forced == "avx2":
        if not _HAS_AVX2_RAWINT4_SUPPORT:
            raise RuntimeError("KT_RAWINT4_BACKEND=avx2 requested, but AVX2RawInt4_MOE is not compiled in.")
        return AVX2RawInt4_MOE

    if _HAS_RAWINT4_SUPPORT:
        return AMXInt4_KGroup_MOE
    if _HAS_AVXVNNI256_RAW_INT4_SUPPORT and _HOST_HAS_AVX_VNNI and avxvnni_group_supported:
        return AVXVNNI256RawInt4_MOE
    if _HAS_AVX2_RAWINT4_SUPPORT:
        return AVX2RawInt4_MOE
    return None


def _select_mxfp4_backend():
    """Select MXFP4 backend: AMX > AVX2 > ARM NEON.

    Override with KT_MXFP4_BACKEND=amx|avx2|neon.
    Returns None if no MXFP4 backend is available.
    """
    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()

    if forced == "amx":
        if not _HAS_MXFP4_SUPPORT:
            raise RuntimeError(
                "KT_MXFP4_BACKEND=amx requested, but AMXFP4_KGroup_MOE is not compiled in. "
                "Recompile with AVX512F + AVX512BW + AVX512_BF16 enabled."
            )
        return AMXFP4_KGroup_MOE

    if forced == "avx2":
        if not _HAS_AVX2_MXFP4_SUPPORT:
            raise RuntimeError(
                "KT_MXFP4_BACKEND=avx2 requested, but AVX2MXFP4_MOE is not compiled in. "
                "Recompile with AVX2 + FMA enabled."
            )
        return AVX2MXFP4_MOE

    if forced == "neon":
        if not _HAS_NEON_MXFP4_SUPPORT:
            raise RuntimeError(
                "KT_MXFP4_BACKEND=neon requested, but NEONMXFP4_MOE is not "
                "compiled in. Rebuild kt-kernel on AArch64."
            )
        return NEONMXFP4_MOE

    if _HAS_MXFP4_SUPPORT:
        return AMXFP4_KGroup_MOE
    if _HAS_AVX2_MXFP4_SUPPORT:
        return AVX2MXFP4_MOE
    if _HAS_NEON_MXFP4_SUPPORT:
        return NEONMXFP4_MOE
    return None


def _select_mxfp8_backend():
    """Select MXFP8 backend: AMX/AVX-512 > AVX2 > ARM NEON.

    Override with KT_MXFP8_BACKEND=amx|avx2|neon.
    Returns None if no MXFP8 backend is available.
    """
    forced = os.getenv("KT_MXFP8_BACKEND", "").strip().lower()

    if forced == "amx":
        if not _HAS_MXFP8_SUPPORT:
            raise RuntimeError(
                "KT_MXFP8_BACKEND=amx requested, but AMXMXFP8_KGroup_MOE is not compiled in. "
                "Recompile with AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI enabled."
            )
        if not _host_has_cpu_flag("amx_tile", "amx_bf16"):
            raise RuntimeError(
                "KT_MXFP8_BACKEND=amx requested, but the host CPU lacks AMX (amx_tile / amx_bf16). "
                "This would SIGILL at first forward. Unset the env to fall back to AVX2."
            )
        return AMXMXFP8_KGroup_MOE

    if forced == "avx2":
        if not _HAS_AVX2_MXFP8_SUPPORT:
            raise RuntimeError(
                "KT_MXFP8_BACKEND=avx2 requested, but AVX2MXFP8_MOE is not compiled in. "
                "Recompile with AVX2 + FMA enabled."
            )
        return AVX2MXFP8_MOE

    if forced == "neon":
        if not _HAS_NEON_MXFP8_SUPPORT:
            raise RuntimeError(
                "KT_MXFP8_BACKEND=neon requested, but NEONMXFP8_MOE is not compiled in. "
                "Recompile on AArch64 with the ARM backend enabled."
            )
        return NEONMXFP8_MOE

    # Auto-select: prefer AMX iff the .so was built with it AND the runtime CPU has AMX.
    # Compile-time-only check would SIGILL on AVX-512 CPUs lacking AMX (pre-Sapphire Rapids).
    if _HAS_MXFP8_SUPPORT and _host_has_cpu_flag("amx_tile", "amx_bf16"):
        return AMXMXFP8_KGroup_MOE
    if _HAS_AVX2_MXFP8_SUPPORT:
        return AVX2MXFP8_MOE
    if _HAS_NEON_MXFP8_SUPPORT:
        return NEONMXFP8_MOE
    return None


class AMXMoEWrapper(BaseMoEWrapper):
    """
    AMX-based MoE wrapper implementation.
    Supports AMXINT4 and AMXINT8 quantization methods.
    """

    _safetensor_loader_instance = None  # Singleton SafeTensorLoader

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "AMXINT4",
        numa_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize AMX MoE Wrapper.

        Args:
            layer_idx: Layer index
            num_experts: Total number of experts
            num_experts_per_tok: Number of experts per token (top-k)
            hidden_size: Hidden dimension size
            moe_intermediate_size: MoE intermediate size
            gpu_experts_mask: Boolean mask indicating which experts are on GPU.
                              Shape: [num_experts], dtype: torch.bool.
                              mask[i] = True means expert i is on GPU.
                              If None, all experts are on CPU.
            cpuinfer_threads: Number of CPU inference threads
            threadpool_count: Number of NUMA subpools
            weight_path: Path to AMX weights (SafeTensor format)
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of experts per token to defer. Defaults to 0.
            method: AMX quantization method ("AMXINT4" or "AMXINT8")
        """
        if method == "AMXINT4" and not _HAS_AMXINT4_SUPPORT:
            raise RuntimeError(
                "AMXINT4 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW (VNNI optional)\n"
                "Please recompile kt_kernel_ext with AVX512 enabled."
            )
        if method == "AMXINT8" and not _HAS_AMXINT8_SUPPORT:
            raise RuntimeError(
                "AMXINT8 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW (VNNI optional)\n"
                "Please recompile kt_kernel_ext with AVX512 enabled."
            )

        # Initialize base class
        super().__init__(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
            numa_nodes=numa_nodes,
        )

        # AMX-specific: Check if we should load merged safetensor weights
        self.load_merged_weight = False
        import glob

        if glob.glob(os.path.join(weight_path, "*.safetensors")):
            self.load_merged_weight = True

        # Initialize SafeTensor loader (singleton)
        if self.load_merged_weight:
            if AMXMoEWrapper._safetensor_loader_instance is None:
                AMXMoEWrapper._safetensor_loader_instance = SafeTensorLoader(weight_path)
            self.safetensor_loader = AMXMoEWrapper._safetensor_loader_instance

        # AMX-specific weight storage
        self.gate_weights = None
        self.up_weights = None
        self.down_weights = None
        self.gate_scales = None
        self.up_scales = None
        self.down_scales = None

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        """
        Load and quantize weights from BF16/FP16 tensors (online quantization).

        Args:
            gate_proj: Gate projection weights [num_experts, intermediate_size, hidden_size]
            up_proj: Up projection weights [num_experts, intermediate_size, hidden_size]
            down_proj: Down projection weights [num_experts, hidden_size, intermediate_size]
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        # Store tensors as instance variables to keep them alive
        self.gate_proj = gate_proj.contiguous()
        self.up_proj = up_proj.contiguous()
        self.down_proj = down_proj.contiguous()

        # Configure MoE with online quantization (cpu_save mode)
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size

        # Enable save mode for online quantization
        moe_config.save = True
        moe_config.load = False

        # Set weight pointers
        moe_config.gate_proj = self.gate_proj.data_ptr()
        moe_config.up_proj = self.up_proj.data_ptr()
        moe_config.down_proj = self.down_proj.data_ptr()

        # Set output path for quantized weights
        moe_config.path = self.weight_path

        # Create MoE module based on AMX method
        if self.method == "AMXINT4":
            self.moe = AMXInt4_MOE(moe_config)
        elif self.method == "AMXINT8":
            self.moe = AMXInt8_MOE(moe_config)
        else:
            raise NotImplementedError(f"Unsupported AMX method: {self.method}")

        # Submit quantization and save task
        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        gate_ptr = 0
        up_ptr = 0
        down_ptr = 0

        gate_ptrs = []
        up_ptrs = []
        down_ptrs = []

        gate_scale_ptrs = []
        up_scale_ptrs = []
        down_scale_ptrs = []

        if self.load_merged_weight:
            base_key = f"blk.{self.layer_idx}"
            w = self.safetensor_loader.load_experts(base_key)

            self.gate_weights = w["gate"]
            self.up_weights = w["up"]
            self.down_weights = w["down"]
            self.gate_scales = w["gate_scale"]
            self.up_scales = w["up_scale"]
            self.down_scales = w["down_scale"]

            # Get pointers to weight arrays
            gate_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.gate_weights
            ]

            up_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.up_weights
            ]

            down_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.down_weights
            ]

            gate_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.gate_scales
            ]

            up_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.up_scales
            ]

            down_scale_ptrs = [
                [
                    ctypes.addressof(ctypes.cast(et.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents)
                    for et in numa_array
                ]
                for numa_array in self.down_scales
            ]

        # Configure MoE
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size

        moe_config.gate_proj = gate_ptr
        moe_config.up_proj = up_ptr
        moe_config.down_proj = down_ptr
        moe_config.gate_projs = gate_ptrs
        moe_config.up_projs = up_ptrs
        moe_config.down_projs = down_ptrs
        moe_config.gate_scales = gate_scale_ptrs
        moe_config.up_scales = up_scale_ptrs
        moe_config.down_scales = down_scale_ptrs

        if self.cpu_save:
            moe_config.save = True
            moe_config.load = False
            base_key = f"model.layers.{self.layer_idx}"
            try:
                w = self.safetensor_loader.load_experts(base_key)
            except (ValueError, KeyError):
                base_key = f"model.language_model.layers.{self.layer_idx}"
                w = self.safetensor_loader.load_experts(base_key)

            self.gate_proj = torch.cat(w["gate_weight"], dim=0).contiguous()
            self.up_proj = torch.cat(w["up_weight"], dim=0).contiguous()
            self.down_proj = torch.cat(w["down_weight"], dim=0).contiguous()

            moe_config.gate_proj = self.gate_proj.data_ptr()
            moe_config.up_proj = self.up_proj.data_ptr()
            moe_config.down_proj = self.down_proj.data_ptr()
        else:
            moe_config.load = True

        if not self.load_merged_weight:
            moe_config.path = self.weight_path

        # Create MoE module based on AMX method
        if self.method == "AMXINT4":
            self.moe = AMXInt4_MOE(moe_config)
        elif self.method == "AMXINT8":
            self.moe = AMXInt8_MOE(moe_config)
        else:
            raise NotImplementedError(f"Unsupported AMX method: {self.method}")

        # Load weights
        self.cpu_infer.submit(self.moe.load_weights_task(physical_to_logical_map_cpu.data_ptr()))
        self.cpu_infer.sync()

        # Clean up temporary weight storage if using merged weights
        if self.load_merged_weight:
            del self.gate_weights
            del self.up_weights
            del self.down_weights
            del self.gate_scales
            del self.up_scales
            del self.down_scales


class NativeMoEWrapper(BaseMoEWrapper):
    """Wrapper for native CPU/SYCL experts stored in compressed SafeTensor format."""

    _native_loader_instance = None
    _native_loader_key = None

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
        method: str = "RAWINT4",
        numa_nodes: Optional[List[int]] = None,
        swiglu_limit: float = 0.0,
        swiglu_alpha: float = 0.0,
        pack_all_experts_on_load: bool = False,
        activation: Optional[str] = None,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
    ):
        # Keep direct NativeMoEWrapper callers compatible with the factory's
        # pre-contract arguments. The exact K3 tuple is translated into the
        # independent SiTU fields; other positive alpha values retain their
        # historical SwiGLU-OAI meaning.
        if activation is None:
            if method == "MXFP4" and swiglu_alpha == 4.0 and swiglu_limit == 25.0:
                activation = "situ"
                situ_beta = swiglu_alpha
                situ_linear_beta = swiglu_limit
                swiglu_alpha = 0.0
                swiglu_limit = 0.0
            else:
                activation = "swiglu_oai" if swiglu_alpha > 0.0 else "silu"
        self._swiglu_alpha = float(swiglu_alpha)
        self._activation_type = activation
        self._situ_beta = None if situ_beta is None else float(situ_beta)
        self._situ_linear_beta = (
            None if situ_linear_beta is None else float(situ_linear_beta)
        )
        self.pack_all_experts_on_load = bool(pack_all_experts_on_load)
        if not math.isfinite(swiglu_limit) or swiglu_limit < 0.0:
            raise ValueError("swiglu_limit must be finite and non-negative.")
        if not math.isfinite(swiglu_alpha) or swiglu_alpha < 0.0:
            raise ValueError("swiglu_alpha must be finite and non-negative.")
        if activation not in ("silu", "swiglu_oai", "situ"):
            raise ValueError(f"Unsupported NativeMoEWrapper activation: {activation!r}")
        if activation == "situ":
            if method not in NATIVE_SITU_METHODS:
                raise ValueError(
                    f"NativeMoEWrapper SiTU is unsupported for method={method!r}; "
                    f"supported native CPU methods are {sorted(NATIVE_SITU_METHODS)}."
                )
            if (
                self._situ_beta is None
                or not math.isfinite(self._situ_beta)
                or self._situ_beta <= 0.0
            ):
                raise ValueError(
                    "NativeMoEWrapper SiTU requires a finite positive situ_beta."
                )
            if self._situ_linear_beta is not None and (
                not math.isfinite(self._situ_linear_beta)
                or self._situ_linear_beta < 0.0
            ):
                raise ValueError(
                    "situ_linear_beta must be finite and non-negative when provided."
                )
            if swiglu_alpha != 0.0 or swiglu_limit != 0.0:
                raise ValueError(
                    "SiTU must not reuse swiglu_alpha/swiglu_limit."
                )
        elif self._situ_beta is not None or self._situ_linear_beta is not None:
            raise ValueError("SiTU beta parameters require activation='situ'.")
        elif activation == "swiglu_oai" and (
            not math.isfinite(swiglu_alpha) or swiglu_alpha <= 0.0
        ):
            raise ValueError(
                "SwiGLU-OAI requires a finite positive swiglu_alpha."
            )
        elif activation == "silu" and swiglu_alpha != 0.0:
            raise ValueError("activation='silu' cannot use swiglu_alpha.")
        # Defence in depth for direct callers: only methods whose gate/up
        # outputs use the shared CPU activation base may consume clamp/OAI/SiTU
        # parameters. SYCL applies activation inside its fused device kernel.
        if swiglu_limit != 0.0 and method not in NATIVE_SWIGLU_METHODS:
            raise ValueError(
                f"NativeMoEWrapper received swiglu_limit={swiglu_limit} with "
                f"method={method!r}; the clamp requires one of the shared native "
                f"CPU/device methods {sorted(NATIVE_SWIGLU_METHODS)}. "
                f"This indicates a missing guard in the caller."
            )
        if method == "RAWINT4" and not (
            _HAS_RAWINT4_SUPPORT or _HAS_AVX2_RAWINT4_SUPPORT or _HAS_AVXVNNI256_RAW_INT4_SUPPORT
        ):
            raise RuntimeError(
                "RAWINT4 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW (for AMX backend), or\n"
                "  - AVX2 + FMA (for AVX2 fallback backend)\n"
                "AVX-VNNI-256 will be selected automatically when available on the current CPU.\n"
                "Please recompile kt_kernel_ext with AVX512 or AVX2 enabled."
            )
        if method == "FP8" and not (_HAS_FP8_SUPPORT or _HAS_AVX2_FP8_SUPPORT or _HAS_NEON_FP8_SUPPORT):
            raise RuntimeError(
                "FP8 backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI (for AMX), or\n"
                "  - AVX2 + FMA (for AVX2 fallback), or\n"
                "  - AArch64 NEON (BF16 BFDOT preferred)\n"
                "Please recompile kt_kernel_ext with the appropriate CPU backend enabled."
            )
        if method == "FP8_PERCHANNEL" and not _HAS_FP8_PERCHANNEL_SUPPORT:
            raise RuntimeError(
                "FP8_PERCHANNEL backend not available. Required ISA:\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI\n"
                "Please recompile kt_kernel_ext with AVX512 + BF16 + VBMI enabled."
            )
        if method == "BF16" and not (_HAS_BF16_SUPPORT or _HAS_AVX2_BF16_SUPPORT or _HAS_NEON_BF16_SUPPORT):
            raise RuntimeError(
                "BF16 backend not available. Required ISA (any one of):\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 (for AMX backend), or\n"
                "  - AVX2 + FMA (for AVX2 fallback backend), or\n"
                "  - AArch64 NEON (arm64 native build, e.g. Ampere One)\n"
                "Please recompile kt_kernel_ext with AVX512+BF16, AVX2 or ARM64 NEON enabled."
            )
        if method == "GPTQ_INT4" and not (
            _HAS_AVX2_GPTQ_INT4_SUPPORT or _HAS_AVXVNNI256_GPTQ_INT4_SUPPORT or _HAS_AVXVNNI256_PACKED_GPTQ_INT4_SUPPORT
        ):
            raise RuntimeError(
                "GPTQ_INT4 backend not available.\n"
                "Please recompile kt_kernel_ext with GPTQ INT4 support enabled.\n"
                "The packed AVX-VNNI-256 backend is selected automatically when available on the current CPU."
            )
        if method == "SYCL_GPTQ_INT4" and not _HAS_SYCL_GPTQ_INT4_SUPPORT:
            raise RuntimeError(
                "SYCL_GPTQ_INT4 backend not available. Rebuild kt_kernel_ext with "
                "CPUINFER_USE_SYCL=1 using a SYCL compiler such as icpx."
            )
        if method == "NVFP4" and not _HAS_AVX2_MXFP4_SUPPORT:
            raise RuntimeError(
                "NVFP4 needs the AVX2 FP4 backend (AVX2MXFP4_MOE), which is not compiled in."
            )
        if method == "MXFP4" and not (
            _HAS_MXFP4_SUPPORT or _HAS_AVX2_MXFP4_SUPPORT or _HAS_NEON_MXFP4_SUPPORT
        ):
            raise RuntimeError(
                "MXFP4 backend not available. Required ISA (any one of):\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 (for AMX/AVX-512 backend)\n"
                "  - AVX2 + FMA (for AVX2 fallback backend)\n"
                "  - AArch64 NEON/BFDOT (for ARM backend)\n"
                "Please recompile kt_kernel_ext with one of the above enabled."
            )
        if method == "MXFP8" and not (_HAS_MXFP8_SUPPORT or _HAS_AVX2_MXFP8_SUPPORT or _HAS_NEON_MXFP8_SUPPORT):
            raise RuntimeError(
                "MXFP8 backend not available. Required ISA (any one of):\n"
                "  - AVX512F + AVX512BW + AVX512_BF16 + AVX512_VBMI (for AMX/AVX-512 backend)\n"
                "  - AVX2 + FMA (for AVX2 fallback backend)\n"
                "  - AArch64 NEON (BF16 BFDOT preferred)\n"
                "Please recompile kt_kernel_ext with one of the above enabled."
            )

        super().__init__(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
            numa_nodes=numa_nodes,
            swiglu_limit=swiglu_limit,
            activation=activation,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

        self._loader_key = NativeMoEWrapper._make_loader_key(method, weight_path)
        self.loader = NativeMoEWrapper._ensure_loader(method, weight_path)

        self.gate_weights = None
        self.up_weights = None
        self.down_weights = None
        self.gate_scales = None
        self.up_scales = None
        self.down_scales = None

    @staticmethod
    def _create_loader(method: str, weight_path: str):
        if method == "RAWINT4":
            return CompressedSafeTensorLoader(weight_path)
        elif method == "FP8":
            return FP8SafeTensorLoader(weight_path)
        elif method == "FP8_PERCHANNEL":
            return FP8SafeTensorLoader(weight_path, scale_suffix="weight_scale")
        elif method == "BF16":
            return BF16SafeTensorLoader(weight_path)
        elif method in ("GPTQ_INT4", "SYCL_GPTQ_INT4"):
            return GPTQSafeTensorLoader(weight_path)
        elif method == "MXFP4":
            return MXFP4SafeTensorLoader(weight_path)
        elif method == "NVFP4":
            return NVFP4SafeTensorLoader(weight_path)
        elif method == "MXFP8":
            return MXFP8SafeTensorLoader(weight_path)
        else:
            raise NotImplementedError(f"Unsupported method for NativeMoEWrapper: {method}")

    @staticmethod
    def _make_loader_key(method: str, weight_path: str):
        return method, os.path.realpath(os.path.abspath(os.fspath(weight_path)))

    @classmethod
    def _ensure_loader(cls, method: str, weight_path: str):
        import time

        loader_key = cls._make_loader_key(method, weight_path)
        if (
            cls._native_loader_instance is None
            or cls._native_loader_key != loader_key
        ):
            if cls._native_loader_instance is not None:
                # A different checkpoint/model is being constructed in this
                # process.  Its wrappers retain their own immutable index, but
                # no stale mmap handle should remain in the singleton cache.
                cls._native_loader_instance.close_all_handles(collect=False)
            started = time.perf_counter()
            logger.info(
                "[KT] Building shared native expert index: method=%s, path=%s",
                method,
                weight_path,
            )
            cls._native_loader_instance = cls._create_loader(method, weight_path)
            cls._native_loader_key = loader_key
            logger.debug(
                "[KT] Shared native expert index ready in %.2fs: %d keys, %d shards",
                time.perf_counter() - started,
                len(cls._native_loader_instance.tensor_file_map),
                len(cls._native_loader_instance.checkpoint_files),
            )
        return cls._native_loader_instance

    @classmethod
    def _release_loader(
        cls, layer_idx: int = -1, *, drop_index: bool = False, loader=None
    ):
        target_loader = loader or cls._native_loader_instance
        if target_loader is not None:
            if drop_index:
                target_loader.clear_index()
            else:
                # Layer tensors have been copied into the native NUMA buffers.
                # Drop mmap handles but retain the one shared 500k-key index;
                # rebuilding that index for every layer is prohibitively slow.
                target_loader.close_all_handles(collect=False)
            if drop_index and target_loader is cls._native_loader_instance:
                cls._native_loader_instance = None
                cls._native_loader_key = None
            if layer_idx >= 0:
                logger.debug(
                    "[KT] Closed native safetensors handles after layer %d; "
                    "shared tensor index retained.",
                    layer_idx,
                )
            elif drop_index:
                logger.info(
                    "[KT] Released NativeMoEWrapper safetensors handles and index."
                )

    @classmethod
    def force_release_loader(cls):
        cls._release_loader(drop_index=True)

    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        physical_to_logical_map_cpu: torch.Tensor,
    ):
        raise NotImplementedError("RAWINT4 wrapper expects pre-quantized safetensor weights.")

    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        import time

        if not getattr(self.loader, "tensor_file_map", None):
            self.loader = NativeMoEWrapper._ensure_loader(
                self.method, self.weight_path
            )

        t0 = time.time()
        logger.info(
            "[KT] Loading native %s experts for layer %d from safetensors",
            self.method,
            self.layer_idx,
        )
        _candidates = [
            f"model.layers.{self.layer_idx}",
            f"language_model.layers.{self.layer_idx}",
            f"language_model.model.layers.{self.layer_idx}",
            f"model.language_model.layers.{self.layer_idx}",
        ]
        weights = None
        load_errors = []
        for base_key in _candidates:
            try:
                weights = self.loader.load_experts(base_key)
                break
            except (TypeError, ValueError, KeyError) as exc:
                load_errors.append(f"{base_key}: {exc}")
                continue
        if weights is None:
            raise ValueError(
                f"No experts found for layer {self.layer_idx} under any prefix: "
                f"{_candidates}. Loader diagnostics: {load_errors}"
            )
        t1 = time.time()
        logger.debug(
            "[KT] Layer %d expert tensors prepared in %.2fs (%d shard handles open)",
            self.layer_idx,
            t1 - t0,
            len(self.loader.file_handle_map),
        )

        # Keep individual tensors instead of stacking - avoid expensive memory copy
        # weights["gate"], weights["up"], weights["down"] are lists of tensors per expert
        self.gate_weights = weights["gate"]  # list of tensors
        self.up_weights = weights["up"]
        self.down_weights = weights["down"]

        # BF16 has no scales, others have scales
        if self.method == "BF16":
            # BF16 doesn't have scales
            self.gate_scales = None
            self.up_scales = None
            self.down_scales = None
        else:
            # Convert scales to bf16 individually
            # self.gate_scales = [t.to(torch.bfloat16).contiguous() for t in weights["gate_scale"]]
            # self.up_scales = [t.to(torch.bfloat16).contiguous() for t in weights["up_scale"]]
            # self.down_scales = [t.to(torch.bfloat16).contiguous() for t in weights["down_scale"]]
            self.gate_scales = weights["gate_scale"]
            self.up_scales = weights["up_scale"]
            self.down_scales = weights["down_scale"]
            if self.method == "RAWINT4":
                assert self.gate_scales[0].dtype == torch.bfloat16, "Expected bf16 scales for RAWINT4"
            elif self.method == "FP8":
                self.gate_scales = [t.to(torch.float32).contiguous() for t in weights["gate_scale"]]
                self.up_scales = [t.to(torch.float32).contiguous() for t in weights["up_scale"]]
                self.down_scales = [t.to(torch.float32).contiguous() for t in weights["down_scale"]]
                assert self.gate_scales[0].dtype == torch.float32, "Expected float32 scales for FP8"
            elif self.method == "FP8_PERCHANNEL":
                self.gate_scales = [t.to(torch.float32).contiguous() for t in weights["gate_scale"]]
                self.up_scales = [t.to(torch.float32).contiguous() for t in weights["up_scale"]]
                self.down_scales = [t.to(torch.float32).contiguous() for t in weights["down_scale"]]
                assert self.gate_scales[0].dtype == torch.float32, "Expected float32 scales for FP8_PERCHANNEL"
            elif self.method == "MXFP4":
                # ue8m0 is losslessly representable in bf16 (8-bit exponent, 0 mantissa);
                # the loader has already done that conversion.
                assert self.gate_scales[0].dtype == torch.bfloat16, "Expected bf16 scales for MXFP4"
            elif self.method == "NVFP4":
                # e4m3 block scale x per-tensor global, folded to bf16 by the loader.
                assert self.gate_scales[0].dtype == torch.bfloat16, "Expected bf16 scales for NVFP4"
            elif self.method == "MXFP8":
                # ue8m0 scales stay as uint8; C++ convert_ue8m0_to_fp32 handles conversion.
                assert self.gate_scales[0].dtype == torch.uint8, "Expected uint8 (ue8m0) scales for MXFP8"

        if self.method in ("FP8", "MXFP4", "MXFP8"):
            _validate_expert_mapping(
                physical_to_logical_map_cpu,
                self.num_experts,
                {
                    "gate": self.gate_weights,
                    "up": self.up_weights,
                    "down": self.down_weights,
                },
            )
            # C++ retains this pointer for layerwise expert staging after the
            # initial load task, so keep the owning tensor alive on the wrapper.
            self.physical_to_logical_map_cpu = physical_to_logical_map_cpu.contiguous()

        if self.method == "FP8":
            if self.loader.is_per_channel():
                raise ValueError(
                    "method='FP8' only supports 128x128 block-wise FP8, but this checkpoint "
                    "contains per-channel scales. Use method='FP8_PERCHANNEL' on a supported "
                    "x86 backend; ARM NEON per-channel FP8 is not implemented."
                )
            _validate_block_fp8_layout(
                self.gate_weights,
                self.gate_scales,
                "gate",
                (self.moe_intermediate_size, self.hidden_size),
            )
            _validate_block_fp8_layout(
                self.up_weights,
                self.up_scales,
                "up",
                (self.moe_intermediate_size, self.hidden_size),
            )
            _validate_block_fp8_layout(
                self.down_weights,
                self.down_scales,
                "down",
                (self.hidden_size, self.moe_intermediate_size),
            )
        elif self.method == "MXFP8":
            _validate_mxfp8_layout(
                self.gate_weights,
                self.gate_scales,
                "gate",
                (self.moe_intermediate_size, self.hidden_size),
            )
            _validate_mxfp8_layout(
                self.up_weights,
                self.up_scales,
                "up",
                (self.moe_intermediate_size, self.hidden_size),
            )
            _validate_mxfp8_layout(
                self.down_weights,
                self.down_scales,
                "down",
                (self.hidden_size, self.moe_intermediate_size),
            )
        elif self.method == "MXFP4":
            _validate_mxfp4_layout(
                self.gate_weights,
                self.gate_scales,
                "gate",
                (self.moe_intermediate_size, self.hidden_size),
            )
            _validate_mxfp4_layout(
                self.up_weights,
                self.up_scales,
                "up",
                (self.moe_intermediate_size, self.hidden_size),
            )
            _validate_mxfp4_layout(
                self.down_weights,
                self.down_scales,
                "down",
                (self.hidden_size, self.moe_intermediate_size),
            )

        t2 = time.time()

        # Build pointer lists: [numa_id][expert_id] -> pointer
        # Since RAWINT4/FP8/BF16 has no numa sharding, numa dimension is 1
        gate_ptrs = [[t.data_ptr() for t in self.gate_weights]]
        up_ptrs = [[t.data_ptr() for t in self.up_weights]]
        down_ptrs = [[t.data_ptr() for t in self.down_weights]]

        # BF16 has no scales, pass empty lists (will use 0/nullptr for consistency)
        if self.method == "BF16":
            gate_scale_ptrs = [[0 for _ in self.gate_weights]]
            up_scale_ptrs = [[0 for _ in self.up_weights]]
            down_scale_ptrs = [[0 for _ in self.down_weights]]
        else:
            gate_scale_ptrs = [[t.data_ptr() for t in self.gate_scales]]
            up_scale_ptrs = [[t.data_ptr() for t in self.up_scales]]
            down_scale_ptrs = [[t.data_ptr() for t in self.down_scales]]
        t3 = time.time()

        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.moe_intermediate_size,
            self.gpu_experts_mask.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.cpu_infer.backend_
        moe_config.max_len = self.chunked_prefill_size
        # Clamp-before-gated-activation; 0.0 = disabled. Re-check the shared
        # activation capability here in case a caller bypasses the factory.
        if self.swiglu_limit != 0.0 and self.method not in NATIVE_SWIGLU_METHODS:
            raise ValueError(
                f"NativeMoEWrapper.load_weights: swiglu_limit="
                f"{self.swiglu_limit} with method={self.method!r}; clamp is "
                f"only valid for shared native CPU methods "
                f"{sorted(NATIVE_SWIGLU_METHODS)}."
            )
        moe_config.swiglu_limit = self.swiglu_limit
        activation_types = {"silu": 0, "swiglu_oai": 1, "situ": 2}
        moe_config.activation_type = activation_types[self._activation_type]
        moe_config.swiglu_alpha = (
            self._swiglu_alpha if self._activation_type == "swiglu_oai" else 0.0
        )
        moe_config.situ_beta = self._situ_beta or 0.0
        moe_config.situ_linear_beta = self._situ_linear_beta or 0.0

        # Use gate_projs instead of gate_proj for per-expert pointers
        moe_config.gate_projs = gate_ptrs
        moe_config.up_projs = up_ptrs
        moe_config.down_projs = down_ptrs
        moe_config.gate_scales = gate_scale_ptrs
        moe_config.up_scales = up_scale_ptrs
        moe_config.down_scales = down_scale_ptrs

        # Keep the physical->logical mapping alive after the asynchronous
        # load.  Layerwise prefill writers receive logical expert IDs and the
        # ARM/x86 native TP exporters resolve them back to their physical
        # BufferB slots through this pointer.
        self.physical_to_logical_map_cpu = physical_to_logical_map_cpu.contiguous()
        moe_config.physical_to_logical_map = self.physical_to_logical_map_cpu.data_ptr()

        # Infer group_size from scale shape (column-major layout)
        # For gate/up projection: in_features = hidden_size
        # So: group_size = hidden_size / scale.shape[1]

        if self.method == "RAWINT4":
            group_size = self.hidden_size // self.gate_scales[0].shape[1]
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = group_size
            moe_config.quant_config.zero_point = False
            backend_cls = _select_rawint4_backend(group_size)
            if backend_cls is None:
                raise RuntimeError(
                    "No RAWINT4 backend is available after runtime selection for "
                    f"group_size={group_size}. AMX (AMXInt4_KGroup_MOE) is preferred; "
                    f"AVX-VNNI-256 supports positive multiples of 32 up to "
                    f"{_AVXVNNI256_RAW_INT4_MAX_GROUP_SIZE}; AVX2 (AVX2RawInt4_MOE) is used as the final fallback."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "MXFP4":
            # MXFP4: E2M1 nibble-packed weights, ue8m0/bf16 per-32 group scale
            # (e.g. DeepSeek-V4-Flash routed experts)
            group_size = self.hidden_size // self.gate_scales[0].shape[1]
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = group_size
            moe_config.quant_config.zero_point = False
            backend_cls = _select_mxfp4_backend()
            if backend_cls is None:
                raise RuntimeError(
                    "No MXFP4 backend available after runtime selection. "
                    "Compile with AVX512_BF16 (AMXFP4_KGroup_MOE), AVX2 (AVX2MXFP4_MOE), "
                    "or AArch64 NEON/BFDOT (NEONMXFP4_MOE)."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "NVFP4":
            # NVFP4: same E2M1 nibble packing as MXFP4, but a per-16 block scale
            # in E4M3 times a per-tensor global scale. The loader has already
            # folded both into one bf16 scale per group, so the FP4 kernel runs
            # this unchanged -- only group_size differs (16 vs 32).
            group_size = self.hidden_size // self.gate_scales[0].shape[1]
            if group_size != 16:
                raise RuntimeError(
                    f"NVFP4 expects group_size 16, derived {group_size} from "
                    f"hidden_size={self.hidden_size} and scale shape "
                    f"{tuple(self.gate_scales[0].shape)}."
                )
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = group_size
            moe_config.quant_config.zero_point = False
            backend_cls = _select_mxfp4_backend()
            if backend_cls is None:
                raise RuntimeError(
                    "No FP4 backend available for NVFP4 after runtime selection. "
                    "Compile with AVX512_BF16 (AMXFP4_KGroup_MOE) or AVX2 (AVX2MXFP4_MOE)."
                )
            if backend_cls is not AVX2MXFP4_MOE:
                # Only the AVX2 kernel has the group-16 path; the AMX FP4 kernel
                # is built around a 32-value k-group.
                raise RuntimeError(
                    "NVFP4 (group_size 16) currently requires the AVX2 FP4 backend. "
                    "Set KT_MXFP4_BACKEND=avx2, or extend the AMX kernel to group-16."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "MXFP8":
            # MXFP8: FP8 E4M3fn byte weights, ue8m0/uint8 per-32 group scale
            # (e.g. MiniMax-M3-Preview)
            group_size = self.hidden_size // self.gate_scales[0].shape[1]
            moe_config.quant_config.bits = 8
            moe_config.quant_config.group_size = group_size
            moe_config.quant_config.zero_point = False
            backend_cls = _select_mxfp8_backend()
            if backend_cls is None:
                raise RuntimeError(
                    "No MXFP8 backend available after runtime selection. "
                    "Compile with AVX512+VBMI, AVX2, or AArch64 NEON support."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "FP8":
            moe_config.quant_config.bits = 8
            moe_config.quant_config.group_size = 128
            moe_config.quant_config.zero_point = False
            if _HAS_FP8_SUPPORT:
                self.moe = AMXFP8_MOE(moe_config)
            elif _HAS_AVX2_FP8_SUPPORT:
                self.moe = AVX2FP8_MOE(moe_config)
            elif _HAS_NEON_FP8_SUPPORT:
                self.moe = NEONFP8_MOE(moe_config)
            else:
                raise RuntimeError(
                    "FP8 MoE is not available in this build: no AMX, AVX2, or NEON FP8 kernel was compiled in."
                )
        elif self.method == "FP8_PERCHANNEL":
            moe_config.quant_config.bits = 8
            moe_config.quant_config.per_channel = True
            moe_config.quant_config.zero_point = False
            if not _HAS_FP8_PERCHANNEL_SUPPORT:
                raise RuntimeError(
                    "FP8_PERCHANNEL MoE requires AMX/AVX512-BF16 support and is not "
                    "available in this build (on ARM64, use method='BF16' instead)."
                )
            self.moe = AMXFP8PerChannel_MOE(moe_config)
        elif self.method == "GPTQ_INT4":
            # GPTQ symmetric INT4: qweight (int32) + scales (fp32)
            group_size = self.gate_scales[0].shape[0]  # scales shape [K/gs, N], first dim = num_groups
            # hidden_size / num_groups = group_size
            actual_gs = self.hidden_size // group_size
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = actual_gs
            moe_config.quant_config.zero_point = False
            backend_cls = _select_gptq_int4_backend(actual_gs)
            if backend_cls is None:
                raise RuntimeError(
                    "No GPTQ_INT4 backend is available after runtime selection for "
                    f"group_size={actual_gs}. AVX-VNNI-256 supports positive multiples of 32 up to "
                    f"{_AVXVNNI256_GPTQ_INT4_MAX_GROUP_SIZE}; AVX2 is used as the fallback when available."
                )
            self.moe = backend_cls(moe_config)
        elif self.method == "SYCL_GPTQ_INT4":
            # Same symmetric GPTQ tensor layout as GPTQ_INT4; execution is on
            # the selected SYCL device.
            num_groups = self.gate_scales[0].shape[0]
            actual_gs = self.hidden_size // num_groups
            moe_config.quant_config.bits = 4
            moe_config.quant_config.group_size = actual_gs
            moe_config.quant_config.zero_point = False
            if not _HAS_SYCL_GPTQ_INT4_SUPPORT:
                raise RuntimeError(
                    "SYCL_GPTQ_INT4 MoE is not available in this build (SYCL backend not compiled in)."
                )
            _preflight_sycl_device()
            self.moe = SYCLGPTQInt4_MOE(moe_config)
        elif self.method == "BF16":
            # BF16 has no quantization config needed
            # Prefer AMX backend, fall back to AVX2, then ARM NEON
            if _HAS_BF16_SUPPORT:
                self.moe = AMXBF16_MOE(moe_config)
            elif _HAS_AVX2_BF16_SUPPORT:
                self.moe = AVX2BF16_MOE(moe_config)
            elif _HAS_NEON_BF16_SUPPORT:
                self.moe = NEONBF16_MOE(moe_config)
            else:
                raise RuntimeError(
                    "BF16 MoE is not available in this build: no AMX, AVX2, "
                    "or ARM NEON (NEONBF16_MOE) kernel was compiled in."
                )
        t4 = time.time()

        # Pass the wrapper-owned mapping to the asynchronous native loader;
        # the C++ MoE retains this pointer for later layerwise staging calls.
        load_task = self.moe.load_weights_task(self.physical_to_logical_map_cpu.data_ptr())
        logger.debug(
            "[KT] Layer %d copying %d experts into native NUMA buffers",
            self.layer_idx,
            self.num_experts,
        )
        with _temporary_all_cpu_experts_mask(
            self.gpu_experts_mask,
            self.pack_all_experts_on_load and self.num_gpu_experts > 0,
        ):
            self.cpu_infer.submit(load_task)
            self.cpu_infer.sync()
        t5 = time.time()
        logger.debug(
            "[KT] Layer %d native NUMA copy finished in %.2fs",
            self.layer_idx,
            t5 - t4,
        )

        del self.gate_weights
        del self.up_weights
        del self.down_weights
        if self.gate_scales is not None:
            del self.gate_scales
            del self.up_scales
            del self.down_scales
        del weights

        NativeMoEWrapper._release_loader(
            layer_idx=self.layer_idx, loader=self.loader
        )
        t6 = time.time()

        logger.info(
            "[KT] Native layer %d loaded: load_experts=%.1fms, "
            "prepare_tensors=%.1fms, build_ptrs=%.1fms, create_moe=%.1fms, "
            "cpp_load_weights=%.1fms, cleanup=%.1fms, total=%.1fms",
            self.layer_idx,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            (t4 - t3) * 1000,
            (t5 - t4) * 1000,
            (t6 - t5) * 1000,
            (t6 - t0) * 1000,
        )

    def submit_write_weight_scale_to_buffer(
        self,
        gpu_tp_count: int,
        expert_id: int,
        w13_weight_ptrs,
        w13_scale_ptrs,
        w2_weight_ptrs,
        w2_scale_ptrs,
    ):
        """
        Submit the write_weight_scale_to_buffer task for RAWINT4 KGroup AMX implementation.

        This method submits the C++-exposed task `write_weight_scale_to_buffer_task` to the
        shared CPUInfer queue. The pointer lists should be plain integer lists (e.g. from
        tensor.data_ptr()).
        """
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot submit write_weight_scale_to_buffer task.")

        if not hasattr(self.moe, "write_weight_scale_to_buffer_task"):
            raise NotImplementedError(
                "write_weight_scale_to_buffer_task is not available for this backend implementation."
            )

        self.cpu_infer.submit(
            self.moe.write_weight_scale_to_buffer_task(
                gpu_tp_count,
                expert_id,
                w13_weight_ptrs,
                w13_scale_ptrs,
                w2_weight_ptrs,
                w2_scale_ptrs,
            )
        )

    def sync_write_weight_scale_to_buffer(self):
        """
        Block until previously submitted write_weight_scale_to_buffer tasks finish.
        """
        # The CPUInfer.sync() call blocks until pending tasks complete.
        self.cpu_infer.sync()

    def run_layerwise_fp8_batch(
        self,
        transport,
        epoch: int,
        layer_id: int,
        expert_count: int,
    ):
        """Run one block-FP8 layer's native writer/H2D transport pipeline.

        The transport owns the hot per-expert protocol.  Python enters once per
        layer, while the C++ producer overlaps writing expert ``e + 1`` with
        each rank's local H2D copy of expert ``e``.
        """
        if self.method != "FP8":
            raise RuntimeError(
                "run_layerwise_fp8_batch is only valid for the block-FP8 NativeMoEWrapper backend"
            )
        if self.moe is None:
            raise RuntimeError("MoE instance not initialized; cannot run FP8 layerwise transport.")
        if not hasattr(self.moe, "run_layerwise_fp8_batch"):
            raise NotImplementedError(
                "The installed kt-kernel extension does not expose run_layerwise_fp8_batch."
            )
        # The native batch calls the shared NUMA distributor directly.  Drain
        # CPUInfer first so it cannot race a previously queued task against the
        # non-reentrant distributor state.
        self.cpu_infer.sync()
        return self.moe.run_layerwise_fp8_batch(
            transport,
            epoch,
            layer_id,
            expert_count,
        )
