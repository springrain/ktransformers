#!/usr/bin/env python
"""MXFP8 (E4M3FN + per-32 UE8M0 scale) CPU MoE accuracy test."""

import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch

from ci.ci_register import register_cpu_ci
from kt_kernel import kt_kernel_ext

register_cpu_ci(est_time=120, suite="default")

EXPERTS = 4
TOPK = 2
HIDDEN = 128
INTERMEDIATE = 256
GROUP_SIZE = 32
IS_ARM64 = platform.machine().lower() in ("aarch64", "arm64")
ACTIVATION_CASES = [(0.0, 0.0), (1.702, 0.01)] if IS_ARM64 else [(0.0, 0.0)]


def mxfp8_backend():
    name = "NEONMXFP8_MOE" if IS_ARM64 else "AVX2MXFP8_MOE"
    backend = getattr(kt_kernel_ext.moe, name, None)
    if backend is None:
        pytest.skip(f"{name} is not available in this build")
    return backend


def decode_e4m3fn(values):
    raw = values.to(torch.int32)
    sign = torch.where((raw & 0x80) != 0, -1.0, 1.0)
    exponent = (raw >> 3) & 0x0F
    mantissa = raw & 0x07
    normal = torch.pow(2.0, exponent.float() - 7.0) * (1.0 + mantissa.float() / 8.0)
    subnormal = torch.pow(torch.tensor(2.0), -6.0) * mantissa.float() / 8.0
    decoded = torch.where(exponent == 0, subnormal, normal)
    decoded = torch.where((exponent == 15) & (mantissa == 7), 0.0, decoded)
    return decoded * sign


def dequant_mxfp8(weights, scales):
    scale_values = torch.pow(2.0, scales.to(torch.int32).float() - 127.0)
    return decode_e4m3fn(weights) * scale_values.repeat_interleave(GROUP_SIZE, dim=-1)


def mlp(inputs, gate, up, down, swiglu_alpha, swiglu_limit):
    gate_output = inputs @ gate.t()
    up_output = inputs @ up.t()
    if swiglu_alpha > 0.0:
        if swiglu_limit > 0.0:
            gate_output = gate_output.clamp(max=swiglu_limit)
            up_output = up_output.clamp(min=-swiglu_limit, max=swiglu_limit)
        activated = gate_output * torch.sigmoid(gate_output * swiglu_alpha) * (up_output + 1.0)
    else:
        activated = torch.nn.functional.silu(gate_output) * up_output
    return activated @ down.t()


def moe_reference(
    inputs,
    expert_ids,
    routing_weights,
    gate,
    up,
    down,
    physical_to_logical_map,
    swiglu_alpha,
    swiglu_limit,
):
    output = torch.zeros_like(inputs, dtype=torch.float32)
    for token in range(inputs.shape[0]):
        for route in range(TOPK):
            expert = int(physical_to_logical_map[int(expert_ids[token, route])])
            output[token] += routing_weights[token, route] * mlp(
                inputs[token : token + 1],
                gate[expert],
                up[expert],
                down[expert],
                swiglu_alpha,
                swiglu_limit,
            )[0]
    return output


def random_mxfp8(shape):
    # Magnitudes below 0x39 keep the test numerically tame while covering zero,
    # subnormal, normal, and signed values. Varying each UE8M0 group exponent
    # catches scale-row and K-group indexing mistakes.
    magnitude = torch.randint(0, 0x39, shape, dtype=torch.uint8)
    sign = torch.randint(0, 2, shape, dtype=torch.uint8) << 7
    weights = (magnitude | sign).contiguous()
    scales = torch.randint(
        121,
        126,
        (*shape[:-1], shape[-1] // GROUP_SIZE),
        dtype=torch.uint8,
    )
    return weights, scales.contiguous()


def verify_arm_layerwise_staging(
    cpu_infer,
    moe,
    gate,
    gate_scale,
    up,
    up_scale,
    down,
    down_scale,
):
    if not IS_ARM64:
        return

    logical_expert = 0
    w13 = torch.empty((2 * INTERMEDIATE, HIDDEN), dtype=torch.uint8)
    w13_scale = torch.empty((2 * INTERMEDIATE, HIDDEN // GROUP_SIZE), dtype=torch.uint8)
    w2 = torch.empty((HIDDEN, INTERMEDIATE), dtype=torch.uint8)
    w2_scale = torch.empty((HIDDEN, INTERMEDIATE // GROUP_SIZE), dtype=torch.uint8)
    cpu_infer.submit(
        moe.write_weight_scale_to_buffer_task(
            1,
            logical_expert,
            [w13.data_ptr()],
            [w13_scale.data_ptr()],
            [w2.data_ptr()],
            [w2_scale.data_ptr()],
        )
    )
    cpu_infer.sync()

    assert torch.equal(w13[:INTERMEDIATE], gate[logical_expert])
    assert torch.equal(w13[INTERMEDIATE:], up[logical_expert])
    assert torch.equal(w13_scale[:INTERMEDIATE], gate_scale[logical_expert])
    assert torch.equal(w13_scale[INTERMEDIATE:], up_scale[logical_expert])
    assert torch.equal(w2, down[logical_expert])
    assert torch.equal(w2_scale, down_scale[logical_expert])


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", [1, 8])
@pytest.mark.parametrize("swiglu_alpha,swiglu_limit", ACTIVATION_CASES)
def test_mxfp8_accuracy(qlen, swiglu_alpha, swiglu_limit):
    cpu_infer = kt_kernel_ext.CPUInfer(60)
    if IS_ARM64:
        # Exercise physical-to-logical remapping across every ARM NUMA TP.
        mapping = torch.arange(EXPERTS - 1, -1, -1, dtype=torch.int64).contiguous()
    else:
        mapping = torch.arange(EXPERTS, dtype=torch.int64).contiguous()

    gate, gate_scale = random_mxfp8((EXPERTS, INTERMEDIATE, HIDDEN))
    up, up_scale = random_mxfp8((EXPERTS, INTERMEDIATE, HIDDEN))
    down, down_scale = random_mxfp8((EXPERTS, HIDDEN, INTERMEDIATE))

    config = kt_kernel_ext.moe.MOEConfig(EXPERTS, TOPK, HIDDEN, INTERMEDIATE, 0)
    config.max_len = 32
    config.gate_proj = gate.data_ptr()
    config.up_proj = up.data_ptr()
    config.down_proj = down.data_ptr()
    config.gate_scale = gate_scale.data_ptr()
    config.up_scale = up_scale.data_ptr()
    config.down_scale = down_scale.data_ptr()
    config.quant_config.bits = 8
    config.quant_config.group_size = GROUP_SIZE
    config.quant_config.zero_point = False
    config.swiglu_alpha = swiglu_alpha
    config.swiglu_limit = swiglu_limit
    config.pool = cpu_infer.backend_

    moe = mxfp8_backend()(config)
    cpu_infer.submit(moe.load_weights_task(mapping.data_ptr()))
    cpu_infer.sync()
    verify_arm_layerwise_staging(
        cpu_infer,
        moe,
        gate,
        gate_scale,
        up,
        up_scale,
        down,
        down_scale,
    )

    expert_ids = torch.stack([torch.randperm(EXPERTS)[:TOPK] for _ in range(qlen)]).contiguous()
    routing = torch.rand((qlen, TOPK), dtype=torch.float32).contiguous()
    inputs = (torch.randn((qlen, HIDDEN), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    output = torch.empty_like(inputs)
    batch_size = torch.tensor([qlen], dtype=torch.int32)

    cpu_infer.submit(
        moe.forward_task(
            batch_size.data_ptr(), TOPK, expert_ids.data_ptr(), routing.data_ptr(),
            inputs.data_ptr(), output.data_ptr(), False,
        )
    )
    cpu_infer.sync()

    reference = moe_reference(
        inputs.float(), expert_ids, routing,
        dequant_mxfp8(gate, gate_scale), dequant_mxfp8(up, up_scale), dequant_mxfp8(down, down_scale),
        mapping, swiglu_alpha, swiglu_limit,
    )
    relative_error = torch.mean(torch.abs(output.float() - reference)) / (
        torch.mean(torch.abs(reference)) + 1e-8
    )
    assert relative_error < 0.1, f"MXFP8 relative error {relative_error.item():.6f}"
