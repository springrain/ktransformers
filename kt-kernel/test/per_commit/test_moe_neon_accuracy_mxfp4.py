#!/usr/bin/env python
"""ARM NEON packed-MXFP4 MoE accuracy and staging tests."""

import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="default")

if platform.machine().lower() not in ("aarch64", "arm64"):
    pytest.skip("NEON MXFP4 test requires an aarch64/arm64 host", allow_module_level=True)

import torch

from kt_kernel import kt_kernel_ext

pytestmark = pytest.mark.cpu

BACKEND = getattr(kt_kernel_ext.moe, "NEONMXFP4_MOE", None)
if BACKEND is None:
    pytest.skip("NEONMXFP4_MOE is not available in this ARM build", allow_module_level=True)

EXPERTS = 4
TOPK = 2
HIDDEN = 96
INTERMEDIATE = 256
GROUP = 32
MAX_LEN = 64

_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _random_mxfp4(shape):
    packed = torch.randint(0, 256, shape[:-1] + (shape[-1] // 2,), dtype=torch.uint8)
    scale_codes = torch.randint(121, 126, shape[:-1] + (shape[-1] // GROUP,), dtype=torch.int16)
    scales = (scale_codes << 7).view(torch.bfloat16).contiguous()
    return packed.contiguous(), scales


def _dequant(packed, scales):
    low = packed.to(torch.int64) & 0x0F
    high = (packed.to(torch.int64) >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return _FP4_VALUES[nibbles] * scales.float().repeat_interleave(GROUP, dim=-1)


def _mlp(x, gate, up, down):
    return (torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())) @ down.t()


def _reference(x, ids, routing, gate, up, down, mapping):
    out = torch.zeros((x.shape[0], HIDDEN), dtype=torch.float32)
    for token in range(x.shape[0]):
        for route in range(TOPK):
            physical_id = int(ids[token, route])
            logical_id = int(mapping[physical_id])
            out[token] += routing[token, route] * _mlp(
                x[token : token + 1], gate[logical_id], up[logical_id], down[logical_id]
            )[0]
    return out


@pytest.mark.parametrize("qlen", [1, 16, 33])
def test_neon_mxfp4_accuracy_and_staging(qlen):
    torch.manual_seed(0x4D584650 + qlen)
    # Use two logical subpools on node 0 so this test covers the TP loader and
    # writer without depending on the host's physical NUMA count.
    worker_config = kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = 2
    worker_config.subpool_numa_map = [0, 0]
    worker_config.subpool_thread_count = [16, 16]
    cpu_infer = kt_kernel_ext.CPUInfer(worker_config)
    mapping = torch.arange(EXPERTS - 1, -1, -1, dtype=torch.int64).contiguous()
    gate, gate_scale = _random_mxfp4((EXPERTS, INTERMEDIATE, HIDDEN))
    up, up_scale = _random_mxfp4((EXPERTS, INTERMEDIATE, HIDDEN))
    down, down_scale = _random_mxfp4((EXPERTS, HIDDEN, INTERMEDIATE))

    config = kt_kernel_ext.moe.MOEConfig(EXPERTS, TOPK, HIDDEN, INTERMEDIATE, 0)
    config.max_len = MAX_LEN
    config.gate_proj = gate.data_ptr()
    config.up_proj = up.data_ptr()
    config.down_proj = down.data_ptr()
    config.gate_scale = gate_scale.data_ptr()
    config.up_scale = up_scale.data_ptr()
    config.down_scale = down_scale.data_ptr()
    config.quant_config.bits = 4
    config.quant_config.group_size = GROUP
    config.quant_config.zero_point = False
    config.pool = cpu_infer.backend_

    moe = BACKEND(config)
    cpu_infer.submit(moe.load_weights_task(mapping.data_ptr()))
    cpu_infer.sync()

    # The staging ABI must preserve packed weights and BF16-compatible scales.
    staged_w13 = torch.empty((2 * INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8)
    staged_s13 = torch.empty((2 * INTERMEDIATE, HIDDEN // GROUP), dtype=torch.bfloat16)
    staged_w2 = torch.empty((HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8)
    staged_s2 = torch.empty((HIDDEN, INTERMEDIATE // GROUP), dtype=torch.bfloat16)
    cpu_infer.submit(
        moe.write_weight_scale_to_buffer_task(
            1,
            0,
            [staged_w13.data_ptr()],
            [staged_s13.data_ptr()],
            [staged_w2.data_ptr()],
            [staged_s2.data_ptr()],
        )
    )
    cpu_infer.sync()
    assert torch.equal(staged_w13[:INTERMEDIATE], gate[0])
    assert torch.equal(staged_w13[INTERMEDIATE:], up[0])
    assert torch.equal(staged_s13[:INTERMEDIATE], gate_scale[0])
    assert torch.equal(staged_s13[INTERMEDIATE:], up_scale[0])
    assert torch.equal(staged_w2, down[0])
    assert torch.equal(staged_s2, down_scale[0])

    ids = torch.stack([torch.randperm(EXPERTS)[:TOPK] for _ in range(qlen)]).contiguous()
    routing = torch.rand((qlen, TOPK), dtype=torch.float32).contiguous()
    inputs = (torch.randn((qlen, HIDDEN), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
    output = torch.empty_like(inputs)
    batch = torch.tensor([qlen], dtype=torch.int32)
    cpu_infer.submit(
        moe.forward_task(
            batch.data_ptr(), TOPK, ids.data_ptr(), routing.data_ptr(), inputs.data_ptr(), output.data_ptr(), False
        )
    )
    cpu_infer.sync()

    reference = _reference(
        inputs.float(), ids, routing,
        _dequant(gate, gate_scale), _dequant(up, up_scale), _dequant(down, down_scale), mapping,
    )
    error = torch.mean(torch.abs(output.float() - reference)) / (torch.mean(torch.abs(reference)) + 1e-8)
    assert error < 0.08, f"NEON MXFP4 relative error={error.item():.6f}"
