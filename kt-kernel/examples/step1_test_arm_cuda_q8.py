#!/usr/bin/env python3
"""Step 1: validate ARM LLAMAFILE Q8_0 execution through a CUDA stream.

This is intentionally a one-layer smoke/stability test, not a full-model
benchmark.  The routed experts are computed by the ARM CPU from the GGUF
weights, while inputs and outputs live on one CUDA GPU.  That exercises the
same pinned-memory and CUDA-stream synchronization used by heterogeneous
inference.

Run from the ktransformers repository root:

    python3 kt-kernel/examples/step1_test_arm_cuda_q8.py

Useful overrides:

    python3 kt-kernel/examples/step1_test_arm_cuda_q8.py \
        --threads 160 --iterations 1000

For synchronous CUDA error reporting:

    CUDA_LAUNCH_BLOCKING=1 \
        python3 kt-kernel/examples/step1_test_arm_cuda_q8.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

# Make the first test use logical CUDA device 0.  An explicit setting from the
# caller is preserved, but this script always selects the first visible GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

from kt_kernel import KTMoEWrapper, __cpu_variant__, kt_kernel_ext
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
from kt_kernel.utils.loader import GGUFLoader


MODEL_CONFIG = Path(
    "/data/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json"
)
GGUF_PATH = Path("/data/ampere/data/Qwen3.6-35B-A3B-Q8_0.gguf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ARM Q8_0 LLAMAFILE + single-CUDA-stream smoke test"
    )
    parser.add_argument("--threads", type=int, default=128, help="CPUInfer worker threads")
    parser.add_argument("--threadpools", type=int, default=1, help="CPUInfer NUMA subpools")
    parser.add_argument("--iterations", type=int, default=1000, help="Measured iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up iterations")
    parser.add_argument("--batch", type=int, default=1, help="Tokens per iteration")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="MoE layer to test; default is the first layer present in the GGUF",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def nested_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def first_int(config: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = config.get(name)
        if value is not None:
            return int(value)
    return None


def find_moe_layers(loader: GGUFLoader) -> list[int]:
    pattern = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
    projections_by_layer: dict[int, set[str]] = {}
    for name in loader.tensor_info:
        match = pattern.match(name)
        if match:
            layer_idx = int(match.group(1))
            projections_by_layer.setdefault(layer_idx, set()).add(match.group(2))
    return sorted(
        layer_idx
        for layer_idx, projections in projections_by_layer.items()
        if projections == {"gate", "up", "down"}
    )


def tensor_description(loader: GGUFLoader, name: str) -> tuple[list[int], str]:
    info = loader.tensor_info[name]
    dtype = info["dtype"]
    dtype_name = getattr(dtype, "name", str(dtype))
    return info["shape"], dtype_name


def resolve_model_dimensions(
    config: dict[str, Any], loader: GGUFLoader, layer_idx: int
) -> tuple[int, int, int, int]:
    text_config = nested_text_config(config)
    gguf_config = loader.get_model_config(layer_idx)

    hidden_size = first_int(text_config, "hidden_size")
    intermediate_size = first_int(
        text_config,
        "moe_intermediate_size",
        "expert_intermediate_size",
    )
    num_experts = first_int(
        text_config,
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
    )
    topk = first_int(
        text_config,
        "num_experts_per_tok",
        "num_experts_per_token",
        "num_selected_experts",
    )

    hidden_size = hidden_size or gguf_config["hidden_size"]
    intermediate_size = intermediate_size or gguf_config["moe_intermediate_size"]
    num_experts = num_experts or gguf_config["num_experts"]
    topk = topk or gguf_config["num_experts_per_tok"]

    values = {
        "hidden_size": hidden_size,
        "moe_intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "num_experts_per_tok": topk,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise RuntimeError(f"Cannot determine model fields: {', '.join(missing)}")

    return int(hidden_size), int(intermediate_size), int(num_experts), int(topk)


def validate_expert_tensors(
    loader: GGUFLoader,
    layer_idx: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
) -> None:
    expected_shapes = {
        "gate": [num_experts, intermediate_size, hidden_size],
        "up": [num_experts, intermediate_size, hidden_size],
        "down": [num_experts, hidden_size, intermediate_size],
    }

    print(f"\nGGUF expert tensors in layer {layer_idx}:")
    for projection, expected_shape in expected_shapes.items():
        name = f"blk.{layer_idx}.ffn_{projection}_exps.weight"
        shape, dtype_name = tensor_description(loader, name)
        print(f"  {projection:4s}: shape={shape}, dtype={dtype_name}")
        if dtype_name != "Q8_0":
            raise RuntimeError(f"Expected {name} to be Q8_0, got {dtype_name}")
        if list(shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected shape for {name}: got {shape}, expected {expected_shape}. "
                "The Hugging Face config and GGUF may not describe the same model."
            )


def validate_q8_scales(loader: GGUFLoader, layer_idx: int) -> None:
    """Check every Q8_0 block scale directly in the memory-mapped GGUF."""
    q8_block_elements = 32
    q8_block_bytes = 34
    invalid_tensors: list[str] = []

    print("\nQ8_0 block-scale validation:")
    for projection in ("gate", "up", "down"):
        name = f"blk.{layer_idx}.ffn_{projection}_exps.weight"
        info = loader.tensor_info[name]
        n_elements = int(info["n_elements"])
        if n_elements % q8_block_elements != 0:
            raise RuntimeError(
                f"{name} has {n_elements} elements, not a multiple of "
                f"Q8_0 block size {q8_block_elements}"
            )

        file_path = loader.tensor_file_map[name]
        mmap_data = loader.file_data_map[file_path]
        n_blocks = n_elements // q8_block_elements
        scales = np.ndarray(
            shape=(n_blocks,),
            dtype="<f2",
            buffer=mmap_data,
            offset=int(info["offset"]),
            strides=(q8_block_bytes,),
        )
        nan_count = int(np.isnan(scales).sum())
        inf_count = int(np.isinf(scales).sum())
        finite = np.isfinite(scales)
        abs_max = (
            float(np.max(np.abs(scales[finite].astype(np.float32))))
            if bool(finite.any())
            else float("nan")
        )
        print(
            f"  {projection:4s}: blocks={n_blocks:,}, nan={nan_count:,}, "
            f"inf={inf_count:,}, max_abs_scale={abs_max:.6e}"
        )
        if nan_count or inf_count:
            invalid_tensors.append(name)

    if invalid_tensors:
        raise RuntimeError(
            "GGUF contains non-finite Q8_0 scales in: " + ", ".join(invalid_tensors)
        )


def make_cpu_inputs(
    batch: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = (
        torch.randn(
            (batch, hidden_size),
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        .mul_(0.02)
        .to(torch.bfloat16)
        .contiguous()
    )
    router_scores = torch.rand(
        (batch, num_experts), dtype=torch.float32, generator=generator
    )
    topk_ids = torch.topk(router_scores, k=topk, dim=-1, sorted=False).indices.contiguous()
    topk_weights = torch.rand(
        (batch, topk), dtype=torch.float32, device="cpu", generator=generator
    )
    topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True))
    return hidden_states, topk_ids, topk_weights.contiguous()


def output_health(label: str, output: torch.Tensor) -> bool:
    values = output.detach().float().cpu()
    nan_count = int(torch.isnan(values).sum().item())
    inf_count = int(torch.isinf(values).sum().item())
    finite = torch.isfinite(values)
    finite_values = values[finite]
    if finite_values.numel():
        value_min = float(finite_values.min().item())
        value_max = float(finite_values.max().item())
        abs_max = float(finite_values.abs().max().item())
    else:
        value_min = value_max = abs_max = float("nan")
    print(
        f"  {label:<18s}: nan={nan_count:,}, inf={inf_count:,}, "
        f"min={value_min:+.6e}, max={value_max:+.6e}, abs_max={abs_max:.6e}",
        flush=True,
    )
    return nan_count == 0 and inf_count == 0 and abs_max > 0.0


def run_direct_cpu(
    wrapper: LlamafileMoEWrapper,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    batch = hidden_states.shape[0]
    output = torch.full_like(hidden_states, float("nan"), device="cpu")
    batch_tensor = torch.tensor([batch], dtype=torch.int32, device="cpu")
    wrapper.cpu_infer.submit(
        wrapper.moe.forward_task(
            batch_tensor.data_ptr(),
            topk_ids.shape[-1],
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            hidden_states.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    wrapper.cpu_infer.sync()
    return output


def run_cuda_stream_with_inputs(
    wrapper: LlamafileMoEWrapper,
    hidden_states_cpu: torch.Tensor,
    topk_ids_cpu: torch.Tensor,
    topk_weights_cpu: torch.Tensor,
    device: torch.device,
    stream: torch.cuda.Stream,
    gpu_probe: torch.Tensor,
) -> torch.Tensor:
    with torch.cuda.stream(stream):
        hidden_states = hidden_states_cpu.to(device)
        topk_ids = topk_ids_cpu.to(device)
        topk_weights = topk_weights_cpu.to(device)
        wrapper.submit_forward(
            hidden_states, topk_ids, topk_weights, stream.cuda_stream
        )
        gpu_probe.add_(1)
        output = wrapper.sync_forward(hidden_states, stream.cuda_stream)

    stream.synchronize()
    return output.detach().cpu()


def diagnostic_preflight(
    wrapper: LlamafileMoEWrapper,
    batch: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    seed: int,
    device: torch.device,
    stream: torch.cuda.Stream,
    gpu_probe: torch.Tensor,
) -> None:
    hidden_states, topk_ids, topk_weights = make_cpu_inputs(
        batch, hidden_size, num_experts, topk, seed
    )

    print("\nDiagnostic preflight with identical inputs:", flush=True)
    cpu_output = run_direct_cpu(wrapper, hidden_states, topk_ids, topk_weights)
    if not output_health("direct CPUInfer", cpu_output):
        raise RuntimeError(
            "direct CPUInfer Q8_0 forward produced invalid output before any CUDA "
            "transfer; the failure is in the ARM LLAMAFILE Q8_0 compute path, not "
            "CUDA stream synchronization"
        )

    cuda_output = run_cuda_stream_with_inputs(
        wrapper,
        hidden_states,
        topk_ids,
        topk_weights,
        device,
        stream,
        gpu_probe,
    )
    if not output_health("CUDA stream path", cuda_output):
        raise RuntimeError(
            "direct CPUInfer passed, but the CUDA stream path produced invalid "
            "output; inspect pinned-memory copies and stream synchronization"
        )

    difference = (cuda_output.float() - cpu_output.float()).abs()
    max_abs_diff = float(difference.max().item())
    matches = torch.allclose(
        cuda_output.float(), cpu_output.float(), rtol=2e-2, atol=2e-3
    )
    print(f"  CPU/CUDA max diff : {max_abs_diff:.6e}", flush=True)
    if not matches:
        raise RuntimeError(
            "CPU and CUDA-stream paths are both finite but disagree beyond tolerance"
        )


def make_inputs(
    batch: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn(
        (batch, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)

    # Router top-k IDs are unique for each token.  Taking top-k from random
    # scores models that property without constructing a randperm per token.
    router_scores = torch.rand(
        (batch, num_experts), device=device, generator=generator
    )
    topk_ids = torch.topk(router_scores, k=topk, dim=-1, sorted=False).indices
    topk_weights = torch.rand(
        (batch, topk), dtype=torch.float32, device=device, generator=generator
    )
    topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True))
    return hidden_states, topk_ids, topk_weights


def run_one(
    wrapper: LlamafileMoEWrapper,
    batch: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    device: torch.device,
    stream: torch.cuda.Stream,
    generator: torch.Generator,
    gpu_probe: torch.Tensor,
) -> torch.Tensor:
    with torch.cuda.stream(stream):
        hidden_states, topk_ids, topk_weights = make_inputs(
            batch, hidden_size, num_experts, topk, device, generator
        )
        wrapper.submit_forward(
            hidden_states, topk_ids, topk_weights, stream.cuda_stream
        )

        # Enqueue an independent GPU kernel between CPU submit and sync.  This
        # verifies that ordinary CUDA work can coexist on the integrated stream.
        gpu_probe.add_(1)
        output = wrapper.sync_forward(hidden_states, stream.cuda_stream)

    stream.synchronize()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("MoE output contains NaN or Inf")
    if float(output.abs().max().item()) == 0.0:
        raise RuntimeError("MoE output is all zeros")
    return output


def main() -> int:
    args = parse_args()
    try:
        require_file(MODEL_CONFIG, "config.json")
        require_file(GGUF_PATH, "GGUF")

        machine = platform.machine().lower()
        if machine not in {"aarch64", "arm64"}:
            raise RuntimeError(f"This test must run on ARM64, got {machine}")
        if __cpu_variant__ != "arm":
            raise RuntimeError(
                f"kt-kernel loaded CPU variant {__cpu_variant__!r}, expected 'arm'"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        if not hasattr(kt_kernel_ext.CPUInfer, "submit_with_cuda_stream"):
            raise RuntimeError(
                "kt_kernel_ext was built without CUDA-stream integration; rebuild with "
                "CPUINFER_USE_CUDA=1"
            )
        if args.threads <= 0 or args.threadpools <= 0:
            raise ValueError("--threads and --threadpools must be positive")
        if args.threads < args.threadpools:
            raise ValueError("--threads must be >= --threadpools")
        if args.iterations <= 0 or args.warmup < 0 or args.batch <= 0:
            raise ValueError("iterations/batch must be positive and warmup non-negative")

        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        print("=" * 72)
        print("Step 1: ARM LLAMAFILE Q8_0 + CUDA stream")
        print(f"  machine       : {machine}")
        print(f"  kt CPU variant: {__cpu_variant__}")
        print(f"  kt extension  : {Path(kt_kernel_ext.__file__).resolve()}")
        print(f"  CUDA device   : {props.name}")
        print(f"  config        : {MODEL_CONFIG}")
        print(f"  GGUF          : {GGUF_PATH}")
        print("=" * 72)

        with MODEL_CONFIG.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        # Parse and mmap the GGUF once, then reuse that loader in the wrapper.
        loader = GGUFLoader(str(GGUF_PATH))
        moe_layers = find_moe_layers(loader)
        if not moe_layers:
            raise RuntimeError("No complete gate/up/down MoE layer found in the GGUF")
        layer_idx = args.layer if args.layer is not None else moe_layers[0]
        if layer_idx not in moe_layers:
            raise ValueError(
                f"Layer {layer_idx} has no complete MoE weights; available layers: {moe_layers}"
            )

        hidden_size, intermediate_size, num_experts, topk = resolve_model_dimensions(
            config, loader, layer_idx
        )
        if intermediate_size % 256 != 0:
            raise RuntimeError(
                f"moe_intermediate_size={intermediate_size} is not divisible by 256, "
                "which the current LLAMAFILE TP path requires"
            )
        if topk > num_experts:
            raise RuntimeError(f"topk={topk} exceeds num_experts={num_experts}")

        print("\nResolved model configuration:")
        print(f"  layer                 : {layer_idx}")
        print(f"  hidden_size           : {hidden_size}")
        print(f"  moe_intermediate_size : {intermediate_size}")
        print(f"  num_experts           : {num_experts}")
        print(f"  num_experts_per_tok   : {topk}")
        print(f"  CPU threads           : {args.threads}")
        print(f"  CPU threadpools       : {args.threadpools}")
        print(f"  batch                 : {args.batch}")
        print(f"  warmup / iterations   : {args.warmup} / {args.iterations}")

        validate_expert_tensors(
            loader, layer_idx, hidden_size, intermediate_size, num_experts
        )
        validate_q8_scales(loader, layer_idx)

        # Standalone KTMoEWrapper does not own GPU expert weights.  Keep every
        # expert on CPU so the returned output is complete; CUDA is used for
        # input/output transfer and stream synchronization in this first test.
        gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool, device="cpu")
        LlamafileMoEWrapper._gguf_loader_instance = loader
        wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=topk,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=args.threads,
            threadpool_count=args.threadpools,
            weight_path=str(GGUF_PATH),
            chunked_prefill_size=max(512, args.batch),
            max_deferred_experts_per_token=0,
            method="LLAMAFILE",
        )

        print("\nLoading one Q8_0 MoE layer into CPUInfer ...", flush=True)
        load_start = time.perf_counter()
        wrapper.load_weights()
        print(f"Weights loaded in {time.perf_counter() - load_start:.2f} s", flush=True)

        KTMoEWrapper.set_capture_batch_sizes([args.batch])
        stream = torch.cuda.Stream(device=device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        gpu_probe = torch.zeros(4096, dtype=torch.float32, device=device)

        diagnostic_preflight(
            wrapper,
            args.batch,
            hidden_size,
            num_experts,
            topk,
            args.seed,
            device,
            stream,
            gpu_probe,
        )

        print(f"\nWarming up ({args.warmup} iterations) ...", flush=True)
        output = None
        for _ in range(args.warmup):
            output = run_one(
                wrapper,
                args.batch,
                hidden_size,
                num_experts,
                topk,
                device,
                stream,
                generator,
                gpu_probe,
            )

        print(f"Running stability test ({args.iterations} iterations) ...", flush=True)
        start = time.perf_counter()
        report_every = max(1, min(50, args.iterations // 10 or 1))
        for iteration in range(1, args.iterations + 1):
            output = run_one(
                wrapper,
                args.batch,
                hidden_size,
                num_experts,
                topk,
                device,
                stream,
                generator,
                gpu_probe,
            )
            if iteration % report_every == 0 or iteration == args.iterations:
                checksum = float(output.float().sum().item())
                print(
                    f"  ok {iteration:5d}/{args.iterations}: checksum={checksum:+.6e}",
                    flush=True,
                )

        elapsed = time.perf_counter() - start
        tokens = args.iterations * args.batch
        print("\n" + "=" * 72)
        print("PASS: all iterations completed with finite, non-zero output")
        print(f"  elapsed       : {elapsed:.3f} s")
        print(f"  average       : {elapsed * 1000 / args.iterations:.3f} ms/iteration")
        print(f"  test rate     : {tokens / elapsed:.3f} token/s (one MoE layer only)")
        print("  MAX_DEFER     : 0")
        print("  GPU experts   : 0 (CUDA transfer/stream test; full hybrid is step 2)")
        print("=" * 72)
        return 0
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
