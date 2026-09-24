# Base classes for MoE CPU inference operations
# SPDX-License-Identifier: Apache-2.0

"""
Base infrastructure for CPU-based MoE inference.

This module contains base classes and utilities shared across all backend implementations.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from kt_kernel import kt_kernel_ext

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# NPU stream-callback bypass.
#
# On Ascend NPU, `CPUInfer::submit_with_cuda_stream` calls `aclrtLaunchCallback`,
# which inserts the function into a per-stream **callback report queue**. ACL
# requires a dedicated subscriber thread (registered via `aclrtSubscribeReport`
# and continuously running `aclrtProcessReport`) to dispatch those callbacks.
#
# Without such a subscriber, queued callbacks **silently never fire**, so the
# CPU forward task never runs and `output_cpu` stays all-zero.
# `sync_with_cuda_stream` likewise schedules its sync_ via the same callback
# queue and silently completes without actually syncing anything.
#
# kt-kernel therefore starts the subscriber itself (see
# `cpu_backend/ascend_callback_worker.cpp`, entered via
# ``init_ascend_callback_worker``); ``subscribe_ascend_stream`` registers each
# stream with it, after which ``submit_with_cuda_stream`` callbacks are
# dispatched and CPU/NPU overlap works.  Set ``KT_FORCE_SYNC_SUBMIT=1`` to fall
# back to the synchronous submit/sync path for debugging.
# -----------------------------------------------------------------------------


# Memoized result of the C++ worker probe. None = not probed yet.
_ascend_worker_degraded: Optional[bool] = None


def _ascend_callback_worker_running() -> Optional[bool]:
    """Probe the C++ callback worker; None when the build predates the probe."""
    probe = getattr(kt_kernel_ext, "is_ascend_callback_worker_running", None)
    if probe is None:
        return None
    try:
        return bool(probe())
    except Exception:
        return None


def _ascend_callback_worker_degraded() -> bool:
    """True iff the ACL callback worker is known NOT to be dispatching reports.

    Probed once and memoized: the worker's state only changes at start-up and at
    interpreter shutdown, and this sits on the per-layer forward path. Builds
    without the probe are treated as healthy (backwards compatible).
    """
    global _ascend_worker_degraded
    if _ascend_worker_degraded is None:
        _ascend_worker_degraded = _ascend_callback_worker_running() is False
    return _ascend_worker_degraded


def _should_bypass_stream_callback(device: torch.device) -> bool:
    """Return True iff we must use the synchronous submit/sync path."""
    if os.environ.get("KT_FORCE_SYNC_SUBMIT", "") == "1":
        return True
    if device.type == "npu" and not _uses_external_npu_report_subscriber():
        # Without a dispatcher thread, aclrtLaunchCallback tasks never fire and
        # submit_with_cuda_stream would hang forever. Degrade to sync instead.
        if _ascend_callback_worker_degraded():
            return True
    return False


def _uses_external_npu_report_subscriber() -> bool:
    return os.environ.get("KT_EXTERNAL_NPU_REPORT_SUBSCRIBER", "") == "1"


def _ensure_ascend_callback_worker() -> None:
    """Start kt-kernel ACL callback worker (idempotent)."""
    if _uses_external_npu_report_subscriber():
        return
    if not hasattr(kt_kernel_ext, "init_ascend_callback_worker"):
        return
    if getattr(_ensure_ascend_callback_worker, "_done", False):
        return
    kt_kernel_ext.init_ascend_callback_worker()
    _ensure_ascend_callback_worker._done = True  # type: ignore[attr-defined]

    global _ascend_worker_degraded
    _ascend_worker_degraded = _ascend_callback_worker_running() is False
    if _ascend_worker_degraded:
        logger.warning(
            "kt-kernel ACL callback worker failed to start (no ACL context?); "
            "falling back to the synchronous CPU-MoE submit/sync path."
        )
    if hasattr(kt_kernel_ext, "shutdown_ascend_callback_worker"):
        import atexit

        atexit.register(kt_kernel_ext.shutdown_ascend_callback_worker)


def _sglang_is_capture_mode() -> bool:
    """True when sglang is inside ``model_capture_mode()`` (graph capture).

    kt-kernel must remain importable standalone, so the sglang dependency is
    optional and any failure is treated as "not capturing".
    """
    try:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return False


def _get_torch_device_module(device: torch.device):
    get_device_module = getattr(torch, "get_device_module", None)
    if callable(get_device_module):
        try:
            return get_device_module(device)
        except Exception:
            pass
    return getattr(torch, device.type, None)


def _device_graph_capture_state(device: torch.device) -> Optional[bool]:
    """Return capture state, or None when the backend cannot report it."""
    if _sglang_is_capture_mode():
        return True
    if device.type in ("cpu", "meta"):
        return False

    backend = _get_torch_device_module(device)
    capture_probe = getattr(backend, "is_current_stream_capturing", None)
    if not callable(capture_probe):
        return None
    try:
        return bool(capture_probe())
    except Exception:
        return None


def _wait_device(device: torch.device) -> None:
    """Block until pending async copies on `device`'s current stream finish.

    Synchronization raises while a stream is being captured, so captured paths
    rely on stream-ordered host callbacks and retain their pinned buffers.
    """
    if device.type in ("cpu", "meta"):
        return

    capture_state = _device_graph_capture_state(device)
    if capture_state is True:
        return
    if capture_state is None:
        raise RuntimeError(
            f"Cannot safely synchronize torch.{device.type}: graph capture "
            "state is unavailable."
        )

    backend = _get_torch_device_module(device)
    synchronize = getattr(backend, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError(
            f"Cannot safely drain pending KT callbacks: torch.{device.type} "
            "does not provide synchronize()."
        )
    synchronize(device)


def _graph_capture_active(device: torch.device) -> bool:
    """True while a stream capture may still record host nodes referencing our buffers."""
    # Unknown accelerator backends fail closed: freeing a callback argument
    # during graph replay is a use-after-free.
    return _device_graph_capture_state(device) is not False


def _forward_task_autofree(sync_submit: bool, device: torch.device) -> bool:
    """Return the legacy single-use hint carried by forward-task bindings.

    CPUInfer now owns actual task destruction. The hint remains in the binding
    ABI, while stream submissions separately pass the authoritative capture
    state to CPUInfer.
    """
    return sync_submit or not _graph_capture_active(device)


@contextmanager
def _temporary_all_cpu_experts_mask(
    gpu_experts_mask: torch.Tensor,
    enabled: bool,
):
    """Temporarily make every expert loadable by CPU backends.

    C++ retains ``gpu_experts_mask.data_ptr()``, so the tensor must be modified
    and restored in place. The caller must keep this context active until the
    asynchronous load task has been synchronized.
    """
    if not enabled:
        yield
        return

    saved_mask = gpu_experts_mask.clone()
    try:
        gpu_experts_mask.zero_()
        yield
    finally:
        gpu_experts_mask.copy_(saved_mask)


def generate_gpu_experts_masks(
    activation_freq: torch.Tensor,
    num_gpu_experts: int,
) -> torch.Tensor:
    """
    Generate GPU experts masks based on activation frequency.

    Selects the top `num_gpu_experts` experts with highest activation frequency
    across all layers to be placed on GPU.

    Args:
        activation_freq: Activation frequency table of shape (num_layers, num_experts).
                         Higher values indicate more frequently activated experts.
        num_gpu_experts: Total number of experts to place on GPU across all layers.

    Returns:
        gpu_experts_masks: Boolean mask of shape (num_layers, num_experts) on CPU.
                           True means the expert should be on GPU.

    Example:
        >>> activation_freq = torch.tensor([
        ...     [0.1, 0.5, 0.3, 0.8],  # layer 0
        ...     [0.2, 0.4, 0.9, 0.1],  # layer 1
        ... ])
        >>> masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=3)
        >>> # Top 3: layer0-expert3 (0.8), layer1-expert2 (0.9), layer0-expert1 (0.5)
        >>> masks
        tensor([[False,  True, False,  True],
                [False, False,  True, False]])
    """
    num_layers, num_experts_per_layer = activation_freq.shape
    total_experts = num_layers * num_experts_per_layer

    # Clamp num_gpu_experts to valid range
    num_gpu_experts = min(num_gpu_experts, total_experts)
    num_gpu_experts = max(num_gpu_experts, 0)

    if num_gpu_experts == 0:
        return torch.zeros(num_layers, num_experts_per_layer, dtype=torch.bool, device="cpu")

    # Flatten and find top-k indices
    flat_freq = activation_freq.view(-1).to(device="cpu")
    _, top_indices = torch.topk(flat_freq, k=num_gpu_experts, largest=True, sorted=False)

    # Create mask
    gpu_experts_masks = torch.zeros(total_experts, dtype=torch.bool, device="cpu")
    gpu_experts_masks[top_indices] = True

    # Reshape to (num_layers, num_experts)
    gpu_experts_masks = gpu_experts_masks.view(num_layers, num_experts_per_layer)

    return gpu_experts_masks


class KExpertsCPUBuffer:
    """
    CPU buffer management for expert computation.

    Manages pinned memory buffers for efficient GPU-CPU data transfer.
    """

    capture_bs: List = list()
    capture_buffers: Dict = dict()
    # Bounded LRU of buffers for sizes not in capture_bs. The old single-slot
    # temp_buffer dropped the previous pinned tensors on every size change
    # while deferred CPU tasks could still hold their raw pointers; workers
    # then wrote into freed/unregistered host memory and the process died
    # much later at an unrelated site (e.g. inside cuGraphLaunch). Buffers
    # here are only dropped via _evict_oldest_temp_buffer, which drains first.
    temp_buffers: OrderedDict = OrderedDict()
    max_temp_buffer_sizes: int = int(os.environ.get("KT_MAX_TEMP_BUFFER_SIZES", "4"))
    buffer_depth: int = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok):
        hidden_size = hidden_states.shape[-1]
        batch_size = hidden_states.shape[0]

        pin_memory = True

        if batch_size in cls.capture_buffers:
            return cls.capture_buffers[batch_size]
        if batch_size in cls.temp_buffers:
            if batch_size in cls.capture_bs or _graph_capture_active(hidden_states.device):
                # The buffer predates its registration (e.g. EagerRunner warm-up
                # runs before set_capture_batch_sizes) or is being recorded into
                # a graph right now. It must leave the LRU: captured host nodes
                # dereference the pinned pointers on every replay, long after
                # eviction would have freed them.
                cls.capture_buffers[batch_size] = cls.temp_buffers.pop(batch_size)
                return cls.capture_buffers[batch_size]
            cls.temp_buffers.move_to_end(batch_size)
            return cls.temp_buffers[batch_size]

        input_tensor_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        immediate_experts_ids_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        deferred_experts_ids_cpu = [
            torch.full((batch_size, num_experts_per_tok), -1, device="cpu", dtype=torch.long, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.bfloat16)
            for _ in range(cls.buffer_depth)
        ]
        bsz_tensor_cpu = [
            torch.full((1,), batch_size, device="cpu", dtype=torch.int32, pin_memory=pin_memory)
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.zeros((batch_size, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            for _ in range(cls.buffer_depth)
        ]

        cur_buffer = (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            output_gpu,
        )
        if batch_size in cls.capture_bs or _graph_capture_active(hidden_states.device):
            # Host nodes captured into CUDA graphs dereference these pinned
            # pointers on every replay; never evict. Registration should cover
            # every captured shape, but prefer extra pinned memory over a
            # dangling pointer if a capture path misses it.
            cls.capture_buffers[batch_size] = cur_buffer
            return cur_buffer

        cls.temp_buffers[batch_size] = cur_buffer
        if len(cls.temp_buffers) > cls.max_temp_buffer_sizes:
            cls._evict_oldest_temp_buffer(hidden_states.device)
        return cur_buffer

    @classmethod
    def _evict_oldest_temp_buffer(cls, device: torch.device) -> None:
        """Drop the LRU temp buffer only after no dangling pointer can remain.

        CPU workers may still hold raw pointers into the evicted pinned tensors:
        tasks submitted via submit_with_cuda_stream only enter the CPU queue
        when the stream's host callback fires, so the device must be drained
        before draining the CPU task queue.
        """
        if _graph_capture_active(device):
            # Cannot synchronize during capture; prefer extra pinned memory
            # over a dangling pointer in a captured host node.
            return
        _wait_device(device)
        cpu_infer = _MoEBase._cpu_infer_instance
        if cpu_infer is not None:
            cpu_infer.sync(0)
        cls.temp_buffers.popitem(last=False)


class _MoEBase:
    """
    Shared base class for inference and SFT MoE wrappers.

    Provides:
    - CPUInfer singleton management
    - Basic configuration validation

    This class is shared between BaseMoEWrapper (inference) and BaseSFTMoEWrapper (SFT).
    """

    _cpu_infer_instance = None
    _cpu_infer_key = None
    _writer_cpu_infer_instance = None
    _writer_cpu_infer_key = None

    @classmethod
    def _get_cpu_infer(
        cls,
        cpuinfer_threads: int,
        threadpool_count: int,
        numa_nodes=None,
    ):
        """
        Get or create the CPUInfer singleton instance.

        Args:
            cpuinfer_threads: Total number of CPU inference threads
            threadpool_count: Number of NUMA subpools (TP count)
            numa_nodes: Explicit list of NUMA node IDs. If None, defaults to sequential.

        Returns:
            CPUInfer singleton instance
        """
        numa_map = (
            list(numa_nodes)
            if numa_nodes is not None
            else list(range(threadpool_count))
        )
        key = (int(cpuinfer_threads), int(threadpool_count), tuple(numa_map))
        if cls._cpu_infer_instance is not None:
            if cls._cpu_infer_key != key:
                raise RuntimeError(
                    "CPUInfer was initialized with a different CPU/NUMA "
                    "configuration in this process"
                )
            return cls._cpu_infer_instance
        if cls._cpu_infer_instance is None:
            if threadpool_count <= 0:
                raise ValueError(
                    f"threadpool_count must be positive, got {threadpool_count}"
                )
            if cpuinfer_threads < threadpool_count:
                raise ValueError(
                    "cpuinfer_threads must be at least threadpool_count so every "
                    f"NUMA subpool has a worker (got {cpuinfer_threads} and "
                    f"{threadpool_count})"
                )
            try:
                if torch.npu.is_available():  # type: ignore[attr-defined]
                    _ensure_ascend_callback_worker()
            except Exception:
                pass
            worker_config = kt_kernel_ext.WorkerPoolConfig()

            if numa_nodes is not None:
                if len(numa_nodes) != threadpool_count:
                    raise ValueError(
                        f"numa_nodes length ({len(numa_nodes)}) must match " f"threadpool_count ({threadpool_count})"
                    )
                subpool_numa_map = list(numa_nodes)
            else:
                subpool_numa_map = list(range(threadpool_count))
            subpool_thread_count = [
                cpuinfer_threads // threadpool_count + (1 if i < cpuinfer_threads % threadpool_count else 0)
                for i in range(threadpool_count)
            ]

            worker_config.subpool_count = threadpool_count
            worker_config.subpool_numa_map = subpool_numa_map
            worker_config.subpool_thread_count = subpool_thread_count
            started = time.perf_counter()
            logger.info(
                "[KT] Initializing CPUInfer: threads=%d, subpools=%d, "
                "numa_nodes=%s, threads_per_subpool=%s",
                cpuinfer_threads,
                threadpool_count,
                subpool_numa_map,
                subpool_thread_count,
            )
            cls._cpu_infer_instance = kt_kernel_ext.CPUInfer(worker_config)
            cls._cpu_infer_key = key
            logger.info(
                "[KT] CPUInfer initialized in %.2fs",
                time.perf_counter() - started,
            )

        return cls._cpu_infer_instance

    @classmethod
    def _get_writer_cpu_infer(
        cls,
        cpuinfer_threads: int,
        threadpool_count: int,
        numa_nodes=None,
    ):
        """Return the process-wide streamed-expert writer executor.

        Writers must not share CPUInfer's FIFO or WorkerPool with the main MoE
        task: otherwise rolling candidate 3+ is queued behind the long CPU
        GEMM. One auxiliary worker per CPU TP/NUMA partition keeps the default
        bounded while allowing host export to overlap the main computation.
        """

        numa_map = (
            list(numa_nodes)
            if numa_nodes is not None
            else list(range(threadpool_count))
        )
        if len(numa_map) != threadpool_count:
            raise ValueError(
                f"numa_nodes length ({len(numa_map)}) must match "
                f"threadpool_count ({threadpool_count})"
            )
        key = (int(cpuinfer_threads), int(threadpool_count), tuple(numa_map))
        if cls._writer_cpu_infer_instance is not None:
            if cls._writer_cpu_infer_key != key:
                raise RuntimeError(
                    "streamed-expert writer CPUInfer was initialized with a "
                    "different CPU/NUMA configuration"
                )
            return cls._writer_cpu_infer_instance

        main_thread_counts = [
            cpuinfer_threads // threadpool_count
            + (1 if i < cpuinfer_threads % threadpool_count else 0)
            for i in range(threadpool_count)
        ]
        main_threads_by_numa = {}
        for numa_id, thread_count in zip(numa_map, main_thread_counts):
            main_threads_by_numa[numa_id] = (
                main_threads_by_numa.get(numa_id, 0) + thread_count
            )
        writer_seen_by_numa = {}
        writer_thread_starts = []
        for numa_id in numa_map:
            writer_offset = writer_seen_by_numa.get(numa_id, 0)
            writer_thread_starts.append(
                main_threads_by_numa[numa_id] + writer_offset
            )
            writer_seen_by_numa[numa_id] = writer_offset + 1

        writer_config = kt_kernel_ext.WorkerPoolConfig()
        writer_config.subpool_count = threadpool_count
        writer_config.subpool_numa_map = numa_map
        writer_config.subpool_thread_count = [1] * threadpool_count
        if not hasattr(writer_config, "subpool_thread_start"):
            raise RuntimeError(
                "kt-kernel lacks auxiliary writer core-offset support; rebuild "
                "the extension from the kt-prefill-stream-top-n branch"
            )
        writer_config.subpool_thread_start = writer_thread_starts
        logger.info(
            "KT MXFP4 Stream-TopN CPU budget: total_threads=%d, "
            "main_gemm_threads=%d, writer_threads=%d, numa_map=%s",
            cpuinfer_threads + threadpool_count,
            cpuinfer_threads,
            threadpool_count,
            numa_map,
        )
        cls._writer_cpu_infer_instance = kt_kernel_ext.CPUInfer(writer_config)
        cls._writer_cpu_infer_key = key
        return cls._writer_cpu_infer_instance

    def _ensure_writer_cpu_infer(self):
        if self.writer_cpu_infer is None:
            cpuinfer_threads, threadpool_count, numa_nodes = (
                self._writer_cpuinfer_config
            )
            self.writer_cpu_infer = self._get_writer_cpu_infer(
                cpuinfer_threads,
                threadpool_count,
                numa_nodes=numa_nodes,
            )
        return self.writer_cpu_infer

    @staticmethod
    def _validate_base_config(
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts_per_tok: int,
    ) -> None:
        """
        Validate basic configuration parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if moe_intermediate_size <= 0:
            raise ValueError(f"moe_intermediate_size must be positive, got {moe_intermediate_size}")
        if num_experts_per_tok <= 0:
            raise ValueError(f"num_experts_per_tok must be positive, got {num_experts_per_tok}")
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed " f"num_experts ({num_experts})"
            )


class BaseMoEWrapper(_MoEBase, ABC):
    """
    Base class for MoE CPU inference operations.
    Provides common functionality for all backend implementations.
    """

    _layer_has_pending_deferred: Dict[int, bool] = {}

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
        swiglu_limit: float = 0.0,
        activation: str = "silu",
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        reserve_stream_writer_threads: bool = False,
    ):
        """
        Initialize base MoE Wrapper.

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
            weight_path: Path to weights
            chunked_prefill_size: Maximum prefill chunk size
            cpu_save: Whether to save weights to CPU memory
            max_deferred_experts_per_token: Number of experts per token to defer on this layer. Defaults to 0 (no defer).
            method: Backend method string
            numa_nodes: Explicit list of NUMA node IDs for subpool mapping.
                        If None, defaults to [0, 1, ..., threadpool_count-1].
        """
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        # Process gpu_experts_mask: convert to bool tensor on CPU, pinned memory for async copy
        # This mask is shared between C and Python (C uses uint8_t*), both can read/write it
        if gpu_experts_mask is None:
            # No GPU experts - all experts on CPU
            self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)
        else:
            # Create a new pinned tensor and copy data into it
            self.gpu_experts_mask = torch.empty(num_experts, dtype=torch.bool, device="cpu", pin_memory=True)
            self.gpu_experts_mask.copy_(gpu_experts_mask)

        self.num_gpu_experts = int(self.gpu_experts_mask.sum().item())

        # GPU copy for mask operations in forward pass (e.g., mask_cpu_expert_ids)
        # This will be lazily initialized when needed
        self._gpu_experts_mask_gpu: Optional[torch.Tensor] = None
        self.weight_path = weight_path
        self.chunked_prefill_size = chunked_prefill_size
        self.cpu_save = cpu_save
        self.max_deferred_experts_per_token = (
            int(max_deferred_experts_per_token) if max_deferred_experts_per_token is not None else 0
        )

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        self.method = method
        # V4-Flash 2604B SwiGLU clamp limit; 0.0 = disabled. NativeMoEWrapper
        # (MXFP4 path) reads this in load_weights() and writes it into
        # MOEConfig.swiglu_limit. Other backends ignore it (C++ act_fn skips
        # the clamp branch when limit==0). Origin: kt-sglang 耦合.
        self.swiglu_limit = float(swiglu_limit)
        self.activation = activation
        self.situ_beta = None if situ_beta is None else float(situ_beta)
        self.situ_linear_beta = (
            None if situ_linear_beta is None else float(situ_linear_beta)
        )

        # Initialize CPU inference engine (singleton via shared base class)
        main_cpuinfer_threads = int(cpuinfer_threads)
        self._reserve_stream_writer_threads = bool(
            reserve_stream_writer_threads
        )
        if self._reserve_stream_writer_threads:
            if str(method).upper() != "MXFP4":
                raise ValueError(
                    "stream writer CPU reservation is only valid for MXFP4"
                )
            # Treat --kt-cpuinfer as the total MXFP4 CPU budget. Reserve one
            # core per CPU TP/NUMA partition for the independent streaming
            # writer so deployments that already use every allowed core do not
            # need another tuning parameter.
            main_cpuinfer_threads -= int(threadpool_count)
            if main_cpuinfer_threads < int(threadpool_count):
                raise ValueError(
                    "MXFP4 Stream-TopN requires at least two CPU threads per "
                    "threadpool (one GEMM worker and one writer worker)"
                )
        self.cpu_infer = self._get_cpu_infer(
            main_cpuinfer_threads,
            threadpool_count,
            numa_nodes=numa_nodes,
        )
        # Created lazily only when Stream-TopN submits its first packed expert.
        self._writer_cpuinfer_config = (
            main_cpuinfer_threads,
            int(threadpool_count),
            None if numa_nodes is None else tuple(numa_nodes),
        )
        self.writer_cpu_infer = None

        # Backend-specific initialization happens in subclasses
        self.moe = None

    @abstractmethod
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
        pass

    @abstractmethod
    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):
        """
        Load weights for this layer and initialize the MoE module.

        Args:
            physical_to_logical_map_cpu: Mapping from physical to logical expert IDs
        """
        pass

    def select_deferred_experts(
        self,
        expert_ids: torch.Tensor,
        expert_scores: torch.Tensor,
        protected_k: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, topk = expert_ids.shape
        device = expert_ids.device

        protected_k = max(0, min(int(protected_k), topk))
        if protected_k == 0:
            deferred_ids = expert_ids.clone()
            immediate_ids = torch.full_like(expert_ids, -1)
            return immediate_ids, deferred_ids

        topk_result = torch.topk(expert_scores, k=protected_k, dim=-1, largest=True, sorted=False)
        protected_indices = topk_result.indices
        protected_ids = torch.gather(expert_ids, -1, protected_indices)

        protected_flag = torch.zeros((self.num_experts,), dtype=torch.int32, device=device)
        protected_flag.scatter_(0, protected_ids.reshape(-1), 1)

        protected_mask_flat = torch.gather(protected_flag, 0, expert_ids.reshape(-1)).ne(0)
        protected_mask = protected_mask_flat.view(batch, topk)

        immediate_ids = expert_ids.clone().masked_fill(~protected_mask, -1)
        deferred_ids = expert_ids.clone().masked_fill(protected_mask, -1)

        return immediate_ids, deferred_ids

    def _check_qlen_fits_cpp_buffers(self, hidden_states: torch.Tensor) -> None:
        """Fail loudly when qlen would overrun the C++ MoE output buffer.

        ``moe-tp.hpp`` sizes ``local_output_numa[i]`` by ``max_possible_qlen()`` =
        ``max(max_len, group_max_len)``, and both are set to ``chunked_prefill_size``
        (``utils/llamafile.py``). ``TP::forward`` then hands the *full* qlen to
        ``MOE::forward``, whose recursion only splits the internal scratch — the
        caller-supplied output pointer still advances across ``qlen * hidden_size``.
        So ``qlen > chunked_prefill_size`` writes past the allocation and corrupts the
        heap, surfacing later as an unrelated ``malloc(): unaligned tcache chunk``
        abort. Raise here instead, mirroring the SFT path (``sft/base.py``).
        """
        qlen = hidden_states.numel() // hidden_states.shape[-1]
        if qlen > self.chunked_prefill_size:
            raise ValueError(
                f"qlen ({qlen}) exceeds chunked_prefill_size ({self.chunked_prefill_size}); "
                "the C++ MoE output buffer is sized by chunked_prefill_size and would be "
                "overrun. Raise --chunked-prefill-size or reduce the prefill chunk."
            )

    def _prepare_forward_cpu_buffers(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], tuple, int, int]:
        """D2H copy into pinned CPU buffers; return deferred ids and buffer handles."""
        self._check_qlen_fits_cpp_buffers(hidden_states)
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        topk_ids_long = topk_ids.to(torch.long)
        if self.max_deferred_experts_per_token > 0:
            protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token
            immediate_ids, deferred_ids = self.select_deferred_experts(topk_ids_long, topk_weights, protected_k)
        else:
            immediate_ids = topk_ids_long
            deferred_ids = None

        input_tensor_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
        weights_cpu[current_slot].copy_(topk_weights, non_blocking=True)
        immediate_experts_ids_cpu[current_slot].copy_(immediate_ids, non_blocking=True)
        if deferred_ids is not None:
            deferred_experts_ids_cpu[current_slot].copy_(deferred_ids, non_blocking=True)

        buffers = (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        )
        return immediate_ids, deferred_ids, buffers, current_slot, next_slot

    def copy_inputs_to_cpu_buffers(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Copy MoE inputs to pinned CPU buffers (for NPU graph host callbacks)."""
        self._prepare_forward_cpu_buffers(hidden_states, topk_ids, topk_weights)

    def forward_on_pinned_buffers(
        self,
        hidden_states: torch.Tensor,
        cuda_stream,
    ) -> None:
        """Run CPU MoE on buffers already filled (sync or stream callback)."""
        self._check_qlen_fits_cpp_buffers(hidden_states)
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        bypass = _should_bypass_stream_callback(hidden_states.device)
        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        immediate_task = self.moe.forward_task(
            bsz_slot_tensor.data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
            True,  # Direct submit is a single-use task.
        )
        if bypass:
            self.cpu_infer.submit(immediate_task)
        else:
            # Correct + fast async path. ``submit_with_cuda_stream`` enqueues the
            # CPU-MoE via an ACL host callback whose firing is not host-observable
            # (NO_BLOCK) -> a later host-side drain can race ahead of it (empty
            # queue -> stale output_cpu read by the H2D; nondeterministic on heavy
            # prefill). Enqueue synchronously on this host thread instead (work is
            # guaranteed submitted; it still runs on the WorkerPool and overlaps the
            # GPU experts queued after). Keep subscribe_ascend_stream — that stream
            # registration is what keeps decode's host-callback dispatch fast; the
            # async submit itself is not required for it.
            if (
                hidden_states.device.type == "npu"
                and not _uses_external_npu_report_subscriber()
                and hasattr(kt_kernel_ext, "subscribe_ascend_stream")
            ):
                kt_kernel_ext.subscribe_ascend_stream(int(cuda_stream))
            self.cpu_infer.submit(immediate_task)

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        has_deferred = (
            self.max_deferred_experts_per_token > 0 and (deferred_experts_ids_cpu[current_slot] >= 0).any().item()
        )
        if has_deferred:
            deferred_task = self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                deferred_experts_ids_cpu[current_slot].size(-1),
                deferred_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[next_slot].data_ptr(),
                False,
                _forward_task_autofree(bypass, hidden_states.device),
            )
            if bypass:
                self.cpu_infer.submit(deferred_task)
            else:
                self.cpu_infer.submit_with_cuda_stream(
                    cuda_stream,
                    deferred_task,
                    _graph_capture_active(hidden_states.device),
                )
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True

    def run_pinned_forward_sync(
        self,
        hidden_states: torch.Tensor,
        cuda_stream,
    ) -> None:
        """Submit + sync CPU MoE on pre-filled buffers (NPU graph host callback).

        Called from ``aclrtLaunchCallback`` / ``_launch_host_func``; must not enqueue
        nested stream callbacks.
        """
        del cuda_stream  # unused — sync path only
        self._check_qlen_fits_cpp_buffers(hidden_states)
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        next_slot = (current_slot + 1) % KExpertsCPUBuffer.buffer_depth
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        immediate_task = self.moe.forward_task(
            bsz_slot_tensor.data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
            True,  # Direct submit is a single-use task.
        )
        self.cpu_infer.submit(immediate_task)
        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        has_deferred = (
            self.max_deferred_experts_per_token > 0 and (deferred_experts_ids_cpu[current_slot] >= 0).any().item()
        )
        if has_deferred:
            deferred_task = self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                deferred_experts_ids_cpu[current_slot].size(-1),
                deferred_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[next_slot].data_ptr(),
                False,
                True,  # Direct submit is a single-use task.
            )
            self.cpu_infer.submit(deferred_task)
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True
        allow_pending = 1 if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx, False) else 0
        self.cpu_infer.sync(allow_pending)

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ):
        """
        Submit forward inference task to CPU (non-blocking).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization
        """
        _immediate_ids, deferred_ids, _buffers, current_slot, next_slot = self._prepare_forward_cpu_buffers(
            hidden_states, topk_ids, topk_weights
        )
        (
            input_tensor_cpu,
            immediate_experts_ids_cpu,
            deferred_experts_ids_cpu,
            weights_cpu,
            output_cpu,
            bsz_tensor_cpu,
            _output_gpu,
        ) = _buffers
        bsz_slot_tensor = bsz_tensor_cpu[current_slot]

        bypass = _should_bypass_stream_callback(hidden_states.device)
        # NPU only: submit the CPU-MoE synchronously on this host thread.
        # ``submit_with_cuda_stream`` enqueues it via an ACL host callback whose
        # firing is not host-observable (NO_BLOCK) -> a later host-side drain can
        # race ahead of it (empty queue -> stale output_cpu read by the H2D;
        # nondeterministic on heavy prefill). The synchronous submit still runs on
        # the WorkerPool and overlaps the device experts queued after it.
        # CUDA keeps the upstream async submit_with_cuda_stream path unchanged.
        sync_submit = bypass or hidden_states.device.type == "npu"
        # Preserve the legacy hint; CPUInfer receives the capture state
        # separately and owns the actual task lifetime.
        autofree = _forward_task_autofree(sync_submit, hidden_states.device)
        if sync_submit:
            # The synchronous submit reads input_tensor_cpu immediately -> the input
            # D2H queued async on the stream by _prepare_forward_cpu_buffers MUST be
            # finished first, else the CPU MoE reads a half-copied input.
            _wait_device(hidden_states.device)

        incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
        immediate_task = self.moe.forward_task(
            bsz_slot_tensor.data_ptr(),
            immediate_experts_ids_cpu[current_slot].size(-1),
            immediate_experts_ids_cpu[current_slot].data_ptr(),
            weights_cpu[current_slot].data_ptr(),
            input_tensor_cpu[current_slot].data_ptr(),
            output_cpu[current_slot].data_ptr(),
            incremental,
            autofree,
        )
        if sync_submit:
            # Keep subscribe_ascend_stream — that stream registration is what keeps
            # decode's host-callback dispatch (sync_forward) fast; the async submit
            # itself is not required for it.
            if (
                not bypass
                and hidden_states.device.type == "npu"
                and not _uses_external_npu_report_subscriber()
                and hasattr(kt_kernel_ext, "subscribe_ascend_stream")
            ):
                kt_kernel_ext.subscribe_ascend_stream(int(cuda_stream))
            self.cpu_infer.submit(immediate_task)
        else:
            self.cpu_infer.submit_with_cuda_stream(
                cuda_stream,
                immediate_task,
                _graph_capture_active(hidden_states.device),
            )

        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = False
        if deferred_ids is not None:
            if sync_submit:
                _wait_device(hidden_states.device)
            deferred_task = self.moe.forward_task(
                bsz_slot_tensor.data_ptr(),
                deferred_experts_ids_cpu[current_slot].size(-1),
                deferred_experts_ids_cpu[current_slot].data_ptr(),
                weights_cpu[current_slot].data_ptr(),
                input_tensor_cpu[current_slot].data_ptr(),
                output_cpu[next_slot].data_ptr(),
                False,
                autofree,
            )
            if sync_submit:
                self.cpu_infer.submit(deferred_task)
            else:
                self.cpu_infer.submit_with_cuda_stream(
                    cuda_stream,
                    deferred_task,
                    _graph_capture_active(hidden_states.device),
                )
            BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True

    def copy_forward_output_to_device(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Copy pinned CPU output to the device tensor (CPU work already finished)."""
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            _input_tensor_cpu,
            _immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            _weights_cpu,
            output_cpu,
            _bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)
        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)
        return output_gpu[current_slot]

    def sync_forward(self, hidden_states: torch.Tensor, cuda_stream) -> torch.Tensor:
        """
        Synchronize and retrieve forward inference results.

        Args:
            hidden_states: Original input hidden states (for getting buffer)
            cuda_stream: CUDA stream for synchronization

        Returns:
            output_gpu: Output tensor on GPU
        """
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        (
            _input_tensor_cpu,
            _immediate_experts_ids_cpu,
            _deferred_experts_ids_cpu,
            _weights_cpu,
            output_cpu,
            _bsz_tensor_cpu,
            output_gpu,
        ) = KExpertsCPUBuffer.get_buffer(flat_hidden_states, self.num_experts_per_tok)

        current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth
        allow_pending = 1 if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx, False) else 0
        bypass = _should_bypass_stream_callback(hidden_states.device)
        if bypass:
            self.cpu_infer.sync(allow_pending)
        else:
            if (
                hidden_states.device.type == "npu"
                and not _uses_external_npu_report_subscriber()
                and hasattr(kt_kernel_ext, "subscribe_ascend_stream")
            ):
                kt_kernel_ext.subscribe_ascend_stream(int(cuda_stream))
            if hidden_states.device.type == "npu":
                # ACL host callbacks do not order the following H2D copy behind
                # the WorkerPool drain.  Wait for the accelerator work that
                # overlapped CPU MoE, then drain on the host before copying the
                # completed pinned output back to the NPU.
                _wait_device(hidden_states.device)
                self.cpu_infer.sync(allow_pending)
            else:
                self.cpu_infer.sync_with_cuda_stream(
                    cuda_stream,
                    allow_pending,
                    _graph_capture_active(hidden_states.device),
                )

        return self.copy_forward_output_to_device(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        cuda_stream,
    ) -> torch.Tensor:
        """
        Execute forward inference synchronously (submit + sync).

        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            topk_ids: Top-k expert IDs [batch_size, num_experts_per_tok]
            topk_weights: Top-k expert weights [batch_size, num_experts_per_tok]
            cuda_stream: CUDA stream for synchronization

        Returns:
            Output tensor on GPU
        """
        self.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
        return self.sync_forward(hidden_states, cuda_stream)

    @staticmethod
    def set_capture_batch_sizes(capture_bs: List[int]):
        """
        Set batch sizes to capture and cache buffers for.

        This allows pre-allocation of CPU buffers for specific batch sizes,
        improving performance by avoiding buffer re-allocation during inference.

        Args:
            capture_bs: List of batch sizes to capture (e.g., [1, 2, 4, 8, 16])

        Example:
            >>> BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])
        """
        KExpertsCPUBuffer.capture_bs = capture_bs

    @staticmethod
    def get_capture_batch_sizes() -> List[int]:
        """
        Get currently configured capture batch sizes.

        Returns:
            List of batch sizes that are being captured
        """
        return KExpertsCPUBuffer.capture_bs

    @staticmethod
    def clear_buffer_cache():
        """
        Clear evictable temp buffers after draining in-flight work.

        Capture buffers are never freed here: CUDA graph host nodes reference
        their pinned pointers on every replay, so dropping them while any
        graph may still be launched is a use-after-free. Must not be called
        during graph capture.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cpu_infer = _MoEBase._cpu_infer_instance
        if cpu_infer is not None:
            cpu_infer.sync(0)
        KExpertsCPUBuffer.temp_buffers.clear()
