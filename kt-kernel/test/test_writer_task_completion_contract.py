"""CPU-only contract tests for task-specific expert writer completion."""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

AMX_PATH = Path(__file__).resolve().parents[1] / "python/utils/amx.py"
EXPERTS_BASE_PATH = Path(__file__).resolve().parents[1] / "python/experts_base.py"
BINDINGS_PATH = Path(__file__).resolve().parents[1] / "ext_bindings.cpp"
WORKER_POOL_PATH = (
    Path(__file__).resolve().parents[1] / "cpu_backend/worker_pool.h"
)
MXFP4_BACKEND_PATHS = (
    Path(__file__).resolve().parents[1] / "operators/amx/fp4-moe.hpp",
    Path(__file__).resolve().parents[1] / "operators/avx2/mxfp4-moe.hpp",
    Path(__file__).resolve().parents[1] / "operators/arm/mxfp4-moe.hpp",
)


def _compile_method(name: str):
    tree = ast.parse(AMX_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeMoEWrapper"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function = copy.deepcopy(method)
    function.name = "invoke"
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(AMX_PATH), "exec"), namespace)  # noqa: S102
    return namespace["invoke"]


def _compile_base_method(name: str, namespace: dict):
    tree = ast.parse(EXPERTS_BASE_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_MoEBase"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function = copy.deepcopy(method)
    function.name = "invoke"
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(EXPERTS_BASE_PATH), "exec"), namespace)  # noqa: S102
    return namespace["invoke"]


class _Completion:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


class _FailingCompletion(_Completion):
    def wait(self):
        super().wait()
        raise RuntimeError("tracked writer failure")


class _CPUInfer:
    def __init__(self):
        self.submitted = []
        self.sync_count = 0

    def submit(self, task):
        self.submitted.append(task)

    def sync(self):
        self.sync_count += 1


class _Wrapper(SimpleNamespace):
    def _ensure_writer_cpu_infer(self):
        return self.writer_cpu_infer

    def ensure_mxfp4_stream_writer(self):
        return self.writer_cpu_infer


class _TrackedMoe:
    def __init__(self, completion):
        self.completions = (
            list(completion) if isinstance(completion, (list, tuple)) else [completion]
        )
        self.calls = []

    def write_weight_scale_to_buffer_tracked_task(self, *args):
        self.calls.append(args)
        completion = self.completions[len(self.calls) - 1]
        return f"tracked-task-{len(self.calls)}", completion


class _LegacyMoe:
    def __init__(self):
        self.calls = []

    def write_weight_scale_to_buffer_task(self, *args):
        self.calls.append(args)
        return "legacy-task"


class TestWriterTaskCompletionContract(unittest.TestCase):
    def test_writer_executor_uses_one_worker_after_main_pool_cores(self):
        created = []

        class _WorkerPoolConfig:
            def __init__(self):
                self.subpool_thread_start = []

        class _WriterCPUInfer:
            def __init__(self, config):
                created.append(config)

        fake_extension = SimpleNamespace(
            WorkerPoolConfig=_WorkerPoolConfig,
            CPUInfer=_WriterCPUInfer,
        )
        get_writer = _compile_base_method(
            "_get_writer_cpu_infer",
            {
                "kt_kernel_ext": fake_extension,
                "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
            },
        )

        class _Owner:
            _writer_cpu_infer_instance = None
            _writer_cpu_infer_key = None

        first = get_writer(_Owner, 168, 1, None)
        second = get_writer(_Owner, 168, 1, None)

        self.assertIs(first, second)
        self.assertEqual(len(created), 1)
        config = created[0]
        self.assertEqual(config.subpool_count, 1)
        self.assertEqual(config.subpool_numa_map, [0])
        self.assertEqual(config.subpool_thread_count, [1])
        self.assertEqual(config.subpool_thread_start, [168])

    def test_mxfp4_writer_preflight_requires_reservation_and_pool_aware_backend(self):
        ensure = _compile_method("ensure_mxfp4_stream_writer")
        writer_cpu_infer = _CPUInfer()

        unreserved = _Wrapper(
            method="MXFP4",
            _reserve_stream_writer_threads=False,
            moe=SimpleNamespace(_kt_pool_aware_writer=True),
            writer_cpu_infer=writer_cpu_infer,
        )
        with self.assertRaisesRegex(RuntimeError, "reserved stream writer"):
            ensure(unreserved)

        stale_backend = _Wrapper(
            method="MXFP4",
            _reserve_stream_writer_threads=True,
            moe=SimpleNamespace(_kt_pool_aware_writer=False),
            writer_cpu_infer=writer_cpu_infer,
        )
        with self.assertRaisesRegex(RuntimeError, "pool-aware writer"):
            ensure(stale_backend)

        ready = _Wrapper(
            method="MXFP4",
            _reserve_stream_writer_threads=True,
            moe=SimpleNamespace(_kt_pool_aware_writer=True),
            writer_cpu_infer=writer_cpu_infer,
        )
        self.assertIs(ensure(ready), writer_cpu_infer)

    def test_tracked_writer_returns_completion_without_global_sync(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        sync = _compile_method("sync_write_weight_scale_to_buffer")
        completion = _Completion()
        cpu_infer = _CPUInfer()
        writer_cpu_infer = _CPUInfer()
        moe = _TrackedMoe(completion)
        wrapper = _Wrapper(
            moe=moe,
            method="MXFP4",
            _reserve_stream_writer_threads=True,
            cpu_infer=cpu_infer,
            writer_cpu_infer=writer_cpu_infer,
        )

        result = submit(wrapper, 2, 7, [1, 2], [3, 4], [5, 6], [7, 8])

        self.assertIs(result, completion)
        self.assertEqual(cpu_infer.submitted, [])
        self.assertEqual(writer_cpu_infer.submitted, ["tracked-task-1"])
        sync(wrapper)
        self.assertEqual(completion.wait_count, 1)
        self.assertEqual(cpu_infer.sync_count, 0)

    def test_non_mxfp4_writer_preserves_shared_queue_behavior(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        completion = _Completion()
        cpu_infer = _CPUInfer()
        writer_cpu_infer = _CPUInfer()
        wrapper = _Wrapper(
            moe=_TrackedMoe(completion),
            method="FP8",
            cpu_infer=cpu_infer,
            writer_cpu_infer=writer_cpu_infer,
        )

        submit(wrapper, 1, 4, [1], [2], [3], [4])

        self.assertEqual(cpu_infer.submitted, ["tracked-task-1"])
        self.assertEqual(writer_cpu_infer.submitted, [])

    def test_static_mxfp4_without_reservation_preserves_shared_queue(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        cpu_infer = _CPUInfer()
        writer_cpu_infer = _CPUInfer()
        wrapper = _Wrapper(
            moe=_TrackedMoe(_Completion()),
            method="MXFP4",
            _reserve_stream_writer_threads=False,
            cpu_infer=cpu_infer,
            writer_cpu_infer=writer_cpu_infer,
        )

        submit(wrapper, 1, 4, [1], [2], [3], [4])

        self.assertEqual(cpu_infer.submitted, ["tracked-task-1"])
        self.assertEqual(writer_cpu_infer.submitted, [])

    def test_noarg_sync_propagates_tracked_writer_failure(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        sync = _compile_method("sync_write_weight_scale_to_buffer")
        completion = _FailingCompletion()
        following = _Completion()
        cpu_infer = _CPUInfer()
        writer_cpu_infer = _CPUInfer()
        wrapper = _Wrapper(
            moe=_TrackedMoe([completion, following]),
            method="MXFP4",
            _reserve_stream_writer_threads=True,
            cpu_infer=cpu_infer,
            writer_cpu_infer=writer_cpu_infer,
        )

        submit(wrapper, 1, 5, [1], [2], [3], [4])
        submit(wrapper, 1, 6, [1], [2], [3], [4])

        with self.assertRaisesRegex(RuntimeError, "tracked writer failure"):
            sync(wrapper)
        # The failed FIFO entry was consumed before wait(), so a later writer
        # remains independently waitable rather than being wedged behind it.
        sync(wrapper)
        self.assertEqual(following.wait_count, 1)
        self.assertEqual(cpu_infer.sync_count, 0)

    def test_legacy_writer_keeps_whole_queue_compatibility(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        sync = _compile_method("sync_write_weight_scale_to_buffer")
        cpu_infer = _CPUInfer()
        writer_cpu_infer = _CPUInfer()
        wrapper = _Wrapper(
            moe=_LegacyMoe(),
            method="MXFP4",
            _reserve_stream_writer_threads=True,
            cpu_infer=cpu_infer,
            writer_cpu_infer=writer_cpu_infer,
        )

        result = submit(wrapper, 1, 3, [1], [2], [3], [4])

        self.assertIsNone(result)
        self.assertEqual(cpu_infer.submitted, [])
        self.assertEqual(writer_cpu_infer.submitted, ["legacy-task"])
        sync(wrapper, result)
        self.assertEqual(cpu_infer.sync_count, 0)
        self.assertEqual(writer_cpu_infer.sync_count, 1)

    def test_tracked_noarg_sync_is_fifo_and_explicit_wait_removes_entry(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        sync = _compile_method("sync_write_weight_scale_to_buffer")
        first = _Completion()
        second = _Completion()
        cpu_infer = _CPUInfer()
        writer_cpu_infer = _CPUInfer()
        wrapper = _Wrapper(
            moe=_TrackedMoe([first, second]),
            method="MXFP4",
            _reserve_stream_writer_threads=True,
            cpu_infer=cpu_infer,
            writer_cpu_infer=writer_cpu_infer,
        )

        submit(wrapper, 1, 1, [1], [2], [3], [4])
        submit(wrapper, 1, 2, [1], [2], [3], [4])

        # An explicit out-of-order wait removes only that completion.  The
        # compatibility no-argument call must still consume the oldest one.
        sync(wrapper, second)
        self.assertEqual((first.wait_count, second.wait_count), (0, 1))
        sync(wrapper)
        self.assertEqual((first.wait_count, second.wait_count), (1, 1))
        self.assertEqual(cpu_infer.sync_count, 0)

    def test_extension_exposes_task_completion_binding(self):
        source = BINDINGS_PATH.read_text(encoding="utf-8")
        worker_pool_source = WORKER_POOL_PATH.read_text(encoding="utf-8")
        self.assertIn('m, "CPUInferTaskCompletion"', source)
        self.assertIn('"write_weight_scale_to_buffer_tracked_task"', source)
        self.assertIn('"rethrow_pending_callback_exception"', source)
        self.assertIn("enqueue_tracked", source)
        self.assertIn("write_weight_scale_to_buffer_with_pool", source)
        self.assertIn('"_kt_pool_aware_writer"', source)
        self.assertIn('"subpool_thread_start"', source)
        self.assertIn("subpool_thread_start", worker_pool_source)
        for backend_path in MXFP4_BACKEND_PATHS:
            backend_source = backend_path.read_text(encoding="utf-8")
            self.assertIn("write_weight_scale_to_buffer_with_pool", backend_source)
            self.assertIn("physical_to_logical_map", backend_source)
            self.assertIn("physical_expert_id", backend_source)
        self.assertIn("std::shared_ptr<MoeClass> moe;", source)
        self.assertNotIn("MoeClass* moe;", source)


if __name__ == "__main__":
    unittest.main()
