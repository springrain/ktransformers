"""CPU-only contract tests for task-specific expert writer completion."""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

AMX_PATH = Path(__file__).resolve().parents[1] / "python/utils/amx.py"
BINDINGS_PATH = Path(__file__).resolve().parents[1] / "ext_bindings.cpp"


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
    def test_tracked_writer_returns_completion_without_global_sync(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        sync = _compile_method("sync_write_weight_scale_to_buffer")
        completion = _Completion()
        cpu_infer = _CPUInfer()
        moe = _TrackedMoe(completion)
        wrapper = SimpleNamespace(moe=moe, cpu_infer=cpu_infer)

        result = submit(wrapper, 2, 7, [1, 2], [3, 4], [5, 6], [7, 8])

        self.assertIs(result, completion)
        self.assertEqual(cpu_infer.submitted, ["tracked-task-1"])
        sync(wrapper)
        self.assertEqual(completion.wait_count, 1)
        self.assertEqual(cpu_infer.sync_count, 0)

    def test_noarg_sync_propagates_tracked_writer_failure(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        sync = _compile_method("sync_write_weight_scale_to_buffer")
        completion = _FailingCompletion()
        following = _Completion()
        cpu_infer = _CPUInfer()
        wrapper = SimpleNamespace(
            moe=_TrackedMoe([completion, following]), cpu_infer=cpu_infer
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
        wrapper = SimpleNamespace(moe=_LegacyMoe(), cpu_infer=cpu_infer)

        result = submit(wrapper, 1, 3, [1], [2], [3], [4])

        self.assertIsNone(result)
        self.assertEqual(cpu_infer.submitted, ["legacy-task"])
        sync(wrapper, result)
        self.assertEqual(cpu_infer.sync_count, 1)

    def test_tracked_noarg_sync_is_fifo_and_explicit_wait_removes_entry(self):
        submit = _compile_method("submit_write_weight_scale_to_buffer")
        sync = _compile_method("sync_write_weight_scale_to_buffer")
        first = _Completion()
        second = _Completion()
        cpu_infer = _CPUInfer()
        wrapper = SimpleNamespace(
            moe=_TrackedMoe([first, second]), cpu_infer=cpu_infer
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
        self.assertIn('m, "CPUInferTaskCompletion"', source)
        self.assertIn('"write_weight_scale_to_buffer_tracked_task"', source)
        self.assertIn('"rethrow_pending_callback_exception"', source)
        self.assertIn("enqueue_tracked", source)
        self.assertIn("std::shared_ptr<MoeClass> moe;", source)
        self.assertNotIn("MoeClass* moe;", source)


if __name__ == "__main__":
    unittest.main()
