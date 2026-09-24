"""Tests for the NativeMoEWrapper shared-index loader lifecycle.

The production wrapper keeps one immutable key-to-shard index across layers,
while closing only the mmap handle cache after each layer. These lightweight
tests model that lifecycle without importing the compiled kt_kernel extension.
"""

import unittest


class MockLoader:
    _create_count = 0

    def __init__(self, cache_key):
        MockLoader._create_count += 1
        self.cache_key = cache_key
        self.file_handle_map = {"dummy.safetensors": object()}
        self.tensor_file_map = {"expert.weight": "dummy.safetensors"}
        self.close_count = 0

    def close_all_handles(self, *, collect=True):
        del collect
        self.close_count += 1
        self.file_handle_map.clear()

    def clear_index(self):
        self.close_all_handles(collect=False)
        self.tensor_file_map.clear()


class FakeNativeMoEWrapper:
    """Small replica of the shared-index / lazy-handle lifecycle."""

    _native_loader_instance = None
    _native_loader_key = None
    _create_loader_calls = 0

    def __init__(self, layer_idx=0, method="MXFP4", weight_path="/fake/path"):
        self.layer_idx = layer_idx
        self.method = method
        self.weight_path = weight_path
        self._loader_key = (method, weight_path)
        self.loader = self._ensure_loader(method, weight_path)

    @classmethod
    def _ensure_loader(cls, method, weight_path):
        key = (method, weight_path)
        if cls._native_loader_instance is None or cls._native_loader_key != key:
            if cls._native_loader_instance is not None:
                cls._native_loader_instance.close_all_handles(collect=False)
            cls._create_loader_calls += 1
            cls._native_loader_instance = MockLoader(key)
            cls._native_loader_key = key
        return cls._native_loader_instance

    def load_weights(self):
        if not self.loader.tensor_file_map:
            self.loader = self._ensure_loader(self.method, self.weight_path)
        # Simulate lazily reopening the shard needed by this layer.
        self.loader.file_handle_map["dummy.safetensors"] = object()
        self._release_loader(layer_idx=self.layer_idx, loader=self.loader)

    @classmethod
    def _release_loader(cls, layer_idx=-1, *, drop_index=False, loader=None):
        del layer_idx
        target = loader or cls._native_loader_instance
        if target is None:
            return
        if drop_index:
            target.clear_index()
            if target is cls._native_loader_instance:
                cls._native_loader_instance = None
                cls._native_loader_key = None
        else:
            target.close_all_handles(collect=False)

    @classmethod
    def force_release_loader(cls):
        cls._release_loader(drop_index=True)


def _reset_state():
    FakeNativeMoEWrapper._native_loader_instance = None
    FakeNativeMoEWrapper._native_loader_key = None
    FakeNativeMoEWrapper._create_loader_calls = 0
    MockLoader._create_count = 0


class TestSharedIndexLifecycle(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_layer_closes_handles_but_retains_index(self):
        wrapper = FakeNativeMoEWrapper(layer_idx=0)
        loader = wrapper.loader
        index_id = id(loader.tensor_file_map)

        wrapper.load_weights()

        self.assertIs(FakeNativeMoEWrapper._native_loader_instance, loader)
        self.assertEqual(loader.file_handle_map, {})
        self.assertEqual(
            loader.tensor_file_map, {"expert.weight": "dummy.safetensors"}
        )
        self.assertEqual(id(loader.tensor_file_map), index_id)

    def test_all_layers_share_one_loader_and_one_index(self):
        loaders = []
        index_ids = []
        for layer_idx in range(5):
            wrapper = FakeNativeMoEWrapper(layer_idx=layer_idx)
            loaders.append(wrapper.loader)
            index_ids.append(id(wrapper.loader.tensor_file_map))
            wrapper.load_weights()

        self.assertTrue(all(loader is loaders[0] for loader in loaders))
        self.assertEqual(len(set(index_ids)), 1)
        self.assertEqual(FakeNativeMoEWrapper._create_loader_calls, 1)
        self.assertEqual(loaders[0].close_count, 5)

    def test_different_checkpoint_gets_a_different_loader(self):
        first = FakeNativeMoEWrapper(weight_path="/model/a").loader
        second = FakeNativeMoEWrapper(weight_path="/model/b").loader

        self.assertIsNot(first, second)
        self.assertEqual(FakeNativeMoEWrapper._create_loader_calls, 2)
        self.assertEqual(
            first.tensor_file_map, {"expert.weight": "dummy.safetensors"}
        )
        self.assertEqual(first.file_handle_map, {})


class TestForceReleaseLoader(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_force_release_drops_handles_and_index(self):
        loader = FakeNativeMoEWrapper(layer_idx=0).loader

        FakeNativeMoEWrapper.force_release_loader()

        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)
        self.assertIsNone(FakeNativeMoEWrapper._native_loader_key)
        self.assertEqual(loader.file_handle_map, {})
        self.assertEqual(loader.tensor_file_map, {})

    def test_next_layer_recreates_after_force_release(self):
        first_wrapper = FakeNativeMoEWrapper(layer_idx=0)
        first_loader = first_wrapper.loader
        FakeNativeMoEWrapper.force_release_loader()

        second_wrapper = FakeNativeMoEWrapper(layer_idx=1)

        self.assertIsNot(second_wrapper.loader, first_loader)
        self.assertEqual(FakeNativeMoEWrapper._create_loader_calls, 2)

    def test_force_release_when_loader_is_none(self):
        FakeNativeMoEWrapper.force_release_loader()
        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
