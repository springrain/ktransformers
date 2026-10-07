# CPU-only contract guards for the Global Expert Cache (GEC) module.
#
# Verifies the structural invariants required by doc/KT-CPU-PICE-GPU.md
# section 29 (V1 acceptance criteria) and runs the full C++ suite when a
# compiler is available.
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite='default')

KT_KERNEL = Path(__file__).resolve().parents[2]
GEC_DIR = KT_KERNEL / 'include' / 'kt' / 'gec'
RUNNER = KT_KERNEL / 'test' / 'gec' / 'run_tests.py'

REQUIRED_HEADERS = {
    'gec_types.hpp',
    'gec_config.hpp',
    'gec_sync.hpp',
    'placement.hpp',
    'inflight_registry.hpp',
    'buffer_pool.hpp',
    'scheduler.hpp',
    'expert_cache.hpp',
    'telemetry.hpp',
    'pipeline.hpp',
}


class TestGecContract(unittest.TestCase):

    def test_required_headers_present(self):
        names = {path.name for path in GEC_DIR.glob('*.hpp')}
        missing = REQUIRED_HEADERS - names
        self.assertFalse(missing, 'missing GEC headers: %s' % sorted(missing))

    def test_no_global_cuda_synchronization(self):
        for header in sorted(GEC_DIR.glob('*.hpp')):
            text = header.read_text(encoding='utf-8')
            self.assertNotIn('cudaStreamSynchronize', text, header.name)
            self.assertNotIn('cudaDeviceSynchronize', text, header.name)
            self.assertNotIn('cudaMemcpy(', text, header.name)

    def test_lifecycle_uses_refcount_not_in_use_state(self):
        text = (GEC_DIR / 'gec_types.hpp').read_text(encoding='utf-8')
        start = text.index('enum class LifecycleState')
        end = text.index('};', start)
        self.assertNotIn('InUse', text[start:end])
        self.assertIn('std::atomic<int> refcount', text)

    def test_three_dimensions_are_separated(self):
        text = (GEC_DIR / 'gec_types.hpp').read_text(encoding='utf-8')
        self.assertIn('LifecycleState state', text)
        self.assertIn('refcount', text)
        self.assertIn('is_logical_ready', text)
        self.assertIn('can_evict', text)

    def test_tp_atomic_ready_gate(self):
        text = (GEC_DIR / 'expert_cache.hpp').read_text(encoding='utf-8')
        self.assertIn('mark_shard_ready', text)
        self.assertIn('all_ready', text)

    def test_inflight_registry_dedup(self):
        text = (GEC_DIR / 'inflight_registry.hpp').read_text(encoding='utf-8')
        self.assertIn('std::pair<bool, uint64_t> register_transfer', text)
        self.assertIn('return {false, it->second.dependency_id};', text)

    def test_buffer_released_only_after_gpu_completion(self):
        text = (GEC_DIR / 'buffer_pool.hpp').read_text(encoding='utf-8')
        self.assertIn('void gpu_complete(int buffer_id)', text)
        self.assertIn('if (buffer.state != BufferState::GpuInUse)', text)

    def test_scheduler_batch_never_waits_to_fill(self):
        text = (GEC_DIR / 'scheduler.hpp').read_text(encoding='utf-8')
        self.assertIn('h2d_selected', text)
        self.assertIn('h2d_batches', text)
        # Top-N admission is budget-bounded; batch size only chunks submits.
        self.assertIn('std::min(eligible.size(), budget)', text)
        pipeline = (GEC_DIR / 'pipeline.hpp').read_text(encoding='utf-8')
        self.assertIn('h2d_batches', pipeline)

    def test_telemetry_has_six_wait_reasons(self):
        text = (GEC_DIR / 'telemetry.hpp').read_text(encoding='utf-8')
        self.assertIn('kWaitReasonCount = 6', text)
        for reason in ('RouterDependency', 'H2DQueued', 'H2DRunning',
                       'H2DEventWait', 'CpuFallback', 'CacheWait'):
            self.assertIn(reason, text)

    def test_h2d_requires_cache_admission(self):
        text = (GEC_DIR / 'pipeline.hpp').read_text(encoding='utf-8')
        self.assertIn('cache_.admit(id, round_start)', text)
        self.assertIn('if (!cache_.admit(id, round_start)) return H2DOutcome::Fallback;', text)
        self.assertIn('FromCache', text)
        self.assertNotIn('admit_on_h2d', text)

    def test_startup_prefill_sequential(self):
        cache = (GEC_DIR / 'expert_cache.hpp').read_text(encoding='utf-8')
        self.assertIn('prefill_sequential', cache)
        self.assertIn('sequential_expert_ids', cache)
        self.assertIn('slot_resident_size', cache)
        self.assertIn('num_gpu_layers onwards', cache)
        pipeline = (GEC_DIR / 'pipeline.hpp').read_text(encoding='utf-8')
        self.assertIn('prefill_sequential', pipeline)

    def test_cpp_suite_passes(self):
        result = subprocess.run([sys.executable, str(RUNNER)],
                                capture_output=True, text=True)
        if result.returncode == 2:
            self.skipTest('no C++ compiler available')
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn('0 failures', result.stdout)


if __name__ == '__main__':
    unittest.main()
