// GEC runtime parameters (doc/KT-CPU-PICE-GPU.md section 5).
//
// All capacity parameters count Logical Experts, never Physical Shards.
#pragma once

#include <stdexcept>

namespace kt::gec {

struct GecConfig {
  // First K layers whose experts stay resident on GPU.
  int num_gpu_layers = 0;

  // Global persistent cache capacity, in Logical Experts.
  int expert_gpu_slots = 0;

  // Per-layer / per-round H2D admission budget, in Logical Experts.
  int layer_h2d_slots = 0;

  // Max Logical Experts per H2D batch. A batch is min(available, batch_size);
  // it never waits to be filled.
  int layer_h2d_batch_size = 1;

  // Number of H2D buffers per GPU. Temporary resource, independent of the
  // persistent cache.
  int h2d_buffer_pool_depth = 1;

  // Priority = current_demand + alpha * historical_hit_count.
  double priority_alpha = 1.0;

  // Uses before a probationary expert becomes a normal cached expert.
  int probation_uses = 1;

  void validate() const {
    if (num_gpu_layers < 0) {
      throw std::invalid_argument("kt-num-gpu-layers must be >= 0");
    }
    if (expert_gpu_slots < 0) {
      throw std::invalid_argument("kt-expert-gpu-slots must be >= 0");
    }
    if (layer_h2d_slots < 0) {
      throw std::invalid_argument("kt-layer-h2d-slots must be >= 0");
    }
    if (layer_h2d_batch_size < 1) {
      throw std::invalid_argument("kt-layer-h2d-batch-size must be >= 1");
    }
    if (h2d_buffer_pool_depth < 1) {
      throw std::invalid_argument("h2d-buffer-pool-depth must be >= 1");
    }
    if (priority_alpha < 0.0) {
      throw std::invalid_argument("priority alpha must be >= 0");
    }
    if (probation_uses < 0) {
      throw std::invalid_argument("probation-uses must be >= 0");
    }
  }

  // Physical shard count for a given TP degree (unit conversion guard).
  static int shard_count(int logical_experts, int tp_size) {
    if (logical_experts < 0 || tp_size < 1) {
      throw std::invalid_argument("logical_experts >= 0 and tp_size >= 1 required");
    }
    return logical_experts * tp_size;
  }
};

}  // namespace kt::gec
