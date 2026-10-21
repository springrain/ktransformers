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

  // Number of H2D buffers per GPU (the staging ring depth). Temporary
  // resource, independent of the persistent cache. Depth 2 = two generations,
  // so the next layer's transfers overlap the current layer's compute (doc
  // section 15.3). This is an internal constant, not exposed on the CLI.
  static constexpr int kStagingRingDepth = 2;
  int h2d_buffer_pool_depth = kStagingRingDepth;

  // Fully-async staging look-ahead depth, in layers. A layer in the staging
  // range prefetches the expert window of the layer `staging_prefetch_layers`
  // ahead on the H2D stream and synchronises only when the compute stream
  // consumes the data (doc sections 15.3-15.4). 0 disables the look-ahead
  // (synchronous per-round behaviour); 1 is the FreeToken prefill depth.
  int staging_prefetch_layers = 0;

  // How long (simulated GPU microseconds) a look-ahead row may sit in the
  // staging pool unobserved before its lease expires and the frustum hands it
  // back to the eviction gate (FreeToken rollback). One layer of compute is
  // the natural window: the look-ahead target layer is the next consumer. 0
  // disables expiry (rows survive until the ring generation wraps). A clock
  // that stands still inside a layer makes the lease one window per round;
  // kLandedLeaseWindows bounds how many of those a landed row survives.
  uint64_t staging_lease_us = 0;

  // Look-ahead pass a landed row may survive as a staging row. One window is
  // the layer the pass materialised for: the boundary right after the landing.
  // Later boundaries (a clock that stands still inside a layer makes every
  // round one window) hand the row back to the eviction gate, so its bytes
  // never surface as a cache hit one layer too late.
  static constexpr int kLandedLeaseWindows = 2;

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
    if (h2d_buffer_pool_depth < 1) {
      throw std::invalid_argument("h2d-buffer-pool-depth must be >= 1");
    }
    if (staging_prefetch_layers < 0) {
      throw std::invalid_argument("staging-prefetch-layers must be >= 0");
    }
    if (staging_prefetch_layers > 0 && h2d_buffer_pool_depth < 2) {
      throw std::invalid_argument(
          "fully-async staging requires h2d-buffer-pool-depth >= 2 "
          "(two generations: prefetch L+1 while L computes)");
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
