/**
 * ARM NEON MXFP8 E4M3FN MoE operator.
 *
 * Weights stay FP8 and UE8M0 scales are expanded once to FP32. The GEMM
 * decodes weights in registers and applies one scale per 32 K elements.
 */
#ifndef CPUINFER_OPERATOR_ARM_NEON_MXFP8_MOE_H
#define CPUINFER_OPERATOR_ARM_NEON_MXFP8_MOE_H

#include "moe_base.hpp"
#include "neon_mxfp8_gemm.hpp"

template <class T = armneon::GemmKernelNeonMXFP8>
class NEON_MXFP8_MOE_TP : public NEON_MOE_BASE<T, NEON_MXFP8_MOE_TP<T>> {
  using Base = NEON_MOE_BASE<T, NEON_MXFP8_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  NEON_MXFP8_MOE_TP() = default;
  NEON_MXFP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size != 32 || quant_config.zero_point) {
      throw std::runtime_error("NEON MXFP8 MoE requires group_size == 32 and no zero_point");
    }
    if (config_.hidden_size % quant_config.group_size != 0 ||
        config_.intermediate_size % quant_config.group_size != 0) {
      throw std::runtime_error("NEON MXFP8 MoE: hidden_size and intermediate_size must be divisible by group_size");
    }
    printf("Created NEON_MXFP8_MOE_TP %d at numa %d (group_size=%d, swiglu_alpha=%.4f, swiglu_limit=%.4f)\n",
           tp_part_idx, numa_node_of_cpu(sched_getcpu()), quant_config.group_size, config_.swiglu_alpha,
           config_.swiglu_limit);
  }

  ~NEON_MXFP8_MOE_TP() = default;

  // MiniMax MXFP8 follows current SGLang SwiGLU-OAI semantics: gate has only
  // an upper clamp, while up is clamped symmetrically. Keep this override on
  // MXFP8 so BF16/FP8/LLAMAFILE and other existing backends remain unchanged.
  armneon::v8f32 custom_activation(armneon::v8f32 gate_val, armneon::v8f32 up_val, float swiglu_limit,
                                   float swiglu_alpha) const {
    if (swiglu_alpha <= 0.0f) return armneon::act_fn(gate_val, up_val, swiglu_limit, swiglu_alpha);
    if (swiglu_limit > 0.0f) {
      const armneon::v8f32 pos_lim = armneon::set1_v8f32(swiglu_limit);
      const armneon::v8f32 neg_lim = armneon::set1_v8f32(-swiglu_limit);
      gate_val = armneon::min_v8f32(gate_val, pos_lim);
      up_val = armneon::min_v8f32(up_val, pos_lim);
      up_val = armneon::max_v8f32(up_val, neg_lim);
    }
    armneon::v8f32 neg_ga = armneon::mul_v8f32(gate_val, armneon::set1_v8f32(-swiglu_alpha));
    neg_ga = armneon::min_v8f32(neg_ga, armneon::set1_v8f32(88.0f));
    const armneon::v8f32 sigmoid_val = armneon::div_v8f32(
        armneon::set1_v8f32(1.0f), armneon::add_v8f32(armneon::set1_v8f32(1.0f), armneon::exp_neon(neg_ga)));
    return armneon::mul_v8f32(armneon::mul_v8f32(gate_val, sigmoid_val),
                              armneon::add_v8f32(up_val, armneon::set1_v8f32(1.0f)));
  }

  float custom_activation(float gate_val, float up_val, float swiglu_limit, float swiglu_alpha) const {
    if (swiglu_alpha <= 0.0f) {
      if (swiglu_limit > 0.0f) {
        gate_val = std::min(gate_val, swiglu_limit);
        up_val = std::min(std::max(up_val, -swiglu_limit), swiglu_limit);
      }
      return gate_val / (1.0f + expf(-gate_val)) * up_val;
    }
    if (swiglu_limit > 0.0f) {
      gate_val = std::min(gate_val, swiglu_limit);
      up_val = std::min(std::max(up_val, -swiglu_limit), swiglu_limit);
    }
    return gate_val / (1.0f + expf(-gate_val * swiglu_alpha)) * (up_val + 1.0f);
  }

  // CRTP buffer creation
  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // GEMM dispatch
  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, [[maybe_unused]] int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    armneon::gemm_mxfp8_kgroup(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, [[maybe_unused]] int qlen) {
    int m = m_local_num_[expert_idx];
    armneon::gemm_mxfp8_kgroup(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                               *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  // Load FP8 weights + ue8m0 scales from checkpoint.
  //   gate_proj/up_proj/down_proj: uint8_t [E, N, K] (FP8 E4M3fn bytes)
  //   gate_scale/up_scale/down_scale: uint8_t [E, N, K/group_size] (ue8m0)
  // We memcpy weights and bit-shift convert scales to FP32 in-place.
  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size != 32 || quant_config.zero_point)
      throw std::runtime_error("NEON MXFP8 MoE requires group_size=32 and zero_point=false");
    if (config_.gate_scale == nullptr)
      throw std::runtime_error("NEON MXFP8 MoE requires native MXFP8 weights with ue8m0 scales");

    // Load FP8 weights (1 byte/element) into BufferB.b.
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          size_t weight_offset = (size_t)logical_expert_id * config_.intermediate_size * config_.hidden_size;
          gate_bb_[expert_idx]->from_raw_mat((uint8_t*)config_.gate_proj + weight_offset, ith, nth);
          up_bb_[expert_idx]->from_raw_mat((uint8_t*)config_.up_proj + weight_offset, ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          size_t weight_offset = (size_t)logical_expert_id * config_.hidden_size * config_.intermediate_size;
          down_bb_[expert_idx]->from_raw_mat((uint8_t*)config_.down_proj + weight_offset, ith, nth);
        },
        nullptr);

    // Convert UE8M0 scales to FP32 in BufferB.d.
    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_count =
              ((size_t)config_.intermediate_size * config_.hidden_size) / config_.quant_config.group_size;
          armneon::convert_ue8m0_to_fp32(gate_bb_[expert_idx]->d,
                                         (const uint8_t*)config_.gate_scale + logical_expert_id * scale_count,
                                         scale_count);
          armneon::convert_ue8m0_to_fp32(
              up_bb_[expert_idx]->d, (const uint8_t*)config_.up_scale + logical_expert_id * scale_count, scale_count);
          armneon::convert_ue8m0_to_fp32(down_bb_[expert_idx]->d,
                                         (const uint8_t*)config_.down_scale + logical_expert_id * scale_count,
                                         scale_count);
        },
        nullptr);
  }

  // --------------------------------------------------------------------------
  // write_weights_to_buffer: copies CPU expert weights to GPU pinned host buffer
  // for layerwise prefill (sglang full-GPU fallback at large prefill token count).
  //
  // Mirrors amx::AMX_MXFP8_MOE_TP::write_weights_to_buffer (amx/mxfp8-moe.hpp:662)
  // with two substitutions for NEON:
  // Weights use std::memcpy and FP32 scales are converted back to the GPU's
  // one-byte UE8M0 representation by armneon::fp32_to_ue8m0.
  //
  // Scale dtype on the GPU side is torch.uint8 (ue8m0), see
  // kt-sglang/python/sglang/srt/layers/quantization/fp8.py:872.
  // --------------------------------------------------------------------------
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    // Unified per-row scatter that works for any (cpu_tp_count, gpu_tp_count)
    // relationship. Replaces the earlier
    // `cpu_tp_count >= gpu_tp_count` direct-write branch which was a silent
    // no-op for a realistic 2-NUMA by tp>=4 deployment. Each task processes a row chunk of this CPU
    // TP's slice; per row we compute target_gpu = global_n / gpu_n_w13 (W13)
    // or scatter across multiple gpu_tp k-slices (W2) and write to the
    // corresponding GPU TP staging buffer at the right offset.
    //
    // MXFP8 specifics vs FP8 block:
    //   - Scale dtype on GPU side is uint8 ue8m0 (1 byte per `group_size`
    //     weights, per row, not per block).
    //   - Source scale (bb->d) is stored as FP32 in kt-kernel BufferB and is
    //     converted to ue8m0 via armneon::fp32_to_ue8m0 at write time.
    //   - Scale layout matches GPU: (E, 2*intermediate, hidden/group_size)
    //     for W13 and (E, hidden, intermediate/group_size) for W2.

    auto& config = config_;
    auto pool = config.pool->get_subpool(tp_part_idx);
    const int group_size = config.quant_config.group_size;
    if (gpu_tp_count <= 0 || full_config.intermediate_size % gpu_tp_count != 0 ||
        (full_config.intermediate_size / gpu_tp_count) % group_size != 0) {
      throw std::runtime_error(
          "NEON MXFP8 GPU staging requires intermediate_size/gpu_tp_count to be divisible by group_size=32");
    }

    // ========= W13 (gate+up): Shape [intermediate, hidden], split by N only =========
    const int cpu_n_w13 = config.intermediate_size;
    const int cpu_k_w13 = config.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;
    const size_t gpu_w13_weight_per_mat = (size_t)gpu_n_w13 * gpu_k_w13;
    const int scales_per_row_w13 = cpu_k_w13 / group_size;
    const size_t gpu_w13_scale_per_mat = (size_t)gpu_n_w13 * scales_per_row_w13;

    // ========= W2 (down): Shape [hidden, intermediate], split by K =========
    const int cpu_n_w2 = config.hidden_size;
    const int cpu_k_w2 = config.intermediate_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;
    const int cpu_scales_per_row_w2 = cpu_k_w2 / group_size;
    const int gpu_scales_per_row_w2 = gpu_k_w2 / group_size;

    constexpr int NUM_W13_TASKS = 32;  // per matrix (gate or up); total 64 W13 tasks
    constexpr int NUM_W2_TASKS = 32;
    const int total_tasks = NUM_W13_TASKS * 2 + NUM_W2_TASKS;

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS * 2) {
            // ---- W13 weight + scale: per-row scatter (one target_gpu per row) ----
            const bool is_up = task_id >= NUM_W13_TASKS;
            const int chunk_idx = task_id % NUM_W13_TASKS;
            const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];

            const int rows_per_task = (cpu_n_w13 + NUM_W13_TASKS - 1) / NUM_W13_TASKS;
            const int row_start = chunk_idx * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w13);
            if (row_start >= cpu_n_w13) return;

            for (int row = row_start; row < row_end; row++) {
              const int global_n = global_n_offset_w13 + row;
              const int target_gpu = global_n / gpu_n_w13;
              const int n_in_gpu = global_n % gpu_n_w13;

              // Weight row: full K (cpu_k_w13 == gpu_k_w13 for W13).
              uint8_t* w_dst = (uint8_t*)w13_weight_ptrs[target_gpu];
              const size_t expert_w_off = is_up ? gpu_w13_weight_per_mat : 0;
              std::memcpy(w_dst + expert_w_off + (size_t)n_in_gpu * gpu_k_w13, bb->b + (size_t)row * cpu_k_w13,
                          cpu_k_w13);

              // Scale row: full K/group_size UE8M0 bytes (FP32 to UE8M0 conversion).
              uint8_t* s_dst = (uint8_t*)w13_scale_ptrs[target_gpu];
              const size_t expert_s_off = is_up ? gpu_w13_scale_per_mat : 0;
              armneon::fp32_to_ue8m0(s_dst + expert_s_off + (size_t)n_in_gpu * scales_per_row_w13,
                                     bb->d + (size_t)row * scales_per_row_w13, scales_per_row_w13);
            }
          } else {
            // ---- W2 weight + scale: per-row + per-k-slice scatter ----
            const int chunk_idx = task_id - NUM_W13_TASKS * 2;
            const auto& bb = down_bb_[expert_id];

            const int rows_per_task = (cpu_n_w2 + NUM_W2_TASKS - 1) / NUM_W2_TASKS;
            const int row_start = chunk_idx * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w2);
            if (row_start >= cpu_n_w2) return;

            for (int row = row_start; row < row_end; row++) {
              // Split at every GPU-TP boundary. CPU and GPU TP counts need not
              // divide one another (for example CPU TP=3, GPU TP=2).
              for (int k_start = 0; k_start < cpu_k_w2;) {
                const int global_k = global_k_offset_w2 + k_start;
                const int target_gpu = global_k / gpu_k_w2;
                const int k_in_gpu = global_k % gpu_k_w2;
                const int k_slice_len = std::min(cpu_k_w2 - k_start, gpu_k_w2 - k_in_gpu);

                // Weight K-slice
                uint8_t* w_dst = (uint8_t*)w2_weight_ptrs[target_gpu];
                std::memcpy(w_dst + (size_t)row * gpu_k_w2 + k_in_gpu, bb->b + (size_t)row * cpu_k_w2 + k_start,
                            k_slice_len);

                // Scale K-slice (k_slice_len/group_size ue8m0 bytes)
                const int scale_slice_len = k_slice_len / group_size;
                const int k_in_gpu_scale = k_in_gpu / group_size;
                const int k_start_scale = k_start / group_size;
                uint8_t* s_dst = (uint8_t*)w2_scale_ptrs[target_gpu];
                armneon::fp32_to_ue8m0(s_dst + (size_t)row * gpu_scales_per_row_w2 + k_in_gpu_scale,
                                       bb->d + (size_t)row * cpu_scales_per_row_w2 + k_start_scale, scale_slice_len);
                k_start += k_slice_len;
              }
            }
          }
        },
        nullptr);
  }
};

// ============================================================================
// TP_MOE specialization handles per-expert FP8 weight and UE8M0 scale loading
// across TP parts. Mirrors NEON_FP8_MOE_TP's TP_MOE but with MXFP8 layout:
//   weights: [E, N, K] FP8 bytes (no nibble pack, no block scale)
//   scales:  [E, N, K/group_size] ue8m0 bytes (NOT pre-converted)
//
// Inside per-TP load_weights() we point config to the staged TP-sliced bytes,
// then derived_class::load_weights() reads from there and expands UE8M0 to FP32.
// ============================================================================
template <typename K>
class TP_MOE<NEON_MXFP8_MOE_TP<K>> : public TP_MOE<NEON_MOE_BASE<K, NEON_MXFP8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<NEON_MOE_BASE<K, NEON_MXFP8_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size != 32 || config.quant_config.zero_point) {
      throw std::runtime_error("MXFP8 MoE only supports group_size=32 and zero_point=false");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    // Full dimensions
    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems = full_weight_elems / (size_t)group_size;  // ue8m0 = 1 byte each

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems = tp_weight_elems / (size_t)group_size;

      // Allocate temporary buffers for TP-sliced FP8 + ue8m0 scales
      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.gate_scale = new uint8_t[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new uint8_t[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new uint8_t[tpc.expert_num * tp_scale_elems];

      // gate/up: split N=intermediate, each expert is [intermediate, hidden] FP8 + [intermediate, hidden/gs] ue8m0
      const size_t gate_up_w_src_off = i * tp_weight_elems;  // bytes
      const size_t gate_up_s_src_off = i * tp_scale_elems;   // bytes (ue8m0=1B)

      // down: split K=intermediate (columns of [hidden, intermediate] FP8 + [hidden, intermediate/gs] ue8m0)
      const size_t down_col_off = (size_t)i * tpc.intermediate_size;
      const size_t down_scale_col_off = down_col_off / (size_t)group_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&](int expert_id_) {
            const size_t physical_expert_id = static_cast<size_t>(expert_id_);
            const size_t logical_expert_id = expert_map(physical_to_logical_map, physical_expert_id);

            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + physical_expert_id * tp_weight_elems;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + physical_expert_id * tp_weight_elems;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + physical_expert_id * tp_weight_elems;
            uint8_t* gate_s_dst = (uint8_t*)tpc.gate_scale + physical_expert_id * tp_scale_elems;
            uint8_t* up_s_dst = (uint8_t*)tpc.up_scale + physical_expert_id * tp_scale_elems;
            uint8_t* down_s_dst = (uint8_t*)tpc.down_scale + physical_expert_id * tp_scale_elems;

            const uint8_t* gate_src;
            const uint8_t* up_src;
            const uint8_t* down_src;
            const uint8_t* gate_s_src;
            const uint8_t* up_s_src;
            const uint8_t* down_s_src;

            if (use_per_expert_ptrs) {
              gate_src = (const uint8_t*)config.gate_projs[0][logical_expert_id] + gate_up_w_src_off;
              up_src = (const uint8_t*)config.up_projs[0][logical_expert_id] + gate_up_w_src_off;
              down_src = (const uint8_t*)config.down_projs[0][logical_expert_id];
              gate_s_src = (const uint8_t*)config.gate_scales[0][logical_expert_id] + gate_up_s_src_off;
              up_s_src = (const uint8_t*)config.up_scales[0][logical_expert_id] + gate_up_s_src_off;
              down_s_src = (const uint8_t*)config.down_scales[0][logical_expert_id];
            } else {
              gate_src =
                  (const uint8_t*)config.gate_proj + logical_expert_id * full_weight_elems + gate_up_w_src_off;
              up_src = (const uint8_t*)config.up_proj + logical_expert_id * full_weight_elems + gate_up_w_src_off;
              down_src = (const uint8_t*)config.down_proj + logical_expert_id * full_weight_elems;
              gate_s_src =
                  (const uint8_t*)config.gate_scale + logical_expert_id * full_scale_elems + gate_up_s_src_off;
              up_s_src =
                  (const uint8_t*)config.up_scale + logical_expert_id * full_scale_elems + gate_up_s_src_off;
              down_s_src = (const uint8_t*)config.down_scale + logical_expert_id * full_scale_elems;
            }

            // gate/up weights + scales: contiguous N slice
            std::memcpy(gate_dst, gate_src, tp_weight_elems);
            std::memcpy(up_dst, up_src, tp_weight_elems);
            std::memcpy(gate_s_dst, gate_s_src, tp_scale_elems);
            std::memcpy(up_s_dst, up_s_src, tp_scale_elems);

            // down weights: column slice within each of `hidden` rows
            // src row = [hidden_row, full_intermediate]; dst row = [hidden_row, tp_intermediate]
            for (int row = 0; row < config.hidden_size; row++) {
              std::memcpy(down_dst + (size_t)row * tpc.intermediate_size,
                          down_src + (size_t)row * config.intermediate_size + down_col_off, tpc.intermediate_size);
            }
            // down scales: column slice within each of `hidden` scale rows
            const int full_kg = config.intermediate_size / group_size;
            const int tp_kg = tpc.intermediate_size / group_size;
            for (int row = 0; row < config.hidden_size; row++) {
              std::memcpy(down_s_dst + (size_t)row * tp_kg, down_s_src + (size_t)row * full_kg + down_scale_col_off,
                          tp_kg);
            }
          },
          nullptr);
    });

    // Temporary tensors are already arranged by physical expert slot.
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      tps[i]->config_.physical_to_logical_map = nullptr;
      tps[i]->load_weights();
    });

    // Free temporary buffers
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)tpc.gate_proj;
      delete[] (uint8_t*)tpc.up_proj;
      delete[] (uint8_t*)tpc.down_proj;
      delete[] (uint8_t*)tpc.gate_scale;
      delete[] (uint8_t*)tpc.up_scale;
      delete[] (uint8_t*)tpc.down_scale;
    });

    this->weights_loaded = true;
  }

  // --------------------------------------------------------------------------
  // write_weight_scale_to_buffer: orchestrator for layerwise prefill.
  // Called once per expert by Python (kt-sglang/.../kt_ep_wrapper.py:_prepare_weight_fp8).
  // Dispatches across all NUMA TP parts; each part runs its own
  // NEON_MXFP8_MOE_TP::write_weights_to_buffer to mirror its slice of the expert
  // into the pre-allocated GPU pinned staging buffer.
  //
  // Mirrors amx/mxfp8-moe.hpp:897-912.
  // SFINAE in ext_bindings.cpp:435 auto-detects this method and exposes
  // `moe.write_weight_scale_to_buffer_task(...)` to Python; no manual binding is needed.
  // --------------------------------------------------------------------------
  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (w13_weight_ptrs.size() != (size_t)gpu_tp_count || w13_scale_ptrs.size() != (size_t)gpu_tp_count ||
        w2_weight_ptrs.size() != (size_t)gpu_tp_count || w2_scale_ptrs.size() != (size_t)gpu_tp_count)
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");

    // The layerwise caller identifies a logical expert. BufferB is indexed by
    // physical slot, so resolve it through the same EPLB mapping used at load.
    int physical_expert_id = expert_id;
    const uint64_t* physical_to_logical_map =
        static_cast<const uint64_t*>(this->config.physical_to_logical_map);
    if (physical_to_logical_map != nullptr) {
      physical_expert_id = -1;
      for (int physical_id = 0; physical_id < this->config.expert_num; ++physical_id) {
        if (physical_to_logical_map[physical_id] == static_cast<uint64_t>(expert_id)) {
          physical_expert_id = physical_id;
          break;
        }
      }
      if (physical_expert_id < 0) {
        throw std::runtime_error("Requested logical MXFP8 expert is absent from physical_to_logical_map");
      }
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this, physical_expert_id](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, physical_expert_id, this->config,
                                            w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_ARM_NEON_MXFP8_MOE_H
