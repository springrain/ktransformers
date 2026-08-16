/**
 * ARM NEON block-scaled FP8 E4M3FN MoE operator.
 *
 * Weight bytes remain compressed in memory. neon_fp8_gemm.hpp expands them in
 * registers and applies one FP32 scale per 128x128 block.
 */
#ifndef CPUINFER_OPERATOR_ARM_NEON_FP8_MOE_H
#define CPUINFER_OPERATOR_ARM_NEON_FP8_MOE_H

#include "moe_base.hpp"
#include "neon_fp8_gemm.hpp"

template <class T = armneon::GemmKernelNeonFP8>
class NEON_FP8_MOE_TP : public NEON_MOE_BASE<T, NEON_FP8_MOE_TP<T>> {
  using Base = NEON_MOE_BASE<T, NEON_FP8_MOE_TP<T>>;
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

  NEON_FP8_MOE_TP() = default;

  NEON_FP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size != 128 || quant_config.zero_point) {
      throw std::runtime_error("NEON FP8 MoE only supports block-wise FP8 (group_size=128, zero_point=false)");
    }
    if (config_.intermediate_size % quant_config.group_size != 0) {
      throw std::runtime_error(
          "NEON FP8 MoE requires each CPU TP intermediate-size slice to be divisible by group_size=128");
    }
    printf("Created NEON_FP8_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~NEON_FP8_MOE_TP() = default;

  // CRTP buffer creation with group_size for BufferB.
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
    armneon::gemm_fp8_block(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, [[maybe_unused]] int qlen) {
    int m = m_local_num_[expert_idx];
    armneon::gemm_fp8_block(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                            *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  // Load FP8 weights + scales from contiguous memory
  void load_weights() {
    auto& quant_config = config_.quant_config;
    int group_size = quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("FP8 MOE requires scale pointers.");
    }

    // Load gate + up weights
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          size_t scale_offset = logical_expert_id * armneon::fp8_div_up(config_.hidden_size, group_size) *
                                armneon::fp8_div_up(config_.intermediate_size, group_size);

          gate_bb_[expert_idx]->from_mat((uint8_t*)config_.gate_proj + weight_offset,
                                         (float*)config_.gate_scale + scale_offset, ith, nth);

          up_bb_[expert_idx]->from_mat((uint8_t*)config_.up_proj + weight_offset,
                                       (float*)config_.up_scale + scale_offset, ith, nth);
        },
        nullptr);

    // Load down weights
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          size_t scale_offset = logical_expert_id * armneon::fp8_div_up(config_.hidden_size, group_size) *
                                armneon::fp8_div_up(config_.intermediate_size, group_size);

          down_bb_[expert_idx]->from_mat((uint8_t*)config_.down_proj + weight_offset,
                                         (float*)config_.down_scale + scale_offset, ith, nth);
        },
        nullptr);
  }

  // Write weights to GPU buffer (for dynamic expert offload / layerwise prefill)
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    auto pool = config.pool->get_subpool(tp_part_idx);
    int group_size = config.quant_config.group_size;
    if (gpu_tp_count <= 0 || full_config.intermediate_size % gpu_tp_count != 0 ||
        (full_config.intermediate_size / gpu_tp_count) % group_size != 0) {
      throw std::runtime_error(
          "NEON FP8 GPU staging requires intermediate_size/gpu_tp_count to be divisible by group_size=128");
    }

    // W13 (gate+up)
    const int cpu_n_w13 = config.intermediate_size;
    const int cpu_k_w13 = config.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;
    const size_t gpu_w13_weight_per_mat = (size_t)gpu_n_w13 * gpu_k_w13;
    const int gpu_n_blocks_k_w13 = armneon::fp8_div_up(gpu_k_w13, group_size);
    const size_t gpu_w13_scale_per_mat = (size_t)armneon::fp8_div_up(gpu_n_w13, group_size) * gpu_n_blocks_k_w13;

    // W2 (down)
    const int cpu_n_w2 = config.hidden_size;
    const int cpu_k_w2 = config.intermediate_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;
    const int cpu_n_blocks_k_w2 = armneon::fp8_div_up(cpu_k_w2, group_size);

    constexpr int NUM_W13_TASKS = 32;
    constexpr int NUM_W2_TASKS = 32;
    const int total_tasks = NUM_W13_TASKS * 2 + NUM_W2_TASKS;

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS * 2) {
            const bool is_up = task_id >= NUM_W13_TASKS;
            const int chunk_idx = task_id % NUM_W13_TASKS;
            const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];

            const int rows_per_task = armneon::fp8_div_up(cpu_n_w13, NUM_W13_TASKS);
            const int row_start = chunk_idx * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w13);
            if (row_start >= cpu_n_w13) return;

            for (int row = row_start; row < row_end; row++) {
              const int global_n = global_n_offset_w13 + row;
              const int target_gpu = global_n / gpu_n_w13;
              const int n_in_gpu = global_n % gpu_n_w13;

              // Copy weight row
              uint8_t* w_dst = (uint8_t*)w13_weight_ptrs[target_gpu];
              const size_t expert_w_off = is_up ? gpu_w13_weight_per_mat : 0;
              std::memcpy(w_dst + expert_w_off + (size_t)n_in_gpu * gpu_k_w13, bb->b + (size_t)row * cpu_k_w13,
                          cpu_k_w13);

              // Copy scale row (if at block boundary)
              if (row % group_size == 0) {
                int n_block = row / group_size;
                int gpu_n_block = n_in_gpu / group_size;
                float* s_dst = (float*)w13_scale_ptrs[target_gpu];
                const size_t expert_s_off = is_up ? gpu_w13_scale_per_mat : 0;
                std::memcpy(s_dst + expert_s_off + gpu_n_block * gpu_n_blocks_k_w13,
                            bb->d + n_block * armneon::fp8_div_up(cpu_k_w13, group_size),
                            armneon::fp8_div_up(cpu_k_w13, group_size) * sizeof(float));
              }
            }
          } else {
            const int chunk_idx = task_id - NUM_W13_TASKS * 2;
            const auto& bb = down_bb_[expert_id];

            const int rows_per_task = armneon::fp8_div_up(cpu_n_w2, NUM_W2_TASKS);
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

                uint8_t* w_dst = (uint8_t*)w2_weight_ptrs[target_gpu];
                std::memcpy(w_dst + (size_t)row * gpu_k_w2 + k_in_gpu, bb->b + (size_t)row * cpu_k_w2 + k_start,
                            k_slice_len);

                // Copy scales for down (at block boundaries)
                if (row % group_size == 0) {
                  int n_block = row / group_size;
                  float* s_dst = (float*)w2_scale_ptrs[target_gpu];
                  int gpu_n_blocks_k_w2 = armneon::fp8_div_up(gpu_k_w2, group_size);
                  int k_block_start = k_in_gpu / group_size;
                  int n_blocks_to_copy = k_slice_len / group_size;
                  std::memcpy(s_dst + n_block * gpu_n_blocks_k_w2 + k_block_start,
                              bb->d + n_block * cpu_n_blocks_k_w2 + k_start / group_size,
                              n_blocks_to_copy * sizeof(float));
                }
                k_start += k_slice_len;
              }  // end k_start loop
            }  // end row loop
          }
        },
        nullptr);
  }
};

// ============================================================================
// TP_MOE specialization, ported from the existing block-FP8 implementation.
// Handles per-expert pointer loading + TP weight/scale splitting
// ============================================================================
template <typename K>
class TP_MOE<NEON_FP8_MOE_TP<K>> : public TP_MOE<NEON_MOE_BASE<K, NEON_FP8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<NEON_MOE_BASE<K, NEON_FP8_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size != 128 || config.quant_config.zero_point) {
      throw std::runtime_error("FP8 MoE only supports block-wise (group_size=128, zero_point=false)");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems = (size_t)armneon::fp8_div_up(config.hidden_size, group_size) *
                                    armneon::fp8_div_up(config.intermediate_size, group_size);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems = (size_t)armneon::fp8_div_up(tpc.intermediate_size, group_size) *
                                    armneon::fp8_div_up(tpc.hidden_size, group_size);

      // Allocate temporary buffers
      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.gate_scale = new float[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new float[tpc.expert_num * tp_scale_elems];

      const size_t gate_up_weight_src_offset = i * tp_weight_elems;
      const size_t gate_up_scale_src_offset = i * tp_scale_elems;
      const size_t down_weight_src_col_offset = i * (size_t)tpc.intermediate_size;
      const size_t down_scale_src_block_k_offset = down_weight_src_col_offset / (size_t)group_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&](int expert_id_) {
            const size_t physical_expert_id = static_cast<size_t>(expert_id_);
            const size_t logical_expert_id = expert_map(physical_to_logical_map, physical_expert_id);

            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + physical_expert_id * tp_weight_elems;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + physical_expert_id * tp_weight_elems;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + physical_expert_id * tp_weight_elems;
            float* gate_scale_dst = (float*)tpc.gate_scale + physical_expert_id * tp_scale_elems;
            float* up_scale_dst = (float*)tpc.up_scale + physical_expert_id * tp_scale_elems;
            float* down_scale_dst = (float*)tpc.down_scale + physical_expert_id * tp_scale_elems;

            const uint8_t* gate_src;
            const uint8_t* up_src;
            const uint8_t* down_src;
            const float* gate_scale_src;
            const float* up_scale_src;
            const float* down_scale_src;

            if (use_per_expert_ptrs) {
              gate_src = (const uint8_t*)config.gate_projs[0][logical_expert_id] + gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_projs[0][logical_expert_id] + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_projs[0][logical_expert_id];
              gate_scale_src = (const float*)config.gate_scales[0][logical_expert_id] + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scales[0][logical_expert_id] + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scales[0][logical_expert_id];
            } else {
              gate_src =
                  (const uint8_t*)config.gate_proj + logical_expert_id * full_weight_elems + gate_up_weight_src_offset;
              up_src =
                  (const uint8_t*)config.up_proj + logical_expert_id * full_weight_elems + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_proj + logical_expert_id * full_weight_elems;
              gate_scale_src =
                  (const float*)config.gate_scale + logical_expert_id * full_scale_elems + gate_up_scale_src_offset;
              up_scale_src =
                  (const float*)config.up_scale + logical_expert_id * full_scale_elems + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scale + logical_expert_id * full_scale_elems;
            }

            // Copy gate/up weights + scales (column slice)
            std::memcpy(gate_dst, gate_src, tp_weight_elems);
            std::memcpy(up_dst, up_src, tp_weight_elems);
            std::memcpy(gate_scale_dst, gate_scale_src, sizeof(float) * tp_scale_elems);
            std::memcpy(up_scale_dst, up_scale_src, sizeof(float) * tp_scale_elems);

            // Copy down weights (row-wise split)
            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
              const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset, (size_t)tpc.intermediate_size);
            }

            // Copy down scales (block-row-wise split)
            const int n_blocks_n = armneon::fp8_div_up(config.hidden_size, group_size);
            const int full_n_blocks_k = armneon::fp8_div_up(config.intermediate_size, group_size);
            const int tp_n_blocks_k = armneon::fp8_div_up(tpc.intermediate_size, group_size);
            for (int bn = 0; bn < n_blocks_n; bn++) {
              const float* src = down_scale_src + (size_t)bn * full_n_blocks_k + down_scale_src_block_k_offset;
              float* dst = down_scale_dst + (size_t)bn * tp_n_blocks_k;
              std::memcpy(dst, src, sizeof(float) * tp_n_blocks_k);
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
      delete[] (float*)tpc.gate_scale;
      delete[] (float*)tpc.up_scale;
      delete[] (float*)tpc.down_scale;
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (w13_weight_ptrs.size() != (size_t)gpu_tp_count || w13_scale_ptrs.size() != (size_t)gpu_tp_count ||
        w2_weight_ptrs.size() != (size_t)gpu_tp_count || w2_scale_ptrs.size() != (size_t)gpu_tp_count) {
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");
    }

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
        throw std::runtime_error("Requested logical FP8 expert is absent from physical_to_logical_map");
      }
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this, physical_expert_id](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, physical_expert_id, this->config,
                                            w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_ARM_NEON_FP8_MOE_H
