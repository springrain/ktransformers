/**
 * ARM NEON packed MXFP4 MoE operator.
 *
 * CPU weights remain in their native nibble-packed representation.  The
 * associated UE8M0 (BF16 on the Python boundary) group scales are expanded to
 * FP32 once, while FP4 values are decoded in the NEON GEMM inner loop.
 */
#ifndef CPUINFER_OPERATOR_ARM_NEON_MXFP4_MOE_H
#define CPUINFER_OPERATOR_ARM_NEON_MXFP4_MOE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "moe_base.hpp"
#include "neon_mxfp4_gemm.hpp"

template <class T = armneon::GemmKernelNeonMXFP4>
class NEON_MXFP4_MOE_TP : public NEON_MOE_BASE<T, NEON_MXFP4_MOE_TP<T>> {
  using Base = NEON_MOE_BASE<T, NEON_MXFP4_MOE_TP<T>>;
  using Base::config_;
  using Base::decode_expert_ids_;
  using Base::decode_k_;
  using Base::decode_output_;
  using Base::decode_weights_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_bc_pool_;
  using Base::gate_up_ba_;
  using Base::m_expert_id_map_;
  using Base::m_local_down_output_ptr_;
  using Base::m_local_gate_output_ptr_;
  using Base::m_local_num_;
  using Base::m_local_up_output_ptr_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;
  using Base::up_bc_pool_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  NEON_MXFP4_MOE_TP() = default;
  NEON_MXFP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    const auto& q = config_.quant_config;
    if (q.group_size != armneon::GemmKernelNeonMXFP4::K_GROUP_SIZE || q.zero_point) {
      throw std::runtime_error("NEON MXFP4 requires group_size=32 and zero_point=false");
    }
    if ((config_.hidden_size % q.group_size) != 0 || (config_.intermediate_size % q.group_size) != 0 ||
        (config_.intermediate_size & 1) != 0) {
      throw std::runtime_error("NEON MXFP4 requires even dimensions divisible by group_size=32");
    }
    printf("Created NEON_MXFP4_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx,
           numa_node_of_cpu(sched_getcpu()), q.group_size);
#if defined(__ARM_FEATURE_DOTPROD)
    printf("NEON MXFP4 path: int8 packed decode + SDOT, fused 2-barrier decode\n");
#else
    printf("NEON MXFP4 path: int8 packed decode + widening MAC fallback, fused 2-barrier decode\n");
#endif
  }

  size_t buffer_a_required_size_impl(size_t m, size_t k) const {
    return T::BufferA::required_size(m, k);
  }
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

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, [[maybe_unused]] int qlen) {
    const int m = m_local_num_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    armneon::gemm_mxfp4(m, config_.intermediate_size, config_.hidden_size, *gate_up_ba_[expert_idx], *bb, *bc, ith,
                        nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, [[maybe_unused]] int qlen) {
    const int m = m_local_num_[expert_idx];
    armneon::gemm_mxfp4(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                        *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  // ---------------------------------------------------------------------
  // Fused decode path (qlen == 1), enabled by the presence of these two
  // methods: the base forward_decode then delegates gate/up+activation and
  // down+merge to us.  Mirrors the llamafile forward_one structure — ONE
  // pool job for gate+up GEMM + SiLU + down-input staging, ONE pool job for
  // down GEMM + weighted merge (2 barriers per layer per token).  The stage
  // path the base would otherwise run costs 3 pool barriers plus two serial
  // passes (activation and weighted merge) on the caller thread, which is
  // what made MXFP4 decode slower than the LLAMAFILE backend despite moving
  // fewer bytes.
  // ---------------------------------------------------------------------

  void decode_gate_up_activation(int activated_experts, int qlen) {
    assert(qlen == 1);
    // The base decode skips gate/up C-buffer wiring when this hook exists
    // (a truly bufferless fused kernel would not need it); our GEMM still
    // writes partial results through BufferC, so do the same setup here.
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    const size_t max_m = 1;
    for (int i = 0; i < activated_experts; i++) {
      auto expert_idx = m_expert_id_map_[i];
      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      gate_bc_pool_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(gate_bc_pool_ptr) +
          align64(Base::buffer_c_required_size(max_m, config_.intermediate_size)));
      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      up_bc_pool_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(up_bc_pool_ptr) +
          align64(Base::buffer_c_required_size(max_m, config_.intermediate_size)));
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int inter = config_.intermediate_size;
    const int nth = T::recommended_nth(inter);
    pool->do_work_stealing_job(
        nth * activated_experts, [](int) { T::config(); },
        [this, nth](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          do_gate_up_gemm(false, expert_idx, ith, nth, 1);
          do_gate_up_gemm(true, expert_idx, ith, nth, 1);
          ggml_bf16_t* gate_ptr = m_local_gate_output_ptr_[expert_idx];
          ggml_bf16_t* up_ptr = m_local_up_output_ptr_[expert_idx];
          gate_bc_[expert_idx]->to_mat(1, gate_ptr, ith, nth);
          up_bc_[expert_idx]->to_mat(1, up_ptr, ith, nth);
          // Same activation code as the staged path, on our column block.
          Base::apply_activation_block(expert_idx, ith, nth);
          // Stage the activated block as the down-GEMM input.  BufferA is
          // plain row-major BF16 (from_mat degenerates to a copy), so a
          // partial-block memcpy is exact.
          auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
          std::memcpy(down_ba_[expert_idx]->data + n_start, gate_ptr + n_start,
                      static_cast<size_t>(n_end - n_start) * sizeof(ggml_bf16_t));
        },
        nullptr);
  }

  void decode_down_projection(int activated_experts, int qlen) {
    assert(qlen == 1);
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int hidden = config_.hidden_size;
    const int nth = T::recommended_nth(hidden);
    float* output = decode_output_;
    const int64_t* expert_ids = decode_expert_ids_;
    const float* weights = decode_weights_;
    const int k = decode_k_;
    pool->do_work_stealing_job(
        nth, [](int) { T::config(); },
        [this, nth, hidden, activated_experts, output, expert_ids, weights, k](int ith) {
          auto [n_start, n_end] = T::split_range_n(hidden, ith, nth);
          int e = n_start;
          for (; e + 8 <= n_end; e += 8) armneon::store_v8f32(output + e, armneon::zero_v8f32());
          for (; e < n_end; ++e) output[e] = 0.0f;
          for (int a = 0; a < activated_experts; a++) {
            int expert_idx = m_expert_id_map_[a];
            float weight = 0.0f;
            for (int j = 0; j < k; j++) {
              if (expert_ids[j] == expert_idx) {
                weight = weights[j];
                break;
              }
            }
            do_down_gemm(expert_idx, ith, nth, 1);
            ggml_bf16_t* dptr = m_local_down_output_ptr_[expert_idx];
            down_bc_[expert_idx]->to_mat(1, dptr, ith, nth);
            const armneon::v8f32 wv = armneon::set1_v8f32(weight);
            e = n_start;
            for (; e + 16 <= n_end; e += 16) {
              armneon::v8f32 d0, d1;
              armneon::load_16xbf16_to_2x8xfp32(dptr + e, &d0, &d1);
              armneon::store_v8f32(output + e,
                                   armneon::fmadd_v8f32(d0, wv, armneon::load_v8f32(output + e)));
              armneon::store_v8f32(output + e + 8,
                                   armneon::fmadd_v8f32(d1, wv, armneon::load_v8f32(output + e + 8)));
            }
            for (; e < n_end; ++e) output[e] += ggml_bf16_to_fp32(dptr[e]) * weight;
          }
        },
        nullptr);
  }

  // Load native packed MXFP4 weights.  Python's MXFP4SafeTensorLoader gives
  // per-expert pointers and BF16 scales; flat-buffer mode is retained for
  // callers using the lower-level C++ API.
  void load_weights() {
    const int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = static_cast<const uint64_t*>(config_.physical_to_logical_map);
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const bool use_per_expert = !config_.gate_projs.empty();

    if (!use_per_expert &&
        (config_.gate_proj == nullptr || config_.up_proj == nullptr || config_.down_proj == nullptr ||
         config_.gate_scale == nullptr || config_.up_scale == nullptr || config_.down_scale == nullptr)) {
      throw std::runtime_error("NEON MXFP4 requires packed weight and scale pointers");
    }
    if (use_per_expert) {
      if (config_.up_projs.empty() || config_.down_projs.empty() || config_.gate_scales.empty() ||
          config_.up_scales.empty() || config_.down_scales.empty() ||
          config_.gate_projs[0].size() != config_.up_projs[0].size() ||
          config_.gate_projs[0].size() != config_.down_projs[0].size() ||
          config_.gate_scales[0].size() != config_.up_scales[0].size() ||
          config_.gate_scales[0].size() != config_.down_scales[0].size() ||
          config_.gate_scales[0].size() != config_.gate_projs[0].size()) {
        throw std::runtime_error("NEON MXFP4: per-expert weight/scale pointer arrays have inconsistent sizes");
      }
      if (tp_part_idx > 0) {
        throw std::runtime_error(
            "NEON_MXFP4_MOE_TP per-expert load with tp_part_idx > 0 requires TP_MOE wrapper");
      }
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map](int expert_idx) {
            if (config_.should_skip_expert(expert_idx)) return;
            const uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            if (lid >= config_.gate_projs[0].size() || config_.gate_projs[0][lid] == nullptr ||
                config_.up_projs[0][lid] == nullptr || config_.down_projs[0][lid] == nullptr) {
              throw std::runtime_error("NEON MXFP4: invalid per-expert weight pointer");
            }
            gate_bb_[expert_idx]->from_raw_mat(static_cast<const uint8_t*>(config_.gate_projs[0][lid]), 0, 1);
            up_bb_[expert_idx]->from_raw_mat(static_cast<const uint8_t*>(config_.up_projs[0][lid]), 0, 1);
            down_bb_[expert_idx]->from_raw_mat(static_cast<const uint8_t*>(config_.down_projs[0][lid]), 0, 1);
          },
          nullptr);
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map, group_size](int expert_idx) {
            if (config_.should_skip_expert(expert_idx)) return;
            const uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            if (lid >= config_.gate_scales[0].size() || config_.gate_scales[0][lid] == nullptr ||
                config_.up_scales[0][lid] == nullptr || config_.down_scales[0][lid] == nullptr) {
              throw std::runtime_error("NEON MXFP4: invalid per-expert scale pointer");
            }
            const size_t count = static_cast<size_t>(config_.intermediate_size) * config_.hidden_size / group_size;
            convert_or_copy(gate_bb_[expert_idx]->d, static_cast<const ggml_bf16_t*>(config_.gate_scales[0][lid]),
                            count);
            convert_or_copy(up_bb_[expert_idx]->d, static_cast<const ggml_bf16_t*>(config_.up_scales[0][lid]), count);
            convert_or_copy(down_bb_[expert_idx]->d, static_cast<const ggml_bf16_t*>(config_.down_scales[0][lid]),
                            count);
          },
          nullptr);
      return;
    }

    // Flat [expert, N, K/2] buffers.
    const size_t packed_per_expert = static_cast<size_t>(config_.intermediate_size) * config_.hidden_size / 2;
    const size_t scale_per_expert = static_cast<size_t>(config_.intermediate_size) * config_.hidden_size / group_size;
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, packed_per_expert](int task_id) {
          const int expert_idx = task_id / nth;
          if (config_.should_skip_expert(expert_idx)) return;
          const int ith = task_id % nth;
          const uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
          const uint8_t* gate = static_cast<const uint8_t*>(config_.gate_proj) + lid * packed_per_expert;
          const uint8_t* up = static_cast<const uint8_t*>(config_.up_proj) + lid * packed_per_expert;
          gate_bb_[expert_idx]->from_raw_mat(gate, ith, nth);
          up_bb_[expert_idx]->from_raw_mat(up, ith, nth);
        },
        nullptr);
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, packed_per_expert](int task_id) {
          const int expert_idx = task_id / nth;
          if (config_.should_skip_expert(expert_idx)) return;
          const int ith = task_id % nth;
          const uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
          const uint8_t* down = static_cast<const uint8_t*>(config_.down_proj) + lid * packed_per_expert;
          down_bb_[expert_idx]->from_raw_mat(down, ith, nth);
        },
        nullptr);
    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map, group_size, scale_per_expert](int expert_idx) {
          if (config_.should_skip_expert(expert_idx)) return;
          const uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
          const size_t off = lid * scale_per_expert;
          convert_or_copy(gate_bb_[expert_idx]->d, static_cast<const ggml_bf16_t*>(config_.gate_scale) + off,
                          scale_per_expert);
          convert_or_copy(up_bb_[expert_idx]->d, static_cast<const ggml_bf16_t*>(config_.up_scale) + off,
                          scale_per_expert);
          convert_or_copy(down_bb_[expert_idx]->d, static_cast<const ggml_bf16_t*>(config_.down_scale) + off,
                          scale_per_expert);
        },
        nullptr);
  }

  // Copy one CPU TP expert into the GPU's packed MXFP4 staging buffers.  The
  // loops intentionally map by global N/K, so cpu_tp_count and gpu_tp_count
  // need not be equal or have a divisibility relationship.
  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs,
                               WorkerPool* work_pool = nullptr) const {
    if (expert_id < 0 || expert_id >= config_.expert_num || !gate_bb_[expert_id] || !up_bb_[expert_id] ||
        !down_bb_[expert_id] || !gate_bb_[expert_id]->b || !up_bb_[expert_id]->b || !down_bb_[expert_id]->b) {
      throw std::runtime_error("NEON MXFP4 staging: invalid expert");
    }
    if (gpu_tp_count <= 0 || w13_weight_ptrs.size() != static_cast<size_t>(gpu_tp_count) ||
        w13_scale_ptrs.size() != static_cast<size_t>(gpu_tp_count) ||
        w2_weight_ptrs.size() != static_cast<size_t>(gpu_tp_count) ||
        w2_scale_ptrs.size() != static_cast<size_t>(gpu_tp_count)) {
      throw std::runtime_error("NEON MXFP4 staging: pointer arrays do not match gpu_tp_count");
    }
    const int group_size = config_.quant_config.group_size;
    const int cpu_n = config_.intermediate_size;
    const int cpu_k = config_.hidden_size;
    if (cpu_tp_count <= 0 || full_config.intermediate_size % cpu_tp_count != 0 ||
        cpu_n != full_config.intermediate_size / cpu_tp_count || tp_part_idx < 0 || tp_part_idx >= cpu_tp_count) {
      throw std::runtime_error("NEON MXFP4 staging: CPU TP dimensions do not match full intermediate size");
    }
    const int gpu_n = full_config.intermediate_size / gpu_tp_count;
    if (gpu_n <= 0 || full_config.intermediate_size % gpu_tp_count != 0 || (gpu_n % group_size) != 0 ||
        (gpu_n & 1) != 0 || (cpu_n % group_size) != 0 || (cpu_n & 1) != 0) {
      throw std::runtime_error("NEON MXFP4 staging: TP dimensions are not group/nibble aligned");
    }
    const int global_n_offset = tp_part_idx * cpu_n;
    if (global_n_offset < 0 || global_n_offset + cpu_n > full_config.intermediate_size) {
      throw std::runtime_error("NEON MXFP4 staging: CPU TP slice exceeds full intermediate dimension");
    }
    const size_t cpu_row_bytes = static_cast<size_t>(cpu_k) / 2;
    if (full_config.hidden_size != cpu_k || (full_config.hidden_size % group_size) != 0) {
      throw std::runtime_error("NEON MXFP4 staging: hidden dimensions do not match");
    }
    const size_t gpu_w13_weight_per_mat = static_cast<size_t>(gpu_n) * cpu_k / 2;
    const int cpu_k_groups = cpu_k / group_size;
    const int gpu_k_groups = cpu_k / group_size;

    auto* selected_pool = work_pool != nullptr ? work_pool : config_.pool;
    auto pool = selected_pool->get_subpool(tp_part_idx);
    constexpr int ROW_TASKS = 32;
    const int total = ROW_TASKS * 2 + ROW_TASKS;
    pool->do_work_stealing_job(
        total, nullptr,
        [=, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs, this](int task_id) {
          if (task_id < ROW_TASKS * 2) {
            const bool is_up = task_id >= ROW_TASKS;
            const int chunk = task_id % ROW_TASKS;
            const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];
            const int rows_per = (cpu_n + ROW_TASKS - 1) / ROW_TASKS;
            const int begin = chunk * rows_per;
            const int end = std::min(cpu_n, begin + rows_per);
            for (int row = begin; row < end; ++row) {
              const int global_n = global_n_offset + row;
              const int target = global_n / gpu_n;
              const int n_in_gpu = global_n % gpu_n;
              uint8_t* wdst = reinterpret_cast<uint8_t*>(w13_weight_ptrs[target]);
              const size_t matrix_off = is_up ? gpu_w13_weight_per_mat : 0;
              std::memcpy(wdst + matrix_off + static_cast<size_t>(n_in_gpu) * cpu_k / 2,
                          bb->b + static_cast<size_t>(row) * cpu_row_bytes, cpu_row_bytes);
              // SGLang's MXFP4 GPU staging scale tensor is BF16 (UE8M0 is
              // exactly representable in BF16), matching the Python loader.
              ggml_bf16_t* sdst = reinterpret_cast<ggml_bf16_t*>(w13_scale_ptrs[target]);
              const size_t scale_matrix = static_cast<size_t>(gpu_n) * gpu_k_groups;
              convert_or_copy(sdst + (is_up ? scale_matrix : 0) + static_cast<size_t>(n_in_gpu) * gpu_k_groups,
                              bb->d + static_cast<size_t>(row) * cpu_k_groups, static_cast<size_t>(gpu_k_groups));
            }
          } else {
            const int chunk = task_id - ROW_TASKS * 2;
            const auto& bb = down_bb_[expert_id];
            const int rows_per = (full_config.hidden_size + ROW_TASKS - 1) / ROW_TASKS;
            const int begin = chunk * rows_per;
            const int end = std::min(full_config.hidden_size, begin + rows_per);
            // W2 is [hidden, intermediate].  Its K dimension is the local
            // intermediate slice, unlike W13 whose K dimension is hidden.
            const int cpu_w2_k = cpu_n;
            const int gpu_w2_k = gpu_n;
            const int global_k_offset = global_n_offset;
            const int cpu_w2_groups = cpu_w2_k / group_size;
            const int gpu_w2_groups = gpu_w2_k / group_size;
            for (int row = begin; row < end; ++row) {
              int local_start = 0;
              while (local_start < cpu_w2_k) {
                const int global_k = global_k_offset + local_start;
                const int target = global_k / gpu_w2_k;
                const int k_in_gpu = global_k % gpu_w2_k;
                const int len = std::min(cpu_w2_k - local_start, gpu_w2_k - k_in_gpu);
                if ((local_start & 1) || (k_in_gpu & 1) || (len & 1) || (local_start % group_size) ||
                    (k_in_gpu % group_size) || (len % group_size)) {
                  throw std::runtime_error("NEON MXFP4 staging: down slice is not nibble/group aligned");
                }
                uint8_t* wdst = reinterpret_cast<uint8_t*>(w2_weight_ptrs[target]);
                std::memcpy(wdst + static_cast<size_t>(row) * gpu_w2_k / 2 + k_in_gpu / 2,
                            bb->b + static_cast<size_t>(row) * cpu_w2_k / 2 + local_start / 2, len / 2);
                ggml_bf16_t* sdst = reinterpret_cast<ggml_bf16_t*>(w2_scale_ptrs[target]);
                convert_or_copy(sdst + static_cast<size_t>(row) * gpu_w2_groups + k_in_gpu / group_size,
                                bb->d + static_cast<size_t>(row) * cpu_w2_groups + local_start / group_size,
                                static_cast<size_t>(len / group_size));
                local_start += len;
              }
            }
          }
        },
        nullptr);
  }
};

// TP specialization: split packed rows/columns without expanding the FP4
// weights.  This mirrors the AVX2 MXFP4 loader but uses the NEON base buffers.
template <typename K>
class TP_MOE<NEON_MXFP4_MOE_TP<K>> : public TP_MOE<NEON_MOE_BASE<K, NEON_MXFP4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<NEON_MOE_BASE<K, NEON_MXFP4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* map = static_cast<const uint64_t*>(config.physical_to_logical_map);
    const int group_size = config.quant_config.group_size;
    if (group_size != 32 || config.quant_config.zero_point) {
      throw std::runtime_error("NEON MXFP4 TP requires group_size=32 and zero_point=false");
    }
    const bool use_per_expert = !config.gate_projs.empty();
    if (!use_per_expert &&
        (config.gate_proj == nullptr || config.up_proj == nullptr || config.down_proj == nullptr ||
         config.gate_scale == nullptr || config.up_scale == nullptr || config.down_scale == nullptr)) {
      throw std::runtime_error("NEON MXFP4 TP requires packed weights and scales");
    }
    if (use_per_expert &&
        (config.up_projs.empty() || config.down_projs.empty() || config.gate_scales.empty() ||
         config.up_scales.empty() || config.down_scales.empty() ||
         config.gate_projs[0].size() != config.up_projs[0].size() ||
         config.gate_projs[0].size() != config.down_projs[0].size() ||
         config.gate_scales[0].size() != config.up_scales[0].size() ||
         config.gate_scales[0].size() != config.down_scales[0].size() ||
         config.gate_scales[0].size() != config.gate_projs[0].size())) {
      throw std::runtime_error("NEON MXFP4 TP: per-expert weight/scale pointer arrays have inconsistent sizes");
    }

    if (use_per_expert) {
      const int full_n = config.intermediate_size;
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto* tp = tps[i].get();
        auto& tc = tp->config_;
        const int local_n = tc.intermediate_size;
        if ((local_n & 1) || (local_n % group_size))
          throw std::runtime_error("NEON MXFP4 TP: local intermediate dimension is not aligned");
        const size_t local_gate_bytes = static_cast<size_t>(local_n) * tc.hidden_size / 2;
        const size_t local_scale_count = static_cast<size_t>(local_n) * tc.hidden_size / group_size;
        const int full_groups = full_n / group_size;
        const int local_groups = local_n / group_size;
        auto subpool = pool->get_subpool(i);
        subpool->do_work_stealing_job(
            tc.expert_num, nullptr,
            [&, i, local_n, local_gate_bytes, local_scale_count, full_groups, local_groups](int eid) {
              if (tc.should_skip_expert(eid)) return;
              const uint64_t lid = expert_map(map, eid);
              if (lid >= config.gate_projs[0].size() || config.gate_projs[0][lid] == nullptr ||
                  config.up_projs[0][lid] == nullptr || config.down_projs[0][lid] == nullptr ||
                  config.gate_scales[0][lid] == nullptr || config.up_scales[0][lid] == nullptr ||
                  config.down_scales[0][lid] == nullptr) {
                throw std::runtime_error("NEON MXFP4 TP: invalid expert pointer");
              }
              auto& tp_ref = *tps[i];
              if (!tp_ref.gate_bb_[eid] || !tp_ref.gate_bb_[eid]->b) {
                throw std::runtime_error("NEON MXFP4 TP: missing destination BufferB");
              }
              const size_t n_byte_off = static_cast<size_t>(i) * local_gate_bytes;
              std::memcpy(tp_ref.gate_bb_[eid]->b, static_cast<const uint8_t*>(config.gate_projs[0][lid]) + n_byte_off,
                          local_gate_bytes);
              std::memcpy(tp_ref.up_bb_[eid]->b, static_cast<const uint8_t*>(config.up_projs[0][lid]) + n_byte_off,
                          local_gate_bytes);
              const size_t scale_off = static_cast<size_t>(i) * local_n * tc.hidden_size / group_size;
              convert_or_copy(tp_ref.gate_bb_[eid]->d,
                              static_cast<const ggml_bf16_t*>(config.gate_scales[0][lid]) + scale_off,
                              local_scale_count);
              convert_or_copy(tp_ref.up_bb_[eid]->d,
                              static_cast<const ggml_bf16_t*>(config.up_scales[0][lid]) + scale_off,
                              local_scale_count);

              const uint8_t* src_down = static_cast<const uint8_t*>(config.down_projs[0][lid]);
              uint8_t* dst_down = tp_ref.down_bb_[eid]->b;
              const size_t local_row_bytes = static_cast<size_t>(local_n) / 2;
              const size_t full_row_bytes = static_cast<size_t>(full_n) / 2;
              const size_t k_byte_off = static_cast<size_t>(i) * local_row_bytes;
              for (int row = 0; row < tc.hidden_size; ++row) {
                std::memcpy(dst_down + static_cast<size_t>(row) * local_row_bytes,
                            src_down + static_cast<size_t>(row) * full_row_bytes + k_byte_off, local_row_bytes);
              }
              const ggml_bf16_t* src_ds = static_cast<const ggml_bf16_t*>(config.down_scales[0][lid]);
              for (int row = 0; row < tc.hidden_size; ++row) {
                convert_or_copy(tp_ref.down_bb_[eid]->d + static_cast<size_t>(row) * local_groups,
                                src_ds + static_cast<size_t>(row) * full_groups + i * local_groups, local_groups);
              }
            },
            nullptr);
      });
    } else {
      // Flat mode: construct TP-sliced temporary tensors, then let the
      // concrete loader copy them into its owned BufferB storage.
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tc = tps[i]->config_;
        const size_t elems = static_cast<size_t>(tc.intermediate_size) * tc.hidden_size;
        const size_t scales = elems / group_size;
        tc.gate_proj = new uint8_t[tc.expert_num * elems / 2];
        tc.up_proj = new uint8_t[tc.expert_num * elems / 2];
        tc.down_proj = new uint8_t[tc.expert_num * elems / 2];
        tc.gate_scale = new ggml_bf16_t[tc.expert_num * scales];
        tc.up_scale = new ggml_bf16_t[tc.expert_num * scales];
        tc.down_scale = new ggml_bf16_t[tc.expert_num * scales];
      });
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tc = tps[i]->config_;
        const size_t local_elems = static_cast<size_t>(tc.intermediate_size) * tc.hidden_size;
        const size_t local_bytes = local_elems / 2;
        const size_t local_scales = local_elems / group_size;
        const size_t full_bytes = static_cast<size_t>(config.intermediate_size) * config.hidden_size / 2;
        const size_t full_scales = static_cast<size_t>(config.intermediate_size) * config.hidden_size / group_size;
        const size_t local_n = tc.intermediate_size;
        const size_t full_n = config.intermediate_size;
        const size_t local_groups = local_n / group_size;
        pool->get_subpool(i)->do_work_stealing_job(
            tc.expert_num, nullptr,
            [&, i, local_bytes, local_scales, full_bytes, full_scales, local_n, full_n, local_groups](int eid) {
              const uint64_t lid = expert_map(map, eid);
              uint8_t* dg = static_cast<uint8_t*>(tps[i]->config_.gate_proj) + static_cast<size_t>(eid) * local_bytes;
              uint8_t* du = static_cast<uint8_t*>(tps[i]->config_.up_proj) + static_cast<size_t>(eid) * local_bytes;
              uint8_t* dd = static_cast<uint8_t*>(tps[i]->config_.down_proj) + static_cast<size_t>(eid) * local_bytes;
              const uint8_t* sg = static_cast<const uint8_t*>(config.gate_proj) + lid * full_bytes + i * local_bytes;
              const uint8_t* su = static_cast<const uint8_t*>(config.up_proj) + lid * full_bytes + i * local_bytes;
              std::memcpy(dg, sg, local_bytes);
              std::memcpy(du, su, local_bytes);
              const uint8_t* sd = static_cast<const uint8_t*>(config.down_proj) + lid * full_bytes;
              for (int row = 0; row < tc.hidden_size; ++row) {
                std::memcpy(dd + static_cast<size_t>(row) * local_n / 2,
                            sd + static_cast<size_t>(row) * full_n / 2 + i * local_n / 2, local_n / 2);
              }
              const ggml_bf16_t* sgs = static_cast<const ggml_bf16_t*>(config.gate_scale) + lid * full_scales +
                                       i * local_scales;
              const ggml_bf16_t* sus = static_cast<const ggml_bf16_t*>(config.up_scale) + lid * full_scales +
                                       i * local_scales;
              std::memcpy(static_cast<ggml_bf16_t*>(tps[i]->config_.gate_scale) + eid * local_scales, sgs,
                          local_scales * sizeof(ggml_bf16_t));
              std::memcpy(static_cast<ggml_bf16_t*>(tps[i]->config_.up_scale) + eid * local_scales, sus,
                          local_scales * sizeof(ggml_bf16_t));
              const ggml_bf16_t* sds = static_cast<const ggml_bf16_t*>(config.down_scale) + lid * full_scales;
              ggml_bf16_t* dds = static_cast<ggml_bf16_t*>(tps[i]->config_.down_scale) + eid * local_scales;
              const size_t full_row_groups = full_n / group_size;
              for (int row = 0; row < tc.hidden_size; ++row) {
                std::memcpy(dds + static_cast<size_t>(row) * local_groups,
                            sds + static_cast<size_t>(row) * full_row_groups + i * local_groups,
                            local_groups * sizeof(ggml_bf16_t));
              }
            },
            nullptr);
      });
      // The temporary tensors are already arranged by physical expert slot;
      // prevent the concrete loader from applying the original EPLB map a
      // second time.
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        tps[i]->config_.physical_to_logical_map = nullptr;
      });
      // The temporary buffers above are already arranged by physical expert
      // slot. Calling DO_TPS_LOAD_WEIGHTS here would re-apply the original
      // physical->logical map and silently permute them a second time.
      pool->dispense_backend()->do_numa_job([&, this](int i) { tps[i]->load_weights(); });
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tc = tps[i]->config_;
        delete[] static_cast<uint8_t*>(tc.gate_proj);
        delete[] static_cast<uint8_t*>(tc.up_proj);
        delete[] static_cast<uint8_t*>(tc.down_proj);
        delete[] static_cast<ggml_bf16_t*>(tc.gate_scale);
        delete[] static_cast<ggml_bf16_t*>(tc.up_scale);
        delete[] static_cast<ggml_bf16_t*>(tc.down_scale);
        tc.gate_proj = tc.up_proj = tc.down_proj = nullptr;
        tc.gate_scale = tc.up_scale = tc.down_scale = nullptr;
      });
    }
    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    write_weight_scale_to_buffer_with_pool(this->config.pool, gpu_tp_count, expert_id, w13_weight_ptrs,
                                           w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
  }

  void write_weight_scale_to_buffer_with_pool(
      WorkerPool* writer_pool, int gpu_tp_count, int expert_id,
      const std::vector<uintptr_t>& w13_weight_ptrs,
      const std::vector<uintptr_t>& w13_scale_ptrs,
      const std::vector<uintptr_t>& w2_weight_ptrs,
      const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (writer_pool == nullptr) throw std::runtime_error("NEON MXFP4 staging: writer pool is null");
    if (w13_weight_ptrs.size() != static_cast<size_t>(gpu_tp_count) ||
        w13_scale_ptrs.size() != static_cast<size_t>(gpu_tp_count) ||
        w2_weight_ptrs.size() != static_cast<size_t>(gpu_tp_count) ||
        w2_scale_ptrs.size() != static_cast<size_t>(gpu_tp_count))
      throw std::runtime_error("NEON MXFP4 staging: pointer arrays do not match gpu_tp_count");

    int physical_expert_id = expert_id;
    const uint64_t* map = static_cast<const uint64_t*>(this->config.physical_to_logical_map);
    if (map != nullptr) {
      physical_expert_id = -1;
      for (int i = 0; i < this->config.expert_num; ++i) {
        if (map[i] == static_cast<uint64_t>(expert_id)) {
          physical_expert_id = i;
          break;
        }
      }
      if (physical_expert_id < 0) throw std::runtime_error("NEON MXFP4 staging: expert is absent from map");
    }
    writer_pool->dispense_backend()->do_numa_job([&, this, writer_pool, physical_expert_id](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, physical_expert_id, this->config,
                                             w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs,
                                             writer_pool);
    });
  }
};

#endif  // CPUINFER_OPERATOR_ARM_NEON_MXFP4_MOE_H
