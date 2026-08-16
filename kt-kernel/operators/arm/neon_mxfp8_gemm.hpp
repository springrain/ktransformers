/** ARM NEON MXFP8 E4M3FN GEMM with per-32-element UE8M0 scales. */
#ifndef CPUINFER_OPERATOR_ARM_NEON_MXFP8_GEMM_H
#define CPUINFER_OPERATOR_ARM_NEON_MXFP8_GEMM_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

#include "neon_bf16_gemm.hpp"
#include "neon_fp8_utils.hpp"

namespace armneon {

struct GemmKernelNeonMXFP8 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;
  static constexpr int K_GROUP_SIZE = 32;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 6144;
  static constexpr double ELEMENT_SIZE = 1.0;

  static void config() {}
  static int recommended_nth(int n) { return std::max(1, (n + N_BLOCK - 1) / N_BLOCK); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range(n, ith, nth); }

  using BufferA = GemmKernelNeonBF16::BufferA;
  using BufferC = GemmKernelNeonBF16::BufferC;

  struct BufferB {
    uint8_t* b = nullptr;
    float* d = nullptr;
    int n = 0;
    int k = 0;
    int k_group_size = K_GROUP_SIZE;
    int k_group_count = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int group_size, void* ptr)
        : n(static_cast<int>(n_)), k(static_cast<int>(k_)), k_group_size(group_size) {
      if (group_size != K_GROUP_SIZE || k % group_size != 0) {
        throw std::runtime_error("NEON MXFP8 requires group_size=32 and K divisible by 32");
      }
      k_group_count = k / group_size;
      b = static_cast<uint8_t*>(ptr);
      d = reinterpret_cast<float*>(b + static_cast<size_t>(n) * k);
    }

    static size_t required_size(size_t n, size_t k, int group_size) {
      return n * k + n * (k / group_size) * sizeof(float);
    }

    void from_raw_mat(const uint8_t* weights, int ith, int nth) {
      const auto [n_start, n_end] = split_range(n, ith, nth);
      if (n_start < n_end) {
        std::memcpy(b + static_cast<size_t>(n_start) * k, weights + static_cast<size_t>(n_start) * k,
                    static_cast<size_t>(n_end - n_start) * k);
      }
    }

    uint8_t* get_submat(int n_begin, int k_begin) { return b + static_cast<size_t>(n_begin) * k + k_begin; }
    float* get_scale(int n_begin, int k_begin) {
      return d + static_cast<size_t>(n_begin) * k_group_count + k_begin / k_group_size;
    }
  };
};

static inline void gemm_mxfp8_kgroup(int m, int n, [[maybe_unused]] int k, GemmKernelNeonMXFP8::BufferA& a,
                                     GemmKernelNeonMXFP8::BufferB& b, GemmKernelNeonMXFP8::BufferC& c, int ith,
                                     int nth) {
  const auto [n_start, n_end] = split_range(n, ith, nth);
  const fp8_e4m3fn_lut decode_lut = load_fp8_e4m3fn_lut();
  constexpr int M_TILE = 4;

  for (int ni = n_start; ni < n_end; ++ni) {
    const uint8_t* weight_row = b.get_submat(ni, 0);
    const float* scales = b.get_scale(ni, 0);

    for (int m_begin = 0; m_begin < m; m_begin += M_TILE) {
      const int rows = std::min(M_TILE, m - m_begin);
      float32x4_t row_accumulators[M_TILE] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                              vdupq_n_f32(0.0f)};

      for (int group = 0; group < b.k_group_count; ++group) {
        const int k_base = group * GemmKernelNeonMXFP8::K_GROUP_SIZE;
        const fp8_bf16x16 first = fp8_e4m3fn_to_bf16x16(weight_row + k_base, decode_lut);
        const fp8_bf16x16 second = fp8_e4m3fn_to_bf16x16(weight_row + k_base + 16, decode_lut);

        for (int row = 0; row < rows; ++row) {
          const ggml_bf16_t* activation = a.data + static_cast<size_t>(m_begin + row) * a.k + k_base;
          float32x4_t group_accumulator = vdupq_n_f32(0.0f);
          group_accumulator = fp8_bfdot(group_accumulator, activation, first.lo);
          group_accumulator = fp8_bfdot(group_accumulator, activation + 8, first.hi);
          group_accumulator = fp8_bfdot(group_accumulator, activation + 16, second.lo);
          group_accumulator = fp8_bfdot(group_accumulator, activation + 24, second.hi);
          row_accumulators[row] = vfmaq_n_f32(row_accumulators[row], group_accumulator, scales[group]);
        }
      }

      for (int row = 0; row < rows; ++row) {
        c.data[(m_begin + row) * n + ni] = vaddvq_f32(row_accumulators[row]);
      }
    }
  }
}

static inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernelNeonMXFP8::BufferA>& a,
                           std::shared_ptr<GemmKernelNeonMXFP8::BufferB>& b,
                           std::shared_ptr<GemmKernelNeonMXFP8::BufferC>& c, int ith, int nth) {
  gemm_mxfp8_kgroup(m, n, k, *a, *b, *c, ith, nth);
}

static inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernelNeonMXFP8::BufferA>& a,
                           std::shared_ptr<GemmKernelNeonMXFP8::BufferB>& b,
                           std::shared_ptr<GemmKernelNeonMXFP8::BufferC>& c, int ith, int nth) {
  gemm_mxfp8_kgroup(m, n, k, *a, *b, *c, ith, nth);
}

}  // namespace armneon

#endif  // CPUINFER_OPERATOR_ARM_NEON_MXFP8_GEMM_H
