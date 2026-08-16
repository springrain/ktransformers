/**
 * ARM NEON block-scaled FP8 E4M3FN GEMM.
 *
 * Activations stay BF16, weights stay FP8, and block partials accumulate in
 * FP32 before applying their FP32 128x128 scale.
 */
#ifndef CPUINFER_OPERATOR_ARM_NEON_FP8_GEMM_H
#define CPUINFER_OPERATOR_ARM_NEON_FP8_GEMM_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "neon_bf16_gemm.hpp"
#include "neon_fp8_utils.hpp"

namespace armneon {

static inline int fp8_div_up(int value, int divisor) { return (value + divisor - 1) / divisor; }

struct GemmKernelNeonFP8 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 16;
  static constexpr int BLOCK_SIZE = 128;
  static constexpr int N_BLOCK = 128;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 1.0;

  static void config() {}

  static int recommended_nth(int n) { return std::max(1, fp8_div_up(n, N_BLOCK)); }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range(n, ith, nth); }

  using BufferA = GemmKernelNeonBF16::BufferA;
  using BufferC = GemmKernelNeonBF16::BufferC;

  struct BufferB {
    uint8_t* b = nullptr;
    float* d = nullptr;
    size_t n = 0;
    size_t k = 0;
    int block_size = BLOCK_SIZE;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int block_size_, void* ptr) : n(n_), k(k_), block_size(block_size_) {
      b = static_cast<uint8_t*>(ptr);
      d = reinterpret_cast<float*>(b + n * k);
    }

    static size_t required_size(size_t n, size_t k, int block_size) {
      return n * k + static_cast<size_t>(fp8_div_up(static_cast<int>(n), block_size)) *
                         fp8_div_up(static_cast<int>(k), block_size) * sizeof(float);
    }

    void from_mat(const uint8_t* weights, const float* scales, int ith, int nth) {
      const auto [n_start, n_end] = split_range(static_cast<int>(n), ith, nth);
      std::memcpy(b + static_cast<size_t>(n_start) * k, weights + static_cast<size_t>(n_start) * k,
                  static_cast<size_t>(n_end - n_start) * k);
      if (ith == 0) {
        const size_t scale_count = static_cast<size_t>(fp8_div_up(static_cast<int>(n), block_size)) *
                                   fp8_div_up(static_cast<int>(k), block_size);
        std::memcpy(d, scales, scale_count * sizeof(float));
      }
    }
  };
};

static inline void gemm_fp8_block(int m, int n, int k, GemmKernelNeonFP8::BufferA& a, GemmKernelNeonFP8::BufferB& b,
                                  GemmKernelNeonFP8::BufferC& c, int ith, int nth) {
  const auto [n_start, n_end] = split_range(n, ith, nth);
  const int block_size = b.block_size;
  const int k_block_count = fp8_div_up(k, block_size);
  const fp8_e4m3fn_lut decode_lut = load_fp8_e4m3fn_lut();
  constexpr int M_TILE = 4;

  for (int ni = n_start; ni < n_end; ++ni) {
    const uint8_t* weight_row = b.b + static_cast<size_t>(ni) * k;
    const float* scale_row = b.d + static_cast<size_t>(ni / block_size) * k_block_count;

    for (int m_begin = 0; m_begin < m; m_begin += M_TILE) {
      const int rows = std::min(M_TILE, m - m_begin);
      float scalar_tails[M_TILE] = {};
      float32x4_t row_accumulators[M_TILE] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                              vdupq_n_f32(0.0f)};

      for (int k_block = 0; k_block < k; k_block += block_size) {
        const int block_length = std::min(block_size, k - k_block);
        float32x4_t accumulators[M_TILE] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};

        int offset = 0;
        for (; offset + 16 <= block_length; offset += 16) {
          const fp8_bf16x16 decoded = fp8_e4m3fn_to_bf16x16(weight_row + k_block + offset, decode_lut);
          for (int row = 0; row < rows; ++row) {
            const ggml_bf16_t* activation = a.data + static_cast<size_t>(m_begin + row) * a.k + k_block + offset;
            accumulators[row] = fp8_bfdot(accumulators[row], activation, decoded.lo);
            accumulators[row] = fp8_bfdot(accumulators[row], activation + 8, decoded.hi);
          }
        }

        const float scale = scale_row[k_block / block_size];
        for (int row = 0; row < rows; ++row) {
          row_accumulators[row] = vfmaq_n_f32(row_accumulators[row], accumulators[row], scale);
          const ggml_bf16_t* activation = a.data + static_cast<size_t>(m_begin + row) * a.k + k_block;
          for (int tail = offset; tail < block_length; ++tail) {
            scalar_tails[row] +=
                scale * ggml_bf16_to_fp32(activation[tail]) * fp8_e4m3fn_to_fp32(weight_row[k_block + tail]);
          }
        }
      }

      for (int row = 0; row < rows; ++row) {
        c.data[(m_begin + row) * n + ni] = vaddvq_f32(row_accumulators[row]) + scalar_tails[row];
      }
    }
  }
}

static inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernelNeonFP8::BufferA>& a,
                           std::shared_ptr<GemmKernelNeonFP8::BufferB>& b,
                           std::shared_ptr<GemmKernelNeonFP8::BufferC>& c, int ith, int nth) {
  gemm_fp8_block(m, n, k, *a, *b, *c, ith, nth);
}

static inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernelNeonFP8::BufferA>& a,
                           std::shared_ptr<GemmKernelNeonFP8::BufferB>& b,
                           std::shared_ptr<GemmKernelNeonFP8::BufferC>& c, int ith, int nth) {
  gemm_fp8_block(m, n, k, *a, *b, *c, ith, nth);
}

}  // namespace armneon

#endif  // CPUINFER_OPERATOR_ARM_NEON_FP8_GEMM_H
