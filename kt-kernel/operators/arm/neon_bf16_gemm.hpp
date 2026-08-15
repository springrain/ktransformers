/**
 * @Description  : ARM NEON BF16 GEMM kernel with trivial Buffer abstractions
 * @Author       : Claude
 * @Date         : 2026-08-09
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * NEON port of avx2/avx2_bf16_gemm.hpp. All buffers keep plain row-major
 * storage (no packing), so BufferA/B/C semantics match the AVX2 kernel one
 * to one - only the SIMD dot product differs.
 *
 * Two dot-product implementations are provided:
 *   1. __ARM_FEATURE_BF16_VECTOR_ARITHMETIC (Armv8.6-A, e.g. Ampere One): native BFDOT
 *      instructions, one instruction per 8 BF16 multiplies.
 *   2. Plain AArch64 (any ARMv8.0+): widen BF16->FP32 and use FMA.
 *
 * GEMM: C[m,n] = sum_k A[m,k] * B[n,k]
 *   A: [M, K] row-major BF16 (input activations)
 *   B: [N, K] row-major BF16 (weights, each row is one output neuron)
 *   C: [M, N] row-major FP32 (output)
 **/
#ifndef CPUINFER_OPERATOR_ARM_NEON_BF16_GEMM_H
#define CPUINFER_OPERATOR_ARM_NEON_BF16_GEMM_H

#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>

#include "neon_bf16_utils.hpp"

namespace armneon {

// Split range [0, total) among nth threads, return [start, end) for thread ith
static inline std::pair<int, int> split_range(int total, int ith, int nth) {
  int per = total / nth;
  int rem = total % nth;
  int start = ith * per + std::min(ith, rem);
  int end = start + per + (ith < rem ? 1 : 0);
  return {start, end};
}

struct GemmKernelNeonBF16 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;      // No M-direction padding needed (matches AVX2)
  static constexpr int N_STEP = 8;      // 8-wide FP32 (2x NEON)
  static constexpr int K_STEP = 8;      // Process 8 K elements at a time
  static constexpr int N_BLOCK = 64;    // N blocking for cache
  static constexpr int K_BLOCK = 256;   // K blocking for cache
  static constexpr double ELEMENT_SIZE = 2.0;  // BF16 = 2 bytes

  // No tile configuration needed
  static void config() {}

  // Thread count for N-dimension parallelism
  // Must return >= 1 to avoid division by zero in moe_base task dispatch
  static int recommended_nth(int n) {
    return std::max(1, n / N_STEP);
  }

  // Split N range for multi-threaded GEMM
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return split_range(n, ith, nth);
  }

  // ========================================================================
  // BufferA: Input activations [M, K] row-major BF16
  // from_mat() = memcpy (no packing needed)
  // ========================================================================
  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void* ptr) : max_m(m), k(k_), data((ggml_bf16_t*)ptr) {}

    static size_t required_size(size_t m, size_t k) {
      return m * k * sizeof(ggml_bf16_t);
    }

    void set_data(void* ptr) { data = (ggml_bf16_t*)ptr; }

    // Copy input rows into buffer (trivial memcpy)
    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(ggml_bf16_t));
      } else {
        // Multi-threaded: split by rows
        auto [m_start, m_end] = split_range(m, ith, nth);
        std::memcpy(data + m_start * k, src + m_start * k,
                    (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  // ========================================================================
  // BufferB: Weight matrix [N, K] row-major BF16
  // from_mat() = memcpy (no transpose/packing needed)
  // ========================================================================
  struct BufferB {
    ggml_bf16_t* b = nullptr;
    size_t n = 0;
    size_t k = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, void* ptr) : n(n_), k(k_), b((ggml_bf16_t*)ptr) {}

    static size_t required_size(size_t n, size_t k) {
      return n * k * sizeof(ggml_bf16_t);
    }

    // Copy weight data (multi-threaded by N dimension)
    void from_mat(const ggml_bf16_t* src, int ith, int nth) {
      auto [n_start, n_end] = split_range((int)n, ith, nth);
      std::memcpy(b + n_start * k, src + n_start * k,
                  (size_t)(n_end - n_start) * k * sizeof(ggml_bf16_t));
    }
  };

  // ========================================================================
  // BufferC: Output matrix [M, N] row-major FP32
  // to_mat() converts FP32 -> BF16 and writes out
  // ========================================================================
  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : max_m(m), n(n_), data((float*)ptr) {}

    static size_t required_size(size_t m, size_t n) {
      return m * n * sizeof(float);
    }

    void set_data(void* ptr) { data = (float*)ptr; }

    // Convert FP32 output to BF16 and write to destination
    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = split_range_n((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + mi * n;
        ggml_bf16_t* dst_row = dst + mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) {
          store_fp32_to_bf16(dst_row + j, load_v8f32(src_row + j));
        }
        // Scalar tail
        for (; j < n_end; j++) {
          dst_row[j] = ggml_fp32_to_bf16(src_row[j]);
        }
      }
    }
  };
};

// ============================================================================
// BF16 dot product: sum_k a[k] * b[k] accumulated in FP32
// ============================================================================

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

// Native BFDOT: each vbfdotq_f32 computes 8 BF16 products and accumulates
// 4 FP32 partial sums in a single instruction.
static inline float dot_bf16(const ggml_bf16_t* a, const ggml_bf16_t* b, int k) {
  float32x4_t c0 = vdupq_n_f32(0.0f);
  float32x4_t c1 = vdupq_n_f32(0.0f);
  float32x4_t c2 = vdupq_n_f32(0.0f);
  float32x4_t c3 = vdupq_n_f32(0.0f);

  int ki = 0;
  for (; ki + 32 <= k; ki += 32) {
    c0 = vbfdotq_f32(c0, vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)(a + ki))),
                     vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)(b + ki))));
    c1 = vbfdotq_f32(c1, vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)(a + ki + 8))),
                     vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)(b + ki + 8))));
    c2 = vbfdotq_f32(c2, vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)(a + ki + 16))),
                     vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)(b + ki + 16))));
    c3 = vbfdotq_f32(c3, vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)(a + ki + 24))),
                     vreinterpretq_bf16_u16(vld1q_u16((const uint16_t*)(b + ki + 24))));
  }

  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(c0, c2), vaddq_f32(c1, c3)));

  // Scalar tail
  for (; ki < k; ki++) {
    sum += ggml_bf16_to_fp32(a[ki]) * ggml_bf16_to_fp32(b[ki]);
  }
  return sum;
}

#else  // !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC

// Portable path: widen BF16 -> FP32 and accumulate with FMA.
static inline float dot_bf16(const ggml_bf16_t* a, const ggml_bf16_t* b, int k) {
  v8f32 c0 = zero_v8f32();
  v8f32 c1 = zero_v8f32();
  v8f32 c2 = zero_v8f32();
  v8f32 c3 = zero_v8f32();

  int ki = 0;
  for (; ki + 32 <= k; ki += 32) {
    c0 = fmadd_v8f32(load_bf16_to_fp32(a + ki), load_bf16_to_fp32(b + ki), c0);
    c1 = fmadd_v8f32(load_bf16_to_fp32(a + ki + 8), load_bf16_to_fp32(b + ki + 8), c1);
    c2 = fmadd_v8f32(load_bf16_to_fp32(a + ki + 16), load_bf16_to_fp32(b + ki + 16), c2);
    c3 = fmadd_v8f32(load_bf16_to_fp32(a + ki + 24), load_bf16_to_fp32(b + ki + 24), c3);
  }

  float sum = hsum_neon(add_v8f32(add_v8f32(c0, c2), add_v8f32(c1, c3)));

  // Scalar tail
  for (; ki < k; ki++) {
    sum += ggml_bf16_to_fp32(a[ki]) * ggml_bf16_to_fp32(b[ki]);
  }
  return sum;
}

#endif  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

// ============================================================================
// NEON BF16 GEMM functions
// C[m,n] = sum_k A[m,k] * B[n,k]
// ============================================================================

// General GEMM (works for both vec_mul m=1 and mat_mul m>1)
static inline void gemm_bf16(
    int m, int n, int k,
    GemmKernelNeonBF16::BufferA& a,
    GemmKernelNeonBF16::BufferB& b,
    GemmKernelNeonBF16::BufferC& c,
    int ith, int nth) {

  auto [n_start, n_end] = split_range(n, ith, nth);

  for (int ni = n_start; ni < n_end; ni++) {
    const ggml_bf16_t* b_row = b.b + (size_t)ni * k;

    for (int mi = 0; mi < m; mi++) {
      const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
      c.data[mi * n + ni] = dot_bf16(a_row, b_row, k);
    }
  }
}

// vec_mul: dispatch to gemm_bf16
static inline void vec_mul(
    int m, int n, int k,
    std::shared_ptr<GemmKernelNeonBF16::BufferA>& a,
    std::shared_ptr<GemmKernelNeonBF16::BufferB>& b,
    std::shared_ptr<GemmKernelNeonBF16::BufferC>& c,
    int ith, int nth) {
  gemm_bf16(m, n, k, *a, *b, *c, ith, nth);
}

// mat_mul: dispatch to gemm_bf16
static inline void mat_mul(
    int m, int n, int k,
    std::shared_ptr<GemmKernelNeonBF16::BufferA>& a,
    std::shared_ptr<GemmKernelNeonBF16::BufferB>& b,
    std::shared_ptr<GemmKernelNeonBF16::BufferC>& c,
    int ith, int nth) {
  gemm_bf16(m, n, k, *a, *b, *c, ith, nth);
}

}  // namespace armneon

#endif  // CPUINFER_OPERATOR_ARM_NEON_BF16_GEMM_H
