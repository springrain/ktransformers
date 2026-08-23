/**
 * ARM NEON packed MXFP4 GEMM.
 *
 * MXFP4 stores two FP4 E2M1 values in each byte and one BF16-compatible
 * (UE8M0) scale for every group of 32 K values.  The packed weights stay in
 * BufferB; FP4 values are expanded to BF16 vectors immediately before the
 * dot product.  On Armv8.6-A targets (including Ampere One) the dot product
 * uses BFDOT, while older AArch64 targets use the portable FP32 FMA path.
 */
#ifndef CPUINFER_OPERATOR_ARM_NEON_MXFP4_GEMM_H
#define CPUINFER_OPERATOR_ARM_NEON_MXFP4_GEMM_H

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

#include "neon_bf16_gemm.hpp"
#include "neon_fp8_utils.hpp"  // fp8_bfdot (also provides UE8M0 helpers)

namespace armneon {

struct GemmKernelNeonMXFP4 {
  using dt = uint8_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;
  static constexpr int K_GROUP_SIZE = 32;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 6144;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}
  static int recommended_nth(int n) { return std::max(1, (n + N_BLOCK - 1) / N_BLOCK); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range(n, ith, nth); }

  // FP4 E2M1 -> BF16 bytes.  The low/high tables are deliberately kept in
  // byte form so vqtbl1q_u8 can decode four packed bytes without a scalar
  // temporary for every nibble.
  static inline uint8x16_t fp4_bf16_lo_table() {
    alignas(16) static constexpr uint8_t values[16] = {
        0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
        0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
    return vld1q_u8(values);
  }

  static inline uint8x16_t fp4_bf16_hi_table() {
    alignas(16) static constexpr uint8_t values[16] = {
        0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
        0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};
    return vld1q_u8(values);
  }

  // Decode four packed bytes (eight nibbles) in logical low/high nibble
  // order: [lo(byte0), hi(byte0), lo(byte1), hi(byte1), ...].
  static inline uint16x8_t fp4x8_to_bf16(const uint8_t* packed) {
    uint32_t raw = 0;
    std::memcpy(&raw, packed, sizeof(raw));
    const uint8x8_t bytes = vcreate_u8(static_cast<uint64_t>(raw));
    const uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
    const uint8x8_t hi = vand_u8(vshr_n_u8(bytes, 4), vdup_n_u8(0x0F));
    const uint8x8x2_t nibble_zip = vzip_u8(lo, hi);
    const uint8x16_t indices = vcombine_u8(nibble_zip.val[0], nibble_zip.val[1]);
    const uint8x16_t lo_bytes = vqtbl1q_u8(fp4_bf16_lo_table(), indices);
    const uint8x16_t hi_bytes = vqtbl1q_u8(fp4_bf16_hi_table(), indices);
    const uint8x16x2_t bf16_bytes = vzipq_u8(lo_bytes, hi_bytes);
    return vreinterpretq_u16_u8(bf16_bytes.val[0]);
  }

  static inline float fp4_scalar(uint8_t nibble) {
    // E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6.  Bit 3 is sign.
    static constexpr float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return values[nibble & 0x0F];
  }

  using BufferA = GemmKernelNeonBF16::BufferA;
  using BufferC = GemmKernelNeonBF16::BufferC;

  struct BufferB {
    uint8_t* b = nullptr;  // n*k/2 packed FP4 bytes
    float* d = nullptr;    // n*(k/32) expanded scales
    int n = 0;
    int k = 0;
    int k_group_size = K_GROUP_SIZE;
    int k_group_count = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int group_size, void* ptr)
        : n(static_cast<int>(n_)), k(static_cast<int>(k_)), k_group_size(group_size) {
      if (group_size != K_GROUP_SIZE || k % group_size != 0 || (k & 1) != 0) {
        throw std::runtime_error("NEON MXFP4 requires group_size=32, even K, and K divisible by 32");
      }
      k_group_count = k / group_size;
      b = static_cast<uint8_t*>(ptr);
      d = reinterpret_cast<float*>(b + static_cast<size_t>(n) * k / 2);
    }

    static size_t required_size(size_t n, size_t k, int group_size) {
      if (group_size == 0) return 0;
      return n * k / 2 + n * (k / static_cast<size_t>(group_size)) * sizeof(float);
    }

    void from_raw_mat(const uint8_t* weights, int ith, int nth) {
      if (b == nullptr) return;
      const auto [n_start, n_end] = split_range(n, ith, nth);
      const size_t row_bytes = static_cast<size_t>(k) / 2;
      if (n_start < n_end) {
        std::memcpy(b + static_cast<size_t>(n_start) * row_bytes,
                    weights + static_cast<size_t>(n_start) * row_bytes,
                    static_cast<size_t>(n_end - n_start) * row_bytes);
      }
    }

    uint8_t* get_submat(int n_begin, int k_begin) {
      return b + static_cast<size_t>(n_begin) * (static_cast<size_t>(k) / 2) + k_begin / 2;
    }
    float* get_scale(int n_begin, int k_begin) {
      return d + static_cast<size_t>(n_begin) * k_group_count + k_begin / k_group_size;
    }
  };
};

// Dot eight BF16 activations against eight decoded FP4 values.
static inline float32x4_t fp4_bfdot(float32x4_t acc, const ggml_bf16_t* activations, uint16x8_t weights) {
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
  const bfloat16x8_t a = vreinterpretq_bf16_u16(vld1q_u16(reinterpret_cast<const uint16_t*>(activations)));
  return vbfdotq_f32(acc, a, vreinterpretq_bf16_u16(weights));
#else
  const uint32x4_t w0_bits = vshll_n_u16(vget_low_u16(weights), 16);
  const uint32x4_t w1_bits = vshll_n_u16(vget_high_u16(weights), 16);
  const float32x4_t a0 = load4_bf16_to_fp32(activations);
  const float32x4_t a1 = load4_bf16_to_fp32(activations + 4);
  acc = vfmaq_f32(acc, a0, vreinterpretq_f32_u32(w0_bits));
  return vfmaq_f32(acc, a1, vreinterpretq_f32_u32(w1_bits));
#endif
}

static inline void gemm_mxfp4(int m, int n, int k, GemmKernelNeonMXFP4::BufferA& a,
                              GemmKernelNeonMXFP4::BufferB& b, GemmKernelNeonMXFP4::BufferC& c, int ith, int nth) {
  if (b.b == nullptr) throw std::runtime_error("gemm_mxfp4: weight buffer is null");
  if (k != b.k || (k % GemmKernelNeonMXFP4::K_GROUP_SIZE) != 0 || (k & 1) != 0)
    throw std::runtime_error("gemm_mxfp4: invalid K/group layout");

  const auto [n_start, n_end] = split_range(n, ith, nth);
  const int group_count = b.k_group_count;
  constexpr int M_TILE = 4;

  for (int ni = n_start; ni < n_end; ++ni) {
    const uint8_t* weight_row = b.b + static_cast<size_t>(ni) * k / 2;
    const float* scales = b.d + static_cast<size_t>(ni) * group_count;

    for (int m_begin = 0; m_begin < m; m_begin += M_TILE) {
      const int rows = std::min(M_TILE, m - m_begin);
      float32x4_t accum[ M_TILE ] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                     vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};

      for (int group = 0; group < group_count; ++group) {
        const int k_base = group * GemmKernelNeonMXFP4::K_GROUP_SIZE;
        float32x4_t group_acc[ M_TILE ] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                           vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        for (int ki = 0; ki < GemmKernelNeonMXFP4::K_GROUP_SIZE; ki += 8) {
          const uint16x8_t decoded = GemmKernelNeonMXFP4::fp4x8_to_bf16(weight_row + (k_base + ki) / 2);
          for (int row = 0; row < rows; ++row) {
            const ggml_bf16_t* activation = a.data + static_cast<size_t>(m_begin + row) * a.k + k_base + ki;
            group_acc[row] = fp4_bfdot(group_acc[row], activation, decoded);
          }
        }
        const float scale = scales[group];
        for (int row = 0; row < rows; ++row) accum[row] = vfmaq_n_f32(accum[row], group_acc[row], scale);
      }
      for (int row = 0; row < rows; ++row) c.data[static_cast<size_t>(m_begin + row) * n + ni] = vaddvq_f32(accum[row]);
    }
  }
}

static inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernelNeonMXFP4::BufferA>& a,
                           std::shared_ptr<GemmKernelNeonMXFP4::BufferB>& b,
                           std::shared_ptr<GemmKernelNeonMXFP4::BufferC>& c, int ith, int nth) {
  gemm_mxfp4(m, n, k, *a, *b, *c, ith, nth);
}

static inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernelNeonMXFP4::BufferA>& a,
                           std::shared_ptr<GemmKernelNeonMXFP4::BufferB>& b,
                           std::shared_ptr<GemmKernelNeonMXFP4::BufferC>& c, int ith, int nth) {
  gemm_mxfp4(m, n, k, *a, *b, *c, ith, nth);
}

}  // namespace armneon

#endif  // CPUINFER_OPERATOR_ARM_NEON_MXFP4_GEMM_H
