/**
 * ARM NEON packed MXFP4 GEMM.
 *
 * MXFP4 stores two FP4 E2M1 values in each byte and one BF16-compatible
 * (UE8M0) scale for every group of 32 K values.  The packed weights stay in
 * BufferB.
 *
 * Dot product strategy (llama.cpp Q4_K/Q8_K style, integer path):
 *  - FP4 E2M1 values x2 are exact integers {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12},
 *    so weights decode to int8 with two vqtbl1q_s8 LUTs per 16 packed bytes
 *    (one for even/low-nibble values, one for odd/high-nibble), plus two
 *    vzip to restore natural order: ~6 NEON ops per group of 32 instead of
 *    ~15 for a full BF16 expansion.
 *  - Activations are quantized once per (row, 32-group) to int8 with an fp32
 *    absmax scale -- the same activation quantization the llama.cpp Q8_K
 *    kernels use -- then int32 accumulate via vdotq_s32 (FEAT_DotProd) with a
 *    portable vmull/vmlal fallback.  Exact: 32 int8 products per lane-group
 *    cannot overflow int32.
 *  - Per group: acc_f32 += (int32 sum) * (w_scale/2 * a_scale).  One vcvt +
 *    one FMA per row; the int accumulator never crosses group boundaries.
 *
 * Compute loop notes (performance-critical):
 *  - Loop order is m-tile -> column -> K.  Each weight group is decoded once
 *    per column and dotted against all ROWS activation rows (decode cost
 *    amortized), while the quantized activation tile is reused across every
 *    column.  At decode (m=1) the per-column cost is ~11 NEON ops per 16
 *    weight bytes -- close to a plain Q8 kernel but reading half the bytes.
 *  - The next column's head is prefetched while the current one streams.
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
#include <vector>

#include "neon_bf16_gemm.hpp"

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

  // FP4 E2M1 value x2 as exact int8.  Nibble order matches the E2M1 layout:
  // indices 0-7 are positive magnitudes, 8-15 their negatives.
  static inline int8x16_t fp4_int8_x2_table() {
    alignas(16) static constexpr int8_t values[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return vld1q_s8(values);
  }

  // Decode one full K group: 16 packed bytes (32 nibbles) -> 32 int8 values
  // (E2M1 x2) in natural order, returned as two vectors v0 (values 0-15) and
  // v1 (values 16-31).  Byte j holds values 2j (low nibble) and 2j+1 (high
  // nibble), so the low/high nibble vectors decode to the even/odd positions
  // and one byte interleave restores the natural order.  ~6 NEON ops total.
  static inline void fp4x32_to_int8(uint8x16_t packed, int8x16_t& v0, int8x16_t& v1) {
    const uint8x16_t lo_idx = vandq_u8(packed, vdupq_n_u8(0x0F));
    const uint8x16_t hi_idx = vshrq_n_u8(packed, 4);
    const int8x16_t lo = vqtbl1q_s8(fp4_int8_x2_table(), lo_idx);  // even values x2
    const int8x16_t hi = vqtbl1q_s8(fp4_int8_x2_table(), hi_idx);  // odd values x2
    v0 = vzip1q_s8(lo, hi);                                        // values 0..15
    v1 = vzip2q_s8(lo, hi);                                        // values 16..31
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

// Quantize one K group (32 BF16 activations) to int8 with an fp32 absmax
// scale -- the same strategy llama.cpp uses for its Q8 activation rows.
// Returns the de-quant scale (amax/127, 0 when the group is all zeros).
static inline float quantize_act_group(const ggml_bf16_t* ap, int8_t* out) {
  float32x4_t f[8];
  for (int i = 0; i < 4; ++i) {
    const uint16x8_t h = vld1q_u16(reinterpret_cast<const uint16_t*>(ap + i * 8));
    f[2 * i] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
    f[2 * i + 1] = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
  }
  float32x4_t am = vabsq_f32(f[0]);
  for (int i = 1; i < 8; ++i) am = vmaxq_f32(am, vabsq_f32(f[i]));
  const float amax = vmaxvq_f32(am);
  if (amax == 0.0f) {
    std::memset(out, 0, 32);
    return 0.0f;
  }
  const float inv = 127.0f / amax;
  int16x8_t s16[4];
  for (int i = 0; i < 4; ++i) {
    const int32x4_t c0 = vcvtnq_s32_f32(vmulq_n_f32(f[2 * i], inv));
    const int32x4_t c1 = vcvtnq_s32_f32(vmulq_n_f32(f[2 * i + 1], inv));
    s16[i] = vcombine_s16(vqmovn_s32(c0), vqmovn_s32(c1));
  }
  vst1q_s8(out, vcombine_s8(vqmovn_s16(s16[0]), vqmovn_s16(s16[1])));
  vst1q_s8(out + 16, vcombine_s8(vqmovn_s16(s16[2]), vqmovn_s16(s16[3])));
  return amax * (1.0f / 127.0f);
}

// Dot 32 int8 activations against 32 decoded int8 weights (values x2).
// Exact in int32: per output lane at most 8 products of |127*127| fit.
static inline int32x4_t q8x32_dot(int32x4_t acc, const int8_t* a, int8x16_t w0, int8x16_t w1) {
  const int8x16_t a0 = vld1q_s8(a);
  const int8x16_t a1 = vld1q_s8(a + 16);
#if defined(__ARM_FEATURE_DOTPROD)
  acc = vdotq_s32(acc, a0, w0);
  return vdotq_s32(acc, a1, w1);
#else
  // Portable widening MACs; split into pairs of 2 products so int16 lanes
  // cannot overflow before vpadalq widens to int32.
  int16x8_t p0 = vmull_s8(vget_low_s8(a0), vget_low_s8(w0));
  p0 = vmlal_s8(p0, vget_high_s8(a0), vget_high_s8(w0));
  int16x8_t p1 = vmull_s8(vget_low_s8(a1), vget_low_s8(w1));
  p1 = vmlal_s8(p1, vget_high_s8(a1), vget_high_s8(w1));
  acc = vpadalq_s16(acc, p0);
  return vpadalq_s16(acc, p1);
#endif
}

// Compute one m-tile (ROWS rows) over columns [n_start, n_end).  The
// activation tile is quantized to int8 once and reused for every column; the
// weight matrix is visited exactly once per tile and each K group is decoded
// a single time and dotted against all ROWS rows.
template <int ROWS>
static inline void gemm_mxfp4_tile(int m_begin, int n_start, int n_end, int n, int k,
                                   const GemmKernelNeonMXFP4::BufferA& a,
                                   const GemmKernelNeonMXFP4::BufferB& b,
                                   GemmKernelNeonMXFP4::BufferC& c) {
  const int group_count = b.k_group_count;
  const size_t weight_row_bytes = static_cast<size_t>(k) / 2;

  // Quantize the whole activation tile up front: cost O(ROWS*K), amortized
  // across all n_end-n_start columns.
  std::vector<int8_t> aq(static_cast<size_t>(ROWS) * group_count * GemmKernelNeonMXFP4::K_GROUP_SIZE);
  std::vector<float> as(static_cast<size_t>(ROWS) * group_count);
  for (int r = 0; r < ROWS; ++r) {
    const ggml_bf16_t* arow = a.data + static_cast<size_t>(m_begin + r) * a.k;
    for (int g = 0; g < group_count; ++g)
      as[static_cast<size_t>(r) * group_count + g] =
          quantize_act_group(arow + static_cast<size_t>(g) * GemmKernelNeonMXFP4::K_GROUP_SIZE,
                             aq.data() + (static_cast<size_t>(r) * group_count + g) *
                                             GemmKernelNeonMXFP4::K_GROUP_SIZE);
  }

  for (int ni = n_start; ni < n_end; ++ni) {
    const uint8_t* weight_row = b.b + static_cast<size_t>(ni) * weight_row_bytes;
    const float* scales = b.d + static_cast<size_t>(ni) * group_count;

#if defined(__GNUC__)
    // Warm up the head of the next column while this one streams.
    if (ni + 1 < n_end) {
      const uint8_t* next_row = weight_row + weight_row_bytes;
      __builtin_prefetch(next_row, 0, 1);
      __builtin_prefetch(next_row + 64, 0, 1);
      __builtin_prefetch(next_row + 128, 0, 1);
      __builtin_prefetch(next_row + 192, 0, 1);
    }
#endif

    float32x4_t accum[ROWS];
    for (int r = 0; r < ROWS; ++r) accum[r] = vdupq_n_f32(0.0f);

    for (int group = 0; group < group_count; ++group) {
#if defined(__GNUC__)
      if ((group & 3) == 0) __builtin_prefetch(weight_row + static_cast<size_t>(group) * 16 + 512, 0, 1);
#endif
      int8x16_t w0, w1;
      GemmKernelNeonMXFP4::fp4x32_to_int8(vld1q_u8(weight_row + static_cast<size_t>(group) * 16), w0, w1);

      // Weight = int8 value * (scale/2); activation = int8 * a_scale.
      const float ws = scales[group] * 0.5f;

      int32x4_t qacc[ROWS];
      for (int r = 0; r < ROWS; ++r)
        qacc[r] = q8x32_dot(vdupq_n_s32(0),
                            aq.data() + (static_cast<size_t>(r) * group_count + group) *
                                            GemmKernelNeonMXFP4::K_GROUP_SIZE,
                            w0, w1);
      for (int r = 0; r < ROWS; ++r)
        accum[r] = vfmaq_n_f32(accum[r], vcvtq_f32_s32(qacc[r]), ws * as[static_cast<size_t>(r) * group_count + group]);
    }
    for (int r = 0; r < ROWS; ++r)
      c.data[static_cast<size_t>(m_begin + r) * n + ni] = vaddvq_f32(accum[r]);
  }
}

static inline void gemm_mxfp4(int m, int n, int k, GemmKernelNeonMXFP4::BufferA& a,
                              GemmKernelNeonMXFP4::BufferB& b, GemmKernelNeonMXFP4::BufferC& c, int ith, int nth) {
  if (b.b == nullptr) throw std::runtime_error("gemm_mxfp4: weight buffer is null");
  if (k != b.k || (k % GemmKernelNeonMXFP4::K_GROUP_SIZE) != 0 || (k & 1) != 0)
    throw std::runtime_error("gemm_mxfp4: invalid K/group layout");

  const auto [n_start, n_end] = split_range(n, ith, nth);
  constexpr int M_TILE = 4;

  for (int m_begin = 0; m_begin < m; m_begin += M_TILE) {
    const int rows = std::min(M_TILE, m - m_begin);
    switch (rows) {
      case 4: gemm_mxfp4_tile<4>(m_begin, n_start, n_end, n, k, a, b, c); break;
      case 3: gemm_mxfp4_tile<3>(m_begin, n_start, n_end, n, k, a, b, c); break;
      case 2: gemm_mxfp4_tile<2>(m_begin, n_start, n_end, n, k, a, b, c); break;
      default: gemm_mxfp4_tile<1>(m_begin, n_start, n_end, n, k, a, b, c); break;
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
