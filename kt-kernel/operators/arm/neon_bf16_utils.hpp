/**
 * @Description  : ARM NEON BF16 utility functions (bf16<->fp32 conversion, activation)
 * @Author       : Claude
 * @Date         : 2026-08-09
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * NEON ports of the AVX2 utilities in avx2/avx2_bf16_utils.hpp.
 * AArch64 NEON registers are 128-bit (4 floats), so an 8-lane FP32 vector
 * (__m256 counterpart) is emulated with two float32x4_t registers.
 *
 * When the Armv8.6-A BF16 vector extension is available
 * (__ARM_FEATURE_BF16_VECTOR_ARITHMETIC), the
 * GEMM path uses native BFDOT instructions; the conversion helpers below stay
 * feature-gate free so the same header also works on plain AArch64 (NEON+FMA).
 **/
#ifndef CPUINFER_OPERATOR_ARM_NEON_BF16_UTILS_H
#define CPUINFER_OPERATOR_ARM_NEON_BF16_UTILS_H

#include <arm_neon.h>

#include <cmath>

#include "llama.cpp/ggml.h"

namespace armneon {

// ============================================================================
// v8f32: 8-lane FP32 vector (__m256 counterpart), two NEON registers
// ============================================================================

struct v8f32 {
  float32x4_t v[2];
};

static inline v8f32 zero_v8f32() { return {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)}; }

static inline v8f32 set1_v8f32(float x) { return {vdupq_n_f32(x), vdupq_n_f32(x)}; }

static inline v8f32 load_v8f32(const float* src) { return {vld1q_f32(src), vld1q_f32(src + 4)}; }

static inline void store_v8f32(float* dst, v8f32 x) {
  vst1q_f32(dst, x.v[0]);
  vst1q_f32(dst + 4, x.v[1]);
}

static inline v8f32 add_v8f32(v8f32 a, v8f32 b) { return {vaddq_f32(a.v[0], b.v[0]), vaddq_f32(a.v[1], b.v[1])}; }

static inline v8f32 sub_v8f32(v8f32 a, v8f32 b) { return {vsubq_f32(a.v[0], b.v[0]), vsubq_f32(a.v[1], b.v[1])}; }

static inline v8f32 mul_v8f32(v8f32 a, v8f32 b) { return {vmulq_f32(a.v[0], b.v[0]), vmulq_f32(a.v[1], b.v[1])}; }

static inline v8f32 div_v8f32(v8f32 a, v8f32 b) { return {vdivq_f32(a.v[0], b.v[0]), vdivq_f32(a.v[1], b.v[1])}; }

static inline v8f32 min_v8f32(v8f32 a, v8f32 b) { return {vminq_f32(a.v[0], b.v[0]), vminq_f32(a.v[1], b.v[1])}; }

static inline v8f32 max_v8f32(v8f32 a, v8f32 b) { return {vmaxq_f32(a.v[0], b.v[0]), vmaxq_f32(a.v[1], b.v[1])}; }

// c = a * b + c
static inline v8f32 fmadd_v8f32(v8f32 a, v8f32 b, v8f32 c) {
  return {vfmaq_f32(c.v[0], a.v[0], b.v[0]), vfmaq_f32(c.v[1], a.v[1], b.v[1])};
}

// ============================================================================
// BF16 <-> FP32 conversion
// ============================================================================

// Load 4 BF16 values and convert to 4 FP32 values.
// BF16 is the upper 16 bits of FP32, so shift left by 16.
static inline float32x4_t load4_bf16_to_fp32(const ggml_bf16_t* src) {
  uint16x4_t bf16 = vld1_u16(reinterpret_cast<const uint16_t*>(src));
  return vreinterpretq_f32_u32(vshll_n_u16(bf16, 16));
}

// Load 8 BF16 values and convert to 8 FP32 values.
static inline v8f32 load_bf16_to_fp32(const ggml_bf16_t* src) {
  return {load4_bf16_to_fp32(src), load4_bf16_to_fp32(src + 4)};
}

// Convert 4 FP32 values to 4 BF16 values with round-to-nearest-even (RNE).
// Matches ggml_compute_fp32_to_bf16 semantics (ggml-impl.h): add
// 0x7FFF + ((val >> 16) & 1) to the FP32 bit pattern, then keep the top half.
static inline uint16x4_t pack4_fp32_to_bf16_rne(float32x4_t v) {
  uint32x4_t i32 = vreinterpretq_u32_f32(v);
  uint32x4_t tie_bit = vandq_u32(vshrq_n_u32(i32, 16), vdupq_n_u32(1));
  uint32x4_t rounded = vaddq_u32(i32, vaddq_u32(vdupq_n_u32(0x7FFF), tie_bit));
  return vshrn_n_u32(rounded, 16);
}

// Convert 8 FP32 values to 8 BF16 values with round-to-nearest-even.
static inline void store_fp32_to_bf16(ggml_bf16_t* dst, v8f32 src) {
  uint16x8_t packed = vcombine_u16(pack4_fp32_to_bf16_rne(src.v[0]), pack4_fp32_to_bf16_rne(src.v[1]));
  vst1q_u16(reinterpret_cast<uint16_t*>(dst), packed);
}

// Load 16 BF16 -> 2x8 FP32 (corresponds to avx2/avx512 pair load helpers)
static inline void load_16xbf16_to_2x8xfp32(const ggml_bf16_t* src, v8f32* out0, v8f32* out1) {
  *out0 = load_bf16_to_fp32(src);
  *out1 = load_bf16_to_fp32(src + 8);
}

// Store 2x8 FP32 -> 16 BF16
static inline void store_2x8xfp32_to_16xbf16(v8f32* in0, v8f32* in1, ggml_bf16_t* dst) {
  store_fp32_to_bf16(dst, *in0);
  store_fp32_to_bf16(dst + 8, *in1);
}

// ============================================================================
// Horizontal sum for v8f32 (8 floats -> 1 float)
// ============================================================================

static inline float hsum_neon(v8f32 x) {
  return vaddvq_f32(vaddq_f32(x.v[0], x.v[1]));
}

// ============================================================================
// Fast exp approximation (NEON port of avx2::exp_avx2 / amx::exp_avx512)
// ============================================================================

static inline v8f32 exp_neon(v8f32 x) {
  const v8f32 log2e = set1_v8f32(1.44269504089f);

  v8f32 y = mul_v8f32(x, log2e);

  // int_part = round_to_nearest_even(y), matching _mm256_cvtps_epi32
  int32x4_t i0 = vcvtnq_s32_f32(y.v[0]);
  int32x4_t i1 = vcvtnq_s32_f32(y.v[1]);
  v8f32 frac_part = {vsubq_f32(y.v[0], vcvtq_f32_s32(i0)), vsubq_f32(y.v[1], vcvtq_f32_s32(i1))};

  // 2^frac polynomial (same coefficients as the AVX2/AMX versions)
  const float P1 = 0.9999999995f;
  const float P2 = 0.6931471805f;
  const float P3 = 0.2402265069f;
  const float P4 = 0.0555041087f;
  const float P5 = 0.0096181291f;
  const float P6 = 0.0013333558f;

  v8f32 frac_exp = {vdupq_n_f32(P6), vdupq_n_f32(P6)};
  frac_exp = fmadd_v8f32(frac_exp, frac_part, set1_v8f32(P5));
  frac_exp = fmadd_v8f32(frac_exp, frac_part, set1_v8f32(P4));
  frac_exp = fmadd_v8f32(frac_exp, frac_part, set1_v8f32(P3));
  frac_exp = fmadd_v8f32(frac_exp, frac_part, set1_v8f32(P2));
  frac_exp = fmadd_v8f32(frac_exp, frac_part, set1_v8f32(P1));

  // 2^int_part: clamp to [-126, 127] then reinterpret ((n + 127) << 23)
  i0 = vmaxq_s32(vminq_s32(i0, vdupq_n_s32(127)), vdupq_n_s32(-126));
  i1 = vmaxq_s32(vminq_s32(i1, vdupq_n_s32(127)), vdupq_n_s32(-126));
  i0 = vshlq_n_s32(vaddq_s32(i0, vdupq_n_s32(127)), 23);
  i1 = vshlq_n_s32(vaddq_s32(i1, vdupq_n_s32(127)), 23);
  v8f32 two_pow_i = {vreinterpretq_f32_s32(i0), vreinterpretq_f32_s32(i1)};

  return mul_v8f32(two_pow_i, frac_exp);
}

// ============================================================================
// SiLU activation: silu(gate) * up = gate * sigmoid(gate) * up
// NEON port of avx2::act_fn
// ============================================================================

static inline v8f32 act_fn(v8f32 gate_val, v8f32 up_val) {
  v8f32 neg_gate_val = sub_v8f32(zero_v8f32(), gate_val);
  // Clamp to avoid exp overflow
  neg_gate_val = min_v8f32(neg_gate_val, set1_v8f32(88.0f));
  v8f32 exp_neg_gate = exp_neon(neg_gate_val);
  v8f32 denom = add_v8f32(set1_v8f32(1.0f), exp_neg_gate);
  v8f32 act_val = div_v8f32(gate_val, denom);

  return mul_v8f32(act_val, up_val);
}

// Overload with swiglu_limit: asymmetric clamp (no alpha).
//   gate = min(gate, limit)            (one-sided pre-silu)
//   up   = clamp(up, -limit, limit)    (symmetric)
// Mirrors avx2::act_fn(g, u, swiglu_limit).
static inline v8f32 act_fn(v8f32 gate_val, v8f32 up_val, float swiglu_limit) {
  if (swiglu_limit > 0.0f) {
    v8f32 pos_lim = set1_v8f32(swiglu_limit);
    v8f32 neg_lim = set1_v8f32(-swiglu_limit);
    gate_val = min_v8f32(gate_val, pos_lim);
    up_val = min_v8f32(up_val, pos_lim);
    up_val = max_v8f32(up_val, neg_lim);
  }
  return act_fn(gate_val, up_val);
}

// "swigluoai" (alpha > 0) / "silu" (alpha == 0) unified entry point.
//   alpha > 0  -> gate * sigmoid(gate * alpha) * (up + 1), symmetric clamp on both
//   alpha == 0 -> falls back to silu (with optional one-sided clamp)
// Mirrors avx2::act_fn(g, u, swiglu_limit, swiglu_alpha).
static inline v8f32 act_fn(v8f32 gate_val, v8f32 up_val, float swiglu_limit, float swiglu_alpha) {
  if (swiglu_alpha > 0.0f) {
    if (swiglu_limit > 0.0f) {
      v8f32 pos_lim = set1_v8f32(swiglu_limit);
      v8f32 neg_lim = set1_v8f32(-swiglu_limit);
      gate_val = min_v8f32(gate_val, pos_lim);
      gate_val = max_v8f32(gate_val, neg_lim);
      up_val = min_v8f32(up_val, pos_lim);
      up_val = max_v8f32(up_val, neg_lim);
    }
    // sigmoid(gate * alpha)
    v8f32 neg_ga = mul_v8f32(gate_val, set1_v8f32(-swiglu_alpha));
    neg_ga = min_v8f32(neg_ga, set1_v8f32(88.0f));
    v8f32 exp_neg = exp_neon(neg_ga);
    v8f32 sigmoid_val = div_v8f32(set1_v8f32(1.0f), add_v8f32(set1_v8f32(1.0f), exp_neg));
    v8f32 up_plus_1 = add_v8f32(up_val, set1_v8f32(1.0f));
    return mul_v8f32(mul_v8f32(gate_val, sigmoid_val), up_plus_1);
  }
  return act_fn(gate_val, up_val, swiglu_limit);
}

}  // namespace armneon

#endif  // CPUINFER_OPERATOR_ARM_NEON_BF16_UTILS_H
