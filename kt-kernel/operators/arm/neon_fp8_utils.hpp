/**
 * ARM NEON helpers for FP8 E4M3FN weights.
 *
 * E4M3FN values are expanded in registers. Normal values are constructed with
 * integer bit operations; the eight subnormal magnitudes use a small NEON
 * table. This avoids materializing a BF16 copy of the weight matrix.
 */
#ifndef CPUINFER_OPERATOR_ARM_NEON_FP8_UTILS_H
#define CPUINFER_OPERATOR_ARM_NEON_FP8_UTILS_H

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "neon_bf16_utils.hpp"

namespace armneon {

struct fp8_bf16x16 {
  uint16x8_t lo;
  uint16x8_t hi;
};

struct fp8_e4m3fn_lut {
  uint8x16_t subnormal_lo;
  uint8x16_t subnormal_hi;
};

static inline uint8x16_t fp8_e4m3fn_subnormal_lo_table() {
  alignas(16) static constexpr uint8_t values[16] = {
      0x00, 0x00, 0x80, 0xc0, 0x00, 0x20, 0x40, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  return vld1q_u8(values);
}

static inline uint8x16_t fp8_e4m3fn_subnormal_hi_table() {
  alignas(16) static constexpr uint8_t values[16] = {
      0x00, 0x3b, 0x3b, 0x3b, 0x3c, 0x3c, 0x3c, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  return vld1q_u8(values);
}

static inline uint16x8_t fp8_e4m3fn_decode_half(uint8x8_t raw, uint8x8_t sub_lo, uint8x8_t sub_hi) {
  const uint8x8_t magnitude = vand_u8(raw, vdup_n_u8(0x7f));
  const uint8x8_t exponent = vand_u8(magnitude, vdup_n_u8(0x78));
  const uint8x8_t sign = vand_u8(raw, vdup_n_u8(0x80));

  uint16x8_t normal = vaddq_u16(vshlq_n_u16(vmovl_u8(magnitude), 4), vdupq_n_u16(0x3c00));
  normal = vorrq_u16(normal, vshlq_n_u16(vmovl_u8(sign), 8));

  const uint8x8x2_t sub_bytes = vzip_u8(sub_lo, sub_hi);
  uint16x8_t subnormal = vreinterpretq_u16_u8(vcombine_u8(sub_bytes.val[0], sub_bytes.val[1]));
  subnormal = vorrq_u16(subnormal, vshlq_n_u16(vmovl_u8(sign), 8));

  const uint16x8_t exponent_is_zero = vceqq_u16(vmovl_u8(exponent), vdupq_n_u16(0));
  uint16x8_t decoded = vbslq_u16(exponent_is_zero, subnormal, normal);

  // E4M3FN reserves magnitude 0x7f for NaN. Match the existing CPU FP8
  // kernels by mapping it to signed zero instead of propagating NaNs.
  const uint16x8_t is_nan = vceqq_u16(vmovl_u8(magnitude), vdupq_n_u16(0x7f));
  const uint16x8_t signed_zero = vshlq_n_u16(vmovl_u8(sign), 8);
  return vbslq_u16(is_nan, signed_zero, decoded);
}

static inline fp8_e4m3fn_lut load_fp8_e4m3fn_lut() {
  return {fp8_e4m3fn_subnormal_lo_table(), fp8_e4m3fn_subnormal_hi_table()};
}

static inline fp8_bf16x16 fp8_e4m3fn_to_bf16x16(const uint8_t* src, const fp8_e4m3fn_lut& lut) {
  const uint8x16_t raw = vld1q_u8(src);
  const uint8x16_t indices = vandq_u8(raw, vdupq_n_u8(0x07));
  const uint8x16_t sub_lo = vqtbl1q_u8(lut.subnormal_lo, indices);
  const uint8x16_t sub_hi = vqtbl1q_u8(lut.subnormal_hi, indices);
  return {
      fp8_e4m3fn_decode_half(vget_low_u8(raw), vget_low_u8(sub_lo), vget_low_u8(sub_hi)),
      fp8_e4m3fn_decode_half(vget_high_u8(raw), vget_high_u8(sub_lo), vget_high_u8(sub_hi)),
  };
}

static inline fp8_bf16x16 fp8_e4m3fn_to_bf16x16(const uint8_t* src) {
  const fp8_e4m3fn_lut lut = load_fp8_e4m3fn_lut();
  return fp8_e4m3fn_to_bf16x16(src, lut);
}

static inline float fp8_e4m3fn_to_fp32(uint8_t value) {
  const uint8_t sign = value & 0x80;
  const uint8_t magnitude = value & 0x7f;
  const uint8_t exponent = (magnitude >> 3) & 0x0f;
  const uint8_t mantissa = magnitude & 0x07;

  uint16_t bits;
  if (exponent == 0) {
    static constexpr uint16_t subnormal[8] = {
        0x0000, 0x3b00, 0x3b80, 0x3bc0, 0x3c00, 0x3c20, 0x3c40, 0x3c60,
    };
    bits = subnormal[mantissa];
  } else if (magnitude == 0x7f) {
    bits = 0;
  } else {
    bits = static_cast<uint16_t>((static_cast<uint16_t>(magnitude) << 4) + 0x3c00);
  }
  bits |= static_cast<uint16_t>(sign) << 8;

  uint32_t fp32_bits = static_cast<uint32_t>(bits) << 16;
  float result;
  std::memcpy(&result, &fp32_bits, sizeof(result));
  return result;
}

static inline float ue8m0_to_fp32(uint8_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 23;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

static inline void convert_ue8m0_to_fp32(float* dst, const uint8_t* src, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const uint8x8_t bytes = vld1_u8(src + i);
    const uint16x8_t words = vmovl_u8(bytes);
    const uint32x4_t lo = vshlq_n_u32(vmovl_u16(vget_low_u16(words)), 23);
    const uint32x4_t hi = vshlq_n_u32(vmovl_u16(vget_high_u16(words)), 23);
    vst1q_f32(dst + i, vreinterpretq_f32_u32(lo));
    vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
  }
  for (; i < count; ++i) dst[i] = ue8m0_to_fp32(src[i]);
}

static inline void fp32_to_ue8m0(uint8_t* dst, const float* src, size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const uint32x4_t lo = vshrq_n_u32(vreinterpretq_u32_f32(vld1q_f32(src + i)), 23);
    const uint32x4_t hi = vshrq_n_u32(vreinterpretq_u32_f32(vld1q_f32(src + i + 4)), 23);
    const uint16x8_t words = vcombine_u16(vmovn_u32(lo), vmovn_u32(hi));
    vst1_u8(dst + i, vmovn_u16(words));
  }
  for (; i < count; ++i) {
    uint32_t bits;
    std::memcpy(&bits, src + i, sizeof(bits));
    dst[i] = static_cast<uint8_t>((bits >> 23) & 0xff);
  }
}

static inline float32x4_t fp8_bfdot(float32x4_t acc, const ggml_bf16_t* activations, uint16x8_t weights) {
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

}  // namespace armneon

#endif  // CPUINFER_OPERATOR_ARM_NEON_FP8_UTILS_H
