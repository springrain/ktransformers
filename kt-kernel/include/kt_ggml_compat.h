#pragma once

// Compatibility surface for the GGML type-traits API used by KT kernels.
//
// The current llama.cpp API keeps layout/dequantization traits in ggml.h and
// CPU dot-product traits in ggml-cpu.h.  KT needs both sets of callbacks while
// loading and quantizing GGUF tensors, so combine them without modifying the
// llama.cpp submodule.

#include <ggml-cpu.h>
#include <ggml-cpu/quants.h>
#include <ggml-common.h>
#include <ggml-impl.h>
#include <ggml-quants.h>
#include <ggml.h>

struct kt_ggml_type_traits {
  ggml_to_float_t to_float;
  ggml_from_float_t from_float;
  ggml_vec_dot_t vec_dot;
  enum ggml_type vec_dot_type;
};

static inline kt_ggml_type_traits ggml_internal_get_type_traits(enum ggml_type type) {
  const struct ggml_type_traits* base_traits = ggml_get_type_traits(type);
  const struct ggml_type_traits_cpu* cpu_traits = ggml_get_type_traits_cpu(type);

  return {
      base_traits != nullptr ? base_traits->to_float : nullptr,
      cpu_traits != nullptr ? cpu_traits->from_float : nullptr,
      cpu_traits != nullptr ? cpu_traits->vec_dot : nullptr,
      cpu_traits != nullptr ? cpu_traits->vec_dot_type : GGML_TYPE_COUNT,
  };
}
