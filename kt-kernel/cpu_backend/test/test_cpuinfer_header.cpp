#ifndef KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS
#define KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS 1
#endif
#define KTRANSFORMERS_CPUINFER_TEST_VENDOR_HEADER "test/fake_cuda_runtime.h"
#ifdef KTRANSFORMERS_CPU_ONLY
#undef KTRANSFORMERS_CPU_ONLY
#endif

#include "../cpuinfer.h"

#include <type_traits>

int main() {
  static_assert(!std::is_copy_constructible_v<CPUInfer>);
  static_assert(!std::is_move_constructible_v<CPUInfer>);
  return 0;
}
