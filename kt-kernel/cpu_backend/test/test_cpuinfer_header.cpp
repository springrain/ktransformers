#ifndef KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS
#define KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS 1
#endif
#define KTRANSFORMERS_CPUINFER_TEST_VENDOR_HEADER "test/fake_cuda_runtime.h"
#ifdef KTRANSFORMERS_CPU_ONLY
#undef KTRANSFORMERS_CPU_ONLY
#endif

#include "../cpuinfer.h"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

int main() {
  static_assert(!std::is_copy_constructible_v<CPUInfer>);
  static_assert(!std::is_move_constructible_v<CPUInfer>);
  static_assert(std::is_same_v<
                decltype(std::declval<CPUInfer&>().enqueue(
                    std::declval<std::function<void()>>())),
                void>);
  static_assert(std::is_same_v<
                decltype(std::declval<CPUInfer&>().enqueue_tracked(
                    std::declval<std::function<void()>>())),
                std::shared_ptr<TaskCompletion>>);
  static_assert(std::is_same_v<
                decltype(std::declval<CPUInfer&>().enqueue_tracked(
                    std::declval<const std::shared_ptr<TaskCompletion>&>(),
                    std::declval<std::function<void()>>())),
                void>);
  return 0;
}
