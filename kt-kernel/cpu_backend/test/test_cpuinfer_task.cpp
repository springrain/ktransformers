#include "../cpuinfer_task.h"

#include <cassert>

namespace {

struct Args {
  int* destroy_count;
  int* run_count;
};

void run(void* opaque) {
  auto* args = static_cast<Args*>(opaque);
  ++*args->run_count;
}

void destroy(void* opaque) noexcept {
  auto* args = static_cast<Args*>(opaque);
  ++*args->destroy_count;
  delete args;
}

CPUInferTask make_task(int* destroy_count, int* run_count) {
  auto* args = new Args{destroy_count, run_count};
  return make_cpuinfer_owned_task(&run, args, &destroy);
}

}  // namespace

int main() {
  static_assert(std::tuple_size_v<CPUInferTask> == 2);
  int destroy_count = 0;
  int run_count = 0;

  assert(!resolve_cpuinfer_capture_state(true, true, false));
  assert(resolve_cpuinfer_capture_state(false, true, true));
  assert(resolve_cpuinfer_capture_state(false, false, false));

  {
    CPUInferTaskCancelGuard guard(make_task(&destroy_count, &run_count));
  }
  assert(destroy_count == 1);

  CPUInferTask released_task = make_task(&destroy_count, &run_count);
  assert(cpuinfer_task_is_owned(released_task));
  {
    CPUInferTaskCancelGuard guard(released_task);
    guard.release();
  }
  assert(destroy_count == 1);
  cpuinfer_task_function(released_task)(cpuinfer_task_opaque(released_task));
  assert(run_count == 1);
  destroy_cpuinfer_task(released_task);
  assert(destroy_count == 2);

  {
    CPUInferTaskCancelGuard guard(make_task(&destroy_count, &run_count));
    guard.cancel();
    guard.cancel();
  }
  assert(destroy_count == 3);

  Args legacy_args{&destroy_count, &run_count};
  CPUInferTask legacy_task{(intptr_t)&run, (intptr_t)&legacy_args};
  assert(!cpuinfer_task_is_owned(legacy_task));
  assert(cpuinfer_task_args(legacy_task) == &legacy_args);
  cpuinfer_task_function(legacy_task)(cpuinfer_task_opaque(legacy_task));
  destroy_cpuinfer_task(legacy_task);
  assert(run_count == 2);
  assert(destroy_count == 3);

  return 0;
}
