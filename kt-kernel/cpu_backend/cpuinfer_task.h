#ifndef CPUINFER_CPUINFER_TASK_H
#define CPUINFER_CPUINFER_TASK_H

#include <cstdint>
#include <utility>

using CPUInferTask = std::pair<intptr_t, intptr_t>;
using CPUInferTaskFunction = void (*)(void*);
using CPUInferTaskDestroy = void (*)(void*);

struct CPUInferTaskPayload {
  CPUInferTaskFunction function;
  void* args;
  CPUInferTaskDestroy destroy;
};

inline bool resolve_cpuinfer_capture_state(bool capture_hint, bool query_succeeded, bool query_capturing) noexcept {
  // A successful CUDA runtime query is authoritative. If the query itself
  // fails, ignore an eager hint and retain callback storage (fail closed).
  (void)capture_hint;
  return query_succeeded ? query_capturing : true;
}

inline void run_cpuinfer_task_payload(void* opaque) {
  auto* payload = static_cast<CPUInferTaskPayload*>(opaque);
  payload->function(payload->args);
}

inline CPUInferTask make_cpuinfer_owned_task(CPUInferTaskFunction function, void* args, CPUInferTaskDestroy destroy) {
  CPUInferTaskPayload* payload = nullptr;
  try {
    payload = new CPUInferTaskPayload{function, args, destroy};
  } catch (...) {
    if (destroy != nullptr && args != nullptr) destroy(args);
    throw;
  }
  return CPUInferTask{(intptr_t)&run_cpuinfer_task_payload, (intptr_t)payload};
}

inline CPUInferTaskFunction cpuinfer_task_function(const CPUInferTask& task) {
  return reinterpret_cast<CPUInferTaskFunction>(task.first);
}

inline bool cpuinfer_task_is_owned(const CPUInferTask& task) noexcept {
  return task.first == (intptr_t)&run_cpuinfer_task_payload;
}

inline void* cpuinfer_task_opaque(const CPUInferTask& task) {
  return reinterpret_cast<void*>(task.second);
}

inline CPUInferTaskPayload* cpuinfer_task_payload(const CPUInferTask& task) {
  return cpuinfer_task_is_owned(task)
      ? static_cast<CPUInferTaskPayload*>(cpuinfer_task_opaque(task))
      : nullptr;
}

inline void* cpuinfer_task_args(const CPUInferTask& task) {
  CPUInferTaskPayload* payload = cpuinfer_task_payload(task);
  return payload != nullptr ? payload->args : cpuinfer_task_opaque(task);
}

inline void destroy_cpuinfer_task(const CPUInferTask& task) noexcept {
  CPUInferTaskPayload* payload = cpuinfer_task_payload(task);
  if (payload == nullptr) return;
  if (payload->destroy != nullptr && payload->args != nullptr) {
    payload->destroy(payload->args);
  }
  delete payload;
}

class CPUInferTaskCancelGuard {
 public:
  explicit CPUInferTaskCancelGuard(const CPUInferTask& task) : task_(task) {}
  ~CPUInferTaskCancelGuard() { cancel(); }

  CPUInferTaskCancelGuard(const CPUInferTaskCancelGuard&) = delete;
  CPUInferTaskCancelGuard& operator=(const CPUInferTaskCancelGuard&) = delete;

  void release() noexcept { active_ = false; }

  void cancel() noexcept {
    if (!active_) return;
    active_ = false;
    destroy_cpuinfer_task(task_);
  }

 private:
  CPUInferTask task_;
  bool active_ = true;
};

#endif
