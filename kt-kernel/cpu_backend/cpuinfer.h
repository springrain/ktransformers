/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#elif KTRANSFORMERS_USE_MACA
#include "vendors/maca.h"
#elif KTRANSFORMERS_USE_ASCEND_NPU
#include "vendors/ascend_npu.h"
#endif

#include "./vendors/vendor.h"
#include "ggml-cpu.h"
#include "task_queue.h"
#include "worker_pool.h"

// Decode (kind << 32) | expert_id into a short human-readable prefix for
// task failure messages.
inline std::string describe_task_tag(int64_t task_tag, const char* what) {
  uint32_t kind = (uint32_t)((uint64_t)task_tag >> 32);
  uint32_t expert_id = (uint32_t)((uint64_t)task_tag & 0xffffffffu);
  char buf[96];
  std::snprintf(buf, sizeof(buf), "[kt task kind=%u expert=%u] ", kind, expert_id);
  return std::string(buf) + what;
}

class CPUInfer {
 public:
  CPUInfer(int thread_num) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
    ggml_cpu_init();
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
    ggml_cpu_init();
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    ggml_cpu_init();
  }

  // watchdog_timeout_ms <= 0 keeps the exact legacy 1-arg behavior: no
  // monitor thread is spawned, so default builds pay zero watchdog cost.
  CPUInfer(WorkerPoolConfig config, int watchdog_timeout_ms) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    ggml_cpu_init();
    start_watchdog_(watchdog_timeout_ms);
  }

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    // Stop the watchdog before task_queue_ below is freed (it dereferences it).
    watchdog_stop_.store(true, std::memory_order_release);
    if (watchdog_thread_.joinable()) watchdog_thread_.join();
    delete task_queue_;  // joins the worker first; queued tasks dereference backend_
    delete backend_;
  }

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  // Tagged twin of enqueue: a nonzero task_tag prefixes task failure
  // messages with (kind, expert); tag 0 keeps the exact legacy text and
  // exception type.
  template <typename Func, typename Obj, typename... Args>
  void enqueue_tagged(int64_t task_tag, Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() {
      if (task_tag == 0) {
        std::invoke(f, *obj, args...);
        return;
      }
      try {
        std::invoke(f, *obj, args...);
      } catch (const std::exception& e) {
        throw std::runtime_error(describe_task_tag(task_tag, e.what()));
      }
    }, task_tag);  // the tag also feeds the watchdog heartbeat display
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
#endif
  }
#endif

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
    // True only when the args cannot be graph-recorded: sync_ then frees them
    // after syncing. Graph-recorded args are re-invoked by every replay.
    bool autofree;
  };

  static void sync_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    CPUInfer* cpuinfer = args->cpuinfer;
    size_t allow_n_pending = args->allow_n_pending;
    bool autofree = args->autofree;
    cpuinfer->task_queue_->sync(allow_n_pending);
    if (autofree) delete args;
  }

  void sync(size_t allow_n_pending = 0) {
    SyncArgs args{this, allow_n_pending, false};
    sync_(&args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    // Single-use unless the stream is capturing: capture records the args pointer
    // into a host node that replays it forever, so only eager launches self-free.
    bool autofree = false;
#if defined(KTRANSFORMERS_USE_CUDA)
    cudaStreamCaptureStatus capture_status{};
    cudaError_t err = cudaStreamGetCaptureInfo((cudaStream_t)user_cuda_stream, &capture_status, nullptr);
    if (err == cudaSuccess) {
      autofree = capture_status == cudaStreamCaptureStatusNone;
    } else {
      (void)cudaGetLastError();  // keep a failed query out of later error checks
    }
#endif
    SyncArgs* args = new SyncArgs{this, allow_n_pending, autofree};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
#endif
  }
#endif

  // Probed from Python between decode steps; both are cheap because the
  // poison state is a latched atomic flag plus a mutex-guarded string.
  bool watchdog_tripped() { return task_queue_->poisoned(); }
  std::string watchdog_text() { return task_queue_->poison_text(); }

 private:
  void start_watchdog_(int watchdog_timeout_ms) {
    if (watchdog_timeout_ms <= 0) return;
    const int64_t budget_ns = int64_t(watchdog_timeout_ms) * 1000000;
    watchdog_thread_ = std::thread([this, watchdog_timeout_ms, budget_ns]() {
      // Poll in 10 slices of the budget so destructor shutdown stays prompt.
      const auto slice = std::chrono::milliseconds(
          std::max<int64_t>(1, int64_t(watchdog_timeout_ms) / 10));
      while (!watchdog_stop_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(slice);
        if (watchdog_stop_.load(std::memory_order_acquire)) return;
        int64_t start_ns = task_queue_->current_task_start_ns();
        if (start_ns == 0) continue;
        int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now_ns - start_ns > budget_ns) {
          task_queue_->poison("[kt watchdog] cpu pool no progress >" +
                              std::to_string(watchdog_timeout_ms) + "ms " +
                              describe_task_tag(task_queue_->current_task_tag(),
                                                "task still running"));
          return;  // latched poison: fire once, then let the dtor reclaim us
        }
      }
    });
  }

 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;
  std::atomic<bool> watchdog_stop_{false};
  std::thread watchdog_thread_;
};

#endif
