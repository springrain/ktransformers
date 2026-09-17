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

#include <atomic>
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

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
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
    });
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
 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;
};

#endif
