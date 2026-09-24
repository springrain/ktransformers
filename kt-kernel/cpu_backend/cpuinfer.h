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
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(KTRANSFORMERS_CPUINFER_TEST_VENDOR_HEADER)
#include KTRANSFORMERS_CPUINFER_TEST_VENDOR_HEADER
#elif defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
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

#if !defined(KTRANSFORMERS_CPUINFER_TEST_VENDOR_HEADER)
#include "./vendors/vendor.h"
#endif
#include "cpuinfer_task.h"
#include "ggml-cpu.h"
#include "task_queue.h"
#include "worker_pool.h"

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

  void enqueue(std::function<void()> task) {
    task_queue_->enqueue(std::move(task));
  }

  template <typename Func, typename Obj, typename... Args>
  std::shared_ptr<TaskCompletion> enqueue_tracked(Func f, Obj* obj, Args... args) {
    return task_queue_->enqueue_tracked([=]() {
      std::invoke(f, *obj, args...);
    });
  }

  std::shared_ptr<TaskCompletion> enqueue_tracked(std::function<void()> task) {
    return task_queue_->enqueue_tracked(std::move(task));
  }

  template <typename Func, typename Obj, typename... Args>
  void enqueue_tracked(const std::shared_ptr<TaskCompletion>& completion,
                       Func f, Obj* obj, Args... args) {
    task_queue_->enqueue_tracked(
        [=]() { std::invoke(f, *obj, args...); }, completion);
  }

  void enqueue_tracked(const std::shared_ptr<TaskCompletion>& completion,
                       std::function<void()> task) {
    task_queue_->enqueue_tracked(std::move(task), completion);
  }

  void submit(CPUInferTask params) {
    CPUInferTaskCancelGuard task_guard(params);
    if (task_queue_->has_pending_exception()) {
      task_queue_->sync(0);
    }
    CPUInferTaskFunction func = cpuinfer_task_function(params);
    void* args = cpuinfer_task_args(params);
    *((CPUInfer**)args) = this;
    func(cpuinfer_task_opaque(params));
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_with_cuda_stream(intptr_t user_cuda_stream, CPUInferTask params, bool capture_active = true) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    CPUInferTaskCancelGuard task_guard(params);
    bool capturing = stream_capture_active(user_cuda_stream, capture_active);
    if (!capturing && task_queue_->has_pending_exception()) {
      synchronize_all_and_rethrow();
    }

    void* args = cpuinfer_task_args(params);
    *((CPUInfer**)args) = this;
    StreamTaskArgs* stream_args = new StreamTaskArgs{params, !capturing};
    cudaError_t err =
        cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&submit_, (void*)stream_args);
    if (err != cudaSuccess) {
      delete stream_args;
      task_guard.cancel();
      throw_launch_failure("CPUInfer host callback", err, user_cuda_stream, capturing);
    }
    task_guard.release();
#endif
  }
#endif

  void record_callback_exception(std::exception_ptr exception) noexcept {
    task_queue_->record_exception(exception);
  }

  void rethrow_pending_callback_exception() {
    task_queue_->rethrow_pending_exception();
  }

  WorkerPool* worker_pool() const noexcept { return backend_; }

  struct StreamTaskArgs {
    CPUInferTask task;
    bool autofree;
  };

  static void submit_(void* opaque) noexcept {
    auto* stream_args = static_cast<StreamTaskArgs*>(opaque);
    void* task_args = cpuinfer_task_args(stream_args->task);
    CPUInfer* cpuinfer = *reinterpret_cast<CPUInfer**>(task_args);
    try {
      cpuinfer_task_function(stream_args->task)(cpuinfer_task_opaque(stream_args->task));
    } catch (...) {
      cpuinfer->record_callback_exception(std::current_exception());
    }
    if (stream_args->autofree) {
      destroy_cpuinfer_task(stream_args->task);
      delete stream_args;
    }
  }

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
    // True only when the args cannot be graph-recorded: sync_ then frees them
    // after syncing. Graph-recorded args are re-invoked by every replay.
    bool autofree;
  };

  static void sync_(void* sync_args) noexcept {
    SyncArgs* args = (SyncArgs*)sync_args;
    CPUInfer* cpuinfer = args->cpuinfer;
    size_t allow_n_pending = args->allow_n_pending;
    bool autofree = args->autofree;
    // Host callbacks cannot propagate C++ exceptions. Keep any worker error
    // latched in TaskQueue until the next normal CPUInfer entry point.
    cpuinfer->task_queue_->sync_noexcept(allow_n_pending);
    if (autofree) delete args;
  }

  void sync(size_t allow_n_pending = 0) {
    task_queue_->sync(allow_n_pending);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void synchronize_and_rethrow(intptr_t user_cuda_stream) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    cudaError_t stream_err = cudaStreamSynchronize((cudaStream_t)user_cuda_stream);
    std::exception_ptr queue_error;
    try {
      task_queue_->sync(0);
    } catch (...) {
      queue_error = std::current_exception();
    }
    if (queue_error) std::rethrow_exception(queue_error);
    if (stream_err != cudaSuccess) {
      throw std::runtime_error(std::string("Failed to synchronize CPUInfer stream: ")
                               + vendor_error_string(stream_err));
    }
#endif
  }

  void synchronize_all_and_rethrow() {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    cudaError_t device_err = cudaDeviceSynchronize();
    std::exception_ptr queue_error;
    try {
      task_queue_->sync(0);
    } catch (...) {
      queue_error = std::current_exception();
    }
    if (queue_error) std::rethrow_exception(queue_error);
    if (device_err != cudaSuccess) {
      throw std::runtime_error(std::string("Failed to synchronize CPUInfer device: ")
                               + vendor_error_string(device_err));
    }
#endif
  }

  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0, bool capture_active = true) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    bool capturing = stream_capture_active(user_cuda_stream, capture_active);
    if (!capturing && task_queue_->has_pending_exception()) {
      synchronize_all_and_rethrow();
    }

    // Single-use unless the stream is capturing: capture records the args pointer
    // into a host node that replays it forever, so only eager launches self-free.
    // The default capture_active=true keeps older non-CUDA callers conservative.
    bool autofree = !capturing;
    SyncArgs* args = nullptr;
    try {
      args = new SyncArgs{this, allow_n_pending, autofree};
    } catch (...) {
      if (!capturing) {
        try {
          synchronize_all_and_rethrow();
        } catch (...) {
          // Preserve the allocation failure while still making a best-effort
          // attempt to drain the already-submitted stream work.
        }
      }
      throw;
    }
    cudaError_t launch_err = cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
    if (launch_err != cudaSuccess) {
      delete args;
      throw_launch_failure("CPUInfer sync callback", launch_err, user_cuda_stream, capturing);
    }
#endif
  }
#endif

 private:
#ifndef KTRANSFORMERS_CPU_ONLY
  static const char* vendor_error_string(cudaError_t error) noexcept {
    const char* message = cudaGetErrorString(error);
    return message != nullptr ? message : "unknown accelerator runtime error";
  }

  bool stream_capture_active(intptr_t user_cuda_stream, bool capture_active) noexcept {
    bool capturing = capture_active;
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
    cudaStreamCaptureStatus capture_status{};
    cudaError_t err = cudaStreamGetCaptureInfo((cudaStream_t)user_cuda_stream, &capture_status, nullptr);
    bool query_succeeded = err == cudaSuccess;
    capturing = resolve_cpuinfer_capture_state(
        capture_active, query_succeeded,
        query_succeeded && capture_status != cudaStreamCaptureStatusNone);
    if (!query_succeeded) {
      // A failed query cannot prove that the callback is single-use. Retain
      // its arguments so a captured graph can never replay a freed pointer.
      (void)cudaGetLastError();
    }
#else
    (void)user_cuda_stream;
#endif
    return capturing;
  }

  [[noreturn]] void throw_launch_failure(const char* callback_name, cudaError_t launch_error,
                                         intptr_t user_cuda_stream, bool capturing) {
    (void)user_cuda_stream;
    std::string message = std::string("Failed to launch ") + callback_name + ": "
        + vendor_error_string(launch_error);
    if (!capturing) {
      try {
        synchronize_all_and_rethrow();
      } catch (const std::exception& drain_error) {
        message += std::string("; failed while draining submitted work: ") + drain_error.what();
      } catch (...) {
        message += "; failed while draining submitted work with an unknown exception";
      }
    }
    throw std::runtime_error(message);
  }
#endif

 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;
};

#endif
