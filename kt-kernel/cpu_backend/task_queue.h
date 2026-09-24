/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-16 10:43:18
 * @Version   : 1.0.0
 * @LastEditors : chenht
 * @LastEditTime : 2024-10-09 11:08:07
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_TASKQUEUE_H
#define CPUINFER_TASKQUEUE_H

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class TaskCompletion {
 public:
  TaskCompletion() = default;

  TaskCompletion(const TaskCompletion&) = delete;
  TaskCompletion& operator=(const TaskCompletion&) = delete;

  bool ready() const noexcept;
  void wait();
  void finish(std::exception_ptr exception = nullptr) noexcept;

 private:
  mutable std::mutex mtx_;
  std::condition_variable cv_;
  bool done_ = false;
  std::exception_ptr exception_;
};

class TaskQueue {
 public:
  TaskQueue();
  ~TaskQueue();

  void enqueue(std::function<void()>);
  void enqueue_tracked(
      std::function<void()>,
      const std::shared_ptr<TaskCompletion>& completion);
  std::shared_ptr<TaskCompletion> enqueue_tracked(std::function<void()>);

  void sync(size_t allow_n_pending);
  // Host callbacks are C ABI boundaries and must not let C++ exceptions
  // escape. Wait for the requested queue depth while leaving any worker
  // exception latched for the next normal sync() call.
  void sync_noexcept(size_t allow_n_pending) noexcept;

  // Record an exception raised while a host callback is submitting work.
  // The first pending exception wins, matching worker-thread semantics.
  void record_exception(std::exception_ptr exception) noexcept;

  bool has_pending_exception() noexcept;
  // Rethrow a latched callback/worker exception without waiting for queued
  // work.  Callers use this immediately after a CUDA host callback boundary to
  // validate that the callback successfully enqueued its CPU task while the
  // task itself remains free to run asynchronously.
  void rethrow_pending_exception();

 private:
  struct Node {
    std::function<void()> task;
    std::shared_ptr<TaskCompletion> completion;
    std::atomic<Node*> next;
    Node() : task(nullptr), completion(nullptr), next(nullptr) {}
    Node(std::function<void()> t,
         std::shared_ptr<TaskCompletion> c = nullptr)
        : task(std::move(t)), completion(std::move(c)), next(nullptr) {}
  };

  std::atomic<Node*> head;
  std::atomic<Node*> tail;
  std::atomic<bool> done;
  std::atomic<size_t> pending;
  std::thread workerThread;
  std::mutex mtx;
  std::condition_variable cv;
  std::exception_ptr first_exception;

  void wait_for_pending(size_t allow_n_pending);
  static void log_exception(std::exception_ptr exception) noexcept;
  void worker();
};

#endif
