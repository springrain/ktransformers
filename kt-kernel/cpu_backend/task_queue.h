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
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class TaskQueue {
 public:
  TaskQueue();
  ~TaskQueue();

  void enqueue(std::function<void()>);

  void sync(size_t allow_n_pending);
  // Host callbacks are C ABI boundaries and must not let C++ exceptions
  // escape. Wait for the requested queue depth while leaving any worker
  // exception latched for the next normal sync() call.
  void sync_noexcept(size_t allow_n_pending) noexcept;

  // Record an exception raised while a host callback is submitting work.
  // The first pending exception wins, matching worker-thread semantics.
  void record_exception(std::exception_ptr exception) noexcept;

  // Re-throw and consume the first latched exception at a normal C++/Python
  // boundary where exception propagation is safe.
  void rethrow_pending_exception();

 private:
  struct Node {
    std::function<void()> task;
    std::atomic<Node*> next;
    Node() : task(nullptr), next(nullptr) {}
    Node(const std::function<void()>& t) : task(t), next(nullptr) {}
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
  void worker();
};

#endif
