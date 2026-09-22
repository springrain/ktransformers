/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

TaskQueue::TaskQueue() : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  workerThread = std::thread(&TaskQueue::worker, this);
}

TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  // Allocate first: a throw after fetch_add would credit pending with no node
  // ever linked, and every later sync() would hang forever.
  Node* node = new Node(task);
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  cv.notify_one();
}

void TaskQueue::wait_for_pending(size_t allow_n_pending) {
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&] {
    return pending.load(std::memory_order_acquire) <= allow_n_pending
        || done.load(std::memory_order_acquire);
  });
}

void TaskQueue::record_exception(std::exception_ptr exception) noexcept {
  if (!exception) return;
  bool recorded = false;
  try {
    std::lock_guard<std::mutex> lock(mtx);
    if (!first_exception) {
      first_exception = exception;
      recorded = true;
    }
  } catch (...) {
    // There is no safe way to report a mutex failure from a C host callback.
  }
  if (recorded) log_exception(exception);
}

bool TaskQueue::has_pending_exception() noexcept {
  try {
    std::lock_guard<std::mutex> lock(mtx);
    return first_exception != nullptr;
  } catch (...) {
    // If the state cannot be inspected safely, force the caller onto its
    // conservative drain path.
    return true;
  }
}

void TaskQueue::sync(size_t allow_n_pending) {
  std::exception_ptr task_exception;
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire);
    });
    if (first_exception && pending.load(std::memory_order_acquire) > 0
        && !done.load(std::memory_order_acquire)) {
      cv.wait(lock, [&] {
        return pending.load(std::memory_order_acquire) == 0
            || done.load(std::memory_order_acquire);
      });
    }
    task_exception = first_exception;
    first_exception = nullptr;
  }
  if (task_exception) std::rethrow_exception(task_exception);
}

void TaskQueue::log_exception(std::exception_ptr exception) noexcept {
  if (!exception) return;
  try {
    std::rethrow_exception(exception);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[kt-kernel] asynchronous CPUInfer task failed: %s\n", error.what());
  } catch (...) {
    std::fprintf(stderr, "[kt-kernel] asynchronous CPUInfer task failed with an unknown exception\n");
  }
}

void TaskQueue::sync_noexcept(size_t allow_n_pending) noexcept {
  try {
    wait_for_pending(allow_n_pending);
    if (allow_n_pending > 0 && has_pending_exception()) {
      wait_for_pending(0);
    }
  } catch (...) {
    record_exception(std::current_exception());
  }
}

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      std::exception_ptr task_exception;
      if (next->task) {
        try {
          next->task();
        } catch (...) {
          task_exception = std::current_exception();
        }
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      bool recorded_exception = false;
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_exception && !first_exception) {
          first_exception = task_exception;
          recorded_exception = true;
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      if (recorded_exception) log_exception(task_exception);
      cv.notify_all();
    } else {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
  }
}
