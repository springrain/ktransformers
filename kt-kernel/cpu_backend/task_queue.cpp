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
#include <iostream>
#include <stdexcept>
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

void TaskQueue::enqueue(std::function<void()> task, int64_t task_tag) {
  // Allocate first: a throw after fetch_add would credit pending with no node
  // ever linked, and every later sync() would hang forever.
  Node* node = new Node(task, task_tag);
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  cv.notify_one();
}

void TaskQueue::sync(size_t allow_n_pending) {
  std::exception_ptr task_exception;
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire)
          || poisoned_flag.load(std::memory_order_acquire);
    });
    // Poison wins over first_exception and stays latched: every later sync()
    // rethrows the same poison instead of draining it once.
    if (poisoned_flag.load(std::memory_order_acquire) && poison_exception) {
      task_exception = poison_exception;
    } else {
      task_exception = first_exception;
      first_exception = nullptr;
    }
  }
  if (task_exception) std::rethrow_exception(task_exception);
}

void TaskQueue::poison(const std::string& what) {
  {
    std::lock_guard<std::mutex> lock(mtx);
    bool expected = false;
    if (!poisoned_flag.compare_exchange_strong(expected, true,
                                               std::memory_order_acq_rel)) {
      return;  // set-once: the first poison wins
    }
    poison_what = what;
    poison_exception = std::make_exception_ptr(std::runtime_error(what));
  }
  cv.notify_all();
}

bool TaskQueue::poisoned() const { return poisoned_flag.load(std::memory_order_acquire); }

std::string TaskQueue::poison_text() {
  std::lock_guard<std::mutex> lock(mtx);
  return poison_what;
}

int64_t TaskQueue::current_task_start_ns() const {
  return task_start_ns.load(std::memory_order_acquire);
}

int64_t TaskQueue::current_task_tag() const {
  return pending_task_tag.load(std::memory_order_relaxed);
}

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      std::exception_ptr task_exception;
      // Heartbeat before the task body runs: an external monitor treats an
      // overdue nonzero start as "no progress" and may poison the queue.
      pending_task_tag.store(next->task_tag, std::memory_order_relaxed);
      task_start_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count(),
                          std::memory_order_release);
      if (next->task) {
        try {
          next->task();
        } catch (...) {
          task_exception = std::current_exception();
        }
      }
      task_start_ns.store(0, std::memory_order_release);
      pending_task_tag.store(0, std::memory_order_relaxed);
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_exception && !first_exception) {
          first_exception = task_exception;
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
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
