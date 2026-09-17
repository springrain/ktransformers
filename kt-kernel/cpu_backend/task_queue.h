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
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class TaskQueue {
 public:
  TaskQueue();
  ~TaskQueue();

  // task_tag rides along for the watchdog heartbeat display only; it never
  // alters execution order or semantics.
  void enqueue(std::function<void()> task, int64_t task_tag = 0);

  void sync(size_t allow_n_pending);

  // Latch a permanent failure: every later sync() rethrows it. Never clears;
  // a stuck task is still joined at shutdown (poison leaves pending alone).
  void poison(const std::string& what);
  bool poisoned() const;
  std::string poison_text();
  int64_t current_task_start_ns() const;
  int64_t current_task_tag() const;

 private:
  struct Node {
    std::function<void()> task;
    std::atomic<Node*> next;
    int64_t task_tag;
    Node() : task(nullptr), next(nullptr), task_tag(0) {}
    Node(const std::function<void()>& t, int64_t tag) : task(t), next(nullptr), task_tag(tag) {}
  };

  std::atomic<Node*> head;
  std::atomic<Node*> tail;
  std::atomic<bool> done;
  std::atomic<size_t> pending;
  std::thread workerThread;
  std::mutex mtx;
  std::condition_variable cv;
  std::exception_ptr first_exception;

  // Watchdog heartbeat: 0 = idle. Written by the consumer thread only, read
  // by an external monitor (CPUInfer watchdog) without taking mtx.
  std::atomic<int64_t> task_start_ns{0};
  std::atomic<int64_t> pending_task_tag{0};

  // Poison state is latched and outlives any individual sync() call, unlike
  // first_exception which is drained once by the first waiter.
  std::atomic<bool> poisoned_flag{false};
  std::exception_ptr poison_exception;
  std::string poison_what;

  void worker();
};

#endif
