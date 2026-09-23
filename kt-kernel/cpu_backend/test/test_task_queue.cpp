#include "../task_queue.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

int main() {
  TaskQueue queue;
  std::atomic<int> completed{0};

  queue.enqueue([&] { completed.fetch_add(1); });
  queue.enqueue([] { throw std::runtime_error("first task failure"); });
  queue.enqueue([&] { completed.fetch_add(1); });

  bool caught = false;
  try {
    queue.sync(0);
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "first task failure";
  }
  assert(caught);
  assert(completed.load() == 2);

  queue.sync(0);
  queue.enqueue([&] { completed.fetch_add(1); });
  queue.sync(0);
  assert(completed.load() == 3);

  {
    TaskQueue callback_queue;
    std::atomic<int> callback_completed{0};

    callback_queue.enqueue([&] { callback_completed.fetch_add(1); });
    callback_queue.enqueue(
        [] { throw std::runtime_error("callback task failure"); });
    callback_queue.enqueue([&] { callback_completed.fetch_add(1); });

    // Host callbacks must only wait for queued work. The exception remains
    // pending until execution returns to a normal, exception-safe sync point.
    callback_queue.sync_noexcept(0);
    assert(callback_completed.load() == 2);
    assert(callback_queue.has_pending_exception());

    bool callback_error_caught = false;
    try {
      callback_queue.sync(0);
    } catch (const std::runtime_error& error) {
      callback_error_caught =
          std::string(error.what()) == "callback task failure";
    }
    assert(callback_error_caught);
    assert(!callback_queue.has_pending_exception());

    callback_queue.sync(0);
  }

  {
    TaskQueue drain_queue;
    std::atomic<bool> trailing_task_completed{false};
    drain_queue.enqueue([] { throw std::runtime_error("drain before throw"); });
    drain_queue.enqueue([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      trailing_task_completed.store(true);
    });

    drain_queue.sync_noexcept(1);
    assert(drain_queue.has_pending_exception());
    try {
      drain_queue.sync(1);
      assert(false);
    } catch (const std::runtime_error& error) {
      assert(std::string(error.what()) == "drain before throw");
    }
    // Even though one pending task was nominally allowed, an exception must
    // drain it before control unwinds to Python.
    assert(trailing_task_completed.load());
  }

  {
    TaskQueue tracked_queue;
    std::atomic<bool> release_trailing{false};
    std::atomic<bool> trailing_started{false};
    std::atomic<bool> trailing_completed{false};

    auto completion = tracked_queue.enqueue_tracked([&] {
      completed.fetch_add(1);
    });
    tracked_queue.enqueue([&] {
      trailing_started.store(true);
      while (!release_trailing.load()) {
        std::this_thread::yield();
      }
      trailing_completed.store(true);
    });

    completion->wait();
    assert(completion->ready());
    assert(!trailing_completed.load());
    // The worker may or may not have entered the trailing task before the
    // tracked completion wakes us, but waiting on the completion must never
    // require that later task to finish.
    (void)trailing_started;

    release_trailing.store(true);
    tracked_queue.sync(0);
    assert(trailing_completed.load());
  }

  {
    // Exercise repeated idle -> notified transitions.  Publishing a node
    // without coordinating with the condition-variable mutex can otherwise
    // lose a wakeup and hang one of these sync calls.
    TaskQueue wakeup_queue;
    std::atomic<int> wakeups{0};
    for (int i = 0; i < 2000; ++i) {
      wakeup_queue.enqueue([&] { wakeups.fetch_add(1); });
      wakeup_queue.sync(0);
    }
    assert(wakeups.load() == 2000);
  }

  {
    TaskQueue tracked_error_queue;
    auto failed = std::make_shared<TaskCompletion>();
    tracked_error_queue.enqueue_tracked(
        [] { throw std::runtime_error("tracked task failure"); }, failed);
    bool tracked_error_caught = false;
    try {
      failed->wait();
    } catch (const std::runtime_error& error) {
      tracked_error_caught =
          std::string(error.what()) == "tracked task failure";
    }
    assert(tracked_error_caught);
    assert(failed->ready());

    // Consuming a tracked error must not poison the whole CPUInfer queue: the
    // caller may need to submit an immediate CPU fallback task.
    tracked_error_queue.sync(0);
    std::atomic<bool> fallback_completed{false};
    tracked_error_queue.enqueue([&] { fallback_completed.store(true); });
    tracked_error_queue.sync(0);
    assert(fallback_completed.load());
  }

  queue.enqueue([] { throw std::runtime_error("first"); });
  queue.enqueue([] { throw std::runtime_error("second"); });
  try {
    queue.sync(0);
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()) == "first");
  }

  return 0;
}
