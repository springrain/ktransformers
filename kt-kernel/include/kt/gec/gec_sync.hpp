// Portable synchronization primitives for the GEC core.
//
// Some MinGW distributions use the win32 thread model where std::mutex is
// unavailable. The simulation core is single-threaded by design, so a null
// mutex keeps the public API identical on such toolchains while real builds
// (Linux/MSVC/POSIX MinGW) use std::mutex.
#pragma once

#if defined(__GLIBCXX__) && !defined(_GLIBCXX_HAS_GTHREADS)
#define GEC_NULL_MUTEX 1
#else
#include <mutex>
#endif

namespace kt::gec {

#if GEC_NULL_MUTEX
class GecMutex {
 public:
  void lock() {}
  void unlock() {}
};
#else
using GecMutex = std::mutex;
#endif

class GecLockGuard {
 public:
  explicit GecLockGuard(GecMutex& mutex) : mutex_(mutex) { mutex_.lock(); }
  ~GecLockGuard() { mutex_.unlock(); }
  GecLockGuard(const GecLockGuard&) = delete;
  GecLockGuard& operator=(const GecLockGuard&) = delete;

 private:
  GecMutex& mutex_;
};

}  // namespace kt::gec
