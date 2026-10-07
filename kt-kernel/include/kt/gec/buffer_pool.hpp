// Per-GPU H2D buffer pool (doc/KT-CPU-PICE-GPU.md sections 15-16).
//
// Buffers are temporary transfer resources, never counted as persistent cache.
// A buffer may only return to FREE after GPU kernel completion; H2D completion
// alone is not enough.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "kt/gec/gec_sync.hpp"
#include "kt/gec/gec_types.hpp"

namespace kt::gec {

enum class BufferState : uint8_t {
  Free = 0,
  H2D = 1,
  ReadyForCompute = 2,
  GpuInUse = 3,
  CompletionWait = 4,
};

inline const char* to_string(BufferState state) {
  switch (state) {
    case BufferState::Free: return "FREE";
    case BufferState::H2D: return "H2D";
    case BufferState::ReadyForCompute: return "READY_FOR_COMPUTE";
    case BufferState::GpuInUse: return "GPU_IN_USE";
    case BufferState::CompletionWait: return "COMPLETION_WAIT";
  }
  return "UNKNOWN";
}

struct H2DBuffer {
  int gpu_id = -1;
  int buffer_id = -1;
  LogicalExpertId logical_expert;
  int tp_rank = -1;
  BufferState state = BufferState::Free;
  uint64_t ready_dependency = 0;
  uint64_t owner_request = 0;
};

class H2DBufferPool {
 public:
  H2DBufferPool(int gpu_id, int depth) : gpu_id_(gpu_id) {
    if (gpu_id < 0) throw std::invalid_argument("gpu_id must be >= 0");
    if (depth < 1) throw std::invalid_argument("buffer pool depth must be >= 1");
    buffers_.reserve(static_cast<size_t>(depth));
    for (int i = 0; i < depth; ++i) {
      H2DBuffer buffer;
      buffer.gpu_id = gpu_id;
      buffer.buffer_id = i;
      buffers_.push_back(buffer);
    }
  }

  int gpu_id() const { return gpu_id_; }
  int depth() const { return static_cast<int>(buffers_.size()); }

  // FREE -> H2D. Returns nullptr when no buffer is free (backpressure).
  H2DBuffer* acquire(const LogicalExpertId& expert, int tp_rank, uint64_t owner_request) {
    GecLockGuard lock(mutex_);
    for (H2DBuffer& buffer : buffers_) {
      if (buffer.state != BufferState::Free) continue;
      buffer.logical_expert = expert;
      buffer.tp_rank = tp_rank;
      buffer.owner_request = owner_request;
      buffer.ready_dependency = 0;
      buffer.state = BufferState::H2D;
      return &buffer;
    }
    return nullptr;
  }

  // H2D -> READY_FOR_COMPUTE. Records the device-local dependency produced by
  // the H2D stream (e.g. a CUDA event).
  void mark_ready(int buffer_id, uint64_t ready_dependency) {
    GecLockGuard lock(mutex_);
    H2DBuffer& buffer = at_(buffer_id);
    if (buffer.state != BufferState::H2D) {
      throw std::runtime_error("buffer is not in H2D state");
    }
    buffer.ready_dependency = ready_dependency;
    buffer.state = BufferState::ReadyForCompute;
  }

  // READY_FOR_COMPUTE -> GPU_IN_USE (compute stream consumed the dependency).
  void begin_gpu_use(int buffer_id) {
    GecLockGuard lock(mutex_);
    H2DBuffer& buffer = at_(buffer_id);
    if (buffer.state != BufferState::ReadyForCompute) {
      throw std::runtime_error("buffer is not ready for compute");
    }
    buffer.state = BufferState::GpuInUse;
  }

  // GPU_IN_USE -> COMPLETION_WAIT -> FREE. Only legal after GPU completion.
  void gpu_complete(int buffer_id) {
    GecLockGuard lock(mutex_);
    H2DBuffer& buffer = at_(buffer_id);
    if (buffer.state != BufferState::GpuInUse) {
      throw std::runtime_error("buffer is not in GPU use");
    }
    buffer.state = BufferState::CompletionWait;
    buffer.state = BufferState::Free;  // completion observed: release
    buffer.ready_dependency = 0;
  }

  // H2D -> FREE. Cancels a buffer whose transfer was never submitted (e.g.
  // the expert turned out to be inflight under another request).
  void cancel(int buffer_id) {
    GecLockGuard lock(mutex_);
    H2DBuffer& buffer = at_(buffer_id);
    if (buffer.state != BufferState::H2D) {
      throw std::runtime_error("only H2D buffers can be cancelled");
    }
    buffer.state = BufferState::Free;
    buffer.ready_dependency = 0;
  }

  // H2D or READY_FOR_COMPUTE -> FREE. Rolls back a buffer whose payload never
  // reached the compute stream (partial-failure path); unlike cancel(), this
  // also covers buffers whose H2D already completed.
  void release_unstarted(int buffer_id) {
    GecLockGuard lock(mutex_);
    H2DBuffer& buffer = at_(buffer_id);
    if (buffer.state != BufferState::H2D &&
        buffer.state != BufferState::ReadyForCompute) {
      throw std::runtime_error("only unstarted buffers can be released");
    }
    buffer.state = BufferState::Free;
    buffer.ready_dependency = 0;
  }

  int free_count() const {
    GecLockGuard lock(mutex_);
    int count = 0;
    for (const H2DBuffer& buffer : buffers_) {
      if (buffer.state == BufferState::Free) ++count;
    }
    return count;
  }

  const H2DBuffer* find(int buffer_id) const {
    GecLockGuard lock(mutex_);
    return &at_(buffer_id);
  }

 private:
  H2DBuffer& at_(int buffer_id) {
    if (buffer_id < 0 || buffer_id >= static_cast<int>(buffers_.size())) {
      throw std::invalid_argument("buffer_id out of range");
    }
    return buffers_[static_cast<size_t>(buffer_id)];
  }

  const H2DBuffer& at_(int buffer_id) const {
    if (buffer_id < 0 || buffer_id >= static_cast<int>(buffers_.size())) {
      throw std::invalid_argument("buffer_id out of range");
    }
    return buffers_[static_cast<size_t>(buffer_id)];
  }

  int gpu_id_;
  mutable GecMutex mutex_;
  std::vector<H2DBuffer> buffers_;
};

}  // namespace kt::gec
