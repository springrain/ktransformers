// Per-GPU H2D buffer pool (doc/KT-CPU-PICE-GPU.md sections 15-16).
//
// Buffers are temporary transfer resources, never counted as persistent cache.
// A buffer may only return to FREE after GPU kernel completion; H2D completion
// alone is not enough.
//
// The pool is a two-generation ring (doc section 15.3, mirroring FreeToken's
// `buffer_id = layer_id % 2` prefill staging): acquire() rotates the cursor so
// consecutive rounds land on different generations, and free_count()/reuse
// waits expose when the ring is out of depth and a round has to stall on the
// previous generation's release.
#pragma once

#include <cstdint>
#include <stdexcept>
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
  // Round that acquired the buffer.  `round % depth` is the generation, so
  // back-to-back rounds never reuse a buffer the previous round still holds.
  // Round 0 is not a generation (rounds start at 1).
  uint64_t round = 0;
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

  // Round-to-buffer mapping that keeps consecutive rounds on separate
  // generations (FreeToken `layer_id % 2` generalised to any depth).
  int generation_of(uint64_t round) const {
    return static_cast<int>(round % static_cast<uint64_t>(buffers_.size()));
  }

  // FREE -> H2D. Returns nullptr when the ring has no free generation
  // (backpressure). When the round's own generation is busy but another is
  // free, the pool takes the other one rather than stalling.
  H2DBuffer* acquire(const LogicalExpertId& expert, int tp_rank,
                     uint64_t owner_request, uint64_t round = 0) {
    GecLockGuard lock(mutex_);
    const int preferred = generation_of(round);
    if (buffers_[static_cast<size_t>(preferred)].state == BufferState::Free) {
      return occupy_(buffers_[static_cast<size_t>(preferred)], expert, tp_rank,
                     owner_request, round);
    }
    const size_t depth = buffers_.size();
    for (size_t i = 1; i < depth; ++i) {
      H2DBuffer& candidate = buffers_[(static_cast<size_t>(preferred) + i) % depth];
      if (candidate.state == BufferState::Free) {
        return occupy_(candidate, expert, tp_rank, owner_request, round);
      }
    }
    return nullptr;
  }

  // The ring slot this round writes to. Rounds start at 1, so the slot a round
  // owns is the one acquired by `round - depth` (FreeToken: layer L and layer
  // L - 2 share a buffer; L waits for L - 2, not for L - 1). nullptr when
  // nothing acquired the slot.
  const H2DBuffer* at_generation(uint64_t round) const {
    GecLockGuard lock(mutex_);
    if (round <= buffers_.size()) return nullptr;
    const H2DBuffer& buffer =
        buffers_[static_cast<size_t>(round % static_cast<uint64_t>(buffers_.size()))];
    return buffer.round == round - buffers_.size() ? &buffer : nullptr;
  }

  // Number of buffers still held by rounds older than `round` (excluding the
  // ones this round already acquired itself).
  int held_by_older_generations(uint64_t round) const {
    GecLockGuard lock(mutex_);
    int held = 0;
    for (const H2DBuffer& buffer : buffers_) {
      if (buffer.state != BufferState::Free && buffer.round < round) ++held;
    }
    return held;
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
  static H2DBuffer* occupy_(H2DBuffer& buffer, const LogicalExpertId& expert,
                            int tp_rank, uint64_t owner_request, uint64_t round) {
    buffer.logical_expert = expert;
    buffer.tp_rank = tp_rank;
    buffer.owner_request = owner_request;
    buffer.round = round;
    buffer.ready_dependency = 0;
    buffer.state = BufferState::H2D;
    return &buffer;
  }

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
