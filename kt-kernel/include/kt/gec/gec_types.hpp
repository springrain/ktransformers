// Global Expert Cache (GEC) core types.
//
// Implements the Logical Expert / Physical Shard dual-layer model from
// doc/KT-CPU-PICE-GPU.md sections 3, 7, 8, 9, 10.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kt::gec {

// A Logical Expert is the unit of routing, cache admission, H2D admission,
// readiness tracking, and eviction.
struct LogicalExpertId {
  int layer_id = -1;
  int expert_id = -1;

  bool operator==(const LogicalExpertId& other) const {
    return layer_id == other.layer_id && expert_id == other.expert_id;
  }
  bool operator!=(const LogicalExpertId& other) const { return !(*this == other); }
  bool operator<(const LogicalExpertId& other) const {
    if (layer_id != other.layer_id) return layer_id < other.layer_id;
    return expert_id < other.expert_id;
  }
};

struct LogicalExpertIdHash {
  std::size_t operator()(const LogicalExpertId& id) const noexcept {
    uint64_t h = (static_cast<uint64_t>(static_cast<uint32_t>(id.layer_id)) << 32) |
                 static_cast<uint32_t>(id.expert_id);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return static_cast<std::size_t>(h);
  }
};

// A Physical Shard is the unit of GPU storage, H2D execution, and kernel reads.
struct PhysicalShardId {
  int layer_id = -1;
  int expert_id = -1;
  int tp_rank = -1;

  bool operator==(const PhysicalShardId& other) const {
    return layer_id == other.layer_id && expert_id == other.expert_id && tp_rank == other.tp_rank;
  }
  bool operator!=(const PhysicalShardId& other) const { return !(*this == other); }
  bool operator<(const PhysicalShardId& other) const {
    if (layer_id != other.layer_id) return layer_id < other.layer_id;
    if (expert_id != other.expert_id) return expert_id < other.expert_id;
    return tp_rank < other.tp_rank;
  }
};

struct PhysicalShardIdHash {
  std::size_t operator()(const PhysicalShardId& id) const noexcept {
    uint64_t h = (static_cast<uint64_t>(static_cast<uint32_t>(id.layer_id)) << 32) |
                 static_cast<uint32_t>(id.expert_id);
    h ^= static_cast<uint32_t>(id.tp_rank) * 0x9e3779b97f4a7c15ULL;
    h ^= h >> 29;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 32;
    return static_cast<std::size_t>(h);
  }
};

// Lifecycle of a Logical Expert inside the persistent cache (doc section 7).
enum class LifecycleState : uint8_t {
  Absent = 0,
  Scheduled = 1,
  Loading = 2,
  Ready = 3,
  Evicting = 4,
};

inline const char* to_string(LifecycleState state) {
  switch (state) {
    case LifecycleState::Absent: return "ABSENT";
    case LifecycleState::Scheduled: return "SCHEDULED";
    case LifecycleState::Loading: return "LOADING";
    case LifecycleState::Ready: return "READY";
    case LifecycleState::Evicting: return "EVICTING";
  }
  return "UNKNOWN";
}

enum class ShardState : uint8_t {
  Absent = 0,
  Loading = 1,
  Ready = 2,
  Evicting = 3,
};

inline const char* to_string(ShardState state) {
  switch (state) {
    case ShardState::Absent: return "ABSENT";
    case ShardState::Loading: return "LOADING";
    case ShardState::Ready: return "READY";
    case ShardState::Evicting: return "EVICTING";
  }
  return "UNKNOWN";
}

// Persistent-cache admission state. Probation protects freshly admitted
// experts from immediate eviction (doc section 19).
enum class AdmissionState : uint8_t {
  NotAdmitted = 0,
  Probation = 1,
  Cached = 2,
};

inline const char* to_string(AdmissionState state) {
  switch (state) {
    case AdmissionState::NotAdmitted: return "NOT_ADMITTED";
    case AdmissionState::Probation: return "PROBATION";
    case AdmissionState::Cached: return "CACHED";
  }
  return "UNKNOWN";
}

// Physical Shard placement record (doc section 10).
struct PhysicalShard {
  PhysicalShardId id;
  int gpu_id = -1;
  int slot_id = -1;
  ShardState state = ShardState::Absent;
  uint64_t ready_dependency = 0;
};

// Logical Expert record (doc section 9). The three dimensions lifecycle /
// dependency / usage are represented by `state`, per-shard readiness, and
// `refcount` respectively, and are fully separated (doc section 8).
struct LogicalExpert {
  LogicalExpertId id;
  LifecycleState state = LifecycleState::Absent;
  std::vector<PhysicalShard> shards;  // indexed by tp_rank
  std::atomic<int> refcount{0};
  bool in_flight = false;
  bool protected_ = false;
  uint64_t last_used = 0;
  uint64_t logical_ready_dependency = 0;
  uint64_t hit_count = 0;
  uint64_t use_count = 0;
  AdmissionState admission_state = AdmissionState::NotAdmitted;

  LogicalExpert() = default;
  explicit LogicalExpert(const LogicalExpertId& expert_id) : id(expert_id) {}
  LogicalExpert(const LogicalExpert&) = delete;
  LogicalExpert& operator=(const LogicalExpert&) = delete;
  LogicalExpert(LogicalExpert&& other) noexcept
      : id(other.id),
        state(other.state),
        shards(std::move(other.shards)),
        refcount(other.refcount.load()),
        in_flight(other.in_flight),
        protected_(other.protected_),
        last_used(other.last_used),
        logical_ready_dependency(other.logical_ready_dependency),
        hit_count(other.hit_count),
        use_count(other.use_count),
        admission_state(other.admission_state) {}
  LogicalExpert& operator=(LogicalExpert&& other) noexcept {
    if (this != &other) {
      id = other.id;
      state = other.state;
      shards = std::move(other.shards);
      refcount.store(other.refcount.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
      in_flight = other.in_flight;
      protected_ = other.protected_;
      last_used = other.last_used;
      logical_ready_dependency = other.logical_ready_dependency;
      hit_count = other.hit_count;
      use_count = other.use_count;
      admission_state = other.admission_state;
    }
    return *this;
  }
};

// A Logical Expert is READY only when every required TP shard is READY
// (TP Atomic Ready rule, doc section 3.3).
inline bool is_logical_ready(const LogicalExpert& expert) {
  if (expert.state != LifecycleState::Ready) return false;
  if (expert.shards.empty()) return false;
  for (const PhysicalShard& shard : expert.shards) {
    if (shard.state != ShardState::Ready) return false;
  }
  return true;
}

// Eviction gate (doc section 18). Every condition must hold.
inline bool can_evict(const LogicalExpert& expert) {
  return expert.state == LifecycleState::Ready &&
         expert.refcount.load(std::memory_order_acquire) == 0 && !expert.in_flight && !expert.protected_;
}

inline std::string to_string(const LogicalExpertId& id) {
  return "(" + std::to_string(id.layer_id) + ", " + std::to_string(id.expert_id) +
         ")";
}

inline std::string to_string(const PhysicalShardId& id) {
  return "(" + std::to_string(id.layer_id) + ", " + std::to_string(id.expert_id) +
         ", " + std::to_string(id.tp_rank) + ")";
}

}  // namespace kt::gec
