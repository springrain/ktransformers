// Inflight H2D registry (doc/KT-CPU-PICE-GPU.md section 11).
//
// Guarantees at most one active H2D per (layer, expert, tp_rank). Concurrent
// requests for the same shard reuse the existing device dependency instead of
// submitting a duplicate transfer.
#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "kt/gec/gec_sync.hpp"
#include "kt/gec/gec_types.hpp"

namespace kt::gec {

class InflightRegistry {
 public:
  struct Entry {
    uint64_t dependency_id = 0;
    uint64_t started_at_us = 0;
  };

  // Attempts to register an active transfer. Returns {inserted, dependency}.
  // If the shard is already inflight, returns {false, existing_dependency}
  // and does NOT create a second transfer.
  std::pair<bool, uint64_t> register_transfer(const PhysicalShardId& shard,
                                              uint64_t dependency_id,
                                              uint64_t started_at_us = 0) {
    GecLockGuard lock(mutex_);
    auto it = map_.find(shard);
    if (it != map_.end()) {
      return {false, it->second.dependency_id};
    }
    map_.emplace(shard, Entry{dependency_id, started_at_us});
    return {true, dependency_id};
  }

  bool contains(const PhysicalShardId& shard) const {
    GecLockGuard lock(mutex_);
    return map_.find(shard) != map_.end();
  }

  std::optional<Entry> find(const PhysicalShardId& shard) const {
    GecLockGuard lock(mutex_);
    auto it = map_.find(shard);
    if (it == map_.end()) return std::nullopt;
    return it->second;
  }

  std::optional<uint64_t> dependency_id(const PhysicalShardId& shard) const {
    GecLockGuard lock(mutex_);
    auto it = map_.find(shard);
    if (it == map_.end()) return std::nullopt;
    return it->second.dependency_id;
  }

  // Marks the transfer finished. Idempotent.
  void complete(const PhysicalShardId& shard) {
    GecLockGuard lock(mutex_);
    map_.erase(shard);
  }

  size_t size() const {
    GecLockGuard lock(mutex_);
    return map_.size();
  }

  void clear() {
    GecLockGuard lock(mutex_);
    map_.clear();
  }

 private:
  mutable GecMutex mutex_;
  std::unordered_map<PhysicalShardId, Entry, PhysicalShardIdHash> map_;
};

}  // namespace kt::gec
