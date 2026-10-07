// TP/EP placement planning (doc/KT-CPU-PICE-GPU.md section 4).
//
// Cache admission must place a Logical Expert completely: every TP shard is
// assigned, or the whole placement fails. Partial admission is forbidden.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "kt/gec/gec_types.hpp"

namespace kt::gec {

struct Topology {
  int tp_size = 1;
  int ep_size = 1;
  int gpu_count = 1;

  void validate() const {
    if (tp_size < 1) throw std::invalid_argument("tp_size must be >= 1");
    if (ep_size < 1) throw std::invalid_argument("ep_size must be >= 1");
    if (gpu_count < 1) throw std::invalid_argument("gpu_count must be >= 1");
  }

  // EP owner for a logical expert. A stable hash keeps placement reproducible
  // across rounds without requiring a full expert-count table.
  int ep_owner(const LogicalExpertId& id) const {
    uint64_t h = (static_cast<uint64_t>(static_cast<uint32_t>(id.layer_id)) << 20) ^
                 static_cast<uint32_t>(id.expert_id);
    h ^= h >> 15;
    return static_cast<int>(h % static_cast<uint64_t>(ep_size));
  }

  // GPU hosting a given TP shard of an expert.
  int gpu_for(const LogicalExpertId& id, int tp_rank) const {
    if (tp_rank < 0 || tp_rank >= tp_size) {
      throw std::invalid_argument("tp_rank out of range");
    }
    int gpu = ep_owner(id) * tp_size + tp_rank;
    if (gpu >= gpu_count) {
      throw std::runtime_error("placement unavailable: topology has fewer GPUs than EP*TP");
    }
    return gpu;
  }
};

class PlacementPlanner {
 public:
  explicit PlacementPlanner(Topology topology) : topology_(topology) {
    topology_.validate();
  }

  const Topology& topology() const { return topology_; }

  // Complete placement for one Logical Expert: exactly one shard per TP rank.
  // `base_slot` is the cache slot index reserved for this expert; every shard
  // of the same expert shares the slot so the expert occupies exactly one
  // Logical slot.
  std::vector<PhysicalShard> plan(const LogicalExpertId& id, int base_slot) const {
    if (base_slot < 0) throw std::invalid_argument("base_slot must be >= 0");
    std::vector<PhysicalShard> shards;
    shards.reserve(static_cast<size_t>(topology_.tp_size));
    for (int tp_rank = 0; tp_rank < topology_.tp_size; ++tp_rank) {
      PhysicalShard shard;
      shard.id = PhysicalShardId{id.layer_id, id.expert_id, tp_rank};
      shard.gpu_id = topology_.gpu_for(id, tp_rank);
      shard.slot_id = base_slot;
      shard.state = ShardState::Absent;
      shards.push_back(shard);
    }
    return shards;
  }

  // All-or-nothing planning for a batch. Throws on the first failure so the
  // caller can roll back the whole round (no partial admission).
  std::vector<PhysicalShard> plan_all(const std::vector<LogicalExpertId>& ids,
                                      int base_slot) const {
    std::vector<PhysicalShard> shards;
    shards.reserve(ids.size() * static_cast<size_t>(topology_.tp_size));
    for (size_t index = 0; index < ids.size(); ++index) {
      std::vector<PhysicalShard> part =
          plan(ids[index], base_slot + static_cast<int>(index));
      shards.insert(shards.end(), part.begin(), part.end());
    }
    return shards;
  }

 private:
  Topology topology_;
};

}  // namespace kt::gec
