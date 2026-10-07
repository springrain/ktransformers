// Global Expert Cache (doc/KT-CPU-PICE-GPU.md sections 6-9, 17-19).
//
// Persistent cache with Logical-Expert granularity. Admission and eviction are
// whole-TP all-or-nothing operations. Probation protects freshly admitted
// experts. Experts in layers < num_gpu_layers are permanently resident and do
// NOT consume expert_gpu_slots; the slot budget is filled from layer
// num_gpu_layers onwards.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kt/gec/gec_sync.hpp"
#include "kt/gec/gec_config.hpp"
#include "kt/gec/gec_types.hpp"
#include "kt/gec/placement.hpp"

namespace kt::gec {

class GlobalExpertCache {
 public:
  enum class LookupResult { Hit, Miss };

  struct Stats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t reh2d = 0;  // transfers of experts previously seen (evicted before)

    double hit_rate() const {
      uint64_t total = hits + misses;
      return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
    }
  };

  GlobalExpertCache(GecConfig config, Topology topology)
      : config_(config), planner_(topology) {
    config_.validate();
  }

  const GecConfig& config() const { return config_; }
  const Topology& topology() const { return planner_.topology(); }
  const Stats& stats() const { return stats_; }

  // Cache lookup. HIT only when the expert is resident and READY.
  LookupResult lookup(const LogicalExpertId& id, uint64_t now_us = 0) {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    if (it == experts_.end() || it->second.state != LifecycleState::Ready) {
      ++stats_.misses;
      return LookupResult::Miss;
    }
    ++stats_.hits;
    ++it->second.hit_count;
    it->second.last_used = now_us;
    return LookupResult::Hit;
  }

  // Whether the scheduler should skip this expert entirely for this round.
  // Mirrors the Candidate Filter inputs of doc section 12.
  bool admission_locked(const LogicalExpertId& id) const {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    if (it == experts_.end()) return false;
    const LogicalExpert& expert = it->second;
    return expert.state == LifecycleState::Evicting ||
           (expert.state == LifecycleState::Ready && is_resident_(it->second) &&
            expert.protected_);
  }

  // Persistent-cache admission for one Logical Expert. Complete TP placement
  // happens or the call fails; no partial admission is possible. Returns false
  // when the cache is full and no victim can be evicted.
  bool admit(const LogicalExpertId& id, uint64_t now_us = 0) {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    if (it != experts_.end() && is_resident_(it->second)) {
      return true;  // already admitted
    }
    const bool permanent = id.layer_id < config_.num_gpu_layers;
    if (!permanent && slot_resident_size_() >= static_cast<size_t>(config_.expert_gpu_slots)) {
      if (!evict_one_locked_()) return false;
    }
    int slot = -1;
    if (!permanent) {
      if (!reusable_slots_.empty()) {
        slot = reusable_slots_.back();
        reusable_slots_.pop_back();
      } else {
        slot = next_slot_++;
      }
    } else {
      // Permanent layers have their own static rows and do not consume the
      // global slot budget.  A negative slot makes that distinction explicit
      // to physical placement consumers.
      slot = -1;
    }
    std::vector<PhysicalShard> placement = planner_.plan(id, slot < 0 ? 0 : slot);
    if (permanent) {
      for (PhysicalShard& shard : placement) shard.slot_id = -1;
    }
    LogicalExpert expert(id);
    expert.state = LifecycleState::Scheduled;
    expert.shards = std::move(placement);
    expert.last_used = now_us;
    if (permanent) {
      expert.admission_state = AdmissionState::Cached;
      expert.protected_ = true;  // never evicted
    } else {
      expert.admission_state = AdmissionState::Probation;
      expert.protected_ = true;  // probation protection (doc section 19)
    }
    experts_.insert_or_assign(id, std::move(expert));
    ++stats_.admissions;
    return true;
  }

  // Layer-major, expert-id-minor ordering: the canonical startup prefill
  // order (layer 0 expert 0, layer 0 expert 1, ..., layer 1 expert 0, ...).
  static std::vector<LogicalExpertId> sequential_expert_ids(int num_layers,
                                                            int experts_per_layer) {
    if (num_layers < 0 || experts_per_layer < 0) {
      throw std::invalid_argument("num_layers and experts_per_layer must be >= 0");
    }
    std::vector<LogicalExpertId> ids;
    ids.reserve(static_cast<size_t>(num_layers) * static_cast<size_t>(experts_per_layer));
    for (int layer = 0; layer < num_layers; ++layer) {
      for (int expert = 0; expert < experts_per_layer; ++expert) {
        ids.push_back(LogicalExpertId{layer, expert});
      }
    }
    return ids;
  }

  // Startup warm start: admits experts in the given order, loading each one
  // through SCHEDULED -> LOADING -> READY so the first routing round already
  // sees resident experts (doc sections 5.2 and 21).
  //
  // Slot semantics: experts in layers < num_gpu_layers are admitted as
  // permanent residents without consuming expert_gpu_slots. The
  // expert_gpu_slots budget is filled sequentially starting from layer
  // num_gpu_layers onwards, and admission stops once the budget is full
  // (freshly prefilled experts are on probation, so nothing is evicted).
  // Returns the number of experts admitted by this call.
  size_t prefill_sequential(const std::vector<LogicalExpertId>& ordered_ids,
                            uint64_t now_us = 0) {
    size_t admitted = 0;
    for (const LogicalExpertId& id : ordered_ids) {
      const LogicalExpert* existing = find(id);
      if (existing != nullptr && is_resident_(*existing)) continue;
      if (!admit(id, now_us)) break;  // capacity reached; probation blocks eviction
      record_h2d(id);
      mark_scheduled(id);
      mark_loading(id);
      for (int tp_rank = 0; tp_rank < topology().tp_size; ++tp_rank) {
        mark_shard_ready(id, tp_rank, now_us);
      }
      ++admitted;
    }
    return admitted;
  }

  // --- Lifecycle transitions (doc section 7) ---

  void mark_scheduled(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    LogicalExpert& expert = require_(id);
    if (expert.state != LifecycleState::Absent && expert.state != LifecycleState::Scheduled) {
      throw std::runtime_error("invalid transition to SCHEDULED");
    }
    expert.state = LifecycleState::Scheduled;
  }

  void mark_loading(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    LogicalExpert& expert = require_(id);
    if (expert.state != LifecycleState::Scheduled) {
      throw std::runtime_error("invalid transition to LOADING");
    }
    expert.state = LifecycleState::Loading;
    expert.in_flight = true;
    expert.logical_ready_dependency = 0;
    for (PhysicalShard& shard : expert.shards) {
      shard.state = ShardState::Loading;
      shard.ready_dependency = 0;
    }
  }

  // Abort a transfer that failed before its device event was published. This
  // is the rollback counterpart of SCHEDULED -> LOADING and prevents a failed
  // H2D request from occupying a global slot forever.
  void cancel_loading(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    LogicalExpert& expert = require_(id);
    if (expert.state != LifecycleState::Scheduled &&
        expert.state != LifecycleState::Loading) {
      throw std::runtime_error("cancel_loading requires SCHEDULED or LOADING");
    }
    expert.state = LifecycleState::Absent;
    expert.in_flight = false;
    expert.refcount.store(0, std::memory_order_release);
    expert.admission_state = AdmissionState::NotAdmitted;
    expert.protected_ = false;
    expert.logical_ready_dependency = 0;
    recycle_slot_locked_(expert);
    for (PhysicalShard& shard : expert.shards) {
      shard.state = ShardState::Absent;
      shard.ready_dependency = 0;
    }
  }

  void bind_shard_dependency(const LogicalExpertId& id, int tp_rank,
                             uint64_t ready_dependency) {
    GecLockGuard lock(mutex_);
    LogicalExpert& expert = require_(id);
    if (tp_rank < 0 || tp_rank >= static_cast<int>(expert.shards.size())) {
      throw std::invalid_argument("tp_rank out of range");
    }
    if (expert.state != LifecycleState::Loading) {
      throw std::runtime_error("shard dependency requires LOADING logical state");
    }
    expert.shards[static_cast<size_t>(tp_rank)].ready_dependency = ready_dependency;
  }

  std::optional<uint64_t> shard_dependency(const LogicalExpertId& id,
                                            int tp_rank) const {
    GecLockGuard lock(mutex_);
    const LogicalExpert& expert = require_(id);
    if (tp_rank < 0 || tp_rank >= static_cast<int>(expert.shards.size())) {
      throw std::invalid_argument("tp_rank out of range");
    }
    const uint64_t dependency =
        expert.shards[static_cast<size_t>(tp_rank)].ready_dependency;
    if (dependency == 0) return std::nullopt;
    return dependency;
  }

  // TP Atomic Ready: a shard becoming READY only flips the Logical Expert to
  // READY when every required shard is READY (doc section 3.3).
  void mark_shard_ready(const LogicalExpertId& id, int tp_rank, uint64_t now_us = 0,
                        uint64_t ready_dependency = 0) {
    GecLockGuard lock(mutex_);
    LogicalExpert& expert = require_(id);
    if (tp_rank < 0 || tp_rank >= static_cast<int>(expert.shards.size())) {
      throw std::invalid_argument("tp_rank out of range");
    }
    if (expert.state != LifecycleState::Loading) {
      throw std::runtime_error("shard ready requires LOADING logical state");
    }
    PhysicalShard& shard = expert.shards[static_cast<size_t>(tp_rank)];
    shard.state = ShardState::Ready;
    shard.ready_dependency = ready_dependency;
    bool all_ready = true;
    for (const PhysicalShard& shard : expert.shards) {
      if (shard.state != ShardState::Ready) { all_ready = false; break; }
    }
    if (all_ready) {
      expert.state = LifecycleState::Ready;
      expert.in_flight = false;
      expert.last_used = now_us;
      expert.logical_ready_dependency = 0;
      for (const PhysicalShard& ready_shard : expert.shards) {
        expert.logical_ready_dependency =
            std::max(expert.logical_ready_dependency, ready_shard.ready_dependency);
      }
    }
  }

  void mark_evicting(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    LogicalExpert& expert = require_(id);
    if (!can_evict(expert)) {
      throw std::runtime_error("eviction gate rejected the expert");
    }
    expert.state = LifecycleState::Evicting;
    for (PhysicalShard& shard : expert.shards) shard.state = ShardState::Evicting;
  }

  void mark_absent(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    if (it == experts_.end()) return;
    if (it->second.state != LifecycleState::Evicting &&
        it->second.state != LifecycleState::Absent) {
      throw std::runtime_error("invalid transition to ABSENT");
    }
    if (it->second.state == LifecycleState::Evicting) {
      recycle_slot_locked_(it->second);
    }
    it->second.state = LifecycleState::Absent;
    it->second.in_flight = false;
    it->second.refcount.store(0);
    it->second.admission_state = AdmissionState::NotAdmitted;
    it->second.protected_ = false;
    it->second.logical_ready_dependency = 0;
    for (PhysicalShard& shard : it->second.shards) {
      shard.state = ShardState::Absent;
      shard.ready_dependency = 0;
    }
  }

  // Evict one complete Logical Expert when the caller is rebalancing the
  // global physical row pool across layers.
  bool evict(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    if (it == experts_.end() || !can_evict(it->second)) return false;
    it->second.state = LifecycleState::Evicting;
    it->second.admission_state = AdmissionState::NotAdmitted;
    recycle_slot_locked_(it->second);
    for (PhysicalShard& shard : it->second.shards) shard.state = ShardState::Absent;
    it->second.state = LifecycleState::Absent;
    it->second.in_flight = false;
    it->second.logical_ready_dependency = 0;
    for (PhysicalShard& shard : it->second.shards) shard.ready_dependency = 0;
    ++stats_.evictions;
    return true;
  }

  // --- Usage / refcount (doc sections 8.3 and 17) ---

  // Acquires a resident READY expert for GPU kernel lifetime. The caller must
  // call release() only after GPU completion.
  bool acquire(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    if (it == experts_.end() || it->second.state != LifecycleState::Ready) return false;
    it->second.refcount.fetch_add(1, std::memory_order_acq_rel);
    return true;
  }

  // Reserves a GPU-use lease before an asynchronous H2D dependency has
  // published READY. The refcount then protects the row across LOADING ->
  // READY and until the compute-stream completion event releases the lease.
  bool acquire_pending(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    if (it == experts_.end() || it->second.state == LifecycleState::Absent ||
        it->second.state == LifecycleState::Evicting) {
      return false;
    }
    it->second.refcount.fetch_add(1, std::memory_order_acq_rel);
    return true;
  }

  // Decrements refcount after GPU completion. Throws on underflow.
  void release(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    LogicalExpert& expert = require_(id);
    int previous = expert.refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (previous <= 0) {
      expert.refcount.store(0);
      throw std::runtime_error("refcount underflow");
    }
  }

  int refcount(const LogicalExpertId& id) const {
    GecLockGuard lock(mutex_);
    return require_(id).refcount.load(std::memory_order_acquire);
  }

  // Records a real use of the expert (GEMM). Promotes probation to cached
  // after config().probation_uses uses.
  void record_use(const LogicalExpertId& id, uint64_t now_us = 0) {
    GecLockGuard lock(mutex_);
    LogicalExpert& expert = require_(id);
    ++expert.use_count;
    expert.last_used = now_us;
    if (expert.admission_state == AdmissionState::Probation &&
        expert.use_count >= static_cast<uint64_t>(config_.probation_uses)) {
      expert.admission_state = AdmissionState::Cached;
      if (id.layer_id >= config_.num_gpu_layers) expert.protected_ = false;
    }
  }

  // Re-H2D tracking: counts H2D transfers for experts that were previously
  // admitted (i.e. evicted and requested again).
  void record_h2d(const LogicalExpertId& id) {
    GecLockGuard lock(mutex_);
    if (!ever_h2d_.insert(id).second) ++stats_.reh2d;
  }

  // --- Queries ---

  // Read-only snapshot of one expert's three dimensions (state / usage /
  // admission). Intended for bindings and diagnostics: the atomic refcount
  // is read under the cache lock and returned as a plain int.
  struct ExpertSnapshot {
    LogicalExpertId id;
    LifecycleState state = LifecycleState::Absent;
    int refcount = 0;
    uint64_t hit_count = 0;
    uint64_t use_count = 0;
    uint64_t last_used = 0;
    uint64_t logical_ready_dependency = 0;
    bool in_flight = false;
    bool protected_ = false;
    AdmissionState admission_state = AdmissionState::NotAdmitted;
  };

  std::optional<ExpertSnapshot> snapshot(const LogicalExpertId& id) const {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    if (it == experts_.end()) return std::nullopt;
    const LogicalExpert& expert = it->second;
    return ExpertSnapshot{expert.id,
                          expert.state,
                          expert.refcount.load(std::memory_order_acquire),
                          expert.hit_count,
                          expert.use_count,
                          expert.last_used,
                          expert.logical_ready_dependency,
                          expert.in_flight,
                          expert.protected_,
                          expert.admission_state};
  }

  // All resident logical expert ids, sorted (layer_id, expert_id). A resident
  // is admitted and not ABSENT; this is the set a GPU routing table should
  // mirror after an admission round.
  std::vector<LogicalExpertId> resident_ids() const {
    GecLockGuard lock(mutex_);
    std::vector<LogicalExpertId> ids;
    for (const auto& pair : experts_) {
      if (is_resident_(pair.second)) ids.push_back(pair.first);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
  }

  const LogicalExpert* find(const LogicalExpertId& id) const {
    GecLockGuard lock(mutex_);
    auto it = experts_.find(id);
    return it == experts_.end() ? nullptr : &it->second;
  }

  size_t resident_size() const {
    GecLockGuard lock(mutex_);
    return resident_size_();
  }

  // Residents that count against expert_gpu_slots (layers >= num_gpu_layers).
  // Permanent layers are extra and never consume the slot budget.
  size_t slot_resident_size() const {
    GecLockGuard lock(mutex_);
    return slot_resident_size_();
  }

 private:
  LogicalExpert& require_(const LogicalExpertId& id) const {
    auto it = experts_.find(id);
    if (it == experts_.end()) {
      throw std::invalid_argument("unknown logical expert");
    }
    return it->second;
  }

  static bool is_resident_(const LogicalExpert& expert) {
    return expert.admission_state != AdmissionState::NotAdmitted &&
           expert.state != LifecycleState::Absent;
  }

  void recycle_slot_locked_(const LogicalExpert& expert) {
    if (expert.shards.empty() || expert.shards.front().slot_id < 0) return;
    reusable_slots_.push_back(expert.shards.front().slot_id);
  }

  size_t resident_size_() const {
    size_t count = 0;
    for (const auto& pair : experts_) {
      if (is_resident_(pair.second)) ++count;
    }
    return count;
  }

  size_t slot_resident_size_() const {
    size_t count = 0;
    for (const auto& pair : experts_) {
      if (is_resident_(pair.second) &&
          pair.first.layer_id >= config_.num_gpu_layers) {
        ++count;
      }
    }
    return count;
  }

  // LRU eviction among eligible experts. Whole-expert (all TP shards) only.
  bool evict_one_locked_() {
    auto victim = experts_.end();
    for (auto it = experts_.begin(); it != experts_.end(); ++it) {
      const LogicalExpert& expert = it->second;
      if (!is_resident_(expert) || !can_evict(expert)) continue;
      if (expert.admission_state == AdmissionState::Probation) continue;  // protected
      bool better = victim == experts_.end() || expert.last_used < victim->second.last_used ||
                    (expert.last_used == victim->second.last_used && expert.id < victim->second.id);
      if (better) {
        victim = it;
      }
    }
    if (victim == experts_.end()) return false;
    victim->second.state = LifecycleState::Evicting;
    victim->second.admission_state = AdmissionState::NotAdmitted;
    for (PhysicalShard& shard : victim->second.shards) shard.state = ShardState::Evicting;
    victim->second.state = LifecycleState::Absent;
    recycle_slot_locked_(victim->second);
    for (PhysicalShard& shard : victim->second.shards) shard.state = ShardState::Absent;
    victim->second.in_flight = false;
    victim->second.logical_ready_dependency = 0;
    for (PhysicalShard& shard : victim->second.shards) shard.ready_dependency = 0;
    ++stats_.evictions;
    return true;
  }

  GecConfig config_;
  PlacementPlanner planner_;
  Stats stats_;
  int next_slot_ = 0;
  std::vector<int> reusable_slots_;
  std::unordered_set<LogicalExpertId, LogicalExpertIdHash> ever_h2d_;
  mutable GecMutex mutex_;
  mutable std::unordered_map<LogicalExpertId, LogicalExpert, LogicalExpertIdHash> experts_;
};

}  // namespace kt::gec
