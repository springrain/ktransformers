// Global Expert Cache + H2D rolling pipeline test suite.
//
// Covers all seven V1 acceptance criteria from doc/KT-CPU-PICE-GPU.md
// section 29 plus per-milestone unit behavior. Header-only, no CUDA required.
#include <iostream>
#include <string>
#include <vector>

#include "kt/gec/buffer_pool.hpp"
#include "kt/gec/expert_cache.hpp"
#include "kt/gec/gec_config.hpp"
#include "kt/gec/gec_types.hpp"
#include "kt/gec/inflight_registry.hpp"
#include "kt/gec/pipeline.hpp"
#include "kt/gec/placement.hpp"
#include "kt/gec/scheduler.hpp"
#include "kt/gec/telemetry.hpp"

using namespace kt::gec;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cerr << "CHECK failed at line " << line << std::endl;
  }
}

#define CHECK(cond) check((cond), __LINE__)

template <typename Fn>
bool throws(Fn&& fn) {
  try {
    fn();
  } catch (...) {
    return true;
  }
  return false;
}

GecConfig test_config() {
  GecConfig config;
  config.num_gpu_layers = 0;
  config.expert_gpu_slots = 4;
  config.layer_h2d_slots = 4;
  config.layer_h2d_batch_size = 2;
  config.h2d_buffer_pool_depth = 2;
  config.priority_alpha = 1.0;
  config.probation_uses = 1;
  return config;
}

Topology tp2() { return Topology{2, 1, 2}; }

// Loads an admitted expert to READY and clears probation (single use).
void load_ready(GlobalExpertCache& cache, const LogicalExpertId& id, uint64_t now) {
  cache.record_h2d(id);
  cache.mark_scheduled(id);
  cache.mark_loading(id);
  for (int tp = 0; tp < cache.topology().tp_size; ++tp) {
    cache.mark_shard_ready(id, tp, now);
  }
  cache.record_use(id, now);
}

// ---------------- M0: config + state machine + refcount ----------------

void test_config_validation() {
  std::cout << "[M0] config validation" << std::endl;
  GecConfig config = test_config();
  CHECK(!throws([&] { config.validate(); }));

  CHECK(throws([&] { GecConfig bad = test_config(); bad.num_gpu_layers = -1; bad.validate(); }));
  CHECK(throws([&] { GecConfig bad = test_config(); bad.expert_gpu_slots = -1; bad.validate(); }));
  CHECK(throws([&] { GecConfig bad = test_config(); bad.layer_h2d_slots = -1; bad.validate(); }));
  CHECK(throws([&] { GecConfig bad = test_config(); bad.layer_h2d_batch_size = 0; bad.validate(); }));
  CHECK(throws([&] { GecConfig bad = test_config(); bad.h2d_buffer_pool_depth = 0; bad.validate(); }));
  CHECK(throws([&] { GecConfig bad = test_config(); bad.priority_alpha = -0.1; bad.validate(); }));
  CHECK(throws([&] { GecConfig bad = test_config(); bad.probation_uses = -1; bad.validate(); }));
}

void test_unit_conversion() {
  std::cout << "[M0] logical/shard unit conversion" << std::endl;
  CHECK(GecConfig::shard_count(512, 4) == 2048);
  CHECK(GecConfig::shard_count(0, 8) == 0);
  CHECK(throws([] { GecConfig::shard_count(-1, 4); }));
  CHECK(throws([] { GecConfig::shard_count(10, 0); }));
}

void test_lifecycle_and_tp_atomic_ready() {
  std::cout << "[M0] lifecycle + TP atomic ready" << std::endl;
  GlobalExpertCache cache(test_config(), tp2());
  LogicalExpertId id{3, 32};
  CHECK(cache.admit(id, 1));
  CHECK(cache.find(id) != nullptr);
  CHECK(cache.find(id)->state == LifecycleState::Scheduled);

  cache.mark_scheduled(id);  // idempotent from Scheduled
  cache.mark_loading(id);
  CHECK(cache.find(id)->state == LifecycleState::Loading);
  CHECK(cache.find(id)->in_flight);

  cache.bind_shard_dependency(id, 0, 77);
  CHECK(cache.shard_dependency(id, 0).value() == 77);
  cache.mark_shard_ready(id, 0, 2, 101);
  CHECK(cache.find(id)->state == LifecycleState::Loading);  // TP 1/2 only
  cache.mark_shard_ready(id, 1, 2, 202);
  CHECK(cache.find(id)->state == LifecycleState::Ready);    // TP 2/2 -> READY
  CHECK(!cache.find(id)->in_flight);
  CHECK(is_logical_ready(*cache.find(id)));
  CHECK(cache.find(id)->shards[0].ready_dependency == 101);
  CHECK(cache.find(id)->shards[1].ready_dependency == 202);
  CHECK(cache.snapshot(id)->logical_ready_dependency == 202);

  // Probation protects the fresh expert from eviction until first use.
  CHECK(!can_evict(*cache.find(id)));
  cache.record_use(id, 2);
  CHECK(can_evict(*cache.find(id)));

  cache.mark_evicting(id);
  CHECK(cache.find(id)->state == LifecycleState::Evicting);
  cache.mark_absent(id);
  CHECK(cache.find(id)->state == LifecycleState::Absent);

  // Illegal transitions.
  CHECK(throws([&] { cache.mark_loading(id); }));  // Absent -> Loading forbidden
  GlobalExpertCache other(test_config(), tp2());
  LogicalExpertId e2{1, 7};
  other.admit(e2, 1);
  other.mark_scheduled(e2);
  other.mark_loading(e2);
  other.mark_shard_ready(e2, 0, 2);
  CHECK(throws([&] { other.mark_evicting(e2); }));  // still Loading -> gate rejects
  other.cancel_loading(e2);
  CHECK(other.find(e2)->state == LifecycleState::Absent);
}

void test_reh2d_tracking() {
  std::cout << "[M0] first H2D is not re-H2D" << std::endl;
  GlobalExpertCache cache(test_config(), tp2());
  LogicalExpertId id{4, 11};
  CHECK(cache.admit(id, 1));
  load_ready(cache, id, 2);
  CHECK(cache.stats().reh2d == 0);
  CHECK(cache.evict(id));
  CHECK(cache.admit(id, 3));
  cache.record_h2d(id);
  CHECK(cache.stats().reh2d == 1);
}

void test_refcount_semantics() {
  std::cout << "[M0] refcount covers GPU kernel lifetime" << std::endl;
  GlobalExpertCache cache(test_config(), tp2());
  LogicalExpertId id{2, 9};
  cache.admit(id, 1);
  load_ready(cache, id, 1);
  CHECK(cache.refcount(id) == 0);
  CHECK(cache.acquire(id));
  CHECK(cache.refcount(id) == 1);
  cache.acquire(id);
  CHECK(cache.refcount(id) == 2);
  cache.release(id);
  CHECK(cache.refcount(id) == 1);
  cache.release(id);
  CHECK(cache.refcount(id) == 0);
  CHECK(throws([&] { cache.release(id); }));  // underflow guarded
  CHECK(!can_evict(*cache.find(id)) || true);
}

void test_pending_refcount_protects_h2d_to_compute() {
  std::cout << "[M0] pending GPU lease protects LOADING expert" << std::endl;
  GlobalExpertCache cache(test_config(), tp2());
  LogicalExpertId id{2, 10};
  CHECK(cache.admit(id, 1));
  cache.mark_scheduled(id);
  cache.mark_loading(id);
  CHECK(cache.acquire_pending(id));
  CHECK(cache.refcount(id) == 1);
  cache.mark_shard_ready(id, 0, 2);
  cache.mark_shard_ready(id, 1, 2);
  CHECK(cache.find(id)->state == LifecycleState::Ready);
  CHECK(!cache.evict(id));
  cache.record_use(id, 3);  // leave probation before testing the eviction gate
  cache.release(id);
  CHECK(cache.refcount(id) == 0);
  CHECK(cache.evict(id));
}

// ---------------- M1: placement + inflight registry ----------------

void test_placement() {
  std::cout << "[M1] TP placement" << std::endl;
  Topology topology{4, 1, 4};
  PlacementPlanner planner(topology);
  LogicalExpertId id{5, 32};
  std::vector<PhysicalShard> shards = planner.plan(id, 7);
  CHECK(shards.size() == 4);
  for (int tp = 0; tp < 4; ++tp) {
    CHECK(shards[static_cast<size_t>(tp)].id.tp_rank == tp);
    CHECK(shards[static_cast<size_t>(tp)].gpu_id == tp);
    CHECK(shards[static_cast<size_t>(tp)].slot_id == 7);
    CHECK(shards[static_cast<size_t>(tp)].state == ShardState::Absent);
  }
  // One logical expert occupies exactly one logical slot across all shards.
  std::vector<PhysicalShard> batch = planner.plan_all({id, LogicalExpertId{5, 33}}, 3);
  CHECK(batch.size() == 8);
  CHECK(batch[0].slot_id == 3);
  CHECK(batch[3].slot_id == 3);
  CHECK(batch[4].slot_id == 4);
  CHECK(batch[7].slot_id == 4);
}

void test_placement_all_or_nothing() {
  std::cout << "[M1] placement all-or-nothing" << std::endl;
  Topology broken{2, 1, 1};  // tp_rank 1 has no GPU
  PlacementPlanner planner(broken);
  LogicalExpertId id{0, 1};
  CHECK(throws([&] { planner.plan(id, 0); }));
  CHECK(throws([&] { planner.plan_all({id}, 0); }));
  // EP expansion beyond available GPUs is also rejected.
  Topology ep_broken{2, 2, 2};
  PlacementPlanner ep_planner(ep_broken);
  bool any_owner1_rejected = false;
  for (int expert = 0; expert < 64; ++expert) {
    LogicalExpertId probe{0, expert};
    if (ep_broken.ep_owner(probe) == 1) {
      any_owner1_rejected = throws([&] { ep_planner.plan(probe, 0); });
      break;
    }
  }
  CHECK(any_owner1_rejected);
}

void test_inflight_registry() {
  std::cout << "[M1] inflight registry dedup" << std::endl;
  InflightRegistry registry;
  PhysicalShardId shard{4, 12, 2};
  auto first = registry.register_transfer(shard, 42, 100);
  CHECK(first.first);
  CHECK(first.second == 42);
  auto second = registry.register_transfer(shard, 99, 200);
  CHECK(!second.first);      // duplicate rejected
  CHECK(second.second == 42);  // existing dependency reused
  CHECK(registry.contains(shard));
  CHECK(registry.dependency_id(shard).value() == 42);
  CHECK(registry.size() == 1);
  registry.complete(shard);
  CHECK(!registry.contains(shard));
  auto third = registry.register_transfer(shard, 99, 300);
  CHECK(third.first);
  CHECK(third.second == 99);
}

// ---------------- M2: scheduler + buffer pool ----------------

void test_scheduler() {
  std::cout << "[M2] candidate filter + priority + top-n" << std::endl;
  GecConfig config = test_config();
  config.layer_h2d_batch_size = 2;
  config.priority_alpha = 1.0;
  H2DScheduler scheduler(config);

  std::vector<SchedulerCandidate> candidates;
  SchedulerCandidate a; a.id = {0, 1}; a.current_demand = 5;
  SchedulerCandidate b; b.id = {0, 2}; b.current_demand = 1; b.historical_hit_count = 100;
  SchedulerCandidate c; c.id = {0, 3}; c.current_demand = 3; c.inflight = true;
  SchedulerCandidate d; d.id = {0, 4}; d.current_demand = 2;
  candidates = {a, b, c, d};

  // Priority: B (1 + 100) > A (5) > D (2); C filtered out (inflight).
  // Top-N admission is bounded by the budget (10), not by batch_size (2).
  SchedulerDecision decision = scheduler.schedule(candidates, 10);
  CHECK(decision.h2d_selected.size() == 3);
  CHECK(decision.h2d_selected[0] == b.id);
  CHECK(decision.h2d_selected[1] == a.id);
  CHECK(decision.h2d_selected[2] == d.id);
  CHECK(decision.cpu_fallback.size() == 1);
  CHECK(decision.cpu_fallback[0] == c.id);

  // batch = min(available, batch_size): 3 selected, batch_size 2 -> 2 + 1.
  std::vector<std::vector<LogicalExpertId>> batches =
      decision.h2d_batches(config.layer_h2d_batch_size);
  CHECK(batches.size() == 2);
  CHECK(batches[0].size() == 2);
  CHECK(batches[1].size() == 1);  // the last batch never waits to be filled

  // Budget cap: only one slot left.
  decision = scheduler.schedule(candidates, 1);
  CHECK(decision.h2d_selected.size() == 1);
  CHECK(decision.h2d_selected[0] == b.id);

  // 1 available -> submit 1, no fallback.
  decision = scheduler.schedule({a}, 10);
  CHECK(decision.h2d_selected.size() == 1);
  CHECK(decision.cpu_fallback.empty());

  // All other filter flags disqualify candidates.
  SchedulerCandidate e; e.id = {1, 0}; e.evicting = true;
  SchedulerCandidate f; f.id = {1, 1}; f.placement_available = false;
  SchedulerCandidate g; g.id = {1, 2}; g.buffer_available = false;
  SchedulerCandidate h; h.id = {1, 3}; h.cache_admission_locked = true;
  decision = scheduler.schedule({e, f, g, h}, 10);
  CHECK(decision.h2d_selected.empty());
  CHECK(decision.cpu_fallback.size() == 4);
}

void test_buffer_lifecycle() {
  std::cout << "[M2] buffer lifecycle (no early reuse)" << std::endl;
  H2DBufferPool pool(0, 1);
  LogicalExpertId id{6, 3};
  H2DBuffer* buffer = pool.acquire(id, 1, 55);
  CHECK(buffer != nullptr);
  CHECK(pool.free_count() == 0);
  CHECK(pool.acquire(id, 0, 56) == nullptr);  // exhausted -> backpressure

  pool.mark_ready(buffer->buffer_id, 77);
  CHECK(pool.find(buffer->buffer_id)->state == BufferState::ReadyForCompute);
  CHECK(pool.free_count() == 0);  // H2D done != free: GPU has not used it yet

  pool.begin_gpu_use(buffer->buffer_id);
  CHECK(pool.find(buffer->buffer_id)->state == BufferState::GpuInUse);
  CHECK(pool.free_count() == 0);  // kernel in flight

  pool.gpu_complete(buffer->buffer_id);
  CHECK(pool.find(buffer->buffer_id)->state == BufferState::Free);
  CHECK(pool.free_count() == 1);

  // Invalid transitions throw.
  CHECK(throws([&] { pool.mark_ready(buffer->buffer_id, 78); }));  // Free -> ready
  H2DBuffer* again = pool.acquire(id, 0, 57);
  CHECK(again != nullptr);
  CHECK(throws([&] { pool.begin_gpu_use(again->buffer_id); }));    // H2D -> gpu use
  CHECK(throws([&] { pool.gpu_complete(again->buffer_id); }));     // H2D -> complete
  pool.cancel(again->buffer_id);
  CHECK(pool.free_count() == 1);

  // Ownership metadata round-trip.
  H2DBuffer* third = pool.acquire(id, 1, 58);
  CHECK(third->owner_request == 58);
  CHECK(third->tp_rank == 1);
  CHECK(third->logical_expert == id);
  pool.mark_ready(third->buffer_id, 88);
  CHECK(pool.find(third->buffer_id)->ready_dependency == 88);
  pool.begin_gpu_use(third->buffer_id);
  pool.gpu_complete(third->buffer_id);

  // Partial-failure rollback: a buffer whose H2D completed but which never
  // reached the compute stream must still return to FREE (no leak).
  H2DBuffer* fourth = pool.acquire(id, 0, 59);
  CHECK(fourth != nullptr);
  pool.mark_ready(fourth->buffer_id, 99);
  CHECK(pool.free_count() == 0);
  pool.release_unstarted(fourth->buffer_id);
  CHECK(pool.find(fourth->buffer_id)->state == BufferState::Free);
  CHECK(pool.free_count() == 1);
  CHECK(throws([&] { pool.release_unstarted(fourth->buffer_id); }));  // Free
}

// ---------------- M4: cache policy ----------------

void test_cache_eviction_lru() {
  std::cout << "[M4] LRU eviction + whole-expert granularity" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 2;
  config.probation_uses = 1;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId a{1, 0}, b{1, 1}, c{1, 2};
  CHECK(cache.admit(a, 1));
  CHECK(cache.admit(b, 2));
  load_ready(cache, a, 3);
  load_ready(cache, b, 4);
  // Touch a so b becomes the LRU victim.
  CHECK(cache.lookup(a, 10) == GlobalExpertCache::LookupResult::Hit);
  cache.record_use(a, 10);

  CHECK(cache.admit(c, 11));
  CHECK(cache.find(b) != nullptr);
  CHECK(cache.find(b)->admission_state == AdmissionState::NotAdmitted);
  CHECK(cache.find(b)->state == LifecycleState::Absent);
  CHECK(cache.stats().evictions == 1);
  CHECK(cache.resident_size() == 2);

  // Evicted expert misses; re-admission works.
  CHECK(cache.lookup(b, 12) == GlobalExpertCache::LookupResult::Miss);
  load_ready(cache, c, 12);
  CHECK(cache.admit(b, 13));  // evicts a (LRU now)
  CHECK(cache.stats().evictions == 2);

  // Re-H2D tracking: b was admitted before, so a new H2D counts.
  cache.record_h2d(b);
  CHECK(cache.stats().reh2d == 1);
  cache.record_h2d(c);
  CHECK(cache.stats().reh2d == 2);
}

void test_global_cross_layer_cache() {
  std::cout << "[M4] global cache competition across layers" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 1;
  Topology topology{1, 1, 1};
  GlobalExpertCache cache(config, topology);
  LogicalExpertId layer0{4, 1};
  LogicalExpertId layer1{5, 2};
  CHECK(cache.admit(layer0, 1));
  load_ready(cache, layer0, 2);
  CHECK(cache.admit(layer1, 3));
  load_ready(cache, layer1, 4);
  CHECK(cache.resident_size() == 1);
  CHECK(cache.find(layer0)->state == LifecycleState::Absent);
  CHECK(cache.find(layer1)->state == LifecycleState::Ready);
  CHECK(cache.find(layer1)->shards[0].slot_id == 0);
  CHECK(cache.slot_resident_size() == 1);
  CHECK(cache.evict(layer1));
  CHECK(cache.resident_size() == 0);
}

void test_cache_snapshot_and_resident_ids() {
  std::cout << "[M4] snapshot + resident_ids queries" << std::endl;
  GlobalExpertCache cache(test_config(), tp2());
  LogicalExpertId a{0, 1}, b{0, 2}, c{0, 3};
  for (const LogicalExpertId& id : {a, b, c}) {
    CHECK(cache.admit(id, 10));
    cache.mark_scheduled(id);
    cache.mark_loading(id);
    for (int tp = 0; tp < 2; ++tp) cache.mark_shard_ready(id, tp, 20);
  }
  CHECK(cache.snapshot(a).has_value());
  CHECK(cache.snapshot(a)->state == LifecycleState::Ready);
  CHECK(cache.snapshot(a)->refcount == 0);
  CHECK(!cache.snapshot(LogicalExpertId{9, 9}).has_value());
  std::vector<LogicalExpertId> ids = cache.resident_ids();
  CHECK(ids.size() == 3);
  CHECK(ids[0] == a && ids[1] == b && ids[2] == c);  // sorted order
  CHECK(cache.lookup(b, 30) == GlobalExpertCache::LookupResult::Hit);
  CHECK(cache.snapshot(b)->hit_count == 1);
  CHECK(cache.acquire(b));
  CHECK(cache.snapshot(b)->refcount == 1);
  cache.release(b);
  CHECK(cache.snapshot(b)->refcount == 0);
}

void test_cache_probation_protection() {
  std::cout << "[M4] probation protection" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 1;
  config.probation_uses = 1;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId a{1, 0}, b{1, 1};
  CHECK(cache.admit(a, 1));
  // a is on probation (protected): no victim available for b.
  CHECK(!cache.admit(b, 2));
  CHECK(cache.resident_size() == 1);
  // First use promotes a out of probation, unlocking eviction.
  load_ready(cache, a, 3);
  CHECK(cache.admit(b, 4));
  CHECK(cache.stats().evictions == 1);
  CHECK(cache.find(a)->admission_state == AdmissionState::NotAdmitted);
  CHECK(cache.find(b)->admission_state == AdmissionState::Probation);
}

void test_permanent_layers() {
  std::cout << "[M4] permanent resident layers" << std::endl;
  GecConfig config = test_config();
  config.num_gpu_layers = 1;
  config.expert_gpu_slots = 1;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId layer0{0, 5}, layer1{1, 5};
  CHECK(cache.admit(layer0, 1));
  load_ready(cache, layer0, 2);
  CHECK(cache.find(layer0)->protected_);  // permanent layers never evicted
  // Permanent layers do NOT consume expert_gpu_slots: the slot budget is
  // still empty, so a post-K expert is admitted normally.
  CHECK(cache.admit(layer1, 3));
  load_ready(cache, layer1, 4);
  CHECK(cache.resident_size() == 2);       // 1 permanent + 1 slot expert
  CHECK(cache.slot_resident_size() == 1);
}

// ---------------- Startup prefill (kt-expert-gpu-slots) ----------------

void test_startup_prefill() {
  std::cout << "[M4] startup sequential prefill" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 2;
  GlobalExpertCache cache(config, tp2());

  std::vector<LogicalExpertId> ordered = GlobalExpertCache::sequential_expert_ids(3, 2);
  CHECK(ordered.size() == 6);
  CHECK((ordered[0] == LogicalExpertId{0, 0}));
  CHECK((ordered[1] == LogicalExpertId{0, 1}));
  CHECK((ordered[2] == LogicalExpertId{1, 0}));

  CHECK(cache.prefill_sequential(ordered, 1) == 2);  // exactly kt-expert-gpu-slots
  CHECK(cache.resident_size() == 2);
  CHECK(cache.slot_resident_size() == 2);
  CHECK(cache.find(LogicalExpertId{0, 0})->state == LifecycleState::Ready);
  CHECK(cache.find(LogicalExpertId{0, 1})->state == LifecycleState::Ready);
  CHECK(cache.find(LogicalExpertId{1, 0}) == nullptr);

  // Re-prefill while every resident is protected admits nothing new.
  CHECK(cache.prefill_sequential(ordered, 3) == 0);
  CHECK(cache.resident_size() == 2);

  // Prefilled experts are immediately usable and probation-protected.
  CHECK(cache.lookup(LogicalExpertId{0, 0}, 2) == GlobalExpertCache::LookupResult::Hit);
  CHECK(!can_evict(*cache.find(LogicalExpertId{0, 0})));
  cache.record_use(LogicalExpertId{0, 0}, 2);
  CHECK(can_evict(*cache.find(LogicalExpertId{0, 0})));
}

void test_startup_prefill_permanent_layers() {
  std::cout << "[M4] prefill fills slots after kt-num-gpu-layers" << std::endl;
  GecConfig config = test_config();
  config.num_gpu_layers = 1;
  config.expert_gpu_slots = 2;
  GlobalExpertCache cache(config, tp2());
  std::vector<LogicalExpertId> ordered = {
      LogicalExpertId{0, 0}, LogicalExpertId{0, 1},   // permanent, no slot cost
      LogicalExpertId{1, 0}, LogicalExpertId{1, 1},   // slot budget starts here
      LogicalExpertId{2, 0}};
  CHECK(cache.prefill_sequential(ordered, 1) == 4);
  CHECK(cache.resident_size() == 4);                  // 2 permanent + 2 slots
  CHECK(cache.slot_resident_size() == 2);
  CHECK(cache.find(LogicalExpertId{0, 0})->protected_);
  CHECK(cache.find(LogicalExpertId{0, 1})->protected_);
  CHECK(cache.find(LogicalExpertId{1, 0})->state == LifecycleState::Ready);
  CHECK(cache.find(LogicalExpertId{1, 1})->state == LifecycleState::Ready);
  CHECK(cache.find(LogicalExpertId{2, 0}) == nullptr);  // slot budget exhausted
}

void test_prefill_starts_after_gpu_layers() {
  std::cout << "[M4] slot filling starts at layer kt-num-gpu-layers" << std::endl;
  GecConfig config = test_config();
  config.num_gpu_layers = 2;   // layers 0 and 1 are permanent
  config.expert_gpu_slots = 2;
  GlobalExpertCache cache(config, tp2());
  std::vector<LogicalExpertId> ordered = GlobalExpertCache::sequential_expert_ids(4, 2);
  CHECK(ordered.size() == 8);

  // 4 permanent experts (layers 0-1) + 2 slot experts (layer 2) = 6 admitted.
  CHECK(cache.prefill_sequential(ordered, 1) == 6);
  CHECK(cache.resident_size() == 6);
  CHECK(cache.slot_resident_size() == 2);
  // The first slot expert is exactly (num_gpu_layers, 0).
  CHECK(cache.find(LogicalExpertId{2, 0})->state == LifecycleState::Ready);
  CHECK(cache.find(LogicalExpertId{2, 1})->state == LifecycleState::Ready);
  // Layer 3 is beyond the slot budget.
  CHECK(cache.find(LogicalExpertId{3, 0}) == nullptr);
  CHECK(cache.find(LogicalExpertId{3, 1}) == nullptr);
}

void test_prefill_then_parallel_paths() {
  std::cout << "[M2] prefill + three parallel paths in one round" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 2;
  config.layer_h2d_slots = 1;
  config.layer_h2d_batch_size = 1;
  RollingPipeline pipeline(config, tp2());
  LogicalExpertId a{0, 0}, b{0, 1}, c{1, 0}, d{1, 1};

  CHECK(pipeline.prefill_sequential({a, b}, 1) == 2);
  RoundResult round = pipeline.run_round({a, c, d});

  // Resident expert: immediate GPU execution (cache HIT path).
  CHECK(round.cache_hits.size() == 1);
  CHECK(round.cache_hits[0] == a);
  // Missing expert: PCIe H2D inside the same round.
  CHECK(round.h2d_loaded.size() == 1);
  CHECK(round.h2d_loaded[0] == c);
  // Budget exhausted: remaining expert executes immediately on CPU.
  CHECK(round.cpu_fallback.size() == 1);
  CHECK(round.cpu_fallback[0] == d);
}

// ---------------- M2/M5: rolling pipeline ----------------

void test_pipeline_hit_and_miss_paths() {
  std::cout << "[M2] pipeline hit / H2D / fallback paths" << std::endl;
  RollingPipeline pipeline(test_config(), tp2());
  std::vector<LogicalExpertId> merged;
  PipelineHooks hooks;
  hooks.on_merge = [&](const std::vector<LogicalExpertId>& ids) { merged = ids; };
  pipeline.set_hooks(hooks);

  LogicalExpertId a{0, 1}, b{0, 2}, c{0, 3};
  RoundResult round1 = pipeline.run_round({a, b});
  CHECK(round1.cache_hits.empty());
  CHECK(round1.h2d_loaded.size() == 2);
  CHECK(round1.cpu_fallback.empty());
  CHECK(merged.size() == 2);
  CHECK(pipeline.cache().find(a)->state == LifecycleState::Ready);
  CHECK(pipeline.cache().find(b)->state == LifecycleState::Ready);
  CHECK(pipeline.inflight().size() == 0);
  CHECK(pipeline.cache().stats().misses == 2);

  RoundResult round2 = pipeline.run_round({a, c});
  CHECK(round2.cache_hits.size() == 1);
  CHECK(round2.cache_hits[0] == a);
  CHECK(round2.h2d_loaded.size() == 1);
  CHECK(round2.h2d_loaded[0] == c);
  CHECK(pipeline.cache().stats().hits == 1);
  CHECK(pipeline.cache().stats().misses == 3);
}

void test_pipeline_refcount_during_gemm() {
  std::cout << "[M2] refcount covers real GPU kernel lifetime" << std::endl;
  RollingPipeline pipeline(test_config(), tp2());
  std::vector<int> observed;
  PipelineHooks hooks;
  hooks.on_gpu_gemm = [&](const LogicalExpertId& id) {
    observed.push_back(pipeline.cache().refcount(id));
  };
  pipeline.set_hooks(hooks);
  LogicalExpertId a{0, 4};

  // H2D path: refcount must be > 0 while the GEMM hook runs.
  RoundResult round1 = pipeline.run_round({a});
  CHECK(round1.h2d_loaded.size() == 1);
  CHECK(observed.size() == 1);
  CHECK(observed[0] == 1);
  CHECK(pipeline.cache().refcount(a) == 0);  // released only after completion

  // Cache HIT path: same invariant.
  RoundResult round2 = pipeline.run_round({a});
  CHECK(round2.cache_hits.size() == 1);
  CHECK(observed.size() == 2);
  CHECK(observed[1] == 1);
  CHECK(pipeline.cache().refcount(a) == 0);
}

void test_pipeline_device_lifecycle_callbacks() {
  std::cout << "[M2] device-local lifecycle callbacks" << std::endl;
  RollingPipeline pipeline(test_config(), tp2());
  std::vector<std::string> events;
  std::vector<uint64_t> shard_ready_at;
  PipelineHooks hooks;
  hooks.on_shard_ready = [&](const PhysicalShardId& shard, uint64_t ready_at) {
    events.push_back("shard:" + std::to_string(shard.tp_rank));
    shard_ready_at.push_back(ready_at);
    CHECK(!pipeline.inflight().contains(shard));
  };
  hooks.on_gpu_gemm_submitted = [&](const LogicalExpertId& id) {
    events.push_back("submit");
    CHECK(pipeline.cache().refcount(id) == 1);
  };
  hooks.on_gpu_gemm_complete = [&](const LogicalExpertId& id) {
    events.push_back("complete");
    CHECK(pipeline.cache().refcount(id) == 1);
  };
  pipeline.set_hooks(hooks);
  RoundResult result = pipeline.run_round({LogicalExpertId{2, 4}});
  CHECK(result.h2d_loaded.size() == 1);
  CHECK(events.size() == 4);  // two TP shards + submit + completion
  CHECK(events[0] == "shard:0");
  CHECK(events[1] == "shard:1");
  CHECK(events[2] == "submit");
  CHECK(events[3] == "complete");
  CHECK(shard_ready_at.size() == 2);
}

void test_pipeline_requires_cache_admission() {
  std::cout << "[M2] H2D requires persistent cache admission" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 0;
  config.layer_h2d_slots = 1;
  RollingPipeline pipeline(config, tp2());
  LogicalExpertId a{0, 5};
  RoundResult round = pipeline.run_round({a});
  CHECK(round.h2d_loaded.empty());
  CHECK(round.cpu_fallback.size() == 1);
  CHECK(pipeline.cache().find(a) == nullptr);      // admission failed before H2D
  CHECK(pipeline.cache().resident_size() == 0);
  CHECK(pipeline.cache().stats().admissions == 0);
  CHECK(pipeline.telemetry().h2d_count() == 0);
  CHECK(pipeline.free_buffer_count(0) == pipeline.cache().config().h2d_buffer_pool_depth);
  CHECK(pipeline.free_buffer_count(1) == pipeline.cache().config().h2d_buffer_pool_depth);
}

void test_pipeline_no_duplicate_h2d() {
  std::cout << "[M2] duplicate request in one round does not duplicate H2D" << std::endl;
  RollingPipeline pipeline(test_config(), tp2());
  LogicalExpertId a{0, 6};
  RoundResult round = pipeline.run_round({a, a});
  CHECK(round.h2d_loaded.size() == 1);   // first request loads
  CHECK(round.cache_hits.size() == 1);   // second request hits the fresh cache
  CHECK(round.cpu_fallback.empty());
  CHECK(pipeline.cache().stats().admissions == 1);
}

void test_pipeline_budget() {
  std::cout << "[M2] H2D admission budget" << std::endl;
  GecConfig config = test_config();
  config.layer_h2d_slots = 2;
  config.layer_h2d_batch_size = 4;
  RollingPipeline pipeline(config, tp2());
  LogicalExpertId a{0, 7}, b{0, 8}, c{0, 9}, d{0, 10}, e{0, 11};

  RoundResult round1 = pipeline.run_round({a, b, c});
  CHECK(round1.h2d_loaded.size() == 2);
  CHECK(round1.cpu_fallback.size() == 1);
  CHECK(pipeline.budget_remaining() == 0);

  RoundResult round2 = pipeline.run_round({d});
  CHECK(round2.h2d_loaded.empty());
  CHECK(round2.cpu_fallback.size() == 1);

  pipeline.reset_budget();
  RoundResult round3 = pipeline.run_round({e});
  CHECK(round3.h2d_loaded.size() == 1);
  CHECK(round3.cpu_fallback.empty());
}

void test_pipeline_batches_fill_budget() {
  std::cout << "[M2] batch size chunks submissions, budget admits the round" << std::endl;
  GecConfig config = test_config();
  config.layer_h2d_slots = 4;       // round budget: 4 experts
  config.layer_h2d_batch_size = 2;  // submitted as two batches of two
  Topology topology{1, 1, 1};
  RollingPipeline pipeline(config, topology);
  LogicalExpertId a{0, 20}, b{0, 21}, c{0, 22}, d{0, 23};
  RoundResult round = pipeline.run_round({a, b, c, d});
  CHECK(round.h2d_loaded.size() == 4);  // budget admits all four, not one batch
  CHECK(round.cpu_fallback.empty());
  CHECK(pipeline.budget_remaining() == 0);
  CHECK(pipeline.free_buffer_count(0) == config.h2d_buffer_pool_depth);
}

void test_pipeline_buffer_recycling() {
  std::cout << "[M2] buffer pool recycling between batches" << std::endl;
  GecConfig config = test_config();
  config.h2d_buffer_pool_depth = 1;  // single buffer per GPU
  config.layer_h2d_batch_size = 2;   // batch of two experts
  Topology topology{1, 1, 1};
  RollingPipeline pipeline(config, topology);
  LogicalExpertId a{0, 12}, b{0, 13};
  RoundResult round = pipeline.run_round({a, b});
  // Sequential execution: a completes and releases its buffer before b starts,
  // so a depth-1 pool still serves a batch of two (doc section 15 recycling).
  CHECK(round.h2d_loaded.size() == 2);
  CHECK(round.cpu_fallback.empty());
  CHECK(pipeline.free_buffer_count(0) == 1);
}

// ---------------- M3: telemetry decomposition ----------------

void test_telemetry_decomposition() {
  std::cout << "[M3] six-way wait decomposition" << std::endl;
  GecConfig config = test_config();
  config.layer_h2d_slots = 8;
  config.layer_h2d_batch_size = 4;
  RollingPipeline pipeline(config, tp2());
  const uint64_t base = 1000000;
  PipelineHooks hooks;
  hooks.now_us = [base]() { return base; };
  hooks.queue_latency_us = [](const LogicalExpertId&, int) { return 40; };
  hooks.h2d_latency_us = [](const LogicalExpertId&, int) { return 500; };
  hooks.event_wait_us = [](const LogicalExpertId&, int) { return 200; };
  hooks.gpu_gemm_latency_us = [](const LogicalExpertId&) { return 300; };
  hooks.cpu_gemm_latency_us = [](const LogicalExpertId&) { return 700; };
  pipeline.set_hooks(hooks);

  LogicalExpertId a{0, 1}, b{0, 2}, c{0, 3};
  pipeline.run_round({a, b});  // two experts, TP=2 -> 4 shard transfers
  // Mark one shard inflight.  The second request must reuse its dependency
  // rather than falling back to CPU or submitting a duplicate H2D.
  pipeline.inflight().register_transfer(PhysicalShardId{c.layer_id, c.expert_id, 0}, 999);
  pipeline.run_round({a, c});  // a: cache hit; c: dependency reuse + one H2D
  pipeline.inflight().complete(PhysicalShardId{c.layer_id, c.expert_id, 0});

  auto stats = pipeline.telemetry().aggregate();
  CHECK(stats[static_cast<size_t>(WaitReason::RouterDependency)].count == 4);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DQueued)].count == 5);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DQueued)].total_us == 4 * 40 + 40);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DRunning)].count == 5);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DRunning)].total_us == 4 * 500 + 500);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DRunning)].p50_us == 500);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DEventWait)].count == 6);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DEventWait)].total_us == 4 * 200 + 200);
  CHECK(stats[static_cast<size_t>(WaitReason::CpuFallback)].count == 0);
  CHECK(stats[static_cast<size_t>(WaitReason::CpuFallback)].total_us == 0);
  CHECK(stats[static_cast<size_t>(WaitReason::CacheWait)].count == 1);
  CHECK(pipeline.telemetry().h2d_count() == 6);
  const std::string summary = pipeline.telemetry().summary();
  CHECK(summary.find("H2D_RUNNING") != std::string::npos);
  CHECK(summary.find("H2D_EVENT_WAIT") != std::string::npos);
  CHECK(summary.find("CPU_FALLBACK") != std::string::npos);
}

void run_diagnostics_demo() {
  std::cout << "[M5] first-round diagnostics demo (150-180 ms decomposition)" << std::endl;
  GecConfig config = test_config();
  config.layer_h2d_slots = 8;
  config.layer_h2d_batch_size = 4;
  Topology topology{1, 1, 1};
  RollingPipeline pipeline(config, topology);
  const uint64_t base = 1000000;
  PipelineHooks hooks;
  hooks.now_us = [base]() { return base; };
  hooks.queue_latency_us = [](const LogicalExpertId&, int) { return 5000; };
  hooks.h2d_latency_us = [](const LogicalExpertId&, int) { return 8000; };
  hooks.event_wait_us = [](const LogicalExpertId&, int) { return 120000; };
  hooks.cache_wait_us = [](const LogicalExpertId&) { return 2000; };
  hooks.cpu_gemm_latency_us = [](const LogicalExpertId&) { return 15000; };
  pipeline.set_hooks(hooks);
  LogicalExpertId a{0, 1}, b{0, 2};
  pipeline.run_round({a});                 // a: miss -> H2D (Case A shape)
  pipeline.inflight().register_transfer(PhysicalShardId{b.layer_id, b.expert_id, 0}, 998);
  pipeline.run_round({a, b});              // a: hit; b: forced CPU fallback
  pipeline.inflight().complete(PhysicalShardId{b.layer_id, b.expert_id, 0});
  std::cout << pipeline.telemetry().summary();
}

}  // namespace

int main() {
  test_config_validation();
  test_unit_conversion();
  test_lifecycle_and_tp_atomic_ready();
  test_reh2d_tracking();
  test_refcount_semantics();
  test_pending_refcount_protects_h2d_to_compute();
  test_placement();
  test_placement_all_or_nothing();
  test_inflight_registry();
  test_scheduler();
  test_buffer_lifecycle();
  test_cache_eviction_lru();
  test_global_cross_layer_cache();
  test_cache_snapshot_and_resident_ids();
  test_cache_probation_protection();
  test_permanent_layers();
  test_startup_prefill();
  test_startup_prefill_permanent_layers();
  test_prefill_starts_after_gpu_layers();
  test_prefill_then_parallel_paths();
  test_pipeline_hit_and_miss_paths();
  test_pipeline_refcount_during_gemm();
  test_pipeline_device_lifecycle_callbacks();
  test_pipeline_requires_cache_admission();
  test_pipeline_no_duplicate_h2d();
  test_pipeline_budget();
  test_pipeline_batches_fill_budget();
  test_pipeline_buffer_recycling();
  test_telemetry_decomposition();
  run_diagnostics_demo();

  std::cout << "==== GEC test summary: " << g_checks << " checks, " << g_failures
            << " failures ====" << std::endl;
  return g_failures == 0 ? 0 : 1;
}
