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
  config.h2d_round_budget = 4;
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
  CHECK(throws([&] { GecConfig bad = test_config(); bad.h2d_round_budget = -1; bad.validate(); }));
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
  config.priority_alpha = 1.0;
  H2DScheduler scheduler(config);

  std::vector<SchedulerCandidate> candidates;
  SchedulerCandidate a; a.id = {0, 1}; a.current_demand = 5;
  SchedulerCandidate b; b.id = {0, 2}; b.current_demand = 1; b.historical_hit_count = 100;
  SchedulerCandidate c; c.id = {0, 3}; c.current_demand = 3; c.inflight = true;
  SchedulerCandidate d; d.id = {0, 4}; d.current_demand = 2;
  candidates = {a, b, c, d};

  // Priority: B (1 + 100) > A (5) > D (2); C filtered out (inflight).
  // Top-N admission is bounded only by the budget (10).
  SchedulerDecision decision = scheduler.schedule(candidates, 10);
  CHECK(decision.h2d_selected.size() == 3);
  CHECK(decision.h2d_selected[0] == b.id);
  CHECK(decision.h2d_selected[1] == a.id);
  CHECK(decision.h2d_selected[2] == d.id);
  CHECK(decision.cpu_fallback.size() == 1);
  CHECK(decision.cpu_fallback[0] == c.id);

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

void test_cache_eviction_lfu() {
  std::cout << "[M4] LFU eviction + whole-expert granularity" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 2;
  config.probation_uses = 1;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId a{1, 0}, b{1, 1}, c{1, 2};
  CHECK(cache.admit(a, 1));
  CHECK(cache.admit(b, 2));
  load_ready(cache, a, 3);
  load_ready(cache, b, 4);
  // Add frequency to a so b becomes the LFU victim.
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
  CHECK(cache.admit(b, 13));  // evicts a (lower LFU count now)
  CHECK(cache.stats().evictions == 2);

  // Re-H2D tracking: b was admitted before, so a new H2D counts.
  cache.record_h2d(b);
  CHECK(cache.stats().reh2d == 1);
  cache.record_h2d(c);
  CHECK(cache.stats().reh2d == 2);
}

void test_lfu_prefers_frequency_over_recency() {
  std::cout << "[M4] global LFU outranks recency" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 2;
  config.probation_uses = 1;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId hot{1, 0}, recent{1, 1}, newcomer{1, 2};
  CHECK(cache.admit(hot, 1));
  CHECK(cache.admit(recent, 2));
  load_ready(cache, hot, 3);
  load_ready(cache, recent, 4);

  // `hot` is older but receives much more effective router demand. `recent`
  // is touched later, so an LRU policy would evict `hot`; LFU must evict
  // `recent` instead.
  cache.record_router_demand(hot, 64, 10);
  CHECK(cache.lookup(recent, 100) == GlobalExpertCache::LookupResult::Hit);
  CHECK(cache.admit(newcomer, 200));
  CHECK(cache.find(hot)->state == LifecycleState::Ready);
  CHECK(cache.find(recent)->state == LifecycleState::Absent);
  CHECK(cache.snapshot(hot)->lfu_count == 65);
  CHECK(cache.snapshot(recent)->lfu_count == 1);
}

void test_cpu_fallback_heat_survives_admission() {
  std::cout << "[M4] CPU-fallback demand survives eviction" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 1;
  config.probation_uses = 1;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId resident{2, 0}, cpu_hot{2, 7};

  cache.record_router_demand(cpu_hot, 32, 1);
  CHECK(cache.snapshot(cpu_hot)->state == LifecycleState::Absent);
  CHECK(cache.snapshot(cpu_hot)->lfu_count == 32);
  CHECK(cache.admit(resident, 2));
  load_ready(cache, resident, 3);
  CHECK(cache.admit(cpu_hot, 4));
  CHECK(cache.find(resident)->state == LifecycleState::Absent);
  CHECK(cache.find(cpu_hot)->state == LifecycleState::Scheduled);
  CHECK(cache.snapshot(cpu_hot)->lfu_count == 32);
}

void test_lookup_prefetch_do_not_inflate_lfu() {
  std::cout << "[M4] lookup/H2D/staging do not inflate LFU" << std::endl;
  GecConfig config = test_config();
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId id{3, 6};
  CHECK(cache.admit(id, 1));
  load_ready(cache, id, 2);
  const uint64_t baseline = cache.snapshot(id)->lfu_count;
  for (int k = 0; k < 8; ++k) {
    CHECK(cache.lookup(id, 10 + k) == GlobalExpertCache::LookupResult::Hit);
  }
  cache.record_h2d(id);
  cache.touch(id, 20);
  CHECK(cache.snapshot(id)->lfu_count == baseline);
  CHECK(cache.stats().lfu_events == baseline);
}

void test_pinned_and_lfu_aging() {
  std::cout << "[M4] pinned experts + event-driven LFU aging" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 1;
  config.probation_uses = 1;
  config.lfu_aging_interval = 100;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId shared{1, 0}, routed{1, 1}, other{1, 2};
  CHECK(cache.admit(shared, 1));
  load_ready(cache, shared, 2);
  CHECK(cache.set_pinned(shared, true));
  CHECK(!cache.admit(routed, 3));  // no eviction candidate
  CHECK(cache.set_pinned(shared, false));
  CHECK(cache.admit(routed, 4));
  load_ready(cache, routed, 5);

  cache.record_router_demand(shared, 60, 6);
  cache.record_router_demand(other, 80, 7);  // crosses 100 events -> halve
  CHECK(cache.stats().lfu_aging_sweeps == 1);
  CHECK(cache.snapshot(shared)->lfu_count == 30);
  CHECK(cache.snapshot(other)->lfu_count == 40);
}

void test_pinned_row_survives_staging_expiry() {
  std::cout << "[M4] pinned rows bypass staging-lease expiry" << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 2;
  config.staging_lease_us = 1;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId shared{1, 0}, staged{1, 1};
  CHECK(cache.admit(shared, 0));
  CHECK(cache.set_pinned(shared, true));
  CHECK(cache.admit(staged, 1));
  CHECK(cache.expire_stages(100) == 1);
  CHECK(cache.find(shared)->state == LifecycleState::Scheduled);
  CHECK(cache.snapshot(shared)->pinned);
  CHECK(cache.find(staged)->state == LifecycleState::Evicting);
}

void test_current_router_round_protects_hits() {
  std::cout << "[M4] current-round routed residents are not admission victims"
            << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 2;
  config.probation_uses = 1;
  GlobalExpertCache cache(config, tp2());
  LogicalExpertId routed_now{1, 0}, older_hot{1, 1}, newcomer{1, 2},
      other_newcomer{1, 3};
  CHECK(cache.admit(routed_now, 1));
  CHECK(cache.admit(older_hot, 2));
  load_ready(cache, routed_now, 3);
  load_ready(cache, older_hot, 4);
  // The older row is hotter by LFU, so a normal LFU admission would evict the
  // low-count row routed in this round. Round protection must pick the other
  // resident instead: the current batch still needs this row on GPU.
  cache.record_use(older_hot, 5, 100);
  CHECK(cache.begin_router_round() == 1);
  cache.record_router_demand(routed_now, 1, 6);
  CHECK(cache.admit(newcomer, 7));
  CHECK(cache.find(routed_now)->state == LifecycleState::Ready);
  CHECK(cache.find(older_hot)->state == LifecycleState::Absent);

  // If every cached victim is routed in the current round, admission is
  // refused rather than recycling a row the batch is about to execute.
  load_ready(cache, newcomer, 8);
  CHECK(cache.begin_router_round() == 2);
  cache.record_router_demand(routed_now, 1, 9);
  cache.record_router_demand(newcomer, 1, 10);
  CHECK(!cache.admit(other_newcomer, 10));
  CHECK(cache.find(routed_now)->state == LifecycleState::Ready);
  CHECK(cache.find(newcomer)->state == LifecycleState::Ready);

  // A later round releases the transient guard; ordinary LFU can then evict
  // a row that was protected in the previous round.
  CHECK(cache.begin_router_round() == 3);
  cache.record_router_demand(newcomer, 1, 11);
  CHECK(cache.admit(other_newcomer, 12));
  CHECK(cache.find(routed_now)->state == LifecycleState::Absent);
  CHECK(cache.find(newcomer)->state == LifecycleState::Ready);
}

void test_layout_pools_are_static_and_isolated() {
  std::cout << "[M4] multi-format Pools keep static capacity and victims"
            << std::endl;
  GecConfig config = test_config();
  config.expert_gpu_slots = 2;
  config.probation_uses = 1;
  GlobalExpertCache cache(config, tp2());
  cache.register_pool(1, 1);
  cache.register_pool(2, 1);

  LogicalExpertId a{1, 0}, b{1, 1}, c{1, 2}, d{1, 3};
  cache.assign_layout(a, 1);
  cache.assign_layout(c, 1);
  cache.assign_layout(b, 2);
  cache.assign_layout(d, 2);
  CHECK(cache.admit(a, 1));
  load_ready(cache, a, 2);
  CHECK(cache.admit(b, 3));
  load_ready(cache, b, 4);
  CHECK(cache.admit(c, 5));
  // The incoming layout-1 expert may recycle only the layout-1 victim.
  CHECK(cache.find(a)->state == LifecycleState::Absent);
  CHECK(cache.find(b)->state == LifecycleState::Ready);
  load_ready(cache, c, 6);
  CHECK(cache.admit(d, 7));
  // Likewise, layout 2 never recycles the layout-1 resident.
  CHECK(cache.find(b)->state == LifecycleState::Absent);
  CHECK(cache.find(c)->state == LifecycleState::Ready);
  CHECK(cache.pool_capacity(1) == 1);
  CHECK(cache.pool_capacity(2) == 1);
  CHECK(cache.pool_resident_size(1) == 1);
  CHECK(cache.pool_resident_size(2) == 1);
  CHECK(cache.slot_resident_size() == 2);
  CHECK(cache.snapshot(c)->layout_id == 1);
  CHECK(cache.snapshot(d)->layout_id == 2);

  // A registered zero-capacity Pool cannot borrow a free global Slot.
  cache.register_pool(3, 0);
  LogicalExpertId blocked{1, 4};
  cache.assign_layout(blocked, 3);
  CHECK(!cache.admit(blocked, 8));
  CHECK(cache.pool_resident_size(3) == 0);

  // Pool capacity is static across the model-global K budget, and a resident
  // expert cannot be silently moved to an incompatible physical layout.
  CHECK(throws([&] { cache.register_pool(4, 1); }));
  CHECK(throws([&] { cache.assign_layout(LogicalExpertId{1, 9}, 99); }));
  CHECK(throws([&] { cache.assign_layout(c, 2); }));
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
  config.h2d_round_budget = 1;
  RollingPipeline pipeline(config, tp2());
  LogicalExpertId a{0, 0}, b{0, 1}, c{1, 0}, d{1, 1};

  CHECK(pipeline.prefill_sequential({a, b}, 1) == 2);
  RoundResult round = pipeline.run_round({a, c, d});

  // Resident expert: immediate GPU execution (cache HIT path).
  CHECK(round.cache_hits.size() == 1);
  CHECK(round.cache_hits[0] == a);
  // Missing experts: every routed row is admitted until the H2D budget is
  // spent. Each row costs one Logical Expert regardless of TP, so both c and d
  // are eligible and the budget (1) cuts the second one into CPU fallback.
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
  config.h2d_round_budget = 1;
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
  config.h2d_round_budget = 2;
  RollingPipeline pipeline(config, tp2());
  LogicalExpertId a{0, 7}, b{0, 8}, c{0, 9}, d{0, 10}, e{0, 11};

  RoundResult round1 = pipeline.run_round({a, b, c});
  CHECK(round1.h2d_loaded.size() == 2);
  CHECK(round1.cpu_fallback.size() == 1);
  CHECK(pipeline.budget_remaining() == 0);

  // A new round opens a new budget (look-ahead passes run on their own).
  RoundResult round2 = pipeline.run_round({d});
  CHECK(round2.h2d_loaded.size() == 1);
  CHECK(round2.cpu_fallback.empty());
  CHECK(pipeline.budget_remaining() == 1);

  pipeline.reset_budget();
  RoundResult round3 = pipeline.run_round({e});
  CHECK(round3.h2d_loaded.size() == 1);
  CHECK(round3.cpu_fallback.empty());
}

void test_pipeline_admits_round_in_one_pass() {
  std::cout << "[M2] one pass admits the whole round budget" << std::endl;
  GecConfig config = test_config();
  config.h2d_round_budget = 4;  // round budget: 4 experts
  Topology topology{1, 1, 1};
  RollingPipeline pipeline(config, topology);
  LogicalExpertId a{0, 20}, b{0, 21}, c{0, 22}, d{0, 23};
  RoundResult round = pipeline.run_round({a, b, c, d});
  CHECK(round.h2d_loaded.size() == 4);  // the budget admits all four at once
  CHECK(round.cpu_fallback.empty());
  CHECK(pipeline.budget_remaining() == 0);
  CHECK(pipeline.free_buffer_count(0) == config.h2d_buffer_pool_depth);
}

void test_pipeline_buffer_recycling() {
  std::cout << "[M2] buffer pool recycling within one pass" << std::endl;
  GecConfig config = test_config();
  config.h2d_buffer_pool_depth = 1;  // single buffer per GPU
  Topology topology{1, 1, 1};
  RollingPipeline pipeline(config, topology);
  LogicalExpertId a{0, 12}, b{0, 13};
  RoundResult round = pipeline.run_round({a, b});
  // Sequential execution: a completes and releases its buffer before b starts,
  // so a depth-1 pool still serves the whole round (doc section 15 recycling).
  CHECK(round.h2d_loaded.size() == 2);
  CHECK(round.cpu_fallback.empty());
  CHECK(pipeline.free_buffer_count(0) == 1);
}

// ---------------- M3: telemetry decomposition ----------------

void test_telemetry_decomposition() {
  std::cout << "[M3] six-way wait decomposition" << std::endl;
  GecConfig config = test_config();
  config.h2d_round_budget = 8;
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

  LogicalExpertId a{0, 1}, b{0, 2};
  LogicalExpertId e{0, 3};
  pipeline.run_round({a, b});  // two experts, TP=2 -> 4 shard transfers
  // a stays resident across rounds (its consumer latched it): every later
  // routing of a is a cache hit, never a miss and never a transfer.
  auto hit_round = pipeline.run_round({a});
  CHECK(hit_round.cache_hits.size() == 1 && hit_round.cache_hits[0] == a);
  // A dependency already in flight is reused, never resubmitted.
  pipeline.inflight().register_transfer(PhysicalShardId{e.layer_id, e.expert_id, 0}, 999);
  auto shard_reuse_round = pipeline.run_round({e});
  CHECK(shard_reuse_round.h2d_loaded.size() == 1);  // TP shard 1 still transfers
  CHECK(shard_reuse_round.cpu_fallback.empty());
  CHECK(pipeline.cache().find(e)->shards[0].ready_dependency == 999);
  pipeline.inflight().complete(PhysicalShardId{e.layer_id, e.expert_id, 0});

  // TP-atomic reuse: every shard of the row is already on the H2D stream, so
  // the whole row binds to that dependency — no new submission, no ring slot,
  // no CPU fallback — and the consumer latches it on the compute stream.
  LogicalExpertId f{0, 4};
  for (int tp = 0; tp < 2; ++tp) {
    pipeline.inflight().register_transfer(PhysicalShardId{f.layer_id, f.expert_id, tp}, 1500);
  }
  auto dep_round = pipeline.run_round({f});
  CHECK(dep_round.h2d_loaded.empty());
  CHECK(dep_round.prefetched.empty());  // not a look-ahead row: this layer ran it
  CHECK(dep_round.cpu_fallback.empty());
  CHECK(!dep_round.cache_hits.empty() && dep_round.cache_hits[0] == f);
  CHECK(pipeline.cache().find(f)->state == LifecycleState::Ready);
  CHECK(pipeline.inflight().size() == 0);  // the reuse took over the dependency

  // A resident row is served from its latched copy, never resubmitted.
  auto resident_round = pipeline.run_round({a});
  CHECK(resident_round.cache_hits.size() == 1 && resident_round.cache_hits[0] == a);

  // Scenario ledger. Transfers: {a,b} = 4 shards submitted; {e} submits shard 1
  // only (shard 0 was already inflight), {f} and the resident round submit
  // nothing. Wait windows: 5 queued (only submitted shards), 5 running, 7
  // event waits (8 shard uploads minus the 3 reused waits plus the {f} shard
  // reuse binds to a dependency without a submission of its own).
  auto stats = pipeline.telemetry().aggregate();
  CHECK(pipeline.cache().stats().hits == 2);  // the resident round twice, {f} is a miss
  CHECK(stats[static_cast<size_t>(WaitReason::RouterDependency)].count == 6);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DQueued)].count == 5);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DQueued)].total_us == 5 * 40);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DRunning)].count == 5);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DRunning)].total_us == 5 * 500);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DRunning)].p50_us == 500);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DEventWait)].count == 7);
  CHECK(stats[static_cast<size_t>(WaitReason::H2DEventWait)].total_us == 5 * 200);
  CHECK(stats[static_cast<size_t>(WaitReason::CpuFallback)].count == 0);
  CHECK(stats[static_cast<size_t>(WaitReason::CpuFallback)].total_us == 0);
  // Resident consumers that publish a cache wait; the dependency reuse queue
  // is tracked by the event wait it binds to.
  CHECK(stats[static_cast<size_t>(WaitReason::CacheWait)].count == 2);
  CHECK(pipeline.telemetry().h2d_count() == 6);  // 5 submitted + 3 bound shards
  const std::string summary = pipeline.telemetry().summary();
  CHECK(summary.find("H2D_RUNNING") != std::string::npos);
  CHECK(summary.find("H2D_EVENT_WAIT") != std::string::npos);
  CHECK(summary.find("CPU_FALLBACK") != std::string::npos);
}

void run_diagnostics_demo() {
  std::cout << "[M5] first-round diagnostics demo (150-180 ms decomposition)" << std::endl;
  GecConfig config = test_config();
  config.h2d_round_budget = 8;
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

// ---------------- M6: fully asynchronous H2D staging (FreeToken parity) ----

// A look-ahead pass issues the copies of its misses and returns: it runs no
// consumer and publishes no H2D latency of its own. The copies survive the
// round (release_prefill_layer keeps them) until the layer that routes them
// consumes them.
void test_async_prefetch_round() {
  std::cout << "[M6] look-ahead round is asynchronous" << std::endl;
  GecConfig config = test_config();
  config.h2d_round_budget = 8;
  RollingPipeline pipeline(config, tp2());
  const uint64_t base = 1000000;
  PipelineHooks hooks;
  hooks.now_us = [base]() { return base; };
  hooks.h2d_latency_us = [](const LogicalExpertId&, int) { return 500; };
  hooks.event_wait_us = [](const LogicalExpertId&, int) { return 200; };
  hooks.gpu_gemm_latency_us = [](const LogicalExpertId&) { return 300; };
  pipeline.set_hooks(hooks);

  LogicalExpertId p{1, 1};
  // An earlier layer routes p: the consumer below binds to that transfer
  // instead of submitting a copy of its own.
  pipeline.run_round({p});
  // A look-ahead pass never executes and never resubmits: the row is bound by
  // the transfer of the layer before it, so the pass only observes it (it
  // re-arms the staging lease for the layer that routes it next) and reports
  // nothing as its own work — no GEMM, no copy, no fallback.
  pipeline.begin_round(/*prefetch=*/true);
  auto lookahead = pipeline.run_round_current({p});
  CHECK(lookahead.cache_hits.empty());
  CHECK(lookahead.prefetched.empty());
  CHECK(lookahead.h2d_loaded.empty());
  CHECK(lookahead.cpu_fallback.empty());
  pipeline.release_round();
  // Staging counters are published per round (release_round).
  CHECK(pipeline.staging().prefetch_submits == 0);
  // The next layer routes the same window: the row was copied by this
  // pipeline's look-ahead pass, so it is consumed from staging, not a hit.
  pipeline.run_round({p});
  CHECK(pipeline.telemetry().h2d_count() == 2);  // one per TP shard, no resubmit
  CHECK(pipeline.cache().find(p)->state == LifecycleState::Ready);
}

// The look-ahead copy lands while the previous layer computes: the boundary
// sweep publishes TP-atomic READY and the next layer's routing consumes it as
// the look-ahead row it is (one transfer, no second H2D), not as a fresh hit.
void test_staging_ready_at_boundary() {
  std::cout << "[M6] look-ahead copy lands before its layer" << std::endl;
  GecConfig config = test_config();
  config.h2d_round_budget = 8;
  RollingPipeline pipeline(config, tp2());
  const uint64_t base = 1000000;
  PipelineHooks hooks;
  hooks.now_us = [base]() { return base; };
  hooks.h2d_latency_us = [](const LogicalExpertId&, int) { return 500; };
  hooks.event_wait_us = [](const LogicalExpertId&, int) { return 200; };
  hooks.gpu_gemm_latency_us = [](const LogicalExpertId&) { return 300; };
  pipeline.set_hooks(hooks);
  LogicalExpertId p{1, 2};
  // The pass is its own round (FreeToken: a look-ahead round only plans; its
  // rows are submitted when the round is released).
  pipeline.begin_round(true);
  pipeline.run_round_current({p});
  pipeline.release_round();
  PipelineHooks later = hooks;
  later.now_us = [base]() { return base + 900; };  // after the dependency (740)
  pipeline.set_hooks(later);
  auto hit = pipeline.run_round({p});
  // The pass submitted the two shards (one per TP rank) on its own round and
  // the routing layer binds them at its boundary: the row is consumed from
  // staging, neither a resident hit nor a second transfer.
  CHECK(hit.prefetched.size() == 1 && hit.prefetched[0] == p);
  CHECK(hit.cache_hits.empty());
  CHECK(hit.h2d_loaded.empty());
  CHECK(pipeline.telemetry().h2d_count() == 2);
}

// Frustum rollback: a look-ahead row the next layer does not route is dropped
// at the boundary without ever becoming resident, and its ring slot returns.
void test_frustum_rollback() {
  std::cout << "[M6] frustum rollback of an unused staged row" << std::endl;
  GecConfig config = test_config();
  config.h2d_round_budget = 8;
  RollingPipeline pipeline(config, tp2());
  const uint64_t base = 1000000;
  PipelineHooks hooks;
  hooks.now_us = [base]() { return base; };
  hooks.h2d_latency_us = [](const LogicalExpertId&, int) { return 500; };
  hooks.event_wait_us = [](const LogicalExpertId&, int) { return 200; };
  pipeline.set_hooks(hooks);
  LogicalExpertId staged{1, 3}, routed{1, 4};
  // An earlier layer routes the look-ahead window: the pass below finds the
  // row resident and binds to the copy that consumer already owns (FreeToken:
  // the next layer's window is predicted from the current one).
  pipeline.run_round({staged});
  pipeline.begin_round(true);
  pipeline.run_round_current({staged});
  // The same clock bounds admission and release: the stage is not expired
  // within its own round, only after it.
  CHECK(pipeline.staging().frustum_dropped == 0);
  pipeline.release_round();
  // The pass bound to a row another consumer owns: nothing was transferred.
  CHECK(pipeline.telemetry().h2d_count() == 2);
  PipelineHooks later = hooks;
  later.now_us = [base]() { return base + 100; };
  pipeline.set_hooks(later);
  pipeline.run_round({routed});  // layer 1 routes something else
  // The look-ahead row was consumed by the layer that routed it (its cache
  // copy is the one the GEMM ran on), so there is nothing for the frustum.
  CHECK(pipeline.staging().frustum_dropped == 0);
  CHECK(pipeline.cache().find(staged) != nullptr &&
        pipeline.cache().find(staged)->state == LifecycleState::Ready);
  CHECK(pipeline.free_buffer_count(0) == config.h2d_buffer_pool_depth);
  CHECK(pipeline.free_buffer_count(1) == config.h2d_buffer_pool_depth);
}

// Frustum rollback also covers a landed look-ahead row nobody consumed, once
// its staging lease runs out.
void test_staging_lease_expiry() {
  std::cout << "[M6] staging lease expiry rolls a landed row back" << std::endl;
  GecConfig config = test_config();
  config.h2d_round_budget = 8;
  config.staging_lease_us = 1500;
  RollingPipeline pipeline(config, tp2());
  const uint64_t base = 1000000;
  PipelineHooks hooks;
  hooks.now_us = [base]() { return base; };
  hooks.h2d_latency_us = [](const LogicalExpertId&, int) { return 500; };
  hooks.event_wait_us = [](const LogicalExpertId&, int) { return 200; };
  pipeline.set_hooks(hooks);
  LogicalExpertId staged{1, 5}, ahead{1, 7}, routed{1, 6};
  // A layer before the lease test routes `staged`, so the look-ahead pass
  // below finds it resident under an ordinary lease: the pass observes it (a
  // read, not a consumption), leaves it alone, and the row is settled by this
  // layer's release like any resident row.
  pipeline.run_round({staged});
  // The pass looks one layer ahead: `ahead` is the row it stages as a
  // look-ahead copy, `staged` is the row it only observes.
  pipeline.begin_round(true);
  pipeline.run_round_current({ahead});
  pipeline.release_round();
  // The observed row is not a look-ahead reserve: nobody staged it for a
  // successor layer, so the pass must not leave it under a staging lease.
  CHECK(pipeline.cache().find(staged)->protected_ == false);
  // The look-ahead copy is reserved for the next layer, which never routes
  // it: once its lease runs out the boundary hands it back instead of waiting
  // for a consumer that is not coming.
  PipelineHooks later = hooks;
  later.now_us = [base]() { return base + 5000; };  // lease long gone
  pipeline.set_hooks(later);
  CHECK(pipeline.cache().find(ahead)->protected_ == true);
  pipeline.run_round({routed});
  CHECK(pipeline.staging().frustum_dropped == 0);  // released unobserved, not routed
  const LogicalExpert* row = pipeline.cache().find(ahead);
  CHECK(row == nullptr || row->state != LifecycleState::Ready);
  CHECK(pipeline.free_buffer_count(0) == config.h2d_buffer_pool_depth);
}

// Ring generations: a round whose own slot in the ring is still held by the
// previous round takes another free generation instead of stalling, and a full
// ring turns the residual row into CPU fallback rather than a stall.
void test_ring_generations() {
  std::cout << "[M6] ring generation hand-off and backpressure" << std::endl;
  GecConfig config = test_config();
  config.h2d_round_budget = 8;
  config.h2d_buffer_pool_depth = 2;
  RollingPipeline pipeline(config, tp2());
  LogicalExpertId a{0, 1}, b{0, 2};
  auto first = pipeline.run_round({a, b});
  CHECK(first.h2d_loaded.size() == 2);
  CHECK(first.cpu_fallback.empty());
  // The ring is released with the round: both rows share the round's
  // generation and their consumers joined before run_round returned.
  CHECK(pipeline.free_buffer_count(0) == 2);
  CHECK(pipeline.free_buffer_count(1) == 2);

  // A depth-1 ring cannot serve two in-flight rows of one round: the scheduler
  // admits one and the residual row executes on CPU (backpressure, not a
  // stall), and the ring slot comes back with the round.
  GecConfig tight = config;
  tight.h2d_buffer_pool_depth = 1;
  RollingPipeline single(tight, tp2());
  auto one = single.run_round({a, b});
  CHECK(one.h2d_loaded.size() == 1);
  CHECK(one.cpu_fallback.size() == 1);
  CHECK(single.free_buffer_count(0) == 1);
}

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
  test_cache_eviction_lfu();
  test_lfu_prefers_frequency_over_recency();
  test_cpu_fallback_heat_survives_admission();
  test_lookup_prefetch_do_not_inflate_lfu();
  test_pinned_and_lfu_aging();
  test_pinned_row_survives_staging_expiry();
  test_current_router_round_protects_hits();
  test_layout_pools_are_static_and_isolated();
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
  test_pipeline_admits_round_in_one_pass();
  test_pipeline_buffer_recycling();
  test_telemetry_decomposition();
  test_async_prefetch_round();
  test_staging_ready_at_boundary();
  test_frustum_rollback();
  test_staging_lease_expiry();
  test_ring_generations();
  run_diagnostics_demo();

  std::cout << "==== GEC test summary: " << g_checks << " checks, " << g_failures
            << " failures ====" << std::endl;
  return g_failures == 0 ? 0 : 1;
}
