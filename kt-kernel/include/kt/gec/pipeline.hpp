// Rolling H2D pipeline orchestration (doc/KT-CPU-PICE-GPU.md sections 14, 21, 22).
//
// Deterministic, dependency-injected simulation harness for the V1 pipeline:
// GPU router -> cache lookup -> HIT path / MISS path (filter, priority, top-N,
// H2D or CPU fallback) -> compute stream -> merge. All CUDA interactions are
// abstracted behind PipelineHooks, so the lifecycle invariants (buffer reuse,
// refcount, inflight dedup, TP atomic ready) can be verified on any host.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kt/gec/buffer_pool.hpp"
#include "kt/gec/expert_cache.hpp"
#include "kt/gec/inflight_registry.hpp"
#include "kt/gec/scheduler.hpp"
#include "kt/gec/telemetry.hpp"

namespace kt::gec {

struct PipelineHooks {
  // Virtual clock (microseconds). Defaults to steady_clock.
  std::function<uint64_t()> now_us;
  // Simulated per-shard H2D transfer duration.
  std::function<uint64_t(const LogicalExpertId&, int)> h2d_latency_us;
  // Simulated per-shard wait in the H2D queue before transfer starts.
  std::function<uint64_t(const LogicalExpertId&, int)> queue_latency_us;
  // Simulated per-shard device dependency wait (event wait after H2D).
  std::function<uint64_t(const LogicalExpertId&, int)> event_wait_us;
  // Simulated GPU GEMM duration per expert.
  std::function<uint64_t(const LogicalExpertId&)> gpu_gemm_latency_us;
  // Simulated cache lookup/lock wait for a HIT expert.
  std::function<uint64_t(const LogicalExpertId&)> cache_wait_us;
  // Simulated CPU GEMM duration per expert.
  std::function<uint64_t(const LogicalExpertId&)> cpu_gemm_latency_us;
  // Observers (no side effects on pipeline state).
  std::function<void(const LogicalExpertId&)> on_gpu_gemm;
  // Lifecycle observers for a real backend.  They are called at enqueue and
  // completion boundaries so a CUDA adapter can bind refcounts and shard
  // readiness to stream events instead of host synchronization.
  std::function<void(const LogicalExpertId&)> on_gpu_gemm_submitted;
  std::function<void(const LogicalExpertId&)> on_gpu_gemm_complete;
  std::function<void(const PhysicalShardId&, uint64_t)> on_shard_ready;
  std::function<void(const LogicalExpertId&)> on_cpu_fallback;
  std::function<void(const std::vector<LogicalExpertId>&)> on_merge;
};

struct RoundResult {
  std::vector<LogicalExpertId> cache_hits;
  std::vector<LogicalExpertId> h2d_loaded;
  std::vector<LogicalExpertId> cpu_fallback;
};

enum class H2DOutcome {
  Loaded,     // transfer executed this round
  FromCache,  // duplicate request reused a resident copy (no duplicate H2D)
  Fallback,   // could not take the H2D path
};

class RollingPipeline {
 public:
  RollingPipeline(GecConfig config, Topology topology, PipelineHooks hooks = {})
      : cache_(std::move(config), std::move(topology)),
        scheduler_(cache_.config()) {
    set_hooks(hooks);
    budget_remaining_ = cache_.config().layer_h2d_slots;
    ensure_pools_();
  }

  // Replaces the hook set (defaults re-applied for unset members).
  void set_hooks(PipelineHooks hooks) {
    hooks_.now_us = hooks.now_us ? hooks.now_us : default_now_us;
    hooks_.h2d_latency_us = hooks.h2d_latency_us ? hooks.h2d_latency_us : zero_shard_latency;
    hooks_.queue_latency_us = hooks.queue_latency_us ? hooks.queue_latency_us : zero_shard_latency;
    hooks_.event_wait_us = hooks.event_wait_us ? hooks.event_wait_us : zero_shard_latency;
    hooks_.gpu_gemm_latency_us = hooks.gpu_gemm_latency_us ? hooks.gpu_gemm_latency_us : zero_expert_latency;
    hooks_.cache_wait_us = hooks.cache_wait_us ? hooks.cache_wait_us : zero_expert_latency;
    hooks_.cpu_gemm_latency_us = hooks.cpu_gemm_latency_us ? hooks.cpu_gemm_latency_us : zero_expert_latency;
    hooks_.on_gpu_gemm = hooks.on_gpu_gemm;
    hooks_.on_gpu_gemm_submitted = hooks.on_gpu_gemm_submitted;
    hooks_.on_gpu_gemm_complete = hooks.on_gpu_gemm_complete;
    hooks_.on_shard_ready = hooks.on_shard_ready;
    hooks_.on_cpu_fallback = hooks.on_cpu_fallback;
    hooks_.on_merge = hooks.on_merge;
  }

  GlobalExpertCache& cache() { return cache_; }
  const GlobalExpertCache& cache() const { return cache_; }
  TelemetryCollector& telemetry() { return telemetry_; }
  const TelemetryCollector& telemetry() const { return telemetry_; }
  InflightRegistry& inflight() { return inflight_; }
  const InflightRegistry& inflight() const { return inflight_; }

  // Resets the per-layer / per-round H2D admission budget.
  void reset_budget() { budget_remaining_ = cache_.config().layer_h2d_slots; }
  int budget_remaining() const { return budget_remaining_; }

  // Startup warm start: sequentially fills the persistent cache with ordered
  // experts up to kt-expert-gpu-slots (GlobalExpertCache::prefill_sequential).
  // After this call, resident experts take the immediate GPU HIT path while
  // PCIe keeps transferring misses in parallel (doc section 21).
  size_t prefill_sequential(const std::vector<LogicalExpertId>& ordered_ids,
                            uint64_t now_us = 0) {
    return cache_.prefill_sequential(ordered_ids, now_us);
  }

  // Free (recycled) buffer count for a GPU, for tests and diagnostics.
  int free_buffer_count(int gpu_id) const {
    auto it = pools_.find(gpu_id);
    if (it == pools_.end()) throw std::invalid_argument("unknown gpu");
    return it->second->free_count();
  }

  // Runs one routing round. `demand[i]` is the current demand weight for
  // `current[i]`; when empty every expert gets demand 1.
  RoundResult run_round(const std::vector<LogicalExpertId>& current,
                        const std::vector<uint64_t>& demand = {}) {
    RoundResult result;
    const uint64_t round_start = hooks_.now_us();
    std::vector<SchedulerCandidate> candidates;
    candidates.reserve(current.size());

    for (size_t i = 0; i < current.size(); ++i) {
      const LogicalExpertId& id = current[i];
      const uint64_t weight = i < demand.size() ? demand[i] : 1;
      record_router_dependency(id, round_start);
      if (cache_.lookup(id, round_start) == GlobalExpertCache::LookupResult::Hit) {
        result.cache_hits.push_back(id);
        run_cache_hit(id);
        continue;
      }
      SchedulerCandidate candidate;
      candidate.id = id;
      candidate.current_demand = weight;
      const LogicalExpert* known = cache_.find(id);
      candidate.historical_hit_count = known ? known->hit_count : 0;
      candidate.inflight = any_shard_inflight(id);
      candidate.reuse_inflight = candidate.inflight;
      candidate.evicting = known && known->state == LifecycleState::Evicting;
      candidate.placement_available = placement_available(id);
      candidate.buffer_available = buffers_available(id);
      candidate.cache_admission_locked = cache_.admission_locked(id);
      candidates.push_back(candidate);
    }

    SchedulerDecision decision = scheduler_.schedule(candidates, budget_remaining_);
    budget_remaining_ -= static_cast<int>(decision.h2d_selected.size());

    // Submit the round selection in batches of at most kt-layer-h2d-batch-size
    // experts; buffer pool recycling applies between batches (doc section 15).
    for (const std::vector<LogicalExpertId>& batch :
         decision.h2d_batches(cache_.config().layer_h2d_batch_size)) {
      for (const LogicalExpertId& id : batch) {
        const H2DOutcome outcome = run_h2d_path(id, round_start);
        if (outcome == H2DOutcome::Fallback) {
          result.cpu_fallback.push_back(id);
        } else if (outcome == H2DOutcome::FromCache) {
          result.cache_hits.push_back(id);
        } else {
          result.h2d_loaded.push_back(id);
        }
      }
    }
    for (const LogicalExpertId& id : decision.cpu_fallback) {
      run_cpu_fallback(id, round_start);
      result.cpu_fallback.push_back(id);
    }

    std::vector<LogicalExpertId> executed;
    executed.reserve(result.cache_hits.size() + result.h2d_loaded.size() + result.cpu_fallback.size());
    executed.insert(executed.end(), result.cache_hits.begin(), result.cache_hits.end());
    executed.insert(executed.end(), result.h2d_loaded.begin(), result.h2d_loaded.end());
    executed.insert(executed.end(), result.cpu_fallback.begin(), result.cpu_fallback.end());
    if (hooks_.on_merge) hooks_.on_merge(executed);
    return result;
  }

  static uint64_t default_now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
  }

 private:
  static uint64_t zero_shard_latency(const LogicalExpertId&, int) { return 0; }
  static uint64_t zero_expert_latency(const LogicalExpertId&) { return 0; }

  void ensure_pools_() {
    const Topology& topology = cache_.topology();
    const int depth = cache_.config().h2d_buffer_pool_depth;
    for (int gpu = 0; gpu < topology.gpu_count; ++gpu) {
      pools_.emplace(gpu, std::make_unique<H2DBufferPool>(gpu, depth));
    }
  }

  bool any_shard_inflight(const LogicalExpertId& id) {
    for (int tp = 0; tp < cache_.topology().tp_size; ++tp) {
      if (inflight_.contains(PhysicalShardId{id.layer_id, id.expert_id, tp})) return true;
    }
    return false;
  }

  bool placement_available(const LogicalExpertId& id) {
    try {
      static_cast<void>(PlacementPlanner(cache_.topology()).plan(id, 0));
      return true;
    } catch (...) {
      return false;
    }
  }

  bool buffers_available(const LogicalExpertId& id) {
    try {
      PlacementPlanner planner(cache_.topology());
      std::vector<PhysicalShard> shards = planner.plan(id, 0);
      for (const PhysicalShard& shard : shards) {
        auto it = pools_.find(shard.gpu_id);
        if (it == pools_.end() || it->second->free_count() < 1) return false;
      }
      return true;
    } catch (...) {
      return false;
    }
  }

  void record_router_dependency(const LogicalExpertId& id, uint64_t round_start) {
    const uint64_t dispatch = hooks_.now_us();
    WaitRecord record;
    record.layer_id = id.layer_id;
    record.expert_id = id.expert_id;
    record.reason = WaitReason::RouterDependency;
    record.wait_start_us = round_start;
    record.wait_end_us = dispatch;
    record.wait_duration_us = dispatch - round_start;
    telemetry_.record_wait(record);
  }

  void run_cache_hit(const LogicalExpertId& id) {
    cache_.record_use(id, hooks_.now_us());
    if (!cache_.acquire(id)) return;  // lost a race with eviction; treated as miss next round
    const uint64_t gemm_start = hooks_.now_us();
    const uint64_t cache_wait = hooks_.cache_wait_us(id);
    if (hooks_.on_gpu_gemm) hooks_.on_gpu_gemm(id);
    if (hooks_.on_gpu_gemm_submitted) hooks_.on_gpu_gemm_submitted(id);
    const uint64_t gemm_end = gemm_start + hooks_.gpu_gemm_latency_us(id);
    WaitRecord record;
    record.layer_id = id.layer_id;
    record.expert_id = id.expert_id;
    record.reason = WaitReason::CacheWait;
    record.wait_start_us = gemm_start;
    record.wait_end_us = gemm_start + cache_wait;
    record.wait_duration_us = cache_wait;
    telemetry_.record_wait(record);
    static_cast<void>(gemm_end);
    if (hooks_.on_gpu_gemm_complete) hooks_.on_gpu_gemm_complete(id);
    cache_.release(id);  // strictly after GPU completion
  }

  H2DOutcome run_h2d_path(const LogicalExpertId& id, uint64_t round_start) {
    // An inflight request is a dependency reuse case, not a CPU fallback.  The
    // transfer_shard() path below binds the new consumer to the existing
    // device dependency and never submits a second H2D.
    // An earlier request in the same round may have already loaded this expert
    // (duplicate routing). Execute on the resident copy: no duplicate H2D.
    const LogicalExpert* existing = cache_.find(id);
    if (existing != nullptr && existing->state == LifecycleState::Ready) {
      run_cache_hit(id);
      return H2DOutcome::FromCache;
    }
    PlacementPlanner planner(cache_.topology());
    std::vector<PhysicalShard> shards;
    try {
      shards = planner.plan(id, 0);
    } catch (...) {
      return H2DOutcome::Fallback;
    }

    const LogicalExpert* before_admission = cache_.find(id);
    const bool owns_admission =
        before_admission == nullptr ||
        before_admission->state == LifecycleState::Absent ||
        before_admission->state == LifecycleState::Scheduled;
    if (!cache_.admit(id, round_start)) return H2DOutcome::Fallback;

    const LogicalExpert* admitted = cache_.find(id);
    if (admitted == nullptr) return H2DOutcome::Fallback;
    if (admitted->state == LifecycleState::Ready) {
      run_cache_hit(id);
      return H2DOutcome::FromCache;
    }
    if (admitted->state == LifecycleState::Scheduled) {
      cache_.mark_scheduled(id);
      cache_.mark_loading(id);
    } else if (admitted->state != LifecycleState::Loading) {
      return H2DOutcome::Fallback;
    }

    // Acquire all buffers for this expert atomically (all-or-nothing).
    std::vector<H2DBuffer*> acquired;
    acquired.reserve(shards.size());
    for (const PhysicalShard& shard : shards) {
      auto it = pools_.find(shard.gpu_id);
      if (it == pools_.end()) {
        for (H2DBuffer* owned : acquired) release_buffer(owned);
        if (owns_admission) cache_.cancel_loading(id);
        return H2DOutcome::Fallback;
      }
      H2DBuffer* buffer = it->second->acquire(id, shard.id.tp_rank, round_start);
      if (buffer == nullptr) {
        for (H2DBuffer* owned : acquired) release_buffer(owned);
        if (owns_admission) cache_.cancel_loading(id);
        return H2DOutcome::Fallback;
      }
      acquired.push_back(buffer);
    }
    std::vector<H2DTimeline> timelines;
    timelines.reserve(shards.size());
    bool ok = true;
    bool submitted_transfer = false;
    for (size_t k = 0; k < shards.size() && ok; ++k) {
      std::optional<H2DTimeline> timeline =
          transfer_shard(id, shards[k], acquired[k], round_start,
                         submitted_transfer);
      if (timeline.has_value()) {
        timelines.push_back(*timeline);
      } else {
        ok = false;
      }
    }
    if (!ok) {
      for (H2DBuffer* buffer : acquired) release_buffer(buffer);
      if (owns_admission) cache_.cancel_loading(id);
      return H2DOutcome::Fallback;
    }
    if (submitted_transfer) cache_.record_h2d(id);

    // H2D completion publishes Logical READY only after every TP shard is
    // ready. Cache admission was reserved before any transfer was submitted.
    for (const PhysicalShard& shard : shards) {
      cache_.mark_shard_ready(id, shard.id.tp_rank, hooks_.now_us(),
                              shard.ready_dependency);
    }
    cache_.record_use(id, hooks_.now_us());

    // All shards ready: logical expert is usable (TP atomic ready handled by
    // the cache after pre-admission and transfer completion).
    const uint64_t gemm_start = hooks_.now_us();
    for (H2DBuffer* buffer : acquired) pool_for(buffer->gpu_id)->begin_gpu_use(buffer->buffer_id);
    const bool cached = cache_.acquire(id);
    if (hooks_.on_gpu_gemm) hooks_.on_gpu_gemm(id);
    if (hooks_.on_gpu_gemm_submitted) hooks_.on_gpu_gemm_submitted(id);
    const uint64_t gemm_end = gemm_start + hooks_.gpu_gemm_latency_us(id);
    for (H2DTimeline& timeline : timelines) {
      timeline.gemm_start_us = gemm_start;
      timeline.gemm_end_us = gemm_end;
      telemetry_.record_h2d(timeline);
    }
    if (hooks_.on_gpu_gemm_complete) hooks_.on_gpu_gemm_complete(id);
    if (cached) cache_.release(id);  // refcount covers real GPU kernel lifetime
    for (H2DBuffer* buffer : acquired) pool_for(buffer->gpu_id)->gpu_complete(buffer->buffer_id);
    return H2DOutcome::Loaded;
  }

  std::optional<H2DTimeline> transfer_shard(const LogicalExpertId& id,
                                            PhysicalShard& shard,
                                            H2DBuffer* buffer, uint64_t round_start,
                                            bool& submitted_transfer) {
    const uint64_t queue_enter = round_start;
    auto registered = inflight_.register_transfer(shard.id, next_dependency_++,
                                                  hooks_.now_us());
    if (!registered.first) {
      // Already inflight: reuse the existing dependency, no duplicate H2D.
      // The deterministic harness has no asynchronous device queue, so the
      // dependency is treated as ready for this consumer; a CUDA adapter can
      // replace this with stream wait/event chaining without changing the
      // registry contract.
      shard.ready_dependency = registered.second;
      pool_for(buffer->gpu_id)->mark_ready(buffer->buffer_id, registered.second);
      const uint64_t ready = hooks_.now_us();
      record_h2d_wait(id, shard.id.tp_rank, WaitReason::H2DEventWait,
                      ready, ready);
      H2DTimeline timeline;
      timeline.expert = id;
      timeline.tp_rank = shard.id.tp_rank;
      timeline.queue_enter_us = ready;
      timeline.h2d_start_us = ready;
      timeline.h2d_end_us = ready;
      timeline.event_wait_start_us = ready;
      timeline.event_ready_us = ready;
      return timeline;
    }
    submitted_transfer = true;
    const uint64_t h2d_start = queue_enter + hooks_.queue_latency_us(id, shard.id.tp_rank);
    const uint64_t h2d_end = h2d_start + hooks_.h2d_latency_us(id, shard.id.tp_rank);
    const uint64_t event_wait_start = h2d_end;
    const uint64_t event_ready = event_wait_start + hooks_.event_wait_us(id, shard.id.tp_rank);
    shard.ready_dependency = registered.second;
    inflight_.complete(shard.id);
    pool_for(buffer->gpu_id)->mark_ready(buffer->buffer_id, registered.second);
    if (hooks_.on_shard_ready) hooks_.on_shard_ready(shard.id, event_ready);
    record_h2d_wait(id, shard.id.tp_rank, WaitReason::H2DQueued, queue_enter, h2d_start);
    record_h2d_wait(id, shard.id.tp_rank, WaitReason::H2DRunning, h2d_start, h2d_end);
    record_h2d_wait(id, shard.id.tp_rank, WaitReason::H2DEventWait, event_wait_start, event_ready);
    H2DTimeline timeline;
    timeline.expert = id;
    timeline.tp_rank = shard.id.tp_rank;
    timeline.queue_enter_us = queue_enter;
    timeline.h2d_start_us = h2d_start;
    timeline.h2d_end_us = h2d_end;
    timeline.event_wait_start_us = event_wait_start;
    timeline.event_ready_us = event_ready;
    return timeline;
  }

  void run_cpu_fallback(const LogicalExpertId& id, uint64_t round_start) {
    const uint64_t start = hooks_.now_us();
    if (hooks_.on_cpu_fallback) hooks_.on_cpu_fallback(id);
    const uint64_t end = start + hooks_.cpu_gemm_latency_us(id);
    record_h2d_wait(id, -1, WaitReason::CpuFallback, start, end);
    static_cast<void>(round_start);
  }

  void record_h2d_wait(const LogicalExpertId& id, int tp_rank, WaitReason reason,
                       uint64_t start, uint64_t end) {
    WaitRecord record;
    record.layer_id = id.layer_id;
    record.expert_id = id.expert_id;
    record.tp_rank = tp_rank;
    record.reason = reason;
    record.wait_start_us = start;
    record.wait_end_us = end;
    record.wait_duration_us = end - start;
    telemetry_.record_wait(record);
  }

  H2DBufferPool* pool_for(int gpu_id) {
    auto it = pools_.find(gpu_id);
    if (it == pools_.end()) throw std::invalid_argument("unknown gpu");
    return it->second.get();
  }

  // Rollback for transfers that never reached the compute stream. Frees H2D
  // and READY_FOR_COMPUTE buffers alike: no GPU kernel has consumed them, so
  // the pool must get every one of them back (otherwise they would leak).
  void release_buffer(H2DBuffer* buffer) {
    if (buffer->state == BufferState::H2D ||
        buffer->state == BufferState::ReadyForCompute) {
      pool_for(buffer->gpu_id)->release_unstarted(buffer->buffer_id);
    }
  }

  GlobalExpertCache cache_;
  H2DScheduler scheduler_;
  InflightRegistry inflight_;
  TelemetryCollector telemetry_;
  std::unordered_map<int, std::unique_ptr<H2DBufferPool>> pools_;
  int budget_remaining_ = 0;
  uint64_t next_dependency_ = 1;
  PipelineHooks hooks_;
};

}  // namespace kt::gec
