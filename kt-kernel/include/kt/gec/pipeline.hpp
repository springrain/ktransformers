// Rolling H2D pipeline orchestration (doc/KT-CPU-PICE-GPU.md sections 14, 21, 22).
//
// Deterministic, dependency-injected simulation harness for the V1 pipeline:
// GPU router -> cache lookup -> HIT path / MISS path (filter, priority, top-N,
// H2D or CPU fallback) -> compute stream -> merge. All CUDA interactions are
// abstracted behind PipelineHooks, so the lifecycle invariants (buffer reuse,
// refcount, inflight dedup, TP atomic ready) can be verified on any host.
//
// Fully-async staging (doc sections 15.3-15.4), mirroring FreeToken's prefill
// overlap: a round whose layer L starts submits layer L+K's misses as a
// look-ahead pass on the H2D stream and does not wait for them. Their device
// work retires at the next round boundary, i.e. while layer L computes. The
// round that routes layer L+K only waits on, and consumes, those transfers; it
// never submits them again.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kt/gec/buffer_pool.hpp"
#include "kt/gec/expert_cache.hpp"
#include "kt/gec/inflight_registry.hpp"
#include "kt/gec/scheduler.hpp"
#include "kt/gec/telemetry.hpp"

namespace kt::gec {

struct PipelineHooks {
  // Virtual clock (microseconds). Defaults to a simulated microsecond clock;
  // the wall clock is monotonic, which breaks event ordering in tests.
  std::function<uint64_t()> now_us;
  // Simulated per-shard H2D transfer duration.
  std::function<uint64_t(const LogicalExpertId&, int)> h2d_latency_us;
  // Simulated per-shard wait in the H2D queue before transfer starts.
  std::function<uint64_t(const LogicalExpertId&, int)> queue_latency_us;
  // Simulated wait of a stage that has to reclaim a ring generation still
  // held by an earlier round (a consumer outlived the ring depth).
  std::function<uint64_t(const LogicalExpertId&, int)> ring_stall_us;
  // Simulated per-shard device dependency wait (event wait after H2D).
  std::function<uint64_t(const LogicalExpertId&, int)> event_wait_us;
  // Simulated GPU GEMM duration per expert.
  std::function<uint64_t(const LogicalExpertId&)> gpu_gemm_latency_us;
  // Simulated cache lookup/lock wait for a HIT expert.
  std::function<uint64_t(const LogicalExpertId&)> cache_wait_us;
  // Simulated CPU GEMM duration per expert.
  std::function<uint64_t(const LogicalExpertId&)> cpu_gemm_latency_us;
  // Simulated preparation phase of a round (router + global cache lookup
  // across GPUs/layers, doc section 14.6). Producers start after it.
  std::function<uint64_t()> prepare_latency_us;
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
  // Staging observers: look-ahead submission (the H2D stream keeps working
  // after the call) and look-ahead completion (dependency now ready).
  std::function<void(const LogicalExpertId&)> on_prefetch_submitted;
  std::function<void(const LogicalExpertId&)> on_prefetch_complete;
};

struct RoundResult {
  std::vector<LogicalExpertId> cache_hits;
  std::vector<LogicalExpertId> h2d_loaded;
  // Executed from a look-ahead pass: consumed, not transferred, this round.
  std::vector<LogicalExpertId> prefetched;
  std::vector<LogicalExpertId> cpu_fallback;
};

enum class H2DOutcome {
  Loaded,      // transfer submitted by this round
  FromCache,   // duplicate request reused a resident copy (no duplicate H2D)
  Prefetched,  // look-ahead pass already owns the transfer; consumed here
  Fallback,    // could not take the H2D path
};

struct ExpertStage {
  LogicalExpertId expert;
  std::vector<PhysicalShard> shards;
  std::vector<H2DBuffer*> buffers;
  // True when this round submitted the stage's transfers, false when it only
  // bound to transfers another round had already submitted.
  bool submitted_here = false;
  bool owns_admission = false;
  bool completed = false;
};

class RollingPipeline {
 public:
  // Monotonic simulated microsecond clock used when the timeline is only
  // partially injected.
  static uint64_t simulated_clock() {
    static uint64_t now = 0;
    return ++now;
  }

  // Timeline position used when the caller only supplies latencies: the
  // global simulated clock, so concurrent streams (prepare, H2D, compute, CPU
  // fallback) can be projected onto one non-decreasing axis. Tests that
  // override latencies but not now_us need this to see stream overlap.
  static uint64_t virtual_now() { return simulated_clock(); }

  RollingPipeline(GecConfig config, Topology topology, PipelineHooks hooks = {})
      : cache_(std::move(config), std::move(topology)),
        scheduler_(cache_.config()) {
    set_hooks(hooks);
    budget_remaining_ = cache_.config().layer_h2d_slots;
    ensure_pools_();
  }

  // Replaces the hook set (defaults re-applied for unset members).
  void set_hooks(PipelineHooks hooks) {
    cached_now_.reset();
    if (hooks.now_us) {
      hooks_.now_us = hooks.now_us;
    } else {
      hooks_.now_us = simulated_clock;
    }
    hooks_.h2d_latency_us = hooks.h2d_latency_us ? hooks.h2d_latency_us : zero_shard_latency;
    hooks_.queue_latency_us = hooks.queue_latency_us ? hooks.queue_latency_us : zero_shard_latency;
    hooks_.ring_stall_us = hooks.ring_stall_us ? hooks.ring_stall_us : hooks_.h2d_latency_us;
    hooks_.event_wait_us = hooks.event_wait_us ? hooks.event_wait_us : zero_shard_latency;
    hooks_.gpu_gemm_latency_us = hooks.gpu_gemm_latency_us ? hooks.gpu_gemm_latency_us : zero_expert_latency;
    hooks_.cache_wait_us = hooks.cache_wait_us ? hooks.cache_wait_us : zero_expert_latency;
    hooks_.cpu_gemm_latency_us = hooks.cpu_gemm_latency_us ? hooks.cpu_gemm_latency_us : zero_expert_latency;
    hooks_.prepare_latency_us = hooks.prepare_latency_us ? hooks.prepare_latency_us : zero_prepare_latency;
    hooks_.on_gpu_gemm = hooks.on_gpu_gemm;
    hooks_.on_gpu_gemm_submitted = hooks.on_gpu_gemm_submitted;
    hooks_.on_gpu_gemm_complete = hooks.on_gpu_gemm_complete;
    hooks_.on_shard_ready = hooks.on_shard_ready;
    hooks_.on_cpu_fallback = hooks.on_cpu_fallback;
    hooks_.on_merge = hooks.on_merge;
    hooks_.on_prefetch_submitted = hooks.on_prefetch_submitted;
    hooks_.on_prefetch_complete = hooks.on_prefetch_complete;
  }

  GlobalExpertCache& cache() { return cache_; }
  const GlobalExpertCache& cache() const { return cache_; }
  // Staging counters published by the rounds released so far, plus the round
  // in flight (begin_round -> release_round). A released round's counters are
  // published by flush_staging_delta_ at its release / next begin_round, so the
  // published total and the in-flight delta never double count.
  StagingStats staging() const {
    StagingStats total = telemetry_.staging();
    total.begin_rounds += staging_delta_.begin_rounds;
    total.prefetch_submits += staging_delta_.prefetch_submits;
    total.consumed_from_prefetch += staging_delta_.consumed_from_prefetch;
    total.stall_waits += staging_delta_.stall_waits;
    total.fallbacks += staging_delta_.fallbacks;
    total.direct_hits += staging_delta_.direct_hits;
    total.frustum_dropped += staging_delta_.frustum_dropped;
    return total;
  }
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
  //
  // The filled rows stay on probation during the fill (doc section 19); once
  // the startup pass completes they are normal cached rows, so the first
  // routed round can take the slot a miss needs.
  size_t prefill_sequential(const std::vector<LogicalExpertId>& ordered_ids,
                            uint64_t now_us = 0) {
    const size_t admitted = cache_.prefill_sequential(ordered_ids, now_us);
    for (const LogicalExpertId& id : ordered_ids) {
      cache_.end_startup_protection(id);
    }
    return admitted;
  }

  // Free (recycled) buffer count for a GPU, for tests and diagnostics.
  int free_buffer_count(int gpu_id) const {
    auto it = pools_.find(gpu_id);
    if (it == pools_.end()) throw std::invalid_argument("unknown gpu");
    return it->second->free_count();
  }

  // Opens a round. With `prefetch` the round models layer L while its misses
  // are submitted as the look-ahead pass for a deeper layer.
  // Publishes the round in flight and starts the next one with empty counters.
  void flush_staging_delta_() {
    telemetry_.record_staging(staging_delta_);
    staging_delta_ = StagingStats{};
  }

  void begin_round(bool prefetch) {
    // The round in flight is over as a reporting window: publish it before its
    // successor opens, so begin_round -> run_round_current can read the
    // look-ahead pass it just ran instead of the round before it.
    flush_staging_delta_();
    ++round_id_;
    current_round_ = std::make_shared<RoundMeta>();
    current_round_->id = round_id_;
    current_round_->prefetch = prefetch;
    reset_budget();
    // A dead stage swept at the round boundary handed its H2D lease back; the
    // row fetching now re-takes it instead of waiting for a ring generation
    // nobody holds any more. Credited once, when the round starts executing
    // (run_round_current), whichever entry point opened it.
    ++staging_delta_.begin_rounds;
  }

  // Runs one complete round (begin + execute + release). `demand[i]` is the
  // current demand weight for `current[i]`; when empty every expert gets 1.
  //
  // With `prefetch`, misses become a look-ahead pass for the next layer and
  // are reported as CPU fallback for the current layer (the look-ahead
  // transfer is not needed now). The next layer's run_round() consumes them.
  RoundResult run_round(const std::vector<LogicalExpertId>& current,
                        const std::vector<uint64_t>& demand = {},
                        bool prefetch = false) {
    begin_round(prefetch);
    RoundResult result = run_round_current(current, demand);
    release_round();
    return result;
  }

  // Executes the consumers of the round opened by begin_round().
  RoundResult run_round_current(const std::vector<LogicalExpertId>& current,
                                const std::vector<uint64_t>& demand = {}) {
    if (!current_round_) begin_round(false);
    if (retake_pending_) {
      // The boundary sweep dropped a dead stage and handed back its H2D lease:
      // the first row fetching in this round takes it instead of stalling for
      // a ring generation that is already free.
      ++budget_remaining_;
      retake_pending_ = false;
    }
    settle_previous_round_();
    RoundResult result;
    const uint64_t round_start = now_();
    // Resident rows of any earlier round are consumable now, whatever their
    // layer: a later layer (cold start, longer sequence) re-routing them is a
    // cache hit, not a miss that finds a READY expert.
    if (cache_.expire_stages(round_start) > 0) refresh_staging_();
    // Ring credit for this round (FreeToken: the layers sharing one buffer
    // serialize on it instead of issuing behind each other). The generation a
    // row takes is one a round older than this one has already released; at
    // most `depth` generations are free from older rounds, and a row that
    // binds to an existing dependency never takes one. The scheduler weighs
    // the rows of the round against that credit one by one, so a depth-1 ring
    // admits one row and cuts the rest into CPU fallback instead of queueing
    // them behind a buffer they could not have. A pass that issues its rows
    // one after the other (synchronous pipeline) keeps one credit per ring
    // slot: the slot a row of the planned batch will write to is only checked
    // on the hand-off (held by an older generation), never charged twice for
    // the same slot, so a depth-1 ring admits one row of the batch and cuts
    // the rest into CPU fallback instead of letting both through.
    ring_credit_.clear();
    sequential_ = cache_.config().staging_prefetch_layers == 0;
    for (auto& entry : pools_) {
      // A pass transfers nothing at planning time: its rows are planned now
      // and submitted by release_round, which always takes an idle ring
      // generation in search order (generation_of is round % depth). The slots
      // the planned batch will write to are therefore still free here, and the
      // hand-off checks them again, so charging them would reject every
      // look-ahead row. Charge only what older rounds still hold.
      ring_credit_[entry.first] =
          current_round_ && current_round_->prefetch
              ? static_cast<int>(entry.second->depth()) -
                    entry.second->held_by_older_generations(current_round_->id)
              : static_cast<int>(entry.second->depth());
    }
    // Produce: router -> global cache lookup for every row. The devices do not
    // run yet; H2D submission, the resident GEMMs and the CPU fallback issue
    // together after this phase (release_round() is the consumer's join).
    const uint64_t prepare_end = round_start + hooks_.prepare_latency_us();
    cached_now_ = prepare_end;
    prepared_now_ = prepare_end;
    dispatched_.clear();
    std::vector<SchedulerCandidate> candidates;
    candidates.reserve(current.size());
    candidate_reuse_.clear();

    for (size_t i = 0; i < current.size(); ++i) {
      const LogicalExpertId& id = current[i];
      const uint64_t weight = i < demand.size() ? demand[i] : 1;
      record_router_dependency(id, round_start, prepare_end);
      const bool pass = current_round_->prefetch;
      // Whether the row's copy is owned by something other than a fetch this
      // round would plan: a stage of this pipeline, a transfer of another
      // requester, or a resident copy.
      const bool staged_row = staged(id);
      const bool bound = staged_row || candidate_buffers_inflight(id);
      const bool landed =
          landed_at_.count(id) > 0 || staged_dispatch_.count(id) > 0;
      if (pass) {
        // A look-ahead pass never executes a consumer and never submits a
        // second copy (FreeToken). A row it predicted for the next layer is
        // planned here exactly like an executing round's miss -- the plan is
        // submitted on the dump stream by release_round -- and the next layer
        // consumes it (ready at the boundary, or bound to the same copies).
        // Only a row somebody else already owns is left alone (no second
        // copy): a stage this pipeline tracks, or a transfer another requester
        // submitted. A resident row the pass finds is retained, not executed:
        // the pass runs no GEMM for its own layer.
        if (staged_row || landed || bound) continue;
        if (cache_.lookup(id, prepare_end) == GlobalExpertCache::LookupResult::Hit) {
          // Resident under an ordinary lease: the pass observes it (a read)
          // and leaves the copy to the layer that routes it.
          cache_.set_protected(id, false);
          continue;
        }
        // A pass looks one layer ahead, past the transfers of the layer it
        // runs for: a copy of this layer that already landed must not pin the
        // row past the layer that routes it.
        (void)cache_.find(id);
        cache_.set_protected(id, false);
      } else {
        // A consumer of this round takes a landed look-ahead row before its
        // own fetch is planned (FreeToken: the probe of the routing layer
        // binds).
        if (landed) claim_lookahead_(id);
        if (bound) {
          // A dependency already owns this row: consume it here (a look-ahead
          // stage earlier or this round, or a transfer another requester
          // submitted), never resubmit.
          run_ready_consumer(id, prepare_end, &result);
          continue;
        }
        if (cache_.lookup(id, prepare_end) == GlobalExpertCache::LookupResult::Hit) {
          // Resident: the GEMM is queued immediately while PCIe keeps working
          // on this round's misses (H2D and compute streams overlap).
          run_hit_consumer(id, prepare_end);
          result.cache_hits.push_back(id);
          continue;
        }
      }
      SchedulerCandidate candidate;
      candidate.id = id;
      candidate.current_demand = weight;
      const LogicalExpert* known = cache_.find(id);
      candidate.historical_hit_count = known ? known->hit_count : 0;
      candidate.inflight = candidate_buffers_inflight(id);
      candidate.reuse_inflight = candidate.inflight;
      candidate.evicting = known && known->state == LifecycleState::Evicting;
      const std::optional<std::vector<PhysicalShard>> shards = plan_shards(id);
      candidate.placement_available = shards.has_value();
      // Ring capacity is charged only by a row that actually transfers. A
      // reuse binds to a dependency already on the H2D stream: it needs no
      // ring slot of its own (and must not be starved when the ring is full).
      H2DReuse reuse = H2DReuse::None;
      if (shards.has_value()) reuse = classify_reuse(id, *shards);
      if (reuse != H2DReuse::None) {
        candidate.inflight = true;
        candidate.reuse_inflight = true;
        // Ring capacity is charged only by a row that actually transfers. A
        // TP-atomic reuse (every shard already on the H2D stream) binds to that
        // dependency: it needs no ring slot of its own and must not be starved
        // when the ring is full. A partial reuse still acquires the ring for
        // the stage it builds (the inflight shards are never submitted twice).
        candidate.buffer_available = reuse == H2DReuse::Dependency ||
                                     buffers_available(*shards, reuse == H2DReuse::None);
      } else {
        candidate.buffer_available = shards.has_value() && buffers_available(*shards, true);
      }
      candidate.cache_admission_locked = cache_.admission_locked(id);
      candidates.push_back(candidate);
      candidate_reuse_[id] = reuse;
    }
    SchedulerDecision decision = scheduler_.schedule(candidates, budget_remaining_);
    if (current_round_->prefetch) {
      // A pass never executes the rows of the layer it runs for: its whole
      // selection is the next layer's look-ahead window.
      current_round_->admitted = decision.h2d_selected;
    }
    budget_remaining_ -= static_cast<int>(decision.h2d_selected.size());

    // Stream overlap (doc section 15): the lookup threads are joined and the
    // whole round selection is submitted in one pass on the H2D stream. A
    // normal round queues the consumer right behind its submission. A
    // look-ahead pass only plans here: its rows are submitted by release_round
    // (the dump stream) and its consumers belong to the layer that routes
    // them, so a row the pass observed stays an observation rather than
    // turning into executed work of the pass itself.
    if (!current_round_->prefetch) {
      for (const LogicalExpertId& id : decision.h2d_selected) {
        const H2DOutcome outcome = run_h2d_path(id, prepare_end);
        if (outcome == H2DOutcome::Fallback) {
          run_cpu_fallback(id, prepare_end);
          result.cpu_fallback.push_back(id);
        } else if (outcome == H2DOutcome::FromCache) {
          result.cache_hits.push_back(id);
        } else if (outcome == H2DOutcome::Prefetched) {
          result.prefetched.push_back(id);
        } else {
          const auto reuse_it = candidate_reuse_.find(id);
          const H2DReuse reuse = reuse_it == candidate_reuse_.end()
                                     ? H2DReuse::None
                                     : reuse_it->second;
          if (reuse == H2DReuse::Dependency) {
            run_reuse_consumer(id, prepare_end, &result);
          } else {
            // Fully owned by this round's stage, or TP-atomic shard reuse
            // through transfer_shard (one H2D, the rest bind).
            dispatch_loaded_(id, result);
          }
        }
      }
    }

    // Consumer in order: P0 = the layer's own routed rows, P1 = look-ahead
    // rows whose consumer is this layer, P2 = CPU fallback (doc section 14.6:
    // A -> D -> C for the one-router-query round). A pass executes nothing.
    if (!current_round_->prefetch) {
      for (const LogicalExpertId& id : current) {
        // Only a live stage is consumed here; a row already bound by this
        // round's reuse consumer path verified it is not a cache hit.
        auto it = stage_index_.find(id);
        if (it == stage_index_.end()) continue;
        dispatch_staged_(id, prepare_end, &result);
      }
    }
    if (current_round_->prefetch) {
      for (const LogicalExpertId& id : decision.cpu_fallback) {
        reserve_fallback_slot_(id);
        run_cpu_fallback(id, prepare_end);
        result.cpu_fallback.push_back(id);
      }
    } else {
      // Cross-stream join: CPU fallback misses run one by one as the compute
      // stream retires the routed rows (a row whose H2D is still transferring
      // keeps the join busy, which is what TP=1 placeholders model).
      size_t joined = 0;
      for (const LogicalExpertId& id : decision.cpu_fallback) {
        reserve_fallback_slot_(id);
        const uint64_t dispatched = join_dispatch_time_(joined);
        run_cpu_fallback(id, std::max(dispatched, prepare_end));
        result.cpu_fallback.push_back(id);
        ++joined;
      }
    }
    cached_now_.reset();

    std::vector<LogicalExpertId> executed;
    executed.reserve(result.cache_hits.size() + result.prefetched.size() +
                     result.h2d_loaded.size() + result.cpu_fallback.size());
    executed.insert(executed.end(), result.cache_hits.begin(), result.cache_hits.end());
    executed.insert(executed.end(), result.prefetched.begin(), result.prefetched.end());
    executed.insert(executed.end(), result.h2d_loaded.begin(), result.h2d_loaded.end());
    executed.insert(executed.end(), result.cpu_fallback.begin(), result.cpu_fallback.end());
    if (hooks_.on_merge) hooks_.on_merge(executed);
    return result;
  }

  // End-of-layer release (FreeToken release_prefill_layer). The round's own
  // transfers are released now; look-ahead transfers survive until the layer
  // they belong to has consumed them.
  void release_round() {
    if (!current_round_) return;
    const uint64_t now = hooks_.now_us();
    // The staging transfers end with the layer's own consumers. This includes a
    // look-ahead pass: the pass executes nothing, so every row it submitted
    // lands here, on the dump stream, under the staging lease, while the
    // previous layer's compute still runs (FreeToken: the look-ahead copy is
    // awaited by the layer that later routes the row, at the entrance of
    // run_round_current, which is also where the landing is published). A stage
    // of an earlier round is only settled by the boundary: dispatching it here
    // would consume it before its layer ever routed it.
    for (StageRef& ref : staging_) {
      if (ref.stage.completed) continue;
      if (cache_stage_expired(ref)) continue;  // frustum already rolled it back
      if (ref.meta != current_round_) continue;
      dispatch_staged_(ref.stage.expert, prepared_now_, nullptr);
    }
    current_round_->released = true;
    current_round_->end_us = now;
    // Frustum rollback (doc section 15.4): a look-ahead row that its layer never
    // consumed is dropped, not kept. A row whose staging lease already expired
    // is treated the same way: the frustum hands it back instead of waiting for
    // a consumer that is not coming.
    for (const LogicalExpertId& id : current_round_->admitted) {
      if (!staged(id)) continue;
      if (!evictable_(id) && !cache_stage_expired(staging_[stage_index_[id]])) continue;
      drop_staged_(id);
    }
    finished_rounds_.push_back(current_round_);
    current_round_.reset();
    staged_dispatch_.clear();
    flush_staging_delta_();
  }

 private:
  static uint64_t zero_shard_latency(const LogicalExpertId&, int) { return 0; }
  static uint64_t zero_expert_latency(const LogicalExpertId&) { return 0; }
  static uint64_t zero_prepare_latency() { return 0; }

  // Per-round ownership: what the round admitted, and whether it has been
  // released. Shared with stages so consumption outliving the round is safe.
  struct RoundMeta {
    uint64_t id = 0;
    bool prefetch = false;
    bool released = false;
    uint64_t start_us = 0;
    uint64_t end_us = 0;
    std::vector<LogicalExpertId> admitted;
  };

  struct StageRef {
    ExpertStage stage;
    std::shared_ptr<RoundMeta> meta;
    // Projected end of the consuming GEMM on the compute stream.
    uint64_t gemm_end_us = 0;
    // Clock of the cache row this stage was submitted for (admission), the
    // origin of the staging lease.
    uint64_t submitted_at_us = 0;
  };

  // A stage whose row the cache released from protection: nobody observed it
  // inside the lease window, so its reservation is dropped as well. The round's
  // own stages are exempt: release_round() has not run yet, so the lease is
  // still held by definition (a zero-latency or constant clock would otherwise
  // roll back a transfer that is about to be consumed).
  bool cache_stage_expired(const StageRef& ref) const {
    const uint64_t lease = cache_.config().staging_lease_us;
    if (lease == 0) return false;
    if (ref.meta == current_round_ && current_round_ && !current_round_->released) {
      return false;
    }
    const LogicalExpert* expert = cache_.find(ref.stage.expert);
    if (expert == nullptr || expert->protected_) return false;
    // An expired row still holds a reservation (its slot was recycled, its
    // lease is held by this pipeline): dropping it hands the row back to the
    // eviction gate. A landed row is released there; a copy still in transit
    // is finished by the next boundary sweep, which counts the rollback.
    return expert->state == LifecycleState::Scheduled ||
           expert->state == LifecycleState::Loading ||
           expert->state == LifecycleState::Ready;
  }

  // True while an expert's row is held by a staging generation: it is then
  // consumed by the H2D consumer path instead of being resubmitted.
  bool staged(const LogicalExpertId& id) const {
    auto it = stage_index_.find(id);
    // The live index only. A completed stage may still sit in staging_ until
    // the next round compacts it, but it no longer binds the row: that row is
    // resident now and its next routing is an ordinary cache hit.
    return it != stage_index_.end() && !staging_[it->second].stage.completed;
  }

  // Rebuilds the id -> index map after expired entries were moved to the back
  // for erasure (staging_ is small: one entry per in-transit transfer).
  // Rebuilds the id -> index map of the live stages and drops the unfinished
  // stages whose cache row expired. Returns true when a stage was dropped.
  bool refresh_staging_() {
    size_t kept = 0;
    for (StageRef& ref : staging_) {
      // The cache released this row's protection at expiry; an unfinished
      // stage has no consumer left, so no row ever waits for it again.
      if (!ref.stage.completed && cache_stage_expired(ref)) {
        for (H2DBuffer* buffer : ref.stage.buffers) release_buffer(buffer);
        reclaim_admission_(ref.stage.expert);
        advance_pending_(ref.stage.expert);
        ++staging_delta_.frustum_dropped;
        continue;
      }
      staging_[kept++] = std::move(ref);
    }
    const bool dropped = kept != staging_.size();
    staging_.resize(kept);
    stage_index_.clear();
    for (size_t i = 0; i < staging_.size(); ++i) {
      // Completed entries are kept only for the round's telemetry; indexing
      // them would route the resident row back through the H2D consumer path.
      if (staging_[i].stage.completed) continue;
      stage_index_[staging_[i].stage.expert] = i;
    }
    return dropped;
  }

  // Where the compute stream currently is (the last queued GPU work), which
  // of this round's rows the prepare phase already dispatched, and the
  // look-ahead rows whose consumer fired in this round.
  uint64_t compute_tail_ = 0;
  std::unordered_set<LogicalExpertId, LogicalExpertIdHash> dispatched_;
  std::unordered_set<LogicalExpertId, LogicalExpertIdHash> staged_dispatch_;
  std::optional<uint64_t> cached_now_;
  // End of the last prepare phase: observations taken between phases (e.g. the
  // next round's clock read) never fall behind the work already issued.
  uint64_t prepared_now_ = 0;
  // A refused consumer is being re-fetched: progress_stage must not hand the
  // row to the refresh at the next round only to have it dropped again.
  bool suppress_expired_fetch_ = false;
  // A stage was swept at a round boundary without being consumed: the next
  // round's first fetch re-takes the H2D lease it took.
  bool retake_pending_ = false;
  // Ring generations this round may still take, per device (see
  // begin-of-round credit in run_round_current).
  std::unordered_map<int, int> ring_credit_;
  // The round's rows are issued one after the other (synchronous pipeline):
  // the ring credit is only a serialisation gate, never charged per row.
  bool sequential_ = false;

  uint64_t now_() {
    const uint64_t now = hooks_.now_us();
    // The simulated clock steps per observation; phases must not step it.
    if (cache_.config().staging_prefetch_layers == 0) {
      cached_now_ = now;
      return now;
    }
    return std::max(now, prepared_now_);
  }

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

  // H2D dependency reuse. FreeToken never submits a second copy of an expert
  // that is already on the H2D stream: the later consumer chains on the
  // existing device dependency. Two shapes reach this pipeline:
  //   Nothing     - this round owns the row end to end (submit, consume).
  //   Shards      - this round keeps its own stage but the ring finds some
  //                 shards already inflight: transfer_shard binds those and
  //                 submits only the rest (one H2D per TP shard).
  //   Dependency  - the whole row is owned elsewhere (another request's
  //                 transfer, or a look-ahead pass). No stage is built at all;
  //                 the consumer runs on the observed dependency.
  enum class H2DReuse { None, Shards, Dependency };

  // Every TP shard already on the H2D stream: nothing of this row needs a new
  // submission, so the consumer can bind without touching the ring.
  bool candidate_buffers_inflight(const LogicalExpertId& id) {
    const LogicalExpert* expert = cache_.find(id);
    // A row bound by
    //  * another request's transfer (the registry has every shard), or
    //  * a reuse this pipeline already bound (the row is LOADING and the next
    //    round polls it),
    // has a consumer-visible dependency and must not be resubmitted.
    if (expert != nullptr && expert->state == LifecycleState::Loading) return true;
    return shard_in_flight_count(id) >= static_cast<size_t>(cache_.topology().tp_size);
  }

  // The H2D dependency of a row no stage of this pipeline owns. Registry shard
  // entries carry the publisher's dependency, the cache row carries that of
  // the run that moved it to LOADING.
  uint64_t observed_ready_us_(const LogicalExpertId& id, uint64_t fallback) const {
    uint64_t ready = 0;
    for (int tp = 0; tp < cache_.topology().tp_size; ++tp) {
      const auto entry = inflight_.find(PhysicalShardId{id.layer_id, id.expert_id, tp});
      if (entry.has_value()) ready = std::max(ready, entry->dependency_id);
    }
    const LogicalExpert* expert = cache_.find(id);
    if (expert != nullptr) ready = std::max(ready, expert->logical_ready_dependency);
    return ready > 0 ? ready : fallback;
  }

  // Classifies one candidate. Exact shards are the row's TP atom; AnyShard is
  // how the registry is queried for a row the caller has no placement for
  // (every TP rank of the logical expert).
  H2DReuse classify_reuse(const LogicalExpertId& id,
                          const std::vector<PhysicalShard>& shards) {
    size_t in_flight = 0;
    for (const PhysicalShard& shard : shards) {
      if (inflight_.contains(shard.id)) ++in_flight;
    }
    const size_t any = shard_in_flight_count(id);
    if (shards.empty() || any == 0) return H2DReuse::None;
    if (in_flight == shards.size() || any >= shards.size()) return H2DReuse::Dependency;
    return H2DReuse::Shards;
  }

  size_t shard_in_flight_count(const LogicalExpertId& id) {
    size_t count = 0;
    for (int tp = 0; tp < cache_.topology().tp_size; ++tp) {
      if (inflight_.contains(PhysicalShardId{id.layer_id, id.expert_id, tp})) ++count;
    }
    return count;
  }

  std::optional<std::vector<PhysicalShard>> plan_shards(const LogicalExpertId& id) {
    try {
      return PlacementPlanner(cache_.topology()).plan(id, 0);
    } catch (...) {
      return std::nullopt;
    }
  }

  // Ring admission for one candidate: the round's generation slot must not
  // still be held by an older generation (FreeToken `layer_id % 2` hand-off).
  // The gate rates a single row's admission, never the whole request: when
  // the ring runs out the scheduler's budget cuts the residual rows into CPU
  // fallback instead of starving the round.
  bool buffers_available(const std::vector<PhysicalShard>& shards,
                         bool take_credit = false) {
    std::unordered_set<int> checked;
    for (const PhysicalShard& shard : shards) {
      if (!checked.insert(shard.gpu_id).second) continue;
      auto it = pools_.find(shard.gpu_id);
      // No pool staged for this device means unrestricted, as before.
      if (it == pools_.end()) continue;
      if (it->second->held_by_older_generations(round_id_) >=
          it->second->depth()) {
        return false;
      }
    }
    if (take_credit) {
      // The row will transfer this round and consume one generation: charge
      // the rows planned before it in the same round.
      for (int gpu_id : checked) {
        auto credit = ring_credit_.find(gpu_id);
        if (credit == ring_credit_.end()) continue;
        if (credit->second <= 0) return false;
        if (!sequential_) --credit->second;
      }
    }
    return true;
  }
  void record_router_dependency(const LogicalExpertId& id, uint64_t round_start,
                                uint64_t prepare_end) {
    WaitRecord record;
    record.layer_id = id.layer_id;
    record.expert_id = id.expert_id;
    record.reason = WaitReason::RouterDependency;
    record.wait_start_us = round_start;
    record.wait_end_us = prepare_end;
    record.wait_duration_us = prepare_end - round_start;
    telemetry_.record_wait(record);
  }

  void record_cache_wait(const LogicalExpertId& id, uint64_t start) {
    const uint64_t cache_wait = hooks_.cache_wait_us(id);
    record_h2d_wait(id, -1, WaitReason::CacheWait, start, start + cache_wait);
  }

  // Queues `latency_us` of GPU work for `id` on the compute stream and fires
  // the kernel boundary observers at enqueue and projected completion.
  uint64_t queue_gpu_kernel_(const LogicalExpertId& id, uint64_t earliest_us,
                             uint64_t latency_us) {
    const uint64_t start = std::max(earliest_us, compute_tail_);
    const uint64_t end = start + latency_us;
    compute_tail_ = end;
    if (hooks_.on_gpu_gemm) hooks_.on_gpu_gemm(id);
    if (hooks_.on_gpu_gemm_submitted) hooks_.on_gpu_gemm_submitted(id);
    if (hooks_.on_gpu_gemm_complete) hooks_.on_gpu_gemm_complete(id);
    return end;
  }

  // Resident expert: no transfer. The GEMM is queued right away, in parallel
  // with the round's H2D submissions.
  void run_hit_consumer(const LogicalExpertId& id, uint64_t prepare_end) {
    if (current_round_ && current_round_->prefetch) return;  // look-ahead floor
    if (!cache_.acquire(id)) return;  // lost a race with eviction: miss next round
    record_cache_wait(id, prepare_end);
    queue_gpu_kernel_(id, prepare_end, hooks_.gpu_gemm_latency_us(id));
    cache_.release(id);  // refcount covers real GPU kernel lifetime
  }

  // A transfer this round submitted for its own layer: the consumer waits on
  // the dependency and runs as soon as the device copy lands.
  void dispatch_loaded_(const LogicalExpertId& id, RoundResult& result) {
    auto it = stage_index_.find(id);
    if (it == stage_index_.end()) return;
    StageRef& ref = staging_[it->second];
    const uint64_t ready = dependency_ready_us_(ref.stage);
    if (cache_.mark_ready(id, ready)) {
      // The lease of an own-round row is held until this consumer runs, so
      // only a clock that jumped by a whole lease can get here. Same rule as a
      // look-ahead row: the lander's bytes belong to the row admitted in its
      // place, the consumer is refused and the expert is fetched again.
      finish_stage_(it->second);
      const int budget = budget_remaining_;
      ++budget_remaining_;
      if (run_h2d_path(id, ready) == H2DOutcome::Fallback) {
        run_cpu_fallback(id, ready);
        result.cpu_fallback.push_back(id);
      }
      budget_remaining_ = budget;
      return;
    }
    complete_stage(ref.stage, ready);
    ref.stage.completed = true;
    stage_index_.erase(id);
    owning_round_.erase(id);
    // Only a row a look-ahead pass handed over counts as consumed from
    // staging; one this round fetched for itself is the ordinary H2D path.
    if (ref.meta->prefetch) ++staging_delta_.consumed_from_prefetch;
    staged_dispatch_.insert(id);
    landed_at_.erase(id);
    // An observed landing consumed by this row's own layer is no longer a
    // reservation: its staging lease ends with the consumer.
    cache_.set_protected(id, false);
    // Consumption-observed learning (FreeToken): the layer that routes the row
    // actually ran its GEMM on this copy, so the row latches into the resident
    // cache. The same rule applies to every consumer path — an own-round
    // transfer, a look-ahead row this layer routes, and a row admitted by CPU
    // fallback — otherwise a transferred row stays probationary forever.
    cache_.record_use(id, ready);
    if (hooks_.on_prefetch_complete) hooks_.on_prefetch_complete(id);
    ref.gemm_end_us = queue_gpu_kernel_(id, ready, hooks_.gpu_gemm_latency_us(id));
    advance_pending_(id);  // staging lease ends at the consumer
    result.h2d_loaded.push_back(id);
  }

  // A look-ahead row reached its consumer: consume, never resubmit.
  void dispatch_staged_(const LogicalExpertId& id, uint64_t prepare_end,
                        RoundResult* result) {
    auto it = stage_index_.find(id);
    if (it == stage_index_.end()) {
      if (result != nullptr) {
        // Bound by another request in this round and already consumed there:
        // execute on the resident copy instead of transferring again.
        run_hit_consumer(id, prepare_end);
        result->prefetched.push_back(id);
      }
      return;
    }
    StageRef& ref = staging_[it->second];
    const uint64_t ready = dependency_ready_us_(ref.stage);
    // Consumption observed: a row whose lease expired while the copy was in
    // flight is refused here (its slot and bytes already belong to the row
    // admitted in its place). The stage is released like a frustum drop and
    // the row is fetched again through the normal H2D path.
    if (cache_.mark_ready(id, ready)) {
      complete_stage(ref.stage, ready);
      finish_stage_(it->second);
      if (!suppress_expired_fetch_) recover_expired_(id, ready, result);
      return;
    }
    complete_stage(ref.stage, ready);
    ref.stage.completed = true;
    stage_index_.erase(id);
    owning_round_.erase(id);
    // The landing now has a consumer: the staging lease it was held under ends
    // with this consumption, so the resident copy goes back to ordinary LRU.
    cache_.set_protected(id, false);
    // A row landed by the look-ahead is consumed here; one this round bound to
    // its own transfer is the ordinary H2D path.
    if (ref.meta->prefetch) ++staging_delta_.consumed_from_prefetch;
    staged_dispatch_.insert(id);
    landed_at_.erase(id);
    // The transfer has been consumed by the grid either way (it pipped the
    // previous round's compute, or the stage was settled at this round's
    // boundary): a real use, which latches the row into the resident cache
    // instead of letting it stay a look-ahead reserve (FreeToken: rows staged
    // but never re-routed stay probationary and evictable, rows a layer
    // actually consumes become cached).
    cache_.record_use(id, ready);
    if (hooks_.on_prefetch_complete) hooks_.on_prefetch_complete(id);
    ref.gemm_end_us = queue_gpu_kernel_(id, ready, hooks_.gpu_gemm_latency_us(id));
    // The staging lease is the only reference this consumer ever took on the
    // look-ahead row (acquire_pending); ending it here hands the row to the
    // eviction gate. No extra per-TP release: nothing else took a reference.
    advance_pending_(id);
    if (result != nullptr) result->prefetched.push_back(id);
  }

  // An expired row reached its consumer after all: release the dead stage, then
  // fetch the expert again from its host source under the consumer's own
  // budget (the look-ahead submit was wasted, so it does not count).
  void recover_expired_(const LogicalExpertId& id, uint64_t start,
                        RoundResult* result) {
    if (hooks_.on_prefetch_complete) hooks_.on_prefetch_complete(id);
    const int budget = budget_remaining_;
    ++budget_remaining_;
    suppress_expired_fetch_ = true;
    const H2DOutcome outcome = run_h2d_path(id, start);
    suppress_expired_fetch_ = false;
    budget_remaining_ = budget;
    if (result != nullptr) {
      if (outcome == H2DOutcome::Loaded) result->h2d_loaded.push_back(id);
      else result->prefetched.push_back(id);
    }
  }

  // The layer routes a row bound by an earlier look-ahead pass or by another
  // request's transfer. A stage this pipeline owns carries the dependency and
  // the row is consumed on it; a stage it does not own (another requester's
  // H2D, or a reuse bound in an earlier round) is chained here.
  void run_ready_consumer(const LogicalExpertId& id, uint64_t prepare_end,
                          RoundResult* result) {
    if (current_round_ && current_round_->prefetch) {
      // Chained, not consumed: a row this round submitted already sits in
      // `prefetched` (a dependency is a miss of the layer, not a hit).
      const size_t before = result->prefetched.size();
      bind_lookahead_(id, prepare_end, result);
      if (result->prefetched.size() == before) result->cache_hits.push_back(id);
      return;
    }
    if (staged(id)) {
      dispatch_staged_(id, prepare_end, result);
      return;
    }
    run_reuse_consumer(id, prepare_end, result);
  }

  // Look-ahead consumer of a row another dependency already owns (FreeToken:
  // the prefetch pass chains on the transfer instead of issuing a second copy).
  // The pass runs no GEMM for its own layer — only the layer that later routes
  // the row consumes it — so the row is left bound and in flight and is
  // reported as prefetched; its consumer site settles READY.
  void bind_lookahead_(const LogicalExpertId& id, uint64_t prepare_end,
                       RoundResult* result) {
    if (staged(id)) {
      // A stage this pipeline tracks: its own consumer settles it later.
      if (result != nullptr) result->cache_hits.push_back(id);
      return;
    }
    // A resident copy: the pass observes it, so it is not a use of the pass
    // (an observation must not refresh the eviction clock, or the row it
    // predicts could never be rolled back). Only the lease is re-armed here.
    const LogicalExpert* resident = cache_.find(id);
    const bool resident_ready =
        resident != nullptr && resident->state == LifecycleState::Ready;
    if (!resident_ready && !cache_.bind_forward(id, prepare_end)) {
      run_cpu_fallback(id, prepare_end);
      if (result != nullptr) result->cpu_fallback.push_back(id);
      return;
    }
    // The pass observes the row (a resident copy it kept, or one a consumer
    // just kept) and re-arms its staging lease instead of consuming it: the
    // layer that routes the row next decides whether it was worth the copy.
    cache_.set_protected(id, true);
    if (!resident_ready) cache_.touch(id, prepare_end);
    if (result != nullptr) result->prefetched.push_back(id);
  }

  // Dependency reuse (FreeToken "bind to the existing H2D dependency"): the
  // row's copy is owned by another requester or by a look-ahead pass that this
  // pipeline does not track as a stage. The consumer neither submits a second
  // transfer nor falls back to the CPU: it waits on the observed device
  // dependency, runs the GEMM when it lands, and latches the row through
  // consumption-observed learning. The binding keeps the row out of the
  // eviction gate until that consumer has run (probation, then Cached).
  void run_reuse_consumer(const LogicalExpertId& id, uint64_t prepare_end,
                          RoundResult* result) {
    if (current_round_ && current_round_->prefetch) {
      bind_lookahead_(id, prepare_end, result);
      return;
    }
    if (!cache_.admit(id, prepare_end)) {
      // The slot budget is already made of other in-flight rows: the only
      // consumer that can run is the host one.
      run_cpu_fallback(id, prepare_end);
      if (result != nullptr) result->cpu_fallback.push_back(id);
      return;
    }
    const bool bound = cache_.bind_forward(id, prepare_end);
    (void)bound;
    const uint64_t ready = std::max(observed_ready_us_(id, prepare_end), prepare_end);
    if (cache_.mark_ready(id, ready)) {
      // The observed dependency expired before its consumer ran: the bytes
      // belong to the row admitted in its place. Fetch the row again instead
      // of consuming memory that is already reassigned.
      recover_expired_(id, ready, result);
      return;
    }
    complete_reuse_(id, ready);
    if (result != nullptr) result->cache_hits.push_back(id);
  }

  // A dependency-bound consumer ran: TP-atomic READY, lease, GEMM. The reuse
  // owns no ring buffer (it never submitted), so nothing is released here.
  void complete_reuse_(const LogicalExpertId& id, uint64_t ready) {
    if (cache_.acquire_pending(id)) ++pending_refs_[id];
    for (int tp = 0; tp < cache_.topology().tp_size; ++tp) {
      const PhysicalShardId shard{id.layer_id, id.expert_id, tp};
      const auto entry = inflight_.find(shard);
      cache_.mark_shard_ready(id, tp, ready,
                              entry.has_value() ? entry->dependency_id : ready);
    }
    // The transfer left the registry: whoever consumes the row next reads the
    // resident copy (READY), not the dependency.
    for (int tp = 0; tp < cache_.topology().tp_size; ++tp) {
      inflight_.complete(PhysicalShardId{id.layer_id, id.expert_id, tp});
    }
    record_h2d_wait(id, -1, WaitReason::H2DEventWait, ready, ready);
    cache_.record_use(id, ready);
    if (hooks_.on_prefetch_complete) hooks_.on_prefetch_complete(id);
    queue_gpu_kernel_(id, ready, hooks_.gpu_gemm_latency_us(id));
    advance_pending_(id);
  }

  // CPU fallback consumes the expert on the host, so the row still takes its
  // persistent-cache placement: the admission happens here instead of in the
  // H2D path, and the H2D budget never covers it (doc section 5.4: a
  // resident expert is a cache hit, not a transfer). Without this the
  // resident pool would lag the router by every row the budget cut.
  void reserve_fallback_slot_(const LogicalExpertId& id) {
    const LogicalExpert* expert = cache_.find(id);
    if (expert != nullptr && expert->state == LifecycleState::Ready) return;
    cache_.admit(id, cached_now_.value_or(prepared_now_));
  }

  H2DOutcome run_h2d_path(const LogicalExpertId& id, uint64_t start) {
    // An inflight request is a dependency reuse case, not a CPU fallback.  The
    // transfer_shard() path below binds the new consumer to the existing
    // device dependency and never submits a second H2D.
    // An earlier request in the same round may have already loaded this expert
    // (duplicate routing). Execute on the resident copy: no duplicate H2D.
    // A look-ahead pass is the exception: it never executes a hit of its own
    // (FreeToken), it only keeps a resident or landed copy under its lease.
    const LogicalExpert* existing = cache_.find(id);
    if (existing != nullptr && existing->state == LifecycleState::Ready &&
        current_round_->prefetch) {
      return H2DOutcome::Prefetched;
    }
    if (existing != nullptr && existing->state == LifecycleState::Ready) {
      run_hit_consumer(id, start);
      return H2DOutcome::FromCache;
    }
    const std::optional<std::vector<PhysicalShard>> planned = plan_shards(id);
    if (!planned.has_value()) return H2DOutcome::Fallback;
    const std::vector<PhysicalShard>& shards = *planned;

    const LogicalExpert* before_admission = cache_.find(id);
    const bool owns_admission =
        before_admission == nullptr ||
        before_admission->state == LifecycleState::Absent ||
        before_admission->state == LifecycleState::Scheduled;
    if (!cache_.admit(id, start)) return H2DOutcome::Fallback;

    const LogicalExpert* admitted = cache_.find(id);
    if (admitted == nullptr) return H2DOutcome::Fallback;
    if (admitted->state == LifecycleState::Ready) {
      run_hit_consumer(id, start);
      return H2DOutcome::FromCache;
    }
    if (admitted->state == LifecycleState::Scheduled) {
      cache_.mark_scheduled(id);
      cache_.mark_loading(id);
    } else if (admitted->state != LifecycleState::Loading) {
      return H2DOutcome::Fallback;
    }

    ExpertStage stage;
    stage.expert = id;
    stage.shards = shards;
    stage.owns_admission = owns_admission;
    return progress_stage(stage, start);
  }

  // Reclaims the ring when this round's generation is still held by an
  // earlier round, then submits (or binds to) the stage's transfers.
  // Rollback is all-or-nothing (doc section 15.4); on success the stage is
  // registered for consumption.
  H2DOutcome progress_stage(ExpertStage& stage, uint64_t start) {
    const LogicalExpertId id = stage.expert;
    uint64_t stall = 0;
    for (const PhysicalShard& shard : stage.shards) {
      auto it = pools_.find(shard.gpu_id);
      if (it == pools_.end()) continue;
      const H2DBuffer* generation = it->second->at_generation(current_round_->id);
      if (generation == nullptr || generation->state != BufferState::Free) {
        stall = std::max(stall, hooks_.ring_stall_us(id, shard.id.tp_rank));
      }
    }
    if (stall > 0) ++staging_delta_.stall_waits;
    start += stall;

    // Acquire all buffers for this expert atomically (all-or-nothing).
    for (const PhysicalShard& shard : stage.shards) {
      auto it = pools_.find(shard.gpu_id);
      if (it == pools_.end()) return rollback_stage(stage);
      H2DBuffer* buffer = it->second->acquire(id, shard.id.tp_rank, start,
                                              current_round_->id);
      if (buffer == nullptr) return rollback_stage(stage);
      stage.buffers.push_back(buffer);
    }

    bool submitted = false;
    for (size_t k = 0; k < stage.shards.size(); ++k) {
      submitted = transfer_shard(stage, k, start) || submitted;
    }
    stage.submitted_here = submitted;
    cache_.record_h2d(id);
    current_round_->admitted.push_back(id);
    if (!submitted) {
      // Every shard already had an owner (look-ahead pass or another consumer
      // in this round): dependency binding, never a second transfer.
      for (H2DBuffer* buffer : stage.buffers) release_buffer(buffer);
      reclaim_admission_(id);
      return H2DOutcome::Prefetched;
    }
    // Keep the logical expert alive until the transfer has finished and the
    // layer that needs it has consumed it (FreeToken's staging lease); a
    // look-ahead row outlives its own round. The lease is taken here, i.e. on
    // the observation of the round that submitted the transfer; its expiry is
    // the cache's boundary sweep (expire_stages) and, for a landed row nobody
    // routes, the pipeline's own lease windows (landed_at_).
    // acquire_pending() can refuse a row the cache already retired; the stage
    // then owns no lease and advance_pending_() must not release one.
    if (cache_.acquire_pending(id)) ++pending_refs_[id];
    stage_index_[id] = staging_.size();
    owning_round_[id] = current_round_.get();
    staging_.push_back(StageRef{stage, current_round_, 0, start});
    if (current_round_->prefetch) {
      // The transfer outlives this call: it is awaited at consumption.
      ++staging_delta_.prefetch_submits;
      if (hooks_.on_prefetch_submitted) hooks_.on_prefetch_submitted(id);
    }
    return H2DOutcome::Loaded;
  }

  void reclaim_admission_(const LogicalExpertId& id) {
    const LogicalExpert* expert = cache_.find(id);
    if (expert != nullptr &&
        (expert->state == LifecycleState::Scheduled ||
         expert->state == LifecycleState::Loading)) {
      cache_.cancel_loading(id);
    }
  }

  H2DOutcome rollback_stage(ExpertStage& stage) {
    for (H2DBuffer* owned : stage.buffers) release_buffer(owned);
    if (stage.owns_admission) reclaim_admission_(stage.expert);
    ++staging_delta_.fallbacks;
    return H2DOutcome::Fallback;
  }

  // Submits (or binds to) one TP shard's transfer. Returns true when this
  // call submitted a new transfer, false when it reused an inflight one.
  bool transfer_shard(ExpertStage& stage, size_t index, uint64_t start) {
    const LogicalExpertId id = stage.expert;
    PhysicalShard& shard = stage.shards[index];
    H2DBuffer* buffer = stage.buffers[index];
    auto registered = inflight_.register_transfer(shard.id, next_dependency_++,
                                                  start);
    if (!registered.first) {
      // Already inflight: reuse the existing dependency, no duplicate H2D.
      // The deterministic harness has no asynchronous device queue, so the
      // dependency is treated as ready for this consumer; a CUDA adapter can
      // replace this with stream wait/event chaining without changing the
      // registry contract.
      shard.ready_dependency = registered.second;
      pool_for(buffer->gpu_id)->mark_ready(buffer->buffer_id, registered.second);
      record_h2d_wait(id, shard.id.tp_rank, WaitReason::H2DEventWait, start, start);
      H2DTimeline timeline;
      timeline.expert = id;
      timeline.tp_rank = shard.id.tp_rank;
      timeline.queue_enter_us = start;
      timeline.h2d_start_us = start;
      timeline.h2d_end_us = start;
      timeline.event_wait_start_us = start;
      timeline.event_ready_us = start;
      telemetry_.record_h2d(timeline);
      return false;
    }
    const uint64_t queue_enter = start;
    const uint64_t h2d_start = queue_enter + hooks_.queue_latency_us(id, shard.id.tp_rank);
    const uint64_t h2d_end = h2d_start + hooks_.h2d_latency_us(id, shard.id.tp_rank);
    const uint64_t event_wait_start = h2d_end;
    const uint64_t event_ready = event_wait_start + hooks_.event_wait_us(id, shard.id.tp_rank);
    shard.ready_dependency = event_ready;
    inflight_.complete(shard.id);
    pool_for(buffer->gpu_id)->mark_ready(buffer->buffer_id, event_ready);
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
    telemetry_.record_h2d(timeline);
    return true;
  }

  // Last device dependency among the stage's shards: TP atomic ready means
  // the consumer waits for every shard.
  static uint64_t dependency_ready_us_(const ExpertStage& stage) {
    uint64_t ready = 0;
    for (const PhysicalShard& shard : stage.shards) {
      ready = std::max(ready, shard.ready_dependency);
    }
    return ready;
  }

  // TP atomic ready for every shard of the stage (the device copy landed).
  void complete_stage(const ExpertStage& stage, uint64_t now) {
    for (const PhysicalShard& shard : stage.shards) {
      cache_.mark_shard_ready(stage.expert, shard.id.tp_rank, now,
                              shard.ready_dependency);
    }
    for (H2DBuffer* buffer : stage.buffers) {
      pool_for(buffer->gpu_id)->begin_gpu_use(buffer->buffer_id);
    }
    for (H2DBuffer* buffer : stage.buffers) {
      pool_for(buffer->gpu_id)->gpu_complete(buffer->buffer_id);
    }
  }

  void run_cpu_fallback(const LogicalExpertId& id, uint64_t start) {
    if (hooks_.on_cpu_fallback) hooks_.on_cpu_fallback(id);
    const uint64_t end = start + hooks_.cpu_gemm_latency_us(id);
    record_h2d_wait(id, -1, WaitReason::CpuFallback, start, end);
    compute_tail_ = std::max(compute_tail_, end);
    // The expert was consumed (on the host) and admitted a residency placement:
    // that is a use, the same as a GPU consumer's. A look-ahead floor is not,
    // and a row admission refused has no cache row to credit.
    if ((!current_round_ || !current_round_->prefetch) && cache_.find(id) != nullptr) {
      cache_.record_use(id, end);
    }
  }

  // Cross-stream join: when may CPU fallback miss `joined` start? The compute
  // stream retires routed rows in order, so the join clears one row per
  // staged GEMM; a row whose H2D is still transferring blocks it (mirroring
  // the GPU stream wait). Returns compute_tail_ when nothing is left.
  uint64_t join_dispatch_time_(size_t joined) const {
    uint64_t next = compute_tail_;
    std::vector<const StageRef*> pending;
    pending.reserve(staging_.size());
    for (const StageRef& ref : staging_) {
      if (!ref.stage.completed && ref.gemm_end_us > 0) pending.push_back(&ref);
    }
    std::sort(pending.begin(), pending.end(),
              [](const StageRef* a, const StageRef* b) { return a->gemm_end_us < b->gemm_end_us; });
    for (size_t k = 0; k < pending.size(); ++k) {
      const uint64_t start = pending[k]->gemm_end_us - gemm_duration_(pending[k]->stage.expert);
      next = std::max(next, start);
      if (k == joined) return next;
    }
    return next;
  }

  uint64_t gemm_duration_(const LogicalExpertId& id) const {
    return hooks_.gpu_gemm_latency_us(id);
  }

  // Round boundary: the previous round's look-ahead transfers have finished
  // while the previous layer computed. TP-atomic READY becomes visible here;
  // the compute stream consumes the rows when their layer routes them.
  void settle_previous_round_() {
    const uint64_t boundary = now_();
    // A row of the previous look-ahead pass settled here belongs to the layer
    // that is about to route it (FreeToken: the pass is one layer ahead), so
    // it stays consumed-from-prefetch and never a direct cache hit.
    const bool promoting = cache_.config().staging_prefetch_layers > 0 ||
                           (current_round_ && current_round_->prefetch);
    // Land the copies before this round routes, so a layer that asks for one
    // still consumes it from staging (FreeToken: a copy the previous layer
    // submitted has landed by the time this layer routes). A landed look-ahead
    // row stays only *reserved* for its own layer: the ledger below marks it,
    // and this round's routing clears the mark only when it consumes the row.
    for (size_t i = 0; i < staging_.size(); ++i) {
      StageRef& ref = staging_[i];
      if (ref.stage.completed || ref.meta == current_round_) continue;
      if (cache_stage_expired(ref)) continue;
      if (!cache_.mark_ready(ref.stage.expert, boundary)) {
        complete_stage(ref.stage, boundary);
        // The transfer landed: the ring slot it held returns to the pool here,
        // at the boundary, not when the layer eventually consumes the row.
        finish_stage_(i);
        // A probe (a row bound by this layer's routing, not a staged
        // look-ahead copy) records what its copy landed for, nothing more: the
        // demand of the earlier layer it chained on. A look-ahead row is the
        // transfer its layer consumes, so it is a use.
        if (ref.meta->prefetch) cache_.record_use(ref.stage.expert, boundary);
        if (promoting) {
          // The landing is observation, not consumption: the row is held for
          // the layer that routes it, by the staging lease (refreshed here)
          // rather than by the admission every copy gets.
          cache_.set_protected(ref.stage.expert, true);
          cache_.touch(ref.stage.expert, boundary);
        }
        if (promoting && ref.meta->prefetch) {
          ++staging_delta_.consumed_from_prefetch;
        }
      } else {
        finish_stage_(i);
      }
      // A look-ahead row landed at this boundary is reserved for the layer
      // that is about to route it: the routing pass claims it right before its
      // own fetch, which is free (the copy already landed), so the row is bound
      // here only to keep it out of the eviction gate until then.
      if (ref.meta->prefetch) {
        const LogicalExpert* row = cache_.find(ref.stage.expert);
        if (row != nullptr && row->state == LifecycleState::Ready &&
            !cache_.bind_forward(ref.stage.expert, boundary)) {
          // Not admitted any more (the ring slot went elsewhere): this cannot
          // be consumed, hand it back like any other frustum row.
          drop_landed_(ref.stage.expert);
        }
        ++landed_at_[ref.stage.expert];
      } else {
        landed_at_.erase(ref.stage.expert);
      }
    }
    if (refresh_staging_() && (cache_.config().staging_prefetch_layers == 0 ||
                               (current_round_ && current_round_->prefetch))) {
      // A swept stage handed its H2D lease back. The synchronous pipeline next
      // fetches through begin_round, a look-ahead pass plans its rows into
      // release_round -- the very round that settles here either way -- and
      // the consumer routing this round's landing would otherwise queue behind
      // a generation nobody holds any more: all three re-take the lease.
      retake_pending_ = true;
    }
    // Frustum rollback (FreeToken): a look-ahead row nobody was routed is
    // handed back before this round's routing, so no later layer ever boots it
    // on bytes it did not ask for, and a landed row the cache has already
    // swept out of protection (lease ran out) likewise. A row that was only
    // settled now never had a layer to be routed by before, and a row this
    // round's routing takes over is not looked at here at all.
    for (size_t i = 0; i < staging_.size(); ++i) {
      StageRef& ref = staging_[i];
      if (ref.stage.completed || ref.meta == current_round_) continue;
      if (cache_stage_expired(ref) || drops_after_round_(ref.stage.expert)) {
        drop_staged_(ref.stage.expert);
      }
    }
    // The boundary is the window counter: a clock that stands still inside a
    // layer cannot tell two boundaries apart, and this never waits longer
    // than the windows the config allows.
    for (auto it = landed_at_.begin(); it != landed_at_.end();) {
      const LogicalExpertId id = it->first;
      const LogicalExpert* expert = cache_.find(id);
      // Only the round that opened at the landing can take it over; once
      // another boundary has gone by unreserved, the row is a frustum row.
      if (expert == nullptr || expert->state != LifecycleState::Ready ||
          it->second >= GecConfig::kLandedLeaseWindows || drops_after_round_(id)) {
        drop_landed_(id);
        it = landed_at_.erase(it);
        continue;
      }
      ++it;
    }
  }

  // The routing layer takes over a look-ahead row (FreeToken: the probe of the
  // the probe of the layer decides which stage it consumes). The reservation is
  // what this claim ends; the binding took no reference of its own.
  void claim_lookahead_(const LogicalExpertId& id) {
    auto landed = landed_at_.find(id);
    if (landed == landed_at_.end()) return;
    landed_at_.erase(landed);
    staged_dispatch_.erase(id);
    // A live stage is consumed from staging; a row already handed back by the
    // cache (mark_ready refused it at the boundary) has nothing left to take.
    if (stage_index_.find(id) != stage_index_.end()) return;
    const LogicalExpert* expert = cache_.find(id);
    if (expert != nullptr && expert->state == LifecycleState::Ready) {
      // Consumption again, exactly like the boundary's own observation.
      cache_.record_use(id, now_());
    }
  }

  // The row the last boundary settled for this round and nobody routed since:
  // a landed look-ahead row is handed back to the eviction gate. A row whose
  // cache entry already expired (mark_ready refused it) has nothing left to
  // reclaim beyond finishing the stage.
  bool drops_after_round_(const LogicalExpertId& id) const {
    const LogicalExpert* expert = cache_.find(id);
    if (expert == nullptr || expert->state == LifecycleState::Evicting) return true;
    return expert->state == LifecycleState::Ready && expert->protected_;
  }

  // Hands a landed row nobody consumed back to the eviction gate: the bytes
  // never surface as a cache hit, the ring slot returns, and the drop is
  // charged to the frustum. A row another requester still holds stays.
  // A consumed row (pipeline entry staged_dispatch_) keeps its finished stage
  // in staging_ until compaction; a finish_stage_ there is harmless.
  void drop_landed_(const LogicalExpertId& id) {
    auto it = stage_index_.find(id);
    if (it != stage_index_.end()) {
      finish_stage_(it->second);
    } else {
      auto finished = std::find_if(staging_.begin(), staging_.end(),
                                   [&id](const StageRef& ref) {
                                     return ref.stage.expert == id && ref.stage.completed;
                                   });
      if (finished != staging_.end()) finished->gemm_end_us = 0;
    }
    // Consumption already ended the staging lease (and its slot); the row
    // still has a resident placement to hand back, so its own lease must not
    // keep it out of the eviction gate.
    cache_.set_protected(id, false);
    continue_evicting_(id);
    ++staging_delta_.frustum_dropped;
  }

  // Drops every unfinished stage, releases its H2D buffers and ends its
  // staging lease (frustum rollback / landing eviction).
  void finish_stage_(size_t index) {
    const ExpertStage stage = staging_[index].stage;
    for (H2DBuffer* buffer : stage.buffers) release_buffer(buffer);
    advance_pending_(stage.expert);
    stage_index_.erase(stage.expert);
    owning_round_.erase(stage.expert);
    // The dead entry stays in staging_ only until the next round's
    // refresh_staging_ compacts it; it must never be dispatched again.
    staging_[index].stage.completed = true;
    staging_[index].gemm_end_us = 0;
  }

  // A staged expert can be dropped only while its dependency is still
  // in-flight (not yet READY): the shards have not been published, so the
  // eviction gate can reclaim the rows. A landed look-ahead row nobody
  // consumed is handed back the same way (its bytes never surface as a hit).
  bool evictable_(const LogicalExpertId& id) const {
    const LogicalExpert* expert = cache_.find(id);
    if (expert == nullptr) return false;
    return (expert->state == LifecycleState::Scheduled ||
            expert->state == LifecycleState::Loading) &&
           !expert->in_flight;
  }

  // Frustum rollback (FreeToken): a look-ahead row nobody consumed is dropped,
  // not kept. An in-flight copy gives its buffers and lease back; a landed one
  // is handed over to the eviction gate (its bytes must never surface as a
  // cache hit). A row whose lease ran out is dropped whatever its state.
  bool drop_staged_(const LogicalExpertId& id) {
    auto it = stage_index_.find(id);
    if (it == stage_index_.end()) return false;
    const ExpertStage stage = staging_[it->second].stage;
    for (H2DBuffer* buffer : stage.buffers) release_buffer(buffer);
    reclaim_admission_(id);
    advance_pending_(id);
    stage_index_.erase(id);
    owning_round_.erase(id);
    landed_at_.erase(id);
    continue_evicting_(id);
    ++staging_delta_.frustum_dropped;
    return true;
  }

  void continue_evicting_(const LogicalExpertId& id) {
    if (cache_.admission_locked(id)) return;  // still held by other requesters
    if (cache_.find(id) != nullptr) cache_.evict(id);
  }

  // Ends one staging lease (acquire_pending) taken for this expert. The row
  // may still be LOADING; the last lease release lets the cache's eviction
  // gate reclaim it.
  void advance_pending_(const LogicalExpertId& id) {
    auto it = pending_refs_.find(id);
    if (it == pending_refs_.end()) return;
    if (--it->second <= 0) {
      pending_refs_.erase(it);
      cache_.release(id);
    }
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

  uint64_t round_id_ = 0;
  std::shared_ptr<RoundMeta> current_round_;
  std::vector<StageRef> staging_;
  std::unordered_map<LogicalExpertId, size_t, LogicalExpertIdHash> stage_index_;
  std::unordered_map<LogicalExpertId, const RoundMeta*, LogicalExpertIdHash> owning_round_;
  std::unordered_map<LogicalExpertId, int, LogicalExpertIdHash> pending_refs_;
  // Boundaries a landed look-ahead row has been observed at by a layer that did
  // not consume it. A consumer is consumption observed; a read taken at the
  // boundary is not, and a clock that stands still inside a layer cannot tell
  // the boundaries apart. The counter can: the row survives
  // kLandedLeaseWindows of them and is then handed back (frustum rollback).
  std::unordered_map<LogicalExpertId, int, LogicalExpertIdHash> landed_at_;
  // Reuse class of this round's candidates (the decision loop consumes it
  // after the scheduler selected the row).
  std::unordered_map<LogicalExpertId, RollingPipeline::H2DReuse, LogicalExpertIdHash>
      candidate_reuse_;
  std::vector<std::shared_ptr<RoundMeta>> finished_rounds_;
  StagingStats staging_delta_;
};

}  // namespace kt::gec
