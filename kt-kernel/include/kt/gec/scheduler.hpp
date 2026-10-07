// H2D candidate filter, priority and Top-N selection
// (doc/KT-CPU-PICE-GPU.md sections 12-15, 20).
//
// Pipeline: Current MISS -> Candidate Filter -> Priority -> Top-N ->
// H2D queue. Everything not selected goes to CPU fallback.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "kt/gec/gec_config.hpp"
#include "kt/gec/gec_types.hpp"

namespace kt::gec {

struct SchedulerCandidate {
  LogicalExpertId id;
  uint64_t current_demand = 0;
  uint64_t historical_hit_count = 0;

  // Candidate Filter inputs (doc section 12). Any true flag disqualifies the
  // candidate from the H2D path for this round.
  bool inflight = false;
  // An inflight request may be selected when the caller can reuse its
  // dependency.  This is distinct from submitting a new transfer.
  bool reuse_inflight = false;
  bool evicting = false;
  bool placement_available = true;
  bool buffer_available = true;
  bool cache_admission_locked = false;

  bool eligible() const {
    return (!inflight || reuse_inflight) && !evicting && placement_available && buffer_available &&
           !cache_admission_locked;
  }
};

struct SchedulerDecision {
  // Top-N H2D admissions for this round: bounded only by the remaining
  // admission budget (kt-layer-h2d-slots). The batch size does not limit how
  // many experts may take the H2D path in a round (doc sections 5.3 and 15).
  std::vector<LogicalExpertId> h2d_selected;
  // Everything else: CPU fallback path.
  std::vector<LogicalExpertId> cpu_fallback;

  // Splits the round selection into H2D batches of at most batch_size
  // experts (kt-layer-h2d-batch-size). The final batch carries the remainder:
  // batch = min(available, batch_size), never waits to be filled.
  std::vector<std::vector<LogicalExpertId>> h2d_batches(int batch_size) const {
    std::vector<std::vector<LogicalExpertId>> batches;
    if (batch_size < 1) return batches;
    const size_t cap = static_cast<size_t>(batch_size);
    for (size_t begin = 0; begin < h2d_selected.size(); begin += cap) {
      const size_t end = std::min(begin + cap, h2d_selected.size());
      batches.emplace_back(h2d_selected.begin() + static_cast<ptrdiff_t>(begin),
                           h2d_selected.begin() + static_cast<ptrdiff_t>(end));
    }
    return batches;
  }
};

class H2DScheduler {
 public:
  explicit H2DScheduler(const GecConfig& config) : config_(config) {
    config_.validate();
  }

  // Priority = current_demand + alpha * historical_hit_count. Current demand
  // dominates; history only orders misses against each other (doc section 13).
  static double priority(const SchedulerCandidate& candidate, double alpha) {
    return static_cast<double>(candidate.current_demand) +
           alpha * static_cast<double>(candidate.historical_hit_count);
  }

  SchedulerDecision schedule(std::vector<SchedulerCandidate> candidates,
                             int budget_remaining) const {
    SchedulerDecision decision;
    std::vector<size_t> eligible;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].eligible()) eligible.push_back(i);
    }
    const double alpha = config_.priority_alpha;
    std::stable_sort(eligible.begin(), eligible.end(),
                     [&candidates, alpha](size_t a, size_t b) {
                       double pa = priority(candidates[a], alpha);
                       double pb = priority(candidates[b], alpha);
                       if (pa != pb) return pa > pb;  // higher priority first
                       return candidates[a].id < candidates[b].id;  // stable tie-break
                     });
    if (budget_remaining < 0) budget_remaining = 0;
    const size_t budget = static_cast<size_t>(budget_remaining);
    // Top-N admission is bounded by the per-round budget only; the batch size
    // chunks the submissions afterwards (doc sections 5.3, 5.4 and 15).
    size_t selected = std::min(eligible.size(), budget);
    for (size_t k = 0; k < selected; ++k) {
      decision.h2d_selected.push_back(candidates[eligible[k]].id);
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
      bool chosen = false;
      for (const LogicalExpertId& picked : decision.h2d_selected) {
        if (picked == candidates[i].id) { chosen = true; break; }
      }
      if (!chosen) decision.cpu_fallback.push_back(candidates[i].id);
    }
    return decision;
  }

 private:
  GecConfig config_;
};

}  // namespace kt::gec
