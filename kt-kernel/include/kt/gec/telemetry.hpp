// GPU Expert Wait telemetry (doc/KT-CPU-PICE-GPU.md sections 23-26).
//
// Decomposes every GPU expert wait into exactly one of six reasons and keeps
// full H2D / GEMM timelines so the 150-180 ms blank time can be attributed.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "kt/gec/gec_sync.hpp"
#include "kt/gec/gec_types.hpp"

namespace kt::gec {

enum class WaitReason : uint8_t {
  RouterDependency = 0,
  H2DQueued = 1,
  H2DRunning = 2,
  H2DEventWait = 3,
  CpuFallback = 4,
  CacheWait = 5,
};

inline constexpr size_t kWaitReasonCount = 6;

inline const char* to_string(WaitReason reason) {
  switch (reason) {
    case WaitReason::RouterDependency: return "ROUTER_DEPENDENCY";
    case WaitReason::H2DQueued: return "H2D_QUEUED";
    case WaitReason::H2DRunning: return "H2D_RUNNING";
    case WaitReason::H2DEventWait: return "H2D_EVENT_WAIT";
    case WaitReason::CpuFallback: return "CPU_FALLBACK";
    case WaitReason::CacheWait: return "CACHE_WAIT";
  }
  return "UNKNOWN";
}

struct WaitRecord {
  int layer_id = -1;
  int expert_id = -1;
  int tp_rank = -1;
  uint64_t wait_start_us = 0;
  uint64_t wait_end_us = 0;
  uint64_t wait_duration_us = 0;
  WaitReason reason = WaitReason::RouterDependency;
};

struct H2DTimeline {
  LogicalExpertId expert;
  int tp_rank = -1;
  uint64_t queue_enter_us = 0;
  uint64_t h2d_start_us = 0;
  uint64_t h2d_end_us = 0;
  uint64_t event_wait_start_us = 0;
  uint64_t event_ready_us = 0;
  uint64_t gemm_start_us = 0;
  uint64_t gemm_end_us = 0;

  uint64_t queue_latency_us() const { return h2d_start_us - queue_enter_us; }
  uint64_t transfer_latency_us() const { return h2d_end_us - h2d_start_us; }
  uint64_t event_wait_us() const { return event_ready_us - event_wait_start_us; }
  uint64_t gemm_latency_us() const { return gemm_end_us - gemm_start_us; }
};

struct ReasonStats {
  uint64_t count = 0;
  uint64_t total_us = 0;
  uint64_t p50_us = 0;
  uint64_t p90_us = 0;
  uint64_t p99_us = 0;
};

class TelemetryCollector {
 public:
  void record_wait(const WaitRecord& record) {
    GecLockGuard lock(mutex_);
    waits_.push_back(record);
  }

  void record_h2d(const H2DTimeline& timeline) {
    GecLockGuard lock(mutex_);
    h2d_timelines_.push_back(timeline);
  }

  std::array<ReasonStats, kWaitReasonCount> aggregate() const {
    GecLockGuard lock(mutex_);
    std::array<ReasonStats, kWaitReasonCount> result{};
    std::array<std::vector<uint64_t>, kWaitReasonCount> durations{};
    for (const WaitRecord& record : waits_) {
      size_t index = static_cast<size_t>(record.reason);
      result[index].count += 1;
      result[index].total_us += record.wait_duration_us;
      durations[index].push_back(record.wait_duration_us);
    }
    for (size_t i = 0; i < kWaitReasonCount; ++i) {
      if (durations[i].empty()) continue;
      std::vector<uint64_t>& sorted = durations[i];
      std::sort(sorted.begin(), sorted.end());
      auto percentile = [&sorted](double q) -> uint64_t {
        double pos = q * static_cast<double>(sorted.size() - 1);
        size_t index = static_cast<size_t>(pos + 0.5);
        return sorted[std::min(index, sorted.size() - 1)];
      };
      result[i].p50_us = percentile(0.50);
      result[i].p90_us = percentile(0.90);
      result[i].p99_us = percentile(0.99);
    }
    return result;
  }

  size_t wait_count() const {
    GecLockGuard lock(mutex_);
    return waits_.size();
  }

  size_t h2d_count() const {
    GecLockGuard lock(mutex_);
    return h2d_timelines_.size();
  }

  std::string summary() const {
    std::array<ReasonStats, kWaitReasonCount> stats = aggregate();
    std::string out;
    for (size_t i = 0; i < kWaitReasonCount; ++i) {
      out += to_string(static_cast<WaitReason>(i));
      out += ": count=" + std::to_string(stats[i].count);
      out += " total_us=" + std::to_string(stats[i].total_us);
      out += " p50_us=" + std::to_string(stats[i].p50_us);
      out += " p90_us=" + std::to_string(stats[i].p90_us);
      out += " p99_us=" + std::to_string(stats[i].p99_us);
      out.push_back(static_cast<char>(10));
    }
    return out;
  }

 private:
  mutable GecMutex mutex_;
  std::vector<WaitRecord> waits_;
  std::vector<H2DTimeline> h2d_timelines_;
};

}  // namespace kt::gec
