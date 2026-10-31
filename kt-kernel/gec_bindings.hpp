// pybind11 bindings for the GEC core (doc/KT-CPU-PICE-GPU.md).
//
// Header-only like the core itself: ext_bindings.cpp includes this file and
// calls bind_gec(m) from PYBIND11_MODULE. The bindings expose the decision
// components (config, topology, scheduler, cache, telemetry); the async
// execution path lives in the caller (sglang's kt_gec driver).
#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kt/gec/expert_cache.hpp"
#include "kt/gec/gec_config.hpp"
#include "kt/gec/gec_types.hpp"
#include "kt/gec/inflight_registry.hpp"
#include "kt/gec/placement.hpp"
#include "kt/gec/scheduler.hpp"
#include "kt/gec/telemetry.hpp"

namespace kt::gec {

inline void bind_gec(pybind11::module_& m) {
  namespace py = pybind11;
  auto gec = m.def_submodule("gec", "Global Expert Cache (doc/KT-CPU-PICE-GPU.md)");

  py::class_<LogicalExpertId>(gec, "LogicalExpertId")
      .def(py::init<int, int>(), py::arg("layer_id"), py::arg("expert_id"))
      .def_readwrite("layer_id", &LogicalExpertId::layer_id)
      .def_readwrite("expert_id", &LogicalExpertId::expert_id)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__lt__", &LogicalExpertId::operator<)
      .def("__repr__",
           [](const LogicalExpertId& id) { return to_string(id); });

  py::class_<PhysicalShardId>(gec, "PhysicalShardId")
      .def(py::init<int, int, int>(), py::arg("layer_id"), py::arg("expert_id"),
           py::arg("tp_rank"))
      .def_readwrite("layer_id", &PhysicalShardId::layer_id)
      .def_readwrite("expert_id", &PhysicalShardId::expert_id)
      .def_readwrite("tp_rank", &PhysicalShardId::tp_rank)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__repr__",
           [](const PhysicalShardId& id) { return to_string(id); });

  py::enum_<LifecycleState>(gec, "LifecycleState")
      .value("ABSENT", LifecycleState::Absent)
      .value("SCHEDULED", LifecycleState::Scheduled)
      .value("LOADING", LifecycleState::Loading)
      .value("READY", LifecycleState::Ready)
      .value("EVICTING", LifecycleState::Evicting);

  py::enum_<AdmissionState>(gec, "AdmissionState")
      .value("NOT_ADMITTED", AdmissionState::NotAdmitted)
      .value("PROBATION", AdmissionState::Probation)
      .value("CACHED", AdmissionState::Cached);

  py::class_<GecConfig>(gec, "GecConfig")
      .def(py::init<>())
      .def_readwrite("num_gpu_layers", &GecConfig::num_gpu_layers)
      .def_readwrite("expert_gpu_slots", &GecConfig::expert_gpu_slots)
      .def_readwrite("h2d_round_budget", &GecConfig::h2d_round_budget)
      .def_readwrite("h2d_buffer_pool_depth", &GecConfig::h2d_buffer_pool_depth)
      .def_readwrite("priority_alpha", &GecConfig::priority_alpha)
      .def_readwrite("probation_uses", &GecConfig::probation_uses)
      .def_readwrite("lfu_aging_interval", &GecConfig::lfu_aging_interval)
      .def_readwrite("staging_prefetch_layers",
                     &GecConfig::staging_prefetch_layers)
      .def_readwrite("staging_lease_us", &GecConfig::staging_lease_us)
      .def("validate", &GecConfig::validate)
      .def_static("shard_count", &GecConfig::shard_count,
                  py::arg("logical_experts"), py::arg("tp_size"));

  py::class_<Topology>(gec, "Topology")
      .def(py::init<int, int, int>(), py::arg("tp_size") = 1,
           py::arg("ep_size") = 1, py::arg("gpu_count") = 1)
      .def_readwrite("tp_size", &Topology::tp_size)
      .def_readwrite("ep_size", &Topology::ep_size)
      .def_readwrite("gpu_count", &Topology::gpu_count)
      .def("validate", &Topology::validate);

  py::class_<SchedulerCandidate>(gec, "SchedulerCandidate")
      .def(py::init<>())
      .def_readwrite("id", &SchedulerCandidate::id)
      .def_readwrite("current_demand", &SchedulerCandidate::current_demand)
      .def_readwrite("historical_hit_count", &SchedulerCandidate::historical_hit_count)
      .def_readwrite("inflight", &SchedulerCandidate::inflight)
      .def_readwrite("reuse_inflight", &SchedulerCandidate::reuse_inflight)
      .def_readwrite("evicting", &SchedulerCandidate::evicting)
      .def_readwrite("placement_available", &SchedulerCandidate::placement_available)
      .def_readwrite("buffer_available", &SchedulerCandidate::buffer_available)
      .def_readwrite("cache_admission_locked", &SchedulerCandidate::cache_admission_locked)
      .def("eligible", &SchedulerCandidate::eligible);

  py::class_<SchedulerDecision>(gec, "SchedulerDecision")
      .def_readonly("h2d_selected", &SchedulerDecision::h2d_selected)
      .def_readonly("cpu_fallback", &SchedulerDecision::cpu_fallback);

  py::class_<H2DScheduler>(gec, "H2DScheduler")
      .def(py::init<const GecConfig&>(), py::arg("config"))
      .def("schedule", &H2DScheduler::schedule, py::arg("candidates"),
           py::arg("budget_remaining"));

  py::class_<GlobalExpertCache::Stats>(gec, "CacheStats")
      .def_readonly("hits", &GlobalExpertCache::Stats::hits)
      .def_readonly("misses", &GlobalExpertCache::Stats::misses)
      .def_readonly("admissions", &GlobalExpertCache::Stats::admissions)
      .def_readonly("evictions", &GlobalExpertCache::Stats::evictions)
      .def_readonly("reh2d", &GlobalExpertCache::Stats::reh2d)
      .def_readonly("lfu_events", &GlobalExpertCache::Stats::lfu_events)
      .def_readonly("lfu_aging_sweeps",
                    &GlobalExpertCache::Stats::lfu_aging_sweeps)
      .def_readonly("router_rounds", &GlobalExpertCache::Stats::router_rounds)
      .def("hit_rate", &GlobalExpertCache::Stats::hit_rate);

  py::class_<GlobalExpertCache::ExpertSnapshot>(gec, "ExpertSnapshot")
      .def_readonly("id", &GlobalExpertCache::ExpertSnapshot::id)
      .def_readonly("state", &GlobalExpertCache::ExpertSnapshot::state)
      .def_readonly("refcount", &GlobalExpertCache::ExpertSnapshot::refcount)
      .def_readonly("hit_count", &GlobalExpertCache::ExpertSnapshot::hit_count)
      .def_readonly("lfu_count", &GlobalExpertCache::ExpertSnapshot::lfu_count)
      .def_property_readonly(
          "use_count",
          [](const GlobalExpertCache::ExpertSnapshot& snapshot) {
            return snapshot.lfu_count;
          })
      .def_readonly("last_used", &GlobalExpertCache::ExpertSnapshot::last_used)
      .def_readonly("logical_ready_dependency",
                    &GlobalExpertCache::ExpertSnapshot::logical_ready_dependency)
      .def_readonly("in_flight", &GlobalExpertCache::ExpertSnapshot::in_flight)
      .def_readonly("protected_", &GlobalExpertCache::ExpertSnapshot::protected_)
      .def_readonly("pinned", &GlobalExpertCache::ExpertSnapshot::pinned)
      .def_readonly("admission_state",
                    &GlobalExpertCache::ExpertSnapshot::admission_state)
      .def_readonly("layout_id",
                    &GlobalExpertCache::ExpertSnapshot::layout_id);

  py::enum_<GlobalExpertCache::LookupResult>(gec, "LookupResult")
      .value("HIT", GlobalExpertCache::LookupResult::Hit)
      .value("MISS", GlobalExpertCache::LookupResult::Miss);

  py::class_<GlobalExpertCache>(gec, "GlobalExpertCache")
      .def(py::init<GecConfig, Topology>(), py::arg("config"), py::arg("topology"))
      .def("lookup", &GlobalExpertCache::lookup, py::arg("id"), py::arg("now_us") = 0)
      .def("admission_locked", &GlobalExpertCache::admission_locked, py::arg("id"))
      .def("admit", &GlobalExpertCache::admit, py::arg("id"), py::arg("now_us") = 0)
      .def_static("sequential_expert_ids", &GlobalExpertCache::sequential_expert_ids,
                  py::arg("num_layers"), py::arg("experts_per_layer"))
      .def("prefill_sequential", &GlobalExpertCache::prefill_sequential,
           py::arg("ordered_ids"), py::arg("now_us") = 0)
      .def("mark_scheduled", &GlobalExpertCache::mark_scheduled, py::arg("id"))
      .def("mark_loading", &GlobalExpertCache::mark_loading, py::arg("id"))
      .def("cancel_loading", &GlobalExpertCache::cancel_loading, py::arg("id"))
      .def("bind_shard_dependency", &GlobalExpertCache::bind_shard_dependency,
           py::arg("id"), py::arg("tp_rank"), py::arg("ready_dependency"))
      .def("shard_dependency", &GlobalExpertCache::shard_dependency,
           py::arg("id"), py::arg("tp_rank"))
      .def("mark_shard_ready", &GlobalExpertCache::mark_shard_ready, py::arg("id"),
           py::arg("tp_rank"), py::arg("now_us") = 0,
           py::arg("ready_dependency") = 0)
      .def("mark_evicting", &GlobalExpertCache::mark_evicting, py::arg("id"))
      .def("mark_absent", &GlobalExpertCache::mark_absent, py::arg("id"))
      .def("evict", &GlobalExpertCache::evict, py::arg("id"))
      .def("acquire", &GlobalExpertCache::acquire, py::arg("id"))
      .def("acquire_pending", &GlobalExpertCache::acquire_pending, py::arg("id"))
      .def("release", &GlobalExpertCache::release, py::arg("id"))
      .def("refcount", &GlobalExpertCache::refcount, py::arg("id"))
      .def("record_use", &GlobalExpertCache::record_use, py::arg("id"),
           py::arg("now_us") = 0, py::arg("weight") = 1)
      .def("record_router_demand", &GlobalExpertCache::record_router_demand,
           py::arg("id"), py::arg("token_count"), py::arg("now_us") = 0)
      .def("begin_router_round", &GlobalExpertCache::begin_router_round)
      .def("router_epoch", &GlobalExpertCache::router_epoch)
      .def("end_startup_protection",
           &GlobalExpertCache::end_startup_protection, py::arg("id"))
      .def("set_pinned", &GlobalExpertCache::set_pinned, py::arg("id"),
           py::arg("pinned") = true)
      .def("is_pinned", &GlobalExpertCache::is_pinned, py::arg("id"))
      .def("record_h2d", &GlobalExpertCache::record_h2d, py::arg("id"))
      .def("snapshot", &GlobalExpertCache::snapshot, py::arg("id"))
      .def("resident_ids", &GlobalExpertCache::resident_ids)
      .def("resident_size", &GlobalExpertCache::resident_size)
      .def("slot_resident_size", &GlobalExpertCache::slot_resident_size)
      .def("register_pool", &GlobalExpertCache::register_pool,
           py::arg("layout_id"), py::arg("capacity"))
      .def("assign_layout", &GlobalExpertCache::assign_layout,
           py::arg("id"), py::arg("layout_id"))
      .def("layout_of", &GlobalExpertCache::layout_of, py::arg("id"))
      .def("pool_capacity", &GlobalExpertCache::pool_capacity,
           py::arg("layout_id"))
      .def("pool_resident_size", &GlobalExpertCache::pool_resident_size,
           py::arg("layout_id"))
      .def("stats", &GlobalExpertCache::stats,
           py::return_value_policy::reference_internal);

  py::class_<InflightRegistry>(gec, "InflightRegistry")
      .def(py::init<>())
      .def("register_transfer", &InflightRegistry::register_transfer,
           py::arg("shard"), py::arg("dependency_id"), py::arg("started_at_us") = 0)
      .def("contains", &InflightRegistry::contains, py::arg("shard"))
      .def("dependency_id", &InflightRegistry::dependency_id, py::arg("shard"))
      .def("complete", &InflightRegistry::complete, py::arg("shard"))
      .def("size", &InflightRegistry::size)
      .def("clear", &InflightRegistry::clear);

  py::enum_<WaitReason>(gec, "WaitReason")
      .value("ROUTER_DEPENDENCY", WaitReason::RouterDependency)
      .value("H2D_QUEUED", WaitReason::H2DQueued)
      .value("H2D_RUNNING", WaitReason::H2DRunning)
      .value("H2D_EVENT_WAIT", WaitReason::H2DEventWait)
      .value("CPU_FALLBACK", WaitReason::CpuFallback)
      .value("CACHE_WAIT", WaitReason::CacheWait);

  py::class_<WaitRecord>(gec, "WaitRecord")
      .def(py::init<>())
      .def_readwrite("layer_id", &WaitRecord::layer_id)
      .def_readwrite("expert_id", &WaitRecord::expert_id)
      .def_readwrite("tp_rank", &WaitRecord::tp_rank)
      .def_readwrite("wait_start_us", &WaitRecord::wait_start_us)
      .def_readwrite("wait_end_us", &WaitRecord::wait_end_us)
      .def_readwrite("wait_duration_us", &WaitRecord::wait_duration_us)
      .def_readwrite("reason", &WaitRecord::reason);

  py::class_<ReasonStats>(gec, "ReasonStats")
      .def_readonly("count", &ReasonStats::count)
      .def_readonly("total_us", &ReasonStats::total_us)
      .def_readonly("p50_us", &ReasonStats::p50_us)
      .def_readonly("p90_us", &ReasonStats::p90_us)
      .def_readonly("p99_us", &ReasonStats::p99_us);

  py::class_<StagingStats>(gec, "StagingStats")
      .def(py::init<>())
      .def_readonly("begin_rounds", &StagingStats::begin_rounds)
      .def_readonly("prefetch_submits", &StagingStats::prefetch_submits)
      .def_readonly("consumed_from_prefetch",
                    &StagingStats::consumed_from_prefetch)
      .def_readonly("stall_waits", &StagingStats::stall_waits)
      .def_readonly("fallbacks", &StagingStats::fallbacks)
      .def_readonly("direct_hits", &StagingStats::direct_hits)
      .def_readonly("frustum_dropped", &StagingStats::frustum_dropped);

  py::class_<TelemetryCollector>(gec, "TelemetryCollector")
      .def(py::init<>())
      .def("record_wait", &TelemetryCollector::record_wait, py::arg("record"))
      .def("aggregate", &TelemetryCollector::aggregate)
      .def("wait_count", &TelemetryCollector::wait_count)
      .def("staging", &TelemetryCollector::staging)
      .def("record_staging", &TelemetryCollector::record_staging,
           py::arg("delta"))
      .def("summary", &TelemetryCollector::summary);
}

}  // namespace kt::gec
