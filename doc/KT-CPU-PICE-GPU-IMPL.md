# Global Expert Cache + H2D 滚动流水 — V1 实现记录

> 依据：`doc/KT-CPU-PICE-GPU.md`（设计方案）、`doc/KT-CPU-PICE-GPU-DEV-PLAN.md`（开发计划）

## 1. 实现范围

V1 按开发计划完成 M0-M5 全部里程碑，交付一个**纯 C++17、头文件式、零 CUDA 依赖**的核心模块：

```text
kt-kernel/include/kt/gec/
```

所有 CUDA 交互（Stream / Event / Memcpy / GEMM）通过 `PipelineHooks` 依赖注入抽象，
因此全部生命周期不变量（Buffer 复用、refcount、TP Atomic Ready、Inflight 去重）
可在任意主机上验证，并可直接接入真实 GPU 后端。

## 2. 里程碑完成状态

| 里程碑 | 状态 | 交付内容 |
|--------|------|----------|
| M0 数据模型与状态机 | 完成 | Logical/Physical 双层模型、生命周期状态机、三维分离、参数校验 |
| M1 Placement 与 Inflight Registry | 完成 | TP/EP 全量 Placement、TP Atomic Ready、H2D 去重注册表 |
| M2 H2D 滚动流水 | 完成 | Filter→Priority→Top-N 调度链、每 GPU Buffer Pool、双路径编排 |
| M3 Telemetry | 完成 | 6 类 Wait 拆解、H2D 时间线、P50/P90/P99 聚合 |
| M4 Cache 策略 | 完成 | Logical 粒度 Admission、整体 Eviction、Probation、Re-H2D 统计 |
| M5 集成验证 | 完成 | 278 项 C++ 检查、12 项 CI 合同测试、150ms 拆解演示 |

## 3. 模块清单

| 头文件 | 职责 | 对应设计章节 |
|--------|------|--------------|
| `gec_types.hpp` | ID、状态枚举、LogicalExpert / PhysicalShard、`is_logical_ready`、`can_evict` | 3 / 7 / 8 / 9 / 10 |
| `gec_config.hpp` | 4 个运行参数 + 校验 + Logical/Shard 单位换算 | 5 |
| `gec_sync.hpp` | 可移植互斥量（MinGW win32 线程模型降级为空锁） | - |
| `placement.hpp` | Topology（TP/EP/GPU）、全量 Placement Planner | 4 |
| `inflight_registry.hpp` | 同一 Shard 最多一个 active H2D，重复请求复用 dependency | 11 |
| `scheduler.hpp` | Candidate Filter、Priority、Top-N（预算 = kt-layer-h2d-slots）、按 kt-layer-h2d-batch-size 分批（`batch = min(available, batch_size)`） | 5 / 12 / 13 / 15 / 20 |
| `buffer_pool.hpp` | 每 GPU Buffer Pool，`FREE→H2D→READY→GPU_IN_USE→COMPLETION_WAIT→FREE` | 15 / 16 |
| `expert_cache.hpp` | 持久缓存：Admission / LRU Eviction / Probation / refcount / Re-H2D | 6 / 17 / 18 / 19 |
| `telemetry.hpp` | 6 类 Wait 指标、H2D 时间线、分位数聚合、summary | 23 / 24 / 26 |
| `pipeline.hpp` | `run_round` 编排：HIT / H2D / Fallback 双路径 + Merge + Hooks | 14 / 21 / 22 |

## 4. 关键设计决策

### 4.1 三维分离

`state`（驻留）、`logical_ready_dependency`（以 shard 就绪态聚合表达）、`refcount`
（使用计数）完全独立；`READY + refcount == 0` 与 `READY + refcount > 0` 均合法，
不引入 `IN_USE` 伪状态。

SGLang 的 GPU execution lease 在 H2D 发布 READY 之前通过 `acquire_pending`
预留 refcount，并在 Compute Stream completion event 后释放，避免新换入 row 在
异步 GPU Kernel 仍运行时被跨层 eviction。

### 4.2 H2D 与 Cache Admission 绑定

生产 GEC 路径中，只有已经确认需要 H2D 的 MISS 才进入 Persistent Cache Admission；
准入成功后才提交 H2D，准入失败直接走 CPU fallback，不存在独立的“不进入 Cache
的 Temporary H2D”路径。`RollingPipeline` 与 SGLang 运行时使用相同的顺序。

### 4.3 同轮重复请求去重

同一轮内重复 Expert：首个请求走 H2D 并驻留后，后续请求以 `H2DOutcome::FromCache`
直接复用驻留副本执行，不产生重复传输（验收标准 2）。

### 4.4 无全局同步

模块不包含 `cudaStreamSynchronize` / `cudaDeviceSynchronize` / 同步 `cudaMemcpy`；
CI 合同测试 `test_no_global_cuda_synchronization` 强制守护（验收标准 6）。

### 4.5 CPU / PCIe / GPU 并行模型

一轮 `run_round` 同时调度三条互不阻塞的路径（设计 21 节）：

```text
CPU:   [CPU GEMM D]        <- Fallback 专家立即执行
PCIe:  [H2D C]             -> 缺失专家并行传输
GPU:   [GPU GEMM A][...]   <- 驻留专家（HIT）立即执行
```

- 驻留专家（HIT）：`refcount++ → GPU GEMM → completion → refcount--`，立即执行；
- 当前层缺失专家：Filter → Priority → 本层 Top-N → H2D Buffer → PCIe 传输 → GPU GEMM；
- 未入选专家：CPU Fallback 立即执行，输出与 GPU 路径汇入 Merge。
- 当前层在 H2D 提交前根据 Router 真实需求完成 Persistent Cache Admission
  和必要的跨层冷槽迁移；执行完成后只更新命中历史、Probation 和 Eviction
  统计，不回头改变已经提交的当前层 H2D。

仿真层按 命中 → H2D → Fallback 的程序顺序同步推进，但三条路径的依赖完全独立
（时间戳均锚定 `round_start`），接入真实异步后端（CPU 线程池 / 拷贝流 / 计算流）
后即形成物理重叠。

### 4.5.1 当前层 Router 的执行归属

SGLang wrapper 的运行时顺序必须与上面的调度模型一致：

```text
materialize top-k
    ↓
GEC lookup
    ├── resident HIT：复用已有 GPU row，不产生 H2D
    └── CPU MISS：按当前 demand 做 Top-N
                    ├── selected：H2D 到 GPU row
                    └── unselected：CPU fallback
    ↓
更新 gpu_experts_mask / logical_to_gpu_index
    ↓
提交 CPU task（只保留最终 mask=False 的 IDs）
    ↓
GPU mask/remap → GPU GEMM → CPU/GPU merge
```

这里的关键约束是 **先更新最终 resident mask，再提交 CPU task**。如果先提交
CPU task、再做 GEC admission，新换入的 Expert 仍可能被 CPU wrapper 看到，造成
CPU GEMM 与 H2D/GPU GEMM 的重复工作。实现中 `mask_gpu_expert_ids_for_cpu()`
会把最终 GPU resident 的路由改成 `-1`，CPU kernel 对这些 sentinel 直接跳过；
未选中的 CPU MISS 才保留给 CPU fallback。

对于 DeepSeek-V4 的 MXFP4，GEC 不再走当前 batch 全量 Expert 的 temporary
GEMM。MXFP4 的 CPU MISS 先通过 CPU buffer 写入 persistent row 的 raw
`w13/w2`，只对新到达的 row 做 Marlin repack，然后使用原 layer 的 compact
resident image 执行 GPU GEMM；GPU HIT 不搬运，未入选 MISS 保留给 CPU fallback。

当 INT4/Marlin 进入 GEC hybrid selected-row copy 路径时，GEC Top-N 仍由
`kt-layer-h2d-slots` 决定，选中 Expert 直接从 CPU raw packed buffer 写入 selected
Persistent GPU row 并做局部 Marlin repack，CPU writer 按 Expert/tensor/row 循环执行。
full-shadow 只保留给 legacy/full-GPU fallback 路径。满足 full-GPU extend 条件时，
packed-format 路径直接处理当前 batch 的全部 active Expert，绕过 GEC 的 H2D
Top-N/CPU fallback 分类。

GEC H2D window 内的 CPU writer 使用与 window 等大的固定 host buffer ring：先提交
window 内所有 Expert 的写入任务，再执行一次 CPU writer drain 和一次 TP barrier，
随后批量提交 H2D/repack。`ready_event` 在每个 window 完成后发布，GPU stream 按
window 等待；MXFP4 的每个 window 只执行一次固定 scratch 容量的 repack，并按
Expert group 输出累加，避免把整个 resident 集合一次性复制到临时张量。

### 4.6 Admission Budget 与 Batch Size 分离

单轮 H2D Admission 由**当前层**的 `kt-layer-h2d-slots` 限制（本层 Top-N），
`kt-layer-h2d-batch-size` 只决定单次提交批次大小
（`batch = min(available, batch_size)`，不凑满），Buffer Pool 深度在批次之间
提供背压（设计 5.3 / 5.4 / 15 节）。例如 `slots=64, batch-size=8` 时，
当前层最多 64 个 Logical Expert 立即走 H2D 路径，分 8 批提交；不需要等待其他层。

临时 Buffer Pool 深度由服务并发和批次大小共同决定：

```text
inflight_batches = max-running-requests
temporary_slots = inflight_batches × kt-layer-h2d-batch-size
```

例如 `max-running-requests=17`、`batch-size=8` 时，Pool 深度为 17，临时
Logical slots 上限为 136；TP=4 时最多展开为 544 个 Physical Shard Buffer。
该上限只约束临时 copy/event pipeline，不占用 `kt-expert-gpu-slots`。

### 4.7 全局容量与跨层物理行迁移

`kt-expert-gpu-slots` 是 `kt-num-gpu-layers` 之后、且通过
`first_k_dense_replace/moe_layer_freq` 过滤的可管理 MoE 层合计的 Logical Expert 容量。启动物理行按
layer-major、expert-id-minor 顺序填充；例如 60 层、每层 256 专家、`slots=2560`
时，若 `kt-num-gpu-layers=0`，前 10 个 eligible MoE 层各拥有 256 行，后续层初始为 0 行；
如果 `kt-num-gpu-layers>0`，则从第一个可管理 MoE 层开始计算这 10 层。

设计上的运行时契约是：每层在 H2D/CPU 执行前将当前 batch 的 Router 激活计数
提交给模型级 `GlobalGecCoordinator`。Coordinator 在 Persistent Cache Admission
阶段跨层比较热度；当高热层需要更多常驻行时，从低热度 donor layer 借行：

1. donor 只释放 `READY && refcount == 0 && !inflight` 的完整 Logical Expert；
2. donor compact GPU 权重行按保留 owner 缩容；
3. target layer 扩容，本轮即可对被提前准入的常驻专家执行 H2D/copy；
4. 所有 TP rank 更新相同的容量、row owner 和 routing mapping。

因此运行一段时间后各层容量可以是不均匀的，但总物理行数始终受全局
`kt-expert-gpu-slots` 限制。`kt-layer-h2d-slots` 只限制当前层当前轮即时 H2D
换入的 Logical Expert 数量；`kt-layer-h2d-batch-size` 只拆分提交批次；二者都
不负责决定跨层 Persistent Cache 的最终保留集合。

当前 SGLang planner 在规划阶段预留/更新常驻行，这与“提前决定是否长期驻留”
的契约一致。普通 hybrid 路径的 INT4/Marlin selected Expert 直接写入
Persistent GPU rows 并做局部 repack，按 Expert/tensor/row 循环；full-shadow 只保留
给 legacy/full-GPU fallback。packed-format full-GPU extend 路径则处理当前 batch
的全部 active Expert。进一步将拷贝改成向量化仍属于后续性能优化。

## 5. 验收标准对照

| # | 设计 29 节验收项 | 验证测试 |
|---|------------------|----------|
| 1 | Logical Expert 正确（TP Atomic Ready） | `test_lifecycle_and_tp_atomic_ready` |
| 2 | 无重复 H2D | `test_inflight_registry`、`test_pipeline_no_duplicate_h2d` |
| 3 | H2D 需要 Cache Admission | `test_pipeline_requires_cache_admission` |
| 4 | Buffer 不提前复用 | `test_buffer_lifecycle`、`test_pipeline_buffer_recycling` |
| 5 | refcount 不提前释放 | `test_refcount_semantics`、`test_pipeline_refcount_during_gemm` |
| 6 | 不存在全局同步 | `test_no_global_cuda_synchronization`（CI 合同） |
| 7 | 150~180 ms 可完整拆解 | `test_telemetry_decomposition`、`run_diagnostics_demo` |

## 6. 测试结果

### C++ 套件（`kt-kernel/test/gec/test_gec.cpp`）

```text
==== GEC test summary: 278 checks, 0 failures ====
```

覆盖 M0-M4 单元行为 + M2/M5 集成路径 + M3 遥测分解，含 Budget/Batch 分离与
Buffer 回滚（`release_unstarted`）回归。

### CI 合同测试（`kt-kernel/test/per_commit/test_gec_contract.py`）

```text
12 passed
```

包含结构守护（禁全局同步、六类指标、batch 语义、解耦开关）与 C++ 套件联动。

### 150ms 拆解演示输出

```text
ROUTER_DEPENDENCY: count=3 total_us=0
H2D_QUEUED:        count=1 total_us=5000
H2D_RUNNING:       count=1 total_us=8000
H2D_EVENT_WAIT:    count=1 total_us=120000
CPU_FALLBACK:      count=1 total_us=15000
CACHE_WAIT:        count=1 total_us=2000
```

合计 150 ms，与设计 27 节 Case A（Event Wait 主导）形态一致。

## 7. 运行方式

```bash
# 方式一：直接编译运行（无需 CMake）
python kt-kernel/test/gec/run_tests.py

# 方式二：CMake
cmake -S kt-kernel/test/gec -B build-gec && cmake --build build-gec && ctest --test-dir build-gec

# 方式三：CI 合同测试（含源码守护 + C++ 套件）
python -m pytest kt-kernel/test/per_commit/test_gec_contract.py -v
```

## 8. 启动顺序预填

`GlobalExpertCache::prefill_sequential`（pipeline 侧同名入口）实现启动时按顺序
初始化 `kt-expert-gpu-slots`：

```text
sequential_expert_ids(num_layers, experts_per_layer)
    -> (layer 0, expert 0), (layer 0, expert 1), ..., (layer 1, expert 0), ...
    -> prefill_sequential(ordered_ids)
        -> layers < kt-num-gpu-layers: 永久驻留，不占 slot
        -> layers >= kt-num-gpu-layers: 顺序填充 slot
        -> SCHEDULED -> LOADING -> READY
        -> 直到 kt-expert-gpu-slots 满（从 kt-num-gpu-layers 之后开始计数）
```

语义：

- 顺序为 layer-major、expert-id-minor；
- 容量以 Logical Expert 计（TP=4 时一个专家占 4 个 Shard）；
- `kt-num-gpu-layers` 内的专家永久驻留且**不占 slot**；
- `kt-expert-gpu-slots` 从 layer `kt-num-gpu-layers` 之后开始顺序填充，
  即首个 slot 专家为 `(kt-num-gpu-layers, 0)`；
- slot 预填专家处于 Probation 保护期，防止启动后立即被驱逐；
- 预填完成后首轮即可 HIT，PCIe 继续并行传输缺失专家。

验证：`test_startup_prefill`、`test_startup_prefill_permanent_layers`、
`test_prefill_starts_after_gpu_layers`、`test_prefill_then_parallel_paths`。

## 9. sglang 集成（M6）

GEC 核心通过 pybind11 暴露为 `kt_kernel.gec`（`kt-kernel/gec_bindings.hpp`，
注册于 `ext_bindings.cpp`），sglang 侧由 `sglang/srt/layers/moe/kt_gec.py`
驱动，接入现有的动态专家更新机制（`KTEPWrapperMethod._update_gpu_experts_from_batch`）：

```text
Router (topk_ids)
    ↓ (current layer only)
GecResidencyPlanner.plan(counts)（当前层即时 H2D + 提前常驻计划）
    ↓ kt_kernel.gec：lookup → Candidate Filter → Priority → 本层 Top-N（kt-layer-h2d-slots）
本层 H2D / CPU Fallback → 本层 GPU/CPU 执行
    ↓ activation counts
GlobalGecCoordinator：跨层 Persistent Cache Admission / 冷槽预留与 Eviction
    ↓ cross-layer row rebalance + TP-consistent mapping
copy_experts_weights_*（对提前准入的常驻专家按 kt-layer-h2d-batch-size 分批；当前
CPU writer 在 batch 内逐 Expert/逐 tensor/逐 row 写入，常驻集合不变的轮次整体跳过
拷贝与映射更新）
    ↓
update_gpu_expert_mappings / KT wrapper mask 同步（复用既有路径）
```

参数语义（三者均为 Logical Expert 计数）：

| 参数 | 语义 | 单独使用 | 配合 --kt-enable-dynamic-expert-update |
|------|------|----------|----------------------------------------|
| `--kt-expert-gpu-slots` | 全局驻留容量（K 之后的可管理 MoE 层合计）；启动按 layer-major 填充，运行时允许跨层迁移物理行 | GEC 常驻容量与运行时热度更新 | 每层 GEC 缓存驱动运行时驻留更新 |
| `--kt-layer-h2d-slots` | 当前层当前轮即时 H2D Admission 预算（换入上限） | 无效（需 slots；无动态更新时不生效） | 生效，缺省 = 每层容量 |
| `--kt-layer-h2d-batch-size` | 单次 H2D 拷贝批次的专家数上限 | 无效（需 slots） | 生效，分批调用拷贝（MXFP4 除外） |

要点：

- `--kt-expert-gpu-slots` 单独使用即激活运行时 GEC 调度；所有 TP rank 使用确定性 coordinator 保持容量和映射一致；
  `--kt-enable-dynamic-expert-update` / `--kt-gpu-prefill-token-threshold` 属于独立的
  legacy threshold-gated layerwise full-GPU prefill 路径。
- `--kt-layer-h2d-slots` / `--kt-layer-h2d-batch-size` 要求先设置
  `--kt-expert-gpu-slots`（启动校验）。
- 启动预填：全局按 layer-major、expert-id-minor 顺序填充，豁免单轮预算；例如
  `kt-num-gpu-layers=0` 的 60×256 专家模型中，`slots=2560` 初始填充前 10 个
  eligible MoE 层；设置 K 后从 K 之后的第一个 eligible MoE 层开始计算。
- 即时执行与常驻更新分离：当前层每轮先根据本层 Router 激活计数做 H2D Top-N，
  并提前决定 Persistent Cache Admission，随后立即执行 GPU/CPU 路径；执行后
  只更新命中历史、Probation 和 Eviction 统计。
- 稳定行分配：层内常驻专家跨轮保持同一 GPU 行；容量变化时只重建受影响层的
  compact row mapping。`--kt-layer-h2d-slots ≥ 当前层容量` 时启动告警。
- GEC MXFP4 路径只重建本轮新准入 row；legacy/full-GPU helper 的整体重建不属于
  GEC selected-row H2D 路径。
- Telemetry：拷贝批次耗时记为 `H2D_RUNNING`，`planner.summary()`
  输出 Hit Rate / Eviction / Re-H2D 与等待分解。

验证：绑定独立编译通过 + 冒烟（生命周期/驱逐/refcount/调度/Telemetry/Inflight）、
规划器行为测试（预填/预算/LRU/不变量）、sglang 仓库测试
`test/registered/unit/layers/moe/test_kt_gec.py`（20 项）。

## 10. 已知限制与后续工作

在当前约束下，控制面、CPU/C++ 仿真、Pool 生命周期、Inflight dependency、
TP layout 同步、六类 Telemetry 和各量化格式的临时 compact-row 执行路径已完成。
以下仅属于 GPU/CUDA 环境验证或性能优化，不作为本机 CPU 验收阻塞项：

1. GEC packed-format H2D 路径使用已准入的 Persistent GPU rows 执行 H2D 和 GEMM；
   compact temporary rows 仍可用于独立的 full-GPU extend 路径，但不作为 Cache
   Admission 失败后的 GEC MISS fallback，也不执行 GEC 的 H2D Top-N/CPU fallback
   分类。进一步将 resident rows 压缩成专用 per-expert kernel workspace 仍可作为
   后续性能优化。
2. 真实 CUDA kernel overlap、NCCL 多卡 resize/refcount/Event 顺序和 PCIe
   带宽数据需要 GPU 硬件压测验证。
3. CPU GEMM 与 H2D 共享 Memory BW 的实际吞吐，需要真实硬件测量；仿真层已
   提供 `cpu_gemm_latency_us` / `h2d_latency_us` 钩子。
