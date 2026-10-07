# Global Expert Cache + H2D 滚动流水 — 开发计划

> 依据：`doc/KT-CPU-PICE-GPU.md`（V1 设计方案）

## 1. 目标与范围

### 1.1 目标

在 MoE + CPU Offload 场景下，将现有的串行路径：

```text
CPU Expert → H2D → GPU GEMM → 等待/Synchronize → CPU Fallback/Merge
```

重构为：

```text
GPU Router → Current Logical Experts → Global Expert Cache
    ├── HIT  → GPU GEMM
    └── MISS → Candidate Filter → Priority → Top-N
                  ├── H2D → Buffer Pool → H2D Stream → GPU GEMM
                  └── CPU Fallback → CPU GEMM
```

并实现 CPU GEMM / PCIe H2D / GPU GEMM 三段流水并行，最终将 150~180 ms 的
GPU Expert Wait 完整拆解为可度量的 6 类指标。

### 1.2 V1 范围内

- Logical Expert / Physical Shard 双层数据模型
- Global Expert Cache（Logical 粒度 Admission/Eviction，TP 整体 Placement）
- H2D 滚动流水（双 Stream + Device-local Dependency + Buffer Pool）
- CPU Fallback 双路径
- GPU Expert Wait Telemetry（6 类指标拆解）

### 1.3 V1 范围外（明确不做）

- 复杂预测模型 / 时间衰减 LFU / GPU-CPU 成本模型
- 同一 Expert 的 CPU GEMM → H2D → GPU GEMM 串行流水
- 全局 `cudaStreamSynchronize()` 类的串行化方案
- `IN_USE` 之类的伪状态（用 `refcount` 表达 Usage）

## 2. 里程碑总览

| 里程碑 | 主题 | 对应设计章节 | 优先级 | 预估 |
|--------|------|--------------|--------|------|
| M0 | 数据模型与状态机 | 3 / 7 / 8 / 9 / 10 | P0 | 1 周 |
| M1 | Placement 与 Inflight Registry | 4 / 11 / 18 | P0 | 1 周 |
| M2 | H2D 滚动流水 | 12-16 / 20-22 | P1 | 2 周 |
| M3 | Telemetry 与首诊 | 23 / 24 / 26 / 27 | P2 | 1 周 |
| M4 | Cache 策略 | 6 / 19 | P3 | 1 周 |
| M5 | 集成验证与调优 | 25 / 26 / 29 | - | 1-2 周 |

> 里程碑串行推进：M0/M1 为 M2 的前置；M3 与 M2 后半可并行；M4 在 M2 稳定后启动。

## 3. M0 — 数据模型与状态机（P0）

### 3.1 任务

1. 定义 `LogicalExpertId = (layer_id, expert_id)`，作为调度 / Cache / H2D / Eviction 的唯一单位。
2. 定义 `PhysicalShardId = (layer_id, expert_id, tp_rank)`，作为 GPU 存储 / H2D / Kernel 读取单位。
3. 实现 `LogicalExpert` 结构：
   - `state`（生命周期）
   - `placement[]`（物理分布）
   - `logical_ready_dependency`
   - `refcount`（atomic）
   - `in_flight / protected_ / last_used / hit_count / admission_state`
4. 实现 `PhysicalShard` 结构：
   - `gpu_id / slot_id / state / ready_dependency`
5. 实现 Lifecycle 状态机：

```text
ABSENT → SCHEDULED → LOADING → READY → EVICTING → ABSENT
```

6. 落实三维分离：`state`（驻留）、`logical_ready_dependency`（可安全使用）、`refcount`（使用计数）。
   - `READY + refcount == 0` 合法；`READY + refcount > 0` 合法；不引入 `IN_USE`。
7. 定义并校验 4 个参数：

```text
--kt-num-gpu-layers K          # 前 K 层 Expert 常驻 GPU
--kt-expert-gpu-slots M        # Persistent Cache 容量，单位 Logical Expert
--kt-layer-h2d-slots N         # 单轮 H2D Admission Budget，单位 Logical Expert
--kt-layer-h2d-batch-size B    # 单次 H2D Batch 上限，单位 Logical Expert
--max-running-requests R       # 服务级并发请求上限，参与临时 H2D Pool 深度计算
```

`M` 是 `kt-num-gpu-layers` 之后、且通过 `first_k_dense_replace/moe_layer_freq` 过滤的
可管理 MoE 层合计的 Persistent Cache 容量。初始化从第一个 eligible MoE 层开始按
layer-major 顺序填充，运行时由当前 Router 需求提前决定常驻集合并驱动跨层物理行滚动；
执行完成后再更新命中历史、Probation 和 Eviction 统计。`N` 只限制
当前层当前轮的即时 H2D 换入数量，`B` 只决定提交拆分。例如
`kt-num-gpu-layers=0` 时，60 x 256 专家位置配合 `M=2560, N=64, B=8`
初始填充前 10 个 eligible 层；设置 K 后从 K 之后的第一个 eligible 层开始。
当前层先按 N 分 8 批执行；Coordinator 在 H2D 提交前根据当前 Router 需求决定
哪些 Expert 进入 M 对应的 Persistent Cache。

当前层的执行归属必须按下面的顺序确定，不能把完整的 Router 命中集合直接
提交给 H2D 或 CPU wrapper：

```text
Router 命中集合
    ↓
Persistent Cache Lookup
    ├── HIT  → 保持 GPU resident，直接复用，不进入 H2D candidate
    └── MISS → 按当前 activation count 排序，只取 Top-N
                    ├── 选中 → H2D → GPU GEMM
                    └── 未选中 → CPU fallback → CPU GEMM
```

因此，`N` 的含义是“当前层当前轮最多从 CPU MISS 中换入多少个 Expert”，
不是从所有 Router 命中中重新选择 GPU Expert；`B` 只把已经选中的 H2D 列表
切成提交批次，不改变 Top-N 结果。GPU resident mask 更新完成后，CPU task 只
提交最终 mask 中为 `False` 的路由，避免同一个 Expert 同时进入 H2D 和 CPU GEMM。

临时 H2D Pool 的计算为：

```text
inflight_batches = R
temporary_logical_slots = inflight_batches × B
```

例如 `R=17, B=8` 时，最多 17 个在途批次，对应 136 个临时 Logical slots；
TP=4 时最多展开为 544 个 Physical Shard Buffer。这部分不计入 `M`。

### 3.2 交付物

- `kt-kernel/include/` 下的核心数据结构与状态机（纯逻辑，无 CUDA 依赖）
- 参数解析与校验（含单位换算：TP=4 时 M 个 Logical = 4M 个 Shard）
- 状态机转换单元测试（合法/非法转换、refcount 边界）

### 3.3 验收标准

- 状态机只允许文档定义的转换路径，非法转换触发断言/异常
- `refcount` 语义测试：submit→completion 期间 >0，completion 后归零
- 参数校验测试：负数/零/超界值被拒绝；单位换算正确

## 4. M1 — Placement 与 Inflight Registry（P0）

### 4.1 任务

1. **Global Placement**：
   - Admission 以 Logical Expert 为单位，一次性决定全部 TP Shard 的去向
   - TP=4：`E32 → TP0/1/2/3 → GPU0/1/2/3`，禁止部分 admission
   - TP+EP：`Logical Expert → EP Owner → TP Shards → Owner GPUs`
2. **TP Atomic Ready**：
   - 所有必要 shard Ready 才置 Logical READY；任一 LOADING 则不 READY
3. **Inflight Registry**：
   - Key = `(layer_id, expert_id, tp_rank)`
   - 同一 Shard 最多一个 active H2D；重复请求复用已有 dependency
4. **整体 Eviction 前置能力**：
   - `can_evict = READY && refcount==0 && !in_flight && !protected`
   - TP 下整体 Eviction，禁止单独淘汰某个 Shard

### 4.2 交付物

- Placement Planner（由 TP/EP topology 驱动）
- Inflight Registry（含去重、dependency 复用）
- Eviction 前置检查接口
- 单元测试：TP=2/4/8、EP 组合、部分失败回滚、重复请求去重

### 4.3 验收标准（对应设计 29.1 / 29.2）

- 任意 TP 拓扑下，Placement 结果完整或整体失败，不存在部分 admitted
- 同一 `(layer, expert, tp_rank)` 在 Registry 中最多一个 active 记录
- TP 3/4 Ready 时 Logical 仍为 LOADING；4/4 Ready 才 READY

## 5. M2 — H2D 滚动流水（P1）

### 5.1 任务

1. **当前层 Candidate Filter → Priority → Top-N → H2D Queue** 调度链：
   - 先排除 Persistent Cache HIT；剩余 CPU MISS 再排除：LOADING/Inflight、EVICTING、Placement unavailable、
      H2D Buffer unavailable、Cache admission locked
   - Priority = `Current Demand + α × Historical Hit Count`（Current Demand 优先）
   - Top-N 后未入选的 CPU MISS 走 CPU Fallback，两条路径明确分离；当前层不等待其他层
      的候选结果
2. **五个容量概念彻底分离**：
   - H2D Admission Budget（`h2d-slots`）
   - H2D Batch Size（`batch-size`，`batch = min(available, batch_size)`，不凑满）
   - H2D Buffer Pool Depth（由 `max-running-requests` 推导；每个请求最多一个当前批次）
   - H2D Stream
   - Compute Stream
3. **H2D Buffer Pool**（每 GPU 独立）：
   - Buffer 状态机：`FREE → H2D → READY_FOR_COMPUTE → GPU_IN_USE → COMPLETION_WAIT → FREE`
   - H2D 完成 ≠ Buffer 可复用；必须等 GPU Kernel 完成
   - Buffer 属于临时资源，不计入 Persistent Cache
   - 普通 hybrid 路径的 INT4/Marlin selected Expert 直接写入 Persistent GPU row
     并做局部 Marlin repack；full-shadow 仅保留给 legacy/full-GPU fallback。满足
     full-GPU extend 条件时，packed-format 路径处理当前 batch 的全部 active Expert，
     绕过 GEC Top-N/CPU fallback 分类
4. **双 Stream 流水**：
   - 每 GPU 独立：H2D Queue / Buffer Pool / H2D Stream / Compute Stream / Local Cache / Device-local Dependency
   - `cudaMemcpyAsync` 在 H2D Stream 上执行，记录 Device-local Ready Dependency
   - Compute Stream 只 `wait` 自己需要的 dependency，禁止全局同步
5. **Cache HIT 路径 refcount 语义**：

```text
HIT → refcount++ → wait(dependency) → GPU GEMM → Completion Event → refcount--
```

   禁止 submit 后立即 `refcount--`。
6. **CPU Fallback 路径**：`CPU Weight → CPU GEMM → CPU Output → Merge`，与 GPU 路径并行。

### 5.2 交付物

- 调度链（Filter/Priority/Top-N/Queue）实现与单测
- 每 GPU 的 Buffer Pool + Stream 管理
- `H2DBuffer` 生命周期管理（含 completion callback）
- 双路径集成：GPU H2D Path 与 CPU Fallback Path 汇入 Merge
- 集成测试：多 Expert 并发、Buffer 争用、不同 batch-size/depth 组合

### 5.3 验收标准（对应设计 29.3 - 29.6）

- H2D candidate 先完成 Persistent Cache Admission；准入失败直接 CPU fallback，避免
  H2D 与 Cache 反复抖动
- Buffer 复用严格发生在 GPU Completion 之后（用假 CUDA Runtime 验证时序）
- `refcount` 递减严格发生在 Completion Event 之后
- 代码路径中不存在将 H2D 与 Compute 串行化的全局 `cudaStreamSynchronize()`
- 不同 Expert 间呈现设计 21 节的三段交叠：`[CPU GEMM] [H2D] [GPU GEMM]` 时间轴重叠

## 6. M3 — Telemetry 与首诊（P2）

### 6.1 任务

1. 实现 6 类 Wait 指标：

```text
ROUTER_DEPENDENCY / H2D_QUEUED / H2D_RUNNING / H2D_EVENT_WAIT / CPU_FALLBACK / CACHE_WAIT
```

2. 每条记录至少包含：`layer_id, expert_id, tp_rank, wait_start, wait_end, wait_duration, reason`
3. 记录关键时间点：H2D Queue Enter / Start / End、Event Wait Start / Ready、GPU GEMM Start / End
4. **严格区分 `H2D_RUNNING` 与 `H2D_EVENT_WAIT`**（设计 24 节）：
   - `RUNNING 小 / EVENT_WAIT 大` → 查 Stream Ordering / Dependency / Event
   - `RUNNING 大` → 查 PCIe / Pinned / NUMA / Batch / 带宽
5. 运行第一轮诊断，回答：**150~180 ms 到底在哪里？**（设计 27 节四个 Case）

### 6.2 交付物

- Telemetry 采集器与结构化输出（JSON/CSV）
- 诊断脚本：自动汇总 6 类指标占比与分布（P50/P90/P99）
- 第一轮诊断报告：明确瓶颈归属（Case A/B/C/D）

### 6.3 验收标准

- 任一次 GPU Expert Wait 都能完整拆解为 6 类指标之一（或明确无等待）
- 指标总和与端到端 wait 误差在可解释范围内
- 产出结论能指明后续优化方向（CUDA 依赖 / 队列 / CPU / PCIe）

## 7. M4 — Cache 策略（P3）

### 7.1 任务

1. **Cache Admission**：当前层 H2D Admission 以 Cache Admission 成功为前提
   - 当前层 H2D Top-N 先确定本批次候选；当前 Router 需求同时参与提前的
     Persistent Cache Admission / 冷槽预留，准入失败的 Expert 走 CPU fallback
2. **Eviction**：整体 Eviction + 前置条件检查（M1 已建接口，此处接入策略）
3. **Probation / Protection**：新加载 Expert 进入保护期，防止用一次即驱逐
4. **Re-H2D Tracking**：监控 Re-H2D Rate，作为策略调优依据

### 7.2 交付物

- Admission/Eviction 策略（V1 保持简单：命中历史 + 保护期，不做复杂模型）
- Re-H2D / Hit / Miss / Eviction Rate 指标
- 压测脚本：不同 `kt-expert-gpu-slots` 下的 Hit Rate 与 Re-H2D 曲线

### 7.3 验收标准

- 无 `H2D → Cache Full → Evict → H2D → Evict` 抖动循环
- Re-H2D Rate 可观测、可对比；Probation 生效前后有量化差异
- Eviction 永远整体发生（TP 全 shard 同进同出）

## 8. M5 — 集成验证与调优

### 8.1 任务

1. **指标采集**（设计 26 节）：
   - GPU：6 类 Wait
   - H2D：Queue Latency / Transfer Latency / Bandwidth
   - Cache：Hit / Miss / Re-H2D / Eviction Rate
   - CPU：Fallback Latency / Memory BW / GEMM Latency
2. **带宽竞争测试**（设计 25 节）：
   - `CPU GEMM Only` vs `CPU GEMM + H2D Concurrent`
   - 验证 CPU GEMM 与 H2D 共享 CPU Memory BW 时的实际收益
3. **参数扫描**：`h2d-slots / batch-size / buffer-pool-depth / expert-gpu-slots` 组合
4. **Acceptance 全量验证**（见第 9 节）

### 8.2 交付物

- 基准脚本（复用/扩展 `kt-kernel/bench/`）
- 参数扫描矩阵与报告
- 最终性能报告：Wait 拆解 + 吞吐/延迟对比

## 9. 总体验收清单（对应设计 29 节）

| # | 验收项 | 判定 |
|---|--------|------|
| 1 | Logical Expert 正确 | Complete Placement → All Shards Ready → Logical Ready → READY |
| 2 | 无重复 H2D | 同一 `(layer, expert, tp_rank)` 最多一个 active H2D |
| 3 | H2D 需要 Cache Admission | H2D 只提交已完成 Persistent Cache Admission 的 Expert |
| 4 | Buffer 不提前复用 | `H2D → READY_FOR_COMPUTE → GPU_IN_USE → Completion → FREE` |
| 5 | refcount 不提前释放 | Kernel Submit → Completion Event → `refcount--` |
| 6 | 无全局同步 | H2D 与 Compute 由 Device-local Dependency 衔接 |
| 7 | Wait 可拆解 | 6 类指标均能给出数值，覆盖 150~180 ms |

## 10. 测试策略

| 层级 | 内容 | 位置 |
|------|------|------|
| 单元测试 | 状态机 / Registry / Placement / Filter | `kt-kernel/test/` |
| 假 CUDA 测试 | Buffer 生命周期 / 时序 / 依赖（无 GPU 环境可跑） | `kt-kernel/cpu_backend/test/fake_cuda_runtime.h` |
| 集成测试 | 双路径 / 多 Expert / TP 组合 / Merge | `kt-kernel/test/` |
| 基准测试 | Wait 拆解 / 带宽 / 参数扫描 | `kt-kernel/bench/` |

测试推进顺序：先在 Fake CUDA Runtime 上验证全部时序断言（Buffer 复用、refcount、
依赖等待），再上真实 GPU 做性能与带宽验证。

## 11. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 跨 GPU 无统一 Event，Logical Ready 判断复杂 | READY 误判/悬挂 | 每个 GPU 只用 Device-local Dependency；Logical Ready 由全部 shard 状态聚合 |
| Buffer/Completion 回调时序错误 | 提前复用/提前 Evict | Fake CUDA Runtime 下穷举时序；Buffer 与 refcount 生命周期独立断言 |
| CPU GEMM 与 H2D 抢 CPU Memory BW | 并行反而变慢 | M5 显式对比串行/并发；必要时限制 H2D 并发或错峰 |
| Cache 抖动 | Re-H2D 飙升 | Admission 分离 + Probation；Re-H2D Rate 作为回归指标 |
| 参数组合爆炸 | 调优成本高 | 先按设计固定默认组合，只扫描关键两维（slots × batch） |

## 12. 建议排期（总计 7-8 周）

```text
W1      M0 数据模型与状态机
W2      M1 Placement + Inflight Registry
W3-W4   M2 H2D 滚动流水（先 Fake CUDA，后真机）
W4-W5   M3 Telemetry（与 M2 后半并行）
W5      M4 Cache 策略
W6-W7   M5 集成验证、参数扫描、首诊闭环
```

关键路径：M0 → M1 → M2 → M3 → M5。M4 可在 M2 稳定后并行插入。

## 13. 实施状态

> 实施记录详见 `doc/KT-CPU-PICE-GPU-IMPL.md`。

| 里程碑 | 状态 | 交付物 |
|--------|------|--------|
| M0 数据模型与状态机 | 已完成 | `kt-kernel/include/kt/gec/gec_types.hpp`、`gec_config.hpp` |
| M1 Placement 与 Inflight Registry | 已完成 | `placement.hpp`、`inflight_registry.hpp` |
| M2 H2D 滚动流水 | 已完成 | `scheduler.hpp`、`buffer_pool.hpp`、`pipeline.hpp` |
| M3 Telemetry 与首诊 | 已完成 | `telemetry.hpp` + 150ms 拆解演示 |
| M4 Cache 策略 | 已完成 | `expert_cache.hpp`（Admission / LRU / Probation / Re-H2D / 启动顺序预填） |
| M5 集成验证与调优 | 已完成（仿真层） | 278 项 C++ 检查 + 12 项 CI 合同测试，全部通过；SGLang GEC 支持跨层物理行滚动 |

验证命令：

```bash
python kt-kernel/test/gec/run_tests.py
python -m pytest kt-kernel/test/per_commit/test_gec_contract.py -v
```

启动行为：`prefill_sequential` 从 layer `kt-num-gpu-layers` 之后开始顺序填充
`kt-expert-gpu-slots` 个 Logical Expert（layer-major、expert-id-minor）；
`kt-num-gpu-layers` 内的专家永久驻留且不占 slot。预填完成后首轮即可 HIT，
PCIe 并行传输缺失专家。运行时 GEC 根据激活热度在 MoE 层之间滚动物理行容量，
donor 层只释放满足 READY/refcount/inflight 安全门的专家，target 层按
`kt-layer-h2d-slots` 和 `kt-layer-h2d-batch-size` 渐进换入。

后续工作：在真实硬件上执行设计 25 节的 CPU GEMM + H2D 带宽竞争测试，并验证
不同量化格式动态扩缩 compact GPU rows 的端到端 CUDA 行为。
