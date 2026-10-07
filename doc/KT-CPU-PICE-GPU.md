# KTransformers Global Expert Cache + H2D 滚动流水方案

## 1. 目标

针对 MoE + CPU Offload 场景，解决当前：

```text
CPU Expert
   ↓
H2D
   ↓
GPU GEMM
   ↓
等待 / Synchronize
   ↓
CPU Fallback / Merge
```

造成的 GPU Expert Wait 和约 150~180 ms 空白时间。

V1 的核心目标不是预测，而是首先建立真正的：

```text
GPU Router
    ↓
Current Logical Experts
    ↓
Global Expert Cache
    ├── HIT  → GPU GEMM
    └── MISS → Top-N H2D / CPU Fallback
```

并实现：

```text
CPU GEMM
PCIe H2D
GPU GEMM
```

三段流水并行。

---

# 2. 核心原则

V1 固定以下原则：

1. **Logical Expert 是调度和 Cache 的基本单位。**
2. **Physical Shard 是 TP 的物理执行单位。**
3. `kt-expert-gpu-slots` 使用 Logical Expert 计数。
4. `kt-layer-h2d-slots` 使用 Logical Expert 计数。
5. `kt-layer-h2d-batch-size` 使用 Logical Expert 计数。
6. H2D Admission 以 Persistent Cache Admission 成功为前提。
7. H2D Buffer Pool 只管理已准入 Expert 的拷贝和 GPU 消费生命周期。
8. 不使用全局 `cudaStreamSynchronize()` 串行化 H2D 和 GPU Compute。
9. Compute Stream 只等待自己需要的 Device-local Dependency。
10. `state`、`logical_ready_dependency`、`refcount` 三个维度完全分离。
11. V1 不引入复杂预测模型、时间衰减 LFU 或 GPU/CPU 成本模型。

---

# 3. Logical Expert 与 Physical Shard

## 3.1 Logical Expert

Logical Expert：

```text
(layer_id, expert_id)
```

例如：

```text
E32
```

是一个完整的逻辑专家。

它是：

* Router 调度单位
* Cache Admission 单位
* H2D Admission 单位
* Ready 判断单位
* Eviction 单位

---

## 3.2 Physical Shard

TP=4 时：

```text
Logical Expert E32
├── TP0 Shard
├── TP1 Shard
├── TP2 Shard
└── TP3 Shard
```

Physical Shard：

```text
(layer_id, expert_id, tp_rank)
```

Physical Shard 是：

* GPU 实际存储单位
* H2D 执行单位
* GPU Kernel 实际读取单位

---

## 3.3 Atomic Rule

一个 Logical Expert 只有在所有必要 TP Shard 都 Ready 后才能使用。

错误：

```text
E32
├── TP0 READY
├── TP1 READY
├── TP2 LOADING
└── TP3 READY
```

不能认为 E32 READY。

正确：

```text
TP0 Ready
TP1 Ready
TP2 Ready
TP3 Ready
    ↓
Logical Expert READY
```

---

# 4. TP Placement

Cache Admission 必须以 Logical Expert 为单位做完整 Placement。

例如 TP=4：

```text
E32
├── TP0 → GPU0
├── TP1 → GPU1
├── TP2 → GPU2
└── TP3 → GPU3
```

禁止独立决定：

```text
E32
├── TP0 admitted
├── TP1 admitted
├── TP2 rejected
└── TP3 admitted
```

正确：

```text
E32 Admission
      ↓
Complete Placement
      ↓
TP0 / TP1 / TP2 / TP3
```

对于 TP + EP：

```text
Logical Expert
      ↓
EP Owner
      ↓
TP Shards
      ↓
Owner GPUs
```

Placement 必须由 TP/EP topology 决定。

---

# 5. 参数定义

```bash
--kt-num-gpu-layers K
--kt-expert-gpu-slots M
--kt-layer-h2d-slots N
--kt-layer-h2d-batch-size B
```

## 5.1 kt-num-gpu-layers

前 K 层 Expert 永久驻留 GPU。

---

## 5.2 kt-expert-gpu-slots

Global Persistent Cache 与物理 GPU 行池的总容量。

**单位：Logical Expert。**

模型的可用位置总数是 `kt-num-gpu-layers` 之后、且通过
`first_k_dense_replace/moe_layer_freq` 过滤的可管理 MoE 层专家数之和。
下面假设 `kt-num-gpu-layers=0`，模型有 60 个 MoE 层、每层 256 个专家：

```text
60 * 256 = 15360 Logical Expert positions
```

设置：

```text
kt-expert-gpu-slots=2560
```

表示整个模型最多同时缓存 2560 个 Logical Expert，而不是每层 2560 个，
也不是固定平均分配成 `2560 / 60`。启动时从第一个 eligible MoE 层开始按
layer-major、expert-id-minor 顺序填充：

```text
first eligible layer: 256
next eligible layer: 256
...
10th eligible layer: 256
remaining eligible layers: 0
```

`kt-num-gpu-layers` 以内的永久 GPU 层不消耗该预算。TP=4 时，2560 个 Logical
Expert 对应最多 10240 个 Physical Shard；参数本身仍不按 Physical Shard 计数。

`kt-expert-gpu-slots` 本身就启用 GEC 的运行时常驻更新：每个 extend/prefill
batch 先由当前层 Router 结果决定本轮需要执行的 MISS，并提前决定哪些 Expert
进入或预留 Persistent Cache；随后执行 H2D/CPU/GPU 路径。执行完成后再更新
命中历史、Probation 和 Eviction 统计，decode batch 直接复用已建立的常驻集合。
热层可以从冷层借用物理 GPU 行，形成跨层滚动分配；
例如运行一段时间后可以是 layer 0 有 200 行、layer 1 有 156 行，只要所有层的
物理行总数不超过全局容量。它与
`kt-enable-dynamic-expert-update`、`kt-gpu-prefill-token-threshold` 相互独立。
后两个参数属于旧的 threshold-gated layerwise full-GPU prefill 路径，不是 GEC
的前置条件。设置 `kt-layer-h2d-slots` 或 `kt-layer-h2d-batch-size` 时仍必须先
设置 `kt-expert-gpu-slots`，因为这两个参数只描述 GEC 的换入预算和批次大小。

---

## 5.3 kt-layer-h2d-slots

当前 Layer / 当前 Round 的 H2D Admission Budget。

**单位：Logical Expert。**

例如：

```text
kt-layer-h2d-slots=100
```

表示本轮最多允许 100 个 Logical Expert 走 GPU H2D 路径。

这个预算只约束**当前层当前 batch 的即时执行**。本层先根据本层 Router
需求选择 H2D 候选，不需要等待其他 MoE 层的 Router 结果，也不参与跨层的
Persistent Cache 排序。

TP=4 时最多对应：

```text
100 × 4 = 400 Physical Shards
```

注意：`kt-layer-h2d-slots` **≥ 当前层当前物理容量时等于关闭该层换入节流**：
每一轮最多仍受全局容量和可迁移冷层行数约束，但路由变化可能造成较大换入量。
sglang 集成在启动时对该配置给出告警；建议预算取当前容量的 1/4 左右，用多轮渐进
跟踪热度漂移。

---

## 5.4 kt-layer-h2d-batch-size

单次 H2D Batch 最大 Logical Expert 数量。

例如：

```text
batch-size=8
```

TP=4：

```text
最多 8 Logical Experts
最多 32 Physical Shards
```

它只是最大值，不要求必须填满。

```text
available = 5
batch_size = 8

submit 5
```

即：

```text
batch = min(available, batch_size)
```

不等待凑满。

sglang 集成中，`available` 不是整个常驻集合，而是本轮新准入的 Logical Expert（与当前
GPU 行的集合差）：常驻行跨轮保持稳定（见 18.1），每轮只拷贝差量；集合不变的轮次整体
跳过拷贝与映射更新。当前 CPU writer 使用固定 host buffer ring，按 batch 内 Expert
逐个提交权重写入，再按 tensor/row 写入目标 GPU row；尚未实现文档原先设想的逐张量
向量化 gather/scatter。

## 5.5 H2D Buffer Pool 与并发度

`kt-layer-h2d-slots` 和 `kt-layer-h2d-batch-size` 描述本层即时 H2D 准入与批次，
不描述 Persistent Cache 容量，也不直接等于 Buffer Pool 数量。服务级
`--max-running-requests` 提供同时运行的请求上限，和批次大小共同推导临时
copy/event pipeline 的在途批次数。每个并发请求最多持有一个当前 H2D 批次：

```text
并发请求上限 = max-running-requests
每批 Logical Experts = kt-layer-h2d-batch-size
最大在途批次数 = Buffer Pool depth
```

例如：

```text
max-running-requests=17
kt-expert-gpu-slots=2560
kt-layer-h2d-slots=64
kt-layer-h2d-batch-size=8
```

表示：

```text
Persistent GPU Cache：最多 2560 个 Logical Expert
单层单轮即时 H2D：最多 64 个 Logical Expert
单个 H2D 批次：8 个 Logical Expert
最大在途批次数：17
Buffer Pool depth：17 个在途批次
```

每个被选中的 Logical Expert 都必须先完成 Persistent Cache Admission，再写入
对应的 GPU row；没有独立的“不进入 Cache 的 Temporary H2D”执行路径。实际
Buffer 数量还受每 GPU 显存、Buffer Pool depth 和设备并发能力约束；最后一个批次
可以小于 8，不等待凑满。若未设置 `max-running-requests`，运行时使用已解析的服务
并发上限或保守的 depth=1。

当前 C++ `RollingPipeline` 和 SGLang `GecCudaPipeline` 都有独立的 per-GPU
Buffer Pool 抽象。该 Pool 控制 copy/event pipeline 的临时资源，不改变
`kt-expert-gpu-slots` 的 Persistent Cache 容量。

---

# 6. H2D Admission 与 Cache Admission

必须明确：当前实现中 H2D Admission 与 Persistent Cache Admission 绑定。

H2D 之前必须先判断是否真的需要换入：Cache HIT 不进入 H2D 候选，Top-N 未选中的
MISS 直接 CPU fallback；只有入选的 MISS 才尝试 Persistent Cache Admission。准入
成功后才提交 H2D 到对应的 Persistent GPU row；准入失败、Placement 不可用或
H2D Buffer 不可用都直接 CPU fallback。实现中不存在先提交 Temporary H2D、再尝试
补做 Cache Admission 的路径。

```text
当前层 Router MISS
      ↓
Candidate Filter / Priority / Top-N
      ↓  （≤ kt-layer-h2d-slots）
Persistent Cache Admission / 冷槽预留
      ├── 成功 → H2D 到 Persistent GPU row → GPU GEMM
      └── 失败 → CPU Fallback → CPU GEMM
```

`kt-layer-h2d-slots` 的 Top-N 是本层即时 H2D Top-N。Persistent Cache Admission
可以在 H2D 之前根据当前 Router 已知的真实需求提前决定，并在需要时先完成跨层
冷槽替换。无法完成准入的 Expert 不进入 H2D 路径，直接保留给 CPU fallback。

当 INT4/Marlin 进入 GEC hybrid selected-row copy 路径时，GEC Top-N 仍按
`kt-layer-h2d-slots` 选择；选中 Expert 直接从 CPU raw packed buffer 写入 selected
Persistent GPU row，再对该 row 做局部 Marlin repack，CPU writer 仍按 Expert/tensor/row
循环执行。full-shadow 只保留给 legacy/full-GPU fallback 路径。支持 full-GPU extend
条件的 packed-format 请求则直接把
当前 batch 的全部 active Expert 写入 compact temporary rows 并执行 GPU GEMM，绕过
GEC 的 H2D Top-N/CPU fallback 分类；这不是 Cache Admission 失败后的 fallback。

避免：

```text
H2D
 ↓
Cache Full
 ↓
Evict
 ↓
H2D
 ↓
Evict
```

造成 Cache Thrashing。

---

# 7. Logical Expert 状态模型

Logical Expert 的生命周期只保留：

```text
ABSENT
   ↓
SCHEDULED
   ↓
LOADING
   ↓
READY
   ↓
EVICTING
   ↓
ABSENT
```

完整流程：

```text
                ┌──────────────┐
                │    ABSENT    │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  SCHEDULED   │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   LOADING    │
                └──────┬───────┘
                       │
              all shard ready
                       │
                       ▼
                ┌──────────────┐
         ┌─────▶│    READY     │◀─────┐
         │      └──────┬───────┘      │
         │             │              │
         │        refcount++          │
         │             │              │
         │             ▼              │
         │        GPU Kernel          │
         │             │              │
         │        refcount--          │
         │             │              │
         │             └──────────────┘
         │
         │ refcount == 0
         │ eviction selected
         ▼
                ┌──────────────┐
                │   EVICTING   │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │    ABSENT    │
                └──────────────┘
```

---

# 8. State / Dependency / Usage 三维模型

这三个概念必须完全分离。

```text
                    Logical Expert
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
      Lifecycle        Dependency        Usage
          │               │                │
        state       ready_dependency     refcount
```

## 8.1 Lifecycle

```text
state
```

表示：

> Expert 是否驻留 GPU，以及是否正在加载/淘汰。

---

## 8.2 Dependency

```text
logical_ready_dependency
```

表示：

> GPU Kernel 什么时候可以安全使用该 Expert。

TP=4：

```text
TP0 ready dependency
TP1 ready dependency
TP2 ready dependency
TP3 ready dependency
        ↓
Logical Ready
```

每个 GPU 使用自己的 Device-local dependency。

不要假设存在一个跨 GPU 的单一 CUDA Event。

---

## 8.3 Usage

```text
refcount
```

表示：

> 当前有多少 GPU Kernel 正在使用该 Expert。

例如：

```text
E32
├── Kernel A → +1
├── Kernel B → +1
└── Kernel C → +1

refcount = 3
```

A 完成：

```text
refcount = 2
```

B 完成：

```text
refcount = 1
```

C 完成：

```text
refcount = 0
```

因此：

```text
READY + refcount == 0
```

完全合法。

同时：

```text
READY + refcount > 0
```

也完全合法。

不需要 `IN_USE` 状态。

---

# 9. Logical Expert 数据结构

```cpp
struct LogicalExpert {
    int layer_id;
    int expert_id;

    LifecycleState state;

    Placement placement[];

    LogicalReadyDependency logical_ready_dependency;

    std::atomic<int> refcount;

    bool in_flight;
    bool protected_;

    uint64_t last_used;
    uint64_t hit_count;

    AdmissionState admission_state;
};
```

---

# 10. Physical Shard

```cpp
struct PhysicalShard {
    int layer_id;
    int expert_id;
    int tp_rank;

    int gpu_id;
    int slot_id;

    ShardState state;

    DeviceReadyDependency ready_dependency;
};
```

---

# 11. Inflight Registry

必须防止重复 H2D。

Key：

```cpp
struct InflightKey {
    int layer_id;
    int expert_id;
    int tp_rank;
};
```

流程：

```text
第一次请求
    ↓
创建 LOADING
    ↓
提交 H2D
    ↓
保存 dependency
```

第二次请求：

```text
发现 LOADING
    ↓
不重复 H2D
    ↓
复用已有 dependency
```

Candidate Filter 必须排除重复提交新的 H2D；如果请求方可以等待并复用已经
注册的 device dependency，则该请求可以进入 dependency-reuse 分支，不应被
直接当作 CPU Fallback。

---

# 12. H2D Candidate Filter

正确流程：

```text
Current MISS
      ↓
Candidate Filter
      ↓
Priority
      ↓
Top-N
      ↓
H2D Queue
```

Candidate Filter 至少排除：

```text
LOADING / Inflight（禁止重复提交；可复用 dependency 的请求除外）
EVICTING
Placement unavailable
H2D Buffer unavailable
Cache admission locked
```

不能直接：

```text
MISS
 ↓
Priority
 ↓
Top-N
```

否则同一个 Expert 可能重复进入 H2D Queue。

---

# 13. Priority

V1 不使用复杂预测模型。

当前 Router 已经知道本轮真实需求，因此使用：

```text
Priority =
    Current Demand
    +
    α × Historical Hit Count
```

其中：

```text
Current Demand
```

优先级最高。

Historical Hit Count 只用于当前 MISS 之间排序。

---

# 14. H2D Pipeline

至少使用两个 CUDA Stream：

```text
H2D Stream
Compute Stream
```

推荐：

```text
CPU Weight
    ↓
cudaMemcpyAsync
    ↓
H2D Stream
    ↓
Device-local Ready Dependency
    ↓
Compute Stream wait
    ↓
GPU GEMM
```

禁止：

```text
H2D Stream
    ↓
cudaStreamSynchronize()
    ↓
Compute Stream
```

Compute Stream 只等待自己需要的 dependency。

---

# 15. H2D Admission Budget 与 Pipeline Depth 分离

以下概念必须独立：

```text
1. H2D Admission Budget
2. H2D Batch Size
3. H2D Buffer Pool Depth
4. H2D Stream
5. Compute Stream
```

例如：

```text
h2d-slots = 100
batch-size = 8
buffer-pool = 3
```

可以：

```text
Batch1 → Buffer0
Batch2 → Buffer1
Batch3 → Buffer2
Batch4 → 等待 Buffer 回收
```

`h2d-slots=100` 不代表同时存在 100 个 H2D Buffer。

---

# 16. H2D Buffer Pool

每个 GPU 独立拥有：

```text
H2D Queue
H2D Buffer Pool
H2D Stream
Compute Stream
Local Cache
Device-local Dependencies
```

Buffer：

```cpp
struct H2DBuffer {
    int gpu_id;
    int buffer_id;

    LogicalExpertId logical_expert;
    int tp_rank;

    BufferState state;

    DeviceReadyDependency ready_dependency;

    RequestId owner_request;
};
```

状态：

```text
FREE
 ↓
H2D
 ↓
READY_FOR_COMPUTE
 ↓
GPU_IN_USE
 ↓
COMPLETION_WAIT
 ↓
FREE
```

关键原则：

```text
H2D 完成
    ≠
Buffer 可以立即复用
```

必须等 GPU Kernel 完成。

---

# 17. GPU Cache HIT

Cache HIT：

```text
Router
  ↓
Cache Lookup
  ↓
HIT
  ↓
refcount++
  ↓
wait(required dependency)
  ↓
GPU GEMM
  ↓
GPU Completion
  ↓
refcount--
```

`refcount` 必须覆盖真实 GPU Kernel lifetime。

错误：

```text
submit GPU GEMM
    ↓
refcount--
    ↓
Evict
```

正确：

```text
submit GPU GEMM
    ↓
GPU Completion Event
    ↓
Kernel 完成
    ↓
refcount--
```

---

# 18. Eviction

最终 Eviction 条件：

```cpp
can_evict =
    state == READY &&
    refcount == 0 &&
    !in_flight &&
    !protected;
```

只有满足全部条件：

```text
READY
  ↓
EVICTING
  ↓
释放所有 TP Shard
  ↓
ABSENT
```

TP 下必须整体 Eviction：

```text
E32
├── TP0
├── TP1
├── TP2
└── TP3
```

不能单独淘汰某一个 Shard。

## 18.1 Stable Row Assignment（物理行稳定分配）

GPU 常驻行与 Logical Expert 的绑定跨轮保持稳定：

```text
Evicted Expert 释放的行
    ↓
由本轮 Admitted Expert 继承
```

行号不随常驻集合的排序结果重排。每个 TP Rank 依据广播的常驻集合独立计算出一致的
行分配（确定性 diff：`(old_row_owner, new_resident_set) → (arrived, freed_rows)`，按升序配对）。

收益：

```text
每轮拷贝量 = 本轮新准入数（≤ Admission Budget）
而不是整个常驻集合（= Capacity）
```

常驻集合不变的轮次是 no-op round：跳过全部拷贝与路由表更新。

GEC MXFP4 路径只对本轮新准入 row 做 Marlin 重建；legacy/full-GPU helper 仍可能
整体重建 resident image，但不属于 GEC selected-row H2D 路径。

## 18.2 Cross-layer Row Rebalancing（跨层物理行迁移）

当前 SGLang GEC 使用模型级 coordinator 管理所有 MoE 层的物理行预算。
跨层迁移属于当前层 H2D 执行之前的 Persistent Cache 预留阶段。当前层根据
本批次 Router 已知需求完成本层 H2D Top-N 后，coordinator 从全局驻留集合中
选择冷槽：

```text
目标层需要新行
    ↓
选择低热度 donor layer
    ↓
仅驱逐 READY && refcount == 0 && !inflight 的 Logical Expert
    ↓
donor layer 缩容并保留其余行
    ↓
目标 layer 扩容，本轮 H2D 可直接写入预留的常驻行
```

跨层迁移发生在当前层 H2D 提交之前，不会覆盖正在执行的 GPU kernel；无法通过
refcount、inflight 或 Physical Shard Ready 检查的 donor 行留到后续轮次。如果
当前层无法获得可用常驻行，Expert 保留给 CPU fallback。
迁移后的行权重在目标层重新建立 compact row mapping，所有 TP rank 使用相同的
确定性容量和 owner 更新。执行完成后只更新命中历史、Probation 和 Eviction
统计，不重新决定本轮已经提交的 H2D 路径。

---

# 19. Cache Thrashing Protection

新加载的 Expert 可以进入：

```text
PROBATION
```

避免：

```text
H2D
 ↓
使用一次
 ↓
立即 Evict
 ↓
再次 H2D
```

重点监控：

```text
Re-H2D Rate
```

如果 Re-H2D 很高，再考虑进一步优化 Cache Policy。

V1 不增加复杂 LFU/Decay 模型。

---

# 20. CPU Fallback

当前 MISS：

```text
Candidate Filter
      ↓
Priority
      ↓
Top-N
```

Top-N：

```text
H2D
 ↓
GPU GEMM
```

剩余：

```text
CPU Fallback
 ↓
CPU GEMM
 ↓
CPU Output
 ↓
Merge
```

两条路径必须明确分开。

---

# 21. 两条数据路径

## GPU H2D Path

```text
CPU Weight
    ↓
H2D
    ↓
    Persistent GPU Cache row
    ↓
GPU GEMM
    ↓
GPU Output
```

## CPU Fallback Path

```text
CPU Weight
    ↓
CPU GEMM
    ↓
CPU Output
    ↓
Merge
```

不要设计成：

```text
CPU GEMM
   ↓
H2D
   ↓
GPU GEMM
```

V1 的目标是不同 Expert 之间流水，而不是同一个 Expert 的 CPU GEMM → H2D → GPU GEMM 串行流水。

例如：

```text
CPU:
[CPU GEMM B] [CPU GEMM D] [CPU GEMM F]

PCIe:
     [H2D A]      [H2D C]      [H2D E]

GPU:
          [GPU GEMM A] [GPU GEMM C] [GPU GEMM E]
```

---

# 22. 完整执行流程

```text
GPU Router
    ↓
Current Logical Experts
    ↓
Cache Lookup
    │
    ├── HIT
    │    ↓
    │  refcount++
    │    ↓
    │  GPU GEMM
    │    ↓
    │  refcount--
    │
    └── MISS
         ↓
    Candidate Filter
         │
         ├── Inflight
         │      ↓
         │  reuse dependency
         │
         └── Available
                ↓
              Priority
                ↓
        本层 Top-N（≤ layer-h2d-slots）
                ↓
       Persistent Cache Admission
          / 冷槽预留与迁移
             /     \
           H2D     CPU Fallback
            │           │
       H2D Batch     CPU GEMM
            │           │
       Buffer Pool   CPU Output
            │           │
       H2D Stream        │
            │             │
      Shard Dependency    │
            │             │
            ▼             │
      Logical READY       │
            │             │
            ▼             │
      Compute Stream      │
            │             │
         GPU GEMM ────────┴──→ Merge
             │
             ▼
       更新 Hit / Probation / Eviction 统计
```

本层 H2D/CPU 执行不等待其他层的当前 H2D 候选。Global Coordinator 在 H2D
提交前根据当前 Router 需求完成 Persistent Cache 预留和冷槽迁移；GPU 执行
完成后只更新后续策略所需的命中历史和保护状态。

---

# 23. GPU Expert Wait Telemetry

必须将 GPU Expert Wait 拆开：

```text
GPU Expert Wait
├── ROUTER_DEPENDENCY
├── H2D_QUEUED
├── H2D_RUNNING
├── H2D_EVENT_WAIT
├── CPU_FALLBACK
└── CACHE_WAIT
```

每次 Wait 至少记录：

```text
layer_id
expert_id
tp_rank

wait_start
wait_end
wait_duration

reason
```

同时记录：

```text
H2D Queue Enter
H2D Start
H2D End

Event Wait Start
Event Ready

GPU GEMM Start
GPU GEMM End
```

---

# 24. 重点区分 H2D_RUNNING 与 H2D_EVENT_WAIT

这是 V1 最重要的诊断点之一。

例如：

```text
H2D_RUNNING = 5 ms
H2D_EVENT_WAIT = 120 ms
```

说明主要问题很可能不是 PCIe 带宽，而可能是：

```text
CUDA Stream Ordering
Dependency
前序 CUDA Operation
错误的 Event Dependency
H2D Stream 被其他任务阻塞
```

反过来：

```text
H2D_RUNNING = 120 ms
```

才应该重点检查：

```text
PCIe Bandwidth
H2D Size
Pinned Memory
NUMA
CPU Memory Bandwidth
H2D Concurrency
```

---

# 25. CPU GEMM 与 H2D 带宽竞争

CPU GEMM 和 H2D 都可能读取 CPU Resident Weight：

```text
CPU GEMM
    ↘
      CPU Memory BW
    ↗
H2D
```

因此不能假设：

```text
CPU GEMM + H2D
```

一定比串行更快。

必须测试：

```text
CPU GEMM Only

CPU GEMM + H2D Concurrent
```

记录：

```text
CPU GEMM latency
H2D bandwidth
CPU Memory BW
GPU Expert Wait
```

---

# 26. 第一阶段测试指标

## GPU

```text
GPU Expert Wait
├── ROUTER_DEPENDENCY
├── H2D_QUEUED
├── H2D_RUNNING
├── H2D_EVENT_WAIT
├── CPU_FALLBACK
└── CACHE_WAIT
```

## H2D

```text
H2D
├── Queue Latency
├── Transfer Latency
└── Bandwidth
```

## Cache

```text
Cache
├── Hit Rate
├── Miss Rate
├── Re-H2D Rate
└── Eviction Rate
```

## CPU

```text
CPU
├── Fallback Latency
├── CPU Memory BW
└── CPU GEMM Latency
```

---

# 27. V1 第一轮测试目标

V1 第一轮**不以优化某一个指标为目标**。

首先回答：

```text
150~180 ms 到底在哪里？
```

例如：

### Case A

```text
H2D_RUNNING      8 ms
H2D_EVENT_WAIT 120 ms
```

重点检查：

```text
CUDA Dependency
Stream Ordering
Event
```

### Case B

```text
H2D_QUEUED      100 ms
```

重点检查：

```text
Scheduler
H2D Admission
Buffer Pool
H2D Queue
```

### Case C

```text
CPU_FALLBACK    130 ms
```

重点检查：

```text
CPU GEMM
CPU Threading
CPU Memory BW
CPU NUMA
```

### Case D

```text
H2D_RUNNING    120 ms
```

重点检查：

```text
PCIe
Pinned Memory
NUMA
H2D Batch
Memory Bandwidth
```

---

# 28. V1 实现优先级

## P0 — Correctness

```text
1. Logical Expert / Physical Shard
2. Global Placement
3. Inflight Registry
4. Logical Ready Dependency
5. TP Atomic Ready
6. refcount GPU Kernel Lifetime
7. H2D Buffer Lifecycle
```

## P1 — Pipeline

```text
8. Candidate Filter
9. H2D Admission
10. Logical Batch
11. H2D Buffer Pool
12. H2D Stream
13. Device-local Dependency
14. Compute Stream Wait
```

## P2 — Telemetry

```text
15. ROUTER_DEPENDENCY
16. H2D_QUEUED
17. H2D_RUNNING
18. H2D_EVENT_WAIT
19. CPU_FALLBACK
20. CACHE_WAIT
```

## P3 — Cache

```text
21. Admission
22. Eviction
23. Probation / Protection
24. Re-H2D Tracking
```

---

# 29. V1 Acceptance Criteria

### 1. Logical Expert 正确

```text
Complete TP Placement
        ↓
All Shards Ready
        ↓
Logical Ready Dependency
        ↓
READY
```

### 2. 无重复 H2D

同一个：

```text
(layer, expert, tp_rank)
```

最多只能存在一个 active H2D。

### 3. H2D Admission 需要 Cache Admission

```text
Persistent Cache Admission
```

成功后才能提交 H2D；准入失败的 MISS 直接走 CPU fallback。

### 4. Buffer 不提前复用

```text
H2D
 ↓
READY_FOR_COMPUTE
 ↓
GPU_IN_USE
 ↓
GPU Completion
 ↓
FREE
```

### 5. refcount 不提前释放

```text
GPU Kernel Submit
 ↓
GPU Completion
 ↓
refcount--
```

### 6. 不存在全局同步

避免：

```text
cudaStreamSynchronize()
```

将 H2D 与 GPU Compute 串行化。

### 7. 150~180 ms 可以被完整拆解

最终必须能够回答：

```text
ROUTER_DEPENDENCY = ?
H2D_QUEUED       = ?
H2D_RUNNING      = ?
H2D_EVENT_WAIT   = ?
CPU_FALLBACK     = ?
CACHE_WAIT       = ?
```

---

# 30. V1 最终架构

## 30.1 当前层的 HIT / MISS 执行契约

对每个 MoE layer，Router 命中的 Expert 必须先查询 Persistent Cache，再决定
执行路径。GPU 已驻留的 Expert 是 **HIT**，直接复用 GPU row，不能再次进入 H2D
候选列表；只有 CPU 上的 **MISS** 才能参与当前轮的 Candidate Filter、Priority
和 Top-N。

```text
Router 命中集合（例如 200 个 Expert）
        ↓
Cache Lookup
   ┌────┴────┐
   HIT       MISS
    │          │
    │          ├── 按当前 activation count 倒序
    │          ├── 取最多 kt-layer-h2d-slots 个
    │          │       ├── H2D → GPU row → GPU GEMM
    │          │       └── 其余 → CPU fallback → CPU GEMM
    │          └── 不会把 GPU HIT 重复 H2D
    └── 直接 GPU GEMM
```

例如 L1 Router 命中 200 个 Expert、其中已有 40 个在 GPU、
`kt-layer-h2d-slots=64`：先剔除 40 个 HIT，只在剩余 160 个 CPU MISS 中按当前
命中次数排序，最多选 64 个 H2D；其余 MISS 走 CPU fallback。若本轮所有路由
都是 HIT，则 H2D 数量为 0；`kt-layer-h2d-batch-size` 只切分已选中的 H2D 列表，
不改变 HIT/MISS 分类和 Top-N 结果。

运行时必须先完成 `gpu_experts_mask` 更新，再提交 CPU task。CPU task 只处理
最终 resident mask 为 `False` 的路由，避免同一个 Expert 同时执行 CPU GEMM 和
GPU/H2D 路径。

MXFP4 使用同一 hybrid 契约：CPU MISS 通过 CPU buffer 写入 persistent GPU row 的 raw
权重，只重打包新换入的 row。满足 full-GPU extend 条件的 packed-format 路径则直接
处理当前 batch 的全部 active Expert，不执行 GEC 的 H2D Top-N/CPU fallback 分类。

GEC 的 batch window 现在由固定 host buffer ring、H2D ready event 和 GPU
consumed event 共同保护。window 内先完成 CPU writer 批量 drain，再进行一次
TP barrier 和 H2D/repack；GPU 按已完成的 window 分组计算并累加输出。这里的
`kt-layer-h2d-batch-size` 仍表示 window 大小，双缓冲只负责 window 的生命周期，
不能单独替代 batch/window 配置。

```text
                         GPU Router
                             │
                             ▼
                  Current Logical Experts
                             │
                             ▼
                       Cache Lookup
                       /           \
                    HIT             MISS
                     │                │
                     │          Candidate Filter
                     │                │
                     │             Priority
                     │                │
                     │              Top-N
                     │             /      \
                     │           H2D       CPU
                     │            │       Fallback
                     │        H2D Queue    │
                     │            │        │
                     │        H2D Batch  CPU GEMM
                     │            │        │
                     │        Buffer Pool  CPU Output
                     │            │        │
                     │       H2D Stream    │
                     │            │        │
                     │      Device Ready   │
                     │            │        │
                     └──────┐     │        │
                            ▼     ▼        ▼
                         Compute Stream
                              │
                              ▼
                           GPU GEMM
                              │
                              ▼
                            Merge
```

最终形成：

```text
Lifecycle
    → state

Dependency
    → logical_ready_dependency

Usage
    → refcount

Scheduling
    → Candidate Filter / Priority / Top-N

Transfer
    → H2D Stream / Buffer Pool

Execution
    → Compute Stream / GPU GEMM

Persistence
    → Global Expert Cache

Fallback
    → CPU GEMM

Diagnosis
    → GPU Expert Wait Telemetry
```
