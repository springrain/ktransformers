# KTransformers GEC 架构设计文档 V1.6

**设计主题：Global Expert Cache + TP=2 CPU/GPU 异构执行 + FreeToken 工程经验借鉴**  
**目标平台：ARM64 + 2× NVIDIA RTX PRO 6000 Blackwell（SM120）**  
**版本定位：TP=2 基础执行路径审计与 GEC 原型验证版**  
**设计状态：架构基线保留；底层执行路径需先完成前置验证**

---

## 1. 设计目标

### 1.1 项目背景

本项目面向 ARM64 + 双 RTX PRO 6000 Blackwell 服务器，构建基于 KTransformers 的 CPU/GPU 混合 MoE 推理架构。

系统的基础条件和设计前提：

- 所有模型文件及专家权重均已加载到 Host RAM。
- CPU 可以直接访问 Host RAM 中的完整专家权重；CPU 专家执行不需要从 GPU 搬回专家权重。
- GPU VRAM 容量有限，只缓存部分热门专家。
- 两张 RTX PRO 6000 以 TP=2 作为目标运行模式。
- CPU 和 GPU 可以并行执行同一个 MoE Layer 中分配给不同执行设备的专家任务，但 TP=2 下 CPU Expert 的权重布局和输出语义必须先审计现有代码。
- 通过全局 LFU 管理 GPU 专家缓存。
- 后台异步加载专家，不因当前缓存未命中而阻塞当前推理。

### 1.2 设计目标

1. 提高 GPU 专家缓存命中率。
2. 充分利用 ARM64 CPU 和双 GPU 的计算能力。
3. 减少专家权重重复搬运和格式转换。
4. 让后台权重加载与当前推理尽可能重叠。
5. 支持不同专家量化格式和物理布局。
6. 统一管理逻辑专家，同时协调 TP Rank 上的物理数据。
7. 建立性能观测和校准体系，使用实际数据指导缓存与调度策略。
8. 在不改变模型计算语义的前提下，保持 CPU/GPU 异构执行的正确性。

### 1.3 目标硬件与验收范围

| 项目 | 目标配置 |
|---|---|
| CPU | ARM64，192 核 |
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU 架构 | SM120 |
| GPU 互联 | PCIe 5.0 x16，无 NVLink |
| Host RAM | 1TB 或以上 |
| OS | Ubuntu 22.04 ARM64 |
| 权重存储 | 模型文件及完整专家权重位于 Host RAM |
| 基础并行模式 | TP=2 |
| 后续扩展 | TP=4/8，暂不纳入本版实现验收 |

**TP=2 是系统级刚性要求。** 单 GPU 测试只用于隔离验证 Kernel、内存访问或传输问题，不能代替 TP=2 的阶段验收。TP=4/8 仅要求基础数据结构保留 Rank 扩展能力，后续再验证。

---

## 2. 总体架构

整体分为四个逻辑平面：

| 模块 | 核心职责 |
|---|---|
| GEC Residency Manager | 专家热度统计、GPU 驻留决策、缓存淘汰 |
| MoE Execution Scheduler | 当前 Layer 的 CPU/GPU 专家任务分配和 Join |
| Transfer Manager | GPU 与 Host 之间的必要异步数据传输及后台加载控制 |
| Performance Monitor | 性能测试、运行时统计、延迟与资源竞争分析 |

### 2.1 架构示意

```text
                         GEC Control Plane
                                |
                 +--------------+--------------+
                 |              |              |
              Global LFU   GEC Manager   Pool Registry
                 |              |              |
                 +--------------+--------------+
                                |
                     Logical Expert Table
                                |
                       TP Rank Coordinator
                                |
                +---------------+---------------+
                |                               |
         Host Weight Store                GPU Slot Pools
          完整专家权重                  Rank-local 物理数据
                |                               |
         CPU Expert Executor              GPU Expert Executor
                |                               |
                +---------------+---------------+
                                |
                       TP Coordination
                                |
                         Layer Join/Combine
                                |
                            Next Layer

       Transfer Manager / Performance Monitor
        管理异步传输、后台 H2D 和性能反馈
```

### 2.2 核心原则

- 专家驻留管理与当前专家计算执行分离。
- 逻辑专家统一管理，物理 Slot Pool 按兼容格式拆分。
- TP=2 的物理数据和执行由 Rank 协调机制管理。
- CPU/GPU 计算并行，完成后按模型原始语义 Join/Combine。
- 后台专家换入不参与当前 Layer 的计算依赖。
- CUDA Graph 和 TMA 不作为 GEC 基础功能成立的前置条件；先验证普通 CUDA Stream 下的动态 Slot。
- 在底层 MXFP4 Kernel 和 KT TP=2 CPU Expert 路径完成审计之前，不冻结最终 Kernel ABI、Slot 寻址方式和 Generation 细节。

---

## 3. Host Weight Store 与 GPU Slot Pool

### 3.1 Host Weight Store

Host RAM 保存完整模型专家权重，是 CPU 专家执行的权重来源。

CPU Expert 直接从 Host RAM 读取所需权重，不需要：

- 从 GPU D2H 获取专家权重。
- 推理时从 SSD 重新读取专家文件。
- 因为 GPU 缓存未命中而等待权重换入。

Host 侧可以根据实际需要保留：

- Canonical / Raw Weight。
- CPU Execution Format。
- 可复用的 GPU Final Format。

Host 权重位于 RAM，不代表所有页面都必须注册为 CUDA Pinned Memory。CPU 可访问的普通 Host Memory 与 GPU DMA 使用的 Pinned Memory 是不同资源。

### 3.2 GPU Slot Pool

GPU VRAM 保存 GEC 选择的热门专家权重。Slot Pool 保存对应 GPU Kernel 可以直接执行的最终权重格式，避免长期同时保存 Raw、Intermediate、Final 等多份重复数据。

GPU Slot Pool 是长期驻留缓存，不是 H2D 临时传输缓冲区。

### 3.3 Host Final Format Cache

Host Final Format Cache 纳入整体设计，但其缓存范围和策略通过后续测量决定。

```text
Host RAM
   +-- Canonical / Raw Weights
   +-- CPU Execution Format
   +-- GPU Final Format Cache
                 |
                 +-- Rank 0 对应物理表示 --> GPU 0 Slot Pool
                 +-- Rank 1 对应物理表示 --> GPU 1 Slot Pool
```

其目的在于避免每次专家换入时重复执行 Dequant、Packing 或 Format Conversion。CPU 与 GPU 的执行格式可以不同，不要求所有格式都永久保留。

---

## 4. GEC 全局专家缓存

### 4.1 Logical Expert Table

全模型维护统一的 Logical Expert Table。逻辑专家身份不随缓存换入、淘汰或 TP 分片变化。

逻辑身份：

```text
LogicalExpert = (layer_id, expert_id)
```

建议记录以下信息：

| 属性 | 含义 |
|---|---|
| logical_expert_id | 全局逻辑专家身份 |
| layer_id | Layer 编号 |
| expert_id | Expert 编号 |
| layout_id | 权重物理布局标识 |
| pool_id | 所属兼容格式 Pool |
| rank_slots | 各 Rank 对应的物理位置 |
| global_state | 逻辑专家整体驻留状态 |
| rank_states | 各 Rank 本地状态 |
| lfu_count | 有效 Router 激活 Token 统计 |
| host_weight | Host 权重位置 |
| generation | 映射版本 |
| active_refs | 在途 Kernel 对 Slot 的使用引用 |

### 4.2 TP Rank 物理表示：Shard/Replica 由审计结果确定

TP=2 下，不能在没有审计当前模型和后端实现的情况下，先验假定每个逻辑专家一定是两个等尺寸 Shard。

不同模型结构和执行后端可能采用不同的 Rank-local 物理表示，例如：

- **Shard**：逻辑专家权重按 TP 维度切分，每个 Rank 保存和执行对应分片。
- **Replica**：某些专家权重在不同 Rank 上复制，具体执行职责由后端决定。
- **其他 Rank-local 表示**：由实际模型结构、Kernel 布局和通信语义决定。

因此 Logical Expert Table 应维护 `rank_slots[]` / `rank_states[]`，但每个 Rank 实际需要保存什么权重、是否必须参与当前专家计算、输出如何组合，必须由 Phase 2 代码审计确认。

**逻辑专家的 GPU READY 条件不是简单写死为“两个相同 Shard 都 READY”，而是：当前执行路径要求的全部 Rank-local 物理数据均已就绪，且对应映射和 Kernel 资源有效。** 对于需要两个 Rank 协同的分片专家，两个 Rank 均 READY 才能发布逻辑 READY。

### 4.3 多格式 Slot Pool

借鉴 FreeToken 的多格式 Slot Pool 设计，根据 `layout_id` 划分多个物理 Pool。

`layout_id` 综合考虑：

- Quant Format。
- Expert Shape。
- Packing 和 Bank Layout。
- Scale 布局。
- Alignment 和 Stride。
- Rank-local Shard/Replica Layout。
- GPU Kernel 所要求的权重格式。

设计原则：

- 相同物理布局的不同 Layer 可以共享 Pool。
- 不同物理布局的专家进入不同 Pool。
- 所有 Pool 由统一 Pool Registry 管理。
- 逻辑专家映射全局统一。
- 每个 Rank 维护其实际需要的物理 Slot 和状态。

### 4.4 全局 LFU

采用全局 LFU 管理专家热度：

- 覆盖模型中所有动态管理的逻辑专家。
- 统计 Router 实际分配给专家的 Token 数量。
- CPU 和 GPU 专家均参与统计。
- 预取、换入和 GPU 命中本身不增加真实访问次数。
- Aging 由有效统计事件推动。
- 共享专家默认 Pinned，不参与普通路由专家淘汰。

LFU 统计全局统一，但物理淘汰必须受 Pool 兼容性约束。不同 Pool 的专家不能直接互相替换 Slot。

#### TP=2 LFU 计数去重

LFU 的统计单位是逻辑专家激活事件，而不是物理 Rank 事件。同一个 Token 对同一个逻辑专家的一次路由激活只能统计一次，不能因为两个 Rank 都处理或收到相同路由信息而重复计数。

应通过明确的统计归属或事件去重保证：

- 统计覆盖所有合格请求。
- CPU/GPU 执行位置不影响统计。
- Rank-local 重复事件不重复累计。
- 后台加载和 GPU 命中不虚增热度。

### 4.5 GPU Slot 容量

保留参数：

```text
--kt-num-gpu-layers N
--kt-expert-gpu-slot K
```

定义：

- `--kt-num-gpu-layers N`：前 N 层必要模型权重按 TP=2 的实际切分方式常驻 GPU。
- `--kt-expert-gpu-slot K`：其余动态管理专家的全局**逻辑专家 Slot 总预算**，不是两张卡的物理 Slot 数量之和。

多格式 Pool 的逻辑容量关系：

\[
\sum_{p=1}^{n}K_p=K
\]

其中 \(K_p\) 是格式 Pool p 可缓存的逻辑专家数量。

每张 GPU 的实际显存占用分别核算：

\[
M_r=\sum_{p=1}^{n}K_p\times S_{p,r}+O_r
\]

其中：

- \(S_{p,r}\)：Pool p 中一个逻辑专家在 Rank r 上实际需要的物理存储大小；它由 Shard、Replica 或实际 Rank-local 表示决定。
- \(O_r\)：该 GPU 上的映射表、Descriptor、Staging 和其他运行时开销。

如果某种布局在某 Rank 上不需要保存对应数据，则按实际需求计算，不能假设所有 Rank 的 Slot 字节数必然相同。

必须分别检查 GPU 0 和 GPU 1 的显存预算，不能只看两张卡的显存总和。

### 4.6 静态 Pool 容量与全局 LFU 的张力

第一版采用静态 Pool 容量分配：

1. 优先预留共享专家 Pinned 容量。
2. 根据各格式专家数量、字节大小和 Router Profile 分配路由专家容量。
3. Global LFU 统一统计热度。
4. Pool Manager 在兼容 Pool 内选择 Victim。
5. 第一版不支持不同格式 Pool 任意借用 Slot。

需要监控：

- Pool Utilization。
- Pool GPU Hit Rate。
- Pool Collision Miss。
- Pool Eviction Rate。
- Pool Free Slots。

后续根据实际数据决定是否增加安全点上的 Pool Rebalance。

---

## 5. 请求级 GEC 准入

保留参数：

```text
--kt-gpu-prefill-token-threshold T
```

在当前分支中定义为请求级 GEC 准入条件：

- Prefill Token 数量达到阈值：请求启用 GEC。
- 不达标：请求正常执行，但不更新 LFU，也不触发 GEC 后台管理。
- 达标请求进入 Decode 后，继续参与 LFU 更新和后台调度。
- 混合 Batch 必须按单个请求判断，不能使用 Batch 总 Token 数替短请求取得资格。
- 准入状态贯穿完整请求生命周期。

这是 GEC 分支的新语义，需要独立的 Request GEC Context，不能直接复用 KTransformers 原有 Chunk-level Token Predicate。

---

## 6. CPU/GPU MoE 异步执行

### 6.1 执行原则

Router 完成专家分配后：

- GPU READY 专家交给 GPU Executor。
- 未驻留 GPU 的专家交给 CPU Executor。
- CPU 和 GPU 在同一个 Layer 内并行计算。
- 所有任务完成后执行 Join 和 Combine。
- 完成当前 Layer 后进入下一 Layer。

不采用 Expert Deferral，也不因缓存未命中触发当前请求的 Demand H2D。

### 6.2 CPU Expert 权重访问

CPU Expert 直接读取 Host RAM 中的专家权重。CPU 专家执行不需要专家权重 D2H，也不需要等待 GPU 缓存加载。

### 6.3 TP=2 CPU Expert 执行语义必须先审计

在审计 KT 当前代码之前，不预设 CPU Expert 一定按 Rank-local Shard 执行，也不预设 CPU 一定执行完整逻辑专家。

必须先确认：

- CPU 权重加载的是完整专家还是 Rank-local 权重。
- FC1/FC2 的分片维度和真实 Shape。
- Scale、Packing 和量化元数据如何组织。
- CPU Expert 由一个 Rank、两个 Rank 还是独立 CPU Worker 执行。
- Token Mapping 是全局还是 Rank-local。
- CPU 输出是完整 Hidden 还是局部结果。
- CPU 输出参与 TP Combine 的具体方式。

Phase 2 以代码审计为第一步，审计结果决定 Phase 3 的唯一主执行方案，不把未经证实的两种方案同时当成既定架构。

### 6.4 CPU/GPU 结果合并

CPU 与 GPU 专家结果必须保持原始 MoE 计算语义：

- Token 对应关系。
- 专家索引和 Routing Weight。
- 数值精度和累加语义。
- 输出布局和顺序。
- TP 通信语义。

CPU 局部加权 Combine 与 CPU 专家分别输出两种路径可作为原型对比，但最终方案必须由 KT TP=2 实际权重布局、输出布局和测量结果决定。

Routing Weight 只能应用一次。

---

## 7. Transfer Manager：异步数据传输

### 7.1 传输类型

| 传输类型 | 方向 | 用途 |
|---|---|---|
| Activation D2H | GPU → Host | CPU Expert 需要 GPU 上的输入时 |
| Expert Result H2D | Host → GPU | GPU 负责后续 Combine 时 |
| Background Expert H2D | Host → GPU | GEC 后台加载专家的 Rank-local 物理数据 |

CPU 专家权重不需要 D2H，因为完整权重一直保存在 Host RAM。

Activation D2H 和 Result H2D 都是条件性传输，不是每次 CPU Expert 执行都必须发生的固定步骤。

### 7.2 Activation 传输

CPU 和 GPU 专家处理的是同一 MoE Layer 的输入 Activation，不是 GPU 专家计算后的结果。

如果 Activation 位于 GPU，而 CPU Expert 需要它，则传输分配给 CPU 专家的 Token 数据。无需无条件传输整个 Layer 的 Activation。

同一 Token 被多个 CPU 专家使用时，可以研究 Gather 去重和输入复用。如果 Activation 已经位于 Host，则不需要 D2H。

TP=2 下，实际需要传输的 Rank-local Activation、是否重复、是否存在共享输入，必须按审计得到的执行布局确定。

### 7.3 CPU 结果传输

- 如果最终 Combine 在 GPU，CPU 结果需要 H2D。
- 如果 CPU 可以先完成局部加权累加，可以减少 Result H2D 数据量。
- 如果后续计算和 Combine 在 CPU，则不需要为了 Combine 强制传回 GPU。
- TP=2 下，结果传输目的 Rank 和输出分片方式由实际 TP 执行语义决定。

### 7.4 Pinned Staging

借鉴 FreeToken 的 Pinned Memory 管理方式：

- Host Weight Store 不必全部注册为 CUDA Pinned Memory。
- 必要时采用有限的 Pinned Staging Buffer。
- Staging Buffer 独立于 GPU Slot Pool。
- 防止锁页资源耗尽影响模型运行。
- 每张 GPU 的传输资源可分别观测，逻辑专家加载事务由 Rank Coordinator 协调。

### 7.5 后台 H2D 反馈闭环

Transfer Manager 管理当前请求必要数据传输和 GEC 后台专家权重加载。

后台 H2D 不是固定并发数，也不应只根据理论 PCIe 带宽确定。需要根据实际前台延迟、每张 GPU 传输压力和 Host Memory 竞争调节。

反馈关系：

```text
Global LFU
    |
选择候选逻辑专家
    |
GEC Manager
    |
检查各 Rank 资源和 Slot 生命周期
    |
Transfer Manager
    |
Rank 0 / Rank 1 后台异步传输
    |
检查全部必要 Rank-local 数据是否完成
    |
发布逻辑专家 READY
    |
收集前台延迟与后台传输指标
    |
调整后台 H2D 调度
```

---

## 8. PCIe 后台 H2D 与 P99 TPOT

### 8.1 设计前提

PCIe 并发偏斜属于硬件和系统资源层面的风险。Transfer Manager 的反馈控制可以降低冲突概率，但不能保证任何负载下都不影响 P99。

因此，不能只以平均吞吐或后台 H2D 完成量作为验收标准。

### 8.2 Decode 窗口暂停机制

Transfer Manager 应允许在 Decode 关键窗口暂停**新后台 H2D 任务的提交**：

- 当前 Decode 负载进入关键窗口时，暂停提交新的后台专家权重传输。
- 已提交的 DMA 任务通常不能像普通队列任务一样立即取消，应安全完成或等待其完成事件。
- 未完成加载的逻辑专家不得提前发布 READY。
- 暂停期间保留候选任务及事务状态。
- 负载回落后，根据恢复条件继续后台加载。
- 应设置恢复迟滞，避免频繁暂停/恢复。

TP=2 下，两张 GPU 的后台加载状态由逻辑事务协调。某个 Rank 已完成而另一个 Rank 未完成时，逻辑专家仍保持未 READY。

### 8.3 P99 TPOT 验收

定义：

\[
\Delta P99(TPOT)=P99(TPOT)_{\text{后台 H2D}}-P99(TPOT)_{\text{Baseline}}
\]

相对增幅：

\[
R_{P99}=\frac{\Delta P99(TPOT)}{P99(TPOT)_{\text{Baseline}}}\times100\%
\]

Baseline 与测试组必须使用相同模型、请求分布、输入输出长度、并发量和运行条件。验收前应根据服务目标明确可接受的绝对增量或相对增幅，不在没有基线数据时臆造固定百分比。

建议比较：

| 场景 | 后台 H2D 策略 | 观察指标 |
|---|---|---|
| Baseline | 关闭后台 H2D | P50/P95/P99 TPOT、TTFT、吞吐 |
| Test A | 低并发后台 H2D | 同上 |
| Test B | 高并发后台 H2D | 同上 |
| Test C | Decode 关键窗口暂停新任务 | 同上 |
| Test D | 自适应反馈调度 | 同上 |

同时记录每张 GPU 的 H2D 带宽、队列等待、Staging 占用和 Rank 间加载偏斜。

---

## 9. FreeToken 工程经验借鉴

| FreeToken 经验 | GEC 设计决策 |
|---|---|
| 多格式 Slot Pool | 采用，逻辑统一、物理分池 |
| CPU Executor 模块化 | 借鉴，Executor 与 Format/CPU/GPU/Sync Backend 解耦 |
| CPU/GPU 混合执行 | 采用，但执行语义以 KT TP=2 审计为准 |
| 带宽 Benchmark | 采用，建立 ARM64 + SM120 + TP=2 Profile |
| Runtime Stats | 采用，分 Cache/Execution/Transfer 三类观测 |
| Pinned Staging | 借鉴，独立于 GPU Slot Pool |
| Spin-Then-Park | 检查 KT 现有 Worker 后再决定 |
| SM120 专用 Kernel | 研究，不假设现有 NVFP4 Kernel 可直接用于 MXFP4 |
| Prefill 双缓冲 | 借鉴传输与计算重叠思想，后续验证 |
| Layer-major Prefill | 后续独立实验 |
| LRU 按需换入 | 不采用，保留全局 LFU 后台驻留 |
| 当前 Step Hybrid Fetch | 不采用 |
| Per-Layer 固定缓存池 | 不采用 |

### 9.1 CPU Executor 模块化

设计上分离：

- Executor：任务调度。
- Format Backend：不同量化格式。
- CPU Backend：ARM64 CPU 计算。
- GPU Backend：GPU 专家 Kernel。
- Sync Backend：CPU/GPU 与 TP 同步。

ARM64 必须独立验证 NEON/SVE 路径，不能直接假设 x86 CPU Kernel 可以复用。

### 9.2 Performance Profile

借鉴 FreeToken 的带宽 Benchmark 思路，针对真实模型和专家 Shape 测量：

- ARM64 CPU MXFP4 Expert GEMM。
- SM120 GPU MXFP4 Expert GEMM。
- Host Memory 权重读取。
- H2D / D2H 实际带宽。
- CPU + GPU 并行执行。
- CPU + Transfer 并行资源竞争。
- TP=2 同步和通信成本。
- 多请求并发下的端到端表现。

不能直接根据 STREAM、memcpy 或理论 PCIe 带宽决定执行策略。

### 9.3 SM120 Kernel 选择原则

SM120 MXFP4 Kernel 不应被描述为完全不可用；但现有 Kernel 路径在目标模型、当前软件版本、TP=2、动态 Slot 和目标权重布局下是否稳定、是否满足性能要求，尚需验证。

Phase 1 首先跑通当前可用的 FlashInfer MXFP4 MoE 路径，建立真实正确性、吞吐和内存访问基线。若环境中存在可用的 CUTLASS/Triton 等替代路径，可作为对照测试，但不能预设一定可用或一定更快。

TMA 不作为基础执行路径的前置条件。先确认现有 Kernel 的输入/输出地址、权重布局、Workspace 和内存访问约束，再决定是否需要 TMA 或其他专用 Kernel 优化。

---

## 10. Prefill 双缓冲与异步传输重叠

### 10.1 双缓冲的目的

FreeToken Prefill 双缓冲的核心是当前批次计算时，另一组临时缓冲区准备或传输下一批数据，以隐藏部分数据准备时间。

```text
时间 T1：Buffer A 被当前计算使用，Buffer B 准备下一批数据
时间 T2：GPU 计算 A，Copy Stream 传输/准备 B
时间 T3：必要事件完成后交换 A / B
时间 T4：GPU 计算 B，Buffer A 准备下一批数据
```

双缓冲不是第二套 GEC Cache，也不意味着所有专家永久保存两份 GPU 权重。

### 10.2 与 GEC 的区别

| 项目 | Prefill 双缓冲 | GEC |
|---|---|---|
| 目的 | 隐藏 Prefill 数据传输等待 | 管理长期热门专家驻留 |
| 资源 | 临时 Staging/传输 Buffer | GPU Slot Pool |
| 决策 | Prefill 数据流水线 | Global LFU |
| 生命周期 | 临时 | 长期驻留 |
| 当前执行 | 可参与当前 Prefill 数据准备 | 不改变当前专家执行位置 |

### 10.3 GEC 中的应用方向

GEC 只借鉴异步传输与计算重叠思想：

- 当前 GPU Expert 计算时，后台可准备未来专家权重。
- 使用独立 Staging Buffer。
- 通过 Event 保证传输完成后才发布 READY。
- 两 Rank 需要的物理数据全部完成后才发布逻辑 READY。
- 在途 Kernel 正在使用的 Slot 不允许提前覆盖。
- 后台传输不能明显恶化 P99 TPOT。

不采用 Prefill 时绕过 LFU、把整层专家全部搬到 GPU 的做法。

### 10.4 Layer-major Prefill

Layer-major Prefill 是不同于普通双缓冲的执行顺序优化，涉及 Chunk 调度、Layer 执行顺序、Attention、KV Cache 和专家权重传输组织。

该设计不纳入 GEC V1.6 核心执行机制，作为后续独立实验；必须在 TP=2 目标环境中验证，不能用 TP=1 结果替代。

---

## 11. CUDA Graph、Workspace 与动态 Slot

### 11.1 风险说明

CUDA Graph 的 Mapping Generation 只能标识逻辑映射版本，不能自动保护 Kernel 内部的 Workspace、Scratch Buffer、TMA Descriptor 或其他内部地址。

必须分别管理：

- Logical Mapping Generation。
- GPU Slot 生命周期和在途引用。
- Kernel Workspace 地址及生命周期。
- TMA Descriptor 的有效性和更新方式。
- Graph Replay 对地址和参数的引用。

### 11.2 验证顺序

**先在不启用 CUDA Graph 的普通 CUDA Stream 路径下验证动态 Slot。**

验证通过后，再把 CUDA Graph 作为独立、受限的优化方向：

1. 审计 Kernel Workspace 的分配和释放方式。
2. 明确 Kernel 内部地址是否会在 Replay 期间变化。
3. 确认 Mapping Table 更新与 Graph Replay 的同步边界。
4. 验证 Slot 不会在在途 Kernel 使用期间回收。
5. 单独验证 Graph Capture、Replay 和受限更新能力。
6. TP=2 下验证两个 Rank 的映射和版本一致性。

### 11.3 回退原则

如果 Graph 与动态 Workspace/Slot 的组合不稳定：

- GEC 核心功能继续使用普通 CUDA Stream。
- 可以按布局使用独立 Graph，前提是其资源与映射生命周期可证明安全。
- 不为追求 Graph 而牺牲动态 Slot 的正确性。

**CUDA Graph 不是 GEC 功能成立的前置条件。**

---

## 12. 前置基础工作与接口冻结门槛

Phase 1 和 Phase 2 是 GEC 叠加之前的基础执行路径确认，不是 GEC 缓存算法本身的验证。

### 12.1 前置工作 A：SM120 MXFP4 GPU Kernel 基线

目标：确认目标环境中可用的 MXFP4 MoE Kernel、真实性能和内存访问约束。

工作顺序：

1. 先跑通当前可用的 FlashInfer MXFP4 MoE 路径。
2. 使用目标模型的真实专家权重、Shape 和 Scale 布局。
3. 测量 Decode 和 Prefill 的吞吐及延迟。
4. 记录权重读取方式、内存访问模式、Workspace 大小和生命周期。
5. 确认 Kernel 对输入/输出指针、地址对齐和权重布局的要求。
6. 若环境中可用，再对照 CUTLASS/Triton 等路径。
7. 在 Kernel 基线明确后，再讨论固定 Slot 地址、动态 Slot 寻址和 Pool 接入。

第一步不以 TMA 或 CUDA Graph 为依赖，不先假定“固定 Slot + TMA”就是基础实现。

交付物：

- 可复现的 SM120 MXFP4 MoE 正确性与性能基线。
- Kernel 输入/输出和权重布局约束。
- Workspace/临时内存生命周期说明。
- Kernel 与后续 Slot Pool 的接入边界建议。

### 12.2 前置工作 B：KT TP=2 CPU Expert 路径审计

目标：确认现有 KTransformers TP=2 下 CPU Expert 的真实权重布局、执行归属和输出语义。

审计内容：

| 审计项 | 必须确认的问题 |
|---|---|
| Expert 权重加载 | CPU 保存完整专家还是 Rank-local Shard/Replica？ |
| 权重布局 | FC1/FC2 的分片维度和真实 Shape？ |
| 量化元数据 | Scale、Packing 和其他元数据如何组织？ |
| Rank 职责 | CPU Expert 由哪个 Rank 或哪些执行实体负责？ |
| Router 输出 | Token Mapping 是全局还是 Rank-local？ |
| CPU GEMM | 实际 CPU Kernel 使用什么权重布局和 Shape？ |
| 输出布局 | CPU 输出是完整 Hidden 还是部分结果？ |
| TP 通信 | CPU 输出如何进入 TP Combine？ |
| 内存管理 | 是否存在 Rank 间权重重复存储？ |
| 回退路径 | GPU 未命中时，TP=2 CPU fallback 是否语义完整？ |

可以把 SGLang 当前对应路径作为对照材料，但不能假定 SGLang 与 KTransformers 的 CPU Expert 布局、调度或 TP 语义相同。设计结论必须以目标 KT 分支的实际代码和运行行为为准。

交付物：

- KT TP=2 CPU Expert 当前数据流图。
- CPU Expert 权重布局及 Shape 说明。
- Rank 职责和 Token Mapping 说明。
- CPU 输出与 TP Combine 的实际语义。
- 确定 Phase 3 应采用的 CPU Expert 执行方案。

### 12.3 GEC 接口冻结门槛

只有前置工作 A、B 完成后，才冻结以下设计：

- GPU Expert Kernel ABI。
- Slot 数据布局和寻址方式。
- Workspace 生命周期约束。
- Rank-local 物理表示。
- Logical Expert READY 条件。
- CPU Expert 权重和输出布局。
- Mapping Generation 与在途引用保护边界。

如果基础 Kernel 或 TP CPU Expert 路径需要重写，应先调整底层接口，再冻结 GEC Slot 和 Generation 约定。不得以“Kernel 必须配合既定 GEC 接口”为前提强行推进。

---

## 13. 性能观测体系

### 13.1 GEC Cache

- GPU Expert Hit Rate。
- LFU Token Frequency。
- Pool Utilization。
- Eviction Count。
- Collision Miss。
- Background H2D。
- Loading Wait。
- Rank-local Slot Utilization。
- Rank Shard/Replica READY 状态。
- 逻辑专家 READY 等待时间。

### 13.2 MoE Execution

- CPU Expert Time。
- GPU Expert Time。
- CPU/GPU Overlap。
- CPU Join Wait。
- GPU Join Wait。
- Combine Time。
- TP Coordination Time。
- CPU Expert 输出布局和合并耗时。

### 13.3 Transfer

- Activation D2H Bytes。
- Result H2D Bytes。
- Expert Weight H2D Bytes。
- Transfer Wait。
- Staging Wait。
- GPU 0 H2D Throughput。
- GPU 1 H2D Throughput。
- 双卡并发传输吞吐。
- Rank 间加载时间偏斜。

### 13.4 服务指标

- TTFT。
- TPOT P50/P95/P99。
- 请求吞吐。
- 后台 H2D 开启/关闭时的延迟差异。
- Decode 窗口暂停前后的延迟变化。

---

## 14. CLI 参数设计

保留核心参数：

| 参数 | 含义 |
|---|---|
| `--kt-num-gpu-layers N` | 前 N 层必要权重按 TP=2 方式常驻两张 GPU |
| `--kt-expert-gpu-slot K` | 全局逻辑专家 Slot 总容量 |
| `--kt-expert-cache-policy lfu` | 全局 LFU 策略 |
| `--kt-gpu-prefill-token-threshold T` | 请求级 GEC 准入阈值 |

`--kt-expert-gpu-slot K` 表示逻辑专家容量；每个逻辑专家实际在各 Rank 需要的物理数据量由审计结果决定，并分别计算两张卡的显存占用。

`--kt-expert-gpu-slot K` 的启动布局由 `--kt-expert-placement-strategy` 选择：`front-loading`（默认，按层顺序填满）或 `uniform`（按 MoE 层均摊）。该选择只影响启动布局，运行期全局 LFU 仍可在层间迁移。

后台 H2D 并发、Staging Buffer 和暂停/恢复策略由内部 Transfer Manager 管理。

不重新引入已经删除的以下参数：

- `--kt-layer-h2d-slots`
- `--kt-h2d-prefetch-depth`
- `--kt-h2d-prefetch`
- `--kt-h2d-max-inflight-experts`

---

## 15. Phase 1–6 原型与验收计划

**Phase 1 和 Phase 2 是 GEC 叠加前的基础前置工作。所有系统级阶段最终均必须通过 TP=2 验收。**

### Phase 1：SM120 MXFP4 GPU Kernel 基线（前置工作 A）

目标：先确认目标环境下的 MXFP4 MoE 执行路径可用，再决定 Slot Pool 如何接入。

工作内容：

1. 跑通当前可用的 FlashInfer MXFP4 MoE 路径。
2. 使用目标模型实际专家权重和 Shape。
3. 建立 Decode/Prefill 正确性、吞吐和延迟基线。
4. 测量内存访问模式和权重布局。
5. 审计 Workspace、临时内存、输入/输出地址约束。
6. 可用时对照其他 SM120 Kernel 路径。
7. 在基线稳定后再讨论固定 Slot 和动态 Slot 的 Kernel 接口。

单卡可用于隔离测试，但阶段结束前必须确认该路径能够进入目标 TP=2 GPU 执行流程。

验收条件：

- 目标 MXFP4 MoE 路径能够正确执行。
- 有可重复的真实性能数据。
- 权重布局、地址和 Workspace 约束明确。
- 不依赖 TMA 或 CUDA Graph 才能完成基础执行。
- TP=2 集成条件和限制已明确。

### Phase 2：KT TP=2 CPU Expert 路径审计（前置工作 B）

目标：先确定现有代码的真实 CPU Expert 执行语义，而不是提前并列假设两套方案。

工作内容：

1. 审计 KT TP=2 的 Expert 权重加载路径。
2. 确定 CPU 权重是完整专家还是 Rank-local 表示。
3. 确认 FC1/FC2、Scale 和 Packing 的真实布局。
4. 确认 Router、Token Mapping 和 CPU Worker 的 Rank 职责。
5. 确认 CPU 输出如何参与 TP Combine。
6. 确认 GPU 缓存未命中时现有 CPU fallback 是否完整。
7. 参考 SGLang 对应实现，但不混同两个项目的执行语义。

验收条件：

- 当前 KT TP=2 CPU Expert 数据流有明确说明。
- 权重布局和输出布局明确。
- Rank 职责明确。
- CPU fallback 的正确性边界明确。
- Phase 3 的 CPU Expert 执行方案能够依据审计结果确定。

### Phase 3：TP=2 单 Layer CPU/GPU 异构执行原型

目标：在不引入完整 GEC 的情况下，验证当前 Layer 的 CPU/GPU 专家执行正确性和性能。

工作内容：

- 根据 Phase 2 审计结果实现实际 CPU Expert 执行路径。
- GPU Expert 使用 Phase 1 验证的 Kernel。
- 验证 Token Gather 和必要 Activation D2H。
- 验证 CPU Expert GEMM。
- 比较 CPU 局部 Combine 与独立输出方案。
- 验证必要 Result H2D 和 TP 通信。
- 完成 Layer Join。

验收条件：

- 与原始 TP=2 执行结果在约定数值容差内一致。
- Token Mapping 正确。
- Routing Weight 只应用一次。
- CPU/GPU 并行执行正确。
- TP 通信和输出语义正确。
- CPU/GPU Join 等待时间可测量。

### Phase 4：GEC 逻辑管理、多格式 Pool 与事务式加载

目标：引入 GEC 核心管理能力。

工作内容：

- Logical Expert Table。
- Global LFU 与请求级准入。
- 多格式 Pool Registry。
- 静态 Pool 容量分配。
- Shared Expert Pinned。
- Pool 内 LFU 淘汰。
- Rank-local Slot 预留。
- 双 Rank 加载事务。
- 全部必要物理数据 READY 后发布逻辑 READY。
- 任一 Rank 加载失败时安全回滚并使用完整 CPU fallback。

动态 Slot 首先在普通 CUDA Stream 下验证，不依赖 CUDA Graph。

验收条件：

- 不同物理布局不会混用 Pool。
- K 的逻辑 Slot 容量语义正确。
- 两张 GPU 分别通过显存预算校验。
- 加载事务只在必要 Rank-local 数据全部完成后发布 READY。
- 淘汰和加载不会造成逻辑映射不一致。
- CPU fallback 正确。

### Phase 5：动态 Slot 生命周期、Generation 与 CUDA Graph 独立验证

目标：确保动态 Mapping 和在途 Kernel 安全，并评估 CUDA Graph 是否值得作为优化启用。

第一部分：普通 CUDA Stream 下的动态 Slot 安全。

- Slot 引用和回收。
- 在途 Kernel 保护。
- Mapping Generation。
- Rank 映射一致性。
- 并发加载和淘汰。
- 多请求并发。

第二部分：CUDA Graph 独立、受限验证。

- Workspace 地址和生命周期审计。
- Graph Capture / Replay。
- Graph Replay 对 Mapping Generation 的引用。
- 映射更新的安全边界。
- TMA Descriptor（如该 Kernel 路径需要）。
- TP=2 两 Rank 映射版本一致性。

验收条件：

- 普通 CUDA Stream 路径可稳定使用动态 Slot。
- 不发生在途 Slot 被提前回收或覆盖。
- 不发生旧 Generation 访问新旧映射混乱。
- CUDA Graph 若不能安全支持动态更新，不阻塞 GEC 核心功能，系统保留 Stream 路径。

### Phase 6：Transfer Manager 闭环与 TP=2 系统验收

目标：建立后台 H2D 反馈控制，并以服务尾延迟验证其影响。

工作内容：

- Transfer Manager。
- 双 GPU H2D 调度。
- Pinned Staging。
- 当前请求传输与后台 H2D 协调。
- Decode 窗口暂停新后台 H2D。
- 负载恢复后的后台任务恢复。
- Prefill 双缓冲原型。
- Performance Profile 和 Runtime Metrics。
- 端到端 TP=2 测试。

验收条件：

- 后台 H2D 可以根据前台负载降低、暂停和恢复。
- 已提交 DMA 安全完成，未完成事务不会提前发布 READY。
- 两张 GPU 的传输负载和等待可独立观测。
- P99 TPOT 增量满足预先定义的服务目标。
- GEC 命中率、后台加载和端到端推理性能之间的关系可测量。
- Prefill 双缓冲是否有收益有实际数据结论。
- 完成完整 TP=2 系统验收。

---

## 16. 风险清单与回退策略

| 风险 | 优先级 | 阶段 | 回退/处理方向 |
|---|---|---|---|
| SM120 MXFP4 Kernel 基线不可用或不稳定 | P0 | Phase 1 | 先修复/替换基础 Kernel，不叠加 GEC |
| Kernel 权重布局或 Workspace 不适合动态 Slot | P0 | Phase 1/4 | 根据真实 Kernel ABI 调整 Slot 布局 |
| KT TP=2 CPU Expert 布局不明确 | P0 | Phase 2 | 先审计代码和运行路径，不提前确定 Shard/完整专家方案 |
| ARM64 MXFP4 CPU 吞吐不足 | P0 | Phase 2/3 | 优化 CPU Kernel、执行粒度和 CPU/GPU 分配 |
| CPU/GPU Combine 数值语义不一致 | P0 | Phase 3 | 以原始 TP=2 语义修正输出布局和加权位置 |
| 多格式 Pool 容量分配不合理 | P1 | Phase 4 | 调整静态 Profile，后续再研究 Rebalance |
| 双 Rank 加载事务失败 | P0 | Phase 4 | 回滚预留 Slot，完整 CPU fallback |
| 动态 Slot 生命周期错误 | P0 | Phase 5 | 保守同步和延迟回收，不能牺牲正确性 |
| CUDA Graph 与 Workspace 冲突 | P1 | Phase 5 | 保留普通 CUDA Stream 路径 |
| PCIe 后台 H2D 导致 P99 恶化 | P0 | Phase 6 | Decode 窗口暂停新后台传输，降低并发 |
| Prefill 双缓冲收益不足 | P2 | Phase 6 | 保留简单异步传输，不增加不必要复杂度 |
| TP=4/8 扩展 | 后续 | V1.6 之后 | 基于 Rank 数组和 Shard 映射扩展 |

---

## 17. 最终架构决策

### 17.1 保留的核心架构

| 设计项目 | 决策 |
|---|---|
| 基础并行模式 | TP=2 |
| TP=1 | 仅用于局部 Kernel/传输基准，不代替系统验收 |
| TP=4/8 | 后续扩展 |
| 模型权重 | Host RAM 保存完整权重 |
| GPU 缓存 | 多格式 Slot Pool |
| 逻辑映射 | 全模型统一 Logical Expert Table |
| 物理映射 | Rank-local Shard/Replica 表示由审计确定 |
| LFU | 全局统计、Pool 内淘汰 |
| Shared Expert | 默认 Pinned |
| GEC 更新 | 事件驱动后台管理 |
| 请求准入 | Request-level Threshold |
| CPU Expert | 直接读取 Host RAM 权重 |
| CPU/GPU 执行 | 同一 Layer 并行，遵循真实 TP 语义 |
| 当前缓存未命中 | CPU 执行，不触发 Demand H2D |
| Expert Deferral | 不采用 |
| 传输管理 | 条件性 D2H / Result H2D / 后台权重 H2D |
| 双 Rank 加载 | 事务式协调，按实际所需 Rank-local 数据发布 READY |
| Prefill 双缓冲 | 借鉴重叠思想，后续测量决定是否采用 |

### 17.2 前置门槛

完整 GEC 开发之前，必须完成：

1. SM120 MXFP4 MoE Kernel 基线、权重布局和 Workspace 审计。
2. KT TP=2 CPU Expert 权重布局、Rank 职责和输出语义审计。

这两项是 GEC 叠加之前的基础执行路径确认，不是 GEC 缓存算法本身的验证。

### 17.3 接口冻结原则

GEC 的 Kernel ABI、Slot 寻址、Mapping Generation 和 Rank-local READY 条件，必须根据 Phase 1/2 的真实结果确定。

如果基础 Kernel 或 KT CPU Expert 路径需要重写，应先调整基础执行接口，再冻结 GEC 设计。不能预设 Kernel 必须适配已有 Slot 方案。

---

## 18. 最终结论

GEC V1.6 保留 V1.5 在逻辑/物理分离、全局 LFU、多格式 Pool、请求级准入和事务式加载方面的核心架构。

本次修订的重点不是推翻 GEC，而是调整实现前置条件和验证顺序：

1. 先建立目标环境下 SM120 MXFP4 MoE Kernel 的真实基线。
2. 先审计 KT TP=2 CPU Expert 的实际权重布局和计算语义。
3. 根据前两项结果确定 Kernel ABI、CPU 执行组织和 Slot 寻址。
4. 在普通 CUDA Stream 下先验证动态 Slot，再独立评估 CUDA Graph。
5. 以双 Rank 事务式加载保证逻辑专家状态一致。
6. 以 P99 TPOT 增量、Decode 窗口暂停和恢复机制验收后台 H2D。

**总体架构：KTransformers 异步 CPU/GPU 执行基础 + FreeToken 多格式 Pool、Executor 模块化、性能校准和传输重叠经验 + GEC 全局 LFU 后台专家驻留 + TP=2 Rank 协调。**

本文件定义设计方向、模块边界、前置条件和阶段验收要求。底层 Kernel、CPU Expert TP 布局、CUDA Graph 兼容性及最终性能收益均以目标代码分支和 TP=2 实测结果为准，不作未经验证的假设。

---

## 参考资料

以下为原 V1.5 保留的 FreeToken 设计参考入口，具体实现和适用范围以实际使用的 FreeToken 版本为准：

1. [FreeToken CLI 文档](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)：多格式 Slot Pool、Prefill 双缓冲开关及运行时统计。
2. [FreeToken Releases](https://github.com/FlashML-org/FreeToken/releases)：CPU Executor 模块化、Pinned Staging、SM120 Kernel、CUDA Graph 和 Worker 优化。
3. [FreeToken Issue #750](https://github.com/FlashML-org/FreeToken/issues/750)：Layer-major Prefill 实验及适用条件。
4. [FreeToken Models 文档](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)：Host 专家权重和 CPU/GPU 混合执行相关说明。
5. [FreeToken Benchmark](https://www.freetoken.wiki/performance/FreeToken-benchmark)：CPU、GPU 和互联带宽校准思路。

**文档版本：V1.6 — TP=2 基础执行路径审计与原型验证版**
