# KT 按层动态专家缓存与 Prefill 流式换入实施技术设计

**状态：首版代码已完成；MXFP4 + KT Marlin 路径待 Linux CUDA / TP 实机验收**

**日期：2026-09-24**

首版实现说明：

- 已实现参数解析、按层固定容量、Decayed-LFU、Stream-TopN、不补位、三路 exact-once 分流、late CPU fallback、双 host/GPU staging、单专家 H2D/repack/wave 2、增量 resident install、decode/mixed freeze、TP 顺序校验和不可逆提交 fail-stop。
- 首个可执行 backend 收敛为 DeepSeek V4 MXFP4 + KT Marlin（SM89/SM120）；其他权重格式仍按本文支持矩阵 fail fast 或留待后续阶段。
- CPU/策略/状态机测试已完成；当前开发环境没有 CUDA GPU，因此真实 kernel 数值、H2D/计算重叠、TP2 多进程和端到端性能仍是合入前硬件验收项。
- 启用 `decayed-lfu` 且 `N_stream > 0` 时，MXFP4 writer 使用独立的内部 CPUInfer FIFO 和辅助 WorkerPool；每个 CPU TP/NUMA 分区默认 1 个 writer worker。此时 `--kt-cpuinfer` 被视为总 CPU 线程预算，内部自动从主 GEMM 池预留每个 threadpool 1 个核心给 writer，例如 `--kt-cpuinfer 168 --kt-threadpool-count 1` 对应 167 个主 worker + 1 个 writer worker，不增加新参数或额外核心。静态 MXFP4 与 `N_stream=0` 保持原有主 CPUInfer 线程数和共享 writer 行为。双 staging 首批 2 个 writer 立即激活；slot 释放后按确定性顺序滚动激活后续 tickets，使 host export、H2D/repack/wave 2 可与主 CPU GEMM 重叠。该资源仍必须观测 DRAM 带宽争抢。
- `N_stream` 定义每窗口最多选择并流式执行的 missing 热点专家总数；双 staging 的 `2` 只定义同时处于 writer/H2D/repack/wave 2 流水中的并发深度。`N_stream > 2` 时全部选中 candidates 仍保持 GPU streaming ownership，并随 slot 释放滚动执行，不回填到主 CPU task。
- writer 预启动后增加 TP prelaunch 共识。任一 rank 在 CPU assignment 过滤、主 CPU task 提交或 enqueue boundary 失败时，所有 rank 统一 abort tickets，禁止部分 rank 进入 wave 2 collective。
- V1.1 的首个 expert H2D 不再等待主 CPU GEMM，但仍等待 CPU 输入 D2H 与 CUDA host callback 的提交确认；该前缀通常是毫秒级，后续可用显式 submission receipt 进一步拆除。
- 双 staging 通过滚动 pump 驱动全部选中 candidates；candidate 2 的 H2D 与 candidate 1 的 wave 2 是否达到理想重叠仍需 Nsight 实机确认，后续可进一步拆分 prepare-ready queue 与独立 finalize 阶段。
- 首版 TP 控制面为保证一致性使用带完整 ticket/stage token 的 CPU-group object collective。其正确性边界已覆盖，但控制面延迟必须实机 profile；后续应改为固定 `int64` tensor token，并合并非关键阶段 collective。

**适用范围：KTransformers + SGLang 的 CPU/GPU 混合 MoE 推理路径**

## 文档导航

- [设计结论与现状](#1-结论摘要)
- [目标、参数与统计口径](#3-目标与非目标)
- [总体架构与数据结构](#6-总体架构)
- [路由热度与替换规则](#8-路由统计与热度信号)
- [Runtime 执行时序](#10-runtime-执行时序)
- [Prefill、decode 与 full-layer streaming](#11-prefilldecode-与-full-layer-streaming-的关系)
- [TP、CUDA Graph 与后端支持](#12-tp一致性与-cuda-graph)
- [失败回退与代码改动](#14-失败处理与回退)
- [实施、测试与性能验收](#16-分阶段实施计划)
- [风险与实施检查清单](#20-风险与待决事项)

## 1. 结论摘要

本文提出一套按层固定容量、仅由 prefill 驱动的有界流式专家缓存：当前 chunk 的 resident hit 立即走 GPU；不属于当前 Stream-TopN 的 non-resident experts 立即走 CPU；当前窗口 Stream-TopN 中缺失的 `0..N_stream` 个 candidate 从 CPU task 中排除，边 H2D/repack 边进入第二批 GPU GEMM，不等待所有 candidate 搬运完成。

推荐启动参数为：

```bash
--kt-num-gpu-experts 32 \
--kt-expert-placement-strategy decayed-lfu \
--kt-prefill-stream-top-n 4
```

核心语义：

- `--kt-num-gpu-layers L` 覆盖的前 `L` 个全局层保持现有 native full-GPU 路径，不进入 KT expert cache。
- `--kt-num-gpu-experts 32` 只作用于 `layer_idx >= L` 的后续 cache-managed MoE 层，为这些层各预分配 32 个固定物理 GPU expert slots。
- `--kt-prefill-stream-top-n 4` 只作用于上述 cache-managed MoE 层；每层每个 prefill chunk 选择路由计数最高的 4 个 distinct experts 作为 streaming hotset。它不是 Router 的 per-token top-k。
- 物理 slot 的数量、shape 和地址在模型加载后保持不变。
- `decayed-lfu` 提供跨 prefill 窗口的历史热度信号；当前窗口 Stream-TopN 以本次真实路由计数为主、历史热度为稳定 tie-break；首版不实现显式的“下一 Chunk 预测”模型。
- 每个 cache-managed MoE 层的所有逻辑专家都维护路由访问分数；非驻留专家也必须计数。
- 当前 batch 先按已发布 resident mapping 启动第一批 GPU GEMM；Stream-TopN 缺失 candidate 使用独立 streaming 路径参与当前 batch。
- Router 之后不能先串行等待 H2D 再启动计算；CPU export、H2D 和 repack 必须与当前层 resident GPU wave 和 CPU-only GEMM 重叠。
- 当前窗口最多考虑 `N_stream` 个热点专家；其中已经 resident 的专家不占搬运名额，也不向后补齐低排名专家。
- 首版 persistent admission 使用历史热度、滞回、最小驻留和内部搬运预算；它不改变当前窗口 Stream-TopN set。显式 expected-benefit 成本模型放入后续规划。
- 权重换入只在 prefill 流程中发生：优先参与产生该统计的当前 chunk，并在安全发布后继续服务后续 prefill chunk；进入 decode 前冻结为稳定 resident 集合。
- decode 不发起统计决策、权重导出、H2D、repack 或 slot 替换。
- 更新只搬运 `new - old` 的专家，不重写没有变化的 resident experts。
- 淘汰只表示复用固定物理 slot，不执行 `cudaFree`、重新分配或改变 tensor 地址。

该设计与现有 `--kt-enable-dynamic-expert-update` 不同。现有逻辑只在满足 threshold 的 layerwise/full-GPU prefill 后，按当前 batch 的 Top-N 整组更新；本文设计独立于 full-GPU prefill，可在 hybrid prefill 路径持续收集路由并增量维护 resident slots。

## 2. 背景与现状

### 2.1 当前 hybrid 路径

普通 hybrid MoE 的执行过程是：

```text
Router 生成 topk_ids / topk_weights
        |
        +-- resident GPU expert --> GPU MoE
        |
        +-- non-resident expert --> TP0 CPU MoE
                                      |
                                  output H2D
        |
CPU output + GPU output
```

CPU expert 与 GPU expert 尝试并行，但最终主路径必须等待 CPU 输出。单层延迟近似为：

```text
T_layer ~= max(T_gpu_resident, T_cpu_miss + T_callback_queue) + T_merge
```

因此缓存只有在降低 CPU miss 工作量并缩短暴露的 CPU wait 时才会降低 step latency。单纯提高命中率但没有把 CPU 分支压到 GPU 分支以下，不一定继续降低端到端时间。

### 2.2 当前动态更新策略

当前实现的动态更新具有以下特征：

1. 只有 `--kt-enable-dynamic-expert-update` 开启，并且本次 `num_tokens` 达到 `--kt-gpu-prefill-token-threshold`、backend 支持 full-GPU gate 时才触发；生产中通常表现为大 prefill，但当前 gate 并不直接以 forward phase 名称判定。
2. 对当前 prefill batch 的 `topk_ids` 临时计数。
3. 选择当前 batch 频率最高的 N 个专家。
4. 复制全部 N 个专家并整体重建映射。
5. 计数在本次更新后丢弃，不保留跨 batch 历史。
6. 普通 hybrid/decode miss 不会触发专家换入。

即使新旧 Top-N 完全一致，当前实现仍可能执行全部 resident 权重复制和后端 prepare。

### 2.3 当前 layerwise/full-GPU prefill

layerwise/full-GPU prefill 会依次构造完整 MoE 层的 GPU 权重镜像。对于一层 E 个专家：

- 已 resident 的专家从原 GPU 权重 D2D 复制到完整层临时 slot。
- 其他专家从 CPU packed 权重导出，经 pinned host staging 后 H2D。
- 完整层执行后，临时 slot 被后续层复用。

该路径本质上是权重 streaming，而不是长期 cache。它适合足够大的 batch，在 CPU 计算节省能够摊销完整层权重导出、H2D、repack 和 TP 同步时使用；不适合普通 decode。

## 3. 目标与非目标

### 3.1 目标

1. 保持现有分层边界：前 `--kt-num-gpu-layers` 层继续全 GPU；只为其后的 cache-managed MoE 层严格维护固定数量的 GPU resident slots。
2. 根据近期真实路由分布自动调整 resident expert 集合。
3. 只更新发生变化的 slots，避免整组 N 个专家重写。
4. 让最多 `N_stream` 个热点 miss 通过流式 H2D + 第二批 GPU GEMM 参与当前 prefill chunk。
5. 保持 CUDA Graph 所需的 tensor 地址与 shape 稳定。
6. 保证所有 TP ranks 的专家权重和 logical-to-slot 映射一致。
7. 加载失败时保留旧 resident 集合或安全回退 CPU，不能产生半提交状态。
8. 支持离线模拟、运行时观测和明确的性能回退开关。

### 3.2 非目标

第一阶段不包含：

- 动态增加或减少每层 GPU slot 数量。
- 每个 request 私有一套专家缓存。
- 从磁盘按 miss 读取专家权重。
- 等待全部 streaming candidates 搬完后再统一开始当前层 GEMM。
- 在 decode 路径发起任何专家权重搬运或 resident 更新。
- 同时支持所有 CPU 量化格式与所有 GPU MoE backend。
- 修改 Router 的数学输出或近似丢弃低权重专家。

## 4. 参数语义与兼容性

### 4.1 启动参数

```bash
--kt-num-gpu-experts 32 \
--kt-expert-placement-strategy decayed-lfu \
--kt-prefill-stream-top-n 4
```

其中：

```text
C        = --kt-num-gpu-experts
N_stream = --kt-prefill-stream-top-n
L_gpu    = --kt-num-gpu-layers or 0
```

`N_stream` 控制每层每个 prefill chunk 的 streaming hotset 大小，并因此限定本窗口最多选择和流式执行的 missing 热点专家总数；它不改变模型 Router 的 per-token top-k，也不受 staging 并发深度裁剪。

建议 help 文案：

```text
Maximum number of missing hot experts selected for current-chunk prefill
streaming in each cache-managed MoE layer. Experts are ranked by this chunk's
route counts. Resident experts remain in the Top-N set but are not transferred,
and lower-ranked experts are not backfilled. Fixed staging slots limit only
concurrency; they are reused until all selected candidates finish or fall back.
```

扩展现有 strategy 枚举：

| 策略 | 语义 |
|---|---|
| `uniform` | 现有静态 placement；每层使用固定初始专家 |
| `frequency` | 现有离线频率 placement；启动后静态 |
| `front-loading` | 现有静态前置分配策略 |
| `random` | 现有静态随机策略 |
| `decayed-lfu` | 启用本文的 prefill 学习、当前 chunk 有界流式换入和 decode 冻结消费 |

为保持现有用户行为，默认值仍为 `uniform`。只有显式选择 `decayed-lfu` 才创建动态 cache controller。

`decayed-lfu` 冷启动规则：

- 配置了可用的 `--init-expert-location` 时，按每层离线 frequency Top-C 初始化。
- 没有离线统计时，按 per-layer uniform 规则初始化。
- 初始 placement 完成后，只有 prefill 路由能够产生增量 promotion。

该参数的 help 文案应从“initial GPU expert placement strategy”扩展为“GPU expert placement and runtime scheduling strategy”。

### 4.2 用户参数边界与内部默认值

首版只新增一个可调运行时宽度参数 `--kt-prefill-stream-top-n`。衰减、过期、滞回、staging 深度、字节预算和 finalization 预算仍全部使用代码内部默认值，通过日志和 A/B 逐步校准。

推荐启动配置为：

```bash
--kt-expert-placement-strategy decayed-lfu \
--kt-prefill-stream-top-n 4
```

参数约束：

- CLI 字段默认 `None`；选择 `decayed-lfu` 且用户未显式配置时，解析为 `N_stream = min(4, C)`。
- 用户显式指定时必须满足 `0 <= N_stream <= C`；越界时启动失败，不静默 clamp。
- `N_stream == 0` 表示只保留统计和现有 resident mapping，不创建 stream/promotion loader task，可用于观测基线。
- 当前窗口不同 active experts 少于 `N_stream` 时，使用 `min(N_stream, unique_active_experts)`。
- `decayed-lfu` 要求 `C > 0`。
- 该参数只在 `--kt-expert-placement-strategy decayed-lfu` 下生效；其他 placement strategy 下显式配置时启动失败，避免无声忽略。
- 推荐从 `1～4` 起步；允许在 `N_stream <= C` 范围内显式实验更大值。双 staging 只限制同时在途的 candidates 为 2，不改变本窗口最多流式执行 `N_stream` 个 missing experts 的语义；更大的值可能形成更长的滚动队列并增加临界路径延迟。

建议的首版内部默认值，而非性能承诺：

```text
prefill update interval     = 1 chunk
decay                       = 0.5 per window
minimum residency           = 2 windows
idle expiry                 = 4 windows
hysteresis                  = 25%
stream selection            = current-window Stream-TopN
stream rank                 = token_count desc, reuse_signal desc, expert_id asc
default N_stream            = min(4, per-layer capacity)
selected candidates         = ordered non-resident filter of Stream-TopN, 0..N_stream
host staging slots          = 2 per compatible loader group
GPU raw/prepared staging    = 2 per compatible loader group
staging semantics           = concurrency depth only; roll over until selected candidates finish
max persistent replacements = actual successful stream loads, at most N_stream per layer/window
fallback deadline policy    = internal, derived from current CPU branch and measured token-bucket CPU cost
```

除 `N_stream` 外，这些值必须通过真实路由 trace 和目标硬件 A/B 调整；调整发生在代码默认配置中，不暴露额外 CLI。

### 4.3 与现有参数的关系

现有层级边界是硬性不变量，本文不得修改：

```text
global layer_idx < L_gpu
    -> native/full GPU routed experts
    -> 不创建 KT CPU wrapper
    -> 不创建 expert cache state
    -> 不应用 C 或 N_stream

global layer_idx >= L_gpu 且该层为 KT-managed MoE
    -> --kt-num-gpu-experts C 定义该层 resident capacity
    -> --kt-prefill-stream-top-n 定义该层当前 chunk hotset 上限
    -> decayed-lfu / CPU-GPU hybrid / streaming 只在这里运行
```

- `uniform/frequency/front-loading/random` 只决定静态 resident experts；`decayed-lfu` 同时决定冷启动 placement 和运行时 prefill 调度。
- cache 模式下，`--kt-num-gpu-experts C` 必须保证每个 `layer_idx >= L_gpu` 的 KT-managed MoE 层严格分配 C 个物理 slots。
- 当前 `frequency`、`front-loading` 和 `random` 都可能按跨层总预算分配，不能直接用于固定 per-layer capacity。
- `decayed-lfu` 的冷启动只允许 per-layer uniform 或 per-layer frequency；不能复用当前跨层总预算的 front-loading/random 结果。
- `--kt-num-gpu-layers` 覆盖的前置全 GPU 层不启用本缓存、不累计 cache policy 统计、不占用 cache promotion/stream budget，也不加载 KT CPU expert backing。不得把 `--kt-num-gpu-experts` 重新解释成包含这些前置层。
- 初始版本建议禁止同时使用 `--kt-gpu-experts-ratio` 与 `decayed-lfu`，避免“全局比例”与“每层固定容量”语义冲突。后续若支持 ratio，应明确转换为每层容量。
- `decayed-lfu` 不依赖 `--kt-gpu-prefill-token-threshold`。
- `--kt-enable-dynamic-expert-update` 保留为 legacy layerwise-prefill 整组更新开关；它与 `--kt-expert-placement-strategy decayed-lfu` 应互斥，避免两套控制器同时改写 slots。
- 首版禁止 `--kt-max-deferred-experts-per-token` 与 `decayed-lfu` 同时启用，除非 deferred CPU task 也接入同一 per-assignment ownership 和独立 buffer 协议。
- Expert LoRA 当前与动态 expert update 不兼容，新缓存首版继承该限制。

## 5. 术语与统计口径

### 5.1 Token-expert assignment

每个 token 被 Router 选中的每一个 top-k expert，计为一次 token-expert assignment。

例如：

```text
4096 tokens * top_k 8 = 32768 assignments
```

如果 expert 7 出现在 500 个 assignment 中，表示它需要处理 500 行 token activation。执行时通常会聚合成一个 `M=500` 的 grouped GEMM，而不是启动 500 次独立 kernel。

### 5.2 Route access 与 cache hit

必须分开统计：

- route access：Router 选中该专家，无论其在 CPU 还是 GPU。
- cache hit：Router 选中该专家，并且其 resident slot 已发布且 READY。
- streaming miss：Router 选中非 resident 专家，且进入当前窗口缺失 Stream-TopN；本次优先由流式换入后的第二批 GPU GEMM 计算。
- CPU-only miss：Router 选中非 resident 专家，但不属于 streaming candidates；本次直接由 CPU 计算。

历史热度统计必须使用所有专家的 route access。若只累计 cache hit，非 resident 热点永远无法积累晋升证据；但 route count 只是 Decayed-LFU 的输入，不能直接等同于最终 cache decision。

### 5.3 逻辑过期时间

过期使用 logical epoch/window，而不是 wall-clock 秒数：

- 服务空闲不会导致热点专家自动失效。
- 只有 prefill chunk 推进统计窗口和过期 epoch。
- decode 期间 epoch 冻结，服务空闲或持续 decode 都不会触发热点衰减和替换。
- 过期只表示“优先成为 victim”，不释放物理显存。

## 6. 总体架构

```text
Chunk N / Layer L Router
            |
            +-- route histogram / current Stream-TopN / Decayed-LFU
            |
            +-- resident active experts -----------------> GPU wave 1
            |
            +-- non-resident, non-candidate experts -----> CPU GEMM
            |
            +-- missing experts inside current Stream-TopN
                    |
                    +-- host export -> H2D -> repack
                    |       overlaps GPU wave 1 and CPU GEMM
                    |
                    +-- each ready expert / small ready group
                            |
                            +-----------------------------> GPU wave 2
            |
            +-- merge(GPU wave 1, GPU wave 2, CPU output)
            |
            +-- safe-boundary persistent slot publish
```

建议拆分为四个组件：

1. `KTExpertCacheState`：单层 prefill 统计、resident 集合和 slot 元数据。
2. `KTExpertCachePolicy`：纯策略逻辑，输入统计与当前 resident，输出 current stream set 和 persistent promotion plan。
3. `KTExpertCacheLoader`：执行单 expert 权重导出、传输、后端 prepare 和 TP 提交。
4. `KTExpertStreamDispatcher`：为当前 chunk 固化三路 assignment、驱动 ready queue、第二批 GPU wave 和 CPU fallback。

策略与权重加载必须解耦，以便离线模拟、单元测试和后端扩展。

禁止实现成以下串行临界路径：

```text
Router -> count -> decide -> H2D wait -> GEMM
```

正确时序不是“等待 H2D 后才开始计算”，而是 Router 完成分桶后同时启动三条流水：resident GPU wave、CPU-only GEMM 和 candidate loader。某个 candidate 一旦 READY，就立即进入当前 chunk 的第二批 GPU wave；不等待其余 candidates。首版不构建下一 Chunk 的显式预测器。

## 7. 数据结构

### 7.1 每层状态

这里的“每层”只指 `layer_idx >= L_gpu` 且实际创建 KT wrapper 的 cache-managed MoE 层。前置 full-GPU layers 不创建下列对象。

建议的数据结构：

```python
@dataclass
class KTExpertCacheState:
    layer_idx: int
    num_experts: int
    capacity: int
    stream_top_n: int

    # GPU-resident statistics; storage is preallocated.
    window_count: torch.Tensor       # int32 [E], GPU
    reuse_signal: torch.Tensor       # float32 [E], GPU
    last_access_epoch: torch.Tensor  # int32 [E], GPU or pinned CPU

    # Authoritative controller metadata.
    slot_to_expert: torch.Tensor      # int32 [C], CPU
    expert_to_slot: torch.Tensor      # int32 [E], CPU

    # Published mirrors used by current runtime.
    method_mask_cpu: torch.Tensor     # self.gpu_experts_mask
    method_mask_gpu: torch.Tensor     # self.gpu_experts_mask_cuda
    method_l2s_cpu: torch.Tensor      # self.logical_to_gpu_index
    method_l2s_gpu: torch.Tensor      # self.logical_to_gpu_index_cuda
    method_s2l_cpu: torch.Tensor      # self.gpu_index_to_logical
    wrapper_mask_pinned: torch.Tensor # wrapper.gpu_experts_mask; C++ holds pointer

    resident_since_epoch: torch.Tensor  # int32 [C], CPU
    slot_generation: torch.Tensor       # int64 [C], CPU
    slot_state: list                    # READY/IN_USE/EVICT_PENDING/INSTALLING_UNPUBLISHED

    # Current-prefill-window streaming state; never exposed to decode graphs.
    stream_generation: int
    stream_candidates: torch.Tensor     # int32 [<=N_stream], CPU/pinned snapshot
    stream_dispatch_state: list         # SELECTED/GPU_CLAIMED/GPU_DONE/CPU_CLAIMED/CPU_DONE
```

当前实现并不是只有一张 mask：

- `KTEPWrapperMethod.self.gpu_experts_mask` 是普通 CPU tensor。
- `self.gpu_experts_mask_cuda` 是 CUDA mirror。
- `KTMoEWrapper.wrapper.gpu_experts_mask` 是独立 pinned tensor，C++ 长期保存其裸指针。
- logical-to-slot 也同时存在 CPU/CUDA mirrors。

cache controller 的 `slot_to_expert/expert_to_slot` 应是唯一权威元数据；上述 runtime tensors 是在 safe boundary 统一发布的 mirrors。除非后续专门重构为共享 pinned storage，否则文档和代码都不能假设这些对象已经共享同一内存。

### 7.2 Stream plan 与 promotion plan

当前 chunk 的执行计划必须与长期 resident 提交计划分离：

```python
@dataclass(frozen=True)
class KTExpertStreamPlan:
    layer_idx: int
    current_chunk_id: int
    resident_generation: int
    stream_top_n: int
    candidate_expert_ids: tuple[int, ...]   # 0..N_stream
    candidate_assignment_indices: tuple     # per candidate
    candidate_topk_weights: tuple
    output_scatter_metadata: tuple
    candidate_token_counts: tuple[int, ...]
```

`KTExpertStreamPlan` 是当前调用的 immutable snapshot，负责 CPU transient exclusion、第二批 GPU 和 late CPU fallback。它不修改 persistent mapping。

```python
@dataclass(frozen=True)
class KTExpertPromotionPlan:
    layer_idx: int
    source_expert: int
    victim_expert: int
    victim_slot: int
    stream_generation: int
    cache_epoch: int
    slot_generation: int
    candidate_reuse_signal: float
    victim_reuse_signal: float
```

`KTExpertPromotionPlan` 只负责把已经成功准备的 candidate 持久安装到 resident slot。`cache_epoch`、`stream_generation` 和 `slot_generation` 用于丢弃过期计划。例如计划排队期间 victim 已变化，则 loader 结果不能再提交到原 slot；这不影响 candidate 已完成的当前 chunk 输出。

### 7.3 Resident slot 与 staging 状态机

CPU expert 与 GPU cache 必须是两套独立状态：

```text
(layer, expert)
    |
    +-- CPU backing resident
    |      backend-owned packed weight
    |
    +-- GPU resident
           slot_id / generation
```

adaptive 模式下 CPU backing 应始终存在，Cache Manager 只管理 GPU residency 和 slot，不把 CPU 权重建模成 cache slot。

Resident slot 状态：

```text
READY
  |
  +--> IN_USE
  |       |
  |       +--> READY
  |
  +--> EVICT_PENDING
          |
          +--> INSTALLING_UNPUBLISHED
                  |
                  +--> READY(new expert)
```

Pinned host staging 与 GPU staging 的生命周期不同，不能合并成一个模糊状态机。

Pinned host staging：

```text
FREE
  |
  +--> WRITING --> H2D_IN_FLIGHT --> TP_RELEASE_PENDING --> FREE
                                      |
                                      +--> POISONED/FAILED
```

GPU raw/prepared staging：

```text
FREE --> H2D --> REPACKING --> READY
                               |
                               +--> GPU_WAVE2_IN_USE --> CONSUMED
                               |                         |
                               |                         +--> INSTALL_D2D --> INSTALL_DONE --> FREE
                               |                         +--> FREE
                               |
                               +--> INSTALL_D2D --> INSTALL_DONE --> FREE
                               |    # 当前输出已由 CPU fallback 拥有
                               +--> FREE
                               +--> FAILED/CANCELLED
```

staging load state 与当前 chunk 的 execution ownership 必须分离。assignment ticket 使用“只执行一次”的 ownership 状态：

```text
SELECTED
   |
   +--> GPU_CLAIMED --> GPU_VALIDATING --> GPU_DONE
   |                         |
   |                         +--> GPU_FAILED_RECOVERABLE --> CPU_CLAIMED
   |
   +--> CPU_CLAIMED --> CPU_DONE
```

硬性规则：

- `IN_USE` resident slot 不允许 eviction 或覆盖。
- candidate 的 loading 发生在独立 host/GPU staging，不影响旧 resident。
- eviction/installation 必须等待 resident slot 的 last-use event。
- host staging 只有在所有 TP ranks 的 H2D event 和 release consensus 完成后才能复用。
- GPU staging 只有在 writer/DMA/repack、可能存在的 GPU wave 2 consumed event，以及把它作为 source 的 install D2D event 全部完成后才能复用。
- assignment 切到 `CPU_CLAIMED` 只禁止当前 chunk 接受 streamed GPU output，不自动取消已经在途的 loader，也不允许提前复用 staging；load 最终成功时仍可安装供后续 chunk 使用。
- 默认只有两个 GPU staging slots，但 staging depth 只是并发深度。`P` 中全部 `0..N_stream` 个 candidates 在主 CPU task 提交前固化为 GPU ownership；首批最多 2 个 tickets 进入 writer/H2D 流水，其余保持等待状态，并在 slot 完成 wave 2、install/release 和 TP fence 后按原顺序滚动激活。任一 candidate 可恢复失败时只对其自身及尚未成功的尾部执行 late CPU fallback，不得重新加入已经运行中的主 CPU task。
- `GPU_CLAIMED` 与 `CPU_CLAIMED` 必须是互斥状态，保证 candidate 的当前 chunk 输出不会被 CPU/GPU 重复计算并重复累加。
- GPU wave 2 必须先写 candidate-private output；只有 kernel/event 与 TP success consensus 完成后才 scatter/merge 到共享 MoE output。可恢复失败可丢弃私有 output 并切换 CPU；已经部分写入共享 output 或 CUDA context 损坏时必须 fail-stop。
- 当前 chunk 的 streaming mapping 是私有 dispatch snapshot；全局 resident mask/mapping 在 CPU task 仍可能读取时不得修改。

### 7.4 Slot footprint

“一个 slot 对应一个 expert”是逻辑概念。一个物理 expert slot 实际包含一组 tensor：

```text
w13 weight
w13 scale/metadata
w2 weight
w2 scale/metadata
backend prepared/repacked image
```

同一 MoE 层的 routed experts 通常具有相同 shape 和 dtype，因此 per-layer slot 应按该层完整 expert footprint 创建。跨层共享或 Global Spare 不能假设所有专家大小相同；只有 footprint、量化布局和 TP signature 兼容的 slots 才能互换。按“任意某个专家的实际大小”临时分配会产生装不下后续 expert 的风险。

## 8. 路由统计与热度信号

### 8.1 统计更新

只对 prefill/extend 行的逻辑 `topk_ids` 做直方图。decode 行不进入 policy 统计；下面代码只表达数学含义：

```python
record_prefill_counts(topk_ids, prefill_row_mask, window_count)
```

实现要求：

- 必须读取 mask/remap 前的逻辑 `topk_ids`，不能统计已经转换成物理 slot 或 `-1` 的 GPU dispatch IDs。
- 仅统计 scheduler 标记为 prefill/extend 的有效行；mixed batch 中的 decode 行、越界 ID 和无效 ID全部排除。
- 首版采用全局 decode freeze：若当前 mixed batch 包含正在使用冻结 generation 的 decode 行，prefill/extend 行可以继续累计统计，但不得生成 stream loader task；其 non-resident assignments 全部走 CPU。scheduler 若要启用当前 chunk streaming，必须提供没有 active decode generation 的 prefill-only epoch。
- decode CUDA Graph、padding 行和 capture/warmup 不执行 policy 统计。
- 统计留在 GPU。
- 使用预分配 buffer，不在 per-layer prefill 热路径动态分配。
- 实际实现应使用固定 shape 的 histogram/scatter kernel，并显式接收 prefill-row mask；布尔索引和 `ones_like(valid_ids)` 会产生动态长度临时 tensor，不应作为高频实现。
- 不在每层调用中执行 `.cpu()`、`.item()` 或 `torch.cuda.synchronize()`。
- 如果 dispatch/backend 已提供 `tokens_per_expert`，优先复用该结果。
- TP 模式下由 TP0 负责生成最终 promotion plan；其他 ranks 不需要把完整统计拉回 host。

### 8.2 Decayed-LFU 历史热度

为避免不同 prefill chunk 大小使大 chunk 永久压制其他请求的热点，窗口观测值建议先归一化。这里计算的是 `reuse_signal`，表示当前及历史 prefill 热度，不是显式的下一 Chunk 预测结果：

```text
observed_freq[e] = window_count[e] / max(sum(window_count), 1)
evidence_weight = min(sum(window_count) / reference_assignments, 1)

reuse_signal[e] =
    decay * reuse_signal[e]
    + (1 - decay) * evidence_weight * observed_freq[e]
```

`reference_assignments` 用来避免只有极少 assignment 的窗口与完整 4K prefill 获得完全相同的历史更新权重。首版还应保留绝对 `window_count` 作为 persistent admission 证据；这些门槛只决定 candidate 是否长期安装到 resident slot，不改变 8.3 节由本窗口真实 Stream-TopN 得到的 streaming set。

窗口结束后：

```text
last_access_epoch[e] = current_epoch  if window_count[e] > 0
window_count.zero_()
```

### 8.3 当前窗口 Stream-TopN 与三路集合

对当前层、当前 prefill chunk 定义：

```text
A = 当前窗口被 Router 激活的 logical experts
G = 进入本层时已发布的 resident expert set，容量 C=32

H = stable_topk(
        A,
        k=min(N_stream, |A|),
        key=(window_count desc, reuse_signal desc, logical_expert_id asc)
    )

P = ordered_filter(H, e not in G)
                          # 保持 H 排名，选中的 0..N_stream 个 candidates
R = A intersect G         # 当前 chunk 第一批 GPU experts
C_cpu = A - G - P         # 当前 chunk 直接交给 CPU 的 experts
```

这里的首要目标是减少产生该统计的当前 chunk 的 CPU expert GEMM，因此 `window_count` 是 Stream-TopN 的主排序键；`reuse_signal` 用于稳定相同或接近计数的排序，并用于长期 resident retention/victim 决策。该 Stream-TopN 不是下一 Chunk 预测。

硬性规则：

- `H` 只取当前窗口真实激活的前 `N_stream` 个专家。
- `H` 中已经 resident 的专家继续进入第一批 GPU wave，不占候选名额。
- 选中的 streaming candidate 数量严格为 `|P|`，范围 `0..N_stream`；实际成功 H2D/GPU wave 2 数可能因失败、deadline 或 backpressure 更少，其余必须 late CPU fallback。
- 不从第 `N_stream + 1` 名及以后补齐。例如 `N_stream=4` 且 hotset 中已有 3 个 resident，只选择剩余 1 个。
- `R`、`P`、`C_cpu` 三个 expert 集合互斥；展开到 token-expert assignments 后，其并集必须等于原始 Router 输出。
- Stream-TopN 选择应使用固定 shape 的 device top-k/reduction；不允许为了少量 expert IDs 对全部路由结果做 host 全排序或全量同步。

### 8.4 Prefill 窗口与 decode 冻结

- Prefill：每个实际 scheduler chunk 作为一个统计窗口。
- 在 prefill-only epoch 中，Router 产生本窗口计数后，立即生成 `H/P/R/C_cpu` 并启动当前 chunk 三路执行。
- resident GPU wave 与 CPU-only GEMM 不等待 candidate H2D；但是当前层最终输出必须等待每个 `P` assignment 由 streamed GPU 或 late CPU fallback 恰好完成一次。
- 成功 stream 的 candidate 可在安全边界提交为 persistent resident，继续服务后续 prefill chunk。
- Decode：不形成统计窗口、不衰减分数、不生成 promotion，也不发起任何 writer/H2D/repack/commit。
- 首版采用全局 prefill-only 更新窗口：一旦进入 decode generation，mapping generation 冻结，直到该 generation 结束。
- mixed prefill+decode batch 不启动 writer/H2D/repack；其 prefill miss 不进入 `P` 执行路径，而是按普通 CPU-only miss 处理。

这是首版明确接受的保守限制，不是最终产品语义。在 continuous batching 中，active decode 可能长期存在，使 mapping 更新永久饥饿；Phase 5 必须通过 generation-aware prepare/publish、同层 shadow slots 或有界 maintenance epoch 解决。若 Phase 5 尚未实现，系统必须通过指标明确暴露 starvation，不能宣称 mixed workload 下动态 cache 正常自适应。

如果后续 trace 证明 prefill 热点与 decode 相关性较弱，优先调整代码内部的衰减和 retention 规则；本文首版仍明确禁止 decode 搬运，不引入 decode-side promotion 或显式下一 Chunk 预测。

窗口 rollover 和 `window_count.zero_()` 必须等待本窗口 Stream-TopN snapshot 以及最后一个 prefill 统计 kernel 的 event，不能与后续 prefill 统计并发清零。

### 8.5 后续规划：预测与成本感知 Expected Benefit

本节不属于首版实现。首版使用 Decayed-LFU 历史热度、滞回、最小驻留和内部搬运预算完成 admission。

后续版本可增加显式下一 Chunk/后续 decode workload 预测，并将最终目标升级为减少暴露在端到端临界路径上的 CPU expert GEMM。

对 candidate `c`、victim `v` 和预测时间范围 `H`：

```text
candidate_saved(c, H) =
    CPU_GEMM_cost(
        layer,
        c,
        predicted_future_tokens(c, H)
    )

victim_loss(v, H) =
    CPU_GEMM_cost(
        layer,
        v,
        predicted_future_tokens(v, H)
    )

promotion_cost(c) =
    CPU_export
    + pinned_staging
    + H2D
    + GPU_repack_or_swizzle
    + TP_consensus
    + install_D2D
    + resource_contention

net_expected_benefit(c, v, H) =
    candidate_saved
    - victim_loss
    - promotion_cost
```

更准确的 latency 收益应按临界路径计算：

```text
critical_path_gain =
    max(T_gpu_before, T_cpu_before)
    - max(T_gpu_after, T_cpu_after)
```

当 CPU 分支已经低于 GPU 分支时，继续提高 hit rate 可能减少 CPU 工作和功耗，但不一定继续降低 step latency。

`CPU_GEMM_cost` 不能简单设为固定“每 token 成本”。同一专家的多个 token 会聚合为 grouped GEMM，因此应按 layer、expert shape、量化后端和 token-count bucket 进行离线或在线测量。

在没有 cost table 的早期实验中，可以用：

```text
predicted_future_tokens * measured_per_token_CPU_cost
```

作为一阶近似，但不能将其视为最终模型。

首版只记录 promotion 全链路成本，不据此在线预测下一 Chunk。相关数据用于后续 cost-aware admission 的离线验证与实现。

## 9. Streaming candidate、victim、过期与滞回

### 9.1 Streaming candidate 选择

当前 chunk 的 streaming candidates 只能来自 8.3 节定义的 `P = ordered_filter(H, e not in G)`，不能再从所有 non-resident experts 中继续扫描。

```text
stream_candidates = [e for e in current_stream_hotset if e not in resident_snapshot]
```

以默认 `N_stream=4` 为例：

- Stream-TopN 全部 resident：选择 0 个 candidate，不选择第 5～8 名。
- hotset 中 3 个 resident：只选择 1 个 candidate。
- hotset 中 2 个 resident：只选择 2 个 candidates。
- hotset 全部 miss：最多选择 4 个 candidates。

相同计数按 `reuse_signal`、logical expert ID 做稳定 tie-break，保证所有 TP ranks 得到一致顺序。

### 9.2 Victim 选择

persistent install 的 victim 仅从 READY、满足最小驻留时间、并且本轮完全未激活的 resident experts 中选择。本轮 Router 产生过任意 assignment 的 resident expert 一律受保护，即使它不在 Stream-TopN 中也不能被替换：

```text
A = {e | window_count[e] > 0}
victim_pool = G - A
```

优先级：

1. 当前 chunk 未激活、已超过 `idle_expire_windows` 且满足最小驻留时间的 resident。
2. 当前 chunk 未激活、`reuse_signal` 最低的 resident。
3. 分数相同时，选择 `last_access_epoch` 最旧，再按 slot ID 确定性排序。

“本轮未激活”是 admission 的硬约束，而不是等待 wave 1 完成后即可放宽的时序条件。实现仍保留 victim-safe CUDA event 作为 D2D install 的防御性屏障，但该 event 不能授权淘汰本轮激活专家。若不存在未激活的安全 victim，candidate 只服务当前 chunk 的 wave 2，随后释放 staging，不改变 persistent resident 集合。刚换入的专家在 `min_residency` 窗口内受保护，避免立即被换出。

### 9.3 当前计算与 persistent admission 分离

streaming candidate 是否参与当前 chunk，与它是否最终成为 persistent resident 是两个决策：

- 当前计算：`P` 中 candidate 只要 loader/backend 可用，就从普通 CPU task 排除，并尝试从 staging 完成当前 chunk 的第二批 GPU GEMM。
- persistent install：只有存在安全 victim，且通过 Decayed-LFU retention、最小驻留、滞回和内部带宽/backpressure 检查时，才在安全边界替换 resident slot。
- 如果没有可用 victim，candidate 仍可从 staging 完成当前 chunk GPU 计算，随后释放 staging，不改变长期 resident 集合。
- 如果 candidate 未在内部 deadline 前进入 `GPU_CLAIMED`，则原子切换为 `CPU_CLAIMED`，提交 late CPU fallback；已经在途的权重可完成并用于后续 persistent install，但不得再向当前 chunk 累加 GPU 输出。deadline 由当前 CPU-only 分支预计完成时间和该 candidate 的实测 token-bucket CPU cost 推导，是代码内部策略，不新增 CLI，也不是下一 Chunk 预测。

后续 cost-aware 版本可对 persistent install 增加：

```text
net_expected_benefit(candidate, victim, horizon)
    > admission_margin_ms
```

首版不实现该在线预测与成本决策，只记录所需指标。

### 9.4 搬运与替换预算

首版每层每窗口的固定语义为：

```text
selected_for_stream = ordered_filter(current_stream_hotset, not resident)
0 <= len(selected_for_stream) <= N_stream
```

`N_stream` 是候选集合硬上限，不是“必须搬满 N 个”的配额。实际成功的 H2D、当前 chunk GPU compute 和 persistent replacements 还受以下内部安全约束：

- 兼容 staging slot 数量和 loader backpressure；slot 数量只决定同时在途数，等待中的 selected candidates 继续保持 GPU ownership。
- 全局在途 writer/H2D/repack 数量。
- 单位 step 的传输字节与 CPU/PCIe 带宽保护。
- victim last-use、TP consensus、backend capability 和 failure fallback。

这些约束不新增其他 CLI；selected candidate 若因 deadline、故障或全局预算无法完成 streaming，必须明确走 late CPU fallback，不能静默漏算，不能偷偷回填到已经启动的主 CPU task，也不能向第 `N_stream + 1` 名以后补位。

### 9.5 全局性与禁止整组覆盖

`decayed-lfu` 的“全局性”是指统计跨 batch、跨请求持续保留，并由整个 serving 实例共享；不是每个 batch 独立生成一套 Top-C。不同 cache-managed 层的专家参数彼此独立，因此热度状态按层维护，但这些层共享全局 stream/promotion 在途数量和带宽预算。前置 full-GPU layers 不进入该统计域。

必须满足以下硬性不变量：

- 不把当前 batch Top-C 直接作为新的完整 resident 集合。
- 不因一次窗口的热点变化同时替换全部 C 个 slots。
- 未被 replacement plan 选中的 resident expert 保持原物理 slot 不变。
- 生产语义每层每窗口最多只处理当前 Stream-TopN 中缺失的 `0..N_stream` 个 experts。
- Stream-TopN 中已经 resident 的数量会直接减少本窗口 candidate 数，不向后补位。
- 一个超大 prefill 窗口不能让 persistent replacement 绕过 hysteresis、最小证据、最小驻留和全局 swap budget；当前 chunk streaming 仍按 `N_stream` 硬上限执行。

实际 candidate 数量为：

```text
P = ordered_filter(current_stream_hotset, not resident)
0 <= |P| <= N_stream
```

例如 `C=32, N_stream=4`，当前 resident 与本窗口 hotset 完全不相交时，最多处理 4 个；若 hotset 中已有 3 个 resident，则只处理 1 个。其余 28 个 resident slots 均保持不变，绝不会因单窗口热点变化重写全部 32 个。

### 9.6 Per-layer minimum 与 Global Spare

“每层 minimum + global spare”可以提高模型整体 cache utilization，但不属于首版固定 per-layer tensor 可以直接支持的能力。

若未来实现动态跨层借 slot，至少需要：

```text
min_slots[layer]
max_slots[layer]
borrow_limit[layer]
borrow_cooldown[layer]
global_spare_bytes
global_transfer_budget
```

例如：

```text
min_slots[layer] = 4
max_slots[layer] = 24
```

任何一层都不能无限借用 global spare。借用决策必须依据该层新增 slot 的边际 expected benefit，而不是只看该层总 token count。

当前 GPU MoE 权重按层创建固定 shape tensor，因此物理 slot 不能直接从 Layer 3 移给 Layer 20。要支持 Global Spare，必须选择以下架构之一：

1. 每层按 `max_slots` 预分配；实现简单，但显存已经全部占用，失去 spare 的主要意义。
2. 建立全局 GPU weight arena，通过 offset/indirection 动态绑定层；需要修改 backend、mapping 和 CUDA Graph 合约。
3. 按 `(dtype, quant/backend, expert shape, TP signature)` 建兼容 slot pool；不同兼容类之间不能互借。

因此实施顺序上，Global Spare 放在当前 chunk 异步流式换入和成本模型之后。第一版继续使用固定 per-layer capacity，只共享全局 promotion 数量、传输字节和 CPU/PCIe 带宽预算。

## 10. Runtime 执行时序

### 10.1 启动阶段

1. 读取 `L_gpu = --kt-num-gpu-layers or 0`；所有 `layer_idx < L_gpu` 的层保持 native/full-GPU，直接跳过 KT cache 初始化。
2. 只对 `layer_idx >= L_gpu` 的 KT-managed MoE 层，根据 `--kt-num-gpu-experts C` 创建 C 个固定 GPU slots。
3. 使用 `--kt-expert-placement-strategy` 选择这些 cache-managed 层的初始 resident experts。
4. cache 模式下 frequency placement 按 cache-managed layer 独立选择 Top-C。
5. 仅为 cache-managed 层初始化固定地址的 mask、logical-to-slot、slot-to-logical 和 cache state tensors。
6. 这些层的 CPU backend 必须 pack 所有逻辑专家，包括启动时已 resident GPU 的专家，确保未来淘汰后可由 CPU 正确计算；前置 full-GPU layers 不因此加载 KT CPU backing。
7. 启动前估算 cache-managed layers 的 all-expert CPU packed storage、pinned host staging 和 GPU staging 内存预算；不足时 fail fast，不能加载一半后静默降级。
8. 按兼容签名默认分别分配每组 2 个 pinned host slots 和 2 个 GPU raw/prepared slots，以支持“一边算上一位、一边搬下一位”，而不是为每层分配额外 staging。

staging 不能无条件跨所有层复用。pool key 至少包含：

```text
(device, dtype, quant/backend, expert shape, TP signature)
```

同一模型各层完全同构时可以自然退化成全局双 staging slots。

### 10.2 当前 prefill batch 计算

以下三路流程只在没有 active decode generation 的 prefill-only epoch 启用。mixed prefill+decode batch 仅记录 prefill 统计，禁止 loader，并将全部 non-resident prefill assignments 交给 CPU。

```text
进入 L 层
  |
Router 生成 prefill topk_ids
  |
在 GPU 生成 window_count，并确定 current Stream-TopN
  |
冻结本次调用使用的 resident snapshot G
  |
拆分互斥 assignments
  |
  +-- R = active resident --------------------> GPU wave 1
  |
  +-- C_cpu = non-resident experts outside P ------> CPU GEMM
  |
  +-- P = Stream-TopN 中全部 missing experts，最多 N_stream
          |
          +-- 双 staging 滚动执行 host export / H2D / repack
          |      与 GPU wave 1、CPU GEMM 重叠
          |
          +-- 每个 candidate READY 后 --------> GPU wave 2
  |
merge(GPU wave 1, GPU wave 2, CPU output)
```

已发布 mapping 只决定第一批 resident GPU wave。当前 batch 可以直接读取达到 `READY` 的 staging candidate，但绝不读取仍处于 HOST_PACKING/H2D/REPACKING 的部分权重，也不要求 candidate 先发布为 persistent resident。

当前调用必须生成 immutable dispatch snapshot：

```text
resident_assignment_ids
stream_assignment_ids       # P，对普通 CPU task 临时排除
cpu_only_assignment_ids
stream topk weights / scatter metadata
```

不能提前修改 persistent `gpu_experts_mask` 来排除 `P`，因为 C++ CPU task 可能长期持有该 pinned mask 的裸指针。必须增加 per-call `stream_exclusion_mask`，或直接给 CPU backend 提交已经过滤好的 `cpu_only_assignment_ids`。

三条流水的重叠范围是同一层：

- GPU wave 1 立即计算所有当前 active resident experts，包括不在 Stream-TopN 的 resident experts。
- CPU 在首批最多两个 writer 激活后立即计算所有 `C_cpu` experts；`P` 中全部 candidates 已从该主 CPU task 排除。首批和后续滚动 host export 均在独立 writer FIFO/WorkerPool 上执行，不排在主 CPU task 后面；`N_stream > 2` 的剩余暴露时间主要来自 writer 带宽、逐 candidate TP 控制和 GPU wave 2，仍须实机测量。
- loader 使用两个 staging slots 滚动准备全部 `P`；candidate 1 ready 后即可计算 candidate 1，slot 安全释放后立即用于后续 candidate，不等待全部 candidates 一次性完成。
- 当前层三路输出合并完成前不能进入下一 transformer layer；只有 persistent install/publish 工作可能在安全协议允许时与后续层重叠。

单层延迟近似为：

```text
T_layer ~= max(
    T_gpu_wave1,
    T_cpu_only,
    T_stream_pipeline_completion
) + T_merge
```

`T_stream_pipeline_completion` 是最后一个 candidate 经 host export、H2D、repack 和 GPU wave 2 完成的时间，包含 staging 复用造成的排队。异步流式换入仍可能落在当前 chunk 的临界路径，但它与已有 GPU/CPU 计算重叠，暴露时间小于先搬完再统一计算的串行路径。

例：`N_stream=4`，L1 当前激活 200 个不同专家，resident capacity 为 32，且 32 个 resident 均被本窗口激活；当前 Stream-TopN 中已有 3 个 resident。则：

```text
GPU wave 1 = 32 experts
stream P   = 1 expert
CPU-only   = 167 experts
```

只选择并尝试搬运这 1 个缺失 expert，不选择第 5～7 名补成 4 个。

MXFP4 CPU writer 使用独立、受限的 FIFO/WorkerPool，不能回退到主 CPUInfer 队列，否则 candidate 3 及以后会排在长 CPU GEMM 后形成串行尾部。辅助池默认每个 CPU TP/NUMA 仅 1 个 worker，并使用独立 core offset；若目标机器没有对应空闲核心，绑核可能降级为未绑定线程，必须通过日志和 trace 观测 CPU/DRAM 争抢。

### 10.3 生成 stream plan 与 promotion plan

Router histogram 完成后立即生成两类计划，不能等当前层计算结束后才选择 streaming candidates。

`KTExpertStreamPlan` 服务当前 chunk：

1. 固化 current chunk ID、resident generation、`N_stream` 和 current Stream-TopN。
2. 按 hotset 原顺序过滤 resident，得到 `P`，数量 `0..N_stream`。
3. 保存每个 candidate 的 assignment indices、top-k weights、output scatter metadata 和 fallback 所需 activation view。
4. 生成确定性 candidate 顺序并广播给各 TP ranks。
5. 立即启动 GPU wave 1、CPU-only task 和 candidate loader。

`KTExpertPromotionPlan` 服务 persistent cache：

1. 更新衰减分数。
2. 只为已经进入 `P` 的 candidates 选择 0～`|P|` 个 eligible victims。
3. 应用过期、最小驻留、滞回和内部传输/backpressure 保护。
4. TP0 生成固定大小的 promotion plan，并与 stream generation 绑定。

首版不估算下一 Chunk 的 token workload，也不在线计算 net expected benefit。

### 10.4 异步单 expert 权重准备

steady-state miss 不是磁盘 miss。权重来源是启动时已经加载到 backend-owned packed expert storage 的专家权重；多数 AMX/AVX 实现内部使用 BufferB。

建议流程：

```text
CPU packed expert
      |
single-expert writer
      |
double-buffered pinned host staging
      |
each TP rank H2D its own shard
      |
double-buffered GPU raw staging
      |
backend-specific single-slot repack/swizzle
      |
GPU prepared staging READY
      |
current-chunk GPU wave 2
      |
EXEC_DONE
      |
optional persistent install or FREE
```

V1.1 双 staging 是容量为 2 的滚动流水，承载当前窗口全部 `0..N_stream` 个 selected candidates：

```text
staging 0: candidate 1 GPU wave 2
staging 1: candidate 2 H2D/repack
candidate 1 consumed/install/release -> staging 0 复用给 candidate 3
candidate 2 consumed/install/release -> staging 1 复用给 candidate 4
依次滚动，直到 P 中全部 candidates GPU_DONE 或进入 late CPU fallback
```

ready queue 的规则是“有一个可算一个；若多个 candidate 已同时 ready，则合成一个小 grouped GEMM”。`ready_group_size` 不得超过当前 staging pool depth；默认双 staging 时单个 ready group 为 1～2 个，但一个窗口可以通过复用 slots 连续执行多个 groups，累计最多处理 `N_stream` 个 missing candidates。staging depth 不得用于缩小 CPU/GPU ownership 集合。

host 与 GPU staging 分别复用：host slot 在所有 TP ranks 完成该 shard H2D 并 release consensus 后即可复用；GPU raw/prepared slot 必须继续保留到最后一个读取权重的 wave 2 kernel 结束，并在需要 persistent install 时继续保留到 install D2D 完成。

H2D 与计算重叠必须落实为不同 CUDA stream 加 event 依赖，而不是把同步 copy 放进另一个 Python 线程：

```text
time -------------------------------------------------------->

CPU worker:    CPU-only GEMM ================================
host loader:   pack E1 ---- pack E2 ---- pack E3 ---- pack E4
copy stream:          H2D E1 ---- H2D E2 ---- H2D E3 ---- H2D E4
prepare stream:          repack E1 - repack E2 - repack E3 - repack E4
compute stream: GPU wave 1 ===== GPU(E1) -- GPU(E2/E3) -- GPU(E4)
```

- copy stream 对每个 candidate 记录 H2D event。
- prepare stream 只等待对应 H2D event，并记录 ready event。
- compute stream 的 GPU wave 2 只等待对应 ready event；不调用 device-wide synchronize。
- 如果 GPU wave 1 已占满计算资源，wave 2 可能排队，但 H2D 仍可利用 copy engine 与其重叠；实际重叠比例必须由 profiler 验证，不能仅凭异步 API 名称推断。

当前 `submit_write_weight_scale_to_buffer()` 已提供单 expert writer primitive，但现有 `sync_write_weight_scale_to_buffer()` 会调用全局 `CPUInfer.sync()`。新 cache loader 不能在热路径复用这种全队列 drain，必须增加：

- task-specific completion handle；或
- 独立/低优先级 writer queue；或
- 只在已知 CPU forward 队列为空的安全边界同步执行。

最终目标是让权重准备与当前 resident GPU/CPU-only critical work 重叠，同时不与 CPU MoE 无节制争抢同一 WorkerPool 和 DRAM 带宽。生产实现需要 task-specific completion 和专用/受限 writer resources；仅把同步调用放入另一个 Python thread 不等于真正异步。

### 10.5 当前 chunk GPU wave 2 与 persistent 提交

推荐先加载到额外的单 expert GPU staging slot，并直接从 staging 执行当前 chunk 的 GPU wave 2，而不是在 candidate ready 前覆盖 victim。GPU backend 必须提供等价于以下能力的接口：

```python
apply_streamed_experts(
    prepared_staging_weights,
    logical_expert_ids,
    candidate_assignments,
    topk_weights,
) -> candidate_private_output
```

该接口不能简单把 arbitrary logical IDs 交给现有 resident fused MoE kernel。它必须完成：

1. compact 每个 candidate 对应的 token/assignment rows。
2. 把 logical expert ID remap 为 staging-local `0..M-1`。
3. 对每个 candidate 按 `topk=1` 执行 expert GEMM，同时保留 Router 原始 top-k weight。
4. 返回 candidate-private output；TP success consensus 后再按原 token/assignment index scatter-add。
5. 保证 routed scaling、top-k weight 和输出累加各执行一次。
6. 在最后一个读取 staging 权重的 kernel 之后记录 consumed event。

优点：

- victim 在 candidate 准备期间仍可继续服务。
- H2D、repack 或 TP 失败时，旧 resident 权重完全不受影响。
- 当前 chunk 的第二批 GPU 不等待全局 mapping publish。
- persistent commit 只需要在安全边界执行受控 D2D 和映射切换。

当前 chunk 执行步骤：

1. 主 CPU task 只接收 `C_cpu`，明确排除 `P`。
2. 各 TP rank 对某 candidate 完成本地 shard H2D/repack 后进行 per-candidate ready consensus。
3. candidate ticket 从 `SELECTED` 原子切换为 `GPU_CLAIMED`；如果已经是 `CPU_CLAIMED`，不得再提交当前 chunk GPU 输出。
4. GPU wave 2 只等待该 candidate/ready group 自己的 ready event，不等待其他 candidates，也不等待 resident publish event。
5. GPU wave 2 写入 candidate-private output。kernel event 完成且所有 TP ranks success consensus 后，ticket 才进入 `GPU_DONE`，随后一次性 scatter/merge；staging 在 consumed event 完成前不得复用。
6. 若 writer/H2D/repack/ready consensus 失败，或内部 deadline 到期，则 ticket 从 `SELECTED` 原子切换为 `CPU_CLAIMED`，使用保留的 assignment metadata 提交 late CPU fallback。
7. 若 GPU wave 2 在共享 output 尚未修改前返回可恢复失败，丢弃 private output，经 TP consensus 后切换到 `CPU_CLAIMED`。若发生异步 CUDA fault、共享 output 可能已被部分写入或 CUDA context 状态不明，必须 fail-stop，不能继续 CPU fallback。
8. 最终输出严格为：

```text
output = resident_gpu_wave1_output
       + streamed_gpu_wave2_output
       + cpu_only_and_late_fallback_output
```

三个 assignment 集合必须互斥且并集等于 Router assignments；gate/top-k weight、routed scaling 和输出 scatter 均只能应用一次。

TP0 的主 CPU-only task 和所有 late fallback tickets 都必须完成后，才形成完整的本层本地 output；随后与各 rank 的 resident/streamed local output 合并，并只执行一次该层既有的最终 TP collective。

persistent 提交步骤：

1. 获取该层 exclusive/quiescent lease：当前 resident GPU wave 已提交 last-use event，且旧 mapping generation 下不允许任何新 batch 再进入该层。
2. 获取 lease 后再次检查 plan 的 `cache_epoch`、`stream_generation` 和 `slot_generation` 仍然有效。
3. commit stream 等待 victim slot 的 GPU wave 1 last-use event。若当前输出由 GPU 完成，还必须等待 candidate consumed event；若 ticket 已由 CPU fallback 完成，则只等待 staging load 的 READY event，并确认该 candidate 不会再启动 GPU wave 2。
4. 若 candidate 没有 eligible victim，当前 chunk 计算完成后直接释放 staging，不改变 persistent cache。
5. 若存在 victim，将 staging 中的 raw 与 prepared 权重 D2D 覆盖 victim 物理 slot，并更新后端 raw canonical snapshot；MXFP4 还必须同步更新 `_kt_mxfp4_raw_weights` 对应 slot。此时 slot 处于 `INSTALLING_UNPUBLISHED`：名称表示“尚未发布”，并非物理隔离；旧 victim 内容已经被破坏，任何新 consumer 都不得使用旧 mapping。
6. GPU staging 必须等 raw/prepared/canonical snapshot 的最后一个 install D2D event 完成后才能复用。记录每个 rank 的 install event，control stream 等待后进行 install consensus；覆盖开始后的任何部分失败均 fail-stop，除非另存 victim 完整备份。
7. 在当前 CPU task 不再读取旧 pinned mask、且没有活跃 decode generation 的 publish boundary，各 rank 的 publish stream 必须先等待 install event，再原地更新 CUDA mirrors：
   - `self.gpu_experts_mask_cuda`
   - `self.logical_to_gpu_index_cuda`
8. 记录 publish event，并在 event 完成后进行 device publish consensus。
9. 每个 TP rank 在下一 CPU task 提交前更新自己的 host mirrors：
   - `self.gpu_experts_mask`
   - `self.logical_to_gpu_index`
   - `self.gpu_index_to_logical`
10. TP0 额外使用原址 `copy_()` 更新 C++ 持有裸指针的 `wrapper.gpu_experts_mask` pinned tensor。
11. 所有 ranks 完成 host mirror 更新后进行最终 publish consensus；该阶段任一异常按不可部分提交处理并 fail-stop。
12. 更新 controller 的 `slot_to_expert`、`expert_to_slot`、`resident_since_epoch` 和 generation。
13. 下一次 main stream/CUDA Graph replay 必须等待 publish event。

映射更新必须使用 `copy_()` 等原址操作，不能替换 tensor 对象。

提交阶段必须被视为一个不可部分失败的事务。首版如果某个 rank 在 candidate D2D 覆盖 victim 之后发生 CUDA copy/stream 错误，或在多张 mapping tensor 更新过程中发生异常，应 fail fast 终止当前 serving process，不能继续使用可能已经跨 rank 分裂的 generation。若要把该类错误也做成可恢复回滚，需要额外保存 victim prepared 权重备份，并不属于首版范围。

### 10.6 Prefill-to-decode finalization

candidate 在 commit 前已经可以作为当前 chunk 的 streaming GPU expert 执行；只有 persistent commit 完成后，它才成为后续调用可见的 resident cache hit。

在进入 decode 前执行一次 finalization barrier：

1. 停止创建新的 promotion。
2. 当前 prefill chunk 的所有 stream tickets 必须达到 `GPU_DONE` 或 `CPU_DONE`，不能遗漏 assignment。
3. 取消尚未开始的低优先级 persistent plan。
4. 已进入 host pack/H2D/repack 的任务必须完成或被底层确认取消，不能带着仍在写 staging 的任务进入 decode。
5. 对已经安全安装的 promotion 执行 safe-boundary publish；无法在内部 finalization budget 内完成的未安装计划丢弃，不阻塞进入 decode。
6. prefill 结束后 resident mapping generation 冻结；该 decode generation 全程只读取，不发起 loader 工作，也不接受其他 prefill 的 mapping publish。

为控制 TTFT，prefill 期间的全局在途 promotion 数必须很小。若需要超时策略，只能丢弃“尚未开始”的计划；已经在途且底层不可取消的 DMA/repack 必须 drain 后才能正式进入 decode。

## 11. Prefill、decode 与 full-layer streaming 的关系

### 11.1 Prefill

大 prefill 中可能有大量不同专家被激活。本文不会把全部 miss 都搬到 GPU，而是只流式处理当前窗口 Stream-TopN 中缺失的 `0..N_stream` 个专家：

- 当前 chunk 的第一批 GPU wave 使用进入本层时的 resident snapshot。
- Stream-TopN 中已经 resident 的专家仍在第一批 GPU wave，不占候选名额。
- Stream-TopN 中缺失的 experts 从普通 CPU task 排除，边搬运边进入当前 chunk 的第二批 GPU wave。
- 其余 non-resident experts 立即由 CPU 计算。
- candidate 1 ready 即可计算 candidate 1，不等待其余 candidates 全部 ready。
- current chunk 计算完成后，成功 candidate 才在 safe boundary 尝试提交为 persistent resident，并在 finalization 后成为 decode 的固定 resident 集合。

这是一种“参数化有界 streaming”：即使 L 层激活 200 个不同专家，也只处理 Stream-TopN 中缺失的 candidates，GPU resident wave 计算已存在热点，CPU 计算其余专家。若 batch 大到完整 layerwise/full-GPU streaming 明确更快，可继续使用该路径；完整层路径会搬运几乎全部 active/all experts，和本文最多 `N_stream` 个 candidate 的 persistent cache 是两个不同执行策略。

两条路径必须共享显式 ownership lock：

- active layerwise round 期间，persistent cache 可以累计统计，但普通 hybrid streaming dispatcher 不再重复提交同一层 candidate。
- layerwise shadow 若能直接提供 candidate，可先 D2D 到 cache staging；resident mapping 仍只在 round 结束后的 safe boundary 发布。该规则仅适用于 full-layer 路径；普通 hybrid prefill 的 candidate 在 staging ready 后立即参与当前 chunk GPU wave 2。
- layerwise transport 与 cache loader 共享 CPU writer、host staging 或 raw canonical snapshot 时，必须由同一调度器串行化资源所有权。
- MXFP4 resident raw snapshot 在 promotion commit 后更新；下一次 layerwise round只能读取已发布 generation 的 snapshot。

### 11.2 Decode

decode 是 cache consumer，不是 cache trainer：

- 不更新 `window_count/reuse_signal`。
- 不生成 candidate/victim。
- 不执行 CPU expert export、H2D、repack 或 resident commit。
- 命中 prefill 已冻结的 resident expert 时走 GPU。
- 未命中时继续走 CPU hybrid 路径。
- 同一 decode generation 内 mapping 保持稳定。

单请求流程中，decode 开始前必须完成 10.6 节 finalization。首版冻结作用域是整个 active decode generation：只要存在正在使用该 generation 的 decode batch，新请求的 prefill 不得发起 expert writer/H2D/repack，也不得发布新的 resident mapping。scheduler 必须把动态 cache 更新限制在 prefill-only epoch；更细粒度的多 generation 并发不属于首版。

该限制在 continuous serving 下可能导致更新饥饿。Phase 5 的目标是让旧 decode 固定使用 generation G，同时允许新 prefill 在 shadow slots 中准备 G+1，并在 TP 原子发布后仅让新 batch 绑定 G+1；在该机制完成前不得放宽首版冻结规则。

### 11.3 All-hit fast path

当某层本 batch 的所有 routed assignments 都命中 GPU 时，后续阶段应增加 fast path：

- 跳过 CPU activation staging。
- 不创建 CPU forward task。
- 不执行 CPU sync 和零输出 H2D。

这不是首版 cache correctness 的前置条件，但它决定高命中率能否进一步消除 hybrid 固定开销。

## 12. TP、一致性与 CUDA Graph

### 12.1 TP 一致性

必须保证：

- 所有 ranks 的同一物理 slot 对应同一 logical expert。
- 每个 rank 收到该 expert 对应的正确 TP shard。
- 任一 rank 加载或 prepare 失败时，不允许其他 ranks 单独发布新映射。
- promotion plan、slot generation 和 commit epoch 在各 ranks 一致。

TP0 可继续负责向所有 rank 的共享 pinned host buffer 写入分片，各 rank 自己执行 H2D。控制面应复用现有 layerwise transport 的 event 与 collective error fence。

当前 chunk streaming 还必须保证：

- 各 rank 对同一 candidate 使用相同 stream generation、assignment partition 和 logical expert ID。
- 每个 rank H2D/repack 自己的 TP shard；所有 ranks 对该 candidate ready 后，才能启动对应 GPU wave 2。
- TP0 先广播确定性的 candidate order；所有 ranks 只能按 `(epoch, layer, candidate_order, phase)` 的固定顺序发起 ready/claim/install collective。不能由各 rank 的本地 ready callback 自主发起 NCCL，否则不同到达顺序会造成 collective 错配或死锁。可使用“有序 ready prefix”或一次固定大小 readiness bitmap collective。
- 优先使用独立 cache-control process group；若复用模型 `device_group`，必须证明 control collective 与本层最终 all-reduce 在所有 streams/ranks 上具有完全一致的全局顺序。
- deadline 与 GPU/CPU claim 由 TP0 作权威决策并广播，所有 ranks 得到相同 ownership 后才执行。late CPU fallback 只由 TP0 计算，其他 ranks 对该 candidate 贡献零。
- 不为每个 candidate 额外执行完整输出 all-reduce；resident、streamed 和 CPU contribution 先合并为本层本地输出，再复用该层原有的最终 TP collective。
- 首版只支持已经确认存在单一 post-expert TP collective 的路径；deferred-finalize、跳过最终 all-reduce 或把 collective 融进 expert kernel 的 backend 必须 fail fast，除非完成专门集成。
- 某 rank 失败时，所有 ranks 对该 candidate 一致切换为 late CPU fallback 或一致终止，不能一部分 rank GPU、一部分 rank CPU。

host staging 也需要跨 TP rank 的释放协议：TP0 只有在每个 rank 的本地 H2D event 完成、所有 ranks 完成 release consensus 且 generation 仍匹配后，才能让 writer 覆盖该 host slot。失败路径同样必须完成共同 fence/poison，不能只等待 TP0 自己的 DMA。

### 12.2 CUDA Graph

要求：

- resident 权重 tensor 地址不变化。
- mask 和 mapping tensor 地址不变化。
- decode graph 内不运行 cache 统计、policy 或 loader kernel。
- graph padding 与 capture/warmup 因此不会进入 `window_count`。
- Python policy 决策只由 prefill 流程触发，不在 graph replay 内运行。
- 当前 prefill GPU wave 2 使用独立 staging 权重和私有 dispatch metadata，不修改 graph 可见的 persistent mapping。
- persistent 权重内容和 mapping 只在 prefill/finalization、没有活跃 decode generation 的安全边界更新。
- commit 前用 CUDA event 保证旧 slot 不再被读取。
- 不允许用每 token `cudaDeviceSynchronize` 代替 event 协议。

### 12.3 并发 batch

如果 model runner 未来允许多个 forward 同时在途，不能依赖“同一层下一次调用一定晚于 loader”这一隐含假设。staging + generation + safe-boundary commit 必须成为正式并发协议，而不是仅靠 Python 调用顺序。

第一版最稳妥的提交边界是：

- 为释放双 staging，某 candidate 的 unpublished install 可以在该层 wave 1/wave 2 已 quiesce、该层 exclusive lease 已取得后执行；此后旧 generation 禁止任何新 consumer 进入该层。
- persistent mapping 的公开发布仍延迟到完整 model batch 安全边界。

- 完整 model batch 已结束。
- 该 batch 的 CPUInfer tasks 已 drain。
- 旧 GPU slot 的 last-use event 已完成。
- staging candidate 若走 GPU，其 wave 2 consumed event 已完成；若走 CPU fallback，则其 loader 已停止或达到可安全安装/释放状态。
- 没有活跃 decode generation。
- 下一 batch 尚未提交。

在共享 pinned CPU mask 尚未实现双缓冲/generation 隔离前，`decayed-lfu` 应禁止 two-batch overlap、SBO/TBO 或其他允许旧 CPU task 与新 mapping 并存的模式，并在启动校验阶段明确报错。多请求被 scheduler 合并进同一个 batch 不属于此处的并发，它们共同使用一个 mapping generation。

## 13. Backend 支持

### 13.1 初始支持原则

`decayed-lfu` 只能在以下能力全部存在时启用：

1. CPU 端保留所有 expert 的有效 packed 权重。
2. 支持按 logical expert ID 导出单 expert。
3. 支持写入指定物理 GPU slot。
4. 支持该 slot 所需的 GPU backend prepare/repack。
5. 支持使用不超过 staging pool depth 的 prepared experts 执行当前 chunk 的小型第二批 GPU GEMM；默认双 staging 对应 1～2 expert ready group，并返回 candidate-private output。
6. CPU backend 支持 per-call transient exclusion 或显式 filtered assignments，不能依赖提前修改 persistent mask。
7. candidate 失败时能够使用保留的 assignment metadata 和独立预分配的 fallback input/output buffer 提交 late CPU fallback；不得覆盖同层主 CPU task 的 buffer 或依赖单 pending-task 假设。
8. 支持稳定地址的 in-place resident 权重更新。

不满足时应启动失败并给出明确错误，不能静默退化为错误权重或部分动态更新。

### 13.2 MXFP4

当前 MXFP4 resident dynamic update 会复制 selected raw experts 后，对整个 resident tensor 调用 `prepare_v4_mxfp4_marlin()`。

新缓存需要增加 expert-granular API，例如：

```python
prepare_v4_mxfp4_marlin_expert(
    raw_w13,
    raw_w13_scale,
    raw_w2,
    raw_w2_scale,
    out_prepared_slot,
)
```

以及从 prepared staging 直接执行小 expert group 的接口，例如：

```python
apply_v4_mxfp4_marlin_streamed_experts(
    prepared_staging,
    candidate_assignments,
    topk_weights,
) -> candidate_private_output
```

streamed runner 必须负责：按 candidate compact token rows、把 logical expert ID remap 为 staging-local `0..M-1`、以单 candidate `topk=1` 保留原始 top-k weight、返回 candidate-private output，并在 TP success consensus 后按原 token/assignment index scatter-add。routed scaling 和 top-k weight 只能应用一次；consumed event 必须记录在最后一个读取 staging 的 GEMM/scatter kernel 之后。

或者为 shape `[1, ...]` 的 raw slice 和 prepared slice 构造经过验证的 slot view。必须新增数值一致性测试，不能假设全 tensor API 的 slice 调用天然安全，也不能假设现有 resident fused MoE kernel 天然能够读取独立 staging 地址。

初始 MXFP4 支持范围建议与当前 KT Marlin dynamic path一致；FlashInfer、TRT-LLM、Humming 或其他后端只有在实现其原生单-slot更新协议后才能启用。

### 13.3 其他格式

BF16、block-FP8、per-channel FP8、MXFP8、INT4 Marlin 等格式需要逐项确认：

- writer 是否存在；
- CPU packed layout 到 GPU raw layout 的转换是否正确；
- postprocess 是否支持单 expert；
- scale、zero point、g_idx 或 permutation 是否随 slot 一起更新；
- 更新后的 GPU 输出是否与完整加载参考一致。

建议的首版支持矩阵：

| CPU/GPU 路径 | 首版建议 | 说明 |
|---|---|---|
| BF16 / native unquantized | 支持 | slot 可直接覆盖，后处理最少 |
| Block-FP8 / per-channel FP8 | 条件支持 | 需要验证 scale 布局和可能的单槽 repack |
| INT4 Marlin | 条件支持 | 需要单槽 transpose、scale permutation 和 repack |
| DeepSeek V4 MXFP4 + KT Marlin（SM89/SM120） | 实验支持 | 必须新增 raw-to-prepared 单 expert 路径 |
| FlashInfer CUTLASS / TRT-LLM / Humming / generic MXFP4 | 暂不支持 | backend-specific 内部布局不能按 Marlin slot 直接覆盖 |
| MXFP8 | 暂不支持 | 当前没有匹配的 resident 单槽转换和提交实现 |
| GPTQ INT4、部分 AVX/AVX2 writer 组合 | 启动时探测 | writer 或单槽转换缺失时 fail fast |
| CPU-only / 无 CUDA | 不支持 | 不存在 GPU resident cache |

启动校验不能只检查 `kt_method` 字符串，必须同时探测：

1. CPU single-expert writer capability。
2. 当前 GPU resident weight layout。
3. backend-specific single-slot postprocess capability。

该探测应使用显式 capability API/枚举或一次不会破坏状态的启动自检，不能仅依赖 `hasattr`。仓库中部分类型虽然暴露同名 writer 接口，实际调用仍可能抛出 `NotImplemented` 或 backend runtime error。

## 14. 失败处理与回退

### 14.1 加载失败

使用 staging 时：

- 旧 victim slot 和映射保持不变。
- staging 标记 FAILED/POISONED；只有对应 writer、各 rank DMA、repack 和可能存在的 GPU/install event 全部完成共同 fence 后，才释放给后续计划。
- 记录失败阶段：host pack、H2D、repack 或 prepare consensus。
- 如果 candidate 尚未进入 `GPU_CLAIMED`，原子切换为 `CPU_CLAIMED`；若 streamed GPU 只写 private output 且在 merge 前报告可恢复失败，也可经 TP consensus 从 `GPU_FAILED_RECOVERABLE` 切换到 `CPU_CLAIMED`。两种情况都使用 `KTExpertStreamPlan` 保留的 activation、assignment、top-k weight 和 scatter metadata 提交 late CPU fallback，当前层 merge 必须等待该结果。
- late fallback 使用独立预分配 buffer 或支持多 ticket 的 CPU backend API；不能复用仍被主 `C_cpu` task 占用的同一 layer input/output buffer。
- 异步 CUDA fault、可能部分写入共享 output 或 CUDA context 损坏不可恢复，必须 fail-stop。
- 如果 GPU wave 2 已成功、persistent install 在覆盖 victim 前失败，则当前 chunk 输出仍然有效，后续请求继续使用旧 resident mapping；覆盖 victim 后的 install/publish 失败按 post-overwrite 规则 fail-stop。

上述可恢复保证仅覆盖 candidate 尚未覆盖 resident slot 的阶段。进入最终 D2D/映射 commit 后若发生部分失败，按 10.5 节要求 fail fast，不能将其当作普通 promotion failure 继续运行。

### 14.2 计划过期

以下情况取消 plan：

- victim slot generation 已变化。
- candidate 已由其他计划 resident。
- candidate 分数已不再满足 admission。
- policy epoch 已重置。
- cache controller 被管理员禁用或冻结。

取消 persistent promotion 只阻止最终 resident 提交，不自动取消仍属于当前 chunk 的 streaming compute。取消 loader/stream ticket 时也不能假设底层任务会立即停止；相关 host/GPU staging 必须保持占用，直到 writer task、H2D、repack，以及可能存在的 GPU consumed/install-D2D event 已完成或被底层确认取消。generation 过期后提前复用 staging 会造成旧任务写入新计划 buffer。

### 14.3 连续失败

连续失败达到阈值后：

- 停止新的 promotion。
- 保留当前 resident 集合。
- 将 adaptive controller 降级为当前已发布的静态 placement。
- 输出一次高优先级日志和可查询状态，不反复刷屏。

### 14.4 OOM

staging allocation 在启动或首次启用时完成。运行时不能临时扩大 resident tensors。若 staging OOM：

- decayed-lfu controller 启动失败，或安全冻结为当前静态 placement。
- 不能退回到每次 promotion 动态分配。

## 15. 代码改动地图

### 15.1 SGLang 参数

文件：

- `third_party/sglang/python/sglang/srt/arg_groups/fields/exec_.py`
- `third_party/sglang/python/sglang/srt/arg_groups/field_order.py`
- `third_party/sglang/python/sglang/srt/arg_groups/kt_hook.py`
- `third_party/sglang/test/registered/unit/server_args/test_kt_server_args.py`

改动：

- 为 `kt_expert_placement_strategy` 增加 `decayed-lfu` choice，并更新 help 文案。
- 新增 `kt_prefill_stream_top_n: Optional[int] = None`，对应 `--kt-prefill-stream-top-n`；在 `decayed-lfu` 下解析有效 `N_stream`。
- 将衰减、过期、滞回、staging 深度和预算实现为内部默认配置，不再增加其他 CLI。
- 校验 decayed-lfu 与旧 dynamic flag、ratio、LoRA、backend 的组合。
- 校验显式 `N_stream` 满足 `0 <= N_stream <= C`；静态 placement 下显式配置该参数直接启动失败。
- decayed-lfu 不要求 positive prefill threshold。

### 15.2 KTEP wrapper

主文件：

- `third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py`

建议新增：

- `KTExpertCacheState`
- `KTExpertCachePolicy` / `DecayedLFUPolicy`
- `KTExpertCostModel`（Phase 6）
- `KTExpertStreamDispatcher`
- `KTExpertCacheManager`
- `KTExpertStreamPlan`
- `KTExpertPromotionPlan`
- `record_prefill_route_accesses()`
- `select_current_stream_hotset()`
- `build_stream_dispatch_snapshot()`
- `split_resident_stream_cpu_assignments()`
- `run_streamed_expert_wave()`
- `submit_late_cpu_fallback()`
- `plan_persistent_promotions()`
- `estimate_net_benefit()`（Phase 6）
- `commit_ready_promotions()`
- `finalize_prefill_cache()`

需要调整：

- `KTConfig` 识别 `kt_expert_placement_strategy == "decayed-lfu"`，解析 `N_stream` 并初始化 cache controller。
- `create_weights()` 必须首先保留现有 `layer_idx < kt_num_gpu_layers -> return None/native full-GPU` 分支；只有后续 KT-managed MoE 层在 strategy 为 `decayed-lfu` 时初始化 cache state，并启用 all-expert CPU pack。
- `apply()` 只对 scheduler 标记的 prefill/extend rows 记录统计；已发布 mapping 只决定第一批 resident GPU wave，同时为当前 batch 构造 stream-candidate 与 CPU-only dispatch。
- CPU task 接收 per-call transient exclusion/filtered assignments，不能把尚未发布的 stream candidate 当作普通 CPU miss 重复计算。
- CPU 主任务最直接的 transient exclusion 实现是复制当前调用的 logical `topk_ids`，把 resident assignments 和 stream assignments 位置写为 `-1` 后提交；late fallback 反向构造仅保留 `CPU_CLAIMED` stream assignments 的 ID tensor。persistent pinned mask 不承担该职责。
- GPU backend 增加直接消费 prepared staging 的小 ready-group wave，先返回 candidate-private output，TP success consensus 后再一次性 scatter/merge 回原 MoE output。
- 首版禁止 `--kt-max-deferred-experts-per-token` 与 `decayed-lfu` 同时启用，除非 deferred CPU IDs/buffer 也接入同一 assignment ownership，避免 stream candidate 被 deferred task 重复计算。
- prefill-to-decode 转换点调用 finalization，drain/commit 有界在途任务并冻结 mapping。
- `decayed-lfu` 启用时不调用旧 `_update_gpu_experts_from_batch()`。
- mapping 更新保留现有 in-place `copy_()` 约束。
- cache mode 下只有 `layer_idx >= L_gpu` 的 KT-managed MoE 层初始 mask 严格 C 个；前置 full-GPU layers 不创建该 mask。

### 15.3 kt-kernel 单专家 writer

相关文件：

- `kt-kernel/python/utils/amx.py`
- `kt-kernel/ext_bindings.cpp`
- `kt-kernel/cpu_backend/cpuinfer.h`
- 对应 CPU backend MoE writer 实现

需要增加：

- task-specific writer completion/future。
- 不依赖全局 `CPUInfer.sync()` 的等待接口。
- 主 CPU-only task 与 late fallback 使用独立预分配 buffer/ticket，或增加可安全并发/排队的增量 expert task API。
- writer 使用受限的独立 WorkerPool/NUMA 资源，或明确的 compute-idle 调度；future 只能解决错误 drain 全队列，不能解决与 CPU MoE 争抢相同 worker 和 DRAM 带宽。
- 可观测的 host-pack 起止时间、字节数和错误状态。
- loader backpressure 或低优先级队列。

### 15.4 MXFP4 单槽 prepare

相关文件：

- `third_party/sglang/python/sglang/srt/layers/quantization/v4_marlin_moe.py`
- `third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py`

需要增加：

- 单 expert raw/prepared staging storage。
- 单 slot Marlin repack 和 scale swizzle。
- 不超过 staging pool depth 的第二批 GPU GEMM/ready-group dispatch；默认双 staging 为 1～2 experts。
- resident raw canonical snapshot 的单 slot 更新。
- 与 layerwise manager 的互斥和 event 协议。

### 15.5 测试

建议新增：

- `third_party/sglang/test/registered/unit/layers/moe/test_kt_expert_cache_policy.py`
- `third_party/sglang/test/registered/unit/layers/moe/test_kt_expert_cache_mapping.py`
- kt-kernel 中对应 backend 的单 expert export/repack correctness 测试。

现有可扩展测试：

- `third_party/sglang/test/registered/unit/layers/moe/test_kt_ep_helpers.py`
- `third_party/sglang/test/registered/unit/server_args/test_kt_server_args.py`

## 16. 分阶段实施计划

### Phase 0：测量与只观测模式

目标：

- 不搬权重。
- 记录按层、按 chunk 的 route/hit/miss 时间序列。
- 测量不同 token-count bucket 的 CPU GEMM 成本。
- 测量单 expert CPU export、H2D、repack、TP 和 install 成本。
- 离线重放 static、LRU、frequency、lifetime-LFU 和 decayed-LFU。

输出：

- 每层 resident hit rate、Stream-TopN 缺失数量和 zero-CPU-only batch 比例。
- replacement 次数、估算字节数和理论 net benefit。
- prefill/decode 分开统计。
- 不同 capacity 下的收益曲线。

### Phase 1：固定 slot 与 hit/miss 正确性

目标：

- 固定 per-layer slot 数量和地址。
- 明确 CPU backing 与 GPU residency 两套状态。
- 验证 logical expert、physical slot 和 TP shard 映射。
- 暂不做自动替换，只允许测试接口手工指定 resident 集合。
- 验证 all-expert CPU pack 和被淘汰专家的 CPU fallback。

### Phase 2：单 candidate 当前 chunk 正确性链路

目标：

- 选择一个明确支持的 backend。
- 首先只允许 `|P| <= 1`，实现 resident GPU wave、CPU-only 和单 candidate staging GPU wave 三路 exact-once 分流。
- candidate 从主 CPU task 排除，加载失败能够提交 late CPU fallback。
- staging GPU wave 在 persistent commit 前执行，证明当前 chunk 可以直接受益。
- 在 prefill/model safe boundary 同步更新一个 resident slot，并实现 last-use/consumed event。
- 验证 TP、数值正确性、pre-commit 回滚、post-install fail-stop、地址稳定和输出只累加一次。

该阶段可以暂时接受 writer 同步，因此只用于验证语义，不能作为性能版本，也不能进入 decode token 热路径。

### Phase 3：Stream-TopN、Frequency、EMA 与 Decayed-LFU

目标：

- 实现持久 prefill 统计、衰减、过期、滞回和 promotion plan。
- 比较 LRU、当前窗口 frequency、EMA 和 Decayed-LFU。
- 实现稳定 current-window Stream-TopN，并验证 `0..N_stream` 个缺失 candidate 严格不向后补位。
- 实现 `KTExpertStreamPlan` 与 `KTExpertPromotionPlan` 分离。
- 保持 slot identity 稳定。
- 验证 hotset 中 resident experts 不搬运，只有按 hotset 原顺序过滤出的 non-resident experts 进入 stream plan。
- 先以同步逐 candidate staging compute 验证多 candidate 决策和 fallback。

### Phase 4：异步流式换入与当前层计算重叠

目标：

- 增加 task-specific completion。
- 使用按兼容签名分组的 staging pool 和 loader queue。
- 为 writer 配置受限的独立 WorkerPool/NUMA 资源，或实现明确的 compute-idle 调度，不能只新增 future 后继续与 CPU MoE 无限制争抢同一 pool。
- Router 后立即同时启动 resident GPU wave 1、CPU-only task 和 Stream-TopN missing loader。
- V1.1 为 MXFP4 建立独立 writer CPUInfer FIFO 和受限辅助 WorkerPool；双 staging 首批 writer 与主 CPU-only task 分别进入不同队列，writer completion 返回后立即启动 H2D/repack/wave 2，slot 释放后继续滚动激活 candidate 3 及以后。全部 `P` 从主 CPU task 排除并最多流式执行 `N_stream` 个；辅助池默认每个 CPU TP/NUMA 1 个 worker，后续根据 trace 决定是否需要自适应限速。
- 实现“candidate 1 搬完即算 candidate 1，candidate 2 继续搬”的双 staging 流水；多个同时 ready 时允许小 grouped GEMM。
- 当前 chunk GPU wave 2 完成后，再在 prefill safe boundary 提交 persistent promotion。
- 在 prefill-to-decode finalization drain 有界的在途任务并冻结 mapping。
- decode 不创建新的 loader task。

### Phase 5（后续规划）：Continuous batching 与 generation-safe 更新

已确认的问题：

- 首版 global decode freeze 在单请求流程中风险较低，但在持续 continuous batching 中可能长期没有 prefill-only epoch。
- active decode 长期存在时，mixed batch 的 non-resident prefill assignments 只能直接走 CPU-only，cache mapping 可能长期不更新。
- 这会造成 update starvation、mapping generation 老化和热点漂移后适应失败；动态 cache 最坏会退化成启动时静态 placement。
- 这里不是 late CPU fallback：loader 从未启动，assignment 从一开始就属于普通 CPU-only 路径。
- 为了制造更新窗口而频繁暂停 decode 会直接形成 ITL 尖峰；完全不暂停则 promotion 可能永久饥饿，因此必须显式解决 generation 生命周期，而不能只靠“以后会出现空闲窗口”的假设。

#### Phase 5A：低成本过渡方案

- 即使保留严格冻结模式，也必须观测 prefill-only epoch 比例、被 decode 抑制的 candidates、generation age 和最长 update starvation。
- 将“权重准备”和“破坏性安装/映射发布”拆成两个独立权限：active decode 期间最多允许受预算约束的 prepare-only，绝不覆盖 resident slot，也不修改 published mapping。
- prepare-only 可包含 route aggregation、host pack、H2D 和 repack 到独立 staging，但必须使用低优先级 stream、全局 byte credits、CPU/NUMA credits 和 decode-ITL backpressure；任一预算不足时继续 CPU-only，不强行准备。
- 若 scheduler 能在所有当前 decode replay、CPU task 和 TP collective 完成后形成全局 quiescent boundary，可获取 generation lease 并执行 install/publish；下一次 replay 必须等待 publish event。
- 若 continuous serving 长期没有自然 quiescent boundary，由 scheduler 创建有硬频率、硬时长预算的 maintenance epoch。该方案是 fallback/对照组，不能隐藏其造成的 decode ITL 尖峰。

Phase 5A 只能缓解饥饿；它要求旧 decode 全部 quiesce 后才能发布，不能支持旧、新 generations 真正并存。

#### Phase 5B：目标架构——双 generation 与同层 shadow slots

1. 每个 model batch 在进入模型前绑定不可变的 `cache_generation_id`，整个 forward/CUDA Graph replay 期间不得改变。
2. 预分配两套固定地址的 mapping/mask banks 和 generation-specific host mirrors/pinned masks，不能让所有在途 batch 读取同一张可变 CPU mask。
3. CPU task 必须接收 generation-specific mask 或 immutable filtered assignments，不能执行期间读取“当前全局 mapping”。
4. 每个 cache-managed layer 预留有界的同层 shadow slots。旧 generation 仍在读取 victim 时，新 candidate 只能安装到 shadow，不能覆盖旧 resident slot。
5. shadow slots 只解决同层 copy-on-write，不属于 Global Spare，也不能跨层借用。
6. 首版最多同时存在两个 generations：`ACTIVE_OLD` 与 `PUBLISHED_NEW`。旧 generation 未退休前禁止创建第三代。
7. prefill 可在旧 decode 使用 generation G 时，低优先级准备 G+1；所有 TP ranks 完成 shadow prepare 并 success consensus 后，TP0 按固定事务顺序原子发布 G+1。
8. backend 不支持 per-token generation 时，scheduler 必须形成 generation-homogeneous batches；不得把 G 和 G+1 混进同一个 fused MoE batch。
9. CUDA Graph 必须使用地址稳定的双 bank mapping + generation selector，或为两个 banks 预建 graph；不能在发布时替换 graph 捕获的 tensor 地址。
10. 旧 generation 只有在 host refcount 归零，并且最后一个 GPU last-use event、CPUInfer task completion 和 graph replay completion 全部完成后才能回收。
11. G 退休后再把 G+1 shadow experts 压实回标准 resident bank，或把 shadow 纳入下一代 arena；压实期间仍需 generation/event 协议。
12. shadow 容量不足、旧 generation 长期不退出、generation batching 碎片过大或 decode ITL 超预算时，停止创建 G+1 并安全退回 global freeze。

硬性约束：

- active decode generation 使用的权重、mapping 和 pinned CPU mask 在其生命周期内不可原址改变。
- prepare-only 任务不得占用或覆盖当前 resident slots。
- 不允许为了获得 cache 更新机会无界 drain decode queue。
- 不支持 per-token generation 时，不允许 mixed-generation fused batch。
- 不允许无上限 generation、shadow slots 或后台 prepare backlog。
- 不得仅根据 host request refcount 回收 generation；必须同时等待 GPU/CPU completion。
- 如果无法证明 generation、TP 顺序、CUDA Graph 或资源隔离安全，则继续使用首版 global freeze，而不是隐式放宽。

验收：

- 在“decode 始终活跃 + 周期性 prefill + 热点切换”压测中，`cache_update_starvation_ms` 有明确上界，mapping generation 能持续推进。
- generation G 的 decode 在 G+1 prepare/publish 期间与严格冻结基线数值一致；新请求只有在 TP 原子发布完成后才能绑定 G+1。
- 任一 rank prepare/publish 失败时，所有 ranks 继续使用 G，不产生半代状态。
- 旧 generation 未完成 GPU/CPU work 前，其 resident/shadow slots、mapping bank 和 pinned mask 绝不复用。
- prepare-only 和 publish 不产生 stale-generation、重复 assignment、TP collective 错序或 CUDA Graph 地址变化。
- decode ITL p50/p95/p99、TTFT 和吞吐与严格冻结基线对比；任何命中率提升都必须扣除 maintenance epoch、后台 H2D 和 host pack 的代价。
- 热点切换后能在有界时间内更新 resident cache，而不是等待所有 decode request 全部结束。
- generation 分桶造成的 batch size 下降、吞吐损失、shadow VRAM 和双 mapping 内存必须单独报告。

### Phase 6（后续规划）：显式预测与成本感知 admission

目标：

- 建立下一 Chunk/后续 decode workload predictor。
- 用实测 `CPU_GEMM_cost(token_bucket)` 估算 candidate_saved。
- 扣除 victim_loss 和完整 promotion_cost。
- 以 net expected benefit 代替纯 frequency score 做最终 admission。
- 加入全局在途数量、传输字节和未来 workload 准备时间预算。
- 增加 all-hit CPU fast path。

### Phase 7：Per-layer min/max 与 Global Spare

目标：

- 评估全局 GPU weight arena 或兼容 slot pool。
- 增加 `min_slots/max_slots/borrow_limit/borrow_cooldown`。
- 以新增 slot 的边际 expected benefit 分配 spare。
- 保证任何单层不能无限借用 global spare。

该阶段是独立架构扩展，不阻塞固定 per-layer capacity 方案上线。

## 17. 测试计划

### 17.1 Policy 单元测试

必须覆盖：

1. 稳定热点：完成首次 promotion 后不再重复搬运。
2. `N_stream=4` 且 hotset 全 resident：`P=0`，不选择第 5 名，不发生 H2D。
3. `N_stream=4` 且 hotset 中分别有 1/2/3 个 resident：只选择其余 3/2/1 个，不向后补位。
4. `N_stream=4` 且 hotset 全 miss：最多产生 4 个 stream candidates。
5. current count 相同：按 reuse signal、logical ID 稳定排序。
6. 热点切换：旧热点经过衰减和过期后可被新热点替换。
7. C+1 循环扫描：滞回与最小驻留限制 persistent cache thrash。
8. 均匀随机：stream candidates 始终不超过 `N_stream`，persistent replacement 不超过内部预算。
9. unique experts 少于 `N_stream`：Stream-TopN 不补零分 expert。
10. 多个 prefill chunk 能逐步适应热点，并在进入 decode 后冻结。
11. stream candidate 可执行当前 GPU wave，但因无 eligible victim 而不提交 resident。
12. plan 排队后过期：generation 校验只取消 persistent commit，不丢失当前 chunk 输出。
13. decode graph、padding 和 capture/warmup 不增加 policy 访问分数。
14. TBO/SBO 等不支持组合在启动时被拒绝。
15. `IN_USE` slot 永远不会进入 `INSTALLING_UNPUBLISHED`。
16. 不同 expert footprint/compatibility class 不能复用同一 staging 或 spare slot。
17. Phase 6 的 net-benefit 规则单独测试，不作为首版 Stream-TopN streaming 的前置条件。
18. 参数边界：默认 `None -> min(4, C)`；显式 `0`、`1`、`C` 正常，负数或大于 `C` 启动失败；静态 strategy 下显式配置启动失败。
19. `N_stream=0`：继续记录统计，但 stream/promotion loader、H2D、wave 2 和 resident replacement 数量均为 0。
20. mixed prefill+decode batch 只记录 prefill 统计，stream loader 数量严格为 0，non-resident prefill assignments 全部走普通 CPU-only，而不是 late CPU fallback。
21. 双 staging + `N_stream>2`：全部 missing candidates 获得当前窗口 GPU ownership并从主 CPU task 排除；任一时刻最多 2 个占用 staging，candidate 3 及以后随 slot 释放按确定性顺序滚动执行。正常路径全部由 wave 2 恰好完成一次，故障尾部合并为一次 late CPU fallback。
22. 各 TP rank 的本地 ready 顺序不同，collective 仍按统一 candidate order 执行且不死锁。
23. candidate 已 `CPU_CLAIMED`、但 loader 后续成功时，可以不等待 consumed event 而执行 persistent install。
24. 主 CPU-only task 与 late fallback 使用独立 buffer，输出不互相覆盖。
25. `layer_idx < L_gpu`：不创建 KT wrapper/cache state，不记录 Decayed-LFU，不产生 stream loader；输出与原 native full-GPU 路径一致。
26. `layer_idx == L_gpu` 的首个 KT-managed MoE 层开始应用 C 个 resident slots 和 `N_stream`，验证边界无 off-by-one。

### 17.2 Mapping 不变量

任意提交后必须满足：

```text
sum(resident_mask) == capacity
resident expert <=> expert_to_slot >= 0
slot_to_expert[expert_to_slot[e]] == e
每个 slot 只有一个 logical expert
每个 resident expert 只有一个 slot
所有 TP ranks 的 slot_to_expert 完全一致
```

同时验证 tensor `data_ptr()` 在更新前后不变化。

当前 chunk 还必须验证：

```text
resident_assignments intersect stream_assignments == empty
resident_assignments intersect cpu_assignments == empty
stream_assignments intersect cpu_assignments == empty
union(all three assignment sets) == original_router_assignments
```

streaming 私有 mapping 不得提前改变 persistent resident mask/mapping。

### 17.3 数值正确性

对于每个支持 backend：

1. 从 CPU packed 权重单独加载 expert X 到 staging。
2. 不 commit resident，直接从 staging 对 expert X 执行当前 chunk GPU wave 2。
3. 将 streamed output scatter 回原 token rows，与完整模型加载或 CPU reference 比较。
4. 再 commit 到指定 slot，验证后续 resident 路径结果一致。
5. 构造 resident + streamed + CPU-only 三路混合输入，验证合并结果与全 CPU/全 GPU reference 一致。
6. 覆盖 gate/up/down、top-k weight、routed scaling、scale、zero point、permutation 和 TP shard。
7. 一个 expert 多 token 时使用 grouped GEMM，验证不会按 assignment 重复启动或重复累加。

### 17.4 故障注入

覆盖：

- host writer 抛错。
- 某一个 TP rank H2D 失败。
- repack 失败。
- plan generation 过期。
- commit 前 cache controller 被禁用或冻结。
- staging OOM。
- loader 队列拥塞。
- candidate deadline 到期。
- GPU wave 2 在 private output commit 前返回可恢复失败，以及异步 CUDA fault 的 fail-stop 路径。
- host staging 在部分 TP rank H2D 较慢或失败时，不会被 TP0 提前复用。

candidate 覆盖 resident slot 之前的可恢复故障必须原子切换到 late CPU fallback，并保证旧 resident 集合仍可正确执行。不得同时接受 streamed GPU 和 fallback CPU 两份输出。进入最终 install/publish 后的部分失败按 10.5 节 fail-stop，不适用普通回滚承诺。

### 17.5 CUDA Graph

验证：

- graph capture 前后 mapping tensor 地址稳定。
- 当前 prefill chunk 能在不修改 persistent mapping 的情况下从 staging 执行 GPU wave 2。
- prefill commit 后的下一次 replay 能读取新权重和 mapping。
- promotion 与 replay 不并发覆盖。
- 更新后结果与 eager reference 一致。
- decode replay、padding 和 capture/warmup 不修改 prefill policy score。

### 17.6 Continuous batching 与 generation 生命周期

严格冻结首版验证：

- decode 始终活跃时 mapping generation 不变化，同时 `prefill_windows_blocked_by_decode`、`suppressed_stream_candidates` 和 starvation 指标正确增长。
- mixed batch 的 resident assignments 仍走 GPU，non-resident prefill assignments 从一开始走 CPU-only，不创建错误的 late-fallback ticket。

Phase 5A 验证：

- active decode 期间 prepare-only 只能写独立 staging，resident slot、published mapping 和 pinned CPU mask 均保持不变。
- maintenance epoch 有硬频率/时长预算，超过预算自动回到 global freeze，并报告 decode ITL 尖峰。

Phase 5B 验证：

- generation G 的长时间 decode 与 G+1 prepare/publish 并行时，G 输出与严格冻结 reference 一致。
- G+1 只有在所有 TP ranks 原子发布成功后才能被新 batch 绑定；任一 rank 故障都保留 G。
- backend 不支持 per-token generation 时，G/G+1 请求不会进入同一个 fused batch。
- 旧 generation 的最后一个 GPU event、CPU task 和 graph replay 未完成前，不复用其 mapping bank、pinned mask、resident 或 shadow slots。
- generation 数量最多为 2；旧 generation 长期不退出时不创建第三代，并安全回退 freeze。
- 覆盖单个超长 decode、持续高并发 decode、周期性 prefill、多租户热点切换、请求取消、TP2/TP4、CUDA Graph 和 generation reclaim timeout。

## 18. 可观测性

### 18.1 必需指标

按层和全局记录：

| 指标 | 含义 |
|---|---|
| `prefill_route_assignments` | 驱动 policy 的 prefill token-expert assignments |
| `prefill_resident_assignments` | 当前 chunk 第一批 GPU wave 的 assignments |
| `prefill_stream_candidate_assignments` | Stream-TopN 缺失 experts 的 assignments |
| `prefill_cpu_only_assignments` | 直接由 CPU 计算的 non-resident assignments |
| `prefill_late_cpu_fallback_assignments` | streaming 失败/超时后转 CPU 的 assignments |
| `decode_cache_hits` / `decode_cache_misses` | 只用于观测冻结 cache 的 decode 效果，不参与替换分数 |
| `decode_active_fraction` | scheduler 时间中存在 active decode generation 的比例 |
| `prefill_only_epoch_ratio` | 允许首版 stream/promotion 的 prefill-only 时间比例 |
| `prefill_windows_blocked_by_decode` | 因 global freeze 无法启动 loader 的 prefill 窗口数 |
| `suppressed_stream_candidates` | 因 active decode 被强制改走 CPU-only 的 candidate 数 |
| `cache_freeze_duration_ms` | 当前 mapping generation 连续被冻结的时间 |
| `cache_update_starvation_ms` | oldest pending/suppressed promotion 等待时间 |
| `mapping_generation_age` | 当前 mapping generation 已服务的 step/token/request 数 |
| `unique_active_experts` | 本窗口不同激活专家数 |
| `effective_stream_top_n` | 本层解析后的 `N_stream` |
| `all_resident_batches` | 所有 assignments 都命中 persistent resident 的 batch 数 |
| `zero_cpu_only_batches` | 除 streaming candidates 外不需要普通 CPU expert 的 batch 数 |
| `stream_candidate_count` | 本窗口按 hotset 顺序过滤 resident 后的 candidate 数，范围 `0..N_stream` |
| `stream_staging_depth` / `stream_queue_depth` | 同时在途的 staging 容量与等待滚动执行的 candidate 数；前者不得改写本窗口 ownership 总数 |
| `stream_gpu_success` / `stream_cpu_fallback` | candidate 当前 chunk 最终由 GPU/CPU 完成的数量 |
| `promotions` / `evictions` | 实际提交次数 |
| `rejected_promotions` | 因滞回、预算、过期或 backend 被拒绝 |
| `promotion_bytes` | host pack、H2D 和 D2D 字节 |
| `host_pack_ms` | 单 expert CPU 导出时间 |
| `h2d_ms` | 单 expert H2D 时间 |
| `repack_ms` | backend prepare 时间 |
| `candidate_ready_ms` | Router 完成到 candidate READY 的时间 |
| `stream_gpu_wave2_ms` | 当前 chunk 第二批 GPU 时间 |
| `commit_ms` | TP consensus 和映射提交时间 |
| `loader_queue_depth` | 后台加载积压 |
| `streaming_overlap_ratio` | candidate host pack/H2D/repack 与同层 GPU wave 1/CPU-only 的重叠比例 |
| `exposed_cpu_wait_ms` | 当前层真正暴露在临界路径上的 CPU wait |

严格冻结首版也必须记录上述 starvation 指标，否则无法判断动态 cache 是否已经在 continuous serving 中退化为静态 placement。

Phase 5 再增加：

- `generation_publish_count` / `generation_publish_ms`
- `active_generation_count`
- `old_generation_drain_ms`
- `generation_blocked_updates`
- `shadow_slot_used` / `shadow_slot_capacity` / `shadow_vram_bytes`
- `generation_batch_fragmentation_ratio` 与按 generation 分桶后的平均 batch size
- `fallback_to_global_freeze` 次数和原因
- decode 活跃期间 loader 的 CPU/DRAM/H2D bytes
- `generation_mismatch`、TP publish abort 和 reclaim timeout

Phase 6 再增加 `predicted_future_tokens`、`candidate_saved_ms`、`victim_loss_ms` 和 `net_expected_benefit_ms`；首版不实现这些在线预测指标。

### 18.2 日志

默认只输出聚合统计。DEBUG 模式可输出：

```text
layer=31 epoch=42
stream_top_n=4
candidate=187 score=0.083
victim=12 score=0.021 slot=7
stream_order=0 current_tokens=512
pack=0.42ms h2d=0.31ms repack=0.18ms wave2=0.27ms
current_result=gpu persistent_commit=success
```

不得在每个 token 输出 resident 列表。

## 19. 性能验收

### 19.1 对照组

至少包含：

1. 静态 `uniform`。
2. 静态 offline `frequency`。
3. 现有 legacy dynamic update。
4. `decayed-lfu` 新缓存。
5. full-GPU layerwise 路径。

### 19.2 工作负载

分别测试：

- 冷 4K prefill。
- 连续多 chunk 长 prefill。
- 单请求 decode。
- 多并发 decode。
- 稳定热点 synthetic routing。
- 周期性热点切换。
- 均匀/高熵路由。
- 多租户相互冲突的热点。
- decode 始终活跃、周期性插入 prefill、并在运行中切换热点的 continuous-batching starvation workload。
- Phase 5 双 generation 下的单个超长 decode、generation-homogeneous batch 碎片和旧 generation 延迟回收。
- 对有效 `N_stream` 做参数扫描，例如 `0/1/2/4/8` 以及 `C` 内的目标值；超过 `C` 的组合只验证启动拒绝。

### 19.3 关键结果

不能只报告 cache hit rate。必须报告：

- prefill tok/s、TTFT。
- decode tok/s、ITL p50/p95/p99。
- TP0 exposed CPU wait。
- TP1 collective wait。
- CPU routed assignments 和 activated experts。
- resident、streamed GPU、CPU-only 和 late fallback assignments。
- candidate ready latency、GPU wave 2 时间和 `streaming_overlap_ratio`。
- all-resident 与 zero-CPU-only layer/batch 比例。
- replacement/sec 和 promotion bytes/sec。
- 不同 `N_stream` 下的 selected candidates、实际 GPU 成功率、fallback 率、H2D bytes 与端到端收益曲线。
- CPU 内存带宽和 WorkerPool 占用。
- GPU 峰值显存与 staging 开销。
- strict freeze 下的 prefill-only epoch ratio、最长 starvation 和 mapping generation age。
- Phase 5 的 generation publish/drain latency、shadow VRAM、generation batch fragmentation 和 freeze fallback 次数。

### 19.4 Go/No-Go 条件

进入默认可用状态前至少满足：

- 现有静态 placement strategies 无行为回归。
- 未发生 promotion 时，prefill 统计开销在目标阈值内；decode 不执行 policy 统计或权重搬运。
- 稳定热点下 replacement 在 warm-up 后趋近于零。
- 热点切换时能够在有界窗口内适应。
- 所有支持 backend 的数值结果在既有量化容差内。
- TP、CUDA Graph 和故障回滚测试全部通过。
- p95/p99 不因后台 loader 争抢 CPUInfer/DRAM 而恶化。
- 单请求 prefill finalization 后的纯 decode trace 中，不存在由 cache scheduler 发起的 expert export、H2D 或 repack。
- profiler 中 resident GPU wave 1 和 CPU-only task 不等待 candidate H2D 才启动。
- candidate 1 ready 后可在其余 candidates 仍搬运时启动 GPU wave 2，不存在“等待所有 candidate ready”的统一 barrier。
- host pack/H2D/repack 与当前层 GPU wave 1/CPU-only 产生可观测重叠。
- Stream-TopN 部分 resident 场景下 selected candidate 数严格等于 hotset 中缺失数量，不向后补位；实际 GPU 成功数与 fallback 数可完整对账。
- `N_stream=0` 与静态 resident 计算路径数值一致，且不产生 cache loader 流量。
- current-chunk 三路方案在目标长 prefill workload 上降低 CPU-only assignments 或 exposed CPU wait，并且端到端 TTFT/prefill tok/s 不退化。
- 在宣称支持高并发 continuous serving 前，必须证明热点切换时 generation 在有界时间内推进；若 strict freeze 下长期不推进，只能标记为单请求/低并发实验能力。
- Phase 5 相对 strict-freeze 基线的 decode ITL p95/p99、吞吐、TTFT 和 batch fragmentation 必须全部报告，不能只报告更高的 cache hit rate。

## 20. 风险与待决事项

1. **Writer 与 CPU GEMM 的带宽争抢**：task-specific completion 和独立 writer FIFO/WorkerPool 已实现，避免了队列级串行；但两者仍共享 DRAM/L3。辅助池默认每个 CPU TP/NUMA 1 个 worker并采用主池之后的 core offset，仍需观测绑核是否成功以及 p95/p99 是否因带宽竞争恶化。
2. **MXFP4 单槽 prepare**：DeepSeek V4 MXFP4 + KT Marlin 已实现单槽 raw-to-prepared；其他格式仍需各自的数值与布局验证。
3. **第二批 GPU backend**：MXFP4 Marlin 已实现 staging-local remap、private output 和逐 candidate wave 2；其他 fused backend 仍不得复用该实现。
4. **CPU transient exclusion**：首版已使用 per-call filtered `topk_ids` 保证 exact-once；deferred-expert 模式在接入相同 ownership 前继续 fail fast。
5. **late CPU fallback**：首版已把失败尾部合并为一次 candidate-only fallback；仍需在目标 GPU 注入 writer/H2D/repack/wave 故障，验证异步错误与 fail-stop 边界。
6. **内存带宽竞争**：独立 writer WorkerPool 消除了队列级串行，但 host export 与 CPU expert GEMM 会并发读取权重并共享 DRAM/L3；单 writer 的带宽影响和 Top-N 尾部仍必须实机观测。
7. **小 GEMM 启动开销**：首版严格逐 expert 执行，可能浪费 GPU；后续可支持小 ready group，但不能等待全部 candidates。
8. **staging 生命周期**：首版使用 ticket-owned event、slot generation、pending install commit 和全 rank host-DMA reuse fence；仍需 CUDA sanitizer/Nsight 验证不存在跨代提前复用。
9. **TP collective 顺序**：首版 token 覆盖 transport、layer、namespace、epoch、ticket、slot、generation、operation 和 stage，错配直接 fail-stop。
10. **大 prefill 覆盖大量专家**：Stream-TopN streaming 不替代所有场景的 full-layer streaming；过大的 `N_stream` 可能重新退化成权重 streaming。
11. **多租户污染**：全局 cache 可能在请求热点间抖动，需要短半衰期、滞回和全局预算。
12. **统计与 graph capture**：不能引入每层 host sync 或动态分配。
13. **旧 dynamic 兼容**：两套更新逻辑不能同时拥有 slot 控制权。
14. **EPLB/placement epoch**：若 logical/physical expert location 外部变化，cache 必须 reset 或重新绑定。
15. **并发 mapping generation**：首版若没有双缓冲 pinned mask，必须禁止 TBO/SBO。
16. **Continuous batching 更新饥饿**：严格 global decode freeze 下，只要持续存在 active decode，mapping 可能长期不更新，动态 cache 退化为静态 placement。Phase 5 必须通过 prepare/publish 分离、双 generation + 同层 shadow slots，或有界 maintenance epoch 解决，并同时守住 decode ITL。
17. **未来预测模型偏差**：Phase 6 若引入预测，必须先用 trace 验证当前 chunk 与下一 chunk/decode 的相关性。
18. **Global Spare 架构成本**：当前 per-layer 固定 tensor 无法零成本跨层借 slot，不能在首版假装支持。
19. **TP 控制面开销**：首版为保正确性在关键阶段使用 CPU-group `all_gather_object` 校验完整 token。Top-N×多层可能累积明显延迟；必须 profile `tp_control_collective_ms`，后续改为固定 `int64` tensor token 并合并非关键阶段。
20. **当前 Chunk 重叠上限**：独立 writer 队列消除了 candidate 3 及以后等待主 CPU FIFO 的问题，但双 staging 仍只允许两个 candidate 同时在途；host export 速度、逐 candidate TP 共识和 GPU wave 2 吞吐可能形成新的 exposed tail。需要记录 writer start/end、slot refill、H2D 和 wave 2 时间，按实测限制有效 `N_stream`。
21. **进程内重建生命周期**：主 CPUInfer 与 writer CPUInfer 当前均为进程级 singleton，配置 key 不一致时 fail fast，线程在进程退出前保持存活。未来若支持同进程卸载并重建不同 CPU/NUMA 配置的 engine，必须增加“停止新 ticket → drain writer completion → 销毁 writer → 销毁 main”的显式 shutdown/reset 协议。

## 21. 实施检查清单

说明：以下清单是“代码 + 目标 GPU/TP 实机验收”的完成标准。首版代码已经覆盖其中的大部分正确性路径，但在 Linux CUDA、TP2、故障注入和 Nsight 性能验收完成前暂不批量勾选，避免把 CPU 单元测试等同于生产验收。

### 参数与状态

- [ ] 为 `kt_expert_placement_strategy` 增加 `decayed-lfu`。
- [ ] 新增 `--kt-prefill-stream-top-n`；默认 `None -> min(4, C)`，显式值校验 `0 <= N_stream <= C`。
- [ ] 除 stream Top-N 外不新增其他用户调优参数，内部默认值可观测但不可由 CLI 修改。
- [ ] 明确与 legacy dynamic、ratio、LoRA 的冲突。
- [ ] deferred-expert 模式未接入 assignment ownership 前，与 `decayed-lfu` 互斥。
- [ ] cache mode 下每个 cache-managed MoE 层严格分配 C 个 slots。
- [ ] 保留 `layer_idx < kt_num_gpu_layers` 的 native full-GPU bypass；这些层不创建 cache state、CPU backing 或 stream task。
- [ ] 只有 `layer_idx >= kt_num_gpu_layers` 的 KT-managed MoE 层严格分配 C 个 slots，并应用 `N_stream`。
- [ ] 启用 all-expert CPU pack。
- [ ] CPU packed backing 与 GPU slot residency 分离，Cache Manager 不管理 CPU slot。
- [ ] 启动前完成 host/GPU staging 与 CPU packed memory preflight。
- [ ] unsupported placement/backend/TBO 组合 fail fast。

### Policy

- [ ] 所有 route access 计数，hit/miss 分开。
- [ ] GPU 预分配统计 buffer。
- [ ] 实现归一化 EMA、过期、LRU tie-break。
- [ ] 实现 min residency、hysteresis 和 swap budget。
- [ ] 首版使用 Decayed-LFU reuse signal、hysteresis 和内部预算完成 admission。
- [ ] current Stream-TopN 以 token count 为主排序，reuse signal/ID 稳定 tie-break。
- [ ] hotset 已 resident 的项不占候选名额，不从第 `N_stream + 1` 名以后补位。
- [ ] unique active experts < `N_stream` 时不补零分专家。
- [ ] 单窗口绝不整组覆盖 C 个 slots。

### Loader

- [ ] 单 expert host writer。
- [ ] task-specific completion。
- [ ] writer 使用受限独立 WorkerPool/NUMA 资源或 compute-idle 调度。
- [ ] 按兼容签名分组的 staging pool。
- [ ] resident、host staging、GPU staging 与 assignment ownership 独立状态机，IN_USE slot 禁止覆盖。
- [ ] per-call transient CPU exclusion；stream candidates 不进入主 CPU task。
- [ ] Router 后 resident GPU wave 1 和 CPU-only task 不等待 candidate H2D。
- [ ] staging 达到 READY 后逐 candidate/小 ready group 启动 GPU wave 2；group size 不超过 staging pool depth。
- [ ] candidate 失败或超时时原子切换到 late CPU fallback。
- [ ] GPU/CPU ownership 互斥，输出 exact-once。
- [ ] streamed GPU 先写 private output，TP success consensus 后才 merge；不可恢复 CUDA fault fail-stop。
- [ ] 主 CPU-only task 与 late fallback 使用独立 buffer/ticket，不覆盖彼此。
- [ ] backend-specific single-slot prepare。
- [ ] backend-specific streamed-expert runner。
- [ ] TP candidate plan、ready/claim/install collective 使用统一全局顺序。
- [ ] TP host staging release consensus、generation 和失败回滚。

### Commit

- [ ] GPU wave 2 等待自己的 candidate-ready event，不等待 persistent publish。
- [ ] host staging 等待所有 TP ranks 的 H2D release consensus；GPU staging 等待可能存在的 wave 2 consumed event 和 install D2D event 后才可复用。
- [ ] 等待旧 slot last-use event。
- [ ] 获取 layer-exclusive lease 后才允许 destructive unpublished install；覆盖开始后禁止旧 generation 新 consumer。
- [ ] 在 safe boundary 提交。
- [ ] mask/mapping 原址更新。
- [ ] 所有 ranks 更新自己的 host mirrors；TP0 额外原址更新 CPU wrapper pinned mask。
- [ ] 更新 raw/prepared resident slot 和 raw canonical snapshot；MXFP4 同步更新 `_kt_mxfp4_raw_weights`。
- [ ] data_ptr/shape 保持稳定。
- [ ] 下一次 main/graph replay 等待 publish event。
- [ ] commit 覆盖后的部分失败执行 fail-stop。
- [ ] 首版 strict-freeze 下，active decode generation 期间不执行 writer/H2D/repack/commit，mapping generation 固定。

### 验证

- [ ] 离线 trace replay。
- [ ] policy/mapping 单元测试。
- [ ] backend 数值测试。
- [ ] TP 与 CUDA Graph 测试。
- [ ] 故障注入。
- [ ] profiler-off 性能 A/B。
- [ ] strict freeze 首版记录 decode active fraction、blocked prefill windows、generation age 和 update starvation。
- [ ] continuous workload 单独验证，不能用单请求 prefill/decode benchmark 代替。
- [ ] Global Spare 留到固定 per-layer 方案与 Phase 5 generation 协议稳定后，在 Phase 7 单独实施。
- [ ] 后续阶段再实现下一 Chunk 预测、candidate_saved/victim_loss 和 net expected benefit。

### Phase 5：Continuous batching

- [ ] prepare-only 与 destructive install/publish 权限分离。
- [ ] bounded maintenance epoch 作为过渡方案，并记录其 ITL 尖峰。
- [ ] batch 绑定不可变 `cache_generation_id`。
- [ ] 双 mapping/mask banks、generation-specific pinned CPU mask 和同层 shadow slots 全部预分配、地址稳定。
- [ ] 最多两个 active generations；禁止无界 generation/shadow 增长。
- [ ] generation-homogeneous batching；不支持 per-token generation 时禁止混代 fused batch。
- [ ] TP 原子发布、失败保留旧 generation。
- [ ] host refcount、GPU last-use event、CPU task 和 graph replay 全部完成后才回收旧 generation。
- [ ] 尾延迟、batch fragmentation、shadow VRAM 或 drain time 超预算时自动退回 strict freeze。

## 22. 最终推荐

首版应采用以下产品语义：

```bash
--kt-num-gpu-experts 32 \
--kt-expert-placement-strategy decayed-lfu \
--kt-prefill-stream-top-n 4
```

- `--kt-num-gpu-layers L` 的既有语义不变：`layer_idx < L` 的前置层继续 native/full GPU，不进入 KT cache。
- `--kt-num-gpu-experts` 只定义 `layer_idx >= L` 的后续 KT-managed MoE 层的固定物理 cache capacity。
- `--kt-prefill-stream-top-n` 也只作用于这些后续 cache-managed MoE 层，控制每层每个 prefill chunk 的 hotset 上限，并定义本窗口最多选择、流式执行的 missing 热点专家总数；实际 candidates 为 hotset 中的 non-resident experts，不向后补位，双 staging 只限制并发深度。
- `decayed-lfu` 根据当前及历史 prefill 路由维护长期热度；当前窗口 Stream-TopN 直接来自本次 Router 计数，首版不实现显式下一 Chunk 预测。
- 当前 active resident experts 立即进入 GPU wave 1；Stream-TopN 中缺失的 `0..N_stream` 个 experts 进入 streaming GPU wave 2；其余 non-resident experts 立即走 CPU-only。
- Router 后不等待 H2D 才启动已有计算；candidate 进入独立 staging，与同层 GPU wave 1、CPU-only 重叠，某个 candidate ready 后立即计算，不等待其余 candidates。
- hotset 中已经 resident 的专家不占候选名额，也不向后补齐；selected candidates 严格为按 Stream-TopN 原顺序过滤 resident 后的集合。
- stream candidate 从普通 CPU task 排除；失败或超时时必须 late CPU fallback，保证每个 assignment 恰好计算一次。
- 当前 chunk 计算使用私有 stream dispatch，persistent resident mapping 只在安全边界原址发布。
- prefill finalization 后 resident 集合冻结，decode 只消费该集合，不搬运专家权重。
- strict global freeze 是首版正确性策略，不是最终 continuous-serving 方案；持续 active decode 可能造成更新饥饿，Phase 5 必须引入 generation-aware prepare/publish 或有界 maintenance fallback。
- 未变化 resident slots 不搬运。
- full-layer streaming 保留为超大 prefill 的独立加速路径。
- 除 `--kt-prefill-stream-top-n` 外，首版不增加其他用户调优参数。

该方案把当前“按单次 prefill Top-N 整组重写”的策略，升级为“固定容量 resident cache + 当前 chunk 参数化 Stream-TopN 流式执行 + safe-boundary 增量提交 + decode 冻结消费”的 KT Expert Scheduler；显式下一 Chunk 预测和成本感知 admission 留到后续阶段。
