# KT GEC Prefill Staging 与跨层异步复制实现基线

状态：设计基线

适用范围：SGLang + KTransformers、TP=多卡、无 NVLink 的 PCIe 场景。

本文描述当前实现、参数契约、运行时行为和验收标准。

## 1. 目标与约束

prefill 阶段按 2048 token、top-k=6、每层 384 个专家计算时(第 16.2 节实测的 DeepSeek-V4.1-Flash:40 层、chunk=2048),每个专家平均被路由 32 次(2048 * 6 / 384),几乎所有专家都会参与计算。仅靠 resident pool 无法覆盖这类稠密路由,未进入 GPU 的专家必须走 CPU fallback。

当前实现使用跨层 staging：

```text
执行 L 层 compute
  + 并行预取 L+1 层
  + 并行执行 L 层的 CPU 专家和 GPU 专家
  + L+1 层执行后再更新自身历史命中表
```

目标是把 GPU 计算集合从 `resident(L)` 扩展为 `resident(L) + staged(L)`,并在 L 层计算期间完成 L+1 层的 host write、H2D 和 postprocess。

## 2. 目标与非目标

### 2.1 目标

- 在 L 层 compute 启动后立即开始 L+1 层 staging 预取。
- L+1 层使用自己的历史命中表选择预取专家。
- staging 容量独立于 `kt-expert-gpu-slots`。
- CPU 专家、GPU 专家、L+1 H2D 三条路径并行。
- staging 命中后,在安全边界更新 `kt-expert-gpu-slots` 常驻池。
- `kt-layer-h2d-slots` 达到该层专家数 N 时支持整层跨层 streaming(示例 N=256,实测模型 384)。
- 不对 decode 路径引入额外串行同步。

### 2.2 非目标

- 不改变模型结构。
- 不改变单层 MoE 的数学定义。
- 不把 staging 专家计入 `kt-expert-gpu-slots` 的逻辑容量。
- 不要求 prefill 和 decode 使用完全相同的预取深度。
- 不依赖未来层 router 来决定下一层权重搬运。

## 3. 术语

### 3.1 Pinned Layers

由 `--kt-num-gpu-layers N` 指定的前 N 层整层常驻 GPU。pinned layers 与专家级 resident 策略正交,可以和 `--kt-expert-gpu-slots` 或 `--kt-enable-dynamic-expert-update` 中的任意一种同时配置（见 3.2）。

这些层：

- 不需要 H2D。
- 不占 `--kt-expert-gpu-slots`。
- 不参与 staging。
- 不参与 GEC 的 LRU/LFU 常驻统计。
- 每层 MoE 默认全 GPU 执行。

注意：这里的 N 是层数,不是专家数。不要写成 `kt-expert-gpu-slots=10`(那是全局 10 个专家 slot)。

### 3.2 Resident Pool

由 `--kt-expert-gpu-slots` 控制的长期常驻专家池。

GPU 侧 resident 布局由两部分叠加而成：

```text
1. --kt-num-gpu-layers N              # 可选的 pinned layers 前缀,整层常驻
2. 专家级 resident 策略,二选一：
   --kt-expert-gpu-slots G            # GEC 常驻池
   --kt-enable-dynamic-expert-update  # legacy 阈值驱动 top-N 准入
```

- pinned layers 与专家级策略正交：`--kt-num-gpu-layers` 可以和两种专家级策略中的任意一种同时配置；pinned 层被整体排除出 KT 管理的 MoE 范围,不占 slot、不参与专家级准入。
- 两种专家级策略互斥：`--kt-expert-gpu-slots`（GEC）和 `--kt-enable-dynamic-expert-update`（legacy）是同一套专家级常驻准入的两种实现,不会同时配置。命令行当前不强制报错；若同时传入,专家选择走 GEC 分支,legacy top-N 不生效。

staging 与上述 resident 布局正交,只按当前生效的 resident 布局做去重和冻结。

逻辑键：

```text
(layer_id, expert_id)
```

职责：

- 保存 decode 和跨请求的长期热点专家。
- 支持 LRU、LFU、衰减命中计数和跨层容量重分配。
- 在安全边界被 staging 命中结果更新。

### 3.3 Staging Pool

用于跨层预取的独立显存池。

逻辑规模：

```text
logical_staging_rows = banks * kt-layer-h2d-slots
```

当前配置：

```text
banks = 2
kt-layer-h2d-slots = 256  # 整层预取
```

`--kt-layer-h2d-slots S` 是每个 bank 的逻辑 staging 窗口,也是当前层最多准入的专家数。staging 不占 `kt-expert-gpu-slots` 的逻辑容量,但必须单独计入 GPU 显存和 KV cache 预算。

只有 `S > 0` 且 `--kt-gpu-prefill-token-threshold` 已设置并被当前 extend batch 满足时,prefill-like forward 才会启用跨层 staging。两个参数联合生效：`kt-layer-h2d-slots` 选择特性并给出 staging 窗口,threshold 决定何时允许搬运；未配置 threshold 或 token 数未达阈值时,专家权重不做任何迁移。

staging 不要求 `--kt-expert-gpu-slots`：`--kt-layer-h2d-slots` 配一个正的
threshold 即可启用（命令行校验接受这两种来源之一）。resident 池为空时,
未被预取选中的专家全部走 CPU fallback。

#### 3.3.1 当前物理实现

当前代码中的 staging pool 是一个模型级双 bank 管理器,不是每层、每请求各一套。

- 每个 bank 的 raw/prepared 存储仍按整层专家数 `N` 分配。
- 两个 bank 共 `2 * N` 行 raw 和 `2 * N` 行 prepared 物理镜像。
- 每次预取只把 `top-S` 个非 resident 专家写入对应物理行。
- 当前层 GPU 计算集合是 `resident(L) + top-S staged(L)`,bank 中未被选中的行会被 mask,不参与 GPU 计算。
- 所有并发请求共享同一对 bank；bank 按 layer position 轮换。
- 因此 `S=64` 表示逻辑窗口为 64,不表示当前只分配 `2 * 64` 行 staging 显存。
- `S > N` 会被夹到 `N`。例如每层 256 个专家时,`S=385` 与 `S=256` 等价,不会获得超过整层的 GPU 计算集合。
- staging 模式下,启动时的 GEC resident 布局保持固定,不在 prefill 的层间执行容量迁移。稠密 prefill 中所有层几乎同样热,跨层借调会把 resident 行反复搬来搬去,形成与 staging 无关的第二条 H2D 流。
- 后续优化目标才是把物理 bank 收缩为 `2 * S`。

### 3.4 History Table

每个 MoE 层自己的专家命中历史：

```text
history[layer_id][expert_id]
```

用途：

- 预测下一轮或下一次执行时的热点专家。
- 选择 staging 预取候选。
- 作为提升到 resident pool 的依据。

L 层 router 只更新 L 层历史。给 L+1 预取时,默认使用 L+1 层自己的历史。

### 3.5 Effective GPU Set

某一层真正执行时,可以走 GPU 的专家集合：

```text
effective_gpu(L)
  = resident(L)
  + staged_hits(L)
```

其余专家：

```text
cpu_experts(L)
  = all_experts(L)
  - effective_gpu(L)
```

### 3.6 Consumed Event

表示某个 staging bank 的权重已经被当前层 compute stream 消费完成。

只有 consumed event 完成后,该 bank 才能被后续层复写。

## 4. 资源模型

当前实现把 GPU 专家资源拆成四部分：

```text
1. pinned layers
   --kt-num-gpu-layers 10

2. resident pool
   --kt-expert-gpu-slots 1024

3. staging pool
   2 个 bank
   每个 bank 按整层专家数 N 分配 raw 和 prepared 镜像
   不进入 resident 逻辑容量

4. temporary repack / scratch
   MXFP4 postprocess、Marlin prepared 中间缓冲
```

示例（第 16.2 节实测的 DeepSeek-V4.1-Flash:40 层、每层 N=384、逻辑窗口 S=128）：

```text
前 10 层（L0–L9,dense）：
    整层常驻 GPU,不经过 KT wrapper

L10–L39（30 层 MoE）：
    resident pool = 1024 个 slot（GEC 全局容量；启动时 L10 384、L11 384、L12 256）
    staging bank A = 384 行 raw + 384 行 prepared
    staging bank B = 384 行 raw + 384 行 prepared
    逻辑 staging 窗口 = 128
```

实测 bank 显存：每 bank/rank 约 7.32 GiB（raw 3.955 + prepared 3.36）,两个 bank 合计
14.63 GiB/rank,启动时从 KV 预算预留
（`Reserving 14.63 GB of the KV budget for lazy KT MXFP4 layerwise prefill slots`）。

resident pool 和 staging pool 是两套独立预算。当前 staging 显存不随 `kt-layer-h2d-slots` 等比例缩小,仍按 `2 * N` 估算。

## 5. 目标不变量

以下条件必须成立：

1. 所有槽位键都是 `(layer_id, expert_id)`。
2. Pinned layer 不进入 resident 或 staging。
3. Resident 和 staging 不能同时持有同一个 `(layer_id, expert_id)`。
4. staging 中的专家不计入 `kt-expert-gpu-slots`。
5. L 层执行时不能 evict L 层正在使用的 resident row。
6. staging bank 只能在 consumed event 完成后复用。
7. 给 L+1 的预取候选只能来自 L+1 层。
8. 默认不使用 L 层 router 结果选择 L+1 层专家。
9. CPU 和 GPU 的专家集合在 router 完成后一次性冻结。
10. TP 各 rank 必须使用同一份 staging selection 和 resident layout。
11. 任何 D2H 统计、排序和广播不得阻塞 compute stream。
12. 任何失败都必须能回退到普通 GEC 或 CPU 路径,不能死锁。

## 6. 总体架构

```text
                 +----------------------+
                 |  Layer L Router      |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Layer L compute      |
                 | resident + staged    |
                 +----------+-----------+
                            |
             +--------------+--------------+
             |                             |
             v                             v
  +------------------+          +----------------------+
  | Layer L GPU GEMM |          | Prefetch L+1         |
  | resident + stage |          | history[L+1] select  |
  +------------------+          +----------+-----------+
             |                             |
             v                             v
  +------------------+          +----------------------+
  | Layer L CPU GEMM |          | H2D -> staging bank  |
  | remaining experts|          | postprocess -> ready |
  +------------------+          +----------+-----------+
             |                             |
             +--------------+--------------+
                            |
                            v
                 +----------------------+
                 | Merge CPU + GPU      |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Queue history[L] D2H |
                 | side stream, async   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Layer L+1 start      |
                 | wait staged ready    |
                 +----------------------+
```

核心组件：

| 组件 | 职责 |
|---|---|
| HistoryTable | 维护每层专家命中、衰减分数和最近窗口 |
| ResidentCache | 管理 `kt-expert-gpu-slots` 长期常驻集合 |
| StagingPool | 管理双缓冲 raw/prepared bank 与按批轮换的 host slot ring |
| PrefetchPlanner | 在 L+1 层候选集合中选择预取专家 |
| LayerExecutor | 并行提交 CPU GEMM 和 GPU GEMM |
| MergeBarrier | 等待 CPU 与 GPU 两路完成并合并 |
| PromotionPolicy | staging 期间 resident 布局固定；显式 safe-boundary promotion 待完成 |
| Telemetry | 每层 timing、staging 命中与 CPU route 统计,每 16 层输出一轮 summary（debug 级） |

## 7. 双缓冲 staging 设计

### 7.1 Bank 划分

staging 使用两个 bank,按层顺序轮换：

```text
bank = layer_position % 2
```

每个 bank 都包含：

```text
GPU raw slot
GPU prepared slot
raw_ready_event
ready_event
consumed_event
```

当前物理布局：

- 两个 bank 都是一整层专家镜像,按 `layer_position % 2` 轮换。
- raw GPU 和 prepared GPU 存储分别按 `num_experts` 行分配。
- `--kt-layer-h2d-slots` 只限制本次选中并写入的行数,不改变物理行数。
- 当前层只计算 `resident(L) + top-S staged(L)`,其余行会被 mask。
- 所有并发请求共享同一对 bank,不是每请求一套。

代码入口：

- `_Mxfp4LayerwisePrefillManager` 持有 `slots[0]` 和 `slots[1]` 两个 bank；`_acquire`/`_prefetch_successor` 用 `1 - current_slot_index` 在两者间轮换,`_advance_round` 在轮次回绕时推进 epoch 并失效旧 bank。
- `_allocate_mxfp4_slot_storage` 为 bank 分配 raw 和 prepared 存储。
- `kt_prefill_staging.effective_experts` 生成 `resident + staged` 的 effective 集合,写入 `slot.effective_ids` 和 `slot.partial`。
- host slot 来自模型级共享 ring(第 7.5 节),不是每 bank 一个；每个 host slot 有自己的 `host_free_event`。

### 7.2 Bank 状态机

```text
EMPTY
  -> HOST_WRITE
  -> H2D
  -> POSTPROCESS
  -> READY
  -> COMPUTE
  -> CONSUMED
  -> REUSABLE
  -> EMPTY
```

异常状态：

```text
ERROR / POISONED
```

状态说明：

| 状态 | 含义 |
|---|---|
| EMPTY | bank 没有被占用 |
| HOST_WRITE | CPU 正在准备 pinned buffer |
| H2D | copy stream 正在搬 raw 权重 |
| POSTPROCESS | 正在生成 Marlin prepared 权重 |
| READY | compute stream 可以使用 |
| COMPUTE | 当前层正在消费 |
| CONSUMED | compute stream 已完成消费 |
| REUSABLE | 可以开始下一轮 host write |
| POISONED | 传输失败且不可安全复用 |

实现上的映射(第 7.1 节 `_Mxfp4PrefillSlot`)：

- `state` 只有 `EMPTY`、`LOADING`、`READY`、`IN_USE` 四个取值：HOST_WRITE/H2D/
  POSTPROCESS 都归在 `LOADING` 里,先后顺序由 stream 和 event 依赖保证；
  轮次回绕或中止时 `invalidate()` 退回 `EMPTY`。
- 复写保护记在 `reuse_guard` 上：`loading` 表示本代加载中,`raw`/`ready`/`consumed`
  分别表示上一代已过 raw fence/postprocess fence/已被 compute 消费；
  异常兜底是 `synchronized`(同步后再复用),不可复用是 `poisoned`。

### 7.3 事件契约

必须至少维护四类事件(`host_free_event` 挂在 host ring 的每个 slot 上,第 7.5 节)：

```text
host_free_event
raw_ready_event
ready_event
consumed_event
```

依赖关系：

- H2D 前等待该 host slot 的 `host_free_event`。
- postprocess 前等待 `raw_ready_event`。
- compute stream 在进入该层前等待 `ready_event`。
- 复用 bank 前等待 `consumed_event`。

### 7.4 Bank 复用规则

一个 bank 可以被复写的充要条件：

```text
bank.consumed_event 已完成
且没有活跃的 compute lease
且 copy stream 没有未完成的 H2D
```

双缓冲的正确性不依赖 Python 侧同步。
主要通过 CUDA event 和 stream 依赖保证顺序。

### 7.5 Host slot ring 与两代滚动

GPU staging 是双 bank,host pinned buffer 是两代滚动的 ring：ring 深度
`2 * --kt-layer-h2d-batch-size`(记作 `2 * k`),batch 序号决定使用哪一代。

```text
generation gen   = 0, 1
batch b 使用 generation b % 2 的 slot 区间 [gen * k, gen * k + k)
batch b 的 H2D 读第 b 代,batch b+1 写另一代 -> 两者重叠
```

一批的固定顺序：

```text
等待本批每个 slot 的 host_free_event
  -> 一次 TP 栅栏,广播本批 host slot 已释放(或上一批 H2D 入队失败)
  -> TP0 提交本批 k 个专家,只 drain 一次 KT writer
  -> 一次 TP 栅栏,广播本次 host 写入完成或失败
  -> 一次性 enqueue 本批 k 个专家的 H2D
  -> 逐 slot record host_free_event
```

- 每个 slot 有独立的 `host_free_event` 和 was_used 标记,异常路径逐 slot 复位,
  不会把未发布 fence 的 slot 交给下一批。
- 每层 KT drain 次数 = `ceil(S / k)`,TP 设备栅栏次数 = `2 * ceil(S / k)`
  (原来分别是 `S` 和 `2 * S`,见第 20 节 item 8)。
- 代内轮换的奇偶性跨层连续：L+1 的第一个 batch 写的是 L 最后一个 batch 的 H2D
  正在读的另一代,因此层边界同样不需要等一次 DMA。
- 写入与 H2D 重叠是必需的:两个方向搬的都是每专家 raw 字节,量级相同。旧的双槽实现
  靠 `position % 2` 交替得到同样的重叠,只是每专家多付一次 drain 和两次栅栏。
- 代价:pinned 内存 = `2 * k * per_expert_raw_bytes`(每 rank)。DeepSeek-V4.1-Flash、
  TP=2 时约 10.5MB/专家/rank；`--kt-layer-h2d-batch-size` 未设置时 layerwise 从
  `k=8` 起步(约 168MB),`k=2` 约 42MB,`k=1` 退化为 2 槽单专家滚动对。该 ring 属于
  模型级共享的 layerwise context,每个进程只分配一份,不随层数增长。

## 8. L+1 预取选择

### 8.1 候选集合

给 L+1 预取时,候选必须严格来自 L+1 层：

```text
candidates(L+1)
  = all_experts(L+1)
  - resident(L+1)
```

不能使用其他层的专家填充 L+1 staging。目标 bank 会在本次预取中被覆盖,因此不把上一代 staging 内容当作排除集合；resident 必须排除,避免同一 `(layer_id, expert_id)` 同时出现在 resident 和 staging。

### 8.2 默认排序依据

默认使用 L+1 层自己的历史命中表：

```text
prefetch_score(L+1, e) = history[L+1][e]
```

L 层 router 只更新 `history[L]`,不作为 L+1 的主排序依据。

### 8.3 当前分数

```text
score[L][e] =
    decayed_hit_count[L][e]
  + 0.25 * recent_window_hits[L][e]
  + 0.5 * last_router_demand[L][e]
```

当前实现：

- `decayed_hit_count` 是主项。
- `recent_window_hits` 用于快速跟随分布变化。
- `last_router_demand` 作为短时加权,不应压过长期历史。
- 定期对所有 layer 的 score 做衰减,避免旧热点永久占位。

### 8.4 冷启动

首次运行没有历史时,当前实现使用 `_cold_cursor` 按专家 id 分块轮转,选择确定的 top-S 候选,并立即启用 staging。不会随机选择,也不会因为冷启动关闭 staging。

### 8.5 预取数量

```text
selected = top_k(candidates(L+1), score, k=kt-layer-h2d-slots)
```

语义：

- `kt-layer-h2d-slots=64`：部分预取(decode 场景)。
- `kt-layer-h2d-slots=128`：实测的 prefill streaming 窗口(部分预取)。
- `kt-layer-h2d-slots=384`：整层预取(= 该层专家数 N；实测模型 N=384)。

如果 L+1 常驻池已经有 R 个专家,staging 预取 S 个,则理论上限覆盖为：

```text
min(N, R + S)          # N = 该层专家数,窗口 S 超过 N 时按 N 夹取
```

只有窗口覆盖整层(`S >= N - R`)时,覆盖率才能稳定接近 100 percent。

## 9. 执行时序

### 9.1 当前执行顺序

1. 确认 L 层权重 ready。
2. 提交 L 层 CPU fallback 和 GPU compute。
3. GPU compute 入队后,把 `history[L]` 的 D2H 放到 side stream。
4. 使用 `history[L+1]` 选择 L+1 候选,并提交 host write、H2D、postprocess。
5. L 层完成并 merge 后,L+1 等待自己的 staged ready。

关键点是 L+1 的 host write 和 H2D 不阻塞 L 层 compute 启动。

### 9.2 首个 KT 层示例

假设（第 16.2 节实测配置）：

```text
--kt-num-gpu-layers 10      # L0–L9 是 dense 层,不经过 KT wrapper
--kt-layer-h2d-slots 128    # 每 bank 的逻辑窗口(整层 = N = 384)
banks = 2
```

执行时序：

```text
L9（dense）：
    不经过 KT wrapper,与 staging 无关

L10（第一个 KT 层）：
    没有上一层可以做接力预取,同步 prime:host write、H2D、postprocess 后执行
    host write 按 k 分批 flush,drain 次数 = ceil(S / k)

L11：
    等待自己的 staged ready
    GPU = resident(11) + bank
    CPU = 其余专家
    另一个 bank 在 L11 计算期间预取 L12

L12：
    等待 bank ready
    GPU = resident(12) + bank
    CPU = 其余专家
    上一代 bank 只能在 L11 consumed 后复用
```

### 9.3 CPU 与 GPU 并行

L 层 router 完成以后,专家集合立即冻结：

```text
gpu_experts(L)
  = resident(L) + staged_hits(L)

cpu_experts(L)
  = all_experts(L) - gpu_experts(L)
```

随后并行执行：

```text
CPU GEMM for cpu_experts(L)
GPU GEMM for gpu_experts(L)
H2D for staging(L+1)
```

层耗时的理想模型：

```text
T_layer(L) ~= max(
    T_cpu_experts(L),
    T_gpu_experts(L),
    T_prefetch(L+1)
)
```

实际还要考虑共享资源竞争：

- CPU 内存带宽。
- PCIe 带宽。
- GPU copy engine。
- GPU SM 和显存带宽。
- MXFP4 postprocess 的额外显存。

## 10. 命中更新与 resident 提升（目标设计）

当前实现只做异步 history 更新。staging 激活期间,GEC resident pool 保持启动时的固定布局,不执行按层 row update；原因是稠密 prefill 没有可区分的层间热度,迁移只会制造额外 H2D。显式 safe-boundary promotion 尚未实现,本节描述待完成目标。

### 10.1 统计更新

L 层 router 完成后,可以异步更新：

```text
history[L][e] += demand
last_used[L][e] = now
recent_window[L][e] += demand
```

统计更新不改变物理 slot。

### 10.2 物理提升

物理提升必须延迟到安全边界。

安全边界定义为：

```text
当前层 compute 已完成
且相关 row refcount == 0
且没有 inflight H2D
且没有正在使用的 execution lease
```

提升流程：

```text
1. L+1 层 router 命中 staging 中的专家
2. 增加该专家在 resident score 中的权重
3. L+1 层 compute 完成
4. 选择低分且 idle 的 resident row
5. staging row 提升为 resident
6. 释放或重标记旧 row
```

### 10.3 提升方式

优先方案：

```text
统一 GPU row pool
resident 和 staging 只是 row 的逻辑状态
```

这样提升时只交换 ownership 元数据。

次优方案：

```text
staging tensor -> resident tensor 做 D2D copy
```

次优方案会增加一次 GPU D2D 拷贝。
如果 staging 和 resident 格式不同,还要额外 postprocess。

### 10.4 淘汰策略

不能被淘汰的 row：

- 当前层正在使用。
- 有 execution lease。
- 有 inflight dependency。
- 处于 LOADING 或 COMPUTE 状态。

候选淘汰顺序：

1. 低 `decayed_hit_count`。
2. 长时间未命中。
3. 没有 staging 命中记录。
4. 同层中可以走 CPU 的非关键专家。

### 10.5 去重

同一个 `(layer_id, expert_id)` 不能同时存在于：

- resident pool。
- staging bank A。
- staging bank B。

每次预取前和提升前都要检查。

## 11. 显存与 KV 预算

### 11.1 基本公式

当前实现的 staging 显存估算：

```text
current_staging_bytes =
    2 * num_experts_per_layer
  * (per_expert_raw_bytes + per_expert_prepared_bytes)
  + host_pinned_bytes
  + postprocess_scratch_bytes
```

其中：

- `num_experts_per_layer` 是每层专家数(本文示例 256；第 16.2 节基线实测的 DeepSeek-V4.1-Flash 为 384,见 `[kt-mask] num_experts=384`)。
- `host_pinned_bytes = 2 * k * per_expert_raw_bytes`(每 rank),`k` 就是 `--kt-layer-h2d-batch-size`(host slot ring 的单代深度,第 7.5 节),与 staging 窗口无关。
  DeepSeek-V4.1-Flash、TP=2 时约 10.5MB/专家/rank；layerwise 路径未设置时按 `k=8`(约 168MB),
  `k=2` 约 42MB。
- `postprocess_scratch_bytes` 是 Marlin repack 和临时缓冲。

未来如果实现物理 bank 收缩,公式才变为：

```text
future_staging_bytes =
    2 * kt-layer-h2d-slots
  * (per_expert_raw_bytes + per_expert_prepared_bytes)
  + host_pinned_bytes
  + postprocess_scratch_bytes
```

### 11.2 启动预留

启动时必须预留：

```text
pinned layers
+ resident pool
+ staging pool
+ postprocess scratch
+ KV cache
```

staging 不占 `kt-expert-gpu-slots` 的逻辑容量,但仍然占物理显存。

### 11.3 OOM 回退

初始化 staging 失败时：

1. 记录 staging disabled reason。
2. 回退到普通 GEC。
3. 或回退到纯 CPU/GPU hybrid。
4. 不得让请求永久挂起。

## 12. 参数语义与推荐配置

### 12.1 参数职责

| 参数 | 当前职责 |
|---|---|
| `--kt-num-gpu-layers 10` | 前 10 层整层常驻 GPU(实测 L0–L9 为 dense 层,不经过 KT wrapper),不参与搬运和 slot 统计 |
| `--kt-expert-gpu-slots 1024` | 动态常驻专家池(GEC 全局容量,实测值),服务 decode 和长期热点 |
| `--kt-num-gpu-experts 0` | 每层均匀分配的旧语义；与 `--kt-expert-gpu-slots` 同时给出时被覆盖(启动日志 `--kt-expert-gpu-slots=1024 overrides --kt-num-gpu-experts/--kt-gpu-experts-ratio`) |
| `--kt-layer-h2d-slots 64` | 每层部分预取,适合 decode 或轻量预取 |
| `--kt-layer-h2d-slots 128` | 实测 prefill streaming 窗口；bank 仍按整层 N 分配,窗口只决定预取集合与 CPU fallback 比例 |
| `--kt-layer-h2d-slots 384` | 每层整层预取(窗口 = 该层专家数 N,实测 N=384),适合 prefill streaming（仍需 threshold 门控） |
| `--kt-layer-h2d-batch-size 8` | H2D 复制粒度,同时决定 host slot ring 的单代深度 `k`(ring = `2 * k`,第 7.5 节)；不是 staging 容量。未设置时各路径 fallback 不同:layerwise host ring 按 `k=8`,GEC 准入批量按 1(每次准入一个专家),GEC CPU staging 深度按 2 |
| `--kt-gpu-prefill-token-threshold 2048` | 与 `--kt-layer-h2d-slots` 联合生效的启用门控；未设置或未达到阈值时不搬运 |

### 12.2 整层 Prefill 推荐配置

```text
--kt-num-gpu-layers 10
--kt-expert-gpu-slots 1024
--kt-num-gpu-experts 0      # 与 GEC 联用时被 kt-expert-gpu-slots 覆盖
--kt-layer-h2d-slots 128    # 实测窗口；整层窗口 = 该层专家数 N(384 专家的模型取 384)
--kt-layer-h2d-batch-size 8 或 16
```

`--kt-gpu-prefill-token-threshold` 与 `kt-layer-h2d-slots > 0` 联合生效：只有 extend batch 达到阈值时才启动 staging。未设置 threshold、或 token 数小于阈值时,不做任何 H2D 迁移,保持普通 GEC / CPU 路径。

语义：

```text
前 10 层（L0–L9,dense）：
    整层 GPU

L10–L39（30 层 MoE）：
    L 层计算时,预取 L+1 到 staging
    staging 与 resident 不共享逻辑容量

decode：
    使用 resident pool
    staging 可以关闭或降为 64
```

当前物理 bank 未缩容前,`S=64`、`S=128` 和 `S=384` 的 staging 显存相同(都按 `2 * N` 分配,实测 N=384 时 14.63 GiB/rank)；区别在预取集合、GPU 计算集合和 CPU fallback 比例。

### 12.3 Decode 推荐策略

```text
resident pool = 1024
staging = 0 或 64
selection = 当前 demand + decayed history
```

decode 路由稀疏,64 个 staging 专家有实际意义。
prefill 路由稠密,只有整层窗口(`S >= N - R`)才有整层意义。

### 12.4 不建议的配置

```text
kt-expert-gpu-slots=10
```

这不是前 10 层常驻 GPU,而是全局只有 10 个专家 slot。
前 10 层常驻应该使用：

```text
--kt-num-gpu-layers 10
```

### 12.5 Staging 与 Pinned Layers 的成本差异

两者不是等价的内存策略：

```text
kt-num-gpu-layers = L
    每层 N 个专家永久驻留(示例 256,实测模型 384)
    权重只在启动时传输一次
    prefill 每个 chunk 不产生 MoE 权重 H2D

kt-expert-gpu-slots = G
    全局 G 个逻辑专家永久驻留
    剩余专家由 staging 按层、按 chunk 反复搬运
    S=N(整层窗口)时,每个 prefill chunk 都会重传所有非 resident 层的整层专家
```

因此,`--kt-expert-gpu-slots 1024` 只有 1024 个永久专家行,而 `--kt-num-gpu-layers 18` 有 `18 * 256 = 4608` 个永久专家行(示例；实测模型按 384 专家算是 `18 * 384 = 6912`)。稠密 prefill 下,后者的 GPU 计算覆盖率和零重复 H2D 都明显更高。staging 的主要价值是覆盖 resident 之外的专家,而不是替代大量 pinned layers。

推荐顺序：

1. 先用 `--kt-num-gpu-layers` 固定能够长期容纳的完整层。
2. 再用 `--kt-expert-gpu-slots` 为剩余层维护 decode 热点和少量跨 chunk 常驻专家。
3. 最后用 `--kt-layer-h2d-slots` 做当前层的增量预取,窗口大小按 H2D 时间与 CPU fallback 时间的平衡点调优。

## 13. 模式划分与回退

### 13.1 Prefill Streaming Mode

进入条件：

```text
forward_mode 是 extend
且 kt-layer-h2d-slots > 0
且 token 数 >= kt-gpu-prefill-token-threshold
且 MXFP4 layerwise staging 可用
```

行为：

- 启动 L 层 compute。
- 同时预取 L+1 层。
- L 层 CPU 和 GPU 专家并行。
- L+1 层使用自己的历史命中表。
- staging 命中后异步更新 history；resident pool 在 staging 期间保持固定布局,显式 safe-boundary promotion 待完成。
- CPU-only 层(`num_gpu_experts == 0`)的 CPU 路由计数直接取 `topk_ids.numel()`,
  不逐层读回设备标量；门控路径已经得到的计数在 round 边界随 summary 汇总
  (§14.5、§15.1)。

### 13.2 Decode Resident Mode

行为：

- 使用 resident pool。
- 根据当前 router demand 和 history 做 admission。
- staging 可以为空。
- 不启动整层 H2D。
- 不在 decode 关键路径做全层专家集合的全量排序。

### 13.3 小 Batch Extend

显式设置 threshold 且 token 数小于阈值时：

- 不做任何 H2D 迁移,保持普通 GEC / CPU 路径。
- 如需让小 batch 也搬运,应调低 `--kt-gpu-prefill-token-threshold`,而不是只调大窗口。
- 不强行整层加载。

### 13.4 失败回退

| 场景 | 处理 |
|---|---|
| staging 初始化 OOM | 禁用 staging,回退普通 GEC |
| L+1 ready 超时 | 有限等待,然后 CPU fallback |
| bank consumed 未完成 | 禁止复写该 bank |
| TP rank 选择不一致 | 中止本轮 staging,回退 resident 路径 |
| 历史表为空 | 使用冷启动策略 |
| router 后处理异常 | 当前层走 CPU/GPU hybrid |
| postprocess 失败 | 标记 bank POISONED,禁止复用 |

不得使用无限等待。
所有等待必须有明确超时或可回退路径。

## 14. 正确性与并发契约

### 14.1 基本顺序

```text
router
  -> 冻结 CPU/GPU 专家集合
  -> 提交 CPU GEMM
  -> 提交 GPU GEMM
  -> 提交 L+1 staging H2D
  -> 等待 CPU 和 GPU 两路完成
  -> merge
  -> 更新历史表
  -> 安全边界更新 resident pool
```

### 14.2 引用计数与 Lease

以下操作必须持有 lease 或 refcount：

- GPU GEMM 读取 resident row。
- GPU GEMM 读取 staging row。
- H2D 写入 staging row。
- postprocess 读取 staging raw row。
- promotion 交换 row ownership。

没有 refcount 为 0 的确认,不得 evict 或复写 row。

### 14.3 TP 一致性

TP 多 rank 必须满足：

- staging selection 在每个 rank 上完全一致（bank id、layer id、expert id）。
- 当前实现不额外做每层 broadcast：resident mask 在启动时已由 rank 0 广播,
  每层历史表由相同的 router id 更新,两个输入一致时 `top_k` 的选择也一致
  （`Planner` 的排序按 `(-score, expert_id)` 确定性 tie-break）。
  如果将来需要 rank 0 权威裁决,放在 epoch 边界而不是每层。
- resident layout 通过 layout epoch 同步。
- 任一 rank 失败必须进入同一回退路径。

### 14.4 确定性

- 相同输入和相同 resident layout 必须产生相同 expert selection。
- stable row assignment 不应每轮重排。
- 同名 logical expert 必须稳定映射到同一物理 row,除非被 evict。

### 14.5 避免串行点

尽量避免：

```text
torch.cuda.synchronize()
counts.tolist() 在关键路径
每层 Gloo broadcast
per-layer object gather
per-layer 阻塞 CPU 写权重
.item() 读回设备标量(除非该值当层就要用于门控)
```

统计更新可以使用：

- pinned buffer 加 event。
- device counter。
- 延迟到下一层边界读取。
- `num_gpu_experts == 0` 时 `has_cpu_fallback` 和路由计数都由 `topk_ids.numel()`
  直接得出,不读回设备标量。
- 门控路径已经得到的路由计数直接 host 累加,随下一份 round summary 输出,
  不为遥测重复读回。

### 14.6 运行门控

当前实现由 ModelRunner 在每个 forward 发布 `forward_batch.forward_mode`,
`kt_ep_wrapper._kt_is_prefill_like()` 读取该发布值,并把 TARGET_VERIFY 视为
prefill-like；`is_extend_in_batch` 只作为发布值缺失时的兼容回退。
因此未启用 DP attention 或 MLP sync 的启动命令也能进入 prefill streaming
（`runtime_context.ForwardFlags` 没有 forward_mode 槽位,不能从 `get_forward()` 读取）。

需要在真机日志确认的门控字段：

```text
KT MXFP4 layerwise prefill: ... partial=True cpu=True/False
```

`partial=True` 表示 `kt-layer-h2d-slots` 没有覆盖整层,未覆盖专家进入 CPU
fallback；`partial=False` 表示整层预取。

## 15. Telemetry 与可观测性

当前已记录每层计数与轮次汇总：`candidate_count`、`selected_count`、
`resident_count`、`staging_hit_count`/`staging_miss_count`、
`cpu_expert_count`/`gpu_expert_count`、`cpu_route_count`、`copy_bytes`、
`copy_ms`、`postprocess_ms`、`bank_reuse_wait_ms`、`ready_wait_ms` 和
`compute_ms`；每 16 层输出一次 `[kt-prefetch] round ...`（debug 级）。
`cpu_route_count` 是该窗口内 staging partial 路径与 hybrid 门控路径的 CPU 路由
计数之和:门控路径在 `num_gpu_experts == 0` 时直接取 `topk_ids.numel()`,其余
情况复用当层门控本来就需要的读回值,两者都不新增设备同步（§14.5）。
每层 info 日志保留 `layer`、`epoch`、`slot`(bank)、`prime`/`prefetch-hit`、
`partial`、`cpu`、`staged`、`resident` 这些 gate 字段；`merge_ms`、
`promotion_count` 和 `eviction_count` 随显式 promotion 待实现。

### 15.1 每层必要字段

```text
layer_idx
layer_position
bank_id
candidate_count
selected_count
resident_count
staging_hit_count
staging_miss_count
cpu_expert_count
gpu_expert_count
cpu_route_count
copy_bytes
copy_ms
postprocess_ms
bank_reuse_wait_ms
ready_wait_ms
compute_ms
merge_ms
promotion_count
eviction_count
```

当前实现记录到 `compute_ms` 为止的字段；`layer_position`/`bank_id` 只体现在每层
info 日志的 `epoch`/`slot`；`merge_ms`、`promotion_count`、`eviction_count` 待补。
`cpu_route_count` 在门控路径由 host 侧已知值累加,不产生设备读回。

### 15.2 关键指标

| 指标 | 目标 |
|---|---|
| staging hit rate | 越高越好,prefill 整层应接近 100 percent |
| ready_wait_ms | 理想为 0,或远小于 compute_ms |
| H2D overlap ratio | L+1 copy 时间应被 L compute 覆盖 |
| CPU expert fraction | prefill 整层模式下应接近 0 |
| GPU expert fraction | prefill 整层模式下应接近 100 percent |
| promotion rate | decode 热点是否成功进入 resident |
| bank reuse wait | 应接近 0；它是 host write 等上一代 H2D 释放 ring slot 的时间,明显变大说明 H2D 比 host write 慢或 ring `k` 太小 |
| host write flush 次数/层 | 应为 `ceil(S / k)`,不随 S 线性增长 |
| per-layer 标量同步次数 | CPU-only 层(`num_gpu_experts == 0`)应为 0；其余层只保留 `has_cpu_fallback` 当层必需的那一次 |

### 15.3 日志建议

每 16 层或每轮汇总一次：

```text
[kt-prefetch] round layers=16 candidates=... selected=... resident=...
  staged_hits=... staged_misses=... cpu=... gpu=... cpu_routes=...
  copy_ms=... postprocess_ms=... bank_reuse_wait_ms=... ready_wait_ms=...
  compute_ms=...
```

debug 级记录每层细节和轮次汇总；每层 info 日志保留 gate 字段
（`partial`、`cpu`、`staged`、`resident`）和异常。
`cpu_routes` 的取值口径见 §15 开头：staging partial 的延迟标量加 hybrid 门控的
host 侧计数,两者都为零同步采集。
每层 trace 的 `cpu_route_classified` 带 `route_count_source`
(`synced`/`mask_empty`/`capture`),可确认是否跳过了标量读回。

## 16. 验收标准

### 16.1 正确性

- 输出与普通 GEC 或整层 GPU 路径一致。
- TP=2 下各 rank 无布局不一致。
- 无重复专家,无 row 泄漏。
- 无 bank 提前复用。
- 无死锁和无界等待。
- OOM 时可回退。

### 16.2 性能

在相同模型、相同 chunk、相同 TP 配置下：

```text
prefill tokens/s >= 900
```

至少应达到当前整层 GPU 路径水平,并希望优于它。

改动前的实测基线(2026-10-14,ARM 单机、TP=2、DeepSeek-V4.1-Flash:40 层、每层 384 专家、
top-k=6、`--kt-layer-h2d-slots 128`、`--kt-layer-h2d-batch-size 8`、chunk=2048)：

```text
input throughput      750 - 775 token/s(2048 token 的完整 chunk)
每层 staging_mxfp4    中位 132 - 194 ms(L10 首层约 80 ms),11 个 staging 层累计约 1.8s/chunk
每层 copy_ms          105 - 229 ms(round 均值；host write + H2D 入队 + 每批栅栏这一段)
bank_reuse_wait_ms    约 0.5
ready_wait_ms         约 7.5
```

该基线里每层对 128 个专家各做一次 KT drain 和两次 TP 栅栏,即第 20 节 item 8 的
每专家往返。按第 7.5 节改成按批(`ceil(S / k)` 次)并把 host ring 扩成两代滚动后需要
重新测量,目标仍是上面的 `prefill tokens/s >= 900`；复测时重点看 `bank_reuse_wait_ms`
是否仍接近 0(为 0 说明两代滚动足够,明显变大则说明 H2D 比 host write 慢)。

其他指标：

- L+1 ready 事件在 L 层 compute 完成前完成的比例大于 90 percent。
- staging hit rate 在 prefill 整层模式下接近 100 percent。
- CPU expert fraction 在 prefill 整层模式下接近 0。
- decode 吞吐不因 staging 逻辑明显下降。

### 16.3 显存

- staging 预留后进程不 OOM。
- KV cache 预算正确扣除 staging、raw、prepared 和 scratch。
- 长上下文请求下不出现碎片导致的不稳定分配。

### 16.4 可观测性

- 每次 staging 命中、miss、copy、wait 都可追踪。
- 可判断层耗时是 CPU 受限、GPU 受限还是 PCIe 受限。
- 可定位 bank 复用等待和 OOM 原因。
- 能区分门控计数是零同步取得的(`mask_empty`)还是逐层读回的(`synced`)。

## 17. 实施里程碑

### M0 数据契约

交付：

- 本文档。
- 术语和状态机评审结论。
- 参数语义冻结。

验收：

- `kt-num-gpu-layers`、`kt-expert-gpu-slots`、`kt-layer-h2d-slots` 的职责无歧义。

### M1 Staging Pool

交付：

- 双 bank raw、prepared 资源(host 侧为 `2 * k` 两代滚动 ring,第 7.5 节)。
- bank 状态机。
- ready、consumed、host_free 事件。

验收：

- 两个 bank 可连续轮换,无提前复写。
- bank 复用时没有活跃 lease。

### M2 History 与 Prefetch Planner

交付：

- 每层历史命中表。
- 衰减分数。
- L+1 候选过滤和 top-k 选择。
- 冷启动策略。

验收：

- 候选只来自目标层。
- resident 和 staging 去重正确。
- `--kt-layer-h2d-slots` 取 64(部分窗口)和整层 N(示例 256、实测模型 384)两种模式都能工作。

### M3 跨层异步复制

交付：

- L compute 启动后立即触发 L+1 预取。
- copy stream 和 compute stream 依赖。
- postprocess stream。

验收：

- L+1 ready 事件明显早于 L 层结束。
- host write 不阻塞 L compute 启动。

### M4 CPU/GPU 并行与 Merge

交付：

- L 层 CPU 和 GPU 专家集合划分。
- CPU 异步任务和 GPU kernel 并行提交。
- merge barrier。

验收：

- 日志显示 CPU 与 GPU 时间重叠。
- 不存在先 CPU 后 GPU 或先 GPU 后 CPU 的串行化。

### M5 Resident 提升

交付：

- staging 命中统计。
- 安全边界提升。
- 低分 resident 淘汰。

验收：

- decode 热点能进入 resident。
- 不 evict 正在使用的 row。
- 无重复 ownership。

### M6 Telemetry 与调优

交付：

- 每层和每轮指标。
- 参数扫描和结果表。
- 性能瓶颈归因。

验收：

- 能区分 host write、PCIe、postprocess、GPU GEMM、CPU GEMM 的耗时占比。

### M7 全量验证

交付：

- 正确性报告。
- 性能报告。
- 长稳和 OOM 回退报告。

验收：

- 达到第 16 节验收标准。

## 18. 测试计划

### 18.1 单元测试

- bank 状态机合法迁移。
- bank 提前复用必须报错。
- consumed event 未完成时禁止复写。
- 候选集合只包含目标层专家。
- resident 和 staging 去重。
- score 衰减和提升。
- TP rank selection 一致性。
- host write 分批边界:整除、尾批更短、每批不超过一代(`k`)、两代代间 slot 不重叠、空窗口与非法 `k` 退化。
- 传输循环契约:按批 drain 与栅栏,不存在 `position % 2` 的 2 槽假设。

### 18.2 集成测试

- TP=1 小模型。
- TP=2 真实配置。
- `kt-layer-h2d-slots=64`。
- `kt-layer-h2d-slots=256`。
- `kt-layer-h2d-slots=384`(实测模型整层窗口)。
- threshold 关闭和开启。
- 前 10 层 pinned、后续 30 层 normal streaming(实测配置)。
- OOM 回退。

### 18.3 性能矩阵

| 变量 | 值 |
|---|---|
| chunked-prefill-size | 1024、2048、4096 |
| h2d-slots | 64、128、256、384(实测模型整层) |
| h2d-batch-size | 2、4、8、16、32 |
| num-gpu-layers | 3、10、18 |
| expert-gpu-slots | 1024、2560 |
| threshold | 0（关闭,不搬运）、1024、2048 |

统一记录：

```text
prefill tokens/s
decode tokens/s
TTFT
GPU memory
KV cache tokens
H2D bytes
ready_wait_ms
CPU/GPU expert fraction
bank_reuse_wait_ms
host write flush 次数/层(应为 ceil(S / k))
pinned 内存占用(随 k 线性)
```

### 18.4 正确性回归

- 同 prompt 多次生成结果一致。
- 与整层 GPU 路径对比 logits 或生成结果。
- 长上下文不丢失。
- 多请求并发不串扰。

### 18.5 长稳测试

- 连续 prefill 数千轮。
- 交替 prefill 和 decode。
- 监控 bank 泄漏、refcount 泄漏和事件未完成。
- 观察显存是否缓慢增长。

## 19. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| Host write 阻塞 | L compute 启动延迟 | 先启动 compute,再做 host write |
| CPU 内存带宽竞争 | CPU GEMM 和 H2D 变慢 | 控制 batch size,测量带宽占比 |
| PCIe 成为瓶颈 | ready_wait 增加 | 提前一层预取,提升 staging 命中率 |
| MXFP4 postprocess 占用显存 | OOM | 单独统计 raw、prepared、scratch |
| 窗口覆盖不足 | CPU 仍是关键路径 | prefill 用整层窗口(`S >= N - R`) |
| 历史表冷启动 | staging 命中率低 | 按 expert 分块轮转,逐步建立历史 |
| router 同步链 | 无法重叠 | 异步更新,避免 `.tolist()`/`.item()` 和每层 Gloo；`num_gpu_experts == 0` 的层直接从 `topk_ids.numel()` 得到计数(§14.5) |
| prefill-like 门控未触发（forward_mode 未发布） | streaming 不生效 | 检查 `path_select` 的 `extend` 字段；ModelRunner 必须发布 forward_mode |
| threshold 未设置或未达到 | 不搬运专家权重 | 与 `--kt-layer-h2d-slots` 联合设置 threshold |
| TP layout 不一致 | 输出错误或死锁 | rank 0 决定并广播 layout |
| resident row 被误 evict | 数据竞争 | lease、refcount、consumed event |
| bank 提前复用 | 权重损坏 | consumed event 和状态机 |
| KV 预算未扣除 staging | 启动 OOM 或 KV 不足 | 启动阶段统一预算 |
| 多请求并发 | 历史和 staging 竞争 | 所有请求共享模型级双 bank,并在同一 forward 中合并 token 后逐层执行 |
| `--kt-layer-h2d-batch-size` 过大 | pinned 内存按 `2 * k` 线性增长 | 未设置时 layerwise 已按 `k=8`(约 168MB/rank),`k=2` 约 42MB/rank；继续加大(16/32)前先确认 memlock 与 SHM 余量 |

## 20. 开放问题

1. prefill 和 decode 是否需要不同的 `kt-layer-h2d-slots` 默认值。
2. staging bank 深度固定 2 是否足够,是否需要可配置 3 或 4。
3. promotion 是采用统一 row pool ownership swap,还是 D2D copy。
4. OOM 时先减小 staging,还是先减小 resident pool。
5. 物理 bank 收缩为 `2 * S` 后,row ownership 和 postprocess 子集如何实现。
6. TP rank 0 权威 selection 广播的实现方式。
7. pinned 层根本不经过 KT wrapper,首个 KT 层必然是同步. 如何解决?
8. submit→sync 改成 worker 线程/事件驱动。
   - 轻量版(批量化,不动线程模型)已实现:host slot ring = 两代滚动,
     每代 `k = --kt-layer-h2d-batch-size` 个 slot(第 7.5 节)。传输循环把
     "每专家 submit+sync" 改成 "提交 k 个专家 → 一次 sync → 一次性 enqueue k 个 H2D",
     相邻 batch 交替使用两代 slot,host write 与上一批 H2D 重叠；TP fence 采用
     "每 k 个一次",没有采用聚合成 pinned 标量挪到层边界再读的方案。
     代码入口:`kt_prefill_staging.host_write_batches` / `host_ring_depth` /
     `host_generation_slots`、`_Mxfp4LayerwisePrefillManager._load_slot` 与
     `_submit_host_write`(`submit` 已不自带 drain)、
     `_initialize_mxfp4_layerwise_pipeline` 传入的 `cpu_buffer_depth=host_ring_depth(k)`
     (参数未设置时 layerwise 按 `k=8`)。
   - 重量版(worker 线程/事件驱动,层内完全不 drain)未实现:KT 侧
     `cpu_backend/task_queue.cpp` 仍是单 FIFO worker 线程,每层仍有 `ceil(S / k)` 次
     `cpu_infer.sync()`。

9. 剩余逐层读回：`num_gpu_experts > 0` 的门控层仍每 call 读一次
   `(cpu_topk_ids >= 0).sum()`(`has_cpu_fallback` 当层就要用,不可延后)；
   `tp_rank != 0` 或无 `cpu_stream` 时该值只喂遥测,可进一步延后到层边界。

## 21. 结论

当前执行模型是：

```text
前 10 层（L0–L9,dense）：
    kt-num-gpu-layers=10,整层常驻 GPU
    不经过 KT wrapper,不参与 H2D、staging、resident 统计

L10–L39（30 层 MoE,实测配置）：
    L 层 compute 启动时,立即预取 L+1 层
    L+1 使用自己的历史命中表
    候选只来自 L+1 层
    排除 resident(L+1)
    staging 独立于 kt-expert-gpu-slots

执行层：
    GPU = resident + staged_hits
    CPU = 剩余专家
    门控计数:gpu_rows == 0 时取 topk_ids.numel(),不读回设备标量(§14.5)
    CPU GEMM、GPU GEMM、L+1 H2D 并行
    host write 按 k 分批 flush,drain 次数 = ceil(S / k)

命中后：
    统计异步更新
    staging 期间 resident pool 保持固定布局
    显式 safe-boundary promotion 待完成
```

关键参数：

```text
--kt-num-gpu-layers 10
--kt-expert-gpu-slots 1024
--kt-layer-h2d-slots 128    # 实测窗口；整层窗口 = 该层专家数 N(384 专家的模型取 384)
--kt-layer-h2d-batch-size 8 或 16
# host slot ring 深度 = 2 * --kt-layer-h2d-batch-size（未设置时 layerwise 按 k=8）
# 必需：--kt-gpu-prefill-token-threshold 2048（与 kt-layer-h2d-slots 联合门控）
```

一句话总结：

> 双缓冲 staging 负责跨层搬运,resident pool 负责长期常驻；L 层执行时预取 L+1,L+1 使用自己的历史命中表；staging 不占 resident 逻辑容量。当前物理 bank 仍按整层分配,显式 promotion 和 bank 缩容待完成。

