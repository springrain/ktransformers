# KT GEC Prefill Staging 与跨层异步复制实现基线

状态：prepared-only 单份 GPU 布局 + 统一 row pool / 边界 promotion + FreeToken 风格批量拷贝已落地，待 CUDA 13 双卡 GPU 实测

对齐说明：本文已按当前代码实现校准。早期设计稿中的"L 层 compute 启动后预取""每层 2 次 TP 栅栏""staging bank 的 lease/refcount"等表述与实现不一致；GEC resident row 另有自己的 lease/refcount（§14.2），不要把两者混为一谈。凡有冲突处以本文的当前实现描述和第 20 节为准。

适用范围：SGLang + KTransformers、TP=多卡、无 NVLink 的 PCIe 场景。

本文描述当前实现、参数契约、运行时行为和验收标准。

三阶段落地状态：

1. compact staging bank、uniform/shared-first resident 初始化、forward/chunk 边界更新已完成。
2. resident/staging 统一 `StagingRowPool`、row-pool metadata-only promotion(decode 行另有 D2D 镜像)、bank 覆盖后的 pending 提交已完成。
3. prepared-only 布局、模型级 raw H2D scratch、CPU canonical，以及 FreeToken 风格 fused D2D / `cudaMemcpyBatchAsync` 已完成；后者保留一次性能力探测和原有逐 expert/逐 run 回退。

## 1. 目标与约束

prefill 阶段按 2048 token、top-k=6、每层 384 个专家计算时(第 16.2 节实测的 DeepSeek-V4.1-Flash:40 层、chunk=2048),每个专家平均被路由 32 次(2048 * 6 / 384),几乎所有专家都会参与计算。仅靠 resident pool 无法覆盖这类稠密路由,未进入 GPU 的专家必须走 CPU fallback。

当前实现使用跨层 staging：

```text
L 层 router 完成后冻结专家集合
  -> full-window 路径在 GEMM(L) 入队前发起 L+1 预取
  -> partial 路径在 GPU GEMM 提交后、CPU join 前发起 L+1 预取
  -> L 层 CPU 专家和 GPU 专家并行
  -> merge 后更新自身历史命中表
```

目标是把 GPU 计算集合从 `resident(L)` 扩展为 `resident(L) + staged(L)`,并尽量让 L+1 层的 host write、H2D 和 postprocess 与 L 层已入队的 attention/GPU compute 重叠。注意 `_load_slot(L+1)` 的 host 控制面会阻塞 Python 线程,因此 GEMM(L) 的 launch 会发生在预取控制面返回之后。

## 2. 目标与非目标

### 2.1 目标

- 在 L 层 attention 之后发起 L+1 层 staging 预取；full-window 路径在 GEMM(L) 入队前,partial 路径在 GPU GEMM 提交后。
- L+1 层使用自己的历史命中表选择预取专家。
- staging 容量独立于 `kt-expert-gpu-slots`。
- CPU 专家、GPU 专家、L+1 H2D 尽量重叠；host 控制面会延迟 GEMM(L) 的 launch。
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
kt-layer-h2d-slots = 256  # N=256 模型的整层预取示例；实测模型 N=384 时取 384
```

`--kt-layer-h2d-slots S` 是每个 bank 的逻辑 staging 窗口,也是当前层最多准入的专家数。staging 不占 `kt-expert-gpu-slots` 的逻辑容量,但必须单独计入 GPU 显存和 KV cache 预算。

只有 `S > 0` 且 `--kt-gpu-prefill-token-threshold` 已设置并被当前 prefill-like batch 满足时,forward 才会启用跨层 staging。两个参数联合生效：`kt-layer-h2d-slots` 选择特性并给出 staging 窗口,threshold 决定何时允许搬运；未配置 threshold 或 token 数未达阈值时,专家权重不做任何迁移。

staging 不要求 `--kt-expert-gpu-slots`：`--kt-layer-h2d-slots` 配一个正的
threshold 即可启用（命令行校验接受这两种来源之一）。resident 池为空时,
未被预取选中的专家全部走 CPU fallback。

参数校验注意：命令行只要显式提供 `--kt-layer-h2d-slots`（包括 0）且未提供
`--kt-expert-gpu-slots`,就要求 `--kt-gpu-prefill-token-threshold` 为正；
运行期 `S=0` 仍会被 `clamp_staging_window` 视为关闭 staging。

#### 3.3.1 当前物理实现

当前代码中的 staging pool 是一个模型级双 bank 管理器,不是每层、每请求各一套。

- staging 选中后,每个 bank 的 GPU prepared pool 按 compact 几何 `R_max + S` 分配,raw 只保留 `S` 行 H2D scratch(prepared-only 默认；`SGLANG_KT_MXFP4_PREPARED_ONLY=0` 的 legacy 回退为 `R_max + S` 行 raw)。`R_max` 是所有 MoE 层中最大的生效 resident 数,`S` 是 `--kt-layer-h2d-slots` 窗口(夹到 N)。默认两个 bank 合计 `2 * (R_max + S)` 行 prepared + `2 * S` 行 raw scratch,不再是每 bank 一份 `R_max + S` 的 raw+prepared 双份。
- prepared-only slot 布局默认随 `--kt-layer-h2d-slots > 0` 启用；对 `num_gpu_experts > 0` 且走 Marlin 后处理的层,checkpoint-native raw 张量会在 prepared 构建后尝试移到 CPU（迁移成功时才只保留 kernel-ready prepared 行；未执行时 raw 仍可能留在 GPU，逐层迁移异常则向上传播，见 §11.1）。诊断时可用 `SGLANG_KT_MXFP4_PREPARED_ONLY=0` 回退旧的 raw+prepared 双 bank 布局。
- 启动时 GEC 全局 resident 预算均匀分布到每个 MoE 层,余数放在前若干层；fused shared experts 处于 trailing logical ids,并在每层 resident 选择中优先保留。这样每层都有 resident floor,同时不会把预算集中到前几层。
- 对 `num_gpu_experts > 0` 的 prepared-only 层,完整 N 行 raw canonical 在 Marlin 后处理后会尝试移到 CPU,作为 resident D2D/GEC/fallback 的权威字节源；只有迁移成功时才以 CPU 为权威副本。`num_gpu_experts == 0` 不进入该后处理分支,raw canonical 仍留在 GPU（该层没有 resident raw 行可复用）；迁移未执行时不能假定 raw 已离开 GPU，逐层迁移异常会向上传播（见 §11.1）。compact canonical 保持 resident row 顺序,并按 victim row 原地刷新。bank H2D 使用每个 bank 自己的 raw scratch；GEC/resident selected-row 更新使用 shared context shadow layer 上缓存的 raw scratch,不会为每个 decoder 层复制一份。
- 每个 bank 用 `StagingRowPool` 统一管理 resident/staging 物理 row。同一 `(layer, expert)` 只占一行,resident 与 staging 只是 row role；promotion 时 promoted row 原地升级为 resident,victim row 置为 free 并清空 mapping,不会把 victim 转成 staging。
- 每次预取只把 `top-S` 个非 resident 专家写入 compact 物理行。当前层 GPU 计算集合仍是 `resident(L) + top-S staged(L)`,未覆盖专家走 CPU fallback。
- 所有并发请求共享同一对 bank；bank 按 `layer_position % 2` 固定轮换。同一层跨 forward 仍使用同一 bank,但 bank 被其他层覆盖后 `_load_slot` 会调用 `pool.clear()`,per-layer mapping 不跨覆盖保留；此时 promotion 走 pending 路径。
- resident 只在 chunk/request 边界更新:forward id 不变时冻结 resident ownership；检测到新的 forward id 后,用上一 chunk 的 staged hit 选择 promotion。
- 边界时若 bank 仍持有该层,row pool 层面只改 owner/role/mapping；随后 `_refresh_original_layer_rows` 仍会把 promoted row D2D 镜像到原层 GEC/decode 行(prepared-only 下只 prepared),所以"metadata-only"指 row ownership,不是零拷贝。若 bank 已被其他层覆盖,决策进入 `_pending_boundary_promotions`,在下次加载该层时走 pinned host write + H2D 提交,不会让 decode 读到已覆盖的 bank 字节。
- 拷贝提交按 FreeToken 风格折叠：resident D2D 优先用一次 JIT multi-row kernel 覆盖所有 prepared bank（W13 同时完成 gate/up half swap）；H2D 优先用一次 `cudaMemcpyBatchAsync` 覆盖整代所有独立 span。JIT 只在首次加载时探测，运行时失败或布局不满足时回退到原逐 expert/逐 run 路径；`SGLANG_KT_DISABLE_PREFILL_FAST_COPY=1` 可强制回退。
- `S > N` 会被夹到 `N`；实际 staged 数还受 `N - |resident(L)|` 限制。`S` 未设置时关闭跨层 staging,继续使用历史 full-row path。
- TP=2/多卡：每个 rank 只解析并操作自己 GPU 上的 layer/prepared 张量（`_kt_layer_cuda_device`），不共享 CUDA tensor；GEC topology 携带 `kt_tp_size`，各 rank 从相同 resident 布局和确定性 history/router 输入独立推导同一 logical resident/staging selection,每层不做 selection 广播或比较。`--kt-expert-gpu-slots` 是每 rank 的 logical slot 总数，每个 logical slot 在每个 rank 上各占一行 TP 分片；不会把两个 rank 的权重塞进同一张卡。

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

只有该 bank 对应 fence 完成后才能被后续层复写：正常 compute 路径是 `consumed_event`；若该代未被 compute 消费,则是 `ready_event` 或 `raw_ready_event`。

## 4. 资源模型

当前实现把 GPU 专家资源拆成四部分：

```text
1. pinned layers
   --kt-num-gpu-layers 10

2. resident pool
   --kt-expert-gpu-slots 1024
   每层 resident 的 kernel-ready prepared 行；raw canonical 默认在 CPU（有 resident GPU 行且 prepared-only 移动成功时）

3. staging pool
   2 个 bank
   每个 bank 分配 R_max + S 行 prepared pool
   每个 bank 分配 S 行 raw H2D scratch(prepared-only；legacy 回退为 R_max+S 行)
   默认不再保留 resident raw bank 或完整 N 行 GPU raw canonical；num_gpu_experts==0、迁移未执行或 full-context 迁移失败时仍可能保留（见 3.3.1）
   不进入 resident 逻辑容量

4. temporary repack / scratch
   shared context raw scratch（GEC/resident selected-row 更新复用一份，按 min(窗口, resident 批量) 限宽）
   首次初始化时临时 full-context raw N 行；bank 建好后尝试移到 CPU（失败时保留在 GPU）
   MXFP4 postprocess 直接写入 prepared pool，不再分配 prepared scratch bank
```

示例（第 16.2 节实测的 DeepSeek-V4.1-Flash:40 层、每层 N=384、逻辑窗口 S=128）：

```text
前 10 层（L0–L9,dense）：
    整层常驻 GPU,不经过 KT wrapper

L10–L39（30 层 MoE）：
    resident pool = 1024 个 slot（GEC 全局容量）
    均匀分布后每层 34 或 35 个 resident,R_max = 35
    staging bank A = 35 + 128 = 163 行 prepared + 128 行 raw scratch
    staging bank B = 35 + 128 = 163 行 prepared + 128 行 raw scratch
    CPU raw canonical = 384 行（prepared-only 移动成功时；不占 GPU）
    逻辑 staging 窗口 S = 128
```

按 `get_v4_mxfp4_marlin_storage_nbytes()` 和
`_mxfp4_raw_slot_storage_nbytes()` 用模型实际的 hidden/intermediate 计算：
两个 bank 的 prepared pool 随 `2 * (R_max + S)` 缩放,raw scratch 随 `2 * S`
缩放；resident pool 的 1024 个 logical slot 按同一 per-expert prepared 大小
另算。首次初始化还会临时分配 full-context raw N 行,bank 建好后尝试移到 CPU（失败时保留在 GPU）。
旧稿的 2500-slot / 21.9 GiB 是另一组逻辑容量,已废弃；具体字节以启动日志
`Reserving ... of the KV budget for lazy KT MXFP4 layerwise prefill slots` 为准。

resident pool 和 staging pool 仍是两套独立预算。staging 显存现在按
`2 * (R_max + S)` 行 prepared + `2 * S` 行 raw scratch 缩放,不再固定按
`2 * N` 估算,也不会为 35 个 MoE 层各保留一份 raw scratch。TP=2 时两张卡
各有一份相同的 logical 布局，但每张卡只存自己那份 TP 权重分片。

## 5. 目标不变量

以下条件必须成立：

1. 所有槽位键都是 `(layer_id, expert_id)`。
2. Pinned layer 不进入 resident 或 staging。
3. Resident 和 staging 不能同时持有同一个 `(layer_id, expert_id)`。
4. staging 中的专家不计入 `kt-expert-gpu-slots`。
5. L 层执行时不能 evict L 层正在使用的 resident row。
6. staging bank 只能在对应 consumed/ready/raw fence 完成后复用。
7. 给 L+1 的预取候选只能来自 L+1 层。
8. 默认不使用 L 层 router 结果选择 L+1 层专家。
9. CPU 和 GPU 的专家集合在 router 完成后一次性冻结。
10. TP 各 rank 必须推导出同一份 staging selection 和 resident layout；当前实现不额外广播或比较每层 selection。
11. 不得在 compute stream 上插入同步；host 侧 TP 控制面可以阻塞 launch,但热路径不得 `stream.synchronize()`。
12. 失败通过 TP 共识在同一 phase 抛出；当前实现没有超时回退,依赖已入队 CUDA event 最终完成,不得等待未发布的 fence。

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
| StagingPool | 管理双缓冲 prepared pool、raw H2D scratch、统一 row pool 与按批轮换的 host slot ring |
| PrefetchPlanner | 在 L+1 层候选集合中选择预取专家 |
| LayerExecutor | 并行提交 CPU GEMM 和 GPU GEMM |
| MergeBarrier | 等待 CPU 与 GPU 两路完成并合并 |
| PromotionPolicy | resident 在 chunk 内冻结；forward/chunk 边界做 row-pool metadata-only promotion,并把 promoted row D2D 镜像到原层 decode 行；被覆盖 bank 延迟到下次加载提交 |
| Telemetry | 每层 timing、staging 命中与 CPU route 统计,每 16 层输出一轮 summary（debug 级） |

## 7. 双缓冲 staging 设计

### 7.1 Bank 划分

staging 使用两个 bank,按层顺序轮换：

```text
bank = layer_position % 2
```

每个 bank 都包含：

```text
GPU prepared pool（R_max + S 行）
GPU raw H2D scratch（S 行；legacy 回退为 R_max + S 行）
raw_ready_event
ready_event
consumed_event
```

当前物理布局：

- 两个 bank 都按 `R_max + S` 行 prepared pool 分配；prepared-only 默认每个 bank 另配 `S` 行 raw H2D scratch,`SGLANG_KT_MXFP4_PREPARED_ONLY=0` 的 legacy 回退为 `R_max + S` 行。bank 按 `layer_position % 2` 轮换。
- GPU 不再保留 resident raw bank；有 resident GPU 行的 prepared-only 层在移动成功后也不再保留完整 N 行 raw canonical（raw canonical 在 CPU，GPU raw 只作为一次性 H2D scratch）。`num_gpu_experts == 0` 或迁移未执行时 raw canonical 仍可能留在 GPU；逐层迁移异常没有本地 catch，会向上传播（见 §11.1）。
- `StagingRowPool` 把 resident 和 staging 放进同一个物理 row 池。每个 row 只有一个 owner,role 为 `resident`、`staging` 或 `free`；promoted row 保留原 row 升级为 resident,victim row 变 free。`_load_slot` 在 bank 换层时调用 `pool.clear()`,per-layer mapping 不跨 bank 覆盖保留。
- 当前层只计算 `resident(L) + top-S staged(L)`,其余 compact 行会被 mask。
- 所有并发请求共享同一对 bank,不是每请求一套。

代码入口：

- `_Mxfp4LayerwisePrefillManager` 持有 `slots[0]` 和 `slots[1]` 两个 bank；`_acquire` 按 `layer_position % 2` 固定 bank,`_prefetch_successor` 预取下一层所在的另一 bank,`_advance_round` 在轮次回绕时推进 epoch 并失效旧 bank。
- `_allocate_mxfp4_slot_storage` 为 bank 分配 `R_max + S` 行 prepared pool；raw 行数在 prepared-only 下是 `S`,在 legacy 回退下是 `R_max + S`。首次初始化临时 full-context raw 在 manager 建立后尝试移到 CPU（失败时保留在 GPU）。
- `kt_prefill_staging.StagingRowPool` 维护 per-layer owner/role/mapping；`_load_slot` 通过 `configure` 建立 mapping,promotion 通过 `promote` 原地改变角色；bank 换层时 mapping 会被清空。
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

当前实现维护以下事件：

```text
host_free_event
raw_ready_event
ready_event
consumed_event
```

- `host_free_event` 挂在模型级 host ring 的每个 slot 上(第 7.5 节)。
- `raw_ready_event`、`ready_event`、`consumed_event` 挂在每个 bank slot 上。

依赖关系：

- host slot 复用前,copy stream 等待该 slot 的 `host_free_event`；TP0 的 host write 在随后的 TP 共识之后提交。
- postprocess 前等待 `raw_ready_event`。
- compute stream 在进入该层前等待 `ready_event`。
- 复用 bank 前,transfer stream 按上一代最后状态等待 `consumed_event`、`ready_event` 或 `raw_ready_event`。

`host_slot_was_used` 目前只写不读：H2D 的 `finally` 会为本次 `batch_slots` 无条件 record `host_free_event`；若 event record 本身失败,代码会 `transfer_stream.synchronize()` 作为异常兜底。不要把它理解为逐 slot 的 was_used 门控。

### 7.4 Bank 复用规则

staging bank 当前没有独立的 lease/refcount 对象,也不会检查显式 lease 计数。实际复用门控是 `slot.reuse_guard` 与 event 依赖：

- `consumed`：上一代 compute 已消费,transfer stream 等 `consumed_event`；
- `ready`：上一代 postprocess 已发布 ready,transfer stream 等 `ready_event`；
- `raw`：上一代 H2D 已过 raw fence,transfer stream 等 `raw_ready_event`；
- `synchronized`：异常路径已同步 copy stream；
- `poisoned`：无法再提供复写 fence,禁止复用。

host slot 另由 `host_slot_free_events` 和一次 NCCL 设备共识串行化。双缓冲的正确性依赖这些 stream/event 依赖,不依赖 Python 侧同步。

### 7.5 Host slot ring 与两代滚动

GPU staging 是双 bank,host pinned buffer 是两代滚动的 ring：ring 深度
`2 * --kt-layer-h2d-slots`(记作 `2 * S`),层序号决定使用哪一代。
一代就是整个 staging 窗口 `S` 本身,与 FreeToken 的 prefill 形状同构
(`_stage_bytes[k] = num_experts * max(rows)` 按 `layer_id % 2` 乒乓),
没有层内子分批、也没有独立的粒度旋钮。

```text
generation gen   = 0, 1
layer L 使用 generation (层内唯一一次写入) 的代区间 [gen * S, gen * S + |staged(L)| + |resident_host_ids(L)|)
layer L 的 H2D 读第 gen 代,layer L+1 写另一代 -> 两者重叠
```

一层的固定顺序：

```text
setup 的 Gloo all-reduce(选择/布局)
  -> transfer stream 等上一代 reuse guard,提交 GPU D2D
  -> 等本代每个 host slot 的 host_free_event(copy stream 依赖)
  -> host-slot reuse 的 NCCL 设备共识
  -> TP0 提交本代 |staged(L)| + |resident_host_ids(L)| 个 host 行(pending promotion 共用本代),有传输时只 drain 一次 KT writer
  -> host-write Gloo all-reduce
  -> host-write publish(每层一次 Gloo broadcast)
  -> 一次性 enqueue 本代 |staged(L)| + |resident_host_ids(L)| 个专家的 H2D
  -> 逐 slot record host_free_event
  -> raw-ready / postprocess 的 Gloo 控制点
```

- `host_slot_was_used` 只写不读,不是门控；异常时若 event record 失败,代码同步 transfer stream 作为兜底(见 7.3)。
- 有实际 host transport(`staged + resident_host_ids` 非空)的 reload 层 KT writer drain 次数为 `1`；无 transport 的 load 记 `0`,warm reuse 不进入 host-write pass。控制面不是 2 次栅栏：需要 `_load_slot` 的正常路径在 TP>1 时至少有 5 次 Gloo all-reduce(setup、GPU D2D、host write、raw-ready、postprocess)、1 次 NCCL all-reduce + `.item()`(host-slot reuse)和 1 次 Gloo broadcast(host-write publish)；warm reuse 分支跳过这些 collective。
- 代内轮换的奇偶性跨层连续：L+1 写的是 L 的 H2D 正在读的另一代,因此层边界
  同样不需要等一次 DMA。
- 写入与 H2D 重叠是必需的:两个方向搬的都是每专家 raw 字节,量级相同。旧的双槽实现
  靠 `position % 2` 交替得到同样的重叠,只是每专家多付一次 drain 和两次栅栏。
- 代价:pinned 内存 = `2 * S * per_expert_raw_bytes`(每 rank),对窗口线性。
  `--kt-layer-h2d-slots` 是 opt-in(`None` 或 `S<=0` 即整个功能关闭),未设置时 ring 深度
  退回历史 2 槽对,默认路径的 pinned 占用完全不变。该 ring 属于模型级共享的
  layerwise context,每个进程只分配一份,不随层数增长。
  **启动时 `_layerwise_host_ring_depth` 会打印 `ring_depth` 与 `S`,任何大 `S`
  都必须是操作者看得见的决定,而不是静默分配。**

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

首次运行没有历史时,当前实现使用全局 `_cold_cursor` 跨层按专家 id 分块轮转,选择确定的 top-S 候选,并立即启用 staging。不会随机选择,也不会因为冷启动关闭 staging。

### 8.5 预取数量

```text
selected = top_k(candidates(L+1), score, k=kt-layer-h2d-slots)
```

当本层存在 pending promotion 时,promoted host 行与本代 staged candidates 共用同一个 host generation,因此实际 staged 数还会被收紧到 `S - |resident_host_ids|`。

语义：

- `kt-layer-h2d-slots=64`：部分预取(prefill 轻量窗口;decode 不启用跨层 staging)。
- `kt-layer-h2d-slots=128`：实测的 prefill streaming 窗口(部分预取)。
- `kt-layer-h2d-slots=384`：整层预取(= 该层专家数 N；实测模型 N=384)。

如果 L+1 常驻池已经有 R 个专家,staging 预取 S 个,则理论上限覆盖为：

```text
min(N, R + S)          # N = 该层专家数,窗口 S 超过 N 时按 N 夹取
```

只有窗口覆盖整层(`S >= N - R`)时,覆盖率才能稳定接近 100 percent。

## 9. 执行时序

### 9.1 当前执行顺序

1. 刷新上一层 history demand；检测到新的 forward id 时执行边界 promotion。
2. `_acquire(L)` 取得 L 层 bank；若 L 尚未预取,这里同步完成 host write、H2D 和 postprocess。
3. full-window 路径在 `_prefetch_early` 中同步发起 L+1 的 host write、H2D、postprocess；此时 attention(L) 已入队,但 GEMM(L) 尚未入队。
4. 等待 L 的 `ready_event`,提交 CPU fallback 或 GPU compute。
5. partial 路径在 GPU compute 提交后、CPU join 前调用 `_prefetch_successor` 预取 L+1。
6. L 层完成并 merge 后,把 `history[L]` 的 D2H 放到 side stream；L+1 等待自己的 staged ready。

关键点：L+1 的 host write 和 H2D 不会阻塞已经在 GPU 上执行的 attention(L),但 `_load_slot(L+1)` 的 host 控制面会阻塞 Python 线程,因此 GEMM(L) 的 launch 会被推迟到预取控制面返回之后。

### 9.2 首个 KT 层示例

假设（第 16.2 节实测配置）：

```text
--kt-num-gpu-layers 10      # L0–L9 是 dense 层,不经过 KT wrapper
--kt-layer-h2d-slots 128    # 每 bank 的逻辑窗口；整层预取需 384
banks = 2
```

执行时序：

```text
L9（dense）：
    不经过 KT wrapper,与 staging 无关

L10（第一个 KT 层）：
    没有上一层可以做接力预取,同步 prime:host write、H2D、postprocess 后执行
    host write 一次搬完本代窗口（≤S）；有实际传输时 drain 次数 = 1

L11：
    等待自己的 staged ready
    GPU = resident(11) + bank
    CPU = 其余专家
    另一个 bank 在 L11 的 GEMM 入队前发起 L12 预取,H2D 与 L11 compute 重叠

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

full-window 路径的执行顺序：

```text
attention(L) 已入队
  -> _prefetch_early(L): _load_slot(L+1) 完成 TP 控制面、TP0 host write、H2D/postprocess 入队
  -> wait_event(ready(L))
  -> CPU fallback / GPU GEMM(L) 入队
```

因此 H2D(L+1) 可以在 GPU 上覆盖 attention(L) 和后续的 GEMM(L),但 Python 线程在预取控制面返回前不会提交 GEMM(L)；这是"更早入队"而不是 CPU 线程级并发。该排序是唯一实现,没有环境开关。

partial 路径改为先提交 CPU fallback 与 GPU GEMM,再在 `_apply_effective_set` 中预取 L+1,最后 join CPU。

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

## 10. 命中更新与 resident 提升

staging 命中在层内只更新 history 和 hit 计数,不改 owner。一个 forward/chunk 内
GEC resident 布局保持固定；检测到新的 `forward_pass_id` 后,才用上一 chunk 的
staged hit 选择 promotion。这样 resident 只会在 chunk/request 边界变化。

### 10.1 统计更新

L 层 router 完成后,可以异步更新：

```text
history[L][e] += demand
last_used[L][e] = now
recent_window[L][e] += demand
```

统计更新不改变物理 slot。

### 10.2 物理提升

物理提升必须延迟到 forward/chunk 边界。上一轮所有层已经完成,因此满足：

```text
上一 chunk 的所有层 compute 已完成
且 pending demand D2H 已归档
且 bank 的 consumed/ready/raw fence 已满足
```

当前实现流程：

```text
1. 每个 staged 专家累计本 chunk 的路由命中
2. 新 forward id 到来时,按 (-hits, -score, id) 选 staged winners
3. 按 (score, id) 选最冷且非 protected 的 resident victims
4. promoted row 原地升级为 resident；victim row 置为 free 并清空 victim mapping,不分配新 row
5. 原层 GEC/decode 行从 promoted row 刷新(prepared-only 只刷新 prepared),更新 mask/logical_to_row
6. 若 bank 已被其他层覆盖,记录 pending,下次加载该层时用 host H2D 提交
```

### 10.3 提升方式

当前方案：

```text
统一 GPU row pool
resident 和 staging 只是 row 的逻辑状态
```

`StagingRowPool.configure` 让同一专家在 resident/staging 间切换时保留物理 row,
`promote` 把 promoted row 的 role 改为 resident,把 victim row 的 role 改为 free 并清空 victim mapping。same-bank promotion 不搬 bank-to-bank 权重,但仍会把 promoted row D2D 镜像到原层稳定 resident/GEC 行(prepared-only 只 prepared)。

只有 bank 已被覆盖的 pending promotion 需要重新装载 promoted 行。它复用 staged
窗口的 host write + H2D 路径,并在 postprocess 后刷新原层 decode 行；这不是
staging row -> resident row 的额外 D2D promotion。

protected/shared 专家可以作为 promoted winner,但永远不会成为 victim。

### 10.4 淘汰策略

不能被淘汰的 row：

- 当前层正在使用。
- 有未完成的 consumed/ready/raw fence。
- 有 inflight dependency。
- 处于 LOADING 或 IN_USE 状态。

当前边界选择按以下顺序淘汰：

1. 排除 `shared_ids`/protected resident。
2. 低 history score。
3. score 相同按 expert id,保证 TP 各 rank 确定性一致。

边界发生在上一 chunk 全部层完成之后,因此不会 evict 正在执行的 row。

### 10.5 去重

同一个 `(layer_id, expert_id)` 不能同时存在于：

- resident pool。
- staging bank A。
- staging bank B。

当前实现由 `bank = layer_position % 2`、每个 bank 只保留一个 current layer 的 mapping,以及 `StagingRowPool.configure` 对 residents/staged 的 overlap 检查来保证；没有额外跨两个 bank 扫描同一层的重复副本。

## 11. 显存与 KV 预算

### 11.1 基本公式

当前实现的 staging 资源峰值估算(GPU 部分与 pinned host 分开)：

```text
current_staging_gpu_peak_bytes =
    full_context_raw_n_temporary
  + 2 * (R_max + S) * per_expert_prepared_bytes
  + 2 * S * per_expert_raw_bytes
  + raw_scratch_shared
  + postprocess_scratch_bytes

host_pinned_bytes =
    2 * S * per_expert_raw_bytes
```

上述 `2 * S` raw 行是默认 prepared-only 布局；`SGLANG_KT_MXFP4_PREPARED_ONLY=0` 的 legacy 回退把每个 bank 的 raw 行数改为 `R_max + S`。

其中：

- `full_context_raw_n_temporary` 只在首次初始化 manager 时存在：它提供兼容的 slot/CPU ring 形状，bank 建好后尝试移到 CPU；该调用有本地 `try/except`，移动失败时只记 warning 并继续留在 GPU，不能把 steady-state 不占 GPU 当作绝对保证。逐层 `process_weights_after_loading` 的迁移没有本地 catch：`_v4_marlin_weights` 不存在时返回 0、raw 留在 GPU；真正迁移异常会向上传播。
- `R_max` 是均匀分布后所有 MoE 层中的最大 resident 数,`S` 是夹到 N 的 staging 窗口。两个 bank 都按 `R_max + S` 行 prepared 分配；raw 行数在 prepared-only 下是 `S`,在 legacy 回退下是 `R_max + S`。
- `raw_scratch_shared` 是 GEC/resident selected-row CPU→GPU 更新复用的 shared context raw scratch，按 `min(S, resident 批量)` 限宽；调用路径分别传 `src_layer` 或 `self.gpu_layer`,具体 owner 取决于路径,但不会按 decoder 层数复制。
- `num_experts_per_layer` 是每层专家数(本文示例 256；第 16.2 节基线实测的 DeepSeek-V4.1-Flash 为 384,见 `[kt-mask] num_experts=384`),用于计算 `full_canonical_raw_n` 与 per-expert 字节数。
- `host_pinned_bytes = 2 * S * per_expert_raw_bytes`(每 rank),`S` 就是 `--kt-layer-h2d-slots`(第 7.5 节)。它**同时是功能开关**:`S` 未设置时整个 layerwise staging 关闭,ring 深度退回 2 槽对,pinned 占用与历史默认路径完全一致。
  该 ring 只由 layerwise prefill 分配,GEC 的 CPU staging 从不使用它(`cpu_buffer_depth=2`)。
  **启动日志 `_layerwise_host_ring_depth` 只打印 `ring_depth` 和 `S`；具体字节数须再用实际 `per_expert_raw_bytes` 相乘得到,不要引用未经验证的估算值。**
- `postprocess_scratch_bytes` 是 Marlin repack 的临时工作区（若有）。prepared-only postprocess 直接写 prepared pool，不再持久保留 prepared scratch bank。
- 注意：`get_mxfp4_layerwise_prefill_reservation_bytes()` 的启动预留实际按 `full_context_raw_n_temporary + 2 * (prepared + S raw)`(prepared-only)计算,没有把延迟分配的 `raw_scratch_shared` 和 postprocess 临时区单独加进去；这些在首次 GEC 更新/首次 postprocess 时由 CUDA 分配器按需分配,做启动预算时应额外留量。

当前 prepared-only bank 已按 `R_max + S` 行 prepared + `S` 行 raw scratch 落地。
Marlin postprocess 仍覆盖 bank 当前物理行范围；进一步只处理 effective 子集属于
后续优化,不改变本公式的保守上界。

### 11.2 启动预留

启动时必须预留：

```text
pinned layers
+ resident pool
+ staging pool
+ GEC raw scratch / postprocess 临时区(按需分配,启动预留函数未单列)
+ KV cache
```

staging 不占 `kt-expert-gpu-slots` 的逻辑容量,但仍然占物理显存。

### 11.3 OOM 回退

初始化 staging 失败时：

1. 记录 staging disabled reason。
2. 回退到普通 GEC。
3. 或回退到纯 CPU/GPU hybrid。
4. 初始化失败必须回退；当前实现没有运行期超时回退。

## 12. 参数语义与推荐配置

### 12.1 参数职责

| 参数 | 当前职责 |
|---|---|
| `--kt-num-gpu-layers 10` | 前 10 层整层常驻 GPU(实测 L0–L9 为 dense 层,不经过 KT wrapper),不参与搬运和 slot 统计 |
| `--kt-expert-gpu-slots 1024` | 动态常驻专家池(GEC 全局容量,实测值),服务 decode 和长期热点 |
| `--kt-num-gpu-experts 0` | 每层均匀分配的旧语义；与 `--kt-expert-gpu-slots` 同时给出时被覆盖(启动日志 `--kt-expert-gpu-slots=1024 overrides --kt-num-gpu-experts/--kt-gpu-experts-ratio`) |
| `--kt-layer-h2d-slots 64` | 轻量 prefill staging 窗口；decode 不走跨层 staging,该值不改变 decode 的 H2D 行为。也是 host slot ring 的单代深度 `S`(ring = `2 * S`,第 7.5 节) |
| `--kt-layer-h2d-slots 128` | 实测 prefill streaming 窗口；bank 按 `R_max + 128` 行 prepared + `128` 行 raw scratch 分配,窗口决定预取集合与 CPU fallback 比例 |
| `--kt-layer-h2d-slots 384` | 整层候选窗口(窗口夹到该层专家数 N,实测 N=384),适合 prefill streaming（仍需 threshold 门控） |
| `--kt-gpu-prefill-token-threshold 2048` | 与 `--kt-layer-h2d-slots` 联合生效的启用门控；未设置或未达到阈值时不搬运 |

注意：显式写 `--kt-layer-h2d-slots 0` 时运行期等价于关闭；但如果没有同时提供
`--kt-expert-gpu-slots`,命令行校验仍会要求正 threshold(`kt_hook.py` 只判断参数是否为 None)。

### 12.2 整层 Prefill 推荐配置

```text
--kt-num-gpu-layers 10
--kt-expert-gpu-slots 1024
--kt-num-gpu-experts 0      # 与 GEC 联用时被 kt-expert-gpu-slots 覆盖
--kt-layer-h2d-slots 128    # 实测窗口；整层窗口 = 该层专家数 N(384 专家的模型取 384)
                            # 同时也是 host slot ring 的单代深度(ring = 2 * S,第 7.5 节)
```

`--kt-gpu-prefill-token-threshold` 与 `kt-layer-h2d-slots > 0` 联合生效：只有 prefill-like batch 达到阈值时才启动 staging。未设置 threshold、或 token 数小于阈值时,不做任何 H2D 迁移,保持普通 GEC / CPU 路径。

语义：

```text
前 10 层（L0–L9,dense）：
    整层 GPU

L10–L39（30 层 MoE）：
    L 层 attention 后、GEMM 入队前预取 L+1(partial 在 GPU GEMM 提交后)
    staging 与 resident 不共享逻辑容量

decode：
    使用 resident pool
    不启用跨层 staging；S 只影响 prefill 路径的 bank/host ring 配置
```

compact bank 的显存随 `S` 增长,`S=64`、`S=128` 和 `S=384` 不再占用相同的物理
行数。实际 staged 数仍受 `N - |resident(L)|` 限制；S 越大,GPU 覆盖越高,但
raw/prepared bank 与 `2 * S` host ring 的占用也越大。

### 12.3 Decode 推荐策略

```text
resident pool = 1024
staging = 0            # decode 不启用跨层 staging
selection = 当前 demand + decayed history
```

`staging_requested_for_batch` 要求 prefill-like,decode 不会触发跨层 H2D。`--kt-layer-h2d-slots=64` 只影响 prefill 路径的窗口和 host ring,不会给 decode 增加 staging 专家。
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
forward_mode 是 prefill-like(extend 或 TARGET_VERIFY)
且 kt-layer-h2d-slots > 0
且 kt-gpu-prefill-token-threshold > 0
且 token 数 >= kt-gpu-prefill-token-threshold
且 MXFP4 layerwise staging 可用
```

行为：

- full-window 路径在 L 层 GEMM 入队前发起 L+1 预取；partial 路径在 GPU GEMM 提交后、CPU join 前发起。
- L 层 CPU 和 GPU 专家并行。
- L+1 层使用自己的历史命中表。
- staging 命中后异步更新 history 和 chunk hit 计数；resident pool 在 chunk 内保持固定布局,新的 forward/chunk 边界执行 promotion。
- CPU-only 层(`num_gpu_experts == 0`)的 CPU 路由计数直接取 `topk_ids.numel()`,
  不逐层读回设备标量；门控路径已经得到的计数在 round 边界随 summary 汇总
  (§14.5、§15.1)。

### 13.2 Decode Resident Mode

行为：

- 使用 resident pool。
- 根据当前 router demand 和 history 做 admission。
- decode 不加载跨层 staging 专家。
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
| L+1 ready 未完成 | 当前没有超时；compute stream 的 `wait_event(ready_event)` 会一直等待,若 event 永不 record 则该层无法前进 |
| bank consumed 未完成 | transfer stream 等 `consumed_event`,不满足则不复写 |
| TP rank selection 不一致 | 当前不做 selection 比较/广播；各 rank 由相同 history/router 推导,失败通过 TP 共识在同一 phase 抛出,没有"回退 resident"分支 |
| 历史表为空 | 使用冷启动策略 |
| router demand D2H 异常 | `_stage_router_demand` 捕获后 debug 记录并跳过该层统计,当前层计算继续 |
| postprocess 失败 | `_postprocess_slot` 仍在 finally 中 record `ready_event`；随后 TP 共识抛错,`_load_slot` 异常路径 `invalidate()` 退回 EMPTY。只有 fence 发布失败且 `transfer_stream.synchronize()` 也失败时才置 `poisoned` |

当前实现没有超时回退；正确性依赖所有入队 event 最终 record。不要在未发布 fence 的路径上复用 bank。

## 14. 正确性与并发契约

### 14.1 基本顺序

```text
router
  -> 冻结 CPU/GPU 专家集合
  -> full-window: 提交 L+1 staging host write / H2D / postprocess
  -> partial: 提交 CPU fallback
  -> 提交 GPU GEMM
  -> partial: 提交 L+1 staging host write / H2D / postprocess
  -> 等待 CPU 和 GPU 两路完成
  -> merge
  -> 更新历史表
  -> 安全边界更新 resident pool
```

full-window 路径把 L+1 预取放在 GEMM(L) 入队前；partial 路径把它放在 GPU GEMM 入队后、CPU join 前。

### 14.2 生命周期与 Lease

staging bank slot (`_Mxfp4PrefillSlot`) 没有独立 lease/refcount 对象；它的 row 生命周期由以下 event/guard 约束：

- staging bank 的 GPU GEMM 读取 row：compute stream 等 `ready_event`,消费后 record `consumed_event`。
- H2D 写 raw scratch：transfer stream 等 `host_slot_free_events` 和上一代 raw/ready/consumed fence。
- postprocess：postprocess stream 等 `raw_ready_event`。
- promotion：只在 forward/chunk 边界、且 bank 仍持有该层时改 metadata；否则进入 pending。
- 复写前由 `reuse_guard` 选择等待 `consumed_event`/`ready_event`/`raw_ready_event`。

staging bank 没有 "lease 计数为 0" 的检查；不要按 lease/refcount 语义扩展 bank 复用。GEC resident cache 是另一套生命周期：`kt_gec.py` 的 `GecExecutionLease`/`GecTransferLease` 持有 refcount；`GecResidencyPlanner.begin_gpu_use()` 在 GPU 使用 resident rows 前登记 lease，`record()` 绑定 compute stream event，`poll_gpu_completions()` 在后续轮次释放已完成的 lease；`dependencies_for()` 让 compute stream 等 H2D dependency。resident row 的 eviction 由这套 lease/refcount 保护，不是 `reuse_guard`。

### 14.3 TP 一致性

TP 多 rank 必须满足：

- staging selection 在每个 rank 上必须推导出完全一致的值（bank id、layer id、expert id），但当前不比较也不广播。
- 当前实现不额外做每层 broadcast：resident mask 在启动时已由 rank 0 广播,
  每层历史表由相同的 router id 更新,两个输入一致时 `top_k` 的选择也一致
  （`Planner` 的排序按 `(-score, expert_id)` 确定性 tie-break）。
  如果将来需要 rank 0 权威裁决,放在 epoch 边界而不是每层。
- resident layout 通过 layout epoch 同步。
- 任一 rank 失败通过 TP 共识在同一 phase 抛出；当前没有 selection 不一致的自动回退。

### 14.4 确定性

- 相同输入和相同 resident layout 必须产生相同 expert selection。
- stable row assignment 不应每轮重排。
- 同名 logical expert 必须稳定映射到同一物理 row,除非被 evict。

### 14.5 避免串行点

尽量避免：

```text
torch.cuda.synchronize()
counts.tolist() 在关键路径(当前只读 pinned host,不触发设备同步)
per-layer object gather
per-layer 阻塞 CPU 写权重
.item() 读回设备标量(除非该值当层就要用于门控)
```

需要 `_load_slot` 的层在 TP>1 时仍保留每层一次 Gloo broadcast(host-write publish)、5 次 Gloo all-reduce 和 1 次 NCCL all-reduce(+`.item()`)；warm reuse 跳过这些控制面成本。

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

每层 `_layer_metrics` 与 round summary 结构覆盖 `candidate_count`、
`selected_count`、`resident_count`、`staging_hit_count`/`staging_miss_count`、
`cpu_expert_count`/`gpu_expert_count`、`cpu_route_count`、`copy_bytes`、
`drain_count`、`copy_submits`、`copy_pairs`、`copy_ms`、`postprocess_ms`、
`bank_reuse_wait_ms`、`ready_wait_ms` 和 `compute_ms`；summary 还带 `layers`。
计数类字段按窗口求和,`copy_bytes`、`drain_count` 和各 ms 字段按层取均值。
每 16 层输出一次 `[kt-prefetch] round ...`（debug 级）。
`cpu_route_count` 是该窗口内 staging partial 路径与 hybrid 门控路径的 CPU 路由
计数之和:门控路径在 `num_gpu_experts == 0` 时直接取 `topk_ids.numel()`,其余
情况复用当层门控本来就需要的读回值,两者都不新增设备同步（§14.5）。
每层 info 日志保留 `layer`、`epoch`、`slot`(bank)、`prime`/`prefetch-hit`、
`partial`、`cpu`、`staged`、`resident` 这些 gate 字段；`merge_ms`、
`promotion_count` 和 `eviction_count` 尚未并入 round summary。边界 promotion
本身会在 debug 日志输出 promoted/queued 行数。
`copy_submits` 和 `copy_pairs` 已在 `_layer_metrics` 中计算,但
`_record_layer_telemetry` 尚未把它们传给 `record_layer`,所以 round 日志当前
恒为 0；per-layer metrics 中仍有实际值。
`bank_reuse_wait_ms` 当前恒为 0,代码没有计时赋值,不能用于判断 ring 是否足够。
当前 telemetry 还有两个口径限制：warm reuse 不经过 `_load_slot`，所以
`candidate_count`/`selected_count`/`resident_count` 没有当层记录（summary 记 0），
`drain_count`/`copy_bytes`/`copy_ms`/`postprocess_ms`/`bank_reuse_wait_ms` 也不写
（这些均值只对 load 层计算）。`drain_count` 在有 host transport 的 load 为 1，
无 transport 的 load 为 0；不要把它当作“每个非 warm 层恒 1”或“warm=0”。

### 15.1 每层字段

当前 round summary 实际记录：

```text
layers
candidate_count
selected_count
resident_count
staging_hit_count
staging_miss_count
cpu_expert_count
gpu_expert_count
cpu_route_count
copy_bytes          # 每层均值,日志名 copy_bytes_mean
drain_count         # load 层均值；有 transport 1、无 transport 0；warm 不写
copy_submits        # 未桥接,summary 当前恒 0
copy_pairs          # 未桥接,summary 当前恒 0
copy_ms
postprocess_ms
bank_reuse_wait_ms   # 当前恒为 0
ready_wait_ms
compute_ms
```

尚未进入 round summary：

- `layer_idx`：每层 info 日志有 `layer`。
- `layer_position`：只用于内部 bank 选择,未输出。
- `bank_id`：info 日志的 `slot` 即 bank。
- `merge_ms`、`promotion_count`、`eviction_count`：尚未记录；promotion 只有边界 debug 日志的 promoted/queued 行数。
- `copy_submits`、`copy_pairs`：per-layer metrics 有值,但当前 summary 桥接缺失,日志显示 0。

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
| bank reuse wait | 当前实现固定写 0,尚未计时；不能据此判断 ring 是否足够。设计目标是 host write 等上一代 H2D 的时间,明显变大说明 H2D 比 host write 慢或 ring `2 * S` 太小 |
| host write flush 次数/层 | 有 transport 的 reload 层为 `1`；无 transport 的 load 为 `0`；warm slot reuse 不写 `drain_count`。整代只 drain 一次 |
| H2D runtime 提交次数/层 | 有搬运时,batch 路径成功为 `1`；回退时按连续 run 计数。per-layer `_layer_metrics` 记录实际 `copy_submits`/`copy_pairs`；round summary 当前未桥接这两个字段,日志恒为 0 |
| per-layer 标量同步次数 | CPU-only 层(`num_gpu_experts == 0`)应为 0；其余层只保留 `has_cpu_fallback` 当层必需的那一次 |

### 15.3 批量拷贝路径与提交合并

暂存集 `staged_expert_ids` 由 `StagingPrefetchPlanner.select` 产出,是**升序**的,
而一次 host 写入对应的槽位由 `host_generation_range` 给出,在代内**连续**。
因此选择中任何一段"目标物理 row 与主机槽号同时加一"的相邻对,
其逐 expert `destination[e].copy_(cpu_buffer[s])` 与一次
`destination[e0:e0+n].reshape(n, -1).copy_(cpu_buffer[s0:s0+n].reshape(n, -1))`
搬运字节完全相同。这里仍用 `contiguous_copy_runs` 切段,返回
`(first_index, length)` 列表；它只合并两侧同时连续的对,不改变选择集。

在这些 run 之上,当前实现优先走两条 FreeToken 风格批处理路径：

- **resident D2D**：所有 prepared bank 使用同一组 `(source_row, destination_row)`，
  通过 `kt::MultiRowCopy::run` 一次 kernel launch 完成 16-byte vector gather；
  W13 的 `swap_halves` 在同一 kernel 内处理。源/目标布局、dtype、连续性和
  对齐不满足时直接返回 `False`，调用方执行原逐 expert/per-tensor `copy_`。
- **staging H2D**：把每个 `(run, bank)` 描述为一个 `BatchCopy`，用一次
  `cudaMemcpyBatchAsync` 提交整代 pinned host -> GPU 拷贝。该路径只在
  CUDA 13 及以上启用，首次使用会做一次真实 16-byte H2D probe；JIT 编译、
  probe 或运行时失败后回退到原逐 run strided `copy_`。

两条路径都只是提交层优化：传输字节、选择集、row ownership 和暂存窗口语义
均不改变。稀疏选择仍然退化为长度为 1 的 run；CUDA 12 或布局不满足时也仍能
工作，不会因为快速路径不可用而禁用 staging。

遥测上新增两个窗口求和字段(见 §15.3 日志):

- `copy_submits` —— 实际发出的 runtime 拷贝提交数;batch 成功时整代记 `1`;
- `copy_pairs` —— 按 run 归并后被搬运的专家行数。

设计上,两者比值 `copy_pairs / copy_submits` 即**合并因子**:密集整层挑选时
接近 `S`,完全稀疏时为 `1.0`。但 `_record_layer_telemetry` 当前没有把这两个
per-layer 值传给 `record_layer`,所以 round 日志中的 `copy_submits`/`copy_pairs`
恒为 0。桥接修复前,不能用日志比值确认合并是否生效；per-layer `_layer_metrics`
中仍有实际计数。

### 15.4 日志建议

每 16 层或每轮汇总一次：

```text
[kt-prefetch] round layers=16 candidates=... selected=... resident=...
  staged_hits=... staged_misses=... cpu=... gpu=... cpu_routes=...
  copy_bytes_mean=... drains=... copy_submits=... copy_pairs=...
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
- 无死锁；当前没有超时回退,等待的 CUDA event 必须最终 record。
- OOM 时可回退。

### 16.2 性能

在相同模型、相同 chunk、相同 TP 配置下：

```text
prefill tokens/s >= 900
```

至少应达到当前整层 GPU 路径水平,并希望优于它。

改动前的实测基线(2026-10-14,ARM 单机、TP=2、DeepSeek-V4.1-Flash:40 层、每层 384 专家、
top-k=6、`--kt-layer-h2d-slots 128`、chunk=2048)：

```text
input throughput      750 - 775 token/s(2048 token 的完整 chunk)
每层 staging_mxfp4    中位 132 - 194 ms(L10 首层约 80 ms),11 个 staging 层累计约 1.8s/chunk
每层 copy_ms          105 - 229 ms(round 均值；host write + H2D 入队 + 每批栅栏这一段)
bank_reuse_wait_ms    约 0.5
ready_wait_ms         约 7.5
```

该基线里每层对 128 个专家各做一次 KT drain 和两次 TP 栅栏,即第 20 节 item 8 的
每专家往返。按第 7.5 节改成每层一次搬完整代(有实际 transport 时 1 次 drain；控制面仍有多阶段 TP 共识)并把 host ring
设为 `2 * S` 后需要重新测量,目标仍是上面的 `prefill tokens/s >= 900`；复测时重点看
`bank_reuse_wait_ms`；但当前实现该字段恒为 0,复测前必须先补计时,否则不能用它判断
两代滚动是否足够。

当前代码又增加了 fused resident D2D 与 CUDA 13 batch H2D。它们不减少 PCIe
传输字节,目标是去掉逐 expert/per-tensor 的 Python 与 runtime 提交开销；因此
上表只能作为改造前基线,不能直接当作新版本成绩。需要在真实 CUDA 13 GPU 上重新
编译 `kt_prefill_copy.cuh`,分别记录 batch 开启与
`SGLANG_KT_DISABLE_PREFILL_FAST_COPY=1` 的 per-layer `copy_ms`/`copy_submits`
（后者尚未桥接到 round summary）和端到端 `prefill tokens/s`,再判断收益是否
达到 FreeToken 同量级。

其他指标：

- L+1 ready 事件在 L 层 compute 完成前完成的比例大于 90 percent。
- staging hit rate 在 prefill 整层模式下接近 100 percent。
- CPU expert fraction 在 prefill 整层模式下接近 0。
- decode 吞吐不因 staging 逻辑明显下降。

### 16.3 显存

- staging 预留后进程不 OOM。
- KV cache 预算扣除 staging 的 prepared/raw bank；GEC raw scratch 和 postprocess 临时区按需分配,需额外留量。
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

- 双 bank 资源,每 bank `R_max + S` 行 prepared + `S` 行 raw H2D scratch(prepared-only 默认；legacy 回退为 `R_max + S` 行 raw；host 侧为 `2 * S` 两代滚动 ring,S 即窗口,第 7.5 节)。
- 统一 `StagingRowPool`,resident/staging row role 可原地切换。
- bank 状态机。
- ready、raw_ready、consumed、host_free 事件。

验收：

- 两个 bank 可连续轮换,无提前复写。
- bank 复用时对应 consumed/ready/raw fence 已满足(由 `reuse_guard` 选择)。

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

- L 层 attention 后触发 L+1 预取；full-window 在 GEMM 入队前,partial 在 GPU GEMM 提交后。
- copy stream 和 compute stream 依赖。
- postprocess stream。

验收：

- L+1 ready 事件明显早于 L 层结束。
- host write 在 attention(L) 之后发起,H2D 可与后续 GPU 执行重叠；host 控制面会延迟 GEMM 的 launch。

### M4 CPU/GPU 并行与 Merge

交付：

- L 层 CPU 和 GPU 专家集合划分。
- CPU 异步任务和 GPU kernel 并行提交。
- merge barrier。

验收：

- 日志显示 CPU 与 GPU 时间重叠。
- 不存在先 CPU 后 GPU 或先 GPU 后 CPU 的串行化。

### M5 Resident 提升（已落地）

交付：

- staging 命中统计。
- forward/chunk 边界提升。
- 低分 resident 淘汰。
- bank 已覆盖时的 pending host H2D 提交。

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
- bank 提前复用必须被 event/guard 阻止,测试验证没有提前复写；`reuse_guard=poisoned` 的 bank 再次加载必须报错。
- consumed/ready/raw event 未完成时,transfer stream 不得复写对应 bank（通过 `wait_event`,不要求 Python 抛错）。
- 候选集合只包含目标层专家。
- resident 和 staging 去重。
- score 衰减和提升。
- TP rank selection 一致性。
- host write 整代边界:一层一次提交本代 `|staged(L)| + |resident_host_ids(L)|` 个 host 行（不超过容量 `S`）、该批连续占用当前代（每代容量 `S`）、两代代间 slot 不重叠、单代请求超过 `S` 必须报错、空传输不提交 host 行、不推进 generation 且 `drain_count=0`、未设窗口或 `S<=0` 时整个功能关闭且 ring 深度退回 2 槽对。
- 传输循环契约:有 host transport 的 reload 层 1 次 drain,无 transport 的 load 为 0；warm reuse 不进入 host-write pass,也不写 `drain_count`。需要 `_load_slot` 时正常路径在 TP>1 下为 5 次 Gloo all-reduce + 1 次 NCCL all-reduce + 1 次 Gloo broadcast,warm reuse 分支不进入这些 collective；不存在 `position % 2` 的 2 槽假设。

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
| h2d-slots | 64、128、256、384(实测模型整层)；同时决定 host ring = `2 * S` |
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
host write flush 次数/层(有 transport 的 reload 1、无 transport 的 load 0、warm reuse 不写)
pinned 内存占用(随 `2 * S` 线性)
```

### 18.4 正确性回归

- 同 prompt 多次生成结果一致。
- 与整层 GPU 路径对比 logits 或生成结果。
- 长上下文不丢失。
- 多请求并发不串扰。

### 18.5 长稳测试

- 连续 prefill 数千轮。
- 交替 prefill 和 decode。
- 监控 bank 泄漏、`reuse_guard` 卡在 loading/poisoned 和事件未完成。
- 观察显存是否缓慢增长。

## 19. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| Host write 阻塞 | GEMM(L) launch 延迟 | attention(L) 已入队后发起预取；host 控制面会推迟 GEMM launch,H2D 与后续 GPU 执行重叠 |
| CPU 内存带宽竞争 | CPU GEMM 和 H2D 变慢 | 控制 batch size,测量带宽占比 |
| PCIe 成为瓶颈 | ready_wait 增加 | 提前一层预取,提升 staging 命中率 |
| MXFP4 postprocess 占用显存 | OOM | 单独统计 raw、prepared、scratch |
| 窗口覆盖不足 | CPU 仍是关键路径 | prefill 用整层窗口(`S >= N - R`) |
| 历史表冷启动 | staging 命中率低 | 按 expert 分块轮转,逐步建立历史 |
| router 同步链 | 无法重叠 | history D2H 异步；但 host-write publish 仍有每层 Gloo broadcast,`num_gpu_experts == 0` 的层直接从 `topk_ids.numel()` 得到计数(§14.5) |
| prefill-like 门控未触发（forward_mode 未发布） | streaming 不生效 | 检查 `path_select` 的 `extend` 字段；ModelRunner 必须发布 forward_mode |
| threshold 未设置或未达到 | 不搬运专家权重 | 与 `--kt-layer-h2d-slots` 联合设置 threshold |
| TP layout 不一致 | 输出错误或死锁 | 启动 resident mask 由 rank 0 生成并广播；GEC 更新轮广播 selected/layout。每层 staging selection 不广播,由相同 history/router 输入确定性推导 |
| resident row 被误 evict | 数据竞争 | GEC `GecExecutionLease`/`GecTransferLease` refcount + stream event；staging bank 另由 `reuse_guard`/event 保护 |
| bank 提前复用 | 权重损坏 | `reuse_guard` + stream/event 依赖 |
| KV 预算未扣除 staging | 启动 OOM 或 KV 不足 | 启动阶段统一预算 |
| 多请求并发 | 历史和 staging 竞争 | 所有请求共享模型级双 bank,并在同一 forward 中合并 token 后逐层执行 |
| `--kt-layer-h2d-slots` 过大 | pinned 内存按 `2 * S` 线性增长 | 未设置时整个功能关闭、ring 保持 2 槽对；一旦设置就按 `2 * S * per_expert_raw_bytes` 分配,加大 `S` 前先确认 memlock 与 SHM 余量；启动日志只打印 `ring_depth`/`S`,字节数按实际 per-expert raw 大小相乘得到 |
| host ring 被误当作"额外专家数" | 期望的显存/内存与实际不符 | `--kt-layer-h2d-slots` 是窗口 + 开关,不是容量；ring 只是同一批专家的两代暂存,不增加可驻留专家数 |

## 20. 开放问题与已决事项

1. prefill 和 decode 是否需要不同的 `kt-layer-h2d-slots` 默认值。
2. staging bank 深度固定 2 是否足够,是否需要可配置 3 或 4。
3. [已决] promotion 采用统一 row pool:promoted row 原地升级为 resident,victim row 置 free；pending 场景复用 host H2D,不做额外 bank-to-bank D2D。
4. OOM 时先减小 staging,还是先减小 resident pool。
5. [已落地] prepared-only 物理 bank 收缩为 `2 * (R_max + S)` 行 prepared + `2 * S` 行 raw scratch；后续可优化 postprocess 只处理 effective 子集。
6. TP rank 0 权威 selection 广播的实现方式。
7. pinned 层根本不经过 KT wrapper,首个 KT 层必然是同步. 如何解决?
8. submit→sync 改成 worker 线程/事件驱动。
   - 轻量版(整代搬运,不动线程模型)已实现:host slot ring = 两代滚动,
     每代 `S = --kt-layer-h2d-slots` 个 slot(第 7.5 节)。传输循环把
     "每专家 submit+sync" 改成 "提交整代 `|staged(L)| + |resident_host_ids(L)|` 个专家 → 一次 sync →
     一次性 enqueue 整代 H2D",相邻层交替使用两代 slot,host write 与上一层 H2D
     重叠；需要 `_load_slot` 的 reload 层在 TP>1 时有 5 次 Gloo all-reduce + 1 次
     NCCL all-reduce + 1 次 Gloo broadcast（warm reuse 跳过），并不是每专家两次栅栏。层内不再有子分批,与 FreeToken 的 `layer_id % 2`
     乒乓形状同构。
     代码入口:`kt_prefill_staging.host_generation_range` / `host_ring_depth`、
     `_Mxfp4LayerwisePrefillManager._load_slot` 与 `_submit_host_write`
     (`submit` 已不自带 drain)、`_initialize_mxfp4_layerwise_pipeline` 传入的
     `cpu_buffer_depth=_layerwise_host_ring_depth(method.kt_config)`
     (参数未设置时返回 2,整个功能关闭)。
   - 重量版(worker 线程/事件驱动,层内完全不 drain)已实现为 **publish 模式**,
     是唯一的实现(没有环境开关,A/B 回退已移除):
     - 与 FreeToken 同形:FreeToken 在 copy stream 上 `ready_events[buffer].record(copy)`,
       消费方 `compute.wait_event(ready_events[buffer])`;KT 的 host writer 是
     CPU 线程池,于是 TP0 提交完整代后直接在 transfer stream 上 `record` 一个
     gate —— 该 gate 同时排在"本层已提交的写入"(KT writer 严格 FIFO)和
     "上一代 H2D 释放"(transfer stream 先 wait `host_slot_free_events`)之后,
     因此语义等价,且 Python 线程不再调用 `cpu_infer.sync()`。
     - `host_slot_free_events[i].synchronize()`(host 忙等)改为
       `transfer_stream.wait_event(...)`:写 host slot 前在该 stream 上排队,
       主机不再等上一代 H2D 完成。
     - 多 TP:只有 TP0 写 host slot,其余 rank 必须等 TP0 的写入。每层一次
       funnel(1 字节 Gloo broadcast,不做 NCCL,避免与控制流上的设备归约
       乱序),把 TP0 的 marker 映射到每个 rank 的 copy stream;每次 publish
       重新分配事件,避免跨层重-arm。
     - 代码入口:`_publish_host_write_phase` / `_host_write_all_ranks`,
       `_load_slot` 中每次加载固定走 publish,不再调用
       `sync_write_weight_scale_to_buffer`。

   - 预取提前(FreeToken 第 4.1 节排序)已实现,且不再提供环境开关:
     - 旧顺序:`wait_event(ready(L))`(apply 内) → GEMM(L) 入队 →
       `_prefetch_successor` → `_load_slot(L+1)`。GPU 上 H2D(L+1) 只依赖
       transfer stream 与 main stream 的并行,没有"L 在算时 L+1 已拷完"的保证。
     - 新顺序:`_acquire(L)` → `_prefetch_early(L)`(`_load_slot(L+1)` 入队:
       D2D resident copy、TP0 host write、H2D、postprocess)→
       `wait_event(ready(L))` → GEMM(L)。
       `_load_slot(L+1)` 的 D2D 仍被 `transfer_stream.wait_event(consumed(L))`
       栅栏,而 `consumed(L)` 在 L 的 GEMM 入队之后才 record；event 记录一次
       即可被之后的 stream wait 引用,这里只是"更早入队、等待更晚的事件",
       不构成依赖环。H2D(L+1) 的设备侧依赖因此不变,提前的只是
       Python/CPU 侧的发起时机。
      - 代价:需要 `_load_slot` 的层在 TP>1 时有多阶段 host 同步(5 次 Gloo all-reduce、
        1 次 NCCL all-reduce 和 1 次 Gloo broadcast),提前到 GEMM(L) 之前会让这些 CPU 往返落在 L 的 launch 前面；单层首个
       apply 由 `prefetch_epoch` 保证不会重复预取。
     - partial 层仍在 `_apply_effective_set` 里做 post-GEMM 预取(CPU fallback
       路径),开启提前预取后本轮余下各层沿用晚预取,`_late_prefetch_target`
       负责在轮次真正需要时补一次,避免环形槽位空转。
     - 代码入口:`_prefetch_early` / `_late_prefetch_target`,`apply`、`_advance_round` 中的
       `prefetch_epoch`/`early_prefetch_only` 状态。

9. 剩余逐层读回：`num_gpu_experts > 0` 的门控层仍每 call 读一次
   `(cpu_topk_ids >= 0).sum()`(`has_cpu_fallback` 当层就要用,不可延后)；
   `tp_rank != 0` 或无 `cpu_stream` 时该值只喂遥测,可进一步延后到层边界。
10. [代码已落地，待 GPU 实测] 提交层批量化：resident D2D 用 multi-bank JIT
    gather，H2D 用 CUDA 13 `cudaMemcpyBatchAsync`。快速路径失败后保留原有
    scalar/per-run 回退；需要在 CUDA 13 机器上验证 JIT 编译、W13 half swap、
    pinned host 批量拷贝和 TP=2 正确性，并对比
    `SGLANG_KT_DISABLE_PREFILL_FAST_COPY=1` 的端到端性能。

### 20.1 当前实现边界（与早期设计稿的差异）

- 预取发起：full-window 路径在 GEMM(L) 入队前同步执行 `_load_slot(L+1)`，
  host 控制面会延迟 GEMM launch；partial 路径在 GPU GEMM 提交后、CPU join 前
  预取。H2D 与已入队的 attention/后续 GEMM 重叠，但不是 CPU 线程级并发。
- Bank 复用：staging bank 没有独立 lease/refcount 对象；`reuse_guard` 选择等待
  `consumed_event`/`ready_event`/`raw_ready_event`。host slot 的
  `host_slot_was_used` 只写不读，不是门控。GEC resident row 的 refcount lease 见 §14.2。
- 失败处理：没有 L+1 ready 超时；`wait_event(ready_event)` 必须等 event 最终
  record。postprocess 异常通常 `invalidate()` 回 EMPTY，只有 fence 发布失败且
  `transfer_stream.synchronize()` 也失败时才置 `poisoned`。
- TP 一致性：不广播也不比较每层 staging selection；依赖相同 history/router
  输入的确定性推导，失败通过 TP 共识在同一 phase 抛出。
- Promotion：row pool 层面 promoted row 升级为 resident、victim row 置 free；
  同 bank 仍会把 promoted row D2D 镜像到原层 decode 行；bank 换层时
  `pool.clear()`，per-layer mapping 不跨覆盖保留。
- Telemetry：`bank_reuse_wait_ms` 恒为 0；`copy_submits`/`copy_pairs` 已在
  per-layer metrics 计算但尚未桥接到 round summary（日志为 0）。
  `candidate_count`/`selected_count`/`resident_count` 和 copy/drain 相关均值
  只对 load 层完整；warm reuse 不写这些字段（前三个由 summary 记 0，copy/drain
  相关均值不覆盖 warm reuse，§15）。
  `layer_position`/`bank_id`/`merge_ms`/`promotion_count`/`eviction_count`
  尚未进入 round summary。
- Decode：`staging_requested_for_batch` 要求 prefill-like，decode 不触发跨层
  staging；`--kt-layer-h2d-slots` 只影响 prefill 路径的 bank/host ring 配置。

## 21. 结论

当前执行模型是：

```text
前 10 层（L0–L9,dense）：
    kt-num-gpu-layers=10,整层常驻 GPU
    不经过 KT wrapper,不参与 H2D、staging、resident 统计

L10–L39（30 层 MoE,实测配置）：
    L 层 attention 后、GEMM 入队前预取 L+1；partial 路径在 GPU GEMM 提交后预取
    L+1 使用自己的历史命中表
    候选只来自 L+1 层
    排除 resident(L+1)
    staging 独立于 kt-expert-gpu-slots

执行层：
    GPU = resident + staged_hits
    CPU = 剩余专家
    门控计数:gpu_rows == 0 时取 topk_ids.numel(),不读回设备标量(§14.5)
    H2D 与已入队的 attention/后续 GEMM 重叠；host 控制面会延迟 GEMM launch
    host write 一次搬完本代窗口（≤S）；有实际 transport 时 drain 次数 = 1
    resident D2D 一次 multi-bank kernel；H2D 一次 CUDA 13 batch memcpy

命中后：
    统计异步更新,chunk 内 resident 保持固定布局
    新 forward/chunk 边界做 row-pool metadata-only promotion,并把 promoted row D2D 镜像到原层 decode 行
    bank 已被覆盖的层缓存 pending,下次加载时 host H2D 提交
```

关键参数：

```text
--kt-num-gpu-layers 10
--kt-expert-gpu-slots 1024
--kt-layer-h2d-slots 128    # 实测窗口；整层窗口 = 该层专家数 N(384 专家的模型取 384)
                            # 同时也是 host slot ring 的单代深度(ring = 2 * S,第 7.5 节)
                            # 未设置该参数时整个功能关闭,ring 退回历史 2 槽对
# 必需：--kt-gpu-prefill-token-threshold 2048（与 kt-layer-h2d-slots 联合门控）
```

一句话总结：

> 双缓冲 prepared-only staging 负责跨层搬运,resident pool 负责长期常驻；L 层 attention 后发起 L+1 预取,L+1 使用自己的历史命中表；staging 不占 resident 逻辑容量。每个 bank 为 `R_max + S` 行 prepared + `S` 行 raw H2D scratch；prepared-only 且 `num_gpu_experts > 0` 的层 raw canonical 留在 CPU（迁移未执行或 `num_gpu_experts == 0` 时仍可能占 GPU）,resident/staging 共用统一 row pool,promotion 只在 forward/chunk 边界更新,并在同 bank 内把 promoted row D2D 镜像到原层 decode 行、把 victim row 释放为 free。resident D2D 与 staging H2D 均按 FreeToken 风格批量提交,不可用时自动回退到原逐行/逐 run 路径。TP=2 时两个 rank 使用同一 logical 布局,各自只持有自己的 TP 权重分片。

