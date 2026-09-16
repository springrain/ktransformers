# Phase-2 计划:consumed_event 前置修复 + batch-dma w0→w1→w2 全阶梯 —— ✅ 代码落盘,静态验证全过

> 日期:2026-09-16。第 2 梯队 Prefill 史诗第一批,依第 0/1 梯队完成的归因基础设施派发。
> 范围裁定:consumed_event 前置修复(绑定硬前置,无 env)+ batch-dma w0→w1→w2 三档;
> **不含** event-fence、**不含** cpu-off-datapath(梯队 2 余下两项另批)。
> 上游依据:[ft-kt-optimization-tiers.md](ft-kt-optimization-tiers.md) §梯队2、
> [ft-kt-migration-audit.md](ft-kt-migration-audit.md) §6 定案。
> 纪律:env 治理矩阵全内联、默认关 = 逐位等价、不 commit、mask pinned 原位翻转契约零触碰、
> 专家数/层数全程参数化(代码/测试零生产拓扑魔数)。
> 行号均为落盘时快照(kt_ep_wrapper.py 6564 行 / 测试文件 1090 行),后续以符号锚点重定位。

## 0. 定位

两项绑定改造,编辑面**仅**两文件:

- `third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py`(fork 独有)
- **新建** `third_party/sglang/test/manual/kt_batch_dma_window_test.py`(CPU stub,N1–N11,无 GPU/无 dist)

1. **consumed_event 前置修复(步骤 A)**:`apply` 中 `consumed_event.record` 相对热更新对该槽 raw 字节的读是陈旧边。修复 = record 移到热更新 gloo 共识之后、`_prefetch_successor` 之前;`torch.cuda.synchronize`(热更新块首部)按审计绑定条款保留不可删。
2. **batch-dma w0→w1→w2(步骤 B/C/D)**:H2D 入列批化封装(w0)→ 单整层窗(w1)→ 双整层窗流水(w2),三档同代码路径,差 only 在 num_windows 与 parity 等待。仅动 `_Mxfp4LayerwisePrefillManager`/`SharedFullContext` 窗口分支;串行 legacy 路径逐字保全即回退面。

## 1. env 总表(全部内联 `os.environ.get`,environ.py 零 diff)

| env | 值域 | 默认 | 读取时机(锚点) | 语义 |
|---|---|---|---|---|
| `SGLANG_KT_PREFILL_BATCH_DMA` | {0,1} | 0 | 每次装载内联:`_kt_prefill_batch_dma_enabled` `:2536` | w0:仅 H2D 入列批化,零新共识零 DRAM |
| `SGLANG_KT_PREFILL_STAGE_LAYER_WINDOW` | {0,1,2} | 0 | `_stage_window_env_mode` `:2555` → init 经 `_negotiate_staging_window_mode` `:1102`(gloo MIN)一次性冻结;运行期改值→`_check_stage_window_drift` `:2572` 一次性 warn + 沿用冻结值 | 0=无窗 / 1=单整层窗 / 2=双窗流水 |
| `SGLANG_KT_PREFILL_BATCH_DMA_DEBUG` | {0,1} | 0 | `_build_staging_geometry` `:1129` 内联,经 `_any_tp_rank_true` `:4414` 冻结入 `debug_rows` | 调试首 4KB digest 钩子 |
| `SGLANG_KT_PREFILL_BYTE_CANARY` | {0,1} | 0 | `:1130` 同上 | 与 DEBUG 共用 digest 入口(登记别名) |
| `SGLANG_KT_PREFILL_NO_BATCH_MEMCPY` | — | — | 仅头部 env 治理段登记 reserved(:96) | phase-3(cudaMemcpyBatchAsync)预留,本期不实现 |

非法值统一 warn-once 按 0/回落。ownership-clone:`SGLANG_KT_SLOT_OWNERSHIP_ASSERT` 与 WINDOW 正交(双轴交叉见 N10)。

**w0 互斥裁决(取代审计字面断言)**:审计字面"BATCH_DMA=1 & WINDOW=0 互斥"的意图是防"双槽 SHM 几何上 host 批量写/单 fence"。w0 本身即该组合,裁决 w0 合法;互斥改由结构性不变量承载——
- `_assert_no_host_write_batch_on_dual_slot` `:2292`:batch submit 必须携带 `staging-window geometry`,缺失即 AssertionError(N11 双击)。
- host 写批化(submit-all + 单 sync)与整层共识只存在于 `geometry.mode >= 1` 的窗口分支;legacy/双槽分支逐字保全。
- 覆盖守卫:`_assert_h2d_plan_coverage` `:2301`(E-4.7)——任何 copy 之前 `Σ entry.rows == len(cpu_expert_ids) per bank`,否则 raise,sticky 零污染(N3)。

## 2. 架构(坐标不动)

- GPU D2D 段、raw_ready fence、postprocess、异常兜底整段位字未动;`_load_slot` 仅在 CPU 专家段打 dispatch:`geometry is not None` → `_load_window_cpu_experts`;否则 legacy 专家级循环逐字保留(其 H2D 入列点再按 BATCH_DMA 走 w0 dispatch)。
- 判定源 = 冻结几何(`self._staging_geometry is not None`),运行期不重读 env。
- 全局不变量:每层 device 共识计数 legacy/w0/w1/w2 均恰 2 次(范围从槽批升格整窗);`_window_generation` 进程级单调 uint64,`abort_round`/epoch 不回卷;TP0-only 仅 host 写与 H2D 入列,集合点零分叉;sticky 通道镜像 `pending_h2d_error` 模式,下层 setup commit 前 pop 并入,集合通信零新增。

## 3. 逐笔施工清单

### 步骤 A:consumed_event 前置修复

1. `__init__` 新增 `self._pending_fence_error: Optional[Exception] = None`(None-init,无防御式 getattr)。
2. `_record_slot_consumed(slot, main_stream)` `:3870-3891`:正常路径栅栏——record 与落位字段(`has_consumed_event=True`、guard→`"consumed"`、state→`"IN_USE"`、`current_slot_index`)逐字同 legacy;record 失败 → `main_stream.synchronize()` 兜底 + guard→`"synchronized"` + sticky 写入(仅 None 时)。
3. `_record_slot_consumed_from_exception` `:3893-3915`:旧 finally 逻辑搬入;`reuse_guard ∈ ("consumed","synchronized")` 早回——这是"恰 record 一次"的唯一机理(裁决 §6-9)。
4. `apply` `:3962-4037` 重排:finally 瘦身为 from_exception 调用;热更新 gloo commit 之后、`_prefetch_successor` 之前插入 `_record_slot_consumed`;`torch.cuda.synchronize(self.device)`(热更新块首行 `:3918`)保留。
5. sticky 并入:`_load_slot` transport setup commit 前 pop `_pending_fence_error`(`:3543-3545`),与本地 error first-non-None 合并提交;`abort_round` 清理段同步置 None。

### 步骤 B:w0(H2D 入列批化)

1. 头部 env 治理段(:79–:96)登记 5 枚 env(英文 ASCII,1–2 行/条)。
2. 共享静态帮手 `_shm_bank_buf_dtype` `:2215` / `_shm_bank_expert_nbytes` `:2223`(bf16 scale 特例逐字搬入)——裁决 §6-1。
3. msgspec 新容器:`_Mxfp4H2DCopyEntry{bank, src_row, dst_row, rows}`、`_StagingWindowGeometry{mode, num_windows, num_experts, bank_strides, bank_expert_nbytes, bank_merge_ok, total_nbytes, ...}`(`:2234` 区);常量 `_MIN_RUN_ENTRY_BYTES = 1 << 18` `:2210`。
4. `_Mxfp4H2DBatchPlanner.plan_layer` `:2260`:输入升序;相邻 id 且源/目的行均连续才并 run;**绝不跨 GPU 常驻洞合并**;`bank_merge_ok=False` 恒 singleton;planner 只读 mask 快照(原位翻转契约零触碰)。
5. `__init__` None-init 字段:`_h2d_planner/_staging_geometry/_window_free_events/_window_was_used/_window_generation/_window_release_consensus_gen/_window_owner/_window_freed/_pending_window_error/...`。
6. planner 惰性建于首轮(manager 创建早于 SHM);`_enqueue_planned_h2d` `:3167`:逐 entry `.copy_(non_blocking=True)`,按 entry 入账 bytes_h2d;was_used/event/失败兜底共用。
7. dispatch:BATCH_DMA=1 且 planner 非 None → planner 单例(w0 逐 (bank,position) singleton,字节/形状与 legacy 同构,N4 两跑集合并集逐位相等 + op 零 sync/零共识);BANK 覆盖断言先于一切 copy。

### 步骤 C:w1(单整层窗)

1. `SharedFullContext.__init__` None 字段;`initialize_cpu_buffers()` 内 `_create_cpu_buffers()` 后立即 `_create_staging_windows()` `:1285`(几何失败全 rank 统一 loud-fail 落 mode=0 + warn-once,字段保 None = 回退面不动)——裁决 §6-3。
2. `_negotiate_staging_window_mode` `:1102`:短路 `not dist.is_initialized() or world==1`;否则 int32 CPU 张量 `dist.all_reduce(op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)`(keyword 调用),降级 warn "negotiated down to %d ..."。
3. `_build_staging_geometry` `:1125-1215`:debug 冻结先行(集合语义先于局部硬断言,防单 rank 跳走分叉);per-bank `stride == num_experts * bank_expert_nbytes` 四硬断言:
   ①帮手公式 == 手工原式(含 bf16 特例)②16B 对齐 ③`nbytes >= _MIN_RUN_ENTRY_BYTES` 否则 `bank_merge_ok=False` + undersized warn(每次构造合并一行、列全欠尺寸 bank)④总行数 == `num_windows * num_experts`。
4. 容量预检 `_staging_window_capacity_ok`(四参纯函数,三源 `/proc/meminfo` MemAvailable + `/dev/shm` statvfs + RLIMIT_MEMLOCK 逻辑 AND);不足 → 协商降级统一落 w1/0。
5. pinned 分配 + `cudaHostRegister` + 跨 rank 指针收集 `_collect_all_rank_staging_pointers` `:1254` → unlink-after-collect → 冻结 `staging_window_mode`。
6. `_load_window_cpu_experts` `:3337-3425`(全 rank 对称主链):generation bump → `_window_pre_write_phase` `:3227`(`window_free_event.synchronize()` 计入 `h2d_stall_ms` → device MIN 共识写回 `_window_release_consensus_gen[win]`;FT check-before-write raise 文本 "staging window {win} recycled at generation {generation} before its reuse consensus for generation {generation - num_windows} completed") → `_ownership_record_window` `:3247`(W 断言:覆写前 `_window_freed[win]` 必真;A/B 槽哨兵窗口模式整体跳过 + 一次性 DEBUG skip note) → TP0-only `_submit_window_writes` `:3278`(geometry 缺失硬断言;submit ×len(ids) + 单 `sync_write_weight_scale_to_buffer()`;异常→`_pending_window_error`) → `_commit_tp_device_runtime_phase`(producer-ready 共识#2,sticky pop 并入) → `_window_debug_row` `:3313`(DEBUG/CANARY 冻结时每窗首 4KB digest;无组单 rank 每观察一行 DEBUG;有组 all_gather_object 比对,不等 → sticky "mismatch",不等进共识通道) → planner → coverage 断言 → `_enqueue_window_h2d`(finally was_used + event record)。
7. 账本:零新 JSON 键——host submit 入既有 `host_write_ms`、窗释放等待入既有 `h2d_stall_ms`、热更新入既有 `hotupdate_ms`;bytes 按 entry 行数入账,>10% 告警路径不变;`unaccounted_declared` 窗口模式追加说明行。N5 op 钉死:`[gen_bump, window_free.synchronize, consensus#1, ownershipW, submit×E(TP0), sync×1, consensus#2, copy×|runs|×4, event.record×4]`。

### 步骤 D:w2(双窗流水)

w1/w2 同代码路径,档差 only 三处:容量阈值 `{1:1×, 2:2×per_win}`(不足协商统一降级);窗口数组按 `num_windows` 建;parity 等待 `generation - num_windows`(w2 即 gen-2)。`abort_round` 不清 `_window_generation`/release 表(N6 不回卷钉死)。无新 env。

## 4. 测试映射(N1–N11,CPU stub 无 GPU;`python test/manual/kt_batch_dma_window_test.py`)

unit-test-admission 三族归类,逐用例"哪个未来 diff 使其变红"已可答:

| # | 族 | 断言对象 |
|---|---|---|
| N1 | 派生性质 | run 合并代数:并集 == popcount;跨 GPU 常驻洞合并恒 0;`bank_merge_ok=False` 恒 singleton |
| N2 | 派生性质 | 四条硬断言 + bf16 公式 `(512*512)*2 = 262144` 恰踩 `_MIN_RUN_ENTRY_BYTES` 验 `>=` 边界(裁决 §6-8);undersized warn 单次构造合并一行列全 bank(裁决 §6-7) |
| N3 | 关键路径簿记 | E-4.7 coverage guard 在任何 copy 前 raise;sticky 双通道零污染 |
| N4 | 派生性质 | w0 两跑 (bank,row) 集合并集逐位相等;singleton mask 下 op 与 legacy 完全同序;op_log 零 sync 零共识 |
| N5 | 关键路径簿记 | w1 op 严格序(§3-C.7);非 TP0 无 submit;每层恰 1 host sync + 恰 2 共识(裁决 §6-4);值级逐位 |
| N6 | 派生性质 | w2 `win = generation % num_windows` 交替;gen-2 reuse 断言在任何 submit 前 raise;`abort_round` 后 generation 不回卷 |
| N7 | 关键路径簿记 | 双 manager 脚本共识 op 逐行相等(除 TP0-only 段);两种失败模式均在同一 phase abort |
| N8 | 派生性质 | capacity 谓词纯函数四参:阈值-1 → False;三源各自独立短缺 → False;全足 → True;单 rank 不足 → 协商降级 |
| N9 | 关键路径簿记 | consumed 修复 op 序 `[compute_commit, update_sync, hot_update_reads, update_commit, consumed_record, prefetch_start]`;异常路径恰 record 一次;record 失败 sync 兜底 + sticky;下层 setup pop 并入;`abort_round` 清 sticky;动态更新关时 record 仍在 prefetch 前 |
| N10 | 关键路径簿记 | W 点火;dedup key=(code, layer, slot, epoch) warn-once;owner 不被违例覆写;level=1 返 RuntimeError;level=0 零副作用;A/B 窗口模式跳过 + skip note 一条 |
| N11 | 关键路径簿记 | env parse/非法值 warn-once;冻结值胜出 + drift warn 一次;MIN 协商脚本化;`_submit_window_writes` 缺 geometry raise;DEBUG=1 digest 每观察一行 + 脚本 mismatch → sticky |

回归盘:`kt_slot_ownership_assert_test.py` N1–N5、`kt_degrade_reason_test.py` N4/N5/N6 全绿。

## 5. 静态验证记录(2026-09-16,全部已执行)

1. `python -m py_compile` 两文件通过。
2. N1–N11 全绿;两条回归全绿(同 4 行)。
3. 魔数:`git diff HEAD` 全部新增行 `grep \b(256|24|41|61)\b` 零命中(N2 合成形状已参数化重排,字节数不变)。
4. env 七读取点全内联 `os.environ.get`;`python/sglang/srt/environ.py` diff = 0。
5. `_PREFILL_STATS_FIELDS_MS` 零 diff;新增行引号 JSON 键仅既有 `host_write_ms/hotupdate_ms`;`should_skip_expert(` 运行时原语零触碰。
6. 新增函数全部 ≤100 LOC(N2/N11 已按场景拆 `_n2_*`/`_n11_*`);测试文件 1090 行 <2k;注释英文 ASCII、子句边界断行;新容器 msgspec.Struct;None-init 无防御式 getattr;≥2 参调用按 keyword。
7. 本文档 + tiers 总表回写。

## 6. 裁定记录(施工期逐条)

1. **C-e5 字节公式**:bank 每专家字节 = `numel() // shape[0] × staging_dtype 宽度`,bf16 scale 特例(staging 恒 bf16 存,不论 f32 源)逐字归 `_shm_bank_expert_nbytes`;硬断言①把"帮手 == 手工原式"钉死,防双路径漂移。
2. **sticky 三通道分立 + 双 pop**:`_pending_h2d_error`(存量)、`_pending_window_error`(窗口)、`_pending_fence_error`(修复)三通道互不串;全部在下层 setup commit 前 pop 与本地 error first-non-None 合并提交;集合通信零新增。
3. **缺口①(几何冻结失败)调用点取舍**:`_create_staging_windows` 置于 `initialize_cpu_buffers()` 内 `_create_cpu_buffers()` 之后立即调用(manager 创建早于 SHM,故几何属 ctx 而非 manager `__init__`);失败全 rank 统一落 mode=0 + warn-once,字段保 None,legacy 路径逐字保全 = 零成本回退。
4. **N5 record×1 裁决**:每层恰 1 host sync(`sync_write_weight_scale_to_buffer`)+ 恰 2 device 共识;host submit 不再计入共识;branch 差额 = event.record × bank 数。
5. **w0 互斥裁决**:见 §1;字面互斥废弃,结构性断言承载意图。
6. **注解前向引用修复(本期唯一触及实现类的 bug)**:`SharedFullContext`(:425)早于 `_StagingWindowGeometry` 定义点(:2234),非字符串注解在类体执行即 NameError(py_compile 不查名);`:1125/:1217/:1296` 三处签名改字符串注解(`-> "_StagingWindowGeometry"` 等),风格同既有 `:456/:3283/:3314`。
7. **under-threshold warn 语义**:每次 `_build_staging_geometry` 调用合并一行、列出全部欠尺寸 bank(非进程级 once;init 只调一次故线上等价);N2 断言行内含双 bank 名守护"合并"性质。
8. **N2 形状 262144 等字节常量**:bf16 scale (8,512,512) → `512×512×2 = 262144 = _MIN_RUN_ENTRY_BYTES`,恰好踩阈值验证 `>=`(merge=True);w13 uint8 同值;w2_weight (8,1024,512) = 524288 高于阈值;uneven/tiny 层 `(4|8,512,128)` / `(8,4096)` 只做 stride 一致性,与阈值无关。
9. **恰 record 一次机理**:异常路径不再二次 record 由 from_exception 的 `guard ∈ ("consumed","synchronized")` 早回承载;guard 作 skip 键,无歧义。
10. **unaccounted 声明行**:窗口模式在 `unaccounted_declared` 追加 window run planner 说明,不改字段集。

## 7. 对拍门槛(实机验收,沿用 ownership-asserts §1.7 离线确定性纪律)

- 固定 8 prompt × (512 prefill + 64 decode),离线 Engine 或单并发串行;`chunked_prefill_size`/`max_running_requests=1` 钉死;双层间冲刷 radix cache;NCCL 版本/拓扑入库。
- 组合矩阵逐格:baseline(两 env 全 unset)vs `{BATCH_DMA=1}` / `{WINDOW=1}` / `{WINDOW=2}`;每格 dump 采样 KT 层 MoE 输出 + 最终 logits,sha256 逐位相等,容许差 = 0。
- layer 0 先甩手(非 KT 层),不参与 dump。
- 画像门(xsya10 实机):`transfer_stream` busy >50% 才按 P1 继续推进;否则整批收益不计入梯队报表。

## 8. 红线表(施工自查已逐条对账)

| # | 红线 | 状态 |
|---|---|---|
| R1 | 默认关 = 逐位等价(legacy 分支逐字保全) | ✅ dispatch 判定源 = 冻结几何;unset 走原路 |
| R2 | env 内联,environ.py 零 diff | ✅ 已核(§5-4) |
| R3 | 几何 env init 冻结,运行期改值一次性 warn + 沿用冻结值 | ✅ N11 |
| R4 | 代码/测试零生产拓扑魔数 | ✅ 已核(§5-3) |
| R5 | 集合点 rank 对称零分叉;TP0-only 仅 host 写 + 入列 | ✅ N7 |
| R6 | 每层 device 共识计数恒 2;集合通信零新增 | ✅ N5 |
| R7 | sticky 镜像 pending_h2d_error 模式,下层 commit 前 pop | ✅ N3/N9/N11 |
| R8 | mask pinned 原位翻转契约零触碰 | ✅ planner 只读快照 |
| R9 | 账本 JSON 字段零新增 | ✅ 已核(§5-5) |
| R10 | `torch.cuda.synchronize`(热更新块首行)不可删 | ✅ apply 重排保留 |
| R11 | NCCL barrier 不提供 host 序;禁 custom all-reduce | ✅ 复用既有 `_commit_tp_device_runtime_phase` |
| R12 | 传输不与热更新窗重叠 | ✅ 窗口分支与热更新窗互不交叠(既有坐标) |
| R13 | 注释英文 ASCII 1–2 行、子句断行;新容器 msgspec;None-init | ✅ 已核(§5-6) |

## 9. 实机尾项登记(本机不做,逐项待 xysa10)

1. nsys 画像门:transfer_stream busy >50% 才按 P1 报数(梯队 2 总表口径)。
2. Phase A 冻 mask bit-equal;Phase B byte canary 跑批;Phase C logits 对拍(§7 组合矩阵)。
3. 三档独立性能门槛:copy 流利用率 ≥85%、H2D ≥18GB/s、tok/s ≥1.25×;不过即 unset 该档。
4. NCCL 2-rank 30min 延迟注入 soak。
5. 热更新 ON 双请求 overlap sha256。
6. lazy-init 注册耗时(pin + cudaHostRegister 段)入库。
7. 回滚演练:`WINDOW=2 → unset` 重启即回 legacy;`BATCH_DMA=0` 热放生(w0 每次装载内联读)。
8. cudaMemcpyBatchAsync 三期评估(仅登记;env `SGLANG_KT_PREFILL_NO_BATCH_MEMCPY` 已预留)。

## 10. 回滚

- 热放生:`SGLANG_KT_PREFILL_BATCH_DMA` 每次装载内联读,置 0 即刻回逐专家入列。
- 窗口:geometry 于 init 冻结,改值须重启;unset/置 0 后 freeze 失败/容量不足/协商降级均统一落 legacy 分支(逐字保全,从未被裁掉任何一行)。
- 极端:last-resort = 两文件 diff 还原(编辑面闭合,零跨文件依赖)。
