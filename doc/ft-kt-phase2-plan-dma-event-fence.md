# Phase-2 计划:cpu-off-datapath(per-rank pinned bank)+ event-fence(E_chunk 环块)—— ✅ 代码落盘,静态验证全过;✅ 2026-09-17 参数化落盘(8 env → `--kt-*` CLI,两主开关默认开,§6-21);❌ 2026-09-17 用户裁定移除 cpu-off-datapath(bank)整支(pinned 镜像 ≈+1 份专家权重 RAM、单机专家桶 ≈×2 不可接受,§6-23);参数总表 8→4 枚,剩者全 fence 系;✅ 2026-09-17 fence×window 兜底复合落盘(§6-24,mutex 互斥废止,F8 翻转 + N12 钉死窗降级⇒ring 接管,四套件 30 测绿)

> 日期:2026-09-16。第 2 梯队 Prefill 史诗**第二批**,即梯队 2 余下两项,用户裁定**同批一次性**落盘。
> 上游依据:[ft-kt-optimization-tiers.md](ft-kt-optimization-tiers.md) §梯队2、
> [ft-kt-migration-audit.md](ft-kt-migration-audit.md) §6 定案、
> [ft-kt-phase2-plan-batch-dma.md](ft-kt-phase2-plan-batch-dma.md)(第一批)。event-fence 硬前置(consumed_event 前置修复)已随第一批入库。
> 纪律(2026-09-17 参数化裁定后,§6-21):8 枚开关全归 ServerArgs `--kt-*` CLI 参数(wrapper 直读 `get_exec().moe`、environ.py 零 diff 维持——无新增 env)、两主开关**默认开**、显式置 0(`--kt-<name> 0`)opt-out 时逐位等价、不 git add/commit、
> mask pinned 原位翻转契约零触碰、专家数/层数全程参数化(零生产拓扑魔数)。
> 行号均为参数化落盘后快照(kt_ep_wrapper.py 7516 行 / kt_bank_dma.py 394 行 / 测试 664+726 行),后续以符号锚点重定位;锚点漂移见 §6-15。(kt_bank_dma.py 与银行测试已随 §6-23 移除,其行号仅供查阅历史提交。)

## 0. 定位

两项绑定改造,编辑面六件(bank 打包器/bank 模块/bank 测试三件随 §6-23 移除):

- `kt-kernel/cpu_backend/cpuinfer.h` + `kt-kernel/ext_bindings.cpp`(步骤 B:task_tag 最小 ABI)
- `third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py`(F 环块全链;D 换源 dispatch 随 §6-23 移除)
- **新建** `third_party/sglang/test/manual/kt_event_fence_ring_test.py`(F1–F8,726 行,CPU stub 无 GPU/无 dist)
- `third_party/sglang/test/manual/kt_batch_dma_window_test.py`(两笔 fake 镜像补齐,非生产行为变更)

1. **cpu-off-datapath(D,步骤 C)**:❌ 2026-09-17 已整体移除(pinned 镜像 ≈+1 份专家权重 RAM 不可接受,§6-23);原条目叙述——消 rank0 代写(≈340GiB/chunk 的 TP0 双写环路)。每 rank 进程本地 pinned bank + pack 期预切片,装载期按 rank 切片直读,TP0 提交段整段跳过,H2D 换源 bank 行。(历史留档)
2. **event-fence(F,步骤 D)**:双槽环路逐专家共识(每层 2×E)→ E_chunk 环块(每层 2×⌈E/E_chunk⌉),复用 `_commit_tp_device_runtime_phase` 原语,禁自写集合原语。按先例现在施工:**默认开、显式置 0(`--kt-<name> 0`)opt-out 时逐位等价**(参数化裁定 §6-21),收益标 TBD,归因门与性能门槛登记 §9。

## 1. 参数总表(4 枚;全归 ServerArgs `--kt-*`,wrapper 直读 `get_exec().moe`,environ.py 零 diff——无新增 env;原 8 枚中 bank 系 4 枚随 §6-23 移除)

| CLI 参数(字段,`NS("exec.moe")`) | 值域 | 默认 | 读取时机(锚点) | 语义 |
|---|---|---|---|---|
| `--kt-prefill-event-fence 0|1`(`kt_prefill_event_fence`,int 注解推导 type + `Arg(choices=[0,1])`) | {0,1} | **1** | `_event_fence_env_enabled` `:3050`(直读袋)→ init `_negotiate_event_fence` `:1468` intent MIN 冻结;CLI 参数运行期不可变(drift 机制废止,§6-20/§6-21) | 1=E_chunk 环块;0⇒分配尺寸/协议逐位冻结,子参数非默认恰一行 ignored warn(server_args `_validate_kt_args` `:4385` 承担,R20) |
| `--kt-prefill-stage-chunk-experts`(`kt_prefill_stage_chunk_experts`,int) | {0,16,32,64}(argparse choices 启动期拦截非法值,原非法值 warn 链随删) | 64 | `_event_fence_chunk_env` `:3056`(直读袋);折叠 `intent = fence and chunk != 0` | 探测失败降级阶梯 64→32→16→1(MIN 统一);chunk=1 ≡ legacy 2×E 节拍(48h 退火基线,choices 不含 1,由阶梯地板达成,§7) |
| `--kt-prefill-no-device-sync 0|1`(`kt_prefill_no_device_sync`,int 注解推导 type + `Arg(choices=[0,1])`) | {0,1} | 0 | 一期恒按 0 处理(grep 门白名单本期为空);=1 恰一行 inert warn(`_validate_kt_args` 承担) | 独立二分预留;fence 关时计入 ignored warn |
| `--kt-prefill-fence-debug 0|1`(`kt_prefill_fence_debug`,int 注解推导 type + `Arg(choices=[0,1])`) | {0,1} | 0 | `_fence_debug_env_local` `:3062`(直读袋)→ `_any_tp_rank_true` MAX 冻结入 `geometry.debug_rows` | 每块首 4KB head digest all_gather_object 对拍;mismatch 入 sticky |
**复合矩阵(2026-09-17 §6-24 兜底复合,原互斥矩阵废止)**:FENCE×WINDOW≥1 ⇒ **双几何皆冻结、协商零互斥、零 warn**;窗口健康 ⇒ window 道独占 host 传输(共识账本 2/层逐位不变),冻结环 standby(环行 SHM 一次性 standby 占用);窗口软降级 ⇒ **ring 接管**(2×⌈E/chunk⌉)而非 legacy;深复合(窗内分块流水)显式出范围。FENCE 不接 INT4/FP8/BF16(仅 MXFP4)。

### 1.1 参数逐枚说明(白话版:叫什么、管什么、本期干什么)

> 形态总纲(**2026-09-17 用户三裁**):开关类一律 `0|1` 显式取值,0=关 1=开,int 字段 `Arg(choices=[0,1])`;不派生 `--no-*` 形态,显式关闭统一写作 `--kt-<name> 0`;唯一非开关是 chunk(值域 {0,16,32,64})。读法:wrapper 直读 `get_exec().moe.<field>`;CLI 参数冻结于启动、运行期不可变(drift 机制随参数化废止,§6-20/§6-21)。

- **`--kt-prefill-event-fence 0|1`(默认 1)**——**事件栅栏环块传输总闸**。CPU 专家按 chunk 尺寸成块经双环搬运,每层付 2×⌈E/chunk⌉ 次设备共识,替代 legacy 的 2×E;控制面收紧是恢复线速的一号手段(§2.2/§6-14)。intent MIN 每 rank 恒跑(含 0)防 collective 挂死;显式 0 ⇒ 分配尺寸与协议逐位冻结(R20),三个子参数被忽略恰一行 warn(prepare `_validate_kt_args` 承担)。与 window≥1 兜底复合(§6-24:双几何皆冻结、协商零互斥零 warn,环 standby;窗降级 ⇒ ring 接管);INT4/FP8/BF16 层不接。
- **`--kt-prefill-stage-chunk-experts={0,16,32,64}`(默认 64,非开关)**——**环块尺寸**。0 折叠 fence intent 为 off(结构同一性,F8);容量探测失败走 64→32→16→1 降级阶梯,全 rank MIN 统一(F5),地板 = legacy 双槽大小(chunk=1 ≡ legacy 节拍,48h 退火基线;choices 不含 1,由阶梯达成)。非法值 argparse 启动期拦截。
- **`--kt-prefill-no-device-sync 0|1`(默认 0)**——**护栏同步精减的二分总闸(预留)**。payload 是「去掉守卫式 host 阻塞同步」的 grep 白名单门:二期逐个评估裁撤哪根护栏、出竞态时二分定位哪根在救命。R10 红线已钉死两处热更新全设备同步**不可移除**,本闸与护栏语义正交,只覆盖白名单登记的其余点位。**本期白名单为空 ⇒ 恒按 0 处理、生产零消费点;传 1 仅启动期恰一行 inert warn**(`_validate_kt_args` 承担);fence 关时计入子参数 ignored warn。
- **`--kt-prefill-fence-debug 0|1`(默认 0)**——**环块取证对拍器**。开时每个环块取首 4KB head digest,all_gather_object 跨 rank 对拍,mismatch 记 sticky(一次性弹出,F7);取值 init 期经 all-rank MAX 冻结入 `geometry.debug_rows`,rank 间永不一半开一半关。纯诊断钩,不改传输语义。
> 消费点速查:fence `_event_fence_env_enabled`(直读袋)→ intent MIN;chunk `_event_fence_chunk_env` 折叠语义;fence-debug `_fence_debug_env_local` → `_any_tp_rank_true` MAX 冻结;no-device-sync 本期零消费(prepare 单读校验)。

## 2. 架构

### 2.1 D:per-rank pinned bank

> ❌ 2026-09-17 bank 特性整体移除(pinned 镜像 ≈+1 份专家权重 RAM 不可接受,§6-23);原 §2.1 设计叙述随删,本节留名占位。

### 2.2 F:E_chunk 环块

- msgspec `_Mxfp4RingGeometry{e_chunk, num_slots=2, ring_rows=2*e_chunk, num_experts, bank_expert_nbytes, total_nbytes, debug_rows}` `:2693`,冻结于 init。
- 协商编舞 `_negotiate_event_fence` `:1441-1481`(在 `_create_cpu_buffers()` 前):`chunk_requested` → `intent = fence 参数 and chunk != 0` → intent 经 `_all_tp_ranks_succeeded` MIN(**参数关也走**,集合序列零分叉)→ frozen 真 ⇒ `_ring_bank_expert_nbytes(` `:1399`,16B 硬断言)→ `_ring_chunk_candidates` `:1422`(阶梯 64→32→16→1,探测失败落 1)→ chunk 经 `_tp_int_min_all_reduce` MIN(降档恰一行 warn)→ debug `_any_tp_rank_true` MAX → 冻结 geometry + `_event_fence_frozen`。(**§6-24 注记**:mutex 检查段与提前 `_staging_window_mode_frozen()` 调用随解禁删除;协商自此不消费窗口 knob,窗口 MIN 恒单站点于 `_create_staging_windows`。)
- `_create_cpu_buffers`:FENCE=1 分支升 `ring_rows = 2*e_chunk` 行(register/指针收集/unlink 段不变);FENCE=0 分支**逐字**;行数经模块级纯助手 `_ring_row_count(geometry)`(裁定 §6-7)。
- 块内 op 严格序(镜像 window):begin-fence((epoch,generation) key 进程单调不回卷 + live-without-owner assert)→ 环块共识#1(sticky pop 并入;相位文本 `ring slot {s} reuse for layer {L} generation {g}`;过期 free-event sync 计 `h2d_stall_ms`)→ ownership assert R(A/B 哨兵结构性 inert + 一次性 skip note)→ TP0-only submit×块长 + 单 `sync_write_weight_scale_to_buffer()`(计入既有 `host_write_ms`)→ 共识#2(sticky pop 并入;`host writes for layer {L} block {i}`)→ debug_rows ⇒ 每块恰一条 4KB head digest → H2D 平拷(`destination[expert_id].copy_(cpu_buffer[src_row], non_blocking=True)`,`src_row = slot*e_chunk + position`;bytes_h2d 仅整块成功后入账)→ finally `was_used=True` 后 `event.record(transfer_stream)`(发布失败回退本地 `transfer_stream.synchronize()`)。
- sticky `_pending_ring_error` 镜像 `pending_h2d_error` 模式:块首共识#1 pop 并入、写共识#2 pop 并入、`_load_slot` raw-fence pop 孪生;enqueue 失败入 sticky,下块 raise 出环;末块 last_error 由 raw-fence commit 收。`abort_round` 清 sticky/shadow 镜像;`_ring_generation` 进程级单调不回卷(`_advance_round` 漂移孪生 warn 随参数化废止,§6-21)。
- **共识计数精确口径**:legacy = 2×E;FENCE=1 ⇒ 2×⌈E/E_chunk⌉;window(w1/w2)= 2/层。集合通信运行时**零新增**。
- **init 集合计数增量**:intent MIN(+1)+ chunk MIN(+1)+ debug MAX(+1)= 相对 legacy init +1～+3 次(fence 参数关时恰 +1);程序内窗口 MIN(mode≥1 时)恒由 `_create_staging_windows` 单站点跑一次,fence 协商不再触发(§6-24),init 集合总数零分叉。实机入库复核(尾项④)。

## 3. 逐笔施工清单

### 步骤 B(C++ 最小 ABI)

`cpuinfer.h`:worker 异常将 `task_tag` 解 (kind, expert_id) 前缀进 `what()`;Args 默认 0,热路径零成本。`ext_bindings.cpp`:Args 尾部追加参数(双 submit 重载合一、inner submit 专用 `delete args_` 路径同步)。旧 binding ⇒ D8 raise "Rebuild kt-kernel"(探针 + 闩锁 + skip 阀门);**aarch64 重编译登记尾项①**。(**注记**,§6-23 后:task_tag ABI 保留,C++ 零改动,pybind 默认 0 透传;D8 探针测试随 bank 测试文件消亡,实机编译确认并入尾项①)

### 步骤 D(F wrapper 侧)

`__init__` ring 七字段 None-init(`_ring_generation/_ring_free_events/_ring_was_used/_ring_owner/_ring_freed/_ring_last_chunk_key/_pending_ring_error` + `_ring_skip_note_logged`);msgspec `_Mxfp4RingGeometry`;`_mxfp4_prefill_expert_bytes` 除数推广 `shape[0]`(裁定 §6-8);参数直读助手(袋读 `:3050-3065` 区;原七枚 latch 与 drift 三检查/三基线字段随参数化整体删除,§6-21,mutex latch 后续随 §6-24 一并删除);`_negotiate_event_fence` 编舞;`_create_cpu_buffers` FENCE=1 分支;环块方法族(begin-fence / `_ring_pre_write_phase` / `_ownership_record_ring` / `_submit_ring_writes` / `_ring_debug_row` / `_enqueue_ring_block` / `_load_ring_cpu_experts`);`_load_slot` dispatch `elif ring_geometry is not None:`(`:4649`,window 后 legacy 前);raw-fence pop 孪生;`abort_round` 孪生清洗。`_submit_host_write` 闭包 `numel() // 2` 不改(裁定 §6-9);账本零新 JSON 字段。

## 4. 测试映射(F1–F8;CPU stub,无 GPU 无 dist;D1–D9 随 §6-23 消亡,编号留档)

unit-test-admission 三族归类。回归盘:`kt_batch_dma_window_test.py` N1–N11、`kt_slot_ownership_assert_test.py` N1–N5、`kt_degrade_reason_test.py` N4/N5/N6 全绿(批窗测试 fake 补 ring 镜像两笔,裁定 §6-16)。

| # | 族 | 断言对象 |
|---|---|---|
| D8 | — | (随 §6-23 **消亡**:task_tag ABI 保留(C++ 零改动、pybind 默认 0 透传),探针测试随 bank 测试文件整删;实机编译确认并入尾项①,编号留档) |
| F1 | 派生性质 | `_ring_blocks` 分区代数:ceil 拆分、末块短、空输入、单块 |
| F2 | 关键路径簿记 | 共识计数 2×⌈E/e_chunk⌉(chunk=1 ≡ legacy 2×E);`_load_slot` dispatch 序审计(window < ring < legacy) |
| F3 | 关键路径簿记 | 块内金色 op 序;非 TP0 零 submit;sticky 块首 pop 恰一次 raise 出环;环行字节逐位(src_row 代数) |
| F4 | 派生性质 | `--kt-prefill-event-fence 0` 结构同一性:恰一次 intent MIN、geometry None、`_ring_row_count(None)==2`、零 warn |
| F5 | 派生性质 | 阶梯:探测 None 落 1、逐档阈值、跳过高于 request 的档(非法值 argparse choices 启动期拦截,原 warn 段随删);peer MIN 降档恰一行 warn、MIN 恰一次 |
| F6 | — | (随 env→CLI 迁移**删除**:no-device-sync 一期恒 False 语义与 inert warn 已上收 server_args `_validate_kt_args`,无 wrapper 侧对象可测;fence 关子参数 ignored warn 同属 prepare 校验面,§6-21;编号跳档) |
| F7 | 关键路径簿记 | DEBUG MAX 冻结恰一次;单 rank 每块恰一条 digest;cross-rank mismatch 入 sticky 不就地 raise |
| F8 | 派生性质 | chunk=0 折叠进 intent(frozen off,`--kt-prefill-stage-chunk-experts=0`);复合断言(§6-24):window≥1 激活下 fence 协商照常冻结全件 geometry、零 warn、窗口 MIN 缓存不预热(单站点钉死);N12 钉死窗降级 ⇒ ring 接管而非 legacy |

## 5. 静态验证记录(2026-09-16,全部已执行)

1. `python -m py_compile` 全触件通过。
2. F1–F8 / N1–N11 / N1–N5 / N4–N6 四套件一次复跑全绿(D1–D9 随 §6-23 消亡)。
3. 魔数:本批 wrapper + 测试全部新增行 `grep \b(256|24|41|61)\b` 零命中。
4. `python/sglang/srt/environ.py` diff = 0(env 时代即零 diff;参数化后 8 名 env 全仓清零,见 10)。
5. `_PREFILL_STATS_FIELDS_MS` 等账本字段零 diff;零新 JSON 键(host submit 入既有 `host_write_ms`、ring 复用等待入既有 `h2d_stall_ms`、H2D 入列入既有 `h2d_enqueue_ms`)。
6. 新容器 msgspec.Struct;None-init 无防御式 getattr;函数 ≤100 LOC;注释英文 ASCII 子句断行;≥2 参调用按 keyword;两测试文件 715/778 行 <2k。
7. C++ 两文件 diff 逐行审(task_tag 热路径零成本)。
8. 本文档 + tiers 总表回写(梯队 2 行 / event-fence)。
9. **审查修复复核(同日 §6-17~20 落地后)**:py_compile 全触件通过;F1–F8 / N1–N11 / N1–N5 / N4–N6 四套件一次复跑全绿(含 F8 折叠/mutex 基线断言);新增行魔数扫描零命中;environ.py 零 diff 维持。
10. **参数化复核(2026-09-17,§6-21 裁定落地后)**:py_compile 全触件通过(server_args / wrapper / 两测试);四套件一次复跑全绿(F6 随迁移删除、编号跳档);8 名 env(`SGLANG_KT_{PREFILL_EVENT_FENCE,PREFILL_STAGE_CHUNK_EXPERTS,PREFILL_NO_DEVICE_SYNC,PREFILL_FENCE_DEBUG,DIRECT_BANK_DMA,DUMP_SLOT_BYTES,BANK_DMA_BATCH,BANK_DMA_LEAN}`)全仓 + 主仓 bench grep 零命中(仅历史叙述段除外:本文 §6-19 与 migration-audit 两文档,均已标注为历史);新增行魔数 `\b(256|24|41|61)\b` 扫描零命中;environ.py 零 diff 维持。**追记(§6-23,2026-09-17)**:bank 系 4 枚参数(kt_direct_bank_dma/kt_dump_slot_bytes/kt_bank_dma_batch/kt_bank_dma_lean)与 D1–D9 随 bank 移除整体删除;现行开关 4 枚全 fence 系。

## 6. 裁定记录(施工期逐条;含计划↔落盘锚点漂移)

1. **批内序与范围**:用户裁定 D+F 同批一次性;event-fence 按先例现在施工(默认关=逐位等价,收益 TBD,归因门+门槛登记尾项);pretiled 前置降级自带,tier-5 本体维持🔒。**(§6-23:bank 特性整体移除,本条 D 侧部分作废——历史留档)**
2. **d2 容量阈值边界 (75,80)→(50,80)+(51,80)**:预检为 ≤/不足判定,期望值修正为两源各自独立短缺两条(D2 双用例)。**#18 裁定起 NUMA 腿改 sum 语义**,该边界期望值由 §6-18 重述。**(§6-23:bank 特性整体移除,本条作废——历史留档)**
3. **D8/N5 ABI 探针同族**:`kt_degrade_reason_test.py` N5 与 D8 复用同一探针族(旧 doc raise/新 doc pass/闩锁/skip 阀门),文案 settle。
4. **weights 全量 hash 启动期不可行**:改 pack-time 算 hash + 启动廉价键(dims/map sha256/tile 表)+ DUMP 金丝雀;偏离逐字公示于此。**(§6-23:bank 特性整体移除,本条作废——历史留档)**
5. **RUN 冻结失败语义**:bank 校验失败/容量不足/NUMA 不符/协商分叉 → 恰一行 reason literal warn 强制落 0,不全 rank 崩(仅校验串 raise 属共识语义)。**(§6-23:bank 特性整体移除,本条作废——历史留档)**
6. **chunk=0 折叠进 intent**:`intent = fence_env and chunk_env != 0`;MIN 跑在折叠后 intent 上,统一落 0(F8);不单独为 chunk=0 临时静音 intent MIN(集合序列零分叉纪律)。
7. **`_ring_row_count` 偏差记录**:计划原裁定"不抽助手";落盘引入模块级纯助手(`_create_cpu_buffers` 行数公式现于数组维度与总量账目两处,消除双写)。纯函数零副作用,裁定意图(FENCE=0 分支逐字)不受影响。
8. **`_mxfp4_prefill_expert_bytes` 除数推广**:字面 2 → `numel() // cpu_buffers[name].shape[0]`:FENCE=1 时 shape[0]=ring_rows;FENCE=0 shape[0]==2,逐位等价。
9. **`_submit_host_write` 闭包 `numel() // 2` 不改**:INT4 packed 专属口径;ring 永不调它。运行时原语零触碰。
10. **NO_DEVICE_SYNC 一期 inert**:grep 门白名单枚举为空 ⇒ 恒返 False;=1 恰一行 inert warn;二期填白名单再激活(guard 护栏类 sync `：：3925/:6201/:2159/:2198` 类永不入白名单,R10)。
11. **BATCH_DMA × ring**:ring 路径忽略 BATCH_DMA(ring 不建 planner、不经 `_enqueue_planned_h2d`)。(**§6-23**:BANK×FENCE 双开 `src_row = expert_id` 分支随 bank 移除,本条 bank 半句作废——历史留档)
12. **A/B 槽哨兵 ring 模式 inert**:结构性跳过 + 一次性 DEBUG skip note;assert R(`_ownership_record_ring`)接管环槽复用纪律(F3 金色序 ownershipR 位)。
13. **begin-fence 三键**:(epoch, generation) key 进程单调不回卷(双请求 overlap 下跨请求复用窗口承诺)+ live-without-owner;`abort_round` 不清 generation。
14. **FENCE init 集合计数增量 +1～+3**:§2.2 明细;env=0 时恰 +1(intent MIN),旧四条 prepare 环路 barrier(INT4/FP8/BF16)与本批无交、留。
15. **行号快照(后续以符号锚点重定位)**:`_negotiate_event_fence` `:1469-1516`、`_staging_window_mode_frozen` `:1417`、`_ring_bank_expert_nbytes` `:1427`、`_ring_chunk_candidates` `:1450`、环块方法族 `_ring_debug_row` `:4232`/`_load_ring_cpu_experts` `:4333`、`_create_cpu_buffers`/abort 漂移区、dispatch `:4636-4664`、env 助手 `:3076-3219` 区、`_Mxfp4RingGeometry` `:2693`。
16. **批窗测试 fake 两笔**:`_make_window_manager` 与 N9 手工块补 ring 八字段(生产 `abort_round` 环道清洗无条件——禁防御式 getattr 纪律不松,伪装备适配是唯一正解)。
17. **GC 回收模板 + 顶层导入正名**(审查修复 #1/#2):计划原写的「半建失败 drain transfer_stream → unregister → free」模板与落盘实现不符——落盘为 `torch.empty(pin_memory=True)` 逐字段分配 + copy + view,半建失败置 `tensors={}` + `gc.collect()` 后 raise,cudaHostAlloc 经 pin_memory 路径无 unregister 簿记,GC 即完整回收(R18 语义不变:bank 永不 free,半建品也不例外)。capability 侧 `_bank_dma` 为**顶层导入**而非懒导入:msgspec/torch 本就由 wrapper 加载,顶层导入无循环导入与 CUDA init 风险,unset 同样零成本。文档 §2.1/§3-C/§4-D7/§10/R18 行同步改写;D7 原宣称的「abort/重复 init 张量 id 不变;半建失败调用序」两条经全文件 grep 坐实**测试未落**,表行改为实机/后续测试尾项。**(§6-23:bank 特性整体移除,本条作废——历史留档)**
18. **NUMA 腿改 sum 语义**(审查修复 #4):`_bank_capacity_precheck` NUMA 腿由「均分假设 per_node=⌈total/len⌉、任一节点 <per_node 即拒」改为「**各节点空闲之和 < total 即拒**」。理由:①Linux 默认 NUMA 策略页随首次触分配、本地节点不足时跨节点 fallback,真实容量约束是聚合空闲而非均摊假设——落盘的逐字段 fill 并无均分实现背书,均分判据会误杀 (free0=0, free1≥total) 这类完全可服务布局;②「任一节点 ≥ total」反向判据则过松;③单节点 len=1 时 sum ≡ total,精确退化。D2 期望同步改写:(50,80)/(51,80) 现为 True,新增 (20,80) False 边界与判别用例 (0,200) True;§6-2 阈值边界由本条重述。容量/NUMA 实测仍登记尾项⑦。**(§6-23:bank 特性整体移除,本条作废——历史留档)**
19. **reserved env 模块级登记补齐**(审查修复 #5):治理段所列 `SGLANG_KT_BANK_DMA_BATCH/LEAN` 原仅定义 `_kt_bank_reserved_env` 助手而**零调用点**,非法值永不 warn,「治理段登记」未兑现。修复:kt_bank_dma.py 模块级在助手定义后直接自调用两枚(import wrapper 即触发),非法值恰一行 warn 按 0;D4 补 bogus 恰一行 + latch 幂等断言。(**历史叙述**:env 时代的审查修复;§6-21 参数化后两枚 reserved 归 server_args 登记、不消费不 warn,上述助手/登记段/latch 整体删除)**(§6-23:bank 特性整体移除,本条作废——历史留档)**
20. **drift 基线字段**(审查修复 #3):三枚 drift 检查原以**冻结值**为基准对比当前 env——mutex veto(F8 坐实 calls==[False,True])、chunk=0 折叠、intent MIN 降 0、chunk MIN/阶梯降档场景下 frozen=False/降档值而 env 未动,`_advance_round` 首进即误报「changed from 0 to 1」型 warn(恰一行进程级,纯日志误导)。修复:`SharedFullContext.__init__` 补三枚 None-init 基线 `_bank_env_baseline`/`_event_fence_env_baseline`/`_event_fence_chunk_env_baseline`;`_init_rank_bank`/`_negotiate_event_fence` **开头**记本地 env 解析值(mutex/早退分支前已赋值,误报免疫);`_advance_round` 三调用点改 `if baseline is not None: _check_xxx(baseline)`(None 纪律化检查;chunk 检保留 `ring_geometry is not None` 门——冻结环恒有基线);三检查函数签名/本体零改(本就是「给基准对 env」)。warn 语义固定为「与 **init 时本地 env 解析值**比较;冻结值(MIN/veto 后果)生效至重启」。D4/F8 补 mutex/折叠场景基线值断言 + drift 零 warn 断言;D7/F8 原漂移双向断言语义不变。(**注记**:§6-21 参数化裁定后 CLI 参数运行期不可变、漂移通道不复存在,本条机制连同三检查/三 latch/三基线字段/`_advance_round` 三调用点**整体删除**,D7 与其 drift 断言段随删;#3 修复成果随机制废止)
21. **8 枚 env → `--kt-*` CLI 参数,两主开关默认开**(用户裁定 2026-09-17,原话:「就是使用参数的方式设置,不使用环境变量的方式」;命名 `--kt-` 开头中横线连接):env 读取路径**整体删除不保留兼容**。落盘:`kt_direct_bank_dma`/`kt_prefill_event_fence` 为 int 字段默认 **1**,argparse `type=int` 由字段注解推导(arg_utils `_infer_type_func`;`Arg` 无 `type` kwarg,注解形态与 chunk 字段一致),显式 `Arg(choices=[0,1])` 取值 0/1(**2026-09-17 用户追裁:「--no-\* 不需要这样的参数」**,故不派生 `--no-*` 关档,显式关闭统一写作 `--kt-<name> 0`,与 env 时代 0/1 取值同构;原 BooleanOptionalAction 施工验证点①随本次追裁作废);其余五枚开关亦统一 0|1 显式取值(**2026-09-17 用户三裁:「--kt-prefill-no-device-sync 类似这样的开关 取值都用 0 和1,0是关闭,1 是开启」**),int 字段默认 **0**,与两主开关同形态(store_true 派生形态整批弃用);chunk 用 `Arg(choices=[0,16,32,64])` 启动期拦截非法值,原非法值 warn-once 链随删;八字段全 `NS("exec.moe")` + `[ktransformers parameter]` help 前缀。**语义翻转**:R1 由「默认关 = 逐位等价」改述「**默认开;显式置 0 opt-out 时逐位等价**」;BANK 默认开而无 `<weight_path>/bank/manifest.json` 时每启动恰一行 `manifest_unreadable` 降级 warn 强制落 0 = **预期常态**(D4 扶正为正测;§7 baseline 口径随改)。**drift 机制整体废止**:CLI 参数冻结于启动、运行期不可变,三检查函数/三 latch/三基线字段/`_advance_round` 三调用点全删(#3 修复成果随机制废止,§6-20 注记)。**职责上收 prepare**:原 `_event_fence_subenvs_ignored`(fence 关且 chunk≠64/no_device_sync/fence_debug 任一非默认 → 恰一行 ignored warn)与 no_device_sync 的 inert warn 均由 server_args `_validate_kt_args`(`:4385`)承担;`_kt_prefill_no_device_sync_enabled` 坐实生产零调用点,整体删除;`_event_fence_mutex_warned` 单 latch 与 `_check_stage_window_drift` **保留**(WINDOW `SGLANG_KT_PREFILL_STAGE_LAYER_WINDOW` 属第一梯队,仍是 env)。kt_bank_dma.py 恢复纯函数零 sglang 依赖,消费全部上收 wrapper 侧直读 `get_exec().moe`(`:3050-3065`/`:1523`/`:4691` 等;model init 期 get_exec 就绪先例 fused_moe_triton/layer.py:441)。本条对 R1/R2/R3/R20 的行级重述见 §8 红线表;§0~§5 各表与 §7/§9/§10 文案同步改于 2026-09-17。**追记(§6-23)**:8 枚 → 4 枚(bank 系 4 枚随删);「两主开关默认开」此后仅指 `--kt-prefill-event-fence` 一枚。
22. **tier-5 pretiled-banks 判死**(2026-09-17,用户二次拍板,与下条同裁定):常驻 +1 份专家权重镜像的架构约束对 pretiled-banks 适用度更强;其前置四件(pack 打包器/manifest 校验/容量预检/热更条款)已随 bank 整支删除归零,本体判死 ❌,移入 [ft-kt-optimization-tiers.md](ft-kt-optimization-tiers.md)「明确不做」节;**架构约束落字:pinned 专家镜像预算为零,任何常驻 +1 份专家权重的设计不再评**。
23. **bank(cpu-off-datapath)整支移除**(2026-09-17 用户裁定逐字:「如果确定 bank 会造成内存翻倍,就去掉bank相关的功能和参数」):前提坐实——per-rank pinned bank = 全 TP 组合计 +1 份专家权重 RAM(widened 镜像),单机部署专家桶 ≈×2(store 因热更/fallback 不可省)。切除范围:bank 系 4 枚参数(`kt_direct_bank_dma`/`kt_dump_slot_bytes`/`kt_bank_dma_batch`/`kt_bank_dma_lean`)+ `kt_bank_dma.py` 整件 + wrapper bank 功能区/换源分支/DUMP 钩子 + init 两集合点(1×MIN + 1×all_gather_object,init 集合计数恢复批量二之前基线;运行期本就零新增)+ D1–D9 测试整件 + bench `pack-bank` 子命令全删;task_tag ABI 保留(C++ 零改动、pybind 默认 0 透传,aarch64 重编需求零新增,实机确认并入尾项①)。staging "field-bank" 术语(`_shm_bank_*`/`bank_expert_nbytes`/`bank_strides`/benchbw 的 RAW_BANK 语义)与 event-fence 全链、batch-dma、tier-1 基建**不动**;environ.py 零 diff 维持;mask pinned 原位翻转契约零触碰。本条生效后 §0~§11 中凡 D 系条目(D1–D9、步骤 A/C、§2.1、R14/R15/R18、尾项 #2/#7、§9 #9 之 bank 两项、§10 bank 行、§11 二期 bank 行)全部作废,正文以删行或随条注记落实。**复活路径:将来若要重评 bank,`git revert <本切除提交>` 即整支回来**。

## 7. 对拍门槛(实机验收,沿用 ownership-asserts §1.7 离线确定性纪律)

- 固定 8 prompt × (512 prefill + 64 decode),离线 Engine 单并发;`chunked_prefill_size`、`max_running_requests=1` 钉死;**热更 ON 强制**;双层间冲刷 radix cache;NCCL 版本/拓扑入库。
- 组合矩阵逐格:baseline(`--kt-prefill-event-fence 0` 显式置 0 = legacy 逐字路径)vs `{FENCE}`(**= 默认开档,零参数**)/ `{FENCE+NO_DEVICE_SYNC}` / `{FENCE}×WINDOW{0,1,2}`;每格 dump KT 层 MoE 输出 + 最终 logits,sha256 逐位相等,**容许差 = 0** + 每层 device 共识计数断言(= 2×⌈E/E_chunk⌉)。(§6-23:{BANK}/{BANK+FENCE} 格与 manifest 常态降级口径随 bank 移除删除)
- 单 rank kill 注入;E_chunk 退火 1×48h → 16/32/64×24h(chunk=1 为 48h 退火基线;choices={0,16,32,64} 不含 1,1 档由容量阶梯地板达成——探测失败 MIN 落 1)。
- 带宽门 **≥17GB/s**(<11 即回落);画像门 `transfer_stream` busy >50% 才继续 P1。
- 收益 gain 标 **TBD**(排期上限 +25%;**不可加警告**:与 batch-dma 收割同一缺口,严禁按宣称值叠加,合并上限 ≈ 传输段恢复线速)。

## 8. 红线表(施工自查已逐条对账)

| # | 红线 | 状态 |
|---|---|---|
| R1 | **默认开;显式置 0 opt-out 时逐位等价**(§6-21 语义翻转) | ✅ F4(参数关恰一次 intent MIN、geometry None、零 warn) |
| R2 | 参数归 ServerArgs `--kt-*`;environ.py 零 diff(无新增 env) | ✅ §5-4/§5-10 |
| R3 | 几何/开关协商期 MIN/MAX 冻结;CLI 参数运行期不可变,drift 机制废止(§6-20 注记/§6-21) | ✅ F4/F5/F7 冻结面;drift 用例随机制删除 |
| R4 | 代码/测试零生产拓扑魔数 | ✅ §5-3 |
| R5 | 集合点 rank 对称零分叉;TP0-only 仅提交段 | ✅ F3 tp1 spine |
| R6 | 共识计数钉死;集合通信运行时零新增 | ✅ F2(2×⌈E/E_chunk⌉、chunk=1 ≡ 2×E) |
| R7 | sticky 镜像、generation 不回卷 | ✅ F3(sticky 恰一次出环);`_ring_generation` 进程单调 |
| R8 | mask pinned 原位翻转契约零触碰 | ✅ 本批零涉 `:5431/:6509` 契约点 |
| R9 | 账本 JSON 字段零新增 | ✅ §5-5 |
| R10 | 护栏 sync(`:3925/:6201/:2159/:2198` 类)不可移除 | ✅ NO_DEVICE_SYNC 跳过枚举;白名单一期恒空 |
| R11 | NCCL 不提供 host 序;禁自写原语 | ✅ 全程复用 `_commit_tp_device_runtime_phase`/`_tp_int_min_all_reduce`/`_any_tp_rank_true`/`_all_tp_ranks_succeeded` |
| R12 | begin-fence 三键 assert(传输不与热更新窗重叠) | ✅ F 块首 begin-fence |
| R13 | msgspec / None-init / LOC / 注释风格 | ✅ §5-6 |
| R16 | shm 机构一期全留 | ✅ 不动 shm 分配/收集/unlink |
| R17 | 16B 对齐断言先、coverage 位置不动 | ✅ ring bank_nbytes 16B 硬断言(staging "field-bank" 术语,非 bank 特性) |
| R19 | 异常打标最小 ABI;旧 binding ⇒ 探针 raise | ✅ N5 探针(D8 随 §6-23 消亡,task_tag ABI 保留);**R14/R15/R18(bank 系)随 §6-23 作废——历史留档** |
| R20 | fence 关分配/协议逐位冻结;子参数非默认恰一行 ignored warn(prepare `_validate_kt_args` 承担) | ✅ F4;ignored warn 属 server_args 校验面 |
| R21 | 四禁(e8m0 制作/ds_fp4 嵌套/BufferB gap/零容忍) | ✅ 全程未触 |

## 9. 实机尾项登记(本机不做,逐项待 xysa10)

1. **aarch64 重编译**:task_tag/forward_task ABI 探针随批编译确认(N5 族;D8 随 §6-23 消亡,ABI 保留 C++ 零改动)。
2. **benchbw 归因门**:F 收益恒 TBD,控制面缺口证实才计入;§6-3 init 集合计数 +1～+3 实机复核入库。
3. §7 对拍矩阵逐格 sha256(容许差=0)+ 每层共识计数断言 + E_chunk 退火(1×48h → 16/32/64×24h)。
4. 性能门槛:transfer_stream 带宽 ≥17GB/s(<11 回落)+ 画像 busy >50% 门。
5. 双请求 overlap + **热更 ON** sha256 + 单 rank kill 注入 + NCCL 2-rank 30min 延迟注入 soak。
6. 回滚演练:`--kt-prefill-event-fence 0` 显式置 0 重启即回 legacy 逐字路径;E_chunk 阶梯地板 1 基线复现(§7 口径)。
7. **Phase-2 登记**:WINDOW 行对接(fence×window 二期解禁)、`NO_BATCH_MEMCPY` 三期评估、dma_done 换代归 3 梯队 memop、BufferB 装载 gap(四禁条款下继续暂锁)。bank 关联原尾项 #2(packer 生产权重)/#7(容量 NUMA 实测回填)与 #9 之 bank×window 解禁、BATCH/LEAN 激活随 §6-23 作废——留档。

## 10. 回滚

- CLI 参数启动冻结不支持热关闭;显式置 0 / 协商降级 → 全 rank 统一落 0,legacy 路径逐字保全(从未被裁掉任何一行)。
- `E_chunk=1` ≡ legacy 2×E 语义(F2),是最近回退档(choices 不含 1,由阶梯地板达成)。
- 极端:last-resort = 八件触件 diff 还原(编辑面闭合,零跨文件隐式依赖)。

## 11. 开放登记(本批不解)

- **K≥3 环槽**:一期 K=2(num_slots=2);K≥3 收益未证实,登记待画像。
- **DEBUG digest 成本线性**:默认关;开档预期每块 ~4KB × bank 数线性,排查专用。
- **aarch64-only 编译风险**:D8/F8 探针先行拦截,实机尾项①闭合。
- **shm 驻留升**:FENCE=1 时 SHM 行数 2→2×e_chunk,容量预检已扩进 ring 几何(阶梯 F5 断点)。
- **二期清单**:fence×window 解禁、dma_done 换代、BufferB gap——均不在本批(BATCH/LEAN 激活与 bank×window 解禁随 §6-23 作废)。
