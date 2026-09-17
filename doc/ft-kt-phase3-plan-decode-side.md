# Phase-3 计划:Decode 并行道(memop 握手 + CPU 池 watchdog + 池拓扑定形) —— ✅ 步骤 A+C 代码落盘,静态验证全过;步骤 B(memop)出列,三岔 2026-09-17 用户裁定冻结(§6-8/§6-13)

> 日期:2026-09-17。第 3 梯队整体第一批,审计 §4-A(critic 漏项 1–3)三项同车派发。
> 施工回写:2026-09-17 步骤 A+C 落盘(3 参 + 测试 W1/W2/W3/W6/W7/W8 六件套全绿,既有四套件 30 测零回归);步骤 B 因 `cuStreamBatchMemOp` 无 ADD 原语出列,三岔经用户拍板**冻结不做**(§6-13);梯队 3 唯一候选性能项 = [bench_reserve_bw_kt.py](../kt-kernel/bench/bench_reserve_bw_kt.py) 三档对照,执行规程见 [ft-kt-xysa10-runbook.md](ft-kt-xysa10-runbook.md)。
> 上游依据:[ft-kt-optimization-tiers.md](ft-kt-optimization-tiers.md) §梯队3、
> [ft-kt-migration-audit.md](ft-kt-migration-audit.md) §4-A.1/A.2/A.3(FT 参照实现行号由审计锚点转引,FT 源码不在本仓)。
> 纪律:开关 int + `Arg(choices=[0,1])`(0=关 1=开,禁 --no-*/store_true/bool;free int 沿 `--kt-cpuinfer` 先例);
> 默认关 = 逐位等价上车;集合通信零新增;不 commit;41 KT 层 / 168 核为口径非硬编码;
> aarch64 重编译归 xysa10(tier-0/1 先例);mask pinned 原位翻转契约零触碰(本期本就不触及)。
> 行号均为勘察时快照(main @ c117d6d 同代),后续以符号锚点重定位。

## 0. 定位

三项同车但**相互独立可逐个回落**,编辑面四块:

- `kt-kernel/cpu_backend/task_queue.h/.cpp`(watchdog 心跳/毒化)
- `kt-kernel/cpu_backend/cpuinfer.h`(T1 memop 节点、watchdog 线程、探针)
- `kt-kernel/cpu_backend/worker_pool.h/.cpp`(T3 reserve 偏移)
- 绑定与 glue:`kt-kernel/ext_bindings.cpp`、`kt-kernel/python/experts_base.py`
- sglang 侧透传:`third_party/sglang/python/sglang/srt/server_args.py`(3 枚新参落盘;memop 参出列未落,§6-8)+ `kt_ep_wrapper.py`(KtConfig 通道,沿 `:5612` cpuinfer_threads 先例)

1. **T1 memop 握手(收益主项)**:41 KT 层 ×2(submit+sync)枚 `cudaLaunchHostFunc` 回调落在 decode 尾(step 口径 ~2.5–4ms,审计 §4-A.1),替换为 `cuStreamBatchMemOp`(WRITE/WAIT_VALUE_64)配对,不占 SM、CUDA-graph 可重放、探测失败结构性回落 host-func。承载面 = CPU-MoE 前向每一步(prefill/decode 同路径),名义 decode 取收益落点。
2. **T2 CPU 池 watchdog(防守先手)**:`TaskQueue::sync` 无限 cv 阻塞(task_queue.cpp:55-67);worker 卡死即全线静默冻结(2026-09 NCCL 永冻案同型取证痛点)。无进度超时 ⇒ 毒化 `first_exception` + notify,sync rethrow 免费抬出带 (kind,expert) 摘要的 RuntimeError;Python 每 forward 一次原子 probe 防毒化后沉默。
3. **T3 池拓扑定形**:每 NUMA 预留 reserve 枚低号核(OS IRQ / coordinator / CUDA 回调线程)+ `torch.set_num_threads` 只钳不升。物理核代表选核**既有化免做**(`worker_pool.cpp:74-89` 已按 hwloc CORE + singlify STRICT 绑核;x86 SMT 行内天然取物理核代表,aarch64 1PU/核 singlify 无操作)——FT 三要素中"物理核代表"对 aarch64 无对象,落其余两项。

## 1. 参数总表(3 枚落盘 + memop 出列 1 枚留档,`NS("exec.moe")`,`[ktransformers parameter]` 前缀,CPUInfer 家族根)

| 参数 | 类型/值域 | 默认(提案,待裁定) | 语义 |
|---|---|---|---|
| ~~`--kt-cpuinfer-memop`~~ | 开关,choices=[0,1] | 0 | **⏸ 施工期整支出列,本参未落盘**(§6-8:cuStreamBatchMemOp 无 ADD 原语 → 三岔待裁);原语义留档:memop 握手主开关;probe 失败/deferred 活跃层自动回落 host-func |
| `--kt-cpuinfer-watchdog` | 开关,choices=[0,1] | 0(✅ 已落盘) | 池 watchdog;挂死 ⇒ fail-loud RuntimeError |
| `--kt-cpuinfer-watchdog-timeout-ms` | free int ≥1000 | 30000 | 无进度判定预算(decode 层 ms 级;30s 远离误报,校准归 §9-4) |
| `--kt-cpuinfer-reserve-cores` | int,choices=[0,1,2] | 0 | 每 NUMA 子池自低号核起预留数(0=现状) |

默认全 0 上车定调:T1 数值零触碰但热路径机制大改;T2 纯观测语义、零 CUDA 零集合、热路径单原子读;T3 静态拓扑。**对拍/画像门槛全过后是否翻默认 1,独立单裁(梯度沿 event-fence §6-21 先例,不搭车)**,本文档不预授权任何默认值翻转。

## 2. 架构(坐标不动)

- Python 调用面 `submit_forward`/`sync_forward` 方法名签名零变;transport 分流在 `CPUInfer::*_with_cuda_stream` 体内,probe 一次 memoize 冻结(冻结语义沿梯队 1 窗口先例)。
- TaskQueue 不变量零动:pending 计数、`first_exception`→sync rethrow 通道(task_queue.cpp:63-66)、cv 唤醒协议(:49-52/:93-99);watchdog 毒化走**同一** first_exception 通道(`[kt watchdog]` 前缀),现成的 `describe_task_tag`(cpuinfer.h:44-50)出 (kind,expert) 摘要。
- `WorkerPoolConfig` 增 `reserve_cores_per_numa`{default 0},旧构造重载行为逐位不变。
- 账面(memop on,decode 尾):host-func 回调 41×2 → 0;流上 memop 节点 ×2/层;集合通信零新增(watchdog/probe 纯主机原子读)。
- NPU 旁路(`_should_bypass_stream_callback` experts_base.py:74-87、`KT_FORCE_SYNC_SUBMIT`)、MUSA/ROCM/MACA 列全部零触碰(无 memop 上游参照,登记非目标)。

## 3. 逐笔施工清单

### 步骤 A(T2 先行:防守先手,memop 注死演练依赖其取证)

1. `task_queue.h`:Node 增 `int64_t task_tag{0}`;队列增 `std::atomic<int64_t> task_start_ns{0}`(0=空闲)与 `pending_task_tag{0}`。`worker()` 在 `next->task()` 前写 start+tag(task_queue.cpp:76-77 处),完成于 :90 清位。`enqueue_tagged` 随 node 写入 tag(cpuinfer.h:93-106 通道)。
2. `cpuinfer.h`:CPUInfer 构造增可选 `watchdog_timeout_ms`(=0 不启动 ⇒ 逐位等价)。watchdog 线程:`task_start_ns` 存活且 `now-start > budget` ⇒ 一次性锁存毒化——~~`first_exception = std::runtime_error(...)`+ `pending` 压至 ≤allow~~(**施工期改判,§6-9**:独立 latched poison 通道,不触碰 pending、不走 first_exception;下文 §6 落字)+ `cv.notify_all()`;毒化**不可清空**(:63 清位通道让读 poison 位)。
3. 销毁序:watchdog join 先于 `delete task_queue_`(cpuinfer.h:74-78 顺序注记加一句)。
4. `ext_bindings.cpp`:CPUInfer 增 `watchdog_tripped()` 导出(毒位原子读,~ns 级)。
5. `experts_base.py sync_forward`:bypass 与流分支在 sync 调用**前**一行 probe;tripped ⇒ raise RuntimeError(C++ 文本原样上抬),防"毒化后继续沉默消费陈输出"。
6. FT weakref daemon 要点的 KT 归属:watchdog 纯 C++ 不持 GIL、不持 Python 对象 ⇒ "worker 死后 daemon 钉活"问题无对象,登记免做(裁定 §6-7)。

### 步骤 B(T1 memop 握手)—— ⏸ 施工期整支出列(§6-8:cuStreamBatchMemOp 无 ADD 原语,三岔待裁;下文为原设计留档,未落盘一行)

1. 驱动 API 面:`vendors/cuda.h` 已链 `<cuda.h>`;`#if CUDART_VERSION >= 12000` 编译期闸 + 运行期 `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1)` 探测一次 memoize。任闸不过 ⇒ memop 支路结构性不存在,函数体落现 `cudaLaunchHostFunc` 实现(cpuinfer.h:115-124/:149-168 一行不删)。
2. **submit 侧**:Python 直接 host 交队列(复用 NPU 同步上交模式 experts_base.py:900/:919,但**不取** `_wait_device` —— 门闩即序);task 包 `(doorbell_ptr, expected_gen)`,worker 分发前自旋/pause 至 `doorbell ≥ expected_gen`;流上 D2H 之后入 WRITE_VALUE_64 节点写 gen。
3. doorbell:cudaHostAlloc 64B 对齐,按 (queue × `KExpertsCPUBuffer.buffer_depth`=2 槽) 索引;生命周期 = CPUInfer,**不随 temp buffer LRU 逐出**(experts_base.py:359-376 条款加一句);释放随 ~CPUInfer。
4. **sync 侧**:TaskQueue `completed_total` 单调计数;`sync_with_cuda_stream` 快照 target = completed_total+(pending−allow);任务完结最后一棒写 pinned done;流上 WAIT_VALUE_64 + GEQ。
5. **CUDA-graph 可重放**:节点烘焙**增量**(ADD_VALUE_64)而非绝对值,WAIT/GT 配单调计数 ⇒ replay 跨步正确;烘焙对象不含运行态值(与 SyncArgs autofree 判据 cpuinfer.h:155-160 对照)。capture 期间 doorbell/done 注册语义沿 KExpertsCPUBuffer 捕获注册直管(:297-304)。
6. **deferred 分流**:`_layer_has_pending_deferred`/deferred_ids 活跃层 ⇒ 该层 submit/sync 双双现 host-func(零状态交集;deferred 默认 0,常态不触发)。
7. ABI 冻结:8 参 `forward_task` 不动(ABI 探针 experts_base.py:185-210 保留);新增面全部走 CPUInfer 新方法。

### 步骤 C(T3 池拓扑)

1. `WorkerPoolConfig.reserve_cores_per_numa`(default 0 ⇒ 逐位等价);`ext_bindings` def_readwrite 增字段;构造校验 `reserve + subpool_thread_count ≤ NUMA 核数`,越界**创建期 hard-fail**(config-first)。
2. `worker_pool.cpp` 每 NUMA 构造(:74-89):CORE 索引 `= reserve + i + threads_id_start`。
3. `experts_base._get_cpu_infer`(:416-436):透传 reserve(None→0);singleton 构造成功后 `torch.set_num_threads(min(torch.get_num_threads(), per_subpool_thread_count))` 一次(只钳不升)。
4. sglang 侧:`server_args` 3 参落位(memop 出列,§6-8;`NS("exec.moe")`,`[ktransformers parameter]` 前缀),KtConfig 通道透传沿 `:5612` 先例;默认 0 形态下现有参数面零变化。

## 4. 测试映射(W1–W9,Python CPU-stub 无 GPU;新增 `third_party/sglang/test/manual/kt_decode_side_test.py`)

unit-test-admission 三族归类,逐用例"哪个未来 diff 使其变红"已可答:

| # | 族 | 断言对象 |
|---|---|---|
| W1 | 派生性质 | `_get_cpu_infer` reserve 透传:None→0,显式 1/2 写入 config;非法值拒绝 |
| W2 | 派生性质 | torch 钳只减不增;subpool=1 时钳值=min(current, count) |
| W3 | 关键路径簿记 | `sync_forward` probe:tripped ⇒ RuntimeError,文本含 `[kt watchdog]` + (kind,expert) 摘要;未 tripped 零告警零行为差(mock cpu_infer) |
| W4 | 派生性质 | ~~memop 探针分流~~ ——⏸ 随步骤 B 出列(§6-8) |
| W5 | 关键路径簿记 | ~~doorbell 槽法~~ ——⏸ 随步骤 B 出列(§6-8) |
| W6 | 派生性质 | server_args 3 参(memop 出列后):ast.parse + 独立 arg_utils 副本解析(Windows 先例);choices/default/NS/门控校验落字 |
| W7 | 关键路径簿记 | KtConfig 增字段 → 构造点 `get_exec().moe` 直读 → common_wrapper_kwargs 逐层一致(源文本钉死;default 0 形态现参数面零 diff) |
| W8 | 派生性质 | 源文本守卫:task_queue 毒化/心跳字段、cpuinfer watchdog 线程/探针、worker_pool reserve 偏移、ext_bindings 字段/探针/双参构造、experts_base probe/透传;且 `cudaLaunchHostFunc` 原调用点存续(防悬空重命名) |
| W9 | 关键路径簿记 | ~~41 层账面~~ ——⏸ 随步骤 B 出列(§6-8) |

> 施工落盘:`third_party/sglang/test/manual/kt_decode_side_test.py` W1/W2/W3/W6/W7/W8 六件套 2026-09-17 全绿;W4/W5/W9 随步骤 B 出列,复活时随三岔裁定重写(memop 原 W4/W5/W9 断言对象作废)。

C++ 级验收(watchdog 注死、memop 节点序、graph replay、reserve 绑核取证)全部归 xysa10 —— Windows 无 aarch64 工具链,tier-0/1 先例。回归盘:既有四套件 30 测全绿(F1–F8、N1–N12、ownership N1–N5、reason N4–N6 零回归)。

## 5. 静态验证记录(✅ 2026-09-17 全过)

- [x] `python -m py_compile` 全部编辑面(experts_base/experts/utils 三件 + kt_ep_wrapper + 新测试件)✔;W1/W2/W3/W6/W7/W8 六件套全绿(one-shot 全绿;W4/W5/W9 随步骤 B 出列)。
- [x] 魔数扫描:`git diff HEAD` + 新测试件源内 `\b(41|61|24|20|168|9040)\b` 生产/测试代码零命中(仅命中既有文档 §6-24 章节号文本与 diff hunk 头,非本期新增)。
- [x] server_args:`ast.parse` ✔ + arg_utils 独立副本解析 ✔(`Arg` 无 `type` kwarg 复核确认;`A[int, Arg(help=,choices=[0,1]), NS("exec.moe")]` / plain-string / `choices=[0,1,2]` 三种落盘形态逐项构造通过)。
- [x] `environ.py` 零 diff ✔;env 零新增 ✔(三枚全为 `--kt-cpuinfer-*` CLI,int + `Arg(choices=...)` 开关/受限值域形态沿先例)。
- [x] 既有四套件回归全绿(F1–F8 / N1–N12 / ownership N1–N5 / degrade N4–N6,共 30 测零回归)。

## 6. 裁定记录(plan 级预登记 1–7;施工期改判/新增 8–12)

1. 四参默认全 0 上车;memop 翻默认 1 独立单裁(§1 定调)。**施工期变更**:memop 整支出列,落盘 3 参默认全 0 维持(§6-8)。
2. deferred 活跃层 per-layer 回落 host-func,不追求 memop 覆盖 deferred。(随 §6-8 出列暂停)
3. watchdog 毒化锁存不可清空(挂死类诊断宁可响到底)。
4. `torch.set_num_threads` 只钳不升。(另见 §6-10 的门控改判)
5. aarch64 无 SMT ⇒ 物理核代表选核既有化免做;x86 列拓扑不重排。
6. memop 回落 = 结构性(编译闸 + 探针 memoize 双闸),热路径零分支;dispatch 一次冻结。(随 §6-8 出列暂停)
7. FT weakref daemon 需求无对象(C++ watchdog 不持 Python 生命周期),免做。
8. **步骤 B(memop)整支出列,三岔待用户拍板**。施工期复核发现 `cuStreamBatchMemOp` **无 ADD 原语**(原设计 §3-B.5 的 ADD_VALUE_64 增量烘焙前提不成立,WRITE_64/WAIT_VALUE_64 仅绝对值);绝对值烘焙与 graph replay 跨步单调冲突,而 decode 图面 shot-to-shot 恒 capture ⇒ eager-only memop 在 graph 下无对象。三岔:(i) **replay-daemon**(图外门闩守护线程,全量重设计,成本最高);(ii) **M-lite eager-only**(仅非图路径生效,decode 常态在图内 ⇒ 收益≈0);(iii) **整支废弃**(维持 `cudaLaunchHostFunc`,由已落盘的 watchdog+topology 兜底防守)。`--kt-cpuinfer-memop`、探针/doorbell/memop 节点全部未落盘,W4/W5/W9 与 §3-B 全文转留档。
9. **watchdog 毒化改判(替代 §3-A.2 两要素)**:latched 独立 poison 通道(`poisoned_flag`/`poison_exception`/`poison_what`,mtx 下 set-once CAS)——毒化**不动 `pending`**(卡死任务若晚完成,原案压计数会下溢)、**不走 `first_exception`**(其属一次性 drain,首个 sync 取走后毒化即沉默,违背裁定 3);`sync()` 等待谓词并入 `poisoned_flag`,awaiter 先 rethrow poison 再 drain first_exception;热路径代价 = 起任务/完工各一原子写 + sync 谓词一原子读。
10. **torch 钳 gated on reserve(改判 §3-C.3 施工程序)**:原案"singleton 构造成功后恒钳"会在默认 0 形态改变 torch intra-op 线程数,破坏裁定 1 的逐位等价;施工期裁为仅 `reserve_cores` 为真值时钳(只钳不升维持),W2 钉死。
11. **SFT 支路硬拒绝**:`KTMoEWrapper.__new__` mode='sft' 对非默认 reserve/watchdog 抛 ValueError(swiglu_limit 先例逐字形态)——reserve/watchdog 属池级 singleton 参数,SFT 与推理共池时不得静默丢弃。
12. **旧 .so 容忍链**:pybind 编不出 feature 探针时 Python 全走动态面——`_cpuinfer_watchdog_raise` 用 `getattr` 元数探测、CPUInfer 1/2 参构造按 timeout 真值分发、`reserve_cores_per_numa` 仅真值时写字段。旧扩展上默认 0 形态逐位等价,W1/W3 钉死。
13. **三岔冻结(2026-09-17 用户拍板,替代 §6-8 的"待裁定"态)**:memop 三岔(replay-daemon / eager-only / 整支废弃)**冻结不做**,§3-B/W4/W5/W9 维持留档;梯队 3 的后续性能议程唯一通道 = reserve 三档实机对照([bench_reserve_bw_kt.py](../kt-kernel/bench/bench_reserve_bw_kt.py),规程 [ft-kt-xysa10-runbook.md](ft-kt-xysa10-runbook.md))。复活条件挂 benchbw verdict:仅当归属落到 "(a)+写腿富余"(PCIe/RC 共享,控制面特性重新估价)才重开议程。

## 7. 对拍门槛(xysa10 实机验收)

- 离线确定性(沿梯队 2 §7 纪律):固定 8 prompt × (512 prefill + 64 decode)、`--max-running-requests=1`,双层间冲刷 radix cache;memop on/off 采样 KT 层 logits sha256 逐位相等(承载面替换计算零触碰,容许差 0)。
- decode TPOT 门槛:41 层 callback 尾延迟收敛 ≥ ~2.5ms/step(审计口径 2.5–4ms);nsys timeline 上 host-func 回调 ≈ 0、memop 节点入流可见。
- watchdog 注死演练:调试注死池 ⇒ RuntimeError 带 (kind,expert) 摘要、进程不冻结不沉默;on/off 各一。
- CUDA-graph replay:decode graph 全部 capture bs 档 replay ×K 步,doorbell/done 单调零回卷断言。
- reserve 带宽对比:benchbw(tier-1 已落盘判官)reserve 0/1/2 三档 GEMV 带宽对照落 JSON。

## 8. 红线表(施工自查;沿梯队 0–2 R1–R13 全量继承,补梯队 3 特有四条)

| # | 红线 | 状态锚点 |
|---|---|---|
| R-m1 | `cudaLaunchHostFunc` 原路径结构性保全(删零行),回落 = 双闸结构性非热分支 | §3-B.1 |
| R-m2 | wait 值无 baked literal:ADD/GT 单调对,replay 跨步安全 | §3-B.5 |
| R-m3 | doorbell/done pinned 生命周期 ≥ 流与图,不入 LRU 逐出面 | §3-B.3 |
| R-m4 | watchdog 热路径 ≤ 每 forward 单原子读,零 CUDA op 零集合 | §3-A.5 |

## 9. 实机尾项登记(本机不做,逐项待 xysa10)

1. aarch64 GCC 重编译(tier-0 先例:CRTP 裸调用补 `this->` 教训,两阶段名字查找自查)。
2. memop on/off sha256 对拍 + TPOT 门槛(§7)。
3. CUDA-graph 全 bs 档 replay soak;doorbell/done 单调断言入库。
4. watchdog 注死演练 + timeout 默认 30000 校准(实测长 prefill chunk 耗时分布后定调)。
5. reserve-cores 0/1/2 三档对照(GEMV 带宽、振荡)——载体 [bench_reserve_bw_kt.py](../kt-kernel/bench/bench_reserve_bw_kt.py)(2026-09-17 落盘:单进程一档、master sweep 汇总、噪声地板判读内置),执行规程与梯队决策翻译见 [ft-kt-xysa10-runbook.md](ft-kt-xysa10-runbook.md)。
6. sm_120 驱动 CAN_USE_STREAM_MEM_OPS_V1 探测实际值落档。

## 10. 回滚

- 参数逐项回落:watchdog/reserve 两项独立,任一置 0 即回现实现(host-func 路径一字未删;watchdog 不启动、无监控线程;reserve=0 拓扑逐位等价且 Python 面零字段写)。
- 极端:last-resort = 四块编辑面 diff 还原;CPUInfer 构造 ABI 旧形态保留(构元数分发 + getattr 探针,旧 .so 上默认 0 逐位等价,§6-12)。
