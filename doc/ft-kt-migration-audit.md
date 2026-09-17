# FreeToken→KT 迁移审计定案(2026-09-13)

> 48 个对抗 agent 工作流(wf_e0af1bb3-04d)产物:7 测绘 + 10 特性评估 + 30 对抗验证(数值/并发/运维三视角)+ 1 完整性复核,两轮执行零失败。本文档为吸收全部验证者加固意见后的**定案**;完整逐项明细见 [ft-kt-migration-audit-full-detail.md](ft-kt-migration-audit-full-detail.md)(361KB,含每项 16 字段评估与 30 份验证记录)。
>
> 生产环境:xysa10(aarch64 168 核,2× RTX PRO 6000 Blackwell sm_120 96GB,PCIe 无 NVLink,TP=2,恒带 `--disable-custom-all-reduce`),DeepSeek-V4-Flash 2604B MXFP4(61 层 × 256 专家,19,161,088 B/专家),`--kt-num-gpu-layers=20`,热更新开关**不可关**。prefill 约 500 tok/s(chunk=4096),实测有效带宽 ~11GB/s(Gen4 x16 线速 ~22GB/s)。
> 硬门槛:config-first;代码改动必须 env 一键回落 + 逐层数值对拍 + 热更新互斥审计;竞态 bug 取证极贵(2026-09 NCCL/custom-AR 永冻案)。

---

## 0. 隐藏 P0:现有代码的正确性活洞(优先级高于一切搬运特性)

**热更新若把初始 24 个 GPU 常驻专家逐出回 CPU,decode 将用未初始化内存计算,静默数值污染。**

证据链(双侧代码亲验):

- 加载期 kt-kernel 对 GPU 常驻专家**跳过打包**:operators/arm/mxfp4-moe.hpp:241/:255/:280/:294/:304(非 TP)与 :477(TP),经 operators/common.hpp:256-258 `should_skip_expert`(掩码为**加载时**初始掩码,kt_ep_wrapper.py:3979 传入、:4112 load_weights 生效;此后仅被 update_kt_wrapper_masks 的 .copy_ 翻转,:3707-3729,全仓无 host 重打包路径)。
- `BufferB` 以**未初始化** `std::aligned_alloc` 分配:operators/arm/moe_base.hpp:155-170。
- NEON forward 唯一门控是 mask:arm/moe_base.hpp:235-237 + common.hpp:256-258,**无 loaded/null 检查**。
- 逐出后其 token 走 CPU 计算,结果经 kt_ep_wrapper.py:4627-4646 的 rank0 合并进入输出。
- 加载结束后 host 字节无处可补:utils/amx.py:1334-1342 del 权重 + 释放 safetensors mmap;唯一临时重导出路径是 prefill rank0 reslice→shm(write_weights_to_buffer, mxfp4-moe.hpp:317-329)。

"先 CPU 后提升再逐出"的专家不受影响(host 字节在);**缺字节精确集合 = 每层初始 24 个常驻**。热更新开关不可关,故此洞在生产形态下是活的;是否可达取决于 `select_top_experts_from_batch`(kt_ep_wrapper.py:4714-4719)是否会把加载期 mask 专家踢出 top-24。

**行动**:①只读审计选择器可达性;②可达 → 修复(逐出前对"从未打包"专家补打包 host 字节,或禁止此类专家降级回 CPU);③不可达 → 写断言把不变量钉死。此项由 decode-cap-fetch 评估的附录挖出(critic 裁定"严重度错位,必须拆为独立 P0"),非任何迁移特性引入。

## 1. 定案总表(30 个对抗验证加固后)

| 优先级 | 特性 | 判定 | 预期收益(41 层口径) | 风险 | 评级变动 |
|---|---|---|---|---|---|
| **P0** | (隐藏项)逐出后 NEON 读未初始化 BufferB | 独立审计 | 防静默数值污染 | 生产活洞 | 由 reject 项附录拆分升级 |
| **P0** | explicit-degrade(降级吱声) | adopt(条件化) | 取证:立刻知道生产躺在哪个静默 gate(:2526 死旗标证明 helper 导入失败告警从未接上,6+1 个静默 return) | low→**medium** | 3 条具名前置 |
| **P0** | observability(打点归因) | adapt | 直接提速 0;**一切搬运项的准入证**:把 8.2s/chunk 拆成 host写/共识/H2D/D2D 四段 | low | 提为搬运史诗前置 |
| **P0** | ownership-asserts(槽位断言) | adapt | 后续所有协议特性的对拍仪器 + 竞态取证 | low→**medium** | 默认改 warn-only(=2) |
| **P1** | batch-dma(批量搬运三段合并) | adapt | **500→650–850 tok/s**(1.3–1.7×) | medium→medium-high | 加 WINDOW 互斥 + 画像门 |
| **P1(条件化)** | event-fence(事件栅栏代 barrier) | adapt | 口径改"恢复线速":**先报 650–900**;770–980 仅当归因证实缺口在控制面 | medium→**high** | 收益被证明高估 |
| **P2** | cpu-off-datapath(CPU 移出通路) | adapt | host 写**实为 ~340GiB/chunk**(rank0 写双 rank shm,mxfp4-moe.hpp:627-630)——收益被低估 2 倍 | medium | 排 batch-dma 之后 |
| **P2** | lru-hit-d2d(命中走 HBM) | adapt | 典型 **+5–15%**(勿按 +20–25% 排期),路由偏斜决定 | medium→medium-high | 共识硬门槛为 adopt 前置 |
| **P2** | pretiled-banks(预排布 host 仓) | adapt | +10~25%(pinned 覆盖到位才成立) | medium→**high** | **三视角全 refuted,降级** |
| **P2** | dual-slot-prefetch | defer | 单做 **+0~5%**(带宽已饱和);控制面修完后 +15~25% | high | 维持 defer |
| **P3** | decode-cap-fetch | **reject** | KT 的 24 常驻+NEON 现状更优;cap fetch 打爆 PCIe(unique ~161/层/步) | high | 维持;附录即隐藏 P0 |
| — | 纯 GPU decode slot cache | **登记非候选** | 同工作集账隐含否决 | — | 避免将来重复评估 |

> 收益可加性警告:batch-dma 1.3–1.7×、event-fence、cpu-off-datapath 收割的是**同一个 11→22GB/s 缺口**,收益不可叠加;全组正确的前置里程碑是 observability(归因)+ benchbw(判官),之后再定谁先。

## 2. 关键评级变动的理由

### pretiled-banks:三验证者全 refuted(19 条问题),P1→P2、medium→high

致命伤是**静默字节损坏通道**(取证最贵类):

1. bootstrap D2H 是槽位覆写栅栏协议枚举之外的**第三个异步读者**——_load_slot 覆写前仅 wait consumed/ready event(kt_ep_wrapper.py:2136-2145/:2186-2192;consumed_event 记于 :2445),bank 行可截获撕裂镜像。
2. 三条消费未建行的真实路径:冷专家、热更新换入、纯 decode 启动(缺 [layer,expert] 有效性位图)。
3. 对拍存在消费期盲区。

降级条件(全部完成前不得回到 P1):①槽位栅栏义务枚举扩展(bank_write_event 纳入 reuse_guard,或首 chunk 保守串行化);②[num_kt_layers,256] 有效性位图,miss/无效行逐专家回退旧 shm staging(shm 路径必须在 =0 下保编译保运行做 parity);③G0 RAM 门槛(2×264GiB bank,先实测 xysa10 空内存与 cudaHostRegister 配额);④residency 默认改 budgeted-pinned + `SGLANG_KT_MXFP4_BANK_MAX_GIB` 硬上限(全 pageable 标为实验性——FT host_banks.py 明示 pageable 不可用于 GPU serving 路径);⑤热更新条款二选一并落文档(建议:保留 raw 快照刷新与 :3652-3659 重建,in-place 刷新对应 bank 行,撤回"raw 拷贝归零"收益声明)。

利好保留:FT 的 nvfp4_marlin "global scale 折叠"对 DSV4 MXFP4 是 **no-op**(只有 ue8m0 per-32 块 scale,offload_cache.py:75-77),可移植的是"repack 一次 + prepared 字节常驻 host bank"的模式而非 schema;字节公式逐式相同(51/32·IH≈18.10MB vs streamed 19.16MB,-5.6%);消灭 3 个全设备 sync/层(:1421/:1778/:1817)、每专家 dist.barrier(:1324)、~583GB GPU repack 流量;decode 与 prefill 消费同一 prepared 布局。sm_120 已白名单(v4_marlin_moe.py:40)。

### explicit-degrade:P0,adopt 但条件化(low→medium)

ops 验证者 refuted:(a) 全篇无对拍,违反硬门槛;(b) :2656 KV-budget 路径是**被遗漏的第 9 个静默 gate**;(c) :2592 `in("0","1")` 判定让用户误留 `=0` 也算显式 opt-out,主流场景沉进被过滤日志级;(d) 落点文件在上游 tip 已删除目标区域、fork 落后 200 commits,上游分歧成本举证为假。

**3 条具名前置**:①gate 行为 bit-identical A/B——8 个 return 站点只许 bool→(bool, reason_literal) 纯增强,CI 合成测试对每 gate 构造单体失败场景断言重构前后逐点一致;xysa10 同 prompt/同 seed logits 必须 bit-identical(纯日志改动);②reason 发射下沉 gate 函数内部(dispatch 只透传),收编第 9 个 gate 与 :4518-4531/:1823 的误导性 info, ABI/能力启动探针并入此 gate 家族;③日志级二选一(默认 warning-once,`SGLANG_KT_SUPPRESS_DEGRADE_WARN` 压制阀),发射点零 CUDA op 零新 collective(:4417 在 decode graph 捕获路径上每 forward 执行,capture 期 host 日志安全、replay 不执行 Python)。另需决策:明知短寿命先落 fork 并登记"upstream-sync 时按新架构重移植",还是推迟至 fork 同步后实现(可顺手 PR 回上游)。

### event-fence:收益口径不自洽,条件化 P1(medium→high)

41 层口径下每 chunk 170GiB/卡,8.2s 反推已需 ~20.7GB/s ≈ 贴 Gen4 线速,留给 fence 收敛的余量仅 1.06×;若 rank0 双写(host 写 ~340GiB/chunk)才是 11GB/s 真因(2026-09 后未归因排除),1.55–1.95× 不可达。**落地顺序钉在归因里程碑(observability + benchbw)之后,对外口径先报 500→650–900。**

绑定硬前置(随特性一同落地,非可选):`consumed_event` 记录移到每层热更新读取之后(:2445 → :2485-2487 Gloo 共识之后、:2491 _prefetch_successor 之前);:2473 与 :4514-4515 两处热更新全设备同步在任何"去 sync"提案中均为**不可移除护栏**。公理入档:NCCL barrier 从不提供 host 序(:1323-1329 vs :1243-1244)。范围修正:一期只接管串行路径,layerwise 原样保留(它是 config 级回退面)。kt_ep_wrapper.py 属 fork 自有新文件、零上游分歧,此项无 submodule 成本。

### lru-hit-d2d:共识从"灰度可选"升为"硬门槛"(medium→medium-high)

两个独立 TP 挂死向量:单 rank 惰性建池 OOM 分歧、逐 rank env 分歧 → 命中集分歧 → miss 循环里每专家两次 device_group all_reduce 的 collective 计数分歧(:2253-2275+:2040-2050)= 2026-09 同型挂死。adopt 前置(模板级改动,有现成先例):①初始化成败全 rank 共识(复用 _all_tp_ranks_succeeded :2704 / _disable_* :2721-2731 模板,任一 rank 失败全 rank 永久禁用);②启动期池配置(POOL_SLOTS/ADMISSION/MIN_RATE)config-hash 跨 rank 握手,不一致全 rank 拒启;③熔断停用动作在 round 边界经 Gloo-AND 两秩确认(灰度期 CONSENSUS 强制开)。另有池槽生命周期竞态(热更新窗口写入 vs gather 消费)需 fence。收益口径修正为典型 +5-15%;先做**路由偏斜探针**拿命中率先验再投。

### cpu-off-datapath:收益方向更强(P2 维持)

critic 修正:host 写实为 **~340GiB/chunk**(rank0 经 do_numa_job 遍历全部 GPU TP 分片写两个 rank 的 shm,arm/mxfp4-moe.hpp:627-630/:374/:397-419),被低估 2 倍;14,152 事件(61×232)与 170GiB(41 层)口径自相矛盾已修正。排 batch-dma 之后(同一批量引擎);对拍层集必须改 **{20,21,40,60}**(0/1 层在 num_gpu_layers=20 下走 :3348-3352 return None 是空转)+ 一条 0 GPU 层 CI 腿。

## 3. P0/P1 项的 env 门与对拍方案

全部沿用仓内内联 `os.environ.get("SGLANG_KT_*")` 先例(kt_ep_wrapper.py:4347/:4552/:4616;environ.py 已实证零 KT 条目,不改它,零上游分歧)。

| 特性 | env 开关 | 对拍方案 |
|---|---|---|
| explicit-degrade | warning-once 默认;压制阀 `SGLANG_KT_SUPPRESS_DEGRADE_WARN` | gate 真值表单测(6 分支 mock);xysa10 同 prompt 同 seed logits bit-identical;gate-fail 注入冒烟(rank0 恰一行、rank1 无行);decode graph 捕获零新告警 |
| observability | `SGLANG_KT_PREFILL_STATS=1`(默认关,关=逐字节等价;运行期读取) | token ids 逐位相等(打点无 device op、无新同步);字节账自校验 = 41×232×19,161,088B ≈ 170GiB/chunk;带宽分项误差 <10%;round 覆盖 layer_order 全序列。**churn 计数用现成 CPU 张量 `new_mask & ~old_mask`(:4779-4791),勿新增 .cpu() D2H** |
| ownership-asserts | `SGLANG_KT_SLOT_OWNERSHIP_ASSERT` ∈ {0,1,2},**默认 2(warn-only)**,≥2 周零 warning 后 config 切 1;非法值按 2 处理并告警 | 层 {0,20,40,60} hidden bitwise ± dump 钩子;负注入单测随 PR(交换 host_slot 奇偶/伪造 reuse_guard/改 epoch),断言必须先于任何 collective 触发(layerwise 走 :2053 共识出口,串行先于 :1324) |
| batch-dma | `SGLANG_KT_PREFILL_BATCH_DMA`(即时读)+ `SGLANG_KT_PREFILL_STAGE_LAYER_WINDOW`(restart-required,初始化期固化,双 rank 必须一致)+ `SGLANG_KT_PREFILL_NO_BATCH_MEMCPY`;**初始化期硬 assert: BATCH_DMA=1 & WINDOW=0 互斥**(shm 恒双槽,232 专家挤 2 槽无完成序保证——设计级硬条件) | Phase A 冻 mask(uniform+固定种,A/B 期暂关动态更新并注明重开)逐 bank dump raw 字节 bit-equal;Phase B 生产态内建 canary(`SGLANG_KT_PREFILL_BYTE_CANARY=1`,rank0/1 逐 bank 16B hash 广播比对);Phase C logits smoke;启动四条硬断言(stride 含 :867-873 bf16 scale 特例/16B 对齐/min-entry ≥256KB 否则整窗单条/popcount) |
| event-fence | `SGLANG_KT_PREFILL_EVENT_FENCE` + `SGLANG_KT_PREFILL_NO_DEVICE_SYNC`(独立二分)+ `SGLANG_KT_PREFILL_FENCE_DEBUG`(rank1 每块首 4KB D2H digest 对 rank0 broadcast);`FENCE=0 ⇒ 分配尺寸与协议逐位冻结` | 权重 sha256 逐位相等 + 5 批 4096tok logits parity + **热更新开启态**固定批(:3416 起选择器同输入可复现)+ overlap 双请求交错 + collective 计数断言 ==2×⌈E/E_chunk⌉ + NCCL 单 rank kill 失败注入;E_chunk=1×48h 退火→16/32/64 各 24h |
| lru-hit-d2d | `SGLANG_KT_PREFILL_HIT_D2D`(主开关)+ POOL_SLOTS(生产只经 config 732/1464)+ ADMISSION(evicted\|miss)+ MIN_RATE(默认 0.02 滚动熔断)+ CONSENSUS(灰度期强制)+ DEBUG(hit-set 哈希对拍) | 热更新开启态逐层对拍;均匀路由回放强制触发熔断(两秩同轮停用);容量触顶逐出确定性;单 rank 人为 OOM → 全 rank 一致回落用例 |

> **历史注记(2026-09-17)**:event-fence 行三枚 env(`SGLANG_KT_PREFILL_EVENT_FENCE/NO_DEVICE_SYNC/FENCE_DEBUG`)连同第二批其余五枚,已随 [ft-kt-phase2-plan-dma-event-fence.md](ft-kt-phase2-plan-dma-event-fence.md) §6-21 参数化裁定改为 `--kt-*` CLI 参数(两主开关默认 1,显式置 0 opt-out);本表保留原 env 设计仅作审计史,现行形态以该文档 §1 参数总表为准。**追记(2026-09-17 §6-23)**:参数化后 8 枚中 bank 系 4 枚(`--kt-direct-bank-dma/--kt-dump-slot-bytes/--kt-bank-dma-batch/--kt-bank-dma-lean`)已随 bank 特性整体移除删除;现行 4 枚全 fence 系,此后「两主开关默认开」仅指 `--kt-prefill-event-fence` 一枚。

**batch-dma 灰度阶梯**(验证者重排):
- w0 = 仅批次化 H2D 入列(每专家 4→batch 条目)+ 常驻流/事件池;零新共识、零 DRAM;
- w1 = 单整层窗,每层 submit-all+单 sync+单 fence,串行写→copy,每层 1-2 次共识(共识前必须先 raw_ready_event.synchronize(),镜像 :2248-2256);
- w2 = 双整层窗流水,两段式 pre-write 共识(host 侧 window_free_event synchronize → NCCL MIN all_reduce;**禁任何入列期共识单段充数**;window-generation uint64 进程级计数,移植 FT offload_cache.py:680 断言"写窗 N+1 前 generation(N-1) 已共识");
- 三期才评估 cudaMemcpyBatchAsync(kt-kernel AOT 扩展 + 16B 探测;<256KB 条目混排静默退同步的 FT 实证坑,aarch64/sm_120 必须 bench 验证异步地板;MXFP4 每专家 4 bank ≥0.5MB 天然免疫,小 bank 走 FT :798-804 排除+整层单条模式)。
每档独立 env、独立对拍门槛,不过即 unset。实现顺序:_Mxfp4LayerwisePrefillManager._load_slot(:2237-2292)先行,串行 Phase-2(:1309-1343)后补。**画像门**:单端点 nsys 采集一个 4k chunk,仅当 transfer_stream busy >50% 关键路径才按 P1 继续。

## 4. critic 补充的 12 项漏项

**A. decode 侧,低成本高收益,与 prefill 史诗正交——建议单独排队:**

1. **memop 握手**:cuStreamWriteValue64/cuStreamWaitValue64 取代 cudaLaunchHostFunc。KT 缺此机制:submit cpuinfer.h:90、sync cpuinfer.h:134,41 层 × 2 次/step 落在 decode 尾延迟(每次 host-func 回调 ~30-50µs ⇒ 省约 2.5–4ms/step);不占 SM(避免电调降频)、CUDA-graph 可重放、探测失败自动回落 host-func。证据:FreeToken python/freetoken/moe/cpu_executor.py:33-55/:202-219/:543-605。
2. **CPU 工作池 watchdog**:coordinator 卡死→毒化 done[] + 每 forward 一次 pinned 读 fail-loud,把无限 stall 变带 err[] 的 RuntimeError;daemon 用 weakref 防 GC 钉死。与 e26c065 的 enqueue 修复互补,与 ownership-asserts 互补。证据:cpu_executor.py:47-48/:287-301/:607-660。
3. **CPU 池拓扑定形**:物理核代表选核 + 预留 coordinator/OS 核 + torch.set_num_threads 钳制;aarch64 无 SMT 但后两项适用。证据:cpu_executor.py:92-141/:221-229/:250-257。

**B. 搬运史诗配套:**

4. **路由偏斜探针**:decode_freq 直方图 + oracle_hit_at_slots/working_set_mean/experts_for_90pct——**lru-hit-d2d 的准入证**,成本近零(仅未捕获 CUDA graph 时准确)。证据:offload_cache.py:244-251/:979-1009。
5. **benchbw 归因基准 + GPU UUID 画像**:实测 CPU GEMV 带宽 vs PCIe gather(含重叠对),落 JSON——**11GB/s 缺口的判官**,event-fence 与 cpu-off-datapath 的收益前提全悬其上。证据:moe/benchbw.py:1-40;bench_profile.py:114-156;engine.py:636-663/:1455-1467。
6. **batch 条目 sizing 纪律**:<~256KB 条目与大条目混排使调用线程阻塞整批(-22% e2e 实证);小 bank 恒发整层一条并排除出 hit gather。并入 batch-dma 设计约束;sm_120+aarch64 需 bench 验证。证据:offload_cache.py:17-26/:432-434/:797-804;batch_memcpy.py:42-68。
7. **ABI/能力启动探针**:陈旧 .so 硬失败而非静默算错(kt-kernel .so 与 python 胶水签名漂移正是此类风险,e26c065 刚加 autofree 第 8 参)——并入 explicit-degrade 的 gate 家族。证据:cpu_executor.py:75-89/:178-186;engine.py:1146-1164。

**C. 加载与运维面(中等优先):**

8. **host bank 快载通路**:pin-after-fill 懒 mmap(137GiB 省 ~47s 零填)+ O_DIRECT 多线程分块直读 + PinPipeline 后台 settle(load ≈ max(read, settle))——直接攻击 2604B 加载时长。证据:host_banks.py:6-16/:88-116/:130-144/:286-348/:427-473。
9. **分层 host 驻留 + pin 预算预检**:PINNED/LOCKED/PAGEABLE 逐层计划,非 pin 层硬校验必须属于 cpu_layer_ids;aarch64 无 WDDM 上限故价值中等;对 340GiB 级 host 镜像有 pin 配额缓解价值。证据:host_banks.py:43-53/:219-257;offload_cache.py:212-215/:320-337/:1015-1026;engine.py:1167-1205/:1208-1223。
10. **运行时 cache 几何重建**:validate-before-teardown + CacheRebuildRejected 可恢复拒绝 + 图重捕;MoE slots/KV 页/window 联动,跨 rank 自由内存取 MIN、差 >2GiB 硬失败——对应 xysa10 的 24 槽驻留/staging 与 KV 预算不重启权衡。注意 sglang 子模块分歧成本(FT 走自有 /v1/cache/status)。证据:offload_cache.py:443-534;engine.py:703-722/:766-905。
11. **prefill warmup 阶梯**:dummy 请求跑两套长度,把 Triton/cublas/编译成本从首个真实请求挪走;需核实 sglang warmup 是否覆盖 KT eager prefill 的 Marlin/重排编译与 staging 初始化。证据:engine.py:932-985。
12. **登记非候选**:纯 GPU decode slot cache(ensure_experts 设备侧 LRU remap + copy_missing)——被 decode-cap-fetch 的工作集账(unique ~161/层/步)隐含否决,登记避免将来重复评估。证据:offload_cache.py:843-853/:187-190;offload_kernels.py:19-41。

## 5. 全局口径公理(所有后续方案必须遵守)

1. **KT 层数按 41 算**:61-20(`--kt-num-gpu-layers=20`,kt_ep_wrapper.py:3348-3352 前 20 层 return None)。全部每 chunk 字节/事件预算按 41 重述:170GiB/chunk、4.14GiB/层/rank、9,512 行(41×232)、host 写 ~340GiB/chunk(双 rank shm)。原评估中 61 层口径的线性项一律作废。
2. **唯一合法对拍基准 = CPU pinned canonical staging 逐字节**:`_kt_mxfp4_raw_weights` 是 detach 别名非快照(kt_ep_wrapper.py:3664-3667/:4073-4080),任何"对快照"方案恒真,一律作废。
3. **NCCL barrier 不提供 host 序**(:1323-1329 vs :1243-1244):fence 等价性论证不得引用它;任何新写入/读取对必须用事件建立 happens-before。
4. **fp32→E8M0 最近幂舍入非幂等**(v4_marlin_moe.py:110-112);加之 :2616-2624 暂存 dtype 差异与 :4482-4487 防双 swizzle——跨后端数值矩阵单一条款:同后端 bit-exact + 跨后端独立黄金基线 + 非 2 幂 scale 断言。
5. **热更新互斥审计为共享项**:pretiled/lru-hit/ownership 均依赖,审计结论绑定 :2473 设备级同步 + _tp_phase_succeeded 族共识 + :4787-4794 in-place .copy_ CUDA-graph 先例。
6. **降级规则**:凡 problemSample 属静默损坏/挂死类 → risk ≥ high 且 priority 降档,除非该条被提升为具名 adopt 前置。

## 6. 推荐落地脊柱

```
立做(零代码)   :--chunked-prefill-size 16384(Stage 0)
本周           :[隐藏 P0] 选择器可达性审计(只读,最高优先)
               → explicit-degrade(3 前置落地;upstream 分歧路线二选一登记)
第 1 班(同车) :observability(SGLANG_KT_PREFILL_STATS)
               + ownership-asserts(默认 =2 warn-only 灰度)
               + benchbw 移植 + 路由偏斜探针
   ↓ 归因报告(11GB/s 缺口归因)决定搬运次序 ↓
第 2 班       :batch-dma w0 → w1 → w2(每档独立 env + 对拍 + 画像门)
第 3 班       :cpu-off-datapath(消 340GiB host 写)
               / event-fence(仅归因证实缺口在控制面后启;consumed_event 前置修复)
第 4 班       :lru 探针结果 → lru-hit-d2d(共识硬门槛先落)
重评           :pretiled-banks(栅栏枚举 + 有效性位图 + G0 RAM 门后回 P1)
               dual-slot(若控制面修完仍有重叠空间,+4.57GiB/rank 需显存水位确认)
独立穿插       :decode 侧 memop 握手 + 工作池 watchdog(与热更新正交,压 decode 尾延迟)
```

## 7. 执行溯源

- 工作流 wf_e0af1bb3-04d:7 map(FT 协议/加载/decode/引擎 + KT staging/热更新/decode 配置)→ 10 assess → 30 adversarial verify(每项 numerics/concurr/ops 三视角,refuted 仅当有 file:line 级硬伤)→ 1 critic(12 漏项 + 5 错排裁定 + 6 条一致性备注)。
- 两轮执行:round 1 遭瞬时网络故障(5 assess 死亡),resume 缓存 28 项重放 + 21 项实况补跑,最终 48/48 零失败。
- 验证者结论摘要:refuted=true 共 4 张(pretiled-banks ×3 + explicit-degrade ops),本文档所有评级均已吸收对应 adjustments。
- 对拍/灰度纪律:一切数值特性遵循"同后端 bit-exact、跨后端独立黄金基线";一切 env 默认关/保守值;灰度阶梯每一档不过即 env 回落,禁止运行时热修。
