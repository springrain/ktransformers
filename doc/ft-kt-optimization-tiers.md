# FT→KT 优化梯队总表(0–5 梯队,含状态)

> 日期:2026-09-14。本文档把对话中定案的**六个梯队**落成文字,统一收口状态。
> 上游依据:[ft-kt-migration-audit.md](ft-kt-migration-audit.md) §1 定案总表 / §6 推荐落地脊柱、
> [ft-kt-phase1-audit-hidden-p0.md](ft-kt-phase1-audit-hidden-p0.md) §7(0 梯队定案)/ §12(Phase-1 派发)、
> 五份 Phase-1 计划文档(benchbw / explicit-degrade / observability / ownership-asserts / routing-probe)。
> 收益栏区分"文档原始宣称"与"对抗复核后的修正口径"——多项宣称已被下调,以复核口径为准。

## 状态图例

| 标记 | 含义 |
|---|---|
| ✅ 完成 | 代码落盘并通过静态验证(实机验证尾项单列) |
| 📋 计划定稿 | 计划文档评审完毕,代码未落盘 |
| ⏸ 暂停 | 曾施工,被还原/中断,待重启指令 |
| 🔒 暂锁 | 前置数据/条件不足,满足条件前不动工 |
| ❌ 废弃 | 对抗验证 refuted,移出路线 |

## 梯队一览

| 梯队 | 主题 | 状态 | 收益量级(复核口径) |
|---|---|---|---|
| **0** | 隐藏 P0 正确性(F3 装载期全量打包 + F4-soft 金丝雀) | **✅ 完成** | 正确性收益:消除静默错值/NaN 级联(不可用速度衡量) |
| **1** | 归因与取证基础设施(benchbw / explicit-degrade / observability / ownership-asserts) | ✅ 代码落盘,静态验证全过(实机尾项见明细) | 无直接增益;是梯队 2 全部收益数字的判官与准入证 |
| **2** | Prefill 传输史诗(batch-dma / event-fence;~~cpu-off-datapath~~) | ✅ 代码全数落盘(第一批 batch-dma w0→w1→w2 + consumed_event 前置修复;第二批 event-fence F1–F8),静态验证全过;原第二批 cpu-off-datapath(bank)D1–D9 于 2026-09-17 整体移除(pinned 镜像 ≈+1 份专家权重 RAM 不可接受,[计划文档 §6-23](ft-kt-phase2-plan-bank-dma-event-fence.md));实机尾项见各计划文档 §9 | 合并上限 ≈ 传输段恢复线速(11→~22GB/s);**两者收割同一缺口,收益不可叠加** |
| **3** | Decode 并行道(memop 握手 / CPU 池 watchdog / 池拓扑定形) | 📋 未始(可与梯队 1 同期推进,正交不冲突) | memop ≈ 省 2.5–4ms/step decode 尾延迟;其余为可观测性/稳定性收益 |
| **4** | 装载与运维(host bank 快载 / warmup 阶梯 / 分层驻留 / cache 几何重建) | 📋 未始 | 直击装载时长(整趟零填约 47s 量级可省)+ 运维免重启 |
| **5** | 条件触发(lru-hit-d2d / ~~pretiled-banks~~ / dual-slot-prefetch) | 🔒 暂锁 | lru 典型 +5–15%(探针先验决定);~~pretiled +10~25%~~ ❌ 2026-09-17 随 bank 裁定判死(移入「明确不做」);dual-slot ≈0 无限期推迟 |

---

## 第 0 梯队:隐藏 P0 正确性 —— ✅ 完成

**定性:两个隐藏 P0 正确性缺陷的加固,非性能优化。** 生产风险:热更新若把初始 GPU 常驻专家逐出回 CPU,decode 将用**未初始化内存**计算,静默数值污染。

| 子项 | 内容 | 收益 / 代价 |
|---|---|---|
| F3 装载期全量打包(修根因) | 装载门/运行时门分离:新增 `should_skip_expert_packing`(装载用)与运行时 `should_skip_expert` 解耦,env `SGLANG_KT_PACK_GPU_RESIDENT_HOST=1` 时装载期打包全部专家 | 根除"逐出后读未初始化 BufferB";代价:装载打包 +~10.3%(秒级)、RAM +0B |
| F4-soft 金丝雀(观测报警) | 运行时五门放行未打包专家时,限流 `KT_WARN hidden-P0` 到 stderr(含 layer/tp_part/expert 坐标) | 静默错值变可观测;默认 OFF 时逐位等价 |

**覆盖范围**:arm 列(NEON)+ x86 avx2 列(mxfp4 / rawint4_avxvnni);amx 列装载面无缺陷(亲读裁定),未改。
**默认关 = 逐位等价**;env 唯一读取点在 Python import-time(`amx.py`),C++ 内零 env;集合通信数 0 变更。

**完成注记**:
- 代码 + 冒烟脚本 `kt-kernel/scripts/smoke_pack_gpu_resident_host.py`(四 scenario)落盘,静态红线核验全过(运行时原语计数、魔数、env 治理、mask pinned 原位翻转契约均未触碰)。
- aarch64 GCC 编译报错已修:CRTP 派生类内 4 处裸调 `mark_packed_experts()` 补 `this->` 限定(两阶段名字查找;纯编译期语法,机器码逐位相同)。
- **实机验证尾项(待 xysa10)**:aarch64 重编译 + smoke 四断言(baseline_off/on、evict_off/on)+ flag on 装载耗时实测(理论 +~10.3%)。

## 第 1 梯队:归因与取证基础设施 —— ✅ 代码落盘(派发序 1/4–4/4 全部完成)

> 地位:**一切传输优化(梯队 2)的准入证**。没有 benchbw 判官和 observability 分段账本,梯队 2 全部收益数字悬空。
> 初版梯队 0 曾含 explicit-degrade / ownership-asserts,Phase-1 派发时并入本梯队;routing-probe 经对抗验证 refuted 后移出(数据需求由 observability 的 GPU 常驻 churn 直方搭车接替)。

| 序 | 子项 | 计划文档 | 状态 | 说明 |
|---|---|---|---|---|
| 1/4 | benchbw 带宽归属探针 | [ft-kt-phase1-plan-benchbw.md](ft-kt-phase1-plan-benchbw.md) | ✅ 代码落盘([bench_bw_kt.py](../kt-kernel/bench/bench_bw_kt.py)) | 实测 CPU GEMV vs PCIe gather(含重叠对),落 JSON——回答"22 vs 11GB/s 缺口归因"。实机尾项:xysa10 机时跑 G2/G3 采集 |
| 2/4 | explicit-degrade(reason 链 + 告警 + ABI 探针) | [ft-kt-phase1-plan-explicit-degrade.md](ft-kt-phase1-plan-explicit-degrade.md) | ✅ 代码落盘(单测 N4/N5/N6 绿) | 把多处静默降级 gate 变成可观测/可拦截;正确性/取证收益。实机尾项:分支边界分析 + xysa10 对拍 |
| 3/4 | observability(`SGLANG_KT_PREFILL_STATS`) | [ft-kt-phase1-plan-observability.md](ft-kt-phase1-plan-observability.md) | ✅ 代码落盘 | 每 chunk 拆成 host写/共识/H2D/D2D/热更新/residual 分项;字节账自校验;churn 直方搭车。实机尾项:离线确定性对拍 + 灰度文档 |
| 4/4 | ownership-asserts(槽位所有权哨兵) | [ft-kt-phase1-plan-ownership-asserts.md](ft-kt-phase1-plan-ownership-asserts.md) | ✅ 代码落盘(单测 N1-N5 绿) | 默认 2=warn-only 灰度;后续协议特性的对拍仪器 + 竞态取证。实机尾项:离线确定性对拍 + 档 0 退出条件(threshold vs capture 双数入 PR) |
| — | routing-probe(路由偏斜探针) | [ft-kt-phase1-plan-routing-probe.md](ft-kt-phase1-plan-routing-probe.md) | ❌ blocked 移出 | 原实施路径被裁定不可救;接替:observability churn 直方 |

> 还原说明:2026-09-14 用户还原工作区"只保留第 0 梯队",benchbw / explicit-degrade / observability 的已落盘代码随之丢弃;五份计划文档仍在。
> 重启记录:2026-09-15 已按五份计划文档从头施工完毕,1/4–4/4 全部 ✅(实机验证尾项见各行"说明"列)。

## 第 2 梯队:Prefill 传输史诗 —— ✅ 全数落盘(第一批 2026-09-16,第二批 2026-09-16)

| 项目 | 状态 | 宣称收益 → 复核口径 |
|---|---|---|
| batch-dma(分段 submit/sync + 整层 batch copy + <256KB 纪律,w0→w1→w2 灰度) | ✅ 代码落盘,静态验证全过([ft-kt-phase2-plan-batch-dma.md](ft-kt-phase2-plan-batch-dma.md);单测 N1–N11 绿,consumed_event 前置修复一并落盘) | 宣称 500→650–850 tok/s(1.3–1.7×) → **gain 标 TBD**:方向被证实,具体数字须先跑画像门(transfer_stream busy >50% 才按 P1 继续)——实机尾项见计划文档 §9 |
| cpu-off-datapath(消 host 写,须容量/NUMA 预检) | ❌ **2026-09-17 整体移除**(用户裁定:pinned bank = 全 TP 组合计 +1 份专家权重 RAM、单机部署专家桶 ≈×2,不可接受;切除范围 = sglang 侧本体/manifest/参数/测试 + 主仓 bench `pack-bank` + tier-5 pretiled-banks 一并判死,详见 [计划文档 §6-23](ft-kt-phase2-plan-bank-dma-event-fence.md);复活路径 = `git revert` 切除提交)。原 ✅ 落盘叙述(单测 D1–D9 绿、pretiled 前置降级自带、env→CLI 参数化)整体转为历史留档 | —(随移除作废,历史留档:host 写实为 ~340GiB/chunk(rank0 双写),收益原被低估 2 倍) |
| event-fence(事件栅栏代 barrier) | ✅ 代码落盘,静态验证全过(同上计划文档;单测 F1–F8 绿;硬前置 consumed_event 前置修复已随第一批落盘;2026-09-17 参数化:env 全部改 `--kt-*` CLI、两主开关默认开,见计划文档 §6-21) | 宣称 1.55–1.95× 不可达 → 口径改"恢复线速";**gain 恒标 TBD**:仅当 benchbw 归因门证实缺口在控制面才计入;E_chunk 阶梯 64→32→16→1(chunk=1 ≡ legacy) |

⚠️ **不可加警告**:两者收割的是**同一个 11→22GB/s 缺口**,严禁按宣称值叠加;合并上限 ≈ 传输段恢复线速。(原 cpu-off-datapath 与 pretiled-banks 已随 2026-09-17 bank 裁定移除/判死,见计划文档 §6-23。)

## 第 3 梯队:Decode 并行道 —— 📋 未始(与梯队 1 同期推进,正交不冲突)

| 项目 | 大概收益 |
|---|---|
| memop 握手(cuStreamWrite/WaitValue64 取代 cudaLaunchHostFunc,探测失败自动回落) | 每 step 可省约 2.5–4ms decode 尾延迟;不占 SM、CUDA-graph 可重放 |
| CPU 池活性 watchdog | 无性能收益;无限 stall → 带 err[] 的 fail-loud,竞态/挂死取证成本骤降 |
| CPU 池拓扑一次定形(物理核代表 + 预留 coordinator/OS 核 + 线程数钳制) | 带宽受限 GEMV 免超额订阅;aarch64 无 SMT,后两项适用,收益中等未量化 |

## 第 4 梯队:装载与运维 —— 📋 未始

| 项目 | 大概收益 |
|---|---|
| host bank 快载(pin-after-fill 懒 mmap + O_DIRECT 分块直读 + PinPipeline 后台 settle) | 省整趟零填(FT 口径 137GiB ≈ 47s 量级),load ≈ max(read, settle);直击长装载 |
| prefill warmup 阶梯 | 消除首个长 prompt 抖动(Triton/cublas/编译成本挪离首个真实请求) |
| 分层驻留 + pin 预算预检 | 数百 GiB 级 host 镜像的 pin 配额缓解,价值中等 |
| cache 几何原地重建 | 驻留/KV 页/window 免重启权衡,运维收益 |

## 第 5 梯队:条件触发 —— 🔒 暂锁(前置数据不足)

| 项目 | 大概收益 | 解锁条件 |
|---|---|---|
| lru-hit-d2d(命中走 HBM) | 典型 +5–15%(勿按 +20–25% 排期),路由偏斜决定 | ①路由偏斜先验(observability churn 直方接替);②共识硬门槛:初始化成败全 rank 共识 + 池配置 config-hash 握手 + 熔断动作 Gloo-AND 两秩确认 |
| dual-slot-prefetch | 单做 +0~5%(带宽已饱和) | 无限期推迟;控制面修完后若仍有重叠空间再重评(需显存水位确认) |

## 明确不做

- **decode-cap-fetch**——reject:KT 现有 GPU 常驻 + NEON 路径更优;cap fetch 会打爆 PCIe。
- **纯 GPU decode slot cache**——登记非候选:被 decode-cap-fetch 的工作集账隐含否决,登记避免将来重复评估。
- **pretiled-banks(预排布 host 仓)**——❌ 2026-09-17 随 bank 裁定判死:前置四件(manifest 位图、容量/NUMA 预检、热更条款、银行本体)全部随 bank 移除归零,且同一 RAM 理由适用度更强(pinned 覆盖本身即专家镜像)。**架构约束落字:pinned 专家镜像预算为零——任何「常驻 +1 份专家权重」的设计不再评。**(原梯队 5 行的降级条件叙述整体转为历史留档,见梯队 2 cpu-off-datapath 行与计划文档 §6-22/§6-23。)

## 全线纪律

1. 所有 chunk 预算按 **41 个 KT 层**口径重述(审计公理 §5.1);专家数/层数为参数化参考值,代码绝不写死生产拓扑。
2. 唯一合法对拍基准 = CPU pinned canonical staging 逐字节;NCCL barrier 不提供 host 序。
3. 新增 env 一律内联 `os.environ.get` + **默认关 = 逐位等价**;不碰 sglang `environ.py`。
4. 不主动 commit(merge-no-commit);mask pinned 张量原位翻转契约不可破坏。
