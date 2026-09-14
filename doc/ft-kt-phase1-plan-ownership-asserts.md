# Phase-1 计划:ownership-asserts —— SGLANG_KT_SLOT_OWNERSHIP_ASSERT 槽位所有权不变量哨兵

> 日期:2026-09-13。**派发序 = 4/4**(observability 之后;F3 独立线并行,与本计划零行级冲突)。
> 状态:验证后修订完成,待实施。上游依据:`doc/ft-kt-phase1-audit-hidden-p0.md` §12。
> 纪律:env 治理矩阵(audit §12.4)、离线确定性对拍、不 commit。

## 0. 定位与派发理由

为 `_Mxfp4LayerwisePrefillManager` 的 host 双槽(`:2238` position%2)生命周期上哨兵:在装载/复用/覆写三枚锚点断言四层不变量,把 2026-09 NCCL 死锁案件家族的"静默错序"前置成线上一行 explicit phase 错误(或 warn-only 日志)。它与 observability 是**串通账**——观测账本回答"慢在哪",本哨兵回答"顺序错没错"。

**采纳 verify/critic 三项硬裁决**:
1. **level-1(INT4/FP8)整组剔除**(open question 6 拍板 + critic 背书`:1313 do_write rank0-only + :1323-1324 全 rank barrier` 错位 = 2026-09 死锁族)——第一阶段 v1 只剩 **MXFP4 manager 四类断言 A/B/D**;级 1 日后若启用,必须重设计为全 rank 预共识 raise(附一条 gloo broadcast 1B)或 warn-only,且永远落在 rank 对称代码点。原锚点勘误:`:1150-1422` 实为 `_prepare_weight_fp8`;真 INT4 在 `:962-1149`(slot=expert_id%2 `:1084`;FP8 slot=idx%2 `:1310`)。
2. **影子键升级为 (epoch, layer_idx, position)** —— 原 (epoch, position) 方案在 41 层轮用同 position 双槽(`_acquire:2391-2394`、`_prefetch_successor:2408`、_advance_round `:1950-1960` 仅新一轮 epoch+1)下断言恒真,等于烧一整轮灰度验一个永不触发族,**列为 v1 阻塞项**。
3. **对拍改离线确定性**(critic:在线 server 组批受时序影响,sha256 parity 按现状不可验收而非噪声,一票否决)——离线 Engine 或单并发串行 server + 固定 8 prompt + radix cache 冲刷 + `chunked_prefill_size`/`max_running_requests=1` 钉死 + NCCL 版本记档。

## 1. 设计(validation 修订后最终版)

### 1.1 不变量(均为 MXFP4 manager,锚点已重锚)
- **A. host_slot 对等**:装载完时 `position % 2 == host_slot` 真值(bitsA);且 host_slot ∈ {0,1}。
- **B. 槽位所有权**:同一 `(epoch, layer_idx, host_slot)` 任意时刻至多一个 position 拥有;覆盖写入前旧 owner 必须已 free 或已由本轮树除。
- **D. reuse_guard 自动机合法**:状态 ∈ {None, consumed, ready, raw, synchronized→loading}:consumed→ready→synchronized→loading 顺序合法,跳过或倒序 = 违例。
- **E(consumed-before-overwrite 合并入 B)**:覆写该槽位 host pinned buffer 前,其消费事件须已触发(对齐 `:2248-2249` host_slot_free_events)。

### 1.2 影子表(v1 阻塞项裁决)
- `_host_slot_owner[2]`:每槽记录 `(epoch, layer_idx, position)`,在 `:2289-2292` position+host_slot 赋值处写入三元组。
- owner 断言 = **同 epoch 内 `owner.layer_idx == slot.layer_idx` 且 position 匹配**;跨层同 position 重用必须发生在新 epoch 后,否则违例。
- **清理挂早回**(:1932-1933 `abort_round` early return 前):影子记录随 round 而弃,cleanup 必须在 return 语句之前;或在 `:1931` 注释"epoch 失效即隐式全清",二选一写清。
- world_size=1 时断言目标 = **RuntimeError with phase message** 经 `:2035-2039` 短路由入 `:2052-2060` 统一出口,而非原地 raise(对齐既有 commit 通道语义)。

### 1.3 emit 通道
违例按 (key=断言类型+layer_idx+slot+epoch) dedup warn-once;level=1 时经 `_commit_tp_device_runtime_phase`(:2052-2060)/`_commit_tp_runtime_phase` 共识出口把 local_error 上抛 —— **永远从 rank 对称代码点**进入该通道(:1997-2023 全 rank gloo / :2040-2046 全 rank device 已亲验;xysa10 layerwise 常态不走 rank 不对称路径,fallback/_prepare_weight_mxfp4:1410→FP8 机制已在级 1 剔除范围外)。

### 1.4 CUDA graph 守卫
**不得**以 `torch.cuda.is_current_stream_capturing()` 为开关(parallel_state.py:1134/:1223 先例:捕获期该谓词语义与用例地对齐不保证)。断言全部落在 prefill 专属路径(layerwise 仅 ≥threshold 进入),运行期守卫留 capture-agnostic 设计:**open question 3 提升为档0 退出条件** —— 从 xysa10 启动参数导出 `kt_gpu_prefill_token_threshold`,与 CUDA-graph 捕获最大 num_tokens 对拍,二者都写进 PR;若 threshold ≤ capture tokens,则 manager 路径本身即预存 capture bomb,该问题单独立项处理(哨兵不当替罪羊)。

### 1.5 env 门(纪律修正 + 治理矩阵落地)
- `SGLANG_KT_SLOT_OWNERSHIP_ASSERT`:0=off / 1=enforce(共识通道 raise)/ **2=warn-only(默认)**;非法值 warn-once 回落 2。
- 读取时机 = **manager `__init__`(`:1879` 附近)首次使用即缓存 int,运行期不可切换**——与仓内 inline 先例(:4347/:4552/:4616 等每次 apply 读)明确不同,这是业主故意(双 rank 语义一致 = 结构约束),文档如实写;改值需重启。**与 audit §12.4 治理矩阵对齐,并顺手把全表写进新增 docstring 小节**(治理矩阵维护权移交本计划,critic Q2)。
- **docstring 不落 `:9-28` KT-DEBUG-ONLY 块**(那是另一纪律区块);新写一段 "Production invariants guard"(0/1/2 语义、默认 2、非法→warn-once+2、读取时机、切换需重启)。
- Wording 修正(D 覆盖说明):写"**自每轮首更新窗口(:2473-2482)之后生效;每轮首层 prime 装载不覆盖**",并把 skip-reason 做成每键一次性日志,杜绝单层单槽单 epoch 重复行。

### 1.6 触及文件
- `kt_ep_wrapper.py`(fork 独有):断言 + 影子表 + emit + docstring 新小节。
- **新建** `test/manual/kt_slot_ownership_assert_test.py`(手工对拍脚本,N1-N5 注入)。
- ~~deepseek_v4.py~~ 已从清单剔除(critic:-arch 默认不做,留名防误启)。
  files[] 滤镜下不再出现。

### 1.7 对拍(离线确定性,critic 一票否决修复)
固定 8 prompt × (512 prefill + 64 decode),**离线 Engine 或单并发串行** 跑两轮:
- baseline(`SGLANG_KT_SLOT_OWNERSHIP_ASSERT=0` 或未设)vs candidate(`=2`);
- dump KT 层 {20, 40, 60} MoE 输出 hidden_states + 最终 logits(fallback 字符串哈希);
- sha256 逐位相等,容许差 = 0;
- 冲刷 radix cache 双轮之间;`chunked_prefill_size` 与 `max_running_requests=1` 钉死参数;NCCL 版本/拓扑入库;
- **layer 0 且先甩手**(非 KT 层∶3348-3352),不参与 dump。

### 1.8 注入用例(N1–N5,全在 world_size=1 单 rank 开跑通 `:1998-2002/:2035-2039`)
- N1 host_slot 翻转:**= monkeypatch `:2238` 赋值**(不改源码;`position % 2` 语义保持不变 → 强触发 B/E)。
- N2 状态机断档:`reuse_guard=consumed` 而 `has_consumed_event=False`(构造 consumed-before-overwrite 违例)。
- N3 epoch 回卷:`slot.epoch` 从 5 回 3(monotonic 违例)。
- N4 默认等价:`unset` vs `=0`,两路 sha256 全对。
- **N5(verify 新增):运行期改 env 无效**——进程内设 `os.environ[...]="1"` 后(env 已缓存)行为不切换,日志仍可打 warn-once;断言目标 = logger 侧而非 raise(与"读一次固化"语义配套)。

## 2. env 门(总表)

| env | 默认 | 读取时机 | 运行期可切换 |
|---|---|---|---|
| `SGLANG_KT_SLOT_OWNERSHIP_ASSERT` | `2`(warn-only) | manager `__init__` 缓存 int | 否(改值须重启) |
| `SGLANG_KT_PARITY_DUMP` | 空=off | apply 入口 inline | 是(对拍钩子) |

非法值 → warn-once + 回落 2。治理矩阵全表移交 docstring 新小节(critic Q2 移交)。

## 3. 实施步骤

1. 符号/grep 重锚全部行号(落在 observability 合入的工作区之上;`:2331-2333` 共享 bracket 复用 observability helper 不重复插桩)。
2. `:1879` 附近缓存 env int;`_host_slot_owner = [None, None]`。
3. 影子表写入点 + owner 断言(三元组语义)。
4. 四类断言入锚(bitsA/B/D/E)。
5. emit 通道挂 commit 函数;dedup + skip-reason once-per-key。
6. docstring 新增 "Production invariants guard" 小节(env 治理矩阵全表)。
7. cleanup-before-early-return 落到 `:1931-1933`。
8. `test/manual/kt_slot_ownership_assert_test.py` 新文件写 N1-N5 注入 + 离线对拍驱动。
9. 档0 退出条件交付:`kt_gpu_prefill_token_threshold` 导出 + 与 capture max num_tokens 对比,两数入 PR 描述。
10. 灰度文档与监控面板说明(warn 键空间、去重键、回落语义)。

## 4. 对拍与负注入

- 硬标准:离线确定性对拍(§1.7)全绿 = 验收前提;在线对拍不可验收的案子由 critic 一票否决钉死,不再复审。
- N1-N5 注入依次验 A/B/D/E + 默认等价 + env 固化;每条断言路径在双 rank 对称点抛 RuntimeError with phase tag。
- 反注入:`SGLANG_KT_SLOT_OWNERSHIP_ASSERT=1` 下不注入违例时一轮无 RuntimeError、无 `(family, code)` dedup 项、host 开销 <0.1%。

## 5. rollout

- **档0(PR merge,默认 2)**:CI + 离线对拍 + N1-N5 全过 + **open-question-3 退出条件交付**;从 PR diff 看全部新增均 derank 静默。
- **档1(xysa10 canary 2 周,显式 `=2`)**:监视 warn-once 键空间增长率≥0/天为 0;一旦有键触发即按 family 走独立排障(哨兵本身不背锅)。
- **档2(enforce `=1` 2 周)灰度**:host 开销 <0.1%、无 dedup 键触发、无 NCCL 超时增量;违规即回 `=2`。
- **档3(常驻 `=1`)**:`=0` 保留为 kill-switch;全表治理矩阵同步更新。

## 6. 热更新交互 / collectives / CUDA graph

- 断言点全在装载/层装载 commit 路径,不触碰 `:2470-2487` 更新窗、`:2253-2275` per-expert 共识、`:2473` synchronize 本体;影子表纯 host。
- 新增 collective = 0;emit 走已有 `_commit_tp_device_runtime_phase` 通道,次数与布尔分支语义保持不变(违例时才入共识出口,无副作用注入)。
- CUDA graph:守卫依据 = layerwise 路径仅 ≥threshold,**预存 capture bomb 排查归入档0 退出条件**;不以 `is_current_stream_capturing()` 为开关,先例已反。

## 7. 工时

约 **5 人日**(实现+影子表 1.5;docstring+治理矩阵 0.5;离线对拍驱动与 N1-N5 2.0;档0 退出条件与灰度文档 0.5;buffer 0.5)。级 1(INT4/FP8)量体裁衣时另估,不在本期。

## 8. 对抗验证记录(verify refuted=False,10 问题全部吸收;critic 追加)

| # | 问题 | 处置 |
|---|---|---|
| P1 | 级 1 原地 raise 错位(rank0-only 写 + 全 rank barrier)= 2026-09 死锁族 | §0/§1.1 级 1 **剔除**;日后启用须全 rank 预共识 raise + 1B gloo broadcast 附约 |
| P2 | 锚点张冠李戴(`:1150`=fp8;INT4 真在 `:962-1149`) | §1.1 勘误;MXFP4 host_slot=`position%2` `:2238` 钉死 |
| P3 | 在线 sha256 parity 注定噪声,非验收腿 | §1.7 离线确定性重写,8 prompt 固批 + cache 冲刷 + 参数钉死 |
| P4 | env_gate 声称"与先例同"不实(先例全部 inline) | §1.5 重写为"缓存 int、不可切换、须重启";N5 注入封语义 |
| P5 | 影子键无 layer_idx → 断言恒真 = 灰度白烧 | §1.2 影子键 **(epoch, layer_idx, position)**,列为 v1 阻塞项裁决 |
| P6 | docstring 落 `:9-28` KT-DEBUG-ONLY 块(纪律区块错配) | §1.5 新写 "Production invariants guard" 小节 |
| P7 | `abort_round :1932-1933` 早回先于 cleanup → 影子残留 | §1.2 cleanup 挂早回/epoch 失效注释,二选一写清 |
| P8 | D 覆盖说明过宽含 prime 装载不穿点 | §1.5 wording 修正 + skip-reason once-per-key |
| P9 | open question 3(threshold vs capture)是预存 capture bomb 风险 | §1.4 提升为**档0 退出条件**,双数入 PR;哨兵不做替罪羊 |
| P10 | N1 用源码编辑注入不可灰度 | §1.8 N1 = monkeypatch `:2238` |
| C-1 | 治理矛盾(env 纪律三家不一致) | §1.5 治理矩阵全表移交 docstring;读取时机如实 |
| C-2 | `deepseek_v4.py` 在 files[] 里(默认不做)防误启 | §1.6 剔除 |
| C-3 | 级 1 若在 xysa10 域过度"严重"评级 | §0 采纳剔除方案,档后另议;验收红线 = 永不对 xysa10 主路入 rank 不对称点 |

## 9. 主控裁定项

1. 默认 `=2`(warn)即 config-first 纸面默认关的偏离 → 与 explicit-degrade 的"预授权 warning-once"**同案一次拍板**,写进治理矩阵。
2. 级 1 重设计预算(全 rank 预共识 raise ≈ 1 人日)是否单独立项;本期不做是否同意。
3. 档0 退出条件(threshold vs capture)若发现 threshold ≤ capture tokens → 是否转独立 P0 立案(哨兵文档只记录不背锅)。
4. 档3 是否常设 kill-switch(`=0`)为回流预案,还是逐步退出 0 档。
