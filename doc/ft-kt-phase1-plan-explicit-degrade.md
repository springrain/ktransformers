# Phase-1 计划:explicit-degrade —— KT 静默降级全清单 reason 链 + 一次性告警 + ABI 探针

> 日期:2026-09-13。**派发序 = 2/4**(benchbw 之后)。状态:**blocked-on-test-design**(critic 定案)——
> 原计划四条主对拍腿被 `:4027-4032` 启动期 raise 全部挡死,须按 §10 重构注错方案后才可开工。
> 上游依据:`doc/ft-kt-phase1-audit-hidden-p0.md` §12。纪律:默认逐位等价、警告-once 预授权待主控拍板(§9)、不 commit。

## 0. 目标（摘要）

给 KT 全部静默降级 gate 加 reason 链与一次性告警：

1. `kt_ep_wrapper.py:2514-2604` 四个布尔 gate 重构为 `(bool, reason)` 孪生函数，布尔逐位不变，False 分支在 gate 内经统一 `_kt_degrade_emit()` 做去重告警；八个 dispatch 站点（`:2655/:2688/:2775/:4030/:4044/:4417/:4443/:4458`）一行不改、只透传 bool。
2. 接通 `:2526` 死旗标（v4-helpers-missing 告警，全文件证实从未被读）。
3. `:2655-2656` KV-budget 静默 continue 注入签名级 reason（第 9 个 gate;kv_cache_configurator.py:2021-2037 消费点 =0 时完全静默）。
4. `:1822-1826` 与 `:4518-4531` 两处把串行 full-GPU 兜底错标为 "layerwise prefill" 的 info 改名（文本级 bug 修复，不受压制阀管辖）。
5. `:2592` 缺陷（`in ("0","1")` 有意 opt-out `=0` 也禁管线）默认保留逐位语义；`SGLANG_KT_V4_TRITON_STRICT_OPTOUT=1` 才纠正；legacy 下读到 `=0` 时告警指路。**`:2592` env 读改为注册期锁存为 method 属性**(critic 修订）：双 rank gate 判定从偶然约束变结构约束，同时消除 capture 窗口边缘。
6. ABI 探针：`kt-kernel/python/experts_base.py` 增 `_ensure_forward_task_abi(moe)`，首个 forward_task 提交前解析 pybind `__doc__` 是否含 `autofree`(e26c065 第 8 参，ext_bindings.cpp:475-477),漂移即 RuntimeError —— 把"首次 prefill 中段 TypeError"前移到启动路径。ARM NEON 类由同一 `bind_moe_module` 模板绑定（ext_bindings.cpp:**:1014-1017**,行号已修正），探针同构覆盖。

默认纯增强：gate 布尔表达式不动、无 CUDA op、无新 collective、无张量分配。

## 1. 设计(validation 修订后最终版)

### 1.1 机制（下沉式 reason 链）
- 模块级 reason 字面量常量组：NOT_REQUESTED / V4_HELPERS_MISSING / CUDA_UNAVAILABLE / GPU_METHOD_NOT_MXFP4 / RAW_ATTRS_MISSING / RAW_SHAPE_UNALIGNED / GPTOSS_ACTIVATION / TRITON_ENV_OVERRIDE / DEVICE_CAPABILITY / RAW_SOURCE_ABSENT / RAW_SOURCE_INCOMPLETE / KV_BUDGET_SKIPPED。
- 统一发射器 `_kt_degrade_emit(family, code, detail="")`：先查 `SGLANG_KT_SUPPRESS_DEGRADE_WARN=="1"` → 返回；**dedup 键 = (family, code)**(verify 定案；detail 折叠进首行文本含层号/签名缩写，消除 41 行/族突发并与 warning-once 语义对齐）;rank0-only `logger.warning("[KT-DEGRADE] family: code (detail)")`。纯 host，无 torch/dist。
- 四 gate 拆 `*_reasoned` 孪生；外壳保持签名语义、内部委托并在 False 时发射 —— dispatch 站点零改动、bool 逐位等价。
- 死旗标 `:2526` 删除，功能由发射器接管（首判 False 时 detail 带缺失符号名/ImportError 文本）。
- KV-budget gate(`:2655-2656`)continue 前按 (signature 缩写， code) 发射 KV_BUDGET_SKIPPED,detail 含方法/权重路径/registry 层数与 OOM 后果提示。
- `:4032` ValueError 文案嵌入 reasoned code（仅文案，不改 raise 条件 —— 热更新前置硬失败语义不变）。
- TRITON legacy/STRICT 双语义 + 注册期锁存（见 §0.5)。STRICT 纠正语义：仅精确 `="1"` 禁用。

### 1.2 ABI 探针
experts_base.py 模块级 `_FORWARD_TASK_ABI_CHECKED` + `_ensure_forward_task_abi(moe)`，在含 `:847` 的提交函数顶部调用一次；先例同 `:865` hasattr 能力探针。
**critic 修订（上游面）**：原计划"experts_base.py 零上游分歧"声明不实 —— kt-kernel 是上游跟踪对象（ddb6bff 刚 merge upstream/main),与 fork 独有的 kt_ep_wrapper.py 不同，存在上游合并冲突面。处置：探针仍落 experts_base.py（检测对象就在该层）,files 清单改标"上游跟踪文件、低冲突面、merge 时注意";若主控要求零上游面，备选 = 探针挪入 fork 胶水层（kt_ep_wrapper 侧在创建 wrapper 后调用，牺牲子类普适性）。
**逃生阀**(verify P5 采纳）:`SGLANG_KT_SKIP_FORWARD_TASK_ABI_CHECK`（默认查）。过筛对象 = dev box 旧 .so 重编摩擦；生产默认开查不损失防护。
**假阳性防线**(verify P5):G0 正向上例 —— xysa10 冒烟起服断言真 .so 下探针通过且 `_FORWARD_TASK_ABI_CHECKED` 置位（覆盖 NEONMXFP4 实类）后才合入；若主控拒新增 env，至少要求 G0 在 xysa10 与任一 dev box 双验证。

### 1.3 xysa10 可达性现实(verify P1/P3,critic 确认 —— 本特性关键事实)

生产热更新开关恒开 → `process_weights_after_loading` 的 `:4027-4032` 链(`:2514-2522`→`:2543-2547`/`:2592-2593`→`:2551-2553`）使门槛=0 / 禁 triton / 缺 v4-helpers 三类 baseline **启动即 ValueError**,xysa10 上"gate False + 服务在线"的四条原对拍腿不可达；**唯一运行期可达的静默降级** = raw_source 缺失路径（`:4044-4079` 建 `_kt_mxfp4_raw_weights` 后，运行期 gate `:4443/:4458`(`:2599-2603`）翻 False → `:4455-4464` if 不命中 → **静默落回 hybrid,无 warning、无 raise**)。

此外 `SGLANG_V4_USE_TRITON_KERNELS` 全仓仅 `:2592` 一处消费；若生产真设了 `=0/1`，同一 raise 链今天服务器根本起不来 —— 生产在线（≈500 tok/s prefill）即**先验证伪**"被该 env 静默降级"假设（verify P2)。档4 STRICT 窗默认不排期，仅当档1 发现 env 残留或 layerwise 未激活证据时启动。

## 2. env 门

| env | 默认 | 读取时机 | 说明 |
|---|---|---|---|
| `SGLANG_KT_SUPPRESS_DEGRADE_WARN` | 未设=告警开 | 发射器每次被调 inline | dedup set 拦截后稳态成本=一次 set 查找；子进程翻转可测试 |
| `SGLANG_KT_V4_TRITON_STRICT_OPTOUT` | 未设=legacy | **注册期锁存为 method 属性**(critic) | `=1` 时仅 `="1"` 禁管线 |
| `SGLANG_KT_SKIP_FORWARD_TASK_ABI_CHECK` | 未设=查 | import 期 | dev box 逃生阀 |

逐位等价论证：gate 布尔表达式树、分支走向、异常类型/时机、collective 序列、CUDA op 一概未动；唯一新副作用 = host `logger.warning`;`=1` 压制时连日志都无新行（两处 info 改名是独立列出的文本级修复）。
**capture 边界**(verify P4 并入）:decode 捕获路径仅 `:4417` 纯 host 判定执行；首次 False 若恰落 capture 期，warning 在 capture 时发出而 replay 不发 —— 行为安全（Python 不入图）,但 parity 断言锁定"首评估早于 capture"作为不变量（见 §4)。

## 3. 触及文件与合并协议

| 文件 | 性质 | 内容 |
|---|---|---|
| `kt_ep_wrapper.py` | fork 独有 | gate 重构/发射器/死旗标/KV-budget reason/两处 info 改名/triton 阀 |
| `kt-kernel/python/experts_base.py` | **上游跟踪（critic 修订）** | ABI 探针 |
| `kt_ep_wrapper.py:9-27` docstring | — | 新 env 说明（env 治理矩阵见 audit 文档 §12.4) |

合并协议（critic Q1 强制）:**落地序 explicit-degrade → observability → ownership**;本计划最先落地 wrapper 侧，symbols/grep 重锚；`:2331-2333` 挂载点归 observability/ownership(本计划不碰）；每步实施前重跑 dispatch 站点 grep（原图：requested@`:2514/:2522/:2688/:4417`、backend@`:2551/:2552/:2775/:4030/:4044`、runtime@`:2598/:2601/:2655/:4443/:4458`，无仓外调用者）。

## 4. 对拍（重构双轨，替换被挡死的四腿）

**轨 a — xysa10 启动期（pre/post 双跑）**:threshold=0 等三 baseline 的期望产出改为"启动期 ValueError 且 post-patch 报错文案携带具体 reason code、pre-patch 不携带"。不做 token-id 对拍。
**轨 b — 热更新可关的测试环境(**`kt_enable_dynamic_expert_update=0`)：执行"gate False 静默降级 + 一次性告警 + token-id 逐位相等"全套（显式标注该环境≠生产拓扑，仅验证发射器与布尔等价性）。

**xysa10 可达注入腿（核心，verify P3 新增 + critic 注错约束）**:
- 注入：启动后对某一 KT 层 `del layer._kt_mxfp4_raw_weights` 的一个 RAW_NAMES 键（或置 None);
- 发一条 ≥threshold 的长 prompt;
- 断言：RAW_SOURCE_INCOMPLETE（或 RAW_SOURCE_ABSENT）恰一行、该请求静默落回 hybrid 路、**token-id 与 pre-patch 同注入场景逐位相等**;
- **注错规避（critic)**:manager 存活 + gate 翻 False 的另一分支走 `:4466-4470` 双 rank RuntimeError = 注错即杀服；注入脚本必须选 `:4455-4464` 静默落回分支的注入面（del 键后先发低于"manager 已建"条件的流量验证分支归属），并在一次性/可弃部署上先跑 `:4466-4470` 分支的探测，确认两分支边界后生产对拍只走静默分支 —— 该分支边界分析文档随实施交付，**未完成前 xysa10 不注入**(blocked-on-test-design 的解除条件）。

**其余 parity 不变量**:N4 monkeypatch capability→DEVICE_CAPABILITY 恰一次；N5 monkeypatch `forward_task.__doc__` 去 autofree→首提交即 RuntimeError；真 .so→通过且 flag 置位；N6 caplog 按 **(family,code)** ≤1 行校验（粒度已同步）;**capture 边界断言**:B 档 + CUDA graph decode 下全部 `[KT-DEGRADE]` 行时间戳早于 "Capture cuda graph" 日志锚点；若有行落 capture 窗口内，emitter 需在捕获检测（`_graph_capture_active` 先例 experts_base.py:163-172）下降级为捕获后补发。

## 5. rollout

- **档0** 开发回路：单测 N4/N5/N6 + import 冒烟；**新增 G0 正向 ABI 真 .so 用例（双验证）**。
- **档1** xysa10 无流量起服（默认告警开）:`env|grep SGLANG_V4` 审计残留（成本为零，保留）；退出=启动日志仅出现预期 `[KT-DEGRADE]` 集合、KV 预留数值与昨日一致、无 ERROR。**先验结论已书面化：生产被 triton env 静默降级的假设已证伪（§1.3)，此审计为残留确认而非开放问题。**
- **档2** 停机窗对拍：轨 a + xysa10 可达注入腿（注错边界文档完成后）+ 轨 b 在测试环境。
- **档3** 生产灰度：混入 ≥1 条过长 prompt 覆盖 KV-budget reason 路径，24h 浸泡；每 (family,code) ≤1 行、prefill tok/s 与 TTFT 波动 <2%、控制面无新告警。
- **档4** STRICT 阀：默认**不排期**；仅档1 发现残留或未激活证据时单独开窗（+0.5 人日，重剖析）。
- **档5** 全程不替用户 commit;wrapper 文件不上游，experts_base 探针随 kt-kernel 上游流（merge 注意面已标）。

## 6. 热更新交互 / collectives / CUDA graph

- 不增删 `:2470-2487` 更新窗口任何语句；发射器不读写 mask/raw_weights/slot 事件；`:2253-2275` 热循环不在其调用链；`:4030-4036` raise 条件不变，热更新恒开天然兼容。
- 无新 collective；分支由 gate 布尔决定而布尔逐位不变 → `:2042-2050` all_reduce 计数不变；gloo `:2704-2717` 同理；`:1323-1324` barrier 所在函数未触碰；reason 发射为 rank0 单边日志，不读取对端状态、不引入新分歧面；kv profiling 期 `:2655` 发射不在任何共识链。
- CUDA graph：捕获路径仅 `:4417`（阈值 int 比较+字符串比较+缓存查表）；首评估早于 capture（注册 `:4019→:2688` 构造期、`:4030/:4044` 权重处理期、`:2655` KV profiling 期均已跑，dedup 集合捕获前收敛 —— parity 新增锚点断言锁定）;`:2592` 注册期锁存后连"capture 窗口边缘"一并消除。

## 7. 工时

**3.5-4 人日**(verify 上调：对拍负腿双轨重构 + 新注入腿 + capture 边界断言 +0.5~1.0；档4 默认不排期抵消 ~0.5)。基线：实现+自审 diff 1.0d；单测/注入脚本 0.5d;xysa10 档1-档3 双跑浸泡 1.0d；文档 0.5d。

## 8. 对抗验证记录(verify refuted=False,7 问题全部吸收；critic 追加)

| # | 问题 | 处置 |
|---|---|---|
| P1 | N1/N2/A 基线三腿+parity C 被 `:4027-4032` 启动 raise 挡死（xysa10 恒开热更新） | §4 双轨重构；critic 标 blocked-on-test-design |
| P2 | triton env 静默降级假设可被先验证伪 | §1.3 书面化；档4 默认不排期 |
| P3 | xysa10 唯一可达静默降级（raw_source 缺失）零覆盖 | §4 新注入腿（含 critic 注错规避） |
| P4 | dedup 粒度 + capture 窗口首发射 | 键=(family,code);§2/§4 capture 边界断言 |
| P5 | ABI 探针假阳性硬失败敞口 | §1.2 G0 正向真 .so + 逃生阀 |
| P6 | ext_bindings 锚点 `:1013`→`:1014-1017` | 已修正 |
| P7 | gate 调用点全图/共识不涉/INFO 改名自洽 | 亲验记录，维持 |
| C-1 | "experts_base 零上游分歧"不实 | §1.2 改标上游跟踪 + 备选挪 fork |
| C-2 | `:2592` 应注册期锁存 | §0.5/§1.1 采纳 |
| C-3 | 默认开告警偏离"默认关逐位等价"字面 | 待主控裁决（§9.1) |

## 9. 主控裁定项

1. **warning-once 预授权**:explicit-degrade 默认开告警（与 ownership 默认 level=2 同案）—— 视为特性规格预授权，还是改为默认压制？请一次拍板，写入 env 治理矩阵（audit §12.4)。
2. ABI 逃生阀 env 是否接受（拒则 G0 双验证后合并）。
3. info 改名 + `:4032` 文案是否同批（倾向同批，diff 极小）。
4. kt-kernel 侧单测落点：新建 tests 文件 vs 并入冒烟脚本。
5. 生产 sglang logger WARNING 级别在部署管道可见性（不可见需 stdio 兜底约定）。
