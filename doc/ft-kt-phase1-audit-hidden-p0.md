# Phase-1 审计报告：隐藏 P0 —— 热更新逐出初始 GPU 常驻专家后 CPU 侧读取从未打包的 BufferB

> 日期：2026-09-13。方法：只读代码审计(16-agent workflow，三路独立审计 + 综合 + 对抗复核 + 终审)。
> 范围：xysa10 生产拓扑(aarch64 168 核 / 2×RTX PRO 6000 Blackwell sm_120 / PCIe 无 NVLink / TP=2)上的
> DeepSeek-V4-Flash 2604B MXFP4 MoE(61 层 × 256 专家，`--kt-num-gpu-layers=20` → 41 个 KT 层，每层 24 个初始 GPU 常驻专家，热更新开关恒开)。
> 所有行号均为审计时工作区亲验锚点。

## 0. 结论

**verdict = reachable(可达，非理论)。** 三条独立审计轨(选择器行为、CPU 字节来源、decode 合并路径)各自独立得出"可达"，综合 agent 复核成立，对抗 verify agent 独立重验 15+ 证据行全部命中(唯一噪音：旧文档把 uniform 回退行写作 `:3232`，实勘为 `:3236`)，终审 critic 第三次独立复验全部挂载行号无误。

失效不是竞态、不需异常流量：在真实生产流量下**几乎必然发生**，且所有前置条件都是部署常量。

## 1. 前置条件(xysa10 生产既定，非竞态)

- `kt_method=MXFP4` + `kt_enable_dynamic_expert_update` 恒开(生产硬约束，不可关)
- `--kt-gpu-prefill-token-threshold > 0`
- `--kt-num-gpu-layers=20` → 41 个 KT 层(20..60)，每层 24 个初始 GPU 常驻专家
- aarch64 → NEON `NEONMXFP4_MOE` 算子族

## 2. 缺陷本体(两行事实)

1. **装载期初始 24 专家跳过 host 打包**：C++ TP per-expert 打包循环 `mxfp4-moe.hpp:477` 对 `should_skip_expert`(`common.hpp:255-258`：仅 `expert_id<0 || >=expert_num || gpu_experts_mask[id]` 掩码门控，**无 loaded/内容检查**)命中的专家直接 `return`。同款早退散布于非 TP 打包点 `:241/:255/:280/:294/:304`。
2. **对应 BufferB 永不写入**：`moe_base.hpp:155-171` 对全部 256 专家**无条件** `std::aligned_alloc(64)` 分配三块 BufferB，且全文件**无 memset/calloc/任何初始化**(裸块由 `owned_aligned_allocs_` :98 持有至析构 :193)。

→ 24 专家 × 41 层 = **984 个专家的 BufferB 自始分配未写**(≈17.56 GiB 未初始化物理页)。

## 3. 触发链(三阶段)

### ① 装载期(洞口成形)

- `_init_kt_gpu_experts_masks` 生成初始掩码(uniform 回退全 1.0 `kt_ep_wrapper.py:3236` → `experts_base.py:227` 全局 top-k，并列时取小 ID `:3450-3455`，**与真实流量分布无关**)。
- rank0 创建 wrapper(`:3950`)并经 `:3979` 传掩码；`experts_base.py:495-499` 拷入 pinned bool 张量，`amx.py:1142` 把 `data_ptr` 交予 C++(`common.hpp:242` 为 `uint8_t*`)。
- `kt_ep_wrapper.py:4112`(全 sglang 树**唯一** `wrapper.load_weights` 调用点，已 grep 核实)→ `amx.py:1330-1331 load_weights_task` → C++ 打包循环对掩码命中者早退(见 §2)。
- 装载收尾 `amx.py:1334-1342` `del` 权重/scales 并 `_release_loader` —— **运行时全树再无 host 重打包原料**；热更新对 CPU 侧的全部作用仪为 `:3729` 一行 pinned `.copy_`。源不可恢复。

### ② 触发逐出(常态化行为，无竞态)

任一 prefill 批 `num_tokens=int(x.shape[0])`(:4345)`>= threshold` 即过 `_full_gpu_gate`(`:4412-4416`，**无 forward_mode 守卫**)→
layerwise manager 层末窗口：`torch.cuda.synchronize`(:2473)→ `_update_gpu_experts_from_batch`(:2474)→
`select_top_experts_from_batch` **只统计当前批**(`:3446 index_add_`；grep 全仓证**无累计/decay/迟滞**，`decode_freq|update_freq|freq_decay|accumulate` 零命中)→
TP broadcast `:4727-4732` → `copy_experts_weights_mxfp4`(:4737-4743)→
`update_gpu_expert_mappings` **从零重建** new_mask(`:3691` zeros + `:3692` 仅 selected=True，旧常驻位清零)→
GPU 侧 `:4792` / `:4794` 与 rank0 wrapper pinned 掩码 `:4797-4799`→`:3729` **原位翻转**(`:4788-4790` 注释明示 CUDA-graph 须 in-place)。

只要该批使任一初始常驻专家跌出批内 top-24(初始集来自静态表或 uniform 并列，真实流量下几乎必然)，该专家位被清。翻转发生在批同步点之后、下一批之前，**无需竞态**。

### ③ 首次读脏字节

其后任一 decode(或低于阈值的 hybrid prefill)批路由命中该专家 →
rank0 `_submit_cpu_forward` 原样提交**未过滤** topk_ids(`:4286-4293`；rank!=0 于 `:4306-4307` 返回 zeros)→
`moe-tp.hpp:203` 唯一防护是**层级** `weights_loaded` 标志(非逐专家)→
`moe_base.hpp:235/:309/:375/:401/:530` 掩码门控放行 →
NEON fused decode 读取从未写入的 BufferB(`b=aligned_alloc` 基址非空，`neon_mxfp4_gemm.hpp:103`；`b==nullptr` 唯一防护不可触发；`from_raw_mat` :112 是唯一写入者且仅打包路径调用)→
加权合并 `:530-539` → rank0 `:4635` 取回后 `:4646 output = output + cpu_output` **静默并入** → 经 deepseek_v2 all_reduce 扩散。

缺陷窗口**永久**(直至进程重启)。

## 4. 第二污染通道(verify 复核追加，波及面扩展)

被逐出的初始专家此后进入 `cpu_expert_ids`，**full-GPU prefill(>threshold)同样吞下脏字节**：

- `_submit_host_write`(:2266 rank0-only → `:2062-2086`)把 BufferB 导出到 GPU SHM(`write_weights_to_buffer` `mxfp4-moe.hpp:320`，校验仪指针非空/索引合法 `:325-328`，方向恒为 BufferB→GPU，从不回写);
- `:2237-2289` 在 transfer_stream 上逐 bank `destination[expert_id].copy_(cpu_buffer[host_slot])`(:2281-2284)把**从未打包的字节 H2D 注入 GPU 全专家层**。

→ 逐出后该专家的脏权重同时污染 CPU NEON 路(decode/hybrid)**与** GPU prefill staging 路。

## 5. 失效模式

**主表征 = 静默错值，非崩溃、非越界、非 UAF、无日志。**

- 块存活(`owned_aligned_allocs_` 持有)→ 非 UAF；读取范围=分配区间 → 非 OOB;`b` 非空 → 唯一防护不触发。
- xysa10 Linux：每块 4-5MB 远超默认 128KiB mmap 阈值 → 走**匿名 mmap 零页**。packed nibble `0x00`(E2M1 值 0)、fp32 scale `0.0` → 该专家输出恰为 **0 向量**，经 routing weight(softmax≤1)缩放后于 `:4646` 并入 —— 等效**该专家静默掉线**的精度腐化，无 NaN。
- **残余风险**:glibc 动态 mmap 阈值在大量 mmap 释放后可升至 32MiB；若 BufferB 块落在回收堆上，字节为任意位型、scale 可为 NaN → 经 fmadd→bf16 cast→`:4646` 加法后沿 RMSNorm 扩散为**全局 NaN 级联**。仍是错值而非崩溃。
- 两种表征全程无声，**生产可观测性为零**。

## 6. 波及面

自首个过阈 prefill 批发生首次逐出起，全部后续流量中凡被路由到被逐出初始专家的 token(41 层 × 256 专家面，每层至多 24 个候选):

- CPU 侧贡献为错值经 `:4646` 静默并入(decode/hybrid);
- GPU 侧 staging 注入错值经 full-GPU prefill 并入(prefill)。
- 逐出是热更新**常态化**行为(批粒度零重建、无迟滞)，波及面随服务时长只增不减。

## 7. 修复方案评估(综合-agent 候选 ×5)

| 方案 | 描述 | 工时 | RAM/VRAM | 风险 | 推荐 |
|---|---|---|---|---|---|
| **F1** 逐出时补打包 repack-on-demote | demote 时从 GPU canonical raw D2H 拷回再按 `mxfp4-moe.hpp:489-515` 同款 TP 切分打包 | 高(新跨语言 API+两阶段协议) | +0；每次逐出 +亚 ms PCIe | **中-高**：热更新互斥窗内新增同步点，与硬规则冲突面大 | ❌ |
| **F2** 永久钉住初始 24 pin-initial | 选择器 only-in 约束，初始集永不逐出 | 低-中 | +0 | **中**:xysa10 每层恰 24 槽且初始集即 24 个 → 约束=抽空全部可动槽位，热更新名存实亡，违反生产硬规则 | ❌ |
| **F3** 装载期打包全部专家(含初始 24) | env 门控(`SGLANG_KT_PACK_GPU_RESIDENT_HOST`，惯例 :4347/:4552/:4616，默认关=逐位等价)取消 `mxfp4-moe.hpp:241/:255/:280/:294/:304/:477` 六处早退 | 低-中(KTConfig/MOEConfig flag + 六处条件化；需 aarch64 重编译) | **RAM +0B**(块本已分配)+VRAM +0；装载打包段 +~10.3%(232→256 专家/层，秒级) | **低**：无运行期语义变化；默认关满足回落 | ✅ 本体 |
| **F4** 有效性位图+硬断言 | C++ per-expert packed 标志，`moe_base.hpp:235/:309/:401/:530` 门控处 mask=CPU 且 !packed 即 throw/告警 | 低(纯 C++) | +256B/层位图 | 单独用=把静默腐化转**线上 crash**;canary 形态风险低 | ⚠️ 仅作 canary |
| **F3 + F4-canary** | F3 为修复本体(env 门控→aarch64 编译冒烟→对拍→灰度翻默认),F4 以 warn-only 日志伴生作检测面 | 中 | 同 F3+F4 | **低** | ✅✅ **推荐** |

**反对项回收**:F1 与热更新互斥审计冲突(每次逐出付 PCIe+新同步面);F2 等价关闭生产自适应；F4 单独部署=可用性事故。
F3 与五个 phase-1 计划**零行级冲突，可独立并行推进**(唯一耦合：装载侧打包量 232→256 专家/层，observability 基线标定应排在 F3 之后)。

## 8. 合入前 interim 缓解(全部 config-first、零行为变更)

1. **启动审计日志**(纯日志)：掩码定稿后(`:3288` 缓存/`:3371` 取行处)rank0 打印每层 GPU 常驻专家计数与 ID，并 WARN 明示"这些专家从未 host 打包，被逐出后 CPU 路径将算错值"。
2. **运行期侦测**(env 门控、默认关的近零风险小补丁)`:4791-4799` 翻转处对照保存的初始掩码计算 `evicted_initial`，非空即 ERROR 级输出 layer/expert 列表供质量事故对时。
3. **摆放卫生**:`--kt-expert-placement-strategy=frequency` 配代表性生产流量采集的 activation-count `.pt`，使初始 24/层为最不可能被逐出者 —— **只降触发频率不能消除**(批粒度零迟滞选择器 `:3416-3460` 保证漂移终将发生)。
4. **事故应急最后手段**:把 `--kt-gpu-prefill-token-threshold` 调到永不触发的量级以事实上冻结逐出(热更新开关保持开启)。代价=放弃自适应，仅在确认线上腐化时采用。
5. **不要**试图用现有 env 旁路 CPU 路径 —— `SGLANG_KT_BYPASS_GPU_MOE`(:4616)只旁路 GPU 侧，无对应开关可停 CPU 半区。

## 9. 证据锚点表(全部亲验；C++ 侧)

| 锚点 | 事实 |
|---|---|
| `common.hpp:255-258` | `should_skip_expert` 仅掩码门控，无 loaded/内容检查 |
| `moe_base.hpp:155-171`,`:98`,`:193` | BufferB 对全部 256 专家无条件 aligned_alloc，全文无 memset/calloc；裸块生命周期 |
| `moe_base.hpp:235/:309/:375/:401/:530` | prefill 去重循环与 decode/加权求和全部纯掩码门控 |
| `mxfp4-moe.hpp:241/:255/:280/:294/:304` | 非 TP 五处 per-expert 打包点全部 `if should_skip_expert return` |
| `mxfp4-moe.hpp:477`(:489-515 佐证) | TP(NUMA) per-expert 加载同款早退；跳过专家的 N 两半与 down 列切片均不写入 |
| `mxfp4-moe.hpp:320/:325-328` | `write_weights_to_buffer` 校验仪指针/索引；方向 BufferB→GPU 导出，从不回写 |
| `neon_mxfp4_gemm.hpp:103`(:112) | ctor 令 `b=ptr` 非空指向未初始化块；`from_raw_mat` 是唯一写入者且仅打包路径调用 |
| `moe-tp.hpp:203` | TP 层 forward 唯一防护是层级 `weights_loaded` 标志，非逐专家 |

(Python 侧：见 §1-§4 内联锚点；含 `:3691-3692`、`:3729`、`:4112`、`:4286-4293`、`:4412-4416`、`:2473-2491`、`:3446`、`:4635/:4646`、amx.py`:1334-1342`、`:1142`、experts_base.py`:495-499`。)

## 10. 对抗验证记录

- **audit:verify(独立第二意见)**:verdict 维持 reachable;15+ 锚点亲验命中；唯一噪音=旧审计文档 uniform 回退行号 `:3232`→实 `:3236`;追加**第二污染通道**(§4)使波及面从 decode/hybrid 扩至 full-GPU prefill;"load_weights 唯一调用点"署名不精确但实质成立(`:4112` wrapper.load_weights → `amx.py:1330`)。
- **critic(终审，第三次独立复验)**:verdict 成立；全部挂载行号亲见无误；确认 F3 与五方案零行级冲突、可独立并行；F3/F4 评估与组合推荐背书。

## 11. 残余待验证清单(xysa10 实测/书面项)

1. **BufferB 页来源实测**:`/proc/<pid>/smaps` 对初始 24 专家的 BufferB 段验证匿名 mmap 零页假设(决定主表征 vs NaN 级联的概率分布)。
2. **glibc mmap 阈值**:检查生产进程 `M_MMAP_THRESHOLD` 动态抬升状态(大量 mmap/free 后可至 32MiB),评估脏页风险敞口。
3. **rank0/rank1 掩码一致性**：翻转 `:4791-4794` 双 rank 各自执行 + rank0 pinned `.copy_` 负责喂 C++;书面论证 rank1 永不读 BufferB(rank!=0 于 `:4306-4307` 直接返回 zeros)后，仅需抽查 rank0 翻转原子性。
4. **最小 testbed 复现**:2-4 专家 toy NEONMXFP4_MOE + 1 个初始 GPU 常驻位，装载后强制逐出并请 decode 命中，读 BufferB 输出应为精确零向量(零页场景)——复现成本远低于生产注入，建议随 F3 对拍一并完成。
5. **`kt_gpu_prefill_token_threshold` xysa10 实际取值**:从生产启动参数书面导出(同时为 ownership-asserts 档0 退出条件与 routing-probe BLOCKED 状态共用输入)。

详见同目录计划文档；routing-probe 已按 verify(refuted)与 critic 裁决**移出第一阶段**。

### 12.2 派发序(critic 定案)

1. **benchbw**(修复 P0/P1 后最先执行)——它是 explicit-degrade 的 KV-budget reason 与 observability unaccounted 归因(22 vs 11 GB/s 判读树)的共同上游前提；其判读缺陷不修，后两方案归属论据全部悬空。
2. **explicit-degrade**(重设计注错腿 + :2592 env 注册期锁存)——状态 **blocked-on-test-design**(见该计划文档 §10)。
3. **observability**(含 churn 搭车指标，接替 routing-probe 数据需求；**F3 落地后再标定基线**)。
4. **ownership-asserts**(补 layer_idx 影子键 + 级 1 重设计 + env 纪律对齐)。
5. **F3 独立线并行**(装载期全部打包 + aarch64 编译冒烟 + 对拍 + 灰度翻默认)。

### 12.3 跨方案合并协议(critic Q1,强制)

explicit-degrade / observability / ownership-asserts **同改 `kt_ep_wrapper.py`** 且行锚交织(docstring 头 `:9-27`、`:1879 __init__`、`:2180-2331/:2463-2485` bracket、`:4027-4032/:4417/:4455-4470` 发射点);`:2331` 被 **双消费**(observability 的 gloo bracket + ownership 的断言点)。规则：

- **落地序 = 派发序**(explicit-degrade → observability → ownership)，后来者以前者合入后的工作区为基线;
- 每个计划实施第 1 步**按符号/grep 重锚**,任何文档行号只作导航不作补丁依据;
- `:2331-2333` (postprocess 提交后的 `_commit_tp_runtime_phase`)为双方共享挂载点，由 observability 先行 bracket,ownership 复用其 helper 不重复插桩。

### 12.4 env 治理矩阵(critic Q2;ownership 实施时同步增补进 `:9-27` docstring)

仓内现有三种纪律并存:(i) 每次前向 inline 读(先例 `:4347/:4552/:4564/:4581/:4616/:4623/:4629/:4654/:4119`);(ii) 首次惰性解析+缓存;(iii) import 期固化。新增 env 一律不碰 `environ.py`，默认关必须逐位等价。总表随各计划合入维护于 ownership 增补的 docstring 小节：

| env | 宿主特性 | 默认 | 读取时机 | 运行期可切换 |
|---|---|---|---|---|
| `SGLANG_KT_PACK_GPU_RESIDENT_HOST` | F3 | 关 | C++ 装载期一次性 | 否(新进程生效) |
| `SGLANG_KT_SUPPRESS_DEGRADE_WARN` | explicit-degrade | 未设=告警开(预授权 warning-once) | 发射器每次 inline | 否(dedup 语义下变更无意义) |
| `SGLANG_KT_V4_TRITON_STRICT_OPTOUT` | explicit-degrade | 未设=legacy | **注册期锁存为 method 属性**(critic 修订) | 否 |
| `SGLANG_KT_SKIP_FORWARD_TASK_ABI_CHECK` | explicit-degrade | 未设=查 | import 期 | 否 |
| `SGLANG_KT_PREFILL_STATS` / `_PATH` | observability | 关 / 空 | 每次 apply inline / flush 时 | 是(逐位等价论证见该计划) |
| `SGLANG_KT_SLOT_OWNERSHIP_ASSERT` | ownership | **2=warn-only** | manager `__init__` 首次使用即缓存 int | 否(改值须重启) |
| `SGLANG_KT_PARITY_DUMP` | ownership 对拍钩子 | 空=off | apply 入口 inline | 是 |
| `SGLANG_KT_BENCHBW_*`(9 个) | benchbw 探针 | 见该计划 | main() 运行期 | 一次性脚本 |

> 两处条件性红线(critic Q2，需主控裁决):explicit-degrade 默认**开**告警、ownership 默认 **2=warn**，均偏离"默认关逐位等价"字面 —— 视 warning-once 为特性规格预授权，还是改为默认压制，请主控一次性拍板并写入上表。
