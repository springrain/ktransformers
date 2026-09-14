# Phase-1 计划:observability —— SGLANG_KT_PREFILL_STATS 分层计时账本(+churn 搭车)

> 日期:2026-09-13。**派发序 = 3/4**(explicit-degrade 之后;**基线标定排在 F3 落地之后** —— F3 会把装载侧打包量从 232→256 专家/层)。
> 状态:验证后修订完成,待实施。上游依据:`doc/ft-kt-phase1-audit-hidden-p0.md` §12。纪律:默认关=逐位等价、不 commit。

## 0. 目标（摘要）

在 fork 独有文件 `kt_ep_wrapper.py` 内实现 `SGLANG_KT_PREFILL_STATS=1` 纯 host 记账：以 `_Mxfp4LayerwisePrefillManager` 的 epoch/chunk 为账本边界，在既有挂载点外套 `time.perf_counter_ns` bracket，把每 chunk 8.2s 拆成 **host写 / 共识 / H2D(enqueue+stall) / D2D(postprocess) / 热更新 / residual** 分项；churn 复用现成 CPU bool mask 做 new&~old；字节账按几何精确累计并与 41×232×19,161,088B 自校验；JSONL 每 rank 一行 append 落盘。**另加 critic 搭车（§1.4)**:GPU 常驻专家翻转 Hamming churn 直方 —— 接替被 refuted 的 routing-probe 的数据需求。

默认关=逐位等价；不引入任何新 device op/collective/D2H → **运行期可安全开关，双 rank 无需同步翻转**（与流折叠类需对称开关的本质差异，论证见 §2)。

## 1. 设计（validation 修订后最终版）

### 1.1 账本
`_PrefillChunkStats`(fork 文件内 dataclass):epoch(begin 时存证，见下)、t0、t_last、layers{layer_idx:{host_write_ms, consensus_ms, h2d_enqueue_ms, h2d_stall_ms, gpu_expert_d2d_enqueue_ms, postprocess_ms, hotupdate_ms, bytes_h2d, churn_in, churn_out, experts_cpu}}、totals、unaccounted_ms、partial、tp_rank、schema 字段 path("manager" | 预留)。

### 1.2 挂载点（共识锚点枚举已修正为 6+2N)

| 分项 | 挂载点 | 说明 |
|---|---|---|
| host_write（计时） | bracket `:2259-2266` `_submit_host_write` | **仅计时在 rank0 门内**（内含 sync,host 阻塞，非 rank0 记 0) |
| **bytes 累计（移出 rank0 门，verify P1)** | `:2237` per-expert 循环层级，所有 rank 无条件 | 字节账是纯几何（`numel()//2*element_size`,`:2066-2068` 公式 ×4 RAW_NAMES,缓冲 `[2,...]` 见 `:884-887`)；否则 rank1 bytes=0，自校验与 parity③(=182,260,269,056B）在非 rank0 必假阳性 |
| 共识（gloo ×6) | `_commit_tp_runtime_phase` **:2180/:2232/:2322/:2331/:2463/:2485**(verify P2:原枚举漏 `:2331` postprocess 提交后一处） | 每层实际 gloo 共识=6(load 4 + apply 2)，热更新开时；parity⑤ 期望式写死 **(6 + 2×232)×41** |
| 共识（device ×2N) | `:2253-2256`(host-slot reuse)+`:2273-2275`(host write)，每专家 2 次 | sub-key 区分 NCCL-device 与 Gloo-cpu |
| h2d_stall | `:2248-2249` host_slot_free_events.synchronize + `:2187-2192` wait_event 序列 | host 可观测 DMA 排水点；有效 PCIe GB/s = bytes/stall（字段名 explicit 标注 stall-based 口径；open question 1 采纳双字段并记 enqueue window 上界，零成本） |
| h2d_enqueue | `:2279-2292` copy 循环 + `:2310-2313` raw-ready fence 尾部 | |
| **gpu_expert_d2d_enqueue（补，verify P5)** | bracket `:2184-2234` 整段 GPU-expert D2D 入队循环；`:2162-2168` 每专家 mask `.item()` 扫描显式列入 unaccounted 已知构成 | ~1-2ms/层；保持 unaccounted<10% 告警可信度 |
| postprocess(D2D repack) | 整体 bracket `_postprocess_slot :2088-2115` | marlin repack enqueue + 异常 fence |
| 热更新 | bracket `:2473` synchronize + `:2470-2487` update 窗口 | 单独分项 |

### 1.3 chunk 生命周期（三处钉死，verify P3/P4)
- **begin** = `_advance_round` epoch+=1(`:1955`)：把 **epoch/t0 存入 _PrefillChunkStats 对象**(P4:`abort_round:1934` 先 epoch+=1 再清状态，若 flush 时直接读 `manager.epoch`,partial 行会记成下一轮的值——flush 一律用存证字段；abort_round 内 flush 须先于/独立于 `:1934` 的递增读取旧值）;pending 先 flush(partial=True)再重置。
- **finalize** = manager.apply 尾部直接以 **`self.successor_layer_idx(layer_idx) is None`(`:1926-1929` 现成谓词）** 判定末层（P3:`_prefetch_successor` 有两条早退且恒返回 None，返回值无法区分路径，禁止观察其控制流），在 `:2491` 之后、`:2500` 之前 flush —— 该点当层全部设备共识已完成、后续无 collective、双 rank 落盘无先后要求。
- **partial flush** = `abort_round`(:1931)+ 下一个 `_advance_round` 开始处各兜底一次（pending 至多 1 份，内存有界）。
- **fallback 序列化路 v1 不记账**(verify P6 拍板 open question 2：该路从不经过 `_advance_round`/`abort_round`/`_prefetch_successor`,chunk begin/finalize/epoch 属主缺位，会产生孤儿 JSON；原 `:1314/:1323/:1326/:1420` bracket 移出步骤、移入 G2 后按请求级生命周期另立；memory 记录 xysa10 生产恒走 MXFP4 layerwise，支持此取舍）。schema `path` 字段预留。

### 1.4 churn（原设计 + critic 搭车)
- **原 churn**:在 `_update_gpu_experts_from_batch` Step3 内、`:4791` 赋值之前：`churn_in = int((gpu_experts_mask_cpu & ~self.gpu_experts_mask).sum())`、`churn_out = int((self.gpu_experts_mask & ~gpu_experts_mask_cpu).sum())`。新旧 mask 均为 CPU bool(`:3691` 创建、`:4791` 注释明示 "CPU tensor, safe to replace"，初始化 `:3808`)，零新 `.cpu()`/D2H；结果暂挂 method pending dict,manager.apply 收尾并入该层记录。
- **critic 搭车(256-bit 翻转 Hamming)**:同一插入点对 top-24 新旧选择做 256 位 host 布尔 XOR+sum 得 Hamming churn（每层 24 位轨迹）。位置在 `:2473 synchronize` 之后、CUDA Graph 捕获路径之外，rank0-only、零 CUDA op、天然 capture-safe。**该直方即 routing-probe 原数据需求（lru-hit-d2d vs 迟滞之争）的接替来源** —— routing-probe 已 refuted 并移出第一阶段，其 oracle 槽曲线问题转由本指标 + 字节账离线分析回填。

### 1.5 落盘与轮转
JSONL 每 rank 一行 append(schema:ts/host/tp_rank/epoch/path/wall_ms/totals/layers/bytes_h2d/bytes_expected/effective_pcie_gbps/unaccounted_ms/partial/warnings[]);TP world>1 且路径无 `{rank}` 时自动 `.rank{tp_rank}` 后缀；异常降级节流 warning，绝不向热路径抛。**轮转（critic）**:`SGLANG_KT_PREFILL_STATS_PATH` 支持按日文件名模板（`{date}` 展开）+ max-bytes 简单轮转（封顶即换 `.%d` 序号），一行成本，杜绝长跑无界增长。
`_StatsSpan` helper **硬约束（verify)**:with 体必须 try/finally 透传异常 —— 挂载点 `:2249/:2301/:2110` 均在 except 路径有既有语义（fence/poison 状态机 `:2330-2371`),bracket 吞异常会破坏之。

### 1.6 自校验
flush 时 `bytes_h2d == bytes_expected`（运行期几何重算，不硬编码 170GiB，仅注释参考 41×232×19,161,088 = 182,260,269,056B)；不等→warnings 加字段不 raise;unaccounted 构成：`gpu_expert_d2d_enqueue`(已补 bracket)+ `:2162-2168` 扫描（显式声明），阈值 10% 标讯；bracket 次数与 registry 层数（`:1918-1924`）口径核对。

## 2. env 门与运行期翻转安全性

- `SGLANG_KT_PREFILL_STATS`：默认关；`=="1"` 开。读取=每次 `KTEPWrapperMethod.apply`(:4311)/manager.apply(:2417) 入口 inline `os.environ.get`（先例 `:4347/:4552/:4616`)→ 局部布尔传各 bracket。**运行期 setenv 下一 chunk 生效**;OFF 时 41×~5 次 env 查找≈20µs/chunk。
- `SGLANG_KT_PREFILL_STATS_PATH`：默认空=rank0 logger.info 单行摘要；flush 时读（每 chunk 至多一次）。
- **运行期翻转安全论证**:ON 只新增 perf_counter 调用、CPU 张量逐位与/求和、append 写文件；不新增 CUDA kernel/事件/流、collective、D2H → 双 rank 设备侧执行序与 collective 序列 ON/OFF 逐位一致 → rank0 开 rank1 关或任意翻转时点都不会集合通信错位。
- **OFF 逐位等价**：每挂载点只多一次 dict 查找+一次失败 if 分支，helper 直返、不构造对象、不碰张量；verify 复核通过（全部挂载点仅 perf_counter/env/CPU bool 逐位运算/文件 IO;churn 操作数 CPU;`:4792-4794` CUDA 就地 copy_ 序列不变；`:2473/:2485-2487` 仅被 wall 计时；token 门槛挡在 capture 路径外）。
- **回滚措辞修正（verify P8)**:unset 即时生效（下一次 `:4311` 内联读取即关），重启只是最保守形态，**无需重启**。
- 开销账：41 层×232 专家×~6 对 perf_counter≈5.7 万次×~0.1µs≈6ms + 每 chunk json.dumps≪ 8.2s×1%=82ms;aarch64/sm_120/PCIe 无特异性阻碍（perf_counter_ns 单调钟；`:891` pinned 使 `:2249` event.synchronize 确为 DMA 排水点，stall-based 有效带宽定义成立）。

## 3. 触及文件与合并协议

唯一改动文件 `kt_ep_wrapper.py`(fork 独有，零上游分歧）。落地序第三位（explicit-degrade → observability → ownership,critic Q1);`:2331-2333` 为与 ownership 共享的挂载点，由本计划先行 bracket,ownership 复用同一 helper；实施第 1 步按符号/grep 重锚全部行号。

## 4. 对拍

位等硬标准（容差=0/`torch.equal`):
1. **OFF≡现状**：固定 server args+seed,3 prompt×512 token，最终 logits sha256 与改动前 build 相等。
2. **ON≡OFF**：同 batch logits 逐位相等；抽 KT 层 {0,20,40} dump MoE 输出 hidden_states `torch.equal`(bitwise)。
3. **chunk 边界**:layers==41、每层 experts_cpu==232、`bytes_h2d==182,260,269,056B` 精确 —— **双 rank 分别成立**（字节累计已移出 rank0 门，非 rank0 不再是 0）。
4. **churn 手对**:rank0 既有 debug 日志（`:4802-4807` selected_experts）与 JSON churn_in 手对 3 层逐一相等 —— **需先调大 DEBUG 日志级别**(verify P7，测试计划写明）。
5. **共识计数**:JSON 每 chunk consensus 次数 == **(6 + 2×232)×41**（热更新开）,6 处 gloo（含 `:2331`)+ 每专家 2 处 device，实现与对拍同一公式。
负注入：(a) PATH 只读 → 仅一次节流 warning,logits 仍位等；(b) `true`/`2`/`0` → 按 `=="1"` 判关，位等；(c) 长任务中途 setenv 翻转 → 下一 chunk 生效，无 NCCL 超时/partial 崩坏/锁死；(d) rank 不对称（rank0 ON/rank1 OFF)→ collective 序列不受影响，两 rank JSONL 各自正常；(e) abort_round 注入 → partial=true 落一行，epoch 为 begin 存证值（非下一轮值）。

## 5. rollout

- **G0** 合入（默认 OFF):CI 冒烟 + OFF≡现状 logits 哈希 + 评审；全绿。
- **G1** 生产单 prefill 低峰 1h:schema 稳定、`bytes_h2d==bytes_expected` 精确、ON vs 历史 OFF chunk wall 中位差 <1%、无 stats 告警 → 否则 unset 回落。
- **G2** canary 1 台 24h（覆盖热更新 churn 高峰+长 prefill):p99 与共识分项较基线漂移 <1%（计时器无反馈效应）、无新 NCCL 超时、轮转策略生效确认。
- **G3** 全量 prefill workers 常开 7 天 → 转常驻 config（回落=unset，无需重启）。**基线标定排在 F3 落地之后**(F3 将打包量 232→256,chunk wall 基线前移；verify/critic 一致的排期约束）。

## 6. 热更新交互 / collectives / CUDA graph

- 逐条核销（挂载点全在 prefill-only）：计时不改 `:2473` synchronize 时点/参数/次数；写盘点刻意落在末层 :2491 之后；每专家共识只测 wall 不 enqueue;`.item()` D2H 次数不变；churn 只读已物化 CPU mask 不写，`:4798-4799` pinned mask copy 原样；abort_round 兜底先于新轮首个 collective。**结论：无语义交互，唯一共享资源 = Python perf_counter 与 CPU 算术。**
- 新增 collective=0;JSONL 不做跨 rank 聚合（每 rank 独立行 + tp_rank/epoch 离线 join)。
- CUDA graph 零影响：layerwise 仅 num_tokens≥threshold 进入（`:4412-4416`);churn hook 两调用点（`:2474/:4509`）均 prefill；新码零 CUDA 调用天然 capture-safe。

## 7. 工时

约 **3 人日**（实现+schema 1.0；自校验+单测 0.5;ON/OFF 位等对拍+五例负注入 0.75;文档+灰度 0.5;buffer 0.25)。

## 8. 对抗验证记录（verify refuted=False,8 问题全部吸收；critic 追加）

| # | 问题 | 处置 |
|---|---|---|
| P1 | 字节账挂 `:2259` rank0 门内 → rank1=0 假阳性 | §1.2 字节累计移出门外，计时仍在门内 |
| P2 | 共识锚点漏 `:2331`（实际 6 处） | §1.2 枚举修正 (6+2×232)×41 |
| P3 | 末层判定误用 `_prefetch_successor` 返回值 | §1.3 钉死 `successor_layer_idx is None` |
| P4 | partial flush epoch off-by-one | §1.3 begin 存证，flush 一律用对象字段 |
| P5 | gpu_expert_d2d/扫描无 bracket，稀释 unaccounted 承诺 | §1.2 补 bracket + 已知构成声明 |
| P6 | fallback 序列化路账本生命周期缺位 | §1.3 v1 砍掉，G2 后另立 |
| P7 | parity④ 依赖 DEBUG 级日志 | §4.4 写明调级别 |
| P8 | "回落=重启"与运行期生效自相矛盾 | §2 措辞改保守：unset 即时生效 |
| C-1 | routing-probe 数据需求回流 | §1.4 256-bit Hamming churn 搭车 |
| C-2 | 统计文件无轮转 | §1.5 按日+max-bytes 轮转 |
| C-3 | 基线应排在 F3 后 | §5 G3 排期约束 |

## 9. 主控裁定项

1. effective PCIe GB/s 分母口径：stall-based（默认）+ enqueue window 上界双字段并记（已采纳零成本），字段命名确认。
2. per-expert 粒度开关（`SGLANG_KT_PREFILL_STATS_DETAIL=1`,232×41 行/chunk）是否预留（默认 per-layer 聚合，磁盘量小两个数量级）。
3. G3 常驻后是否桥 Prometheus（新分歧面）还是长期 JSONL-only —— 倾向后者。
4. partial chunk 默认不参与带宽周报均值（仅标记），确认。
