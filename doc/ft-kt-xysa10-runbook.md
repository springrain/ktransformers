# xysa10 实机执行规程:benchbw 判官 + reserve 三档(同一维护窗口)

> 日期:2026-09-17。承接两处登记:[ft-kt-phase1-plan-benchbw.md](ft-kt-phase1-plan-benchbw.md) §5 rollout
> (判官本体已落盘 [bench_bw_kt.py](../kt-kernel/bench/bench_bw_kt.py),规程收口在本文件);
> [ft-kt-phase3-plan-decode-side.md](ft-kt-phase3-plan-decode-side.md) §9-5(reserve 0/1/2,载体
> [bench_reserve_bw_kt.py](../kt-kernel/bench/bench_reserve_bw_kt.py) 2026-09-17 随本规程新增)。
> **memop 冻结**:用户裁定本序列不含 memop 三岔的任何一条,梯队 3 维持 A+C 落盘态。
> 纪律:两脚本对生产文件零触碰、不 init torch.distributed、与服务器互斥;全程 merge-no-commit。

## 0. 目标

| 序 | 任务 | 回答什么 | 挂了谁 |
|---|---|---|---|
| 1 | benchbw G1/G2 全矩阵 | 11 vs ~22GB/s 缺口归 host 供给 / PCIe 共享 / 控制面 | 梯队 2 fence×window 全部 gain 数字(标 TBD)的兑现通道 |
| 2 | reserve 0/1/2 三档 sweep | 预留低频核对池 GEMV 带宽/抖动有无实测增益 | 梯队 3 唯一候选性能项;"无信号 → 默认 0 永驻" |

两个任务共用同一停服窗口,benchbw 先行(它也是 reserve 的维稳前置——先确认 rig 在带内,后跑档对照才不会把 rig 病读成档位差)。

## 1. 硬前置(在窗口开始前)

1. **kt-kernel 重编译**:extend `.so` 必须带上 tier-2/tier-3 的 C++ 落盘(task_queue 心跳/毒化、worker_pool reserve、ext_bindings 新面)。aarch64 GCC 两阶段名字查找自查(tier-0 CRTP `this->` 教训,phase-3 §9-1)。
2. 重编译后绑定自检(任一不过即停手,先修构建):
   ```bash
   python - <<'EOF'
   from kt_kernel import kt_kernel_ext as e
   c = e.WorkerPoolConfig()
   assert hasattr(c, "reserve_cores_per_numa"), "tier-3 reserve field missing"
   assert hasattr(e.CPUInfer, "watchdog_tripped"), "tier-3 watchdog probe missing"
   print("bind check OK")
   EOF
   ```
3. benchbw 不需要 `kt_kernel_ext`(torch-only),但 reserve sweep 需要;两任务同一窗口,故统一在此门后。
4. 确认服务器进程已停(两脚本自带护栏:`SGLANG_KT_BENCHBW_FORCE` 不是常规逃生阀,误并发等于废掉整窗数据)。

## 2. 执行序列

```bash
mkdir -p doc/bench/$(date +%Y%m%d)
cd <repo root>

# 步 1:benchbw G1 —— 全量首跑,出口 = solo 校准在 LnkSta 相对带内 + 字节账全绿
python kt-kernel/bench/bench_bw_kt.py
# 校准出带(verdict=c / exit 5)⇒ 停手修 rig,本窗不产判读。

# 步 2:benchbw G2 —— 争用腿一次(GEMV∥DMA 体制差值进 verdict 支撑字段)
SGLANG_KT_BENCHBW_CONTENDED=1 python kt-kernel/bench/bench_bw_kt.py

# 步 3:reserve 三档 sweep(master 自 spawn 0/1/2 三子进程,各出一份 tier JSON + 一份对照 JSON)
python kt-kernel/bench/bench_reserve_bw_kt.py
# 旧 .so 上步 3 会以 exit 3 拒绝("lacks reserve_cores_per_numa")⇒ 回 §1-1 修构建。

# 步 4:存档(两脚本 SGLANG_KT_*_OUT 系 env 可把 JSON 直落到该目录)
mv bench_bw_kt_*.json bench_reserve_bw_kt_*.json doc/bench/$(date +%Y%m%d)/
```

## 3. verdict → 梯队决策翻译表(benchbw JSON 的 verdict.attribution 字段)

| verdict | 含义 | 梯队决策 |
|---|---|---|
| **(b)** dual_sum/solo ≥1.7,控制面归属 | 每专家 2 次设备共识 / barrier / all_reduce+.item() 吃掉带宽 | **梯队 2 兑现通道打开**:生产离线确定性对拍(8 prompt 固定种子)+ transfer_stream busy 画像门(>50% ⇒ P1)按 E_chunk 64→32→16 阶梯落 tok/s 数字;memop 冻结维持 |
| **(a)** 比值 ≤1.15,上游共享;P1 写腿 ≈11/卡 | host 供给(rank0 跨 NUMA 写)是天花板 | 梯队 2 收益上限**锁死**,围栏口径改"不做大改";评估转梯队 4(供给/放置治理) |
| **(a)+写腿 ≫** | 供给富余,归 PCIe 链路/RC 共享 | 带宽类收益上限锁死,控制面/CPU 侧特性(memop 类)重新估价——届时才复活三岔 |
| **(c)** solo 出校准带 | rig-invalid(IOMMU/NUMA 误标/分配路径病) | 修 rig,不产任何梯队判读 |
| **(d)** P3 膝点 ≈22 | CPU/DRAM 两侧同顶 | 热更新窗口形状特性优先;reserve 档增益期望下调 |

## 4. reserve 三档读数规则

对照 JSON(`bench_reserve_bw_kt_compare_*.json`)已内置判读行:

- **信号条件**:某 qlen 行 `p50_delta_pct_vs_tier0 > 2 × noise_floor_cv_pct_max` ⇒ 该形态下 reserve 有实测增益;否则一律"无信号",**默认 0 永驻,不推荐开**。
- 各 qlen 形态分别判:qlen=8 代表 decode 微批量、256/2048 代表 prefill 满载;信号可只出现在部分形态,部署推荐值按生产主导形态取。
- 有信号时:`--kt-cpuinfer-reserve-cores` 推荐档回填到 phase-3 §6 裁定记录(作为新裁定,不改已落代码——参数面既有)。
- 对拍纪律:三档共享同一 env 几何(脚本对 qlen 行错位会 exit 3 拒绝合并);`SGLANG_KT_RESERVEBW_QUANT=int8` 仅在 int4 kernel 构建缺失时作退路,**两档 quant 不可混对照**。

## 5. 收尾回写

窗口结束后把结论落回文档(不 commit,随下一次 merge 上车):
1. tiers 总表梯队 2 行:verdict 归属 + fence×window gain 落数或"上限锁死"字样;
2. tiers 总表梯队 3 行:reserve 信号结论(推荐档 / 默认 0 永驻);
3. phase-3 §9-1/§9-4/§9-5 逐项核销(重编译自查、watchdog 注死演练可选搭车、reserve 三档)。

## 6. 本规程不含(memop 冻结条款)

replay-daemon / eager-only / 整支废弃三岔**不在本窗口**;不跑 memop 探针、不写 doorbell、不动 `cudaLaunchHostFunc` 调用面。若步 2  verdict 落在 "(a)+写腿≫" 行,才有复活三岔的议程(届时另立裁定)。
