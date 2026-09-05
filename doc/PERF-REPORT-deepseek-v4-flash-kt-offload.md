# DeepSeek-V4-Flash + KT CPU 卸载推理性能诊断报告

**日期：2026-09-04**
**结论：瓶颈 = GPU 每步时间轴（~17ms 物理成分）+ 逐层 CPU 专家同步依赖（deferred 可完全隐藏，但因质量策略否决；无损路线见第 7 节）**
**状态：诊断全部闭环（36 t/s 无损基线）；deferred 路径实测可达 58 t/s（+61%）但属数值近似，仅作诊断证据保留**

---

## 1. 测试环境

### 1.1 硬件

| 部件 | 规格 | 实测确认 |
|---|---|---|
| CPU | 2× Intel Xeon Gold 6530（Emerald Rapids） | 32 物理核 / 64 线程 × 2 = **64C/128T**；`amx_bf16 / amx_int8 / avx512_vnni` 指令集确认存在 |
| NUMA | 2 节点 | node0: CPU 0-31, 64-95；node1: CPU 32-63, 96-127 |
| 内存 | DDR5-5600 ×8 通道 ×2 路 | 理论 ~716 GB/s，实测可达估 ~500-600 GB/s（**未实测**，mlc 未跑） |
| GPU | **4×** NVIDIA RTX PRO 6000 Blackwell Server（SM120，96GB） | 共 384GB；PCIe `gen.max=5, width.max=16`；空闲降速 Gen1 属正常电源管理（负载下实测 >4GB/s 证明升速成功） |
| GPU 拓扑 | 两两 `NODE`，同 NUMA 1 | 无 NVLink；P2P 理论上可行，无跨 socket（SYS）死刑 |
| GPU-CPU 亲和 | 4 卡全挂 NUMA 1（CPU 32-63, 96-127） | `nvidia-smi topo -m` 确认 |

### 1.2 软件 / 服务配置

- 模型：DeepSeek-V4-Flash（MXFP4 打包路由专家 + FP8 其余，`kv_cache_dtype=fp8_e4m3`，attention backend `dsv4`，sampling backend `flashinfer`）
- 推理栈：ktransformers fork 的 sglang（TP 进程内嵌 kt-kernel CPU 线程池）
- decode CUDA Graph **已启用**（`backend=full`，bs=1..256 捕获成功，耗时 151.93s，占显存 2.38GB）；prefill 图禁用（capture-pool 内存压力规则自动关闭）
- 基线参数：`--kt-method MXFP4 --kt-cpuinfer 64 --kt-threadpool-count 2 --kt-num-gpu-experts 16 --tensor-parallel-size 2 --mem-fraction-static 0.85 --moe-runner-backend flashinfer_mxfp4`
- 每卡 GPU 权重仅 **10.71GB**（路由专家大头在 CPU，GPU 仅有 dense + attention + 16 个常驻专家）→ TP=1 显存余量充足
- 测试负载：固定中文长 prompt，`temperature=0, max_new_tokens=256, ignore_eos`，bs=1，读服务日志 `gen throughput`

## 2. 诊断工具链

| 工具 | 用途 |
|---|---|
| `nvidia-smi dmon -s put` / `--query-gpu` | GPU 占用率、功耗、PCIe 吞吐、链路协商速率 |
| `perf stat -e cycles,instructions,cycle_activity.stalls_total` | CPU IPC 与内存停顿周期占比 |
| 线程缩放（32/64 重启对比） | CPU 算力 vs 内存带宽的因果实验 |
| `SGLANG_KT_HYBRID_TIMING=1 --log-level debug` | MoE wrapper 段内计时；⚠️ `SGLANG_KT_HYBRID_TIMING_DEEP=1` 插 `torch.cuda.synchronize()`，与 CUDA Graph 捕获**互斥**，不可用 |
| sglang 内置 torch profiler（`/start_profile`，30 步） | GPU kernel 级归因；⚠️ TP=1 + log-level debug 组合下观测到挂死，恢复：`POST /stop_profile` 或重启 |

## 3. 排除法证据链

### 3.1 GPU 算力 —— ❌ 排除

decode 期间 `dmon`：功耗 131-142W（满血 600W 的 ~1/4）、显存控制器利用率 13%。GPU1 的 `sm=100%` 后经 profile 证实是**自旋等待**而非计算（见 3.5）。

### 3.2 PCIe —— ❌ 排除

负载下 `rxpci` 最高 8.7 GB/s，离 Gen5 x16 满速（50+ GB/s）差 6 倍；且 8.4 GB/s 的观测值在物理上证明链路已升速（Gen1 上限 4 GB/s）。

### 3.3 CPU 纯算力 —— ❌ 排除

线程缩放（deferred=0）：`--kt-cpuinfer` 64→32，tok/s 36→26（仅 -28%，远非线性减半）。IPC 恒 ~1.53-1.55 不随线程数变化。

### 3.4 内存带宽打满 —— ❌ 排除

- 按每步流过 ~3.8GB 专家权重计（同模型 config 解剖真值，见 §10.3；旧文"~2-3GB"为估算值），有效带宽仅 ~136 GB/s，离平台上限差约 4 倍
- NUMA 对照：`numactl --interleave=all` vs `--kt-numa-nodes 0 1` → **36 = 36 零差异**，perf 指标一致（IPC 1.53，stall 33.5%）→ 跨 socket UPI 假设排除
- stall 33.5% + IPC 1.53 = 典型"带宽接近饱和但未顶死 + 访存并发度未榨出"的中间态
- **机器特异补充（同日 ARM 平台交叉证据）**：本结论（带宽未成墙）只限于这台 2×8 通道 Xeon 机；同日实测的 AmpereOne ARM 单机（8ch@4400，STREAM 墙 200GB/s）上，**内存带宽恰恰是唯一生效的硬件瓶颈**，见 §10。带宽是否成墙由"通道供给 ÷ 每 token CPU 字节"决定，不是 KT 栈的固有属性

### 3.5 NCCL/通信 —— ⚠️ 部分成立，曾误诊

- Profile #1（`--disable-custom-all-reduce` 生效时）：`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` 882ms，占 GPU busy 的 **73%**（每步 ~16ms ≈ 122 次/步 × ~130µs/次）
- 删除 `--disable-custom-all-reduce` 后 → tok/s **纹丝不动**（36）
- Profile #2：NCCL 降至 0.6ms，但 `sglang::cross_device_reduce_1stage` 顶上 **1069.7ms**
- **TP=1 彻底删除全部 reduce kernel → 34 t/s，依然没变**

**结论修正**：这段"通信时间"不是传输成本、也不是 rank 间互等，而是 **TP=2 的 reduce 会合点恰好是每层"等 CPU 专家结果"的汇聚点**——等待在 profiler 里挂名在通信 kernel 头上。TP=1 删掉了 kernel 的名字，等待仍以图内气泡的形式存在。

### 3.6 最终锁定的瓶颈结构 ✅

```
步进 = GPU 每步时间轴 (~17ms: 61 层 × ~280µs 小 kernel 串行接力 + 图内事件同步开销)
     + 露出的 CPU 专家同步依赖 (~10.7ms × (1 − deferred 豁免率))
```

模型对全部实测点闭合：

| 配置 | 实测步进 | 模型分解 | 吻合 |
|---|---|---|---|
| TP=2, deferred=0 | 27.8ms (36 t/s) | 17.2 + 10.6 | ✅ |
| TP=2, deferred=8 | 17.2ms (58 t/s) | 17.2 + ~0 | ✅ |
| TP=1, deferred=0 | 29.4ms (34 t/s) | 18.8 + 10.6（单卡 GPU 段略慢 1.6ms） | ✅ |
| cpuinfer 64→32（d=0） | 27.8→38.5ms | CPU 段 10.7→21.4 | ✅ |

## 4. 优化措施与收益

| 措施 | 机制 | 收益 | 评价 |
|---|---|---|---|
| `--kt-max-deferred-experts-per-token 2/4/8` | 每层仅死等路由权重最高的 top_k−N 个专家，权重最低的 N 个延一层合并进残差 | **36→43→48→58 t/s（+61%）** | ⚠️ **数值近似手段（低权重专家晚一层进残差），因质量策略已否决，仅作上限参考** |
| `--kt-num-gpu-experts 16→32` | 缩小 CPU 专家计算段 | deferred=0: +6%（36→38）；deferred≥2: **+0%** | CPU 算不在关键路径；建议退回 16，显存留给 KV |
| 删 `--disable-custom-all-reduce` | 换快通信库 | +0% | 时间不是通信成本；`--disable-custom-all-reduce` 的原始出处未确认（疑教程默认值），建议不再加回 |
| TP=1 | 消灭 122 会合点/步 | 34 t/s ≈ 36，**不提速但白省 3 张卡** | 部署形态优化，多副本/释放资源 |
| `--kt-cpuinfer 96→64` | 贴物理核数，去超线程浪费 | 性能持平，核数省 1/3 | 清理项 |
| `--kt-numa-nodes 0 1` | 线程池绑 NUMA | 零收益（interleave 假设被否） | 可写可不写 |

## 5. 关键证据摘录

- **kt-time 插桩**（捕获/prefill 期，eager）：MoE wrapper 每层 total ~1.0ms（submit 0.25 / mask 0.18 / gpu 0.42 / sync 0.11 / merge 0.03 / cpu_wait 0.07）。**重要架构事实：decode 图回放期间 Python 层 `apply()` 不执行**（无任何 num_tokens=1 计时报文）——证实 kt 的 CPU 专家提交是以 **host 函数节点录进 CUDA 图**内的架构（`submit_with_cuda_stream`）
- **Profile #2 Top kernel**：`cross_device_reduce_1stage 1069ms`、`cutlass gemm 92ms`、`gemvx 66+10ms`、`sparse_mla_decode_dsv4 18ms`、`_page_split 21ms`、`mhc_pre_big_fuse 11.7ms` —— attention/GEMM 真实计算占比很小，GPU 段主体是"小 kernel 接力 + 会合等待"
- **dmon（TP=2 decode）**：GPU0 sm=40% / GPU1 sm=100%（自旋等待伪装）/ GPU2、3 闲置 / rxpci 最高 8.7GB/s / 单卡功耗 131-142W
- **perf（64 线程）**：IPC 1.53，stalls 33.5%；deferred 8 时 IPC 降至 1.43（CPU 被同步抽打减少，良性）

## 6. 待办与未验证项（诚实清单）

| 事项 | 状态 |
|---|---|
| **质量策略约束** | **用户明确要求数值无损，deferred（近似手段）已否决，不参与生产配置**；deferred 数据仅作为"藏着多少可重叠等待"的诊断价值保留 |
| 投机解码复测（EAGLE/DSPARK） | **最高优先，数学无损**（草稿提议 + 目标模型 rejection sampling 验证，输出分布与原模型严格相同）。此前 35→8 t/s 的负收益结论得自旧瓶颈结构；新结构下 verify m=4 摊薄 GPU 时间轴，经济学可能已反转 |
| `--kt-num-gpu-experts` 逐级上调（48/64） | **无损**。deferred=0 时 CPU 段在关键路径上，16→32 已实测 +6%（36→38） |
| AMX 内核变体确认 | **无损，潜在翻倍 CPU 段**。需确认实际加载的是 AMX 而非 AVX2 变体（`pip show kt-kernel`、启动日志、wheel 内 .so 命名） |
| TP=1 + deferred=8（预测 53-58 t/s） | 不再追求（deferred 否决），数据仅作模型验证 |
| dmon 气泡验证（TP=1 deferred=0 时 sm% 应降至 50-65%） | 未做，零成本，用于区分 GPU 忙碌 vs 空转 |
| mlc 物理带宽标定 | 未跑，~500-600 GB/s 为平台推算值 |
| torch profiler TP=1 挂死 | 已知问题，替代方案 dmon |

## 7. 建议最终配置（数值无损基线 36 t/s）

```bash
# 质量无损生产配置 (TP=1, 34-36 t/s, 释放 3 张卡):
CUDA_VISIBLE_DEVICES=0 sglang serve \
  --host 0.0.0.0 --trust-remote-code \
  --model <DeepSeek-V4-Flash 路径> --kt-weight-path <同左> \
  --mem-fraction-static 0.85 --chunked-prefill-size 4096 --enable-mixed-chunk \
  --kt-method MXFP4 --kt-cpuinfer 64 --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --tensor-parallel-size 1 --moe-runner-backend flashinfer_mxfp4 \
  --port 3000
```

**无损提速路线（按优先级）**：

1. **EAGLE/DSPARK 投机解码复测**——数学无损（草稿提议 + rejection sampling 验证，输出分布不变），在 TP=1 基线上加回原投机参数即可对比。旧结构下 35→8 的结论可能已失效
2. **`--kt-num-gpu-experts` 上调**（32→48→64）：deferred=0 时 CPU 段在关键路径，16→32 已实测 +6%
3. **确认 AMX 内核变体生效**：若加载的是 AVX2 变体，换 AMX 构建可直接缩短 CPU 专家段
4. **多用户场景**：TP=1 单副本 + 剩余 3 卡各起 1 副本，聚合吞吐 ~4×36 ≈ 144 t/s，摆脱全部跨卡协同问题

## 8. 附：本轮顺带修复（未提交，工作区改动）

- `third_party/sglang/python/sglang/srt/arg_groups/overrides.py`：`_deepseek_v4_sm120_moe` 中，SM120 上将 unset/`auto` 的 speculative MoE runner backend 钉到目标 backend —— 修复 EAGLE 草稿 CUDA graph 捕获时的 `Hidden size mismatch`（draft 的 MXFP4 打包专家落入 fp4 无感知的 triton 路径所致）。

## 9. 总账

**诊断结论（零内核改动，全部实测）**：瓶颈从最初猜测的"内存带宽"经七轮排除修正为"GPU 每步时间轴（61 层小 kernel 接力，~17ms）+ 逐层 CPU 专家同步依赖（~10.7ms）"；TP 被证明自始至终无关紧要（TP=1 = TP=2，且白省 3 张卡）；deferred 被证明能完全隐藏 CPU 依赖（36→58 t/s）但因属**数值近似手段**按质量策略弃用。

**无损基线：36 t/s（TP=2）/ 34 t/s（TP=1）。无损提速的三条开放通道**：
1. **EAGLE/DSPARK 投机解码**（数学无损）——旧结构下 35→8 的负收益结论需在新结构下复测，潜在 +30~70%
2. **`--kt-num-gpu-experts` 上调**（已实测 16→32 给 +6%）——缩短关键路径上的 CPU 段
3. **AMX 内核变体确认**——若当前为 AVX2 变体，换构建可直接压缩 CPU 专家段

**需工程投入的剩余天花板（17ms GPU 时间轴）**：并发摊薄（bs↑，零成本）或 GPU 内核级融合（周级工作量）。

---

## 10. 附：第二现场的反向案例 —— AmpereOne ARM 单机上"内存带宽就是唯一的墙"（同日）

> **2026-09-05 修订沿革**（仅此处留档，正文均为定稿）：单专家字节 23MB → **15.73MB**（改用官方 config 解剖；旧值系"时间×假设满墙带宽"循环反推）；每 token CPU 字节 5.2GB → **3.6-3.8GB**；DRAM 墙统一为 **STREAM 200GB/s**（自写纯读 238.7 降为存疑旁注）；内核有效带宽 ≈**170GB/s = STREAM 的 85%**。连带作废：top_k=8 的表述（实为 6）、"0.8×墙"判据、"Q8 同放置只能 17-18 t/s"、"LLAMAFILE 快=路由运气"、内核余量 +25%（实 ~+5-10%）、TP=1 GPU 段"~10ms"的积木预估（2026-09-05 实测 15.1ms，见 §10.4 解剖三）、TP=1 HBM 流量估 6.55GB（nsys 实锤 **8.1GB**：共享专家/lm_head 实为 BF16 非 FP8）。

### 10.1 环境对照

| 部件 | 本报告主机（x86） | ARM 对照机 |
|---|---|---|
| CPU | 2× Xeon Gold 6530（128 线程） | **单路 AmpereOne 192 核**（无 SMT） |
| 内存 | DDR5-5600 ×2 路 16 通道，名义 ~716GB/s，估实 ~500-600 | DDR5 **8 通道 × 2DPC × 64GB @4400**（16 槽全插满，0 空槽；单 NUMA node），**名义峰值 281.6GB/s** |
| GPU | 4× PRO 6000 | 2× PRO 6000，TP=2 |
| 启动参数 | 基线 §1.2 | `numactl --interleave=all sglang serve ... --kt-method MXFP4 --kt-cpuinfer 168 --kt-threadpool-count 1 --kt-num-gpu-experts 16/32 --tensor-parallel-size 2`（无投机解码） |

### 10.2 关键实测数字

| 项目 | 数字 | 来源 |
|---|---|---|
| DRAM 墙（统一口径） | **200 GB/s**（经典 STREAM，168 线程）。名义峰值 281.6（4400MT/s×8ch×8B）；sysbench memory 不可用（buffer 复用，测的是 L2 假象） | 实测 |
| 内核有效带宽 | **≈170 GB/s = STREAM 的 85%**。两种 GPU 名额互证：3.79GB÷22.75ms=167（GPU=16）、3.59GB÷20.7ms=173（GPU=32） | 字节（config 解剖）÷ 相位插桩时间（`KT_MOE_PHASE_TIMING=1`） |
| 每 token CPU 字节 | **3.79GB（GPU=16）/ 3.59GB（GPU=32）**；兑换率 1GB ≈ 5.9ms | 43 层 × avg 5.6/5.3 CPU 专家（插桩）× 15.73MB（config） |
| token 时间结构 | 33.4ms = CPU 段 20.7ms（62%）+ GPU 段 12.7ms（38%）；DRAM 占空比 62% | 插桩 + 残余法 |
| 实测吞吐 | v1 kernel 27 → v2 int8-sdot **28.29**（+4.7%）→ 融合两屏障 +0.3% → GPU=32 **29.92**；LLAMAFILE（UD-Q8_K_XL GGUF）= **30.2-30.5** | 服务日志 gen throughput |
| 带宽敏感系数（判据实验） | GPU 专家 16→32：字节 −5.4% → 总速 **+5.5%**；指令砍半仅 +4.7% | 带宽受限的两条指纹 |

### 10.3 字节账与有效带宽（方法论 + 闭环）

**单专家字节（config 解剖，精确真值）**

官方 config.json（deepseek-ai/DeepSeek-V4-Flash，2026-09-05 经代理核对；部署即官方权重）：hidden 4096、moe_inter 2048、256 路由专家、top_k 6、43 层全 MoE、共享专家 1。

```
单专家参数 = 3 × 4096 × 2048 = 25,165,824（2517 万）
MXFP4 字节 = 25.17M × 0.625B（4bit 0.5B + 每 32 值 4B scale 0.125B）= 15.73 MB
锚点：路由专家总参数 43×256×25.17M ≈ 277B，全模型 ≈ 284B
```

**字节 ↔ 有效带宽（两种 GPU 名额互证，非循环）**

```
字节/token = 43 × 5.6 × 15.73MB = 3.79GB（GPU=16）   有效带宽 = 3.79GB ÷ 22.75ms = 167 GB/s
           = 43 × 5.3 × 15.73MB = 3.59GB（GPU=32）             = 3.59GB ÷ 20.7ms  = 173 GB/s
```

三因子各自独立直读、无反推：层数 43（config）、CPU 专家/层（插桩 `avg_experts`）、单专家 15.73MB（config 解剖）。校验：gate+up : down 的相位时间比 340.5:178.7 = 1.91 ≈ 字节比 2:1。

**闭环检验（有效带宽取 171GB/s 反演吞吐）**

| 配置 | CPU 字节/token | CPU 段 | +GPU 段 | 预测 t/s | 实测 |
|---|---|---|---|---|---|
| GPU=16（TP=2） | 3.79GB | 22.2ms | +12.7 = 34.9ms | 28.7 | 28.2-28.4 ✓ |
| GPU=32（TP=2） | 3.59GB | 21.0ms | +12.7 = 33.7ms | 29.7 | 29.92 ✓ |
| GPU=0，TP=1（2026-09-05 登记） | 4.06GB | 23.4-23.8 | 实测 CPU 段 24.2ms（562µs×43，有效 168GB/s，族内 ✓）→ GPU 段残余 15.1ms → 39.3ms | 27.4-27.7（前提 TP=2 GPU 段；TP=1 实落 25.5） | **25.46** |

> **TP=1 税负实测（2026-09-05）**：CPU 侧预登记全中（avg_experts=6.0 精确、gateup:down=1.98≈2、有效带宽 168GB/s 族内）；**TP=1 的 GPU 段 15.1ms > TP=2 的 12.7ms（+2.4ms）**，与 x86 同形态（17.2→18.8，+1.6ms）——"TP=1 更快"是错觉，删会合省下的 < 独扛全部流量与 kernel 的新增额。结论：单实例最优维持 **TP=2 + GPU=32（29.92）**；TP=1 的价值在双副本聚合（2×25.46 ≈ 50.9 t/s）。GPU 段全解剖（nsys 实锤）见 §10.4 解剖三。待测自由度：TP=2+GPU=0（预测 27.4-27.7）、TP=1+GPU=32（预测 ~28）。

**通用公式（任意 kt 部署的先验估算）**

```
token 时间 ≈ (CPU专家/层 × 层数 × 单专家字节) ÷ 内核有效带宽 + GPU 段
单专家字节 = 3 × hidden × moe_inter × 量化每参数字节   ← 一律解剖求值，禁止"时间×墙"反推
内核有效带宽 ≈ η × STREAM 墙（MXFP4 v2 实测 η≈0.85，须按内核各自标定；勿用名义峰值）
```

适用范围：只计 CPU 权重流；activation/KV、共享专家、dense、attention 在 GPU/HBM 域，单独计账。

### 10.4 每个 token 的时间组成（CPU/GPU 双侧全解剖）

**解剖一：相位 × 读/算分拆（GPU=16 逐项插桩；GPU=32 仅 total：482.1µs×43 = 20.7ms）**

| 相位 | 每层（GPU=16） | 读字节 | 其中纯读时间（÷200GB/s） | 算术+残余 |
|---|---|---|---|---|
| prep | 9.9µs | KB 级 | — | ~10µs |
| gateup（2/3 字节） | 340.5µs | 58.7MB | 294µs | ~47µs（14%） |
| down（1/3，含加权合并） | 178.7µs | 29.4MB | 147µs | ~32µs（18%） |
| **CPU 合计/token** | **529.1µs×43 = 22.75ms** | 3.79GB | 19.0ms | ~3.8ms |
| GPU 段（attention+dense+路由+同步，残余法） | | | | 12.7ms |
| **token 合计** | | | | **35.5ms → 28.3 t/s ✓** |

**解剖二：CPU 段跨配置三分解（插桩直读，按 STREAM 200 拆）**

| 配置 | CPU 段 | 纯读（字节÷200） | 算术 | 残余 |
|---|---|---|---|---|
| GPU=16，TP=2，5.6 专家/层 | 22.75ms（529µs×43） | 19.0ms | 算+残余合计 ~3.8ms（未细分） | — |
| GPU=32，TP=2，5.3 专家/层 | 20.7ms（482µs×43） | 17.9ms | ~1.7ms（v1→v2 指令砍半 +4.7% 反推，已被 v2 减半过一次） | ~1.1ms（减法余项） |
| GPU=0，TP=1，6.0 专家/层 | 24.17ms（562µs×43） | 20.3ms | ~1.9ms | ~1.9ms（6 专家合并每层必走而上抬） |

读/算在流水线中交织，插桩无法直读分离；分离手段只有两个换工况实验：①指令砍半差分（已做，即上表算术列的依据）；②缓存驻留（未做，配方：mask 强制同一专家 → 工作集 ~16MB 驻留 L2 → DRAM 读≈0 → 相位时间=算术+残余真值）。内核有效带宽 = STREAM 墙 200 的 **85%**，剩余可压榨空间 ~+5-10% 总速（预取/访存流水实验为唯一候选）。

**解剖三：GPU 段（TP=1 nsys 实测全开箱；TP=2 构成 = 同构推断）**

方法论：GPU 段总量一律残余法实测（±0.3ms）；成分来自 2026-09-05 nsys 1337-token 稳态窗（`--cuda-graph-trace=node`；观测开销可忽略——采集期相位插桩与饱和态同值）；TP=2 构成 = 在 TP=1 实测底上"流量类折半 + 会合余项反推"，总量对账闭合。

| 成分（每 token） | TP=1 + GPU=0（nsys 实测） | TP=2 + GPU=32（推断） |
|---|---|---|
| attention 投影 FP8（236×17.8µs，@1.16TB/s=HBM 65%） | 4.20ms | ~2.1ms（字节折半） |
| 共享专家 BF16（43×47µs，**BF16 实锤**，@1.07TB/s） | 2.02ms | ~1.0ms |
| 小 GEMV 族（q_a/kv_a/索引器，效率仅 0.3-0.6TB/s） | 1.18ms | ~1.2ms（按层不按卡） |
| 稀疏注意力机关（sparse_mla + merge + _page_split） | 1.60ms | ~1.6ms |
| lm_head BF16 GEMV（719µs，1.06GB @1.47TB/s=82%） | 0.72ms | ~0.36ms |
| mhc fuse 双子（170×4.4µs） | 0.76ms | ~0.76ms |
| 量化/填充/逐元素/router/rope/norm 小核群（~470 个） | ~2.26ms | ~2.3ms |
| 命中 GPU 路由专家流量 | 0（GPU=0） | ~0.36ms（0.47GB 双卡 @~1.3TB/s） |
| 图内缝隙（节点调度 + kt 提交/合并事件） | 2.36ms | ~2.4ms |
| TP 跨卡会合 | 0 | ~0.6-1.5ms（= 12.7 总量 − 其余各项，余项反推） |
| **GPU 段合计（残余法实测）** | **15.1ms ✓** | **12.7ms ✓** |

要点：① 效率随 kernel 尺寸递增（lm_head 82% > 大 GEMM 65% > 小 GEMV 30-50%）——"接力税"的真身是小 kernel 跑不满带宽，缝只占 2.4ms；② TP=1 每 token HBM 流量 **8.1GB**（attention 4.87 + 共享 BF16 2.16 + lm_head 1.06 + KV ~0.05），TP=2 每卡 ~4.3GB——**共享专家与 lm_head 实为 BF16（nsys 实锤），此前按 FP8 的 6.55GB 估算作废**，"TP=1 GPU 段 15.1 > 积木预估 ~10ms"的缺口由此归位（+2.4ms 流量 + 页机关/小核低估）；③ 缝仅 2.4ms ⇒ 图融合上限 ~6%，空间在 kernel 本体——sglang 疆域 ROI：**共享专家 BF16→FP8 省 ~0.9ms/token（+2.3%，纯字节差最肥）** ＞ lm_head BF16→FP8 省 ~0.3ms（输出层数值敏感，需质量 A/B）＞ 小 GEMV/页机关合并 2-3ms（工程量大）。

**总账**：三配置 CPU/GPU/token 逐列相加全部闭合（预测 vs 实测表见 §10.3 闭环）。一层时间轴：TP=2/GPU=32 = 777µs/层（GPU 忙 295 + 等 CPU 482）；TP=1/GPU=0 = 913µs/层（GPU 忙 351 + 等 CPU 562）。

**占空比结构**

```
DRAM 忙时占比 = 20.7 ÷ 33.4 = 62%（GPU 段 12.7ms 内总线全闲）
占空比 100% 上限 = 3.59GB ÷ 171GB/s = 21.0ms ≈ 47.6 t/s；实测 29.9 = 47.6 × 0.62 ✓
```

三类动手项的归属：砍字节（§10.8-1，动 3.6GB）、收占空比（§10.8-2，动 62%）、抬墙（§10.8-3，动 STREAM 墙及 η）。GPU 段属 sglang 疆域，默认不动。

### 10.5 MXFP4 29.9 ≈ LLAMAFILE 30 的归因

两后端共用同一套放置代码（uniform = 每层 0-15 号索引盲置 GPU）；top_k=6 下盲放命中期望 = 6×16/256 = 0.375，CPU 专家期望 ≈ 5.63，MXFP4 实测 5.3-5.6 恰为盲放期望。**同 mask、同放置、同一条墙，吞吐相同是必然而非巧合**：LLAMAFILE 用的 UD-Q8_K_XL 为动态混合量化，路由专家实际位宽低于 Q8 名义（否则其相位数据将要求 ~287GB/s 有效带宽、超越物理墙），两边实际读取字节相近（均 ~15-16MB 量级）。未闭环项：该 GGUF 专家张量的真实位宽（查 dtype 可定案）。

推而广之的格式经济学（同放置同墙，CPU 段字节反比、GPU 段恒定稀释总比值）：**MXFP4 0.625B/参 → 28-30 t/s（实测）；FP8 ~1.03B → ~20；MXFP8 ~1.125B → ~19；BF16 2.0B → ~12**。格式（字节数）之间差 2-3×，后端（内核）之间 ≤±5%——MXFP4 v2 与 LLAMAFILE 两套完全不相干的内核收敛到 1% 内即为实证。

### 10.6 一般化判据

先分清四档"带宽"（本机取值）：

| 量 | 定义 | 取值 |
|---|---|---|
| 名义峰值 | 4400 MT/s × 8 通道 × 8B | 281.6 GB/s（任何实测达不到） |
| **STREAM 墙（统一口径）** | 经典 STREAM，168 线程；现实负载的墙 | **200 GB/s** |
| **内核有效带宽** | CPU 字节 ÷ CPU 段实测时间（含算术+残余，§10.4） | **MXFP4 v2 ≈ 170（= STREAM 的 85%）** |
| 段外均值 | CPU 字节 ÷ 全 token 时间 | ≈ 107（含 GPU 段空转，仅供总账） |

**判据 = 灵敏度实验**（不依赖任何带宽标定）：

```
带宽受限 ⟺ 增减 CPU 字节 x% ⇒ CPU 段时间同比例变化、总速反向变化 ≈ x% × CPU 段占比
本机闭合：GPU 专家 16→32，字节 −5.4% ⇒ 总速 +5.5%；佐证：指令砍半仅 +4.7%（时间受字节数控制）
```

- x86 主机：同模型同字节量（≈3.8GB/token），但平台墙 ~500-600GB/s、CPU 依赖在步进中仅暴露 ~38%（§3.6）→ 不成墙，瓶颈在 GPU 时间轴（报告主体）
- ARM 机：CPU 段以该内核有效上限持续，两项实验闭合 → **带宽受限**；优化只剩三类：减字节、提 η、抬墙
- **新部署判定顺序：先按 config 解剖算每 token CPU 字节 → 做灵敏度实验 → 再谈优化方向**

### 10.7 ARM 侧内核改动（有效带宽已满，独立提交）

`kt-kernel/operators/arm/`（neon_mxfp4_gemm.hpp v2 + moe_base.hpp + mxfp4-moe.hpp）：FP4×2 精确映射 int8 LUT + 激活 32 组 absmax 量化 + `vdotq_s32` int32 精确点积；forward_decode 两屏障融合结构（gateup+激活、down+加权合并各一屏障），合并顺序与原实现逐位一致；量化误差有界（组级 ~2.9% / GEMV 级 ~0.2%，20 万随机组模拟）。计算天花板 ~1.2TB/s，实测有效带宽与余量判定见 §10.4 解剖二；规划与预测一律用有效带宽，勿用名义/纯读口径。fp8/bf16/mxfp8 路径无需改动（BF16 天花板 ~5TB/s；FP8/MXFP8 ~1.3-1.5TB/s，仅 1TB/s 平台才值得微调）。

### 10.8 ARM 侧剩余无代码提速路径（按优先级，均与本报告 §7 兼容）

1. **频率放置（主开关，+~80%）**：`--kt-expert-placement-strategy frequency --init-expert-location <logical_count.pt>`（frequency 策略按全局热点挑 928 个名额，CPU 专家 5.3→~1.5 → 字节 3.59→1.02GB，÷171GB/s ≈ 6ms + GPU 12.7ms ≈ 18.7ms → 估 **~53-54 t/s**）。uniform 抽签已实测证实（16 加 32 仅 5.6→5.3）
2. **deferred 专家重叠（诊断价值同上）**：`--kt-max-deferred-experts-per-token 4-8` 预计把占空比 62%→~95% 收至 ~75-78 t/s（上限 = GPU 段 12.7ms）；与本报告 §4/§6 同属数值近似手段，质量策略一视同仁
3. **1DPC 内存重插（可选硬件）**：每通道拔 1 条（1TB→512GB），频率回 5200-6400，带宽 +48~82%；仅当 CPU 常驻 <450GB 时成立
4. **投机解码在本机上已实测否决**（接受率 ~2.2 < 专家并集膨胀 ~1.8-2.5×，每接受 token 摊到字节不降反升）——注意这是对"MoE 路由熵高 + CPU 卸载占比大"形态的一般性警告，与 §6 待办中 x86 机的 EAGLE 复测**互相独立，不通用**
5. 可能的 BIOS 超频项：memory operating speed 4400→4800（+9%，需自测稳定性）

### 10.9 ARM 侧硬件核查结论（排除"服务器有问题"）

dmidecode + numactl + lscpu 全链自查：单插槽（无 P1_ 定位名）、16 槽全插满 0 空槽、Channel0-7 × Dimm0/1、全 DIMM 同配 4400、单 NUMA node 含全部 192 核和 1TB；sysbench "2.6TB/s" 与 "18.5GB/s" 两个离谱数字均为其 buffer 复用模型导致的 L2 假象（正/反方向），与 DRAM 无关。**结论：机器按设计运行，无硬件故障；与"12 通道 DDR5-5200"芯片规格书的差距来自板级只引出 8 通道 + 2DPC 降速，属 SKU 特性**
