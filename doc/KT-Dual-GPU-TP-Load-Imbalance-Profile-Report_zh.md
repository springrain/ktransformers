# KT 双 GPU TP 负载不均 Profile 分析报告

**分析日期：2026-09-23**<br>
**分析对象：KTransformers + SGLang，TP=2，已启用 `--disable-custom-all-reduce`**

> **结论摘要：Decode trace 和最新长 prefill trace 得到相同的结构性结论：两个 rank 的非 NCCL GPU 计算几乎完全相等，TP1 的额外 GPU busy 主要是提前进入 NCCL AllReduce 后等待 TP0。最新 GPU-only prefill trace 实际处理的是 4,096-token forward，而不是 16K；单步约 4.496 秒，对应约 911 input tok/s，与线上约 890 tok/s 接近。该步中 TP1 有 94.9% 的 GPU 时间线被 NCCL 覆盖，13 个固定 collective 平均等待约 317 ms；TP0 同期 99.4% 的时间没有 GPU kernel、memcpy 或 memset，主要受 TP0-only CPU expert/offload 临界路径限制。**

## 1. 问题背景

现场观察到两张 NVIDIA RTX PRO 6000 Blackwell Server Edition 的负载长期不对称：

| 项目 | GPU0 / TP0 | GPU1 / TP1 |
|---|---:|---:|
| 显存占用 | 66,917 MiB | 66,769 MiB |
| `nvidia-smi` GPU-Util | 6% | 100% |
| 功耗 | 95 W / 600 W | 110 W / 600 W |

两卡显存只相差约 148 MiB，与 TP 静态分配基本对称的预期相符，但显存接近本身不能证明执行工作量均衡；真正的计算量证据来自后文“排除 AllReduce 后两 rank 的 kernel 时间只差 0.014%”。异常点是 GPU1 利用率显示 100%，但功耗只有 110 W，明显不像大规模 GEMM、Attention 或 MoE GPU 计算真正跑满。

表中的 6% 是一次 `nvidia-smi` 瞬时采样；trace 对 29 个 decode step 做区间合并后，TP0 平均 GPU busy 约为 52%。两者并不矛盾：采样窗口、请求阶段和 profiler 开销不同都会改变瞬时读数。稳定结论是 TP0 存在大段 GPU 空洞，而 TP1 被长驻 NCCL kernel 填满了这些时间段。

当前启动参数包含：

```text
--disable-custom-all-reduce
```

因此本次 trace 中集合通信使用 NCCL，表现为：

```text
ncclDevKernel_AllReduce_bf16_RING
```

而不是自定义通信路径中的：

```text
sglang::cross_device_reduce_1stage
```

## 2. 输入文件与环境

### 2.1 Trace 文件

#### Decode 与旧短 prefill trace

| Rank | 文件 | 大小 | SHA-256 |
|---|---|---:|---|
| TP0 | `kt-tp-wait-1790092382.1432543-TP-0.trace.json.gz` | 7,819,481 B | `47cbfaae576678a6064ea66cfb72331f6c3ce7b68810d334d16bfffda4553d42` |
| TP1 | `kt-tp-wait-1790092382.1432543-TP-1.trace.json.gz` | 7,906,742 B | `c3b9c27984b4177cb71a1c121d74687ac51f68e80eab660f886de357a620de6b` |

事件数量：

| Rank | Trace events | CUDA kernel events | 唯一 kernel 名称数 |
|---|---:|---:|---:|
| TP0 | 284,301 | 72,563 | 110 |
| TP1 | 294,367 | 71,406 | 110 |

#### 最新长 prefill GPU-only trace

| Rank | 文件 | 大小 | SHA-256 |
|---|---|---:|---|
| TP0 | `kt-prefill-16k-1790136805.4312267-TP-0.trace.json.gz` | 484,691 B | `0b692bb2d7c8124eaaa981fcddb0c0a51ee10b7985311dbd5139a4bae2d7bc37` |
| TP1 | `kt-prefill-16k-1790136805.4312267-TP-1.trace.json.gz` | 458,371 B | `34387dee41b9839e8419a1be64c747b128331cfe32341ab2a273525349075ea8` |

| Rank | Trace events | CUDA kernel events | 唯一 kernel 名称数 |
|---|---:|---:|---:|
| TP0 | 24,745 | 2,176 | 98 |
| TP1 | 23,849 | 2,150 | 98 |

### 2.2 Trace 内记录的运行环境

| 项目 | 值 |
|---|---|
| Host | `xysa1` |
| Distributed backend | NCCL |
| World size | 2 |
| NCCL | 2.29.7 |
| CUDA driver | 13.4 |
| CUDA runtime | 13.0 |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | SM 12.0 |
| SM 数量 | 188 / GPU |
| 显存 | 102,071,664,640 B / GPU |

### 2.3 两次采集覆盖的请求阶段

Decode/旧短 prefill trace 中共有 30 个 forward step：

- 1 个 `EXTEND`：batch size 1，20 个 prefill tokens；
- 29 个 `DECODE`：batch size 1；
- 每个 forward 固定出现 87 次 NCCL AllReduce。

旧 prefill 只有 20 tokens，不能代表生产长 prompt。最新 GPU-only trace 只采集一个长 prefill forward；虽然文件名和测试目标为 16K，但 kernel grid 和 32 MiB activation 搬运共同证明该 forward 的实际 token 数为 **4,096**。这说明本次 scheduler 实际只发出了 4K forward；究竟是 resolved chunk、`max_prefill_tokens`、调度 token budget、mixed/dynamic chunk 还是缓存状态造成，需要通过 `/server_info` 和启动日志核对。

## 3. 分析方法

### 3.1 Kernel 聚合

从两个 gzip Chrome trace 的 `traceEvents` 中筛选：

```text
cat == "kernel" && dur 存在
```

按 kernel 名称统计：

- 调用次数；
- `dur` 累计时间；
- 平均、median 和最大持续时间。

PyTorch Chrome trace 中 `dur` 的单位为微秒。本报告中将其除以 1000 转为毫秒。

### 3.2 AllReduce 配对

两个 rank 每个 step 都有相同数量的 87 次 AllReduce，因此可以按照：

```text
step 序号 + step 内 collective 序号
```

对 TP0 和 TP1 的同一个 collective 进行配对，比较：

- 进入时间；
- kernel 持续时间；
- 结束时间；
- 等待窗口内另一张 GPU 是否有 kernel 或 memcpy。

如果一个 rank 提前进入、kernel 持续时间显著更长，但两个 rank 几乎同时结束，则长出来的时间是 rendezvous/barrier 等待，而不是 payload 更大或有效计算更多。

### 3.3 累计时间与墙钟时间的区别

报告同时使用两种统计口径：

1. **Summed kernel time**：所有 kernel 的 `dur` 直接求和；不同 stream 重叠时会重复计时。
2. **Kernel union / GPU busy wall time**：合并同一观察窗口内所有 kernel 时间区间，重叠部分只计算一次。

因此 summed kernel time 适合比较同名 kernel 的工作量，union time 适合解释 `nvidia-smi` 看到的 GPU busy 比例。

### 3.4 绝对性能口径限制

旧 CPU+GPU trace 适合回答“两个 rank 为什么不对称”，不适合直接回答线上 prefill/decode 吞吐和请求延迟。

原因包括：

- 开启了 CPU+GPU Torch Profiler，CPU op、CUDA runtime、kernel 和 annotation 都被逐事件记录；
- 唯一的 prefill 只有 20 tokens，是小 shape、首个被采集的 EXTEND；
- 20-token prefill 很可能低于 `--kt-gpu-prefill-token-threshold`，与生产长 prompt 的 layerwise/full-GPU 路径不是同一执行策略；
- prefill/eager 路径需要逐个 launch 大量算子，而 decode 主要回放 CUDA Graph，profiler 对两者的扰动不同；
- 没有同输入、同配置下 profiler-off/on 的 A/B，无法从单份 trace 反推出 profiler 开销。

本次 EXTEND 内，TP0 记录了约 13,997 个 `cpu_op` 和 6,312 个 `cuda_runtime` 事件，TP1 也有约 13,802 和 6,043 个；单个 decode step 只有约 38 个 `cpu_op`。这说明 prefill 绝对时间尤其容易被 profiler 的逐算子记录放大。

因此本报告遵循以下口径：

1. 跨 rank 的同名 kernel、collective 进入/结束顺序和相对差异，可用于根因归因；
2. Trace 内的毫秒数仅称为“profile 诊断值”或“观察窗口”，不能直接换算线上 tok/s；
3. 线上性能以关闭 profiler 后的 TTFT、ITL、prefill tok/s、decode tok/s 和请求延迟为准；
4. 用户提供的线上基线为 decode 约 54 tok/s、prefill 约 890 input tok/s。

最新 prefill trace 只开启 GPU activity，事件量显著下降；其实际 4,096-token GPU 时间线跨度约 4.496 秒，折算约 911 input tok/s，与线上约 890 tok/s 只差约 2.4%。因此它比旧 20-token trace 更具代表性，可以用于本次 4K 有效 chunk 的 prefill 时间线和瓶颈占比分析，但仍不等同于完整 HTTP TTFT。

## 4. Decode/旧短 prefill trace 关键结果

### 4.1 TP1 多出的 kernel 时间几乎全部是 AllReduce

| 指标 | TP0 | TP1 | TP1 - TP0 |
|---|---:|---:|---:|
| 全部 CUDA kernel 累计时间 | 463.635 ms | 723.863 ms | +260.228 ms |
| NCCL AllReduce 次数 | 2,610 | 2,610 | 0 |
| NCCL AllReduce 累计时间 | 106.773 ms | 366.952 ms | +260.179 ms |
| 排除 AllReduce 后的 kernel 时间 | 356.862 ms | 356.911 ms | +0.049 ms |
| AllReduce 占全部 kernel 时间 | 23.030% | 50.694% | +27.664 pct |

定量归因：

```text
TP1 多出的总 kernel 时间      = 260.228 ms
TP1 多出的 AllReduce 时间      = 260.179 ms
AllReduce 对差异的解释比例     = 99.981%
```

排除 AllReduce 后，两卡 kernel 时间只差 0.049 ms，即 **0.014%**。这基本排除了以下假设：

- TP1 分到了更多 Transformer 层；
- TP1 分到了更多 GPU experts；
- TP1 的 Attention、GEMM 或 MoE GPU 计算量明显更大；
- 权重 TP 切分严重不均。

### 4.2 主要有效计算 kernel 基本一致

| Kernel 类别 | TP0 累计时间 | TP1 累计时间 |
|---|---:|---:|
| CUTLASS SM120 GEMM 主 kernel | 96.738 ms | 97.020 ms |
| BF16 GEMV variant 6 | 32.328 ms | 32.374 ms |
| CUTLASS grouped MoE GEMM #1 | 24.917 ms | 24.997 ms |
| BF16 GEMV variant 7 | 21.182 ms | 21.298 ms |
| FlashInfer sparse MLA decode | 18.455 ms | 18.732 ms |
| BF16/FP32 GEMV | 15.389 ms | 15.449 ms |
| CUTLASS grouped MoE GEMM #2 | 14.766 ms | 14.780 ms |
| MHC pre-fuse | 11.719 ms | 11.846 ms |

这些核心计算项在两个 rank 上高度一致，与“TP 切分均衡、差异来自同步等待”的判断一致。

## 5. Decode 分析

### 5.1 本次 trace 中 decode step 的 GPU 时间线统计

29 个 decode step 对齐后的平均值：

| 本次 profile 内每步指标 | TP0 | TP1 |
|---|---:|---:|
| Step 关联 GPU 事件覆盖窗口 | 约 18.498 ms | 约 18.456 ms |
| 非 NCCL kernel union | 8.714 ms | 8.719 ms |
| NCCL AllReduce union | 0.938 ms | 9.360 ms |
| 任意 GPU kernel busy union | 9.634 ms | 18.063 ms |
| GPU busy 比例 | 约 52.1% | 约 97.9% |
| 窗口内未观测到 kernel/memcpy | 约 8.864 ms | 约 0.393 ms |

最重要的两项是：

```text
非 NCCL kernel union：TP0 8.714 ms，TP1 8.719 ms
差异：约 0.005 ms，约 0.05%
```

而 AllReduce：

```text
TP0：0.938 ms/step
TP1：9.360 ms/step
TP1 额外驻留：8.422 ms/step
```

TP1 相对 TP0 多出的 GPU busy 时间中，约 **99.93%** 来自 AllReduce。

这里的约 18.46 ms 是 profiler 下所有关联 GPU stream 的事件覆盖窗口，不是 HTTP 请求墙钟，也不是可以直接换算线上 tok/s 的严格 ITL。排除首个 prefill→decode 过渡 step 后，两个 rank 的稳态 GPU 活动包络都约为 18.40 ms；用户线上 decode 约 54 tok/s 与其数值接近，但线上性能仍应以 profiler-off 日志为准。

这组数字可以定性解释现场现象：

- TP0 GPU busy 约一半；
- TP1 几乎整个 step 都有 kernel 驻留；
- 但两卡真正用于模型计算的时间完全相同。

### 5.2 固定 13 个 collective 是主要等待点

每个 decode step 有 87 次 AllReduce。其中固定的第（从 1 开始计数）：

```text
63、65、67、...、87
```

共 13 个 collective，在 29 个 decode step 中重复出现，共 377 次。

| 指标 | TP0 同位置 collective | TP1 同位置 collective |
|---|---:|---:|
| 平均持续时间 | 9.12 µs | 665.17 µs |
| TP1 median | - | 647.62 µs |
| TP1 范围 | - | 595.78–1,596.80 µs |

配对后的时序特征：

- TP1 平均提前约 656.73 µs 进入 collective；
- 两个 rank 的 collective 结束时间中位差只有约 0.82 µs；
- 最大结束时间差只有约 1.60 µs。

也就是说：

```text
TP1 先进入 NCCL AllReduce
    -> kernel 持续驻留等待 TP0
    -> TP0 到达后两边几乎同时退出
```

这不是 TP1 的通信 payload 更大。同一个 collective 在两个 rank 上具有相同的 dtype、逻辑 tensor shape 和 launch 配置；长短差异来自 rank 到达时间差。旧 trace 没有为 decode CUDA Graph 内部 kernel 保留可可靠解读的元素数字段，因此这里不从 trace 元数据反推具体 payload 字节数。

### 5.3 等待窗口与 TP0 GPU 空洞高度对应

本次 trace 中上述 377 个等待窗口：

| 指标 | 数值 |
|---|---:|
| 等待窗口合计 | 247.587 ms |
| 本次 profile 内平均每个 decode step | 8.537 ms |
| 窗口内 TP0 无 kernel/memcpy 的时间 | 245.498 ms |
| TP0 每步对应空洞 | 8.466 ms |
| 这些窗口解释的 TP0 idle 比例 | 约 95.5% |

等待窗口内 TP0 约 **99.16%** 的时间没有 GPU kernel 或 memcpy。这说明 TP1 等待的不是 TP0 上的另一段大型 GPU 计算，而是 TP0 GPU 时间线之外的 host/CPU/offload 依赖。

### 5.4 CPU offload 数据流证据

| 操作 | TP0 次数 | TP1 次数 |
|---|---:|---:|
| Pinned HtoD | 406 | 16 |
| Pinned DtoH | 1,200 | 30 |
| DtoD | 104 | 91 |
| `cudaLaunchHostFunc` | 26 | 0 |

更强的对应关系是：

- 377 个 TP1 长等待窗口中，每个窗口都对应一次 TP0 pinned HtoD，共 377 次；
- 同一批窗口内还出现 1,083 次 TP0 pinned DtoH；
- TP1 没有相应规模的 CPU expert 数据交换。

这与 KT 当前“CPU expert 只由 TP rank 0 执行”的实现高度吻合。这里的 TP0 临界路径不仅包含 CPU expert 算术本身，也包含 host callback、任务调度、DtoH/HtoD 和结果合并；Torch trace 没有展开 native CPU worker，不能把全部空洞进一步精确拆成单独的 CPU 算术时间。

## 6. Prefill 分析

### 6.1 文件名为 16K，但实际 forward 是 4,096 tokens

最新 trace 有两条独立证据证明实际 prefill forward 的 token 数是 4,096，而不是 16,384：

1. `sparse_mla_prefill_mg*` kernel 的 grid 为 `[4096, 1, 1]`；
2. TP0 每个 CPU-managed 层都有一次 33,554,432 B activation 搬运，恰好等于：

```text
4096 tokens × hidden_size 4096 × BF16 2 bytes
= 33,554,432 bytes
```

该 forward 的全局 GPU 活动跨度约为 4.497 秒：

```text
4096 / 4.4966 ≈ 910.9 input tok/s
```

与线上约 890 input tok/s 接近，说明这份 GPU-only trace 已捕获到具有生产代表性的有效 4K prefill chunk。

这也说明：测试脚本中的 `CHUNKED_PREFILL_SIZE=16384` 只是期望值，并不会修改正在运行的服务。当前 scheduler 实际只发出了 4K forward。可能来源包括 resolved `chunked_prefill_size`、`max_prefill_tokens`、调度 token budget、并发/mixed chunk、dynamic chunking 或缓存命中；仅凭 GPU trace 不能唯一确定，必须通过 `/server_info` 和启动日志核对。Layerwise 初始化失败或 OOM fallback 可以解释“为什么仍走 hybrid path”，但通常不能单独解释“为什么 forward 被切成 4K”，两者应分开检查。

### 6.2 4K prefill GPU 时间线拆分

| 指标 | TP0 | TP1 |
|---|---:|---:|
| 全局 GPU 活动观察窗口 | 4,496.625 ms | 4,496.625 ms |
| 非 AllReduce kernel union | 219.185 ms | 219.015 ms |
| NCCL AllReduce union | 147.895 ms | 4,265.911 ms |
| Memcpy 累计时间 | 22.254 ms | 0.029 ms |
| 任意 GPU 活动 union | 389.411 ms | 4,485.105 ms |
| GPU busy 比例 | 8.66% | 99.74% |
| 未观测到 GPU kernel/memcpy/memset | 4,107.214 ms | 11.520 ms |

两个 rank 的非 AllReduce GPU 工作只差约 0.170 ms，不到 0.1%，再次证明有效 GPU 计算没有静态失衡。

TP1 的 AllReduce 覆盖了整个 prefill 时间线的 **94.87%**；TP0 AllReduce 只占 **3.29%**。87 次 paired NCCL 中两 rank 同时驻留的区间合计约 140.449 ms，平均约 1.614 ms/次，只占窗口约 3.12%；其余大部分是单边提前到达后的等待。因此现场看到的“TP1 约 100%、TP0 很低”在长 prefill 中同样成立，但 TP1 并不是在做更多 GEMM/Attention，也不是一直在传输数据，而是在 NCCL rendezvous 中等待。

### 6.3 固定 13 个 collective 占据主要等待时间

与 decode 一样，长 prefill 中固定的第 63、65、67、...、87 个 collective 出现极端不对称，共 13 个：

| 指标 | TP0 | TP1 |
|---|---:|---:|
| 13 个 collective 总时间 | 20.314 ms | 4,142.834 ms |
| 平均单次 | 1.563 ms | 318.680 ms |
| TP1 median | - | 315.493 ms |
| TP1 最大值 | - | 337.228 ms |

时序配对显示：

- TP1 平均提前 317.129 ms 进入；
- 提前范围为 310.203–335.677 ms；
- 两 rank 结束时刻的绝对差中位数只有约 6.4 µs；
- 结束时刻最大绝对差约 178 µs。

这 13 个 TP1 长 AllReduce 合计 4.143 秒，占 TP1 AllReduce 时间的约 97.1%，也占整个 4K prefill GPU 时间线的约 92.2%。其本质是 TP1 提前到达后等待 TP0，而不是 13 次数据量突然增大的通信。

87 次 AllReduce 的结构可以进一步定位到层：第 1 次是入口同步，之后每个 transformer block 各有 attention 后和 MLP 后两次同步，即 `1 + 43 × 2 = 87`。长等待恰好出现在第 63、65、...、87 次，因此对应 0-based 的 L30–L42 共 13 个 block 的 MLP 后同步。TP1 在这些层的 AllReduce 驻留时间依次为：

```text
L30 330.379 ms  L31 312.286 ms  L32 314.335 ms  L33 315.493 ms
L34 320.186 ms  L35 314.588 ms  L36 314.308 ms  L37 321.642 ms
L38 314.195 ms  L39 315.585 ms  L40 320.855 ms  L41 337.228 ms
L42 311.755 ms
```

### 6.4 TP0 在等待窗口中主要受 CPU expert/offload 限制

在 TP1 提前等待 TP0 的 4.123 秒窗口内，TP0 的 GPU 活动为：

| TP0 活动 | 时间/数量 |
|---|---:|
| GPU kernel union | 1.056 ms |
| Memcpy union | 22.069 ms |
| 无 GPU kernel/memcpy/memset | 约 4,099.5 ms，99.44% |
| 32 MiB DtoD | 13 次，共 416 MiB |
| 32 MiB DtoH | 13 次，共 416 MiB |
| 32 MiB HtoD | 13 次，共 416 MiB |

单层 32 MiB DtoH/HtoD 实际只需要约 0.8–0.9 ms，13 层总 DMA 约 22 ms，只占等待窗口约 0.5%。因此主要瓶颈不是 PCIe 传输带宽，而是 DtoH 与 HtoD 之间的 TP0-only CPU expert/offload host gap。GPU-only trace 无法继续把这约 4.1 秒拆成 CPU 算术、DRAM 访存和队列时间，但代码路径和 activation 往返可以确认它属于 CPU expert/offload 临界路径。

以第一次 32 MiB DtoH 为边界，时间线进一步分成：

| 阶段 | 墙钟 | TP0 GPU busy | TP0 GPU idle | Busy 比例 |
|---|---:|---:|---:|---:|
| 第一次 DtoH 之前 | 290.022 ms | 282.860 ms | 7.162 ms | 97.53% |
| 第一次 DtoH 之后 | 4,206.601 ms | 106.551 ms | 4,100.050 ms | 2.53% |

也就是说，前约 290 ms 的纯 GPU 路径运行良好，真正拖慢 prefill 的是后续 13 段 hybrid CPU-offload 尾部。Trace 中还出现 26 次 `cudaLaunchHostFunc`，正好约每段 2 次；一条 4.242 秒的 `cudaDeviceSynchronize` 主要是在等待这些异步 callback、CPU expert 结果和 stream 依赖完成，不应误读成 CUDA API 自身计算了 4.242 秒。

Trace 共出现 43 个 sparse-MLA/transformer block 阶段。L0–L29 出现了 30 组 GPU expert compute（每组两次 grouped GEMM），而 L30–L42 没有对应 GPU expert compute，却各出现一组完整 activation 往返和长同步等待。因此可以确认本次**实际执行结果**是“前 30 个 block 有 GPU expert compute、后 13 个 block 走 TP0 CPU/offload”。但仅凭 trace 不能唯一反推出这是由 `kt_num_gpu_layers=30` 造成，`front-loading`、`frequency`、ratio/count 组合或实际 per-layer placement mask 也可能形成相似分布；需结合 `/server_info` 和启动日志确认具体配置来源。

### 6.5 本次 16K 测试没有真正形成 16K forward

立即检查 resolved 配置：

```bash
curl -s http://192.168.50.211:40010/server_info | jq '{
  chunked_prefill_size,
  max_prefill_tokens,
  kt_gpu_prefill_token_threshold,
  kt_num_gpu_layers,
  kt_num_gpu_experts,
  kt_gpu_experts_ratio,
  kt_expert_placement_strategy,
  init_expert_location,
  kt_enable_dynamic_expert_update,
  enable_dynamic_chunking,
  enable_mixed_chunk,
  context_length,
  launch_command
}'
```

一种高度可疑、但仍需 `/server_info` 证实的失配是：

```text
chunked-prefill-size            = 16384
实际 max/effective prefill       = 4096
kt-gpu-prefill-token-threshold  > 4096
```

无论具体裁切原因是什么，trace 可以确定当前 4K forward 没有进入 full-GPU layerwise 权重流水，而是继续走 hybrid CPU experts；否则不会出现 13 组完整的 32 MiB activation DtoH/HtoD 和 300+ ms TP0 host gap。除查清 4K 裁切来源外，还应独立检查启动日志是否出现 `layerwise prefill disabled/OOM; using hybrid` 一类 fallback 信息，以解释 layerwise path 未生效的原因。

### 6.6 Prefill 最优先 A/B 配置

先根据 `/server_info` 选择测试分支。

如果 resolved effective chunk 本来就是 4K，先做低风险验证：

```text
--chunked-prefill-size 4096
--max-prefill-tokens 4096
--kt-gpu-prefill-token-threshold 4096
```

该组合满足 threshold 等于 chunk，目标是确认 4K batch 能否进入 layerwise/full-GPU path。如果 resolved 配置声称 16K、但空载单请求 trace 仍只有 4K，应先查 mixed chunk、dynamic chunking、调度 budget 或 fallback 日志，不应直接把 threshold 降到 4K 掩盖裁切原因。

在显存和 context length 允许时，再测试真正的 16K 三项对齐：

```text
--chunked-prefill-size 16384
--max-prefill-tokens 16384
--kt-gpu-prefill-token-threshold 16384
```

目标是让实际 forward 达到 16K，并在达到 threshold 时进入 layerwise/full-GPU prefill。当前 DeepSeek-V4-Flash 教程还要求 threshold 不小于 chunked size，上述组合满足约束。

如果 16K 组合 OOM，则按同样原则回退为：

```text
8192 / 8192 / 8192
```

每组必须先做一次同 shape 预热，再 `/flush_cache`，最后在 profiler-off 条件下比较 prefill tok/s 和 TTFT。随后抓一份 GPU-only trace，确认：

- sparse prefill kernel grid 已从 4096 变为目标 token 数；
- 32 MiB activation 往返消失或显著减少；
- 13 个 300+ ms AllReduce 等待点消失或缩短；
- TP0 GPU busy 上升，TP1 NCCL 覆盖下降。

第二组 A/B 是动态 expert 更新开/关。若 runtime dynamic update 对 prefill 造成明显同步成本，可用离线 `frequency` placement 替代，并单独确认 decode 54 tok/s 是否保持。

第三组是先确认当前 expert placement 的来源。如果 `/server_info` 和启动日志确认 `--kt-num-gpu-layers 30` 正在主导该分布，再在显存安全范围内测试 `30 -> 31 -> 32`。如果实际由 `front-loading`、`frequency` 或 ratio/count mask 形成，则应按对应策略调整，不能直接把 trace 形态等同于 `kt_num_gpu_layers=30`。

按 V4-Flash 结构粗估，单 expert MXFP4 约 15.73 MB，即约 15.0 MiB；256 experts 的单层全局权重约 3.75 GiB。TP=2 理论权重约 1.88 GiB/卡/层，13 层约 24.4 GiB/卡，另外还要计入元数据、重排副本和临时 buffer。虽然现场表面余量约 31 GiB，但仍需保留 KV cache、CUDA Graph、layerwise slots、activation 和碎片空间，因此只能逐层压测。

按本次 trace 的 4K hybrid 路径，每减少一个 CPU-managed 层，最多可去掉一个约 310–336 ms 的暴露等待段；实际收益还要扣除新增 GPU MoE 计算，并以 profiler-off A/B 为准。

### 6.7 旧 20-token prefill 结论已被替代

旧 CPU+GPU trace 中 20-token EXTEND 的 285 ms 受 profiler 和小 shape 严重影响，不再用于 prefill 性能判断。其“跨 rank 有效计算相等、存在 rendezvous 偏斜”的结构性结论与最新长 prefill trace 一致，但所有 prefill 定量结论以本节最新 GPU-only trace 为准。

## 7. 为什么低功耗也会显示 100% GPU-Util

Decode 中的长 NCCL kernel launch 配置为：

```text
grid  = [1, 1, 1]
block = [512, 1, 1]
```

Prefill 中为 4 个 CTA，而每张 GPU 有 188 个 SM。

因此等待期间并不是 188 个 SM 都在执行高吞吐矩阵计算；基本只是极少量 NCCL block 持续驻留、轮询或等待 peer。`nvidia-smi` 的 GPU-Util 更接近“采样窗口内是否持续有 kernel 执行”，不能区分：

- 高吞吐 GEMM；
- Attention；
- 通信；
- barrier/rendezvous 等待。

所以以下两件事可以同时成立：

```text
GPU1 GPU-Util = 100%
GPU1 功耗仅约 110 W / 600 W
```

它表示 GPU1 时间线上几乎一直有 NCCL kernel 驻留，不表示 GPU1 的 Tensor Core、显存带宽或 188 个 SM 被有效跑满。

## 8. 对应代码路径

### 8.1 TP0 独占 CPU expert wrapper 和 stream

`third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py:3976-4059`：

- 只有 `tp_rank == 0` 创建 `_cpu_stream`；
- 只有 `tp_rank == 0` 创建 `KTMoEWrapper`；
- CPU experts 的权重和运行时状态只存在于 TP0 路径。

### 8.2 非 TP0 不提交或等待 CPU expert

`third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py:4290-4357`：

- 非 TP0 的 CPU submit 直接返回；
- 非 TP0 的 CPU sync 返回零张量；
- TP1 因此会比 TP0 更早走到后续同步点。

### 8.3 TP0 执行 staging、CPU submit、sync 和 merge

`third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py:4587-4693`：

1. TP0 将 hidden states 复制到 staging buffer；
2. TP0 提交 CPU expert 计算；
3. GPU experts 与 CPU experts 尝试并行；
4. TP0 等待 CPU 结果；
5. TP0 将 CPU output 合并到 routed-expert output。

这正好对应 trace 中 TP0 独有的大量 DtoH/HtoD 和 host callback。

### 8.4 每层最终进入 TP AllReduce

`third_party/sglang/python/sglang/srt/models/deepseek_v2.py:1185-1188`：

```python
if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
    is_tp_path=True,
):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

因此典型时间线是：

```text
TP0: GPU 分片计算 -> DtoH/CPU expert/HtoD/merge -> 较晚进入 AllReduce -> 很快退出
TP1: GPU 分片计算 --------------------------> 较早进入 AllReduce -> 驻留等待 TP0
```

## 9. 最终结论

### 9.1 已确认

以下确认的是结构性因果关系和同一次 trace 内的 rank 相对差异，不是生产环境的绝对延迟：

1. **两卡有效模型 GPU 计算量是均衡的。** Decode 排除 AllReduce 后的累计 kernel 时间只差 0.014%；最新 4K prefill 的非 NCCL kernel 累计时间为 TP0 219.605 ms、TP1 219.395 ms，只差约 0.096%。
2. **GPU1 的 100% 不是有效算力跑满。** Decode trace 中 TP1 的 GPU 活动窗口约有一半被 NCCL 覆盖；4K prefill 中更达到约 94.9%。长 NCCL kernel 只有 4 个 CTA，而 GPU 有 188 个 SM。
3. **在 decode/旧短 prefill trace 中，TP1 多出的 kernel 时间有 99.981% 来自 AllReduce。** 这不是权重、Attention 或 GEMM 切分不均。
4. **TP1 在固定 collective 上提前到达并等待 TP0。** TP1 的同一 collective 平均 665.17 µs，TP0 只有 9.12 µs，二者最终几乎同时结束。
5. **TP0 的迟到与 TP0-only CPU expert/offload 路径高度对应。** TP0 有 406 次 HtoD、1,200 次 DtoH，TP1 只有 16 和 30 次；377 个主要等待窗口逐一对应 TP0 HtoD。该路径包括 host 调度、数据搬运、CPU 计算和结果合并。
6. **根因是架构级不对称。** 当前 KT 混合 MoE 将 CPU experts 集中到 TP rank 0，而每层又需要 TP collective 会合。
7. **所谓 16K 测试实际是 4K forward。** Kernel grid 和 32 MiB activation 明确给出 4,096 tokens；其约 4.497 秒跨度对应约 911 tok/s，与线上约 890 tok/s 吻合。
8. **长 prefill 的主瓶颈是 13 段 TP0 CPU expert/offload host gap。** 合计约 4.10 秒，占观察窗口约 91.2%；PCIe DMA 合计仅约 22 ms，不是主要瓶颈。
9. **当前 4K forward 没有进入 full-GPU layerwise prefill。** 应优先核对 resolved chunk、`max_prefill_tokens`、threshold 和 fallback 日志，再决定使用 4K 对齐或真正的 16K 对齐配置。

### 9.2 根因表述

本问题不是“GPU1 分配到的计算太多”，而是：

> **TP0 承担 CPU expert/offload 临界路径并产生 GPU 空洞；TP1 完成自己的 TP 分片后提前进入 NCCL AllReduce，通过常驻通信 kernel 等待 TP0。`nvidia-smi` 将这种等待显示成 GPU1 100%。**

### 9.3 `--disable-custom-all-reduce` 不是根因

当前参数使等待显示为 NCCL：

```text
ncclDevKernel_AllReduce_bf16_RING
```

如果删除该参数，等待可能改为自定义 AllReduce kernel，但 rank 到达偏斜仍然存在。更换 collective backend 只能改变等待发生在哪个 kernel 中，不能消除 TP0 的 CPU expert 临界路径。

## 10. 如何提升有效 GPU 利用率

### 10.1 优化目标不能只看 `nvidia-smi`

当前 GPU1 已经显示约 100%，但本次 trace 的 TP1 GPU 时间线中约一半被 NCCL kernel 覆盖。继续把 GPU-Util 数字做高没有意义，真正目标应是：

1. 降低 TP0 的 CPU expert 临界路径；
2. 减少 TP1 提前进入 collective 后的驻留等待；
3. 提高 GEMM、Attention、GPU MoE 等非 NCCL kernel 的占比；
4. 最终提高无 profiler 条件下的 prefill/decode tok/s 和并发吞吐。

本次 profile 的诊断量可简化为：

```text
Step 关联 GPU 事件覆盖窗口   ≈ 18.46 ms
每个 rank 的非 NCCL GPU 计算 ≈ 8.72 ms/step
TP1 AllReduce 驻留           ≈ 9.36 ms/step
TP0 AllReduce 驻留           ≈ 0.94 ms/step
由到达偏斜造成的差值         ≈ 8.42 ms/step
```

其中 8.42 ms 只是在该 profiler 配置下观测到的 TP0/TP1 AllReduce 驻留差。Profiler 可能改变 host launch、CPU/offload 和 rank 到达节奏，因此不能用 `18.46 - 8.42` 预测线上 step，也不能据此承诺具体加速倍数。

正确的优化判据是：在相同 profiler 配置下，该差值和长 collective 持续时间相对下降；同时在关闭 profiler 后，线上 prefill/decode tok/s、TTFT 和 ITL 确实改善。

### 10.2 第一优先级：减少落到 CPU 的 expert 工作

这是最直接、最可能同时提升 TP0 和 TP1 有效利用率的方案。

#### 方案 A：增加常驻 GPU experts

`--kt-num-gpu-experts` 的语义是**每个 MoE 层常驻 GPU 的 expert 数量**，不是整个模型的全局总数；增加 1 个会在多个 MoE 层上同时增加权重，因此显存增幅可能很大。应先根据模型层数、单 expert 大小和 TP 分片估算显存，再按小步增加：

```text
--kt-num-gpu-experts <当前值 + 1>
--kt-num-gpu-experts <当前值 + 2>
--kt-num-gpu-experts <当前值 + 4>
```

也可以使用全局比例：

```text
--kt-gpu-experts-ratio <ratio>
```

如果同时配置两者，当前实现中 `--kt-gpu-experts-ratio` 会覆盖 `--kt-num-gpu-experts`，A/B 测试时不要让两个参数互相干扰。

现场每卡约使用 67 GiB/98 GiB，表面上还有约 31 GiB 空间，但不能全部用于 expert 权重，还要为以下内容留余量：

- KV cache；
- CUDA Graph；
- layerwise prefill 临时层；
- prefill activation；
- allocator 碎片和峰值请求。

建议每次只增加一个小档位；由于该数量会乘以全部 MoE 层，`+1` 也可能增加数百 MiB 甚至超过 1 GiB，具体取决于 expert 大小、层数、精度和 TP 分片。每档都要用最大计划上下文和最大 `chunked-prefill-size` 做 OOM 压力测试。

预期在 trace 中看到：

- CPU expert 工作量和 13 个长 AllReduce 节点的持续时间下降；
- TP0 DtoH/HtoD 的次数可能保持不变，因为 hybrid wrapper 仍会按 CPU-managed 层搬运完整 activation；只有整层绕过 KT 或进入 full-GPU layerwise path 时，对应搬运和等待节点才会消失；
- TP0 idle 从约 8.86 ms/step 下降；
- TP1 AllReduce 从约 9.36 ms/step 下降；
- decode tok/s 上升。

#### 方案 B：使用热点 expert 放置

如果当前仍使用默认的 `uniform` 放置，它不保证常驻 GPU 的 expert 正好是业务流量最常命中的 expert。应使用代表性请求收集路由频率，然后测试：

```bash
--kt-expert-placement-strategy frequency \
--init-expert-location /path/to/logical_count.pt
```

具体的分布采集流程可参考 [experts-sched-Tutorial](en/kt-kernel/experts-sched-Tutorial.md)。如果模型、量化方法和当前版本支持，也可测试：

```text
--kt-enable-dynamic-expert-update
```

动态更新要求正确配置正数的 `--kt-gpu-prefill-token-threshold`。如果使用 `kt run`，当前 CLI 可能已经自动添加动态更新参数，应先检查实际展开后的服务启动命令和启动日志，避免误以为尚未启用。

热点放置比单纯增加随机/均匀 GPU expert 数量更节省显存。判断它是否生效，不看“GPU experts 数量”，而看：

- CPU expert 实际命中率；
- TP0 CPU/offload gap 和 DtoH/HtoD 持续时间；
- 13 个长等待节点的持续时间；
- 最终 tok/s。

#### 方案 C：将前 N 层的 routed experts 固定在 GPU

最新 trace 确认当前前 30 个 block 有 GPU expert compute、后 13 个 block 走 CPU/offload，但不能仅凭该形态唯一推断配置值。先核对 `kt_num_gpu_layers`、placement strategy、ratio/count 和启动日志中的 per-layer placement；如果确认当前确由 `kt_num_gpu_layers=30` 主导，再小步测试：

```text
--kt-num-gpu-layers 31
--kt-num-gpu-layers 32
```

这些层中的 routed experts 会绕过 KT CPU expert 路径，从而依次覆盖当前 L30、L31 等长等待点；Attention、dense/shared 等其他模块并不会因为这个参数整体搬迁。按本模型粗估，每增加一层理论 routed-expert 权重约增加 1.88 GiB/卡，实际还要计入元数据、重排副本和临时 buffer，必须逐层检查峰值显存、OOM 和实际 tok/s，不能一次把 13 层全部补齐。

### 10.3 第二优先级：缩短 TP0 CPU expert 时间

即使 GPU expert 数量不变，只要 TP0 更快完成 CPU experts，TP1 的 AllReduce 等待也会同步缩短。

#### CPU 和 NUMA 配置

先确认硬件拓扑：

```bash
nvidia-smi topo -m
numactl -H
lscpu -e=CPU,NODE,SOCKET,CORE
```

需要重点核对：

- GPU0 所在的 NUMA node；
- CPU expert 权重实际分配在哪些 NUMA node；
- `--kt-threadpool-count` 是否与使用的 NUMA node 数量匹配；
- `--kt-numa-nodes` 是否指向本地内存带宽更合适的节点；
- `--kt-cpuinfer` 是否超过物理核心数量而引入超线程争用。

建议做小型参数扫描，而不是直接把线程数开到最大：

```text
--kt-cpuinfer 32 / 48 / 64 / 96
--kt-threadpool-count 1 / 2 / 实际计划使用的 NUMA 节点数
```

每次只修改一个变量，并记录 CPU IPC、内存带宽、decode tok/s 和 TP1 AllReduce 时间。CPU expert 常见瓶颈可能是内存带宽而不是核心数量，线程继续增加不一定更快。

#### 核对实际 CPU kernel

确认当前安装的 `kt-kernel`、量化方法和 CPU 指令路径符合硬件能力。例如 x86 平台应确认没有意外回退到较慢的通用/AVX 路径；ARM 平台则应确认使用了相应的 NEON/native backend。

目标不是单独提高 CPU benchmark，而是让以下指标下降：

```text
TP0 CPU submit -> CPU finish -> HtoD finish -> AllReduce enter
```

### 10.4 第三优先级：提高真实并发和 GPU kernel 规模

当前采集是 batch size 1。单请求 decode 的 GEMV、小 GEMM 和小 Attention kernel 很难充分利用 188 个 SM。增加并发可以让 SGLang 动态 batching 形成更大的 GPU 工作量。

建议测试固定输入/输出长度下的并发：

```text
1 -> 4 -> 8 -> 16 -> 32
```

同时记录：

- 聚合 output tok/s；
- 单请求 p50/p95/p99 延迟；
- TP0 CPU 带宽和利用率；
- TP1 AllReduce 等待；
- 每瓦 tok/s。

并发增大可能有两种结果：

1. GPU kernel 变大、CPU/GPU 固定开销被摊薄，聚合吞吐提高；
2. CPU experts 和内存带宽先饱和，TP0 临界路径进一步拉长。

因此并发不是根因修复，但通常是最容易提高“有效 GPU 工作量”的服务侧手段。

### 10.5 Prefill 专项优化

用户线上冷 prefill 约为 **890 input tok/s**。最新 GPU-only trace 证明它实际来自一个 4K hybrid CPU/GPU forward，而不是 16K layerwise/full-GPU forward；约 91.2% 的时间暴露在 13 段 CPU expert/offload host gap 中。因此优化顺序已经可以明确。

#### 第一步：核对服务真正生效的参数

首先读取 `/server_info`，确认 `chunked_prefill_size`、`max_prefill_tokens`、threshold、dynamic/mixed chunk、`kt_num_gpu_layers`、`kt_num_gpu_experts`、`kt_gpu_experts_ratio`、`kt_expert_placement_strategy` 和 `init_expert_location`，再从启动日志核对实际 per-layer placement mask。测试脚本里的 `CHUNKED_PREFILL_SIZE=16384` 不会修改服务端配置。

如果 resolved 参数为 16K，但空载单请求仍只形成 4K forward，检查启动日志中的 scheduler budget 和 layerwise fallback/OOM 信息。在找到裁切原因前，不要只改一个 threshold 参数。

#### 第二步：让 effective chunk 与 layerwise gate 对齐

优先做两组 profiler-off A/B：

```text
# 保守 4K 对齐
chunked-prefill-size=4096
max-prefill-tokens=4096
kt-gpu-prefill-token-threshold=4096

# 真正 16K 对齐
chunked-prefill-size=16384
max-prefill-tokens=16384
kt-gpu-prefill-token-threshold=16384
```

每组先预热一次，再 `/flush_cache`，重复 3–5 次记录 prefill tok/s、TTFT、峰值显存和 decode p95/p99。16K 如果 OOM，则测试三项均为 8192。DeepSeek-V4-Flash 当前教程要求 threshold 不小于 chunked size。

#### 第三步：若继续保留 hybrid path，减少 CPU-managed 层

如果 `/server_info` 和 placement 日志确认当前由 `kt_num_gpu_layers=30` 主导，按 `30 -> 31 -> 32` 逐档测试。每增加一层理论 routed-expert 权重约增加 1.88 GiB/卡，并可能消除一个约 315 ms/4K-chunk 的 CPU gap；实际还要为元数据、重排副本、KV、CUDA Graph、activation 和 allocator 留出余量。

同时可小步增加 `kt_num_gpu_experts`，并使用 `frequency` placement 减少 CPU expert 命中。由于它是“每个 MoE 层”的数量，仍只按 `+1/+2` 测试。

#### 第四步：优化 CPU expert 本身

若 13 层必须留在 CPU，优化目标是缩短每层约 315 ms 的 host gap：

- 核对 AMX/AVX/NEON 实际 kernel，没有回退到慢路径；
- 扫描 `kt_cpuinfer`、`kt_threadpool_count` 和 `kt_numa_nodes`；
- 让 CPU 权重、线程池和 GPU0 尽可能 NUMA 本地；
- 测量 DRAM 带宽，判断是算力还是内存带宽受限。

#### 第五步：评估动态 expert 更新的成本

当 layerwise path 真正生效后，再做 dynamic update 开/关 A/B。`--kt-enable-dynamic-expert-update` 有利于后续 decode 的热点 expert 放置，但会增加统计、broadcast、同步和权重更新。可用离线 `frequency` placement 作为无运行时更新的对照。

#### 第六步：利用 Radix prefix cache

如果大量请求共享 system prompt、工具描述或固定文档前缀，启用 Radix cache 往往比继续优化冷 prefill kernel 更有效。测冷 prefill 时应 flush cache；测真实业务有效吞吐时则应保留 cache，并分别报告：

- cache miss prefill tok/s；
- cache hit ratio；
- TTFT；
- 实际被复用的 prefix tokens。

如果启动命令包含 `--disable-radix-cache`，可以单独建立一组移除该参数的业务流量 A/B，但要重新评估 KV cache 显存占用和淘汰策略。

#### 代码级优化方向

若配置调优后仍受限，hybrid 路径的首要工程目标是缩短或分散 TP0-only CPU expert 临界路径：

- CPU expert 是否能按 TP rank/NUMA 分片；
- DtoH、CPU compute、HtoD 是否能与更多 GPU 工作流水重叠；
- 13 个逐层 rendezvous 是否能减少或延后；
- full-GPU layerwise 路径中的权重 prepare、双 buffer 和动态更新同步是否能进一步流水化。

这些改动需要同时验证数值正确性、CUDA Graph、不同 batch 和 OOM fallback。

### 10.6 如果单卡可容纳：优先评估两个 TP=1 实例

TP=2 会分摊 GPU 权重与计算，但也引入每层 rendezvous；当前 trace 中，两卡有效计算量几乎相同，而 TP1 存在显著额外 NCCL 驻留。它表明 rendezvous 是值得验证的成本项，但 TP=2 是否仍有净收益，必须通过相同工作负载下 TP=1/TP=2 的无 profiler A/B 确认。如果模型和目标 KV cache 能在单卡上运行，应评估：

```text
GPU0: TP=1 实例 A
GPU1: TP=1 实例 B
前端负载均衡到两个实例
```

这种部署可以：

- 删除跨卡 TP AllReduce；
- 让两张卡都处理真实请求，而不是一张卡等待另一张卡；
- 在多用户场景下提高聚合吞吐。

但必须实际验证：

- 单卡显存是否容纳 GPU 权重、KV cache 和最大上下文；
- 两个实例是否重复加载大量 CPU expert 权重；
- 两个实例是否争抢同一套 CPU 核和 DRAM 带宽；
- 单实例时延与双实例聚合吞吐。

如果单卡不能容纳目标上下文，或者 CPU 内存带宽无法支撑双实例，则继续使用 TP=2，并优先做 GPU expert 热点放置和 CPU 路径优化。

### 10.7 不应优先做的事情：只更换 AllReduce backend

本次 trace 的 13 个主要等待 collective 中：

```text
TP0 同位置平均持续时间 ≈   9.12 µs
TP1 同位置平均持续时间 ≈ 665.17 µs
到达偏斜                 ≈ 656.73 µs
```

这说明在本次 profile 中，真正的数据传输和正常同步只占很小一部分，绝大部分时间是 TP1 等 TP0。删除 `--disable-custom-all-reduce` 可能改变 kernel 名称和少量通信开销，但不能从架构上消除 rank 到达偏斜；实际偏斜数值会随 profiler、backend 和运行条件变化。

因此在当前阶段：

- 保留稳定的 NCCL 路径作为基线；
- 不要把切换 custom all-reduce 当作主优化；
- 只有在 CPU expert 临界路径显著缩短后，再比较 NCCL 与 custom all-reduce 的纯通信差异。

### 10.8 最直接但需要开发的根治方案

配置调优只能减少问题，架构上的根治方向是取消“CPU experts 全部集中在 TP0”。

#### 方案 A：CPU expert 按 TP rank 分片

让每个 TP rank 负责与自身权重分片对应的 CPU expert 计算和结果搬运，使两个 rank 都有近似对称的 host/CPU 工作量，再进入 collective。

这一方案只有在 CPU 核、NUMA 本地内存和 DRAM 带宽可以真正并行支撑多个 rank 时，才会缩短 step。若多个 rank 仍争抢同一内存带宽瓶颈，它可能只让两卡看起来更对称，却没有提高吞吐，甚至增加复制和同步开销。

预期效果：

- TP0 的 DtoH/HtoD 压力下降；
- TP1 不再像本次 trace 中那样显著提前到达固定同步点；
- 两卡 collective 持续时间接近；
- 两卡有效 GPU busy 更对称。

需要处理：

- CPU 权重分片格式；
- NUMA 放置；
- rank-local top-k 路由；
- partial output 的数值一致性；
- CUDA Graph 和 host callback 生命周期。

#### 方案 B：加深 CPU/GPU 流水重叠

把下一层 CPU expert 的输入准备、CPU submit 或 HtoD 与当前层仍可执行的 GPU 工作重叠，减少 TP0 主 GPU stream 的空洞。

本次 trace 中 TP0 的大部分 GPU 空洞可由 13 个会合等待窗口解释，因此这是明确的工程优化靶点；绝对毫秒数需要用相同 profiler 配置做修改前后 A/B，不能直接外推线上收益。

#### 方案 C：减少 rendezvous 数量

如果模型依赖允许，可评估：

- 合并部分 collective；
- 延后不影响下一层依赖的结果归并；
- 使用 reduce-scatter/all-gather 重构数据流；
- 避免 CPU output 每层都成为全局同步点。

这类改动涉及模型并行语义和数值正确性，需配套单元测试、端到端精度测试和多 batch CUDA Graph 测试。

#### 方案 D：近似重叠选项

`--kt-max-deferred-experts-per-token` 可以隐藏部分 CPU 等待，但当前语义会把部分低权重 expert 延后处理，属于近似执行路径，可能改变逐层数值结果。如果要求严格数值无损，不应作为默认生产优化；如果业务允许，需要单独进行质量评估。

### 10.9 推荐的执行顺序

| 优先级 | 实验 | 目标 | 通过标准 |
|---|---|---|---|
| P0 | `/server_info` 与启动日志核查 | 找到 16K 配置实际只发 4K forward 的原因 | resolved 参数和实际 kernel grid 一致 |
| P0 | 4K 三项对齐 | 验证 layerwise gate | 13 个 300 ms 等待段消失/缩短，无 OOM |
| P0 | 16K 三项对齐 | 真正使用 16K forward | kernel grid=16384，prefill tok/s 上升 |
| P0 | `kt_num_gpu_layers: 30->31->32`（确认 placement 来源后） | 减少 CPU-managed 层 | 每档消除一个对应层的搬运/长等待，显存可控 |
| P1 | GPU experts +1/+2 与 frequency placement | 减少 CPU expert 命中 | CPU/offload gap 与长等待时长下降；搬运次数可不变 |
| P1 | CPU 线程/NUMA/内核扫描 | 缩短 TP0 临界路径 | 单层约 315 ms host gap 下降 |
| P1 | dynamic update 开/关 | 平衡 prefill 与 decode | prefill 上升且 decode 54 tok/s 可接受 |
| P1 | 并发 1/4/8/16 | 提高聚合吞吐 | output tok/s 提升，p99 可接受 |
| P1 | TP=1 单实例 | 判断 TP 是否有净收益 | tok/s 接近或超过 TP=2，显存可控 |
| P1 | 两个 TP=1 副本 | 提高双卡聚合吞吐 | 总吞吐高于单个 TP=2 |
| P2 | CPU experts 多 rank 分片 | 根治 rank 不对称 | TP0/TP1 collective 时间接近 |

### 10.10 每轮实验必须记录的指标

不要只记录 GPU-Util。至少记录：

| 指标 | 本次 profile 诊断值或线上基线 | 优化方向 |
|---|---:|---|
| 实际 prefill forward tokens | 4,096 | 与目标 chunk/threshold 对齐 |
| 4K prefill GPU 观察窗口 | 约 4,496.6 ms | profiler-off tok/s 上升、TTFT 下降 |
| Prefill TP0 / TP1 GPU busy | 8.66% / 99.74% | TP0 上升、TP1 等待下降 |
| Prefill TP1 NCCL 覆盖 | 94.87% | 显著下降 |
| 13 段 TP0 host gap | 约 4,099.5 ms | layerwise 或减少 CPU-managed 层 |
| Prefill host-device DMA | 约 22 ms | 非当前首要瓶颈 |
| Decode GPU 活动覆盖窗口 | 约 18.46 ms/step | 仅用于同配置 A/B |
| Decode 非 NCCL kernel union | 约 8.72 ms/step/rank | 占 trace 窗口比例上升 |
| TP0 AllReduce | 约 0.94 ms/step | 保持或下降 |
| TP1 AllReduce | 约 9.36 ms/step | 显著下降 |
| TP1-TP0 AllReduce 差 | 约 8.42 ms/step | 相同 profiler 配置下显著下降 |
| TP0 idle | 约 8.86 ms/step | 显著下降 |
| Decode/旧 trace TP0 HtoD/DtoH | 406 / 1,200 次/trace | 整层 GPU 化时次数下降；否则关注持续时间 |
| 线上 prefill | 约 890 input tok/s | profiler-off 条件下上升 |
| 线上 decode | 约 54 output tok/s | profiler-off 条件下上升 |
| TTFT / ITL | 另行记录 | 下降 |
| p50/p95/p99 | 另行记录 | 不超过业务目标 |
| 功耗与每瓦 tok/s | 另行记录 | 有效提升 |

Trace 指标只用于同采集配置下的相对比较；最终仍应以关闭 profiler 后的吞吐、延迟、精度和稳定性共同判断。

### 10.11 建议补抓的 Profile

最新 trace 已经覆盖一个具有生产代表性的 4,096-token forward，足以确认当前根因；在拿到 `/server_info` 和启动日志前，**不需要为了重复证明同一根因立即重抓**。

下一次采集应围绕配置 A/B，而不是重复当前配置：

- 实际业务 P50 输入长度；
- 实际业务 P95 输入长度；
- 4K 三项对齐并确认 layerwise 生效后的 trace；
- 真正 16K 三项对齐后的 trace；
- TP=1 基线；
- TP=2 且提高 GPU expert 数量后的对照组；
- TP=2 且启用 frequency placement 的对照组；
- 并发 1、4、8、16 的对照组。

`PROFILE_STEPS` 应按实际 resolved chunk 决定：如果服务仍以 4K forward 处理 16K 输入，至少使用 `PROFILE_STEPS=4` 才能覆盖完整输入；如果单个 forward 已经真正达到 16K，`PROFILE_STEPS=1` 即可。若目标只是确认单个 chunk 是否进入 layerwise path，1 step 也足够。

长 prefill 重采时建议：先用同 shape 内容完成一次 layerwise slot 预热，然后调用 `/flush_cache`，再以新内容或唯一前缀发送正式请求；Profiler 只开 GPU activity，关闭 stack、shape 和 detailed annotations。这样既避免首次懒加载，也避免 Radix cache 直接跳过 prefill。

仓库已提供针对 16K 目标的完整脚本：[profile_long_prefill_16k.sh](../kt-kernel/scripts/profile_long_prefill_16k.sh)。

脚本会依次执行服务健康检查、读取并校验 `/server_info`、同 shape 预热、Radix cache 清理、三轮 profiler-off 基线、GPU-only profile、`nvidia-smi dmon` 以及容器 trace 路径收集。脚本中的 `CHUNKED_PREFILL_SIZE` 是校验目标，不会修改已经启动的服务参数。

每组至少记录：

- prefill/decode tok/s；
- TP0/TP1 功耗和显存；
- AllReduce union time；
- 非 NCCL kernel union time；
- TP0 HtoD/DtoH 次数和时间；
- 13 个固定长 collective 的平均、P50、P95 和最大时间。

## 11. 分析限制

1. 旧 CPU+GPU Torch Profiler 对 prefill/eager 路径扰动很大；其中 20-token EXTEND 的 285 ms 不用于生产性能判断。最新 prefill trace 只启用 GPU activity，4096-token 的约 4.497 秒 GPU 时间线与线上 890 tok/s 接近，但它仍不是完整 HTTP TTFT。
2. 最新采集只覆盖一个 4K forward，足以定位单 chunk 的临界路径，但不能代表多请求并发、P50/P95 输入分布或完整 16K 请求的所有 chunk。
3. Trace 能确定实际 forward 为 4K，却不能单独确定为何服务没有发出 16K forward；最终原因需要 `/server_info`、启动命令和 fallback/OOM 日志。
4. Kernel `dur` 直接求和会重复计算跨 stream overlap；本报告在解释 GPU busy 时使用了 interval union。
5. Trace 不包含完整功耗、频率和 SM 吞吐遥测；95 W/110 W 来自用户提供的 `nvidia-smi` 快照。
6. GPU-only trace 没有展开 KT native CPU worker 内部的算术、线程池和 DRAM 访存，因而不能把约 4.1 秒 host gap 精确拆分；但 rank 配对、GPU 空洞、host callback、activation 往返和代码路径足以将其归入 TP0 CPU expert/offload 临界路径。

## 12. 一句话总结

> **两张 GPU 执行了基本相同的有效模型计算；区别在于 GPU1 完成 TP 分片后，又用一个低占用、长驻的 NCCL kernel 等待承担 CPU experts 的 TP0。解决方向应是缩短或分散 TP0 CPU-offload 临界路径，让等待时间转化为有效 GPU 工作，而不是把 GPU1 的 100% 当成算力已经充分利用。**
