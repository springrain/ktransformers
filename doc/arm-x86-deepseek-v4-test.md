

# ARM(AmpereOne)与x86(Intel Xeon)平台DeepSeek V4系列推理性能对比测试报告

## 一、核心结论

| 模型 | AmpereOne + 2×RTX PRO 6000 | x86 Xeon 6530 + 2×RTX PRO 6000 | 性能对比 |
|---|---|---|---|
| **DeepSeek‑V4‑Flash**(13B激活) | **34 token/s** | **35 token/s** | 基本持平(x86略高2.9%) |
| **DeepSeek‑V4‑Pro**(49B激活) | **14 token/s** 🏆 | **8 token/s** | **ARM领先75%** |

> **核心洞察**：在GPU硬件完全相同的前提下，当MoE模型激活参数从13B暴涨至49B时，x86主机端的CPU侧矩阵计算能力出现严重瓶颈(吞吐暴跌77%)，而AmpereOne凭借超多核心优势，将性能衰减控制在59%，实现大幅反超。

---

## 二、测试环境

### 2.1 ARM平台(AmpereOne)

- **CPU**：AmpereOne(`ampere1/ampere1a/ampere1b`)，高核心数(如192核)架构
- **构建**：`-mcpu=native` 默认原生构建
- **关键指令集**：NEON(asimd)、FP16(asimdhp/fphp)、BF16(bf16)、INT8 dot(asimddp)、I8MM(i8mm)
- **GPU**：**双路 NVIDIA RTX PRO 6000**(与x86平台完全相同)
- **内存**: 1T DDR5
- **推理后端**：HeterogeneousComputing + LLAMAFILE GGUF CPU + GPU协同计算

### 2.2 x86平台(Intel Xeon Gold 6530)

- **CPU**：双路 Intel Xeon Gold 6530(Sapphire Rapids)，共64核/128线程
- **关键指令集**：AMX(INT8/BF16/INT4)、AVX‑512 VNNI
- **GPU**：**双路 NVIDIA RTX PRO 6000**
- **内存**: 1T DDR5
- **推理后端**：HeterogeneousComputing，CPU(AMX)与GPU协同计算

---

## 三、ARM平台实测性能与执行路径

### 3.1 吞吐量实测

| 模型 | 实测吞吐 | 激活参数量 | 关键特征 |
|---|---|---|---|
| **DeepSeek‑V4‑Flash** | **34 token/s** | 130亿(13B) | 与x86异构性能持平 |
| **DeepSeek‑V4‑Pro** | **14 token/s** | 490亿(49B) | 远超x86异构方案(+75%) |

### 3.2 各量化格式执行路径(已验证)

| KT格式 | 实际执行方式 | 硬件加速程度 |
|---|---|---|
| **BF16** | 原生BFDOT | ✅ 完全硬件加速 |
| **FP16** | 原生NEON FP16 | ✅ 完全硬件加速 |
| **FP32** | 原生NEON FP32/FMA | ✅ 完全硬件加速 |
| **Q8_0** | 原生INT8 dot(`vdotq_s32`)+ 软件scale | ⚡ 硬件加速+后处理 |
| **Q4_0** | 软件拆4‑bit → INT8 dot | ⚡ 部分硬件加速 |
| **MXFP4** | LUT解码 → INT8 dot | ⚡ 部分硬件加速 |
| **FP8** | 软解码E4M3FN → BF16 → BFDOT | ⚠️ 软解+硬算 |
| **MXFP8** | 软解码FP8+UE8M0 → BF16 BFDOT | ⚠️ 软解+硬算 |

> ARM平台**无原生INT4指令**，RAWINT4/GPTQ_INT4不可用；FP8/MXFP8为软解码路径。

---

## 四、x86平台实测性能(基线数据)

| 模型 | 实测吞吐 | 激活参数量 | 备注 |
|---|---|---|---|
| **DeepSeek‑V4‑Flash** | **~35 token/s** | 130亿(13B) | 与ARM持平 |
| **DeepSeek‑V4‑Pro** | **~8 token/s** | 490亿(49B) | 严重衰减，仅为ARM的57% |

---

## 五、深度对比分析：GPU相同时，为什么AmpereOne在Pro版上碾压x86？

### 5.1 性能缩放曲线

| 指标 | AmpereOne + 2×GPU | x86 + 2×GPU |
|---|---|---|
| Flash→Pro激活参数增幅 | 13B → 49B(**×3.77**) | 13B → 49B(**×3.77**) |
| Flash→Pro吞吐衰减幅度 | 34 → 14(**下降58.8%**) | 35 → 8(**下降77.1%**) |
| **缩放效率**(性能/参数比) | **更优** | **较差** |

### 5.2 根本原因剖析：CPU侧矩阵计算负载的急剧膨胀

在KTransformers的MoE混合推理架构中，CPU承担着**双重关键职责**，两者都需要极高的算力：

**控制面(路由与调度，相对轻量)：**
- Token Routing：每个token经过Router计算，决定激活哪几个专家
- 专家权重寻址：确定各专家权重在内存中的位置
- 动态Batch重组：不同token去往不同专家，频繁重组micro-batch

**数据面(大规模矩阵计算，极度吃重)——这是性能差异的核心！**
- 在混合卸载(Hybrid Offloading)策略中，**部分专家(冷专家)被固定在CPU内存中，由CPU直接执行矩阵乘法**
- ARM NEON指令集(`bf16`、`i8mm`、`asimddp`)此时满负荷运转，负责计算分配给CPU侧的专家权重与激活向量的矩阵乘(GEMM/GEMV)
- **这意味着：CPU不是简单“喂GPU”，而是“和GPU并行计算不同的专家”**

当激活参数从13B增至49B时，**CPU侧需要计算的冷专家参数量暴涨数倍**。此时两个平台的差异被急剧放大：

| 负载维度 | AmpereOne(192核) | x86 Xeon 6530(64核) |
|---|---|---|
| **Flash版(13B激活)** | CPU计算少量冷专家(负载轻)，GPU计算热专家，两者并行协同良好 | 同左，64核尚能应付，因此两者跑出相近的34/35 token/s |
| **Pro版(49B激活)** | 192个物理核心凭借极强的NEON并行吞吐，硬生生扛下了暴增的CPU侧矩阵乘法计算量，同时持续输出结果给GPU做后续层计算 | 64个核心在面对暴增的CPU侧矩阵乘法时，算力被瞬间榨干；同时由于CPU计算不及时，导致依赖CPU输出结果的GPU也无法满载，整条流水线进入"CPU Compute-Bound"状态，吞吐直接腰斩至8 token/s |

**结论**：AmpereOne反杀x86的根本原因，**不是"喂料快"，而是"算得多且算得快"**。在Pro版本49B激活参数的巨大计算压力下，192核的绝对物理算力碾压了64核(含超线程)，使得CPU-GPU混合流水线在ARM端依然流畅运转，而在x86端则因CPU侧算力枯竭导致整条流水线严重阻塞。

### 5.3 硬件拓扑因素的放大效应

除了核心数差异，AmpereOne的**单NUMA节点**设计进一步强化了其优势：

| 硬件特性 | AmpereOne | x86 Xeon 6530 |
|---|---|---|
| 物理核心数 | **192核**(无超线程) | 64核(128线程，有超线程) |
| NUMA拓扑 | **单节点**(无跨节点访问) | 双节点(跨路访问有延迟) |
| PCIe带宽利用 | 单节点下所有核心平等访问，无跨路延迟 | 跨NUMA节点访问PCIe设备有额外延迟 |

在Pro版本的巨大计算压力下，x86不仅面临核心数不足的问题，还要承受跨NUMA节点访问内存和PCIe设备的额外延迟，进一步加剧了CPU侧的算力瓶颈。

---

## 六、分场景综合结论与选型建议

### 6.1 不同场景下的最佳平台

| 使用场景 | 推荐平台 | 预期性能 | 核心理由 |
|---|---|---|---|
| **DeepSeek‑V4‑Flash 部署** | x86 / ARM **均可** | ~34‑35 token/s | 13B激活下CPU侧计算压力小，两者持平 |
| **DeepSeek‑V4‑Pro 部署** | **AmpereOne(ARM)** 🏆 | **14 token/s** | CPU侧192核的矩阵计算能力扛住了暴增的负载，保障GPU持续满载，比x86快75% |
| **追求极致单卡/双卡利用率** | **AmpereOne(ARM)** | 更高 | 避免x86主机端瓶颈，让GPU物尽其用 |
| **依赖INT4/GPTQ原始格式** | **x86(AMX)** | N/A | ARM无原生INT4后端 |

### 6.2 针对DeepSeek‑V4‑Pro的硬核建议

> **务必优先选用AmpereOne平台！**
>
> 在**相同GPU配置(双路RTX PRO 6000)** 下，AmpereOne以14 token/s完胜x86的8 token/s，**性能领先75%**。这意味着：
>
> - 两块旗舰显卡在x86上因CPU侧算力不足而被严重拖累，仅发挥出不到60%的潜力；
> - 迁移到AmpereOne后，无需更换GPU即可免费获得近一倍的实际吞吐提升；
> - 从TCO角度看，与其升级更贵的GPU，不如将主机端切换至AmpereOne。

### 6.3 潜在限制提醒

1. **Flash版两者平手**：小MoE(13B激活)下主机调度压力不大，按采购成本/功耗随意选择即可。
2. **ARM的FP8为软解码**：若使用FP8格式，ARM需先解码再喂给GPU，建议优先选用Q8_0、BF16或Q4_K格式的GGUF。
3. **生态成熟度**：x86在GPTQ/AWQ等格式工具链上更成熟，ARM目前以GGUF/LLAMAFILE生态为主，需确认模型是否已转换为GGUF格式。

### 6.4 后续优化方向

- **ARM平台**：针对192核进行NUMA亲和性绑定与投机解码优化，目标将Pro版推至20‑30 token/s。
- **x86平台**：优化MoE专家预取与双路NUMA感知调度，减少CPU→GPU数据准备延迟，力争重返10 token/s以上，避免GPU资源浪费。