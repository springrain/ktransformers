# Phase-1 计划:benchbw —— KT PCIe/DRAM 带宽归属探针

> 日期:2026-09-13。**派发序 = 1/4**(最先执行)。状态:验证后修订完成,待实施。
> 上游依据:`doc/ft-kt-phase1-audit-hidden-p0.md` §12。对拍纪律:默认逐位等价、不 commit(merge-no-commit)。
> 交付物唯一新增文件:`kt-kernel/bench/bench_bw_kt.py`(主仓,torch-only,不 import sglang/kt_kernel_ext,不改任何生产文件)。

## 0. 定位与派发理由

将 FreeToken `python/freetoken/moe/benchbw.py`(已通读 997 行)移植为 KT 独立探针,回答:**生产每卡 11GB/s vs Gen4 x16 线速 ~22GB/s 的缺口归谁**(host 供给 / PCIe 链路共享 / 协议控制面),直供第二阶段特性排序。

它是 explicit-degrade 的 KV-budget reason、observability 的 unaccounted 归因判读树之**共同上游前提**(critic)——其判读缺陷不修,后两方案归属论据全部悬空,故排第一阶段最先执行。

## 1. 设计(validation 修订后最终版)

单文件探针(~800 行),结构沿用 FT 分层(发现→测量原语→编排→JSON 落盘→报告)。

### P0 机器发现
物理核 = `thread_siblings_list` 去重 ∩ `sched_getaffinity`(移植 FT `cpu_executor.py:92-141` 语义,纯 torch/os/sysfs 重写);NUMA = sysfs cpulist;**PCIe 代际/宽度探测(`nvidia-smi -q` 或 `lspci` LnkSta Gen/width)→ 写入 JSON meta**(critic Q4-近零:关闭 open_question#3 —— RTX PRO 6000 Blackwell 为 Gen5 卡,硬编码 Gen4 带 [20,25]GB/s 有 rig-invalid 风险);GPU↔NUMA 经 PCI BDF→`numa_node`。非 Linux/无 CUDA → loud exit;**单卡/无双卡时 dual 模式显式 loud-skip 并写 JSON skip 原因**(verify P8)。

### P1 host pinned 带宽 rig(复现 rank0 写双 rank shm,修订核心)
- **生产形态是写,不是读**(critic 升 P0 + verify P1):`kt_ep_wrapper.py:2240-2241` 注释 "Rank 0 writes every rank's SHM"、`:2266` 仅 rank0 调 `_submit_host_write`、`:2075-2085` 把 peer 指针交 KT writer 写入。双路 ARM 跨 socket 写带宽通常为同向读 50-70%,用读数判归属可能翻转整个 (a)/(b) verdict。
- **测量矩阵修订为 {read, write} × {local, remote} × {1, 2, 4, 8, 16} 线程**,写腿用 pinned `fill_`(与 FT 读腿同 buffer/同计时骨架,~+0.3 人日);**verdict (a) 的 host 供给判据改用写腿数字**,读腿保留作 DRAM 上限参考。
- **shm 生命周期握手(verify P1 修正)**:子进程 create+first-touch(`sched_setaffinity` 钉 NUMA,:840-845 语义)+`cudaHostRegister`(:891 形态)→ **post Event A**;父进程 open 同名段(:943-948 形态)→ **post Event B**;子进程 unlink 保 mmap(对齐生产 `:931-960`→`:923-929` 真实顺序)。POSIX 上先 unlink 再 open 必 FileNotFoundError,原版顺序做反了。

### P2 PCIe H2D gather
- 19,161,088 B/专家 × N(N 扫 1/8/24/64/232)四 bank 切分拷贝复现 `:2281-2284`;CUDA events 计时 + `torch.cuda._sleep`(FT :433-434)防 CPU 提前入队;solo0/solo1/dual(mp.Barrier 区外对齐,计时区零集合通信);NUMA 放置变体(node0/node1/交错);per-copy 粒度扫描(KB→19MB)量化控制面下限。
- **BANK_SPLIT 文档修正(verify P6)**:生产 scale 被强制 bfloat16(`:868-872`),同维度生产字节比应为 **32:4:16:2**;默认标签改为"近似生产粒度,非生产精确切分",初始化时断言 Σrounded_banks == EXPERT_BYTES(19,161,088 不能被 51 整除,per-bank 凑整有恒等式兜底)。切分不进判读阈值。
- **判定树去 Gen4 硬编码(verify P3)**:校准带 = **[0.75, 1.10] × 理论线速(按 P0 探测的 LnkSta 推导;探测失败回退 [20,25] 并标 `unknown-gen`)**;verdict (a)/(b) 用 **dual_sum/solo 比值域(≤1.15 共享 / ≥1.7 独立 / 中间 inconclusive)**替代绝对 22/40 阈值。
- **可选 contended 腿(verify P4)**:`SGLANG_KT_BENCHBW_CONTENDED=1`(默认关)时 P2 dual 测量窗口内并发跑 P3 满核 pinned 读工作集作 DRAM 背景负载(复现 FT `measure_overlap_bw:543` "standalone numbers cannot predict this split" 体制 —— 生产 11GB/s 是 168 核 NEON GEMV 与双卡 DMA 同在 DRAM 上跑出来的);verdict (b) 要求 idle 与 contended 两套数字,差值写入 verdict 支撑字段。

### P3 168 核 CPU 聚合读带宽
移植 FT `measure_cpu_mem_bw:184-253`,膝点扫描 {40,80,120,168}。**与 `bench_moe_neon_perf.py:1-40` 的交叉对拍降级为 reference-only 字段**(verify P7:方法学不对齐 —— pinned 单线程 float32 sum vs unpinned bf16 4GiB,失败只是误报),不进 rig-invalid 硬门槛。

### 启动护栏(verify P5)
脚本启动查双卡 `memory.used`(nvidia-smi / `torch.cuda.mem_get_info`)与进程表,显存占用超阈值或检出 sglang 进程即**拒绝运行**,除非 `SGLANG_KT_BENCHBW_FORCE=1`(与 config-first 相称;生产同机误跑会抢 DRAM/PCIe + cudaHostRegister page-lock 压力扰动 graph 捕获期)。

### 判定树(终版)
- **(a)** dual_sum/solo ≈1(≤1.15)→ 上游共享。查 P1 写腿:跨 NUMA 供给 ≈11/卡 → 归属 host 供给(二期优先 NUMA 感知 bank 放置);写腿供给 ≫ → 归属 PCIe 链路/RC 共享(带宽类收益上限锁死,控制面/CPU 侧特性上调)。
- **(b)** dual_sum/solo ≥1.7 → 硬件可达双线速 → 生产缺口归协议控制面(`:2253/:2273` 每专家 2 次设备共识、`:1324` barrier、`:2042-2050` all_reduce+.item()),二期把 E_chunk 批拷贝/fence 流水/共识合并排最高;contended 差值写入支撑字段。
- **(c)** solo 出校准带 → rig-invalid,先修 rig(分配路径/IOMMU/NUMA 误标),不出判读。
- **(d)** P3 膝点 x:x≫22 CPU 不吃亏;x≈22 两侧同顶,热更新窗口形状特性优先。

## 2. env 门(探针一次性脚本,main() 运行期读,写入 JSON config 块)

| env | 默认 | 说明 |
|---|---|---|
| `SGLANG_KT_BENCHBW_EXPERT_BYTES` | `19161088` | 字节账恒等式基准 |
| `SGLANG_KT_BENCHBW_BANK_SPLIT` | `32:4:16:2` | 近似生产粒度(bf16 scale);uint8 scale 口径可选 |
| `SGLANG_KT_BENCHBW_EXPERT_COUNTS` | `1,8,24,64,232` | N 扫描 |
| `SGLANG_KT_BENCHBW_GPUS` | `0,1` | |
| `SGLANG_KT_BENCHBW_THREADS_SWEEP` | `1,2,4,8,16` | P1 用 |
| `SGLANG_KT_BENCHBW_OUT` | `./bench_bw_kt_<host>_<epoch>.json` | 原子落盘(FT :682-688 移植) |
| `SGLANG_KT_BENCHBW_QUICK` | `0` | `1`=iters/缓冲减半,<30min 兜底 |
| `SGLANG_KT_BENCHBW_ONLY` | 空=全部 | `p1`/`p2`/`p3` 子集 |
| `SGLANG_KT_BENCHBW_PIN` | `shm-register` | `alloc`=cudaHostAlloc;`off`=负注入(页换源,预期塌缩) |
| `SGLANG_KT_BENCHBW_CONTENDED` | `0` | `1`=GEMV∥DMA 争用腿 |
| `SGLANG_KT_BENCHBW_FORCE` | `0` | 越过启动护栏 |

等价性:不改任何生产文件(已 grep 实证无生产代码 import `kt-kernel/bench/*.py`),服务器进程指令流逐字节不变;脚本对 environ 只 get 不 set。

## 3. 实施步骤

1. 骨架:argparse+env 解析+原子 JSON writer+meta(git/dirty,照 `bench_write_buffer.py:50-60` 体例)。
2. 机器发现:物理核/NUMA/**PCIe LnkSta Gen×width→meta**/GPU→numa_node;单卡 dual 腿 loud-skip。
3. 启动护栏:显存/进程检查 + FORCE 逃生阀。
4. P1 rig:shm Event 握手(A→open→B→unlink);**{read,write}×{local,remote}×threads** 矩阵;写腿入主判据。
5. P2 rig:校准(LnkSta 相对带)→ 逐 N chunk copy + events + `_sleep`;dual Barrier 对齐;NUMA 变体;per-copy 扫描;contended 分支。
6. P3:FT 移植 + 膝点;交叉对拍 reference-only。
7. 判定树:比值域判读 + 支撑数字;rig-invalid 时只出诊断。
8. 自校验:字节账恒等(Σcopy == N×19,161,088,四 bank 恒等,同 `audit.md:86` 口径);3 次重复 CV<5%;event vs perf_counter 偏差>10% 标 skew;退出前 assert 未 init torch.distributed。
9. 本机 win32 无法演练:全部 Linux 依赖(sysfs/sched_setaffinity/shared_memory/cudart)try+清晰报错;xysa10 联调留档(G1)。

## 4. 对拍(=测量保真度)

- 字节账整数恒等;校准带 LnkSta 相对;CV<5%;双计时偏差<10%。
- **负注入(内置,env 触发)**:(a) `PIN=off` 页换源 → H2D 必须塌缩(<5GB/s);(b) peer shm 尺寸减半 → 字节账/stride 断言必须在计时**前**炸出(scale 区域减半分配越界教训);(c) 子进程起测前被杀 → 父进程 barrier timeout(120s)内非零退出不挂死(生产有 watchdog,探针不得掩盖)。
- **交叉对拍**:P3 vs `bench_moe_neon_perf.py` reference-only;P2 单卡 gather vs 生产 `SGLANG_KT_PREFILL_STATS` 窗口纯拷贝分项 —— "生产每专家周期 − 探针纯拷贝" = 控制面开销估计,供判读 (b) 定量。

## 5. rollout

- **G0** 开发机(Linux+CUDA)`--quick` 全程;退出=三探针出数或 loud-skip、JSON 合法、solo 校准在带内。
- **G1** xysa10 停服窗口全量首跑(<30min);退出=双卡 solo gather 均在 LnkSta 相对带、字节账全绿、P3 膝点出数;校准出带即停手修 rig,不产判读。
- **G2** xysa10 全矩阵(dual × NUMA × N,含一次 CONTENDED=1);退出=归入 (a)/(b) 之一且支撑数字齐,verdict JSON 交付二期排序。
- **G3** 常驻:拓扑/驱动变更后或二期重排前复跑;与基线漂移 <10%。全程与服务器互斥;探针失败只损失一次测量,生产零影响。

## 6. 热更新交互 / collectives / CUDA graph(核销)

- 独立进程,从不 import kt_ep_wrapper;`:2473-2491` 更新窗口、`:2253-2275` 共识、`:2042-2050` all_reduce、`:1323-1324` barrier、`:4504-4513/:4689-4732` serial 热更新 —— **全无路径交叉**(rollout 互斥挡硬件并发)。集合通信数=0(不 init 任何 process group,协调全走 localhost mp.Barrier/Event)。
- **补核销(verify P9)**:`:4787-4794` decode 路径 CUDA-graph 安全 in-place 更新窗 —— 无交互,同 `:2474` 理由(独立进程零路径交叉)。
- CUDA graph:零影响;约束唯一 = 不得在服务器捕获/重放期同机跑(护栏已挡)。

## 7. 工时

约 **3.5 人日**(开发 3.0:骨架/发现/JSON 0.75;P1 shm rig+**写腿** 1.3;P2 gather+dual+contended 1.0;P3 移植 0.25;判定树 0.25)+ xysa10 联调校准 0.5;区间 2.5-4。

## 8. 对抗验证记录(verify refuted=False,8 问题全部吸收)

| # | 问题 | 处置 |
|---|---|---|
| P1 | 测读、生产是跨 NUMA 写(critic 升 P0) | §1.P1 加写腿,verdict (a) 改用写数 |
| P1 | shm 顺序做反(unlink 先于 open 必失败) | §1.P1 mp.Event 握手,对齐 `:931-960→:923-929` |
| P3 | Gen4 阈值硬编码(Gen5 卡风险) | §1.P2 LnkSta 相对校准带 + dual_sum/solo 比值域 |
| P4 | GEMV∥DMA 争用体制被丢弃 | CONTENDED=1 可选腿 |
| P5 | 无防呆护栏,误并发生产 | §1 启动护栏 + FORCE 阀 |
| P6 | BANK_SPLIT 32:2:16:1 与生产 bf16 scale 不符 | 改 32:4:16:2 + 恒等式断言(进不了判读阈值,doc 修正) |
| P7 | P3 交叉对拍方法学不对齐 | 降级 reference-only |
| P8 | G0 单卡无路标;`:4787-4794` 核销遗漏 | loud-skip + §6 补核销 |

critic 另定:PCIe gen/width 探测写入 JSON meta(零成本关闭 open_question#3);**派发序提前至第一**(本计划即承接)。

## 9. 主控裁定项

1. 四 bank 精确切分未闭合(19,161,088 = 2^13×2339 反查不整;按 config 4096/2048 uint8 口径算 13,369,344、按 `:867-873` bf16 shm 口径算 14,155,776,均≠)——按总额+比例参数化,不阻塞;请拿 xysa10 checkpoint 实参定稿。
2. JSON 落盘目录约定(建议 `doc/bench/` 按日期,verdict 块字段直供二期排序)。
3. contended 观测默认严禁并发生产;如需生产争用观测另立窗口。
4. P4 全复现档(rank0 跨写 shm + 双卡同拷一体)默认不做,列二期备用。
