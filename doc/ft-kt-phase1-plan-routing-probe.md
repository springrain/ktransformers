# Phase-1 计划:routing-probe —— **BLOCKED(移出第一阶段)**

> 日期:2026-09-13。状态:**verify 裁定 refuted=true,critic 终审确认移出第一阶段并标 blocked**。
> 本文档只作留存:否决理由 + 可施救包(供后续阶段启用)+ 数据需求接替指向。
> 上游依据:`doc/ft-kt-phase1-audit-hidden-p0.md` §12。**严禁**按原计划新建任何 `scripts/` 文件 —— blocked 的意义即阻止这份顺手创建。

## 0. 一句话

原拟 `SGLANG_KT_ROUTING_PROBE` 在每 chunk 统计各专家 top-k 路由命中直方,回答 lru-hit-d2d vs 迟滞之争;verify 在对拍意图与注解逐条亲验后裁定 **refuted=true**(P0/P1 两条硬伤,P2-P6 六条次伤),critic 终审背书"移出第一阶段并标 blocked,数据需求由 observability churn 搭车接替"。

## 1. 否决理由(refuted 全案)

### P0(致命):首处 `aten::nonzero` → `memcpy_and_sync` → `cudaErrorStreamCaptureUnsupported`
- 设计核心步是"对 token 级选中做 boolean mask 过滤再累计命中"。boolean mask 索引必走 `aten::nonzero`,后者内部 `at::cuda::memcpy_and_sync` 强制 D2H 同步。
- 挂载点 `:4343-4344` **正处 CUDA Graph 捕获路径**:首个 decode 捕获即抛 `cudaErrorStreamCaptureUnsupported`。`:3440-3446` 的先例是 eager-only(且躲在 `:2473` 同步后),不能作为捕获安全证据。
- 结论:**只要开探针,首次捕获即崩**,无任何 env 默认值可免。

### P1(致命):`tokens.add_(valid.numel())` 把捕获常量烘进图
- `numel()` 在捕获期被求值并固化为常量;replay 期真实 token 数变化时该字段返回历史捕获值 → 统计口径必错。
- 与 P0 叠加,该设计"既能炸 capture 又能在不炸的 replay 下静默统计错数"。

### P2:round 计数语义错
- abort 中途幂等(`:1931-1939` 可被调多次),真"轮翻转"只在 `:1955` 生效;原案按调用次数累加会虚增。

### P3:dump 路径是死分支
- `:4430` 需要 manager 已建;非 MXFP4 或低于 threshold 的部署永远没有 dump 出口。

### P4:"双 rank 相等"被夸大
- `:4714-4719` rank0-only 统计 + broadcast `:4727-4732` 是单向,rank1 视角数字由 rank0 决定;"双 rank 各自独立验证一致性"的论断不成立。

### P5:atexit 兜底失控
- 探针若失败,atexit 回调在熄火路径仍执行,可能向已半关的 CUDA 资源发问。

### P6:档2 成功标准本末倒置
- 原案档2 首条是"直方相关",verify 明正:**首条必须是 capture 成功**,直方是否可解释在其次。

## 2. 可施救包(供后续阶段启用,非本阶段承诺)

任何打捞版必须先过 P0/P1 两道:

```python
# 完全静态形状的 clamp-bucket 替代 boolean mask + numel
safe = flat_ids.clamp_(0, E)                                   # 无 nonzero、无同步
delta = torch.zeros(E+1, dtype=torch.int64, device=dev)
delta.index_add_(0, safe, torch.ones_like(safe, dtype=torch.int64))
counts.add_(delta[:E])
tokens.add_(delta[:E].sum())                                   # int 加法,非 numel()
```

配套修订:
- 轮计数 = 观测 `round_active` True→False 翻转(`:4430` 附近)一次,幂等 abort 不再虚增;
- parity 前置 = manager 已初始化再激活;
- 双 rank 各自 per-rank local reference bincount 对照(不在 broadcast 后);
- atexit 挂 `_PROBE_ON` 且 fail-soft(任何异常吞掉只打 warn-once);
- 档2 首条改为 **capture 成功**,统计正确性列第二。

## 3. 数据需求接替(第一阶段已落地)

lru-hit-d2d vs 迟滞之争原由 routing-probe 供数,**改由 observability 计划在 `:4791-4792` 增设的 256 位 host XOR+sum Hamming churn 直方接替**(详见 `ft-kt-phase1-plan-observability.md` §1.4):
- 位置在 `:2473 synchronize` 之后、捕获路径之外,rank0-only、零 CUDA op,天然 capture-safe;
- 挂在 `SGLANG_KT_PREFILL_STATS` 账本下,无新增 env;
- 离线配合字节账回放即可反推 oracle 槽曲线。

## 4. 后续何时复活

只在以下全部成立时再议:
1. observability churn 数据到 case,且确实需要 token 级命中直方补充;
2. 有独立的方案六(owner = cluster rails 的承载者)接手;
3. P0/P1 的替代实现(clamp-bucket)先在最小 sandbox 过 capture + replay 双轨;
4. 主控书面批准解除 blocked。

在那之前,**严禁任何路线触碰 `:4343-4344` boolean mask 索引与 `numel()` 入 add_**,严禁新建 `scripts/parity*`、`scripts/atexit*` 等文件。
