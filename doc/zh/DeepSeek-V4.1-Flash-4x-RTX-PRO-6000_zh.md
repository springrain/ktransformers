# 在 4× RTX PRO 6000 上运行 DeepSeek-V4.1-Flash

本文复现社区启动器
[0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000)
的服务配置，基于 KTransformers 的 SGLang fork。attention/MoE 内核以及
SM120 prefill 页拆分修复已合入本仓库，本文只迁移启动参数与验收检查。

## 硬件要求

- **GPU**：4× NVIDIA RTX PRO 6000 Blackwell（96GB，SM120），无需 NVLink
- **内存**：建议 ≥ 256 GB。FP8 Engram 表（约 189 GiB）通过
  `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` 放置在主机内存；shared 布局下
  四个 TP rank 共享一份物理拷贝。
- **存储**： checkpoint 约 510 GB。

模型：`deepseek-ai/DeepSeek-V4.1-Flash`，已验证的上游 revision 为
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`。

## 启动

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared   # 默认值，四卡共享一份

python -m sglang.launch_server \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --trust-remote-code \
  --load-format safetensors \
  --tp 4 \
  --ep-size 4 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --context-length 409600 \
  --max-running-requests 8 \
  --cuda-graph-max-bs-decode 8 \
  --min-free-slots-delay 1 \
  --random-seed 0 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-decoder-swa-bounded-replay \
  --tool-call-parser deepseekv41 \
  --reasoning-parser deepseek-v41 \
  --host 0.0.0.0 \
  --port 8010
```

参数说明：

- **自动生效，无需显式传入**：`--attention-backend dsv4`、
  `--moe-runner-backend flashinfer_mxfp4`、`--page-size 256` 由
  `DeepseekV4ForCausalLM` 的 model override 在 SM120 + FP4 路由专家的
  checkpoint 上自动选择。
- **DSPARK** 使用 checkpoint 内置的 draft head（`dspark_*` 配置键），
  无需单独的 draft 模型。block size 5 与上游实测配置一致。
- **prefill CUDA graph** 会按 capture-pool 内存压力规则自动关闭，
  decode/verification 图保持开启。
- **上下文长度**：可接受范围为 400,000 至模型上限 1,048,576。上游实测的
  最大容量配置为 `MEMORY_FRACTION=0.95`、`MAX_TOTAL_TOKENS=4200000`、
  `CONTEXT_LENGTH=524288`，达到了 4,063,744 个已填充 KV token。
- **KT CPU 专家卸载**（`--kt-method MXFP4 --kt-cpuinfer ...`）在本机非必需：
  计算权重已可放入 4×96 GB 显存。仅在想用内存/CPU 换取更多 KV 缓存显存时
  再启用。

## 验收检查

等效于启动器的 smoke 测试，在 `/health` 就绪后执行：

```bash
# 1. 算术（新鲜推理）
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","temperature":0,
       "chat_template_kwargs":{"thinking":false},
       "messages":[{"role":"user","content":"What is 19 + 23? Reply only with the number."}]}'
# 期望 content 为 "42"，finish_reason 为 "stop"

# 2. 结构化输出（JSON schema）
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","temperature":0,
       "chat_template_kwargs":{"thinking":false},
       "messages":[{"role":"user","content":"Return an object whose answer is the integer 42."}],
       "response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,
         "schema":{"type":"object","properties":{"answer":{"type":"integer"}},
         "required":["answer"],"additionalProperties":false}}}}'
# 期望 {"answer": 42}

# 3. 工具调用往返：声明一个函数，将其结果回填后确认最终回答使用了该结果。
```

上游测试套件还验证了工具调用往返，以及按 checkpoint 满额
1,024 image tokens/图 的原生图像请求。

## 未迁移项（有意为之）

- **checkpoint 下载 + SHA256/git-blob 校验**（boot.py 的 `prepare`）：
  属于发行工具；用 `huggingface-cli` 一次性下载即可。
- **API key 文件管理**：需要时直接使用 `--api-key`。
- **NVMe Engram 行存储**（`adapter/row_store.cpp`）：仅适用于内存装不下
  ~189 GiB Engram 表的主机；本文假设内存充足，使用 fork 原生的
  host-table 模式。
- **`sitecustomize.py` 的 SM120 indexer 猴子补丁**：针对更新版上游行为
  （ratio-1/2 indexer 总是走 FP4 DeepGEMM kernel）。本 fork 中 FP4 indexer
  由 `--enable-deepseek-v4-fp4-indexer` 显式开启，且开启后 backend 已自动
  强制使用正确的 SM120 metadata planner。
- **Docker 打包**：复用 KTransformers 镜像或源码安装即可。