# Qwen3.8-27B 在 24GB 显卡上的本地部署（RTX 3090 / 4090 / A5000）

在**任意 24 GB 显存的 NVIDIA 显卡**上用 [llama.cpp](https://github.com/ggml-org/llama.cpp) 本地部署 **Qwen3.8-27B** 的完整配方：一键安装脚本、启动/验证/基准测试脚本，以及全部实测数据。

**实测平台为 RTX A5000，同一套配置适用于其他 24 GB 显卡**（RTX 3090 / 3090 Ti / 4090 / A5000 等）。

> **English version: [README.md](README.md)**

---

## 特性

- **针对混合架构调优** —— 模型为 `qwen35` 混合架构（SSM + 注意力，64 层，约 16 层带 KV），长上下文显存开销仅为传统 27B 密集模型的约 1/4。
- **MTP 投机解码** —— 使用模型自带 MTP 头（`--spec-type draft-mtp`），生成**无损提速 30–65%**。
- **128K 上下文 + KV q8_0** —— 速度优先的稳定组合（160K/196K 已验证不稳定，见 FAQ）。
- **官方采样参数** —— 内置 Qwen3.8-27B 思考模式推荐参数。
- **一键安装** —— `install.ps1` 自动下载引擎与模型，支持断点续传。
- **OpenAI 兼容 API** —— 可直接接入 opencode 及任何 OpenAI 兼容客户端。

## 硬件要求

| 项 | 要求 |
|---|---|
| GPU | 任意 **24 GB 显存**的 NVIDIA 显卡（实测：RTX A5000；RTX 3090 / 3090 Ti / 4090 同样适用） |
| 驱动 | 支持 CUDA 13.x（实测 596.86） |
| 内存 | 最低 32 GB，**推荐 64 GB**（权重 mlock 常驻） |
| 磁盘 | 空闲约 30 GB |
| 系统 | Windows 10/11（PowerShell 5.1+）；其他平台请使用对应 llama.cpp 构建 |

### 显卡差异说明

- **RTX 3090 / 3090 Ti**：与 A5000 相同的内存带宽（936 GB/s）——性能应与下文实测一致。
- **RTX 4090**：带宽 1008 GB/s——会比下文数据略快。
- **A5000**：下文数据即在此卡测得（功耗预算 230W）。
- 其他任意 24 GB 显存、驱动支持 CUDA 13.x 的显卡可直接使用，无需改动。

## 快速开始

### 方式 A：一键安装

```powershell
# 下载本仓库后：
.\install.ps1                     # 引擎 + 模型
.\install.ps1 -SkipModel          # 仅引擎
.\install.ps1 -Proxy http://127.0.0.1:7890   # 需要代理时
```

### 方式 B：手动安装

1. **llama.cpp**（b10566，CUDA 13.3）：<https://github.com/ggml-org/llama.cpp/releases/download/b10566/llama-b10566-bin-win-cuda-13.3-x64.zip>
   - 解压到 `llama-cpp/`
   - 注意：CUDA 12.4 版构建缺少 `cudart64_12.dll`/`cublas64_12.dll`，会静默回退到 CPU——**请使用 CUDA 13.3 版**。
2. **模型**：`Qwen3.8-27B-UD-Q4_K_XL.gguf`（16.35 GB），来自 <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF> → 放入 `models/`

### 启动与验证

```powershell
powershell -ExecutionPolicy Bypass -File .\start-server.ps1
powershell -ExecutionPolicy Bypass -File .\verify-server.ps1
```

服务地址：`http://127.0.0.1:8080/v1`（OpenAI 兼容）。

## 配置说明

`start-server.ps1` 使用以下参数（均已验证）：

```
-c 131072 -fa 1
--cache-type-k q8_0 --cache-type-v q8_0
--no-mmproj --parallel 1
-ngl 99
-t 8
--spec-type draft-mtp
--spec-draft-poll 100
--reasoning-preserve
--load-mode mmap+mlock
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
```

| 参数 | 原因 |
|---|---|
| `--spec-type draft-mtp` | 模型自带 MTP 头作草稿；无损提速 30–65%。默认 `--spec-draft-n-max 3` 最优（8 反而更慢） |
| `-t 8` | 显式 8 个 CPU 线程（给其他应用留 CPU 余量；自动检测在其他机器上可能用满逻辑核） |
| `-c 131072` + KV q8_0 | 速度优先的稳定组合；160K/196K 在此卡上不稳定（见 FAQ） |
| `--spec-draft-poll 100` | 异步草稿轮询，隐藏草稿延迟；想省 CPU 可降到 50 |
| `--reasoning-preserve` | 历史中保留思考痕迹，长会话更连贯 |
| `--load-mode mmap+mlock` | 16.35 GB 权重常驻内存，防止换页重读 |
| 采样参数 | Qwen3.8-27B 官方**思考模式**推荐值（temp 1.0 / top_p 0.95 / top_k 20 / min_p 0） |
| `--temp 1.0` | 注意：温度越低 MTP 接受率越高、速度越快（temp 0 约 42–53 tok/s，temp 1.0 约 38 tok/s） |

### 官方采样参数（Qwen3.8-27B）

| 参数 | 思考模式 | 非思考模式 |
|---|---|---|
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | 1.5 |
| repetition_penalty | 1.0 | 1.0 |

## 性能基准

硬件：RTX A5000（24564 MiB）、Ryzen 7 5800X、DDR4-3600、llama.cpp b10566、`benchmark.ps1`。

| 场景 | 结果 |
|---|---|
| Prompt 处理（2K / 4K 未缓存） | 863–917 tok/s |
| 生成（短上下文，temp 1.0） | 34–42 tok/s（均值 ~38，MTP 开启） |
| 生成（temp 0 贪婪） | ~41.7 tok/s（代码类最高 52.7） |
| 生成 @64K 上下文 | ~28 tok/s |
| 生成 @128K 上下文 | ~26–32 tok/s（波动来自 MTP 接受率） |
| TTFT（短 prompt） | 0.2–0.4 s |
| TTFT @64K / @128K | ~60 s / ~145 s |
| 显存 | 23.3 / 24.6 GiB |
| MTP 接受率 | temp 0：55–75%；temp 1.0：48–62% |

无 MTP 基线：~32 tok/s。**MTP 收益：+30–65%。**

## opencode 接入

```jsonc
// ~/.config/opencode/opencode.jsonc（全局配置）
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B (Local)",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 131072, "output": 65536 }
        }
      }
    }
  }
}
```

服务端还支持按请求传入 `reasoning_effort`（`low`/`medium`/`high`/`xhigh`/`none`）和 `reasoning_budget_tokens`——已通过 API 实测验证。

## 已知问题与经验教训

1. **显存余量 <1.5 GB 时不要调 `-ub`（ubatch）超过 512** —— CUDA graph 捕获会静默失败，速度暴跌约 7 倍且无任何报错。
2. **保持 ≥1.5 GB 显存余量** —— 160K 上下文 / V q4_0 KV / `-ub 2048+` 在此卡上都会触发慢路径。
3. **`ngram-simple` 投机解码在此环境是净损失**（CPU +27%、显存 +0.5 GB、128K 生成退化）——单独 MTP 更好。
4. **MTP 头默认被当作"unused tensor"忽略** —— 只有 `--spec-type draft-mtp` 才会激活它。
5. **`--spec-draft-n-max 8` 比默认 3 更差** —— 自由文本下草稿开销超过长链收益。
6. **长上下文 TTFT 才是真正的瓶颈**（128K 约 145 s）——prefill 受带宽限制，密集/混合模型没有 CPU 侧解法。
7. **`reasoning_effort` 可按请求设置** —— API 侧已验证（`"none"` 完全关闭思考，思考块 31→0）。

## License

[MIT](LICENSE) © 2026 Mike Lee

## 参考

- llama.cpp：<https://github.com/ggml-org/llama.cpp>
- 模型：<https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- opencode：<https://opencode.ai>
