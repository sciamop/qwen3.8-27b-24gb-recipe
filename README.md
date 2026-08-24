# Qwen3.8-27B on a 24GB GPU (RTX 3090 / 4090 / A5000)

A complete, battle-tested recipe for running **Qwen3.8-27B** locally on **any 24 GB NVIDIA GPU** with [llama.cpp](https://github.com/ggml-org/llama.cpp): one-click installer, start/verify/benchmark scripts, and real measured numbers.

**Tested on an RTX A5000; the same setup applies to other 24 GB cards** (RTX 3090 / 3090 Ti / 4090 / A5000, etc.).

> **中文版：[README-zh.md](README-zh.md)**

---

## Features

- **Hybrid architecture tuned** — the model is a `qwen35` hybrid (SSM + attention, 64 layers, ~16 attention layers), so long-context VRAM cost is ~1/4 of a dense 27B.
- **MTP speculative decoding** — uses the model's built-in MTP head (`--spec-type draft-mtp`) for a **lossless 30–65% generation speedup**.
- **128K context, KV q8_0** — the speed-first stable configuration (160K/196K proven unstable, see FAQ).
- **Official sampling presets** — Qwen3.8-27B thinking-mode parameters baked in.
- **One-click install** — `install.ps1` downloads engine + model with resume support.
- **OpenAI-compatible API** — works with opencode and any OpenAI-compatible client.

## Requirements

| Item | Requirement |
|---|---|
| GPU | Any NVIDIA GPU with **24 GB VRAM** (tested: RTX A5000; RTX 3090 / 3090 Ti / 4090 also work) |
| Driver | CUDA 13.x capable (tested: 596.86) |
| RAM | 32 GB minimum, **64 GB recommended** (model is pinned with mlock) |
| Disk | ~30 GB free |
| OS | Windows 10/11 (PowerShell 5.1+); other platforms: use the corresponding llama.cpp build |

### Card-specific notes

- **RTX 3090 / 3090 Ti**: same 936 GB/s memory bandwidth as the A5000 — expect identical performance.
- **RTX 4090**: 1008 GB/s bandwidth — slightly faster than the numbers below.
- **A5000**: results below were measured on this card (230W power budget).
- Any other 24 GB card with a CUDA 13.x-capable driver should work unchanged.

## Quick Start

### Option A: One-click install

```powershell
# download the repository, then:
.\install.ps1                     # engine + model
.\install.ps1 -SkipModel          # engine only
.\install.ps1 -Proxy http://127.0.0.1:7890   # if a proxy is required
```

### Option B: Manual

1. **llama.cpp** (b10566, CUDA 13.3): <https://github.com/ggml-org/llama.cpp/releases/download/b10566/llama-b10566-bin-win-cuda-13.3-x64.zip>
   - extract into `llama-cpp/`
   - Note: the CUDA 12.4 build ships without `cudart64_12.dll`/`cublas64_12.dll` and silently falls back to CPU — use the **CUDA 13.3** build.
2. **Model**: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (16.35 GB) from <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF> → place in `models/`

### Start & verify

```powershell
powershell -ExecutionPolicy Bypass -File .\start-server.ps1
powershell -ExecutionPolicy Bypass -File .\verify-server.ps1
```

Server endpoint: `http://127.0.0.1:8080/v1` (OpenAI-compatible).

## Configuration

`start-server.ps1` runs with the following (all parameters validated):

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

| Parameter | Why |
|---|---|
| `--spec-type draft-mtp` | Model's own MTP head as draft; lossless 30–65% faster generation. Default `--spec-draft-n-max 3` is optimal (8 was slower). |
| `-t 8` | Explicit 8 CPU threads (leaves CPU headroom for other apps; auto-detect may use all logical cores on other machines). |
| `-c 131072` + KV q8_0 | Speed-first stable combo. 160K/196K contexts are unstable on this GPU (see FAQ). |
| `--spec-draft-poll 100` | Async draft polling; hides draft latency. Lower to 50 to save CPU. |
| `--reasoning-preserve` | Keep thinking traces in history for better long-session continuity. |
| `--load-mode mmap+mlock` | Pin 16.35 GB weights in RAM (no page-out re-reads). |
| Sampling | Official Qwen3.8-27B **thinking mode** values (temp 1.0 / top_p 0.95 / top_k 20 / min_p 0). |
| `--temp 1.0` | Note: lower temperature = higher MTP acceptance = faster (temp 0 → ~42–53 tok/s vs ~38 @ temp 1.0). |

### Official sampling presets (Qwen3.8-27B)

| Parameter | Thinking Mode | Non-thinking |
|---|---|---|
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | 1.5 |
| repetition_penalty | 1.0 | 1.0 |

## Benchmark

Hardware: RTX A5000 (24564 MiB), Ryzen 7 5800X, DDR4-3600, llama.cpp b10566, `benchmark.ps1`.

| Scenario | Result |
|---|---|
| Prompt eval (2K / 4K, uncached) | 863–917 tok/s |
| Generation (short context, temp 1.0) | 34–42 tok/s (avg ~38, MTP on) |
| Generation (temp 0 / greedy) | ~41.7 tok/s (code up to 52.7) |
| Generation @64K context | ~28 tok/s |
| Generation @128K context | ~26–32 tok/s (variance from MTP acceptance) |
| TTFT (short prompt) | 0.2–0.4 s |
| TTFT @64K / @128K | ~60 s / ~145 s |
| VRAM | 23.3 / 24.6 GiB |
| MTP acceptance | temp 0: 55–75%; temp 1.0: 48–62% |

Baseline without MTP: ~32 tok/s. **MTP gain: +30–65%.**

## opencode Integration

```jsonc
// ~/.config/opencode/opencode.jsonc (global config)
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

The server also accepts `reasoning_effort` per request (`low`/`medium`/`high`/`xhigh`/`none`),
and `reasoning_budget_tokens` — verified working via the API.

## Known Issues & Lessons Learned

1. **Do NOT raise `-ub` (ubatch) above 512** with <1.5 GB VRAM headroom — CUDA graph capture silently fails and speed collapses ~7x with no error message.
2. **Keep ≥1.5 GB VRAM headroom** — 160K context / V q4_0 KV / `-ub 2048+` all triggered the slow path on this card.
3. **`ngram-simple` spec decoding was a net loss** here (CPU +27%, VRAM +0.5GB, 128K generation regressed) — MTP alone is better.
4. **MTP head is loaded as "unused tensor" by default** — only `--spec-type draft-mtp` activates it.
5. **`--spec-draft-n-max 8` is worse than default 3** — draft overhead exceeds the longer-chain benefit on freeform text.
6. **Long-context TTFT is the real bottleneck** (~145 s @128K) — prefill is bandwidth-bound; no CPU-side fix exists for dense/hybrid models.
7. **`reasoning_effort` is settable per request** — API-side verified (`"none"` disables thinking entirely, 31→0 reasoning chunks).

## License

[MIT](LICENSE) © 2026 Mike Lee

## References

- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Model: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- opencode: <https://opencode.ai>
