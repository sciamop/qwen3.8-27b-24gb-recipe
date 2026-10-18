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

1. **llama.cpp** (b10621, CUDA 13.3): <https://github.com/ggml-org/llama.cpp/releases/download/b10621/llama-b10621-bin-win-cuda-13.3-x64.zip>
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
-c 131072 --flash-attn on
--cache-type-k q8_0 --cache-type-v q8_0
--no-mmproj --parallel 1
-ngl all --no-host
-t 8 -ub 1024
--spec-type draft-mtp
--spec-draft-poll 1
--no-reasoning-preserve
--reasoning-effort medium
--load-mode mmap+mlock
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
--presence-penalty 0.0 --repeat-penalty 1.0
```

| Parameter | Why |
|---|---|
| `--spec-type draft-mtp` | Model's own MTP head as draft; lossless 30–65% faster generation. Default `--spec-draft-n-max 3` is optimal (4/8 were slower). |
| `-ub 1024` | Short-context prefill **+4–5%** (936/964 vs 866/915 tok/s); verified no effect on long-context prefill or decode. |
| `-ngl all` | All 65 layers incl. the MTP head into VRAM (parses to `-2` internally; equivalent to `99` on this model, more robust). |
| `--no-host` | Bypass host buffer, freeing extra device buffers. Verified no regressions. |
| `-t 8` | Explicit 8 CPU threads (leaves CPU headroom for other apps; auto-detect may use all logical cores on other machines). |
| `-c 131072` + KV q8_0 | Speed-first stable combo. 160K/196K contexts are unstable on this GPU (see FAQ). |
| `--spec-draft-poll 1` | Enable async draft polling (hides draft latency). Bool flag (`0`/`1`); active polling level follows `--poll`. |
| `--no-reasoning-preserve` | **Explicitly** drop thinking traces from the history. The template default *keeps* them and opencode always sends `reasoning_content` back, so this flag is the only way to actually drop them — longer sessions, faster incremental prefill. Current-round thinking still streams normally. |
| `--reasoning-effort medium` | Server default thinking depth (was template default `xhigh`). Shorter thinking = less latency per turn. Overridable per request via `reasoning_effort`. |
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

Hardware: RTX A5000 (24564 MiB), Ryzen 7 5800X, DDR4-3600, llama.cpp b10621, `benchmark.ps1`.

| Scenario | Result |
|---|---|
| Prompt eval (2K / 4K, uncached) | 950 / 973 tok/s |
| Generation (short context, temp 1.0) | ~38–40 tok/s (MTP on; variance from acceptance) |
| Generation (temp 0 / greedy) | ~41.7 tok/s (code up to 52.7) |
| Generation @64K context | ~35 tok/s (~62% acceptance) |
| Generation @128K context | ~32 tok/s (~69% acceptance) |
| TTFT (short prompt) | ≈0.3 s |
| TTFT @64K / @128K (true 64K/128K) | ~86 s / ~218 s |
| VRAM | ~23.2 GiB |
| MTP acceptance | temp 0: 55–75%; temp 1.0: 48–62% |

Baseline without MTP: ~32 tok/s. **MTP gain: +30–65%.**

## Vision offload (llambada build)

`start-llambada.ps1` runs the llamAmpere fork (successor of [llama-cpp-qwen-ampere](https://github.com/JakeATX/llama-cpp-qwen-ampere)) (turbo3 V cache) text-only, and `vision-proxy.py` gives it vision by borrowing a small vision model on another machine. Dropping the 27B's own projector (`--no-mmproj`, ~0.9 GB plus its compute buffer) took context from 131K to **168K** (`-c 172032`).

```
clients ──► :8080  vision-proxy.py ── images ──► remote vision model (OpenAI-compatible)
                       │  image parts replaced by its text description
                       ▼
                 127.0.0.1:8079  llama-server (27B, --no-mmproj)
```

- Everything passes through unchanged except chat requests with images: OpenAI `image_url` parts (`/v1/chat/completions`) and Anthropic `image` blocks (`/v1/messages`, including inside `tool_result`). Streaming is relayed as-is.
- Each image is described once, without the conversation, and cached by hash, so a resent history produces identical text and the 27B's prompt cache stays valid.
- If the vision model is unreachable, the image becomes a short "could not be described" note; the request still succeeds.
- `/props` reports `vision: true` so clients that check it will send images.
- **Trade-off:** the 27B reasons over a description, never the pixels. A new image adds the vision model's latency (~10–15 s for a 4B).

The server binds loopback only so nothing on the LAN can bypass the proxy. `local/autostart.ps1` (the `llama-server-qwen38` scheduled task) starts the proxy, then the server; the proxy runs on a Python that has an inbound firewall rule.

| Measured (UD-Q4_K_XL, RTX 4090, temp 1.0, real C++ prompts) | |
|---|---|
| Decode @ ~6K / ~83K context (with vision, before the switch) | 87–94 / 80–83 tok/s |
| 160K-token prompt filled, `-c 172032` | peak 23,738 / 24,564 MiB; prefill 1,243 tok/s; decode 52.7 tok/s |
| First-time image / cached image | 13.9 s / 0.4 s end to end |

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
          "limit": { "context": 131072, "input": 131072, "output": 65536 }
        }
      }
    }
  },
  // compact ~16K tokens before the 128K hard limit (opencode defaults to ~32K headroom)
  "compaction": { "reserved": 16000 },
  // optional: delegate compaction to a cloud model instead of the local GPU
  "agent": {
    "compaction": { "model": "deepseek/deepseek-v4-flash" }
  }
}
```

The server defaults to `--reasoning-effort medium`; it also accepts `reasoning_effort` per request
(`low`/`medium`/`high`/`xhigh`/`none`) and `reasoning_budget_tokens` — verified working via the API.

## Known Issues & Lessons Learned

1. **`-ub 1024` is safe on this card (measured 0.8 GB headroom)** — short-context prefill +4–5%. But `-ub 2048+` with <1.5 GB headroom silently fails CUDA graph capture (~7x slower, no error message).
2. **KV stays `q8_0` — `q4_0`/`iq4_nl` KV is a slow path on qwen35** (attention kernels lack CUDA-graph support), regardless of VRAM headroom — measured 112/66 tok/s even with 2.5 GB free. Earlier diagnosis blaming headroom was wrong.
3. **`ngram-simple` spec decoding was a net loss** here (CPU +27%, VRAM +0.5GB, 128K generation regressed) — MTP alone is better.
4. **MTP head is loaded as "unused tensor" by default** — only `--spec-type draft-mtp` activates it.
5. **`--spec-draft-n-max` 4 and 8 are both worse than default 3** — draft overhead exceeds the longer-chain benefit on freeform text.
6. **Long-context TTFT is the real bottleneck** (~218 s @ true 128K) — prefill is bandwidth-bound; no CPU-side fix exists for dense/hybrid models. (Earlier ~145 s figure came from the old benchmark token estimate of 4.0 chars/token, i.e. only ~93K context.)
7. **`reasoning_effort` is settable per request** — API-side verified (`"none"` disables thinking entirely, 31→0 reasoning chunks).

## License

[MIT](LICENSE) © 2026 Mike Lee

## References

- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Model: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- opencode: <https://opencode.ai>
