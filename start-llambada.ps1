$ErrorActionPreference = "Stop"

# start-llambada.ps1 -- start the llamAmpere build ("llambada") on port 8080.
#
# Sibling of start-server.ps1 (the stock llama.cpp recipe). Wrap it with
# local\autostart.ps1 the same way and it inherits the log rotation and the
# cmd.exe-level stderr redirect that wrapper exists to provide -- llama.cpp logs
# to stderr, and PowerShell-side redirection turns that into ErrorRecords that
# ErrorActionPreference=Stop kills the server with.
#
# Differences from the stock recipe, none of them cosmetic:
#
#   -ctv turbo3      Value cache at ~3 bits. This is the fork's whole reason for
#                    existing at long context -- it is what fits >48K on this
#                    card. Stock ran q8_0/q8_0 and capped at 49152.
#   -rea auto        The fork has NO --reasoning-effort. Its equivalent is
#                    -rea [on|off|auto]. Stock ran "medium"; the closest this
#                    build offers is auto (template default). If Qwen3.8's
#                    thinking latency bites in Claude Code, change to -rea off.
#   --temp 1.0       The fork is tuned at temperature 1.0 -- every published
#                    llamAmpere number assumes it. Stock ran 0.0 because greedy
#                    raises draft acceptance. For the stock's speed bias back,
#                    change this one value to 0.0.
#   --sleep-idle-seconds 180
#                    Unload after 3 minutes idle (stock used 300). Sleeping is
#                    not a suspend: server-context.cpp handle_sleeping_state()
#                    calls destroy(), freeing model, KV and the vision tower, so
#                    the ~23 GB goes back to the GPU; the next request calls
#                    load_model() again and blocks until it finishes. Measured
#                    2026-09-20 on this box, idle -> slept at 181s:
#                      VRAM      22173 -> 1572 MiB
#                      host RAM  15.8  -> 0.8 GB working set (mlock is released
#                                with the mapping, so it does not pin anything
#                                while asleep)
#                      wake      12.8s end-to-end, ~10s of it the reload. A cold
#                                start from a stopped server was 26s.
#                    Two costs to know about:
#                      - that 12.8s is paid by whichever request arrives first
#                      - the RAM prompt cache does NOT survive: load_model()
#                        constructs a fresh server_prompt_cache, so a woken
#                        server re-prefills a long prompt from scratch, on top
#                        of the reload. Add --cache-disk-path to keep it across
#                        sleeps; that tier is rescanned from disk on
#                        construction (server-task.h, disk_scan()).
#                    Set to -1 to disable if a pause mid-session costs more than
#                    the idle VRAM is worth.
#   -c 172032        168K, up from stock's 49152. Measured 2026-10-18 with a
#                    160K-token prompt filled and decoded: whole-card peak
#                    23,738 of 24,564 MiB with the desktop up. The next 8K step
#                    costs ~210 MB; this is the same margin the old 131K+vision
#                    config ran on for weeks.
#   --no-mmproj      Vision is NOT loaded here. Images are handled by
#                    vision-proxy.py, which owns the public port 8080, sends each
#                    image to the 4B vision model on the vision box, and hands this server a
#                    text description instead. Dropping the projector (~0.9 GB
#                    plus its compute buffer) is what bought 131K -> 168K.
#   --host 127.0.0.1 --port 8079
#                    Loopback only, so nothing on the LAN can bypass the proxy
#                    and send image parts this server cannot read.
#
# --fit is deliberately NOT passed, so the fork's default applies: ON (its
# common.h sets fit_params = true), i.e. it may adjust unset arguments to fit
# device memory. On this recipe it prints
#   fit could not prove a viable placement; restoring the pre-fit parameters
# and then starts on the parameters below anyway. The line is harmless, but it
# is the planner reporting that this config sits at the edge of 24 GB.
# llambada.bat passes --fit off for a deterministic load. Leaving it on here
# keeps a rescue path if a larger -c pushes past the edge.
#
# The exe stays in the build tree because the fork was linked with the CUDA
# runtime DYNAMICALLY: llama-server.exe needs cudart64_12.dll plus the ggml DLLs
# beside it. To fold it into the recipe the way stock does, rebuild with
# -DBUILD_SHARED_LIBS=OFF first, then copy a single self-contained exe.

$scriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$forgeDir     = Join-Path $env:USERPROFILE "llamAmpere"
$serverExe    = Join-Path $forgeDir  "build-sm86\bin\llama-server.exe"
$vocabMap     = Join-Path $forgeDir  "docs\mtp-vocab\atx_65536.txt"
$modelFile    = Join-Path $scriptDir "models\Qwen3.8-27B-UD-Q4_K_XL.gguf"
$chatTemplate = Join-Path $scriptDir "qwen3.8-chat-template.jinja"

if (!(Test-Path $serverExe))    { Write-Error "llambada llama-server.exe not found: $serverExe"; exit 1 }
if (!(Test-Path $modelFile))    { Write-Error "Model file not found: $modelFile"; exit 1 }
if (!(Test-Path $chatTemplate)) { Write-Error "Chat template not found: $chatTemplate"; exit 1 }
if (!(Test-Path $vocabMap))     { Write-Error "MTP vocab map not found: $vocabMap"; exit 1 }

$existing = Get-NetTCPConnection -LocalPort 8079 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Error "Port 8079 is already in use (PID $($existing.OwningProcess)). Stop the running llama-server first."
    exit 1
}

# The CUDA runtime is linked dynamically, so cudart64_12.dll must be on PATH at
# run time. With CUDA 11.8 also installed the system PATH generally points at
# 11.8's bin, which carries cudart64_110.dll and not ..._12. Prepend 12.9.
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:CUDA_PATH = $cuda
$env:PATH = "$cuda\bin;$env:PATH"

# Worth 13% on the fork's own measurement, and the flag everyone forgets.
$env:GGML_Q8_TURBO3_MMA_FUSED = "1"

# llama-server demands bearer auth if LLAMA_API_KEY is set, and an inherited
# value also breaks Ollama's bundled llama.cpp runner. Clear it for this process.
$env:LLAMA_API_KEY = $null

Write-Host "Starting llambada (llamAmpere)..." -ForegroundColor Cyan
Write-Host "  Model:   $modelFile" -ForegroundColor Gray
Write-Host "  Context: 172032, K q8_0 / V turbo3, FA on, MTP-3 + 64K vocab shortlist" -ForegroundColor Gray
Write-Host "  Idle:    unloads after 180s, reloads on next request" -ForegroundColor Gray
Write-Host "  Vision:  none here -- vision-proxy.py on :8080 describes images via the vision box" -ForegroundColor Gray
Write-Host "  Listen:  127.0.0.1:8079 (behind vision-proxy.py)" -ForegroundColor Gray
Write-Host "  Measured: ~75 tok/s on real code at 123K context" -ForegroundColor Gray

& $serverExe `
  -m $modelFile `
  --alias llambada --alias qwen3.8-27b `
  --chat-template-file $chatTemplate `
  -c 172032 --flash-attn on `
  --cache-type-k q8_0 --cache-type-v turbo3 `
  --no-mmproj --parallel 1 `
  -ngl 99 --no-host `
  -t 8 -ub 1024 `
  --spec-type draft-mtp `
  --spec-draft-n-max 3 --spec-draft-p-min 0 `
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 `
  --spec-draft-vocab-map $vocabMap `
  --spec-draft-poll 1 `
  --no-reasoning-preserve `
  -rea auto `
  --load-mode mmap+mlock `
  --cache-prompt --cache-ram 8192 --ctx-checkpoints 24 --checkpoint-min-step 10240 `
  --sleep-idle-seconds 180 `
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 `
  --presence-penalty 0.0 --repeat-penalty 1.0 `
  --host 127.0.0.1 --port 8079
