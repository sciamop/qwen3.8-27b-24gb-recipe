#Requires -Version 5.1
<#
.SYNOPSIS
    One-click installer for the Qwen3.8-27B local deployment on RTX A5000.
    Qwen3.8-27B A5000 单卡部署一键安装脚本。

.DESCRIPTION
    Downloads llama.cpp (b10566, CUDA 13.3) and the Qwen3.8-27B GGUF model,
    then prepares the llama-cpp/ and models/ directories.
    下载 llama.cpp 引擎与模型权重，准备 llama-cpp/ 与 models/ 目录。

.PARAMETER SkipModel
    Skip downloading the model (engine only). 跳过模型下载，仅安装引擎。

.PARAMETER Proxy
    Proxy URL for downloads, e.g. http://127.0.0.1:7890. 下载代理地址。

.PARAMETER Force
    Re-download existing files. 已存在文件重新下载。

.PARAMETER LlamaCppUrl
    Override llama.cpp release zip URL. 覆盖引擎下载地址。

.PARAMETER ModelUrl
    Override model GGUF URL. 覆盖模型下载地址。

.EXAMPLE
    .\install.ps1
    .\install.ps1 -SkipModel
    .\install.ps1 -Proxy http://127.0.0.1:7890
#>
param(
    [switch]$SkipModel,
    [string]$Proxy = "",
    [switch]$Force,
    [string]$LlamaCppUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b10566/llama-b10566-bin-win-cuda-13.3-x64.zip",
    [string]$ModelUrl = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_XL.gguf"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$llamaCppDir = Join-Path $scriptDir "llama-cpp"
$modelsDir = Join-Path $scriptDir "models"
$llamaServerExe = Join-Path $llamaCppDir "llama-server.exe"
$modelFile = Join-Path $modelsDir "Qwen3.8-27B-UD-Q4_K_XL.gguf"

$script:stepCount = 0
function Write-Step($msgEn, $msgZh) { $script:stepCount++; Write-Host "[$script:stepCount/5] $msgEn / $msgZh" -ForegroundColor Cyan }
function Write-Info($msgEn, $msgZh) { Write-Host "  - $msgEn / $msgZh" -ForegroundColor Gray }

# ---------- Step 1: environment checks ----------
Write-Step "Checking environment..." "检查环境..."
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Error "nvidia-smi not found. This deployment requires an NVIDIA GPU with at least 24GB VRAM. / 未找到 nvidia-smi，本部署需要至少 24GB 显存的 NVIDIA 显卡。"
}
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "curl.exe not found (required for downloads). / 未找到 curl.exe（下载依赖）。"
}

$gpuLine = & nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader 2>$null | Select-Object -First 1
if ($gpuLine) {
    $gpuParts = $gpuLine -split ","
    $totalMB = [int]($gpuParts[0] -replace "[^0-9]", "")
    Write-Info "GPU detected: $($gpuParts[0].Trim()) total / $($gpuParts[1].Trim()) free" "检测到 GPU：共 $($gpuParts[0].Trim()) / 空闲 $($gpuParts[1].Trim())"
    if ($totalMB -lt 24000) {
        Write-Host "  [WARN] VRAM is below 24GB; this model may not fit. / 警告：显存小于 24GB，可能无法加载本模型。" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [WARN] Could not query GPU memory. / 警告：无法查询显存信息。" -ForegroundColor Yellow
}

$freeGB = (Get-PSDrive (Split-Path $scriptDir -Qualifier).TrimEnd(':')).Free / 1GB
$needGB = 30
if ($SkipModel) { $needGB = 5 }
if ($freeGB -lt $needGB) {
    Write-Host "  [WARN] Free disk space is only $([math]::Round($freeGB,1)) GB; at least $needGB GB recommended. / 警告：磁盘剩余 $([math]::Round($freeGB,1)) GB，建议至少 $needGB GB。" -ForegroundColor Yellow
}

if ($Proxy) {
    Write-Info "Using proxy: $Proxy" "使用代理：$Proxy"
}

# ---------- Step 2: download llama.cpp ----------
Write-Step "Downloading llama.cpp (b10566, CUDA 13.3)..." "下载 llama.cpp 引擎（b10566, CUDA 13.3）..."
New-Item -ItemType Directory -Path $llamaCppDir -Force | Out-Null
if ((Test-Path $llamaServerExe) -and -not $Force) {
    Write-Info "llama-server.exe already exists, skipping. (use -Force to re-download)" "已存在 llama-server.exe，跳过（-Force 强制重下）"
} else {
    $zip = Join-Path $env:TEMP "llama-b10566-bin-win-cuda-13.3-x64.zip"
    $curlArgs = @("-L", "-C", "-", "--retry", "5", "--retry-all-errors", "-o", $zip, $LlamaCppUrl)
    if ($Proxy) { $curlArgs = @("-x", $Proxy) + $curlArgs }
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "llama.cpp download failed (exit $LASTEXITCODE). / 引擎下载失败。" }
    Write-Info "Extracting..." "解压中..."
    Expand-Archive -Path $zip -DestinationPath $llamaCppDir -Force
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path $llamaServerExe)) { Write-Error "llama-server.exe missing after extraction. / 解压后未找到 llama-server.exe。" }
}

# ---------- Step 3: download model ----------
Write-Step "Downloading Qwen3.8-27B GGUF (16.35 GB, resumable)..." "下载模型权重（16.35 GB，断点续传）..."
New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
if ($SkipModel) {
    Write-Info "Skipped (as requested). / 已按参数跳过模型下载。"
} elseif ((Test-Path $modelFile) -and -not $Force) {
    Write-Info "Model already exists, skipping. (use -Force to re-download)" "已存在模型文件，跳过（-Force 强制重下）"
} else {
    $curlArgs = @("-L", "-C", "-", "--retry", "5", "--retry-all-errors", "-o", $modelFile, $ModelUrl)
    if ($Proxy) { $curlArgs = @("-x", $Proxy) + $curlArgs }
    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "Model download failed (exit $LASTEXITCODE). / 模型下载失败。" }
    $sizeGB = [math]::Round((Get-Item $modelFile).Length / 1GB, 2)
    Write-Info "Downloaded: $sizeGB GB" "下载完成：$sizeGB GB"
    if ($sizeGB -lt 15) {
        Write-Host "  [WARN] Model file is only $sizeGB GB (expected ~16.35 GB). The download may be incomplete; re-run with -Force to re-download. / 警告：模型文件仅 $sizeGB GB（预期约 16.35 GB），下载可能不完整；请用 -Force 重新下载。" -ForegroundColor Yellow
    }
}

# ---------- Step 4: verify ----------
Write-Step "Verifying installation..." "验证安装..."
$ok = $true
if (-not (Test-Path $llamaServerExe)) { Write-Host "  [FAIL] llama-server.exe missing / 缺少 llama-server.exe" -ForegroundColor Red; $ok = $false }
else { Write-Info "llama.cpp: OK" "引擎：OK" }
if (-not $SkipModel) {
    if (-not (Test-Path $modelFile)) { Write-Host "  [FAIL] model missing / 缺少模型文件" -ForegroundColor Red; $ok = $false }
    else {
        $modelSizeGB = [math]::Round((Get-Item $modelFile).Length / 1GB, 2)
        Write-Info "model: $modelSizeGB GB" "模型：$modelSizeGB GB"
        if ($modelSizeGB -lt 15) { Write-Host "  [WARN] model size $modelSizeGB GB is below expected ~16.35 GB / 警告：模型大小 $modelSizeGB GB 低于预期约 16.35 GB" -ForegroundColor Yellow }
    }
}
if (-not $ok) { Write-Error "Installation incomplete. / 安装不完整。" }

# ---------- Step 5: done ----------
Write-Step "Done! Next steps:" "完成！下一步："
Write-Host ""
Write-Host "  Start the server / 启动服务:" -ForegroundColor Green
Write-Host "    powershell -ExecutionPolicy Bypass -File $(Join-Path $scriptDir 'start-server.ps1')"
Write-Host ""
Write-Host "  Verify / 验证:" -ForegroundColor Green
Write-Host "    powershell -ExecutionPolicy Bypass -File $(Join-Path $scriptDir 'verify-server.ps1')"
Write-Host ""
Write-Host "  Benchmark / 基准测试:" -ForegroundColor Green
Write-Host "    powershell -NoProfile -File $(Join-Path $scriptDir 'benchmark.ps1')"
Write-Host ""
Write-Host "  Server endpoint / 服务地址: http://127.0.0.1:8080/v1 (OpenAI-compatible)"
