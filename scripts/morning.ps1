# Sets up the V2 folder on the demo laptop, REUSING what already works there (no reinstall):
#   - the Python environment (.venv) from the old folder, linked, not copied
#   - the models already in Ollama (qwen3-vl:4b-instruct, qwen2.5-coder:3b, neuro-doc-qa)
# Run from the NEW folder:
#     powershell -ExecutionPolicy Bypass -File scripts\morning.ps1
# Safe to run again. The old folder is not changed, so it stays the fallback.
param(
    [string]$OldRepo = "C:\Users\fepes\Documents\Hackathon 11.26\Ctrl-Alt-Defeat-main",
    [string]$Data = "C:\data\practice_pack"
)
$ErrorActionPreference = "Continue"   # PS 5.1: "Stop" turns harmless native stderr (pip notices) into fatal errors
Set-Location (Split-Path $PSScriptRoot -Parent)
function OK($m) { Write-Host "[OK  ] $m" -ForegroundColor Green }
function Info($m) { Write-Host "[..  ] $m" }
function BAD($m) { Write-Host "[FAIL] $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path "neuroassist\workspace.py")) { BAD "Run this from the NEW (v2) folder. This folder has no neuroassist\workspace.py." }
OK "In the v2 folder: $(Get-Location)"

# 1. Python environment: link the old, working .venv (no pip, no Smart App Control surprises)
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if (-not (Test-Path "$OldRepo\.venv\Scripts\python.exe")) {
        BAD "No .venv in the old folder '$OldRepo'. Re-run with -OldRepo `"<old folder path>`""
    }
    New-Item -ItemType Junction -Path ".venv" -Target "$OldRepo\.venv" -ErrorAction Stop | Out-Null
    OK "Linked .venv from the old folder"
} else { OK ".venv present" }
$py = ".venv\Scripts\python.exe"
& $py -c "import gradio, ollama, faster_whisper, cv2, pypdf, huggingface_hub" 2>$null
if ($LASTEXITCODE) {
    Info "A package is missing; installing requirements (only what's missing)"
    & $py -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE) { BAD "pip install failed. Use the old folder for the demo and send the error to the team." }
}
OK "Python packages"

# 2. Old NEUROASSIST_* settings (from setx) would override the new settings file: remove them
$old = [Environment]::GetEnvironmentVariables("User").Keys | Where-Object { $_ -like "NEUROASSIST_*" }
foreach ($k in $old) {
    Write-Host "       removing old setting $k=$([Environment]::GetEnvironmentVariable($k, 'User'))"
    [Environment]::SetEnvironmentVariable($k, $null, "User")
    Remove-Item "Env:$k" -ErrorAction SilentlyContinue
}
Get-ChildItem Env: | Where-Object { $_.Name -like "NEUROASSIST_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
OK "Old NEUROASSIST_* settings cleared (settings now live in neuroassist.env)"

# 3. Ollama memory settings, then restart Ollama so they apply
$ollamaEnv = @{ OLLAMA_NUM_PARALLEL = "1"; OLLAMA_MAX_LOADED_MODELS = "2"; OLLAMA_FLASH_ATTENTION = "1"; OLLAMA_KV_CACHE_TYPE = "q8_0" }
foreach ($k in $ollamaEnv.Keys) {
    [Environment]::SetEnvironmentVariable($k, $ollamaEnv[$k], "User")
    Set-Item "Env:$k" $ollamaEnv[$k]
}
if ($env:OLLAMA_HOST -eq "0.0.0.0" -or $env:OLLAMA_HOST -like "0.0.0.0:*") { $env:OLLAMA_HOST = "127.0.0.1:11434" }
Info "Restarting Ollama with the memory settings"
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
$app = "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
if (Test-Path $app) { Start-Process $app } else { Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden }
$up = $false
foreach ($i in 1..40) {
    Start-Sleep -Milliseconds 500
    try { Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 2 | Out-Null; $up = $true; break } catch {}
}
if (-not $up) { BAD "Ollama did not start. Open the Ollama app from the Start menu, then run this script again." }
OK "Ollama running"
$models = (Invoke-RestMethod "http://127.0.0.1:11434/api/tags").models.name
Write-Host "       models: $($models -join ', ')"

# 4. Models + settings file + speech models + one live test question
$dataArg = @()
if (Test-Path $Data) { $dataArg = @("--data", $Data) } else { Write-Host "       (practice pack not found at $Data; eval needs -Data)" }
& $py -m neuroassist.installer @dataArg
if ($LASTEXITCODE) { BAD "Model setup failed (see above). The old folder still works for the demo." }

# 5. Logic checks with a stand-in model (10 s)
& $py tests\offline_checks.py | Select-Object -Last 1
if ($LASTEXITCODE) { BAD "Offline checks failed (see above)." }

Write-Host ""
Write-Host "READY. Next:" -ForegroundColor Green
Write-Host "   run.bat                                                     <- opens the app"
Write-Host "   .venv\Scripts\python.exe eval\run_eval.py --tag v2          <- full eval (~20-30 min)"
