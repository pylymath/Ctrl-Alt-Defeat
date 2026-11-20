# NeuroAssist one-step setup for Windows. From the repo folder, in PowerShell:
#
#     powershell -ExecutionPolicy Bypass -File install.ps1
#     powershell -ExecutionPolicy Bypass -File install.ps1 -Finetunes "D:\neuroassist-finetunes"
#     powershell -ExecutionPolicy Bypass -File install.ps1 -HfRepo "USER/neuroassist-finetunes"
#
# It: finds Python 3.10-3.12 and Ollama (offers to install them with winget), creates .venv, installs the
# Python packages (CPU-only PyTorch, small), pulls the models, imports our fine-tune if given, downloads the
# speech + sound models, writes neuroassist.env, and asks one test question. Then run:  run.bat
# Nothing is changed outside this folder except Ollama's model store and the Python/Ollama installs you approve.
param([string]$Finetunes = "", [string]$HfRepo = "", [ValidateSet("", "lite", "standard")][string]$Profile = "",
      [switch]$SkipTest)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Ask($q) { (Read-Host "$q [y/N]") -match '^(y|yes)$' }

# --- Python ---------------------------------------------------------------------------------------------
$pyExe = $null; $pyArgs = @()
foreach ($cand in @(@{e = "py"; a = @("-3.12") }, @{e = "py"; a = @("-3.11") }, @{e = "py"; a = @("-3.10") },
                    @{e = "python"; a = @() })) {
    if (-not (Get-Command $cand.e -ErrorAction SilentlyContinue)) { continue }
    try {
        $a = $cand.a
        $v = & $cand.e @a -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and "$v" -match '^3\.(10|11|12)$') { $pyExe = $cand.e; $pyArgs = $cand.a; break }
    } catch {}
}
if (-not $pyExe) {
    Write-Host "Python 3.10-3.12 was not found."
    if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Ask "Install Python 3.12 with winget now?")) {
        winget install -e --id Python.Python.3.12 --accept-package-agreements
        $pyExe = "py"; $pyArgs = @("-3.12")
    } else { Write-Host "Install Python 3.12 from https://www.python.org/downloads/ and run this again."; exit 1 }
}
Write-Host "[OK  ] Python: $pyExe $($pyArgs -join ' ')"

# --- Ollama ---------------------------------------------------------------------------------------------
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama (runs the AI models locally) was not found."
    if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Ask "Install Ollama with winget now?")) {
        winget install -e --id Ollama.Ollama --accept-package-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    } else { Write-Host "Install Ollama from https://ollama.com/download and run this again."; exit 1 }
}
Write-Host "[OK  ] Ollama: $((ollama --version) -join ' ')"

# --- Python packages ------------------------------------------------------------------------------------
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[..  ] Creating .venv"
    & $pyExe @pyArgs -m venv .venv
}
$vpy = ".venv\Scripts\python.exe"
Write-Host "[..  ] Installing Python packages (first time: a few minutes)"
& $vpy -m pip install --upgrade pip --quiet
# CPU-only PyTorch: the sound classifier runs on the CPU, and this is ~200 MB instead of ~2.5 GB.
& $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
& $vpy -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE) { Write-Host "pip install failed (see above)."; exit 1 }
Write-Host "[OK  ] Python packages installed"

# --- Models, settings, test -------------------------------------------------------------------------------
$args2 = @()
if ($Finetunes) { $args2 += @("--finetunes", $Finetunes) }
if ($HfRepo) { $args2 += @("--hf-repo", $HfRepo) }
if ($Profile) { $args2 += @("--profile", $Profile) }
if ($SkipTest) { $args2 += "--skip-test" }
& $vpy -m neuroassist.installer @args2
exit $LASTEXITCODE
