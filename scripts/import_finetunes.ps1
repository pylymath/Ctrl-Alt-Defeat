# Imports the Colab fine-tunes into Ollama and prints the settings that switch the app to them.
# Usage (PowerShell, from the repo root):
#     .\scripts\import_finetunes.ps1 -ModelsDir "C:\models\neuroassist-finetunes"
# ModelsDir is the Drive folder "neuroassist-finetunes", downloaded as is:
#     neuro-card-writer\  *.gguf + Modelfile
#     neuro-coder\        *.gguf + Modelfile
#     neuro-doc-qa\       *.gguf + Modelfile
#     neuro-vision\merged\  safetensors + Modelfile   (imported with --quantize q4_K_M)
#     neuro-whisper-small\ct2\  faster-whisper model (no Ollama import needed)
param([Parameter(Mandatory = $true)][string]$ModelsDir)
$ErrorActionPreference = "Stop"

$text = @(
    @{ Name = "neuro-card-writer"; Setting = "NEUROASSIST_CARD_MODEL" },
    @{ Name = "neuro-coder";       Setting = "NEUROASSIST_CODE_MODEL" },
    @{ Name = "neuro-doc-qa";      Setting = "NEUROASSIST_TEXT_MODEL" }
)
$settings = [ordered]@{}

foreach ($m in $text) {
    $dir = Join-Path $ModelsDir $m.Name
    if (-not (Test-Path (Join-Path $dir "Modelfile"))) { Write-Host "skip  $($m.Name): no Modelfile in $dir"; continue }
    Write-Host "import $($m.Name) ..."
    Push-Location $dir
    try { ollama create $m.Name -f Modelfile; if ($LASTEXITCODE) { throw "ollama create $($m.Name) failed" } }
    finally { Pop-Location }
    $settings[$m.Setting] = $m.Name
}

$vision = Join-Path $ModelsDir "neuro-vision\merged"
if (Test-Path (Join-Path $vision "Modelfile")) {
    Write-Host "import neuro-vision (quantizing to q4_K_M, takes a few minutes) ..."
    Push-Location $vision
    try { ollama create neuro-vision --quantize q4_K_M -f Modelfile; if ($LASTEXITCODE) { throw "ollama create neuro-vision failed" } }
    finally { Pop-Location }
    $settings["NEUROASSIST_VISION_MODEL"] = "neuro-vision"
    $settings["NEUROASSIST_COUNT_MODE"] = "direct"          # trained on direct counts
    $settings["NEUROASSIST_VIDEO_MODE"] = "direct"          # trained on all frames in one call
    $settings["NEUROASSIST_VIDEO_DIRECT_MAX_FRAMES"] = "16"
} else { Write-Host "skip  neuro-vision: no Modelfile in $vision" }

$ct2 = Join-Path $ModelsDir "neuro-whisper-small\ct2"
if (Test-Path (Join-Path $ct2 "model.bin")) {
    $settings["NEUROASSIST_WHISPER_MODEL"] = (Resolve-Path $ct2).Path
} else { Write-Host "skip  whisper: no model.bin in $ct2 (the 'hf' folder is not what the app loads)" }

Write-Host "`nImported. Ollama now has:"; ollama list
Write-Host "`nTo use them in THIS PowerShell window (run the eval with and without before keeping any):"
foreach ($k in $settings.Keys) { Write-Host "    `$env:$k=`"$($settings[$k])`"" }
Write-Host "`nTo keep one permanently (new windows only):  setx NAME `"value`""
