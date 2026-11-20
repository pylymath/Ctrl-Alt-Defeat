# One-time Ollama memory settings for the demo laptop. Run, then QUIT Ollama from the tray and start it again.
#     .\scripts\ollama_settings.ps1
# Why: on a big GPU, Ollama may allow 4 requests at once and reserve context memory (KV cache) for each.
# With our 12k-token video context that alone is about 7 GB. One user = one request at a time.
setx OLLAMA_NUM_PARALLEL 1          | Out-Null   # one request at a time: 1x KV cache, not 4x
setx OLLAMA_MAX_LOADED_MODELS 2     | Out-Null   # vision/text model + QA model resident; coder swaps in
setx OLLAMA_FLASH_ATTENTION 1       | Out-Null   # less memory for long contexts
setx OLLAMA_KV_CACHE_TYPE q8_0      | Out-Null   # KV cache at 8 bit: about half the memory, same answers in practice
Write-Host "Set. Quit Ollama (tray icon > Quit) and start it again, then check with:  ollama ps   and   nvidia-smi"
