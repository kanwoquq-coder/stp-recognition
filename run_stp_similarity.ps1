param(
    [string]$Query = "",
    [ValidateRange(1, 50)]
    [int]$Top = 7,
    [ValidateRange(1, 200)]
    [int]$CoarseTop = 20,
    [ValidateRange(1, 200)]
    [int]$FusionTop = 20,
    [string]$DatabasePath = "",
    [string]$EmbeddingBaseUrl = "",
    [string]$EmbeddingModel = "",
    [string]$LlmBaseUrl = "",
    [string]$LlmModel = "",
    [ValidateRange(1, 3600)]
    [int]$LlmTimeout = 180,
    [ValidateRange(0, 10)]
    [int]$LlmMaxRetries = 1,
    [ValidateSet("auto", "json", "none")]
    [string]$LlmResponseFormat = "none",
    [ValidateSet("auto", "on", "off")]
    [string]$LlmThinking = "off",
    [switch]$Vision,
    [string]$QueryViewRoot = "C:\Users\phillip\Desktop\project\3D",
    [string]$CandidateViewDir = "C:\Users\phillip\Desktop\project\pic",
    [switch]$NoReport,
    [string]$PythonExecutable = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

# Embedding API is used for text-index building and query-vector generation.
# Prefer EMBEDDING_* variables; API_KEY/BASE_URL remain backward compatible.
[string]$embeddingApiKey = [string][Environment]::GetEnvironmentVariable("EMBEDDING_API_KEY", "Process")
if ([string]::IsNullOrWhiteSpace($embeddingApiKey)) {
    $embeddingApiKey = [Environment]::GetEnvironmentVariable("EMBEDDING\_API\_KEY", "Process")
}
if ([string]::IsNullOrWhiteSpace($embeddingApiKey)) {
    $embeddingApiKey = [Environment]::GetEnvironmentVariable("API_KEY", "Process")
}
if ([string]::IsNullOrWhiteSpace($embeddingApiKey)) {
    $secureKey = Read-Host "Enter the Embedding API key" -AsSecureString
    $credential = [System.Management.Automation.PSCredential]::new("api", $secureKey)
    $embeddingApiKey = $credential.GetNetworkCredential().Password
}
if ([string]::IsNullOrWhiteSpace($embeddingApiKey)) {
    throw "Embedding API key is empty. Set EMBEDDING_API_KEY or API_KEY."
}
$embeddingApiKey = $embeddingApiKey.Trim().Replace('\_', '_')

if ([string]::IsNullOrWhiteSpace($EmbeddingBaseUrl)) {
    $EmbeddingBaseUrl = [Environment]::GetEnvironmentVariable("EMBEDDING_BASE_URL", "Process")
    if ([string]::IsNullOrWhiteSpace($EmbeddingBaseUrl)) {
        $EmbeddingBaseUrl = [Environment]::GetEnvironmentVariable("EMBEDDING\_BASE\_URL", "Process")
    }
    if ([string]::IsNullOrWhiteSpace($EmbeddingBaseUrl)) {
        $EmbeddingBaseUrl = [Environment]::GetEnvironmentVariable("BASE_URL", "Process")
    }
}
if ([string]::IsNullOrWhiteSpace($EmbeddingBaseUrl)) {
    throw "Embedding API URL is empty. Set EMBEDDING_BASE_URL or pass -EmbeddingBaseUrl."
}
$EmbeddingBaseUrl = $EmbeddingBaseUrl.Trim().Replace('\_', '_')
if ($EmbeddingBaseUrl -match '^\[([^\]]+)\]\([^\)]+\)$') {
    $EmbeddingBaseUrl = $Matches[1]
}

if ([string]::IsNullOrWhiteSpace($EmbeddingModel)) {
    $EmbeddingModel = [Environment]::GetEnvironmentVariable("EMBEDDING_MODEL", "Process")
    if ([string]::IsNullOrWhiteSpace($EmbeddingModel)) {
        $EmbeddingModel = [Environment]::GetEnvironmentVariable("EMBEDDING\_MODEL", "Process")
    }
}
if ([string]::IsNullOrWhiteSpace($EmbeddingModel)) {
    throw "Embedding model is empty. Set EMBEDDING_MODEL or pass -EmbeddingModel."
}
$EmbeddingModel = $EmbeddingModel.Trim().Replace('\_', '_')
if ($EmbeddingModel -notmatch '(?i)embedding|bge|gte|e5') {
    Write-Warning (
        "Embedding model '$EmbeddingModel' does not look like an embedding model. " +
        "A chat model such as qwen-plus cannot build the text vector index."
    )
}

# The LLM API is used only for final reranking.
[string]$llmApiKey = [string][Environment]::GetEnvironmentVariable("LLM_API_KEY", "Process")
if ([string]::IsNullOrWhiteSpace($llmApiKey)) {
    $llmApiKey = [Environment]::GetEnvironmentVariable("LLM\_API\_KEY", "Process")
}
if ([string]::IsNullOrWhiteSpace($llmApiKey)) {
    $llmApiKey = "EMPTY"
}
$llmApiKey = $llmApiKey.Trim().Replace('\_', '_')
if ([string]::IsNullOrWhiteSpace($LlmBaseUrl)) {
    $LlmBaseUrl = [Environment]::GetEnvironmentVariable("LLM_BASE_URL", "Process")
    if ([string]::IsNullOrWhiteSpace($LlmBaseUrl)) {
        $LlmBaseUrl = [Environment]::GetEnvironmentVariable("LLM\_BASE\_URL", "Process")
    }
}
if ([string]::IsNullOrWhiteSpace($LlmBaseUrl)) {
    $LlmBaseUrl = "http://10.100.0.35:8000/v1"
}
$LlmBaseUrl = $LlmBaseUrl.Trim().Replace('\_', '_')
if ($LlmBaseUrl -match '^\[([^\]]+)\]\([^\)]+\)$') {
    $LlmBaseUrl = $Matches[1]
}
if ([string]::IsNullOrWhiteSpace($LlmModel)) {
    $LlmModel = [Environment]::GetEnvironmentVariable("MODEL", "Process")
    if ([string]::IsNullOrWhiteSpace($LlmModel)) {
        $LlmModel = [Environment]::GetEnvironmentVariable("LLM_MODEL", "Process")
    }
    if ([string]::IsNullOrWhiteSpace($LlmModel)) {
        $LlmModel = [Environment]::GetEnvironmentVariable("LLM\_MODEL", "Process")
    }
}
if ([string]::IsNullOrWhiteSpace($LlmModel)) {
    $LlmModel = "Qwen3.5-35B-A3B"
}
$LlmModel = $LlmModel.Trim().Replace('\_', '_')

if ([string]::IsNullOrWhiteSpace($DatabasePath)) {
    $DatabasePath = Join-Path $PSScriptRoot "chroma_parts"
}

if (-not (Test-Path -LiteralPath $Query -PathType Leaf)) {
    throw "Query STEP file does not exist: $Query"
}

$scriptPath = Join-Path $PSScriptRoot "stp_similarity.py"
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Cannot find modified search program: $scriptPath"
}

$cliArgs = @(
    $scriptPath,
    "search",
    "--query", $Query,
    "--top", $Top.ToString(),
    "--coarse", $CoarseTop.ToString(),
    "--api-key", $embeddingApiKey,
    "--base-url", $EmbeddingBaseUrl,
    "--embed-model", $EmbeddingModel,
    "--llm-api-key", $llmApiKey,
    "--llm-base-url", $LlmBaseUrl,
    "--llm-model", $LlmModel,
    "--llm-timeout", $LlmTimeout.ToString(),
    "--llm-max-retries", $LlmMaxRetries.ToString(),
    "--llm-thinking", $LlmThinking,
    "--llm-response-format", $LlmResponseFormat,
    "--use-three-way",
    "--fusion-top", $FusionTop.ToString(),
    "--db-path", $DatabasePath
)

if ($Vision) {
    if (-not (Test-Path -LiteralPath $CandidateViewDir -PathType Container)) {
        throw "Candidate view directory does not exist: $CandidateViewDir"
    }

    $queryStem = [IO.Path]::GetFileNameWithoutExtension($Query)
    $queryViewFolder = Join-Path $QueryViewRoot $queryStem
    $queryViewNames = @(
        "front.png", "top.png", "left.png",
        "right.png", "bottom.png", "back.png"
    )
    $queryViews = @(
        $queryViewNames |
            ForEach-Object { Join-Path $queryViewFolder $_ } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    )
    if ($queryViews.Count -lt 3) {
        throw "At least 3 query views are required in: $queryViewFolder"
    }

    $cliArgs += @("--view-dir", $CandidateViewDir, "--query-views")
    $cliArgs += $queryViews
} else {
    $cliArgs += @("--no-vision", "--no-feature-extraction")
}

if ($NoReport) {
    $cliArgs += "--no-report"
}

Write-Host "Program: $scriptPath"
Write-Host "Query: $Query"
Write-Host "Embedding endpoint: $EmbeddingBaseUrl"
Write-Host "Embedding model: $EmbeddingModel"
Write-Host "LLM endpoint: $LlmBaseUrl"
Write-Host "LLM model: $LlmModel"
Write-Host "Candidates: coarse=$CoarseTop, fusion=$FusionTop, result=$Top"
Write-Host ("Mode: " + $(if ($Vision) { "vision" } else { "text-only" }))

& $PythonExecutable @cliArgs
$processExitCode = if (Test-Path Variable:LASTEXITCODE) { $LASTEXITCODE } else { 0 }
exit $processExitCode
