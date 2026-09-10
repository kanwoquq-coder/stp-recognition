param(
    [string]$Query = "C:\Users\phillip\Desktop\project\teststp\测试件3.stp",
    [int]$Top = 7,
    [string]$DatabasePath = ".\chroma_parts",
    [switch]$Vision,
    [string]$QueryViewRoot = "C:\Users\phillip\Desktop\project\3D",
    [string]$CandidateViewDir = "C:\Users\phillip\Desktop\project\pic"
)

$ErrorActionPreference = "Stop"

# Prefer the correct variable name. Also accept a name/value accidentally
# containing escaped underscores after being copied from formatted text.
$embeddingApiKey = [Environment]::GetEnvironmentVariable("EMBEDDING_API_KEY", "Process")
if ([string]::IsNullOrWhiteSpace($embeddingApiKey)) {
    $embeddingApiKey = [Environment]::GetEnvironmentVariable("EMBEDDING\_API\_KEY", "Process")
}

if (-not [string]::IsNullOrWhiteSpace($embeddingApiKey)) {
    $embeddingApiKey = $embeddingApiKey.Replace("\_", "_")
}

if ([string]::IsNullOrWhiteSpace($embeddingApiKey)) {
    $secureKey = Read-Host "Enter the Embedding API key" -AsSecureString
    $credential = New-Object System.Management.Automation.PSCredential("api", $secureKey)
    $embeddingApiKey = $credential.GetNetworkCredential().Password
}

if ([string]::IsNullOrWhiteSpace($embeddingApiKey)) {
    Write-Error "The Embedding API key is empty."
    exit 2
}

if (-not (Test-Path -LiteralPath $Query -PathType Leaf)) {
    Write-Error "Query STEP file does not exist: $Query"
    exit 3
}

$scriptPath = Join-Path $PSScriptRoot "stp.py"
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    Write-Error "Cannot find: $scriptPath"
    exit 4
}

$cliArgs = @(
    $scriptPath,
    "search",
    "--query", $Query,
    "--top", $Top.ToString(),
    "--api-key", $embeddingApiKey,
    "--base-url", "https://llm-8qgclixatgifoso3.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    "--embed-model", "text-embedding-v3",
    "--llm-api-key", "EMPTY",
    "--llm-base-url", "http://10.100.0.35:8000/v1",
    "--llm-model", "Qwen3.5-35B-A3B",
    "--llm-thinking", "off",
    "--llm-response-format", "none",
    "--use-three-way",
    "--fusion-top", "30",
    "--db-path", $DatabasePath
)

if ($Vision) {
    if (-not (Test-Path -LiteralPath $CandidateViewDir -PathType Container)) {
        Write-Error "Candidate view directory does not exist: $CandidateViewDir"
        exit 5
    }

    $queryStem = [IO.Path]::GetFileNameWithoutExtension($Query)
    $queryViewFolder = Join-Path $QueryViewRoot $queryStem
    $queryViewNames = @("front.png", "top.png", "left.png", "right.png", "bottom.png", "back.png")
    $queryViews = @(
        $queryViewNames |
            ForEach-Object { Join-Path $queryViewFolder $_ } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    )

    if ($queryViews.Count -lt 3) {
        Write-Error "At least 3 query views are required in: $queryViewFolder"
        exit 6
    }

    $cliArgs += @("--view-dir", $CandidateViewDir, "--query-views")
    $cliArgs += $queryViews
    $cliArgs += @("--coarse", "10")
} else {
    $cliArgs += @("--no-vision", "--no-feature-extraction")
}

Write-Host "Query: $Query"
Write-Host "LLM endpoint: http://10.100.0.35:8000/v1"
Write-Host "LLM model: Qwen3.5-35B-A3B"
Write-Host ("Mode: " + $(if ($Vision) { "vision" } else { "text-only" }))

& python @cliArgs
exit $LASTEXITCODE
