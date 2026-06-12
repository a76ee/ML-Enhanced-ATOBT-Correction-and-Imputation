$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$TrainingDir = Join-Path $Root "data\training"
$InputFiles = @(
    Get-ChildItem -LiteralPath $TrainingDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @(".csv", ".xlsx", ".xlsm") } |
        Sort-Object Name |
        ForEach-Object { $_.FullName }
)
if ($InputFiles.Count -eq 0) {
    throw "No training files found in $TrainingDir. Put local CSV/XLSX files there first."
}

$SourceOutputDir = Join-Path $Root "reproduced_outputs\source_outputs"
$BestOutputDir = Join-Path $Root "reproduced_outputs\outputs"
New-Item -ItemType Directory -Force -Path $SourceOutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $BestOutputDir | Out-Null

& $Python (Join-Path $Root "scripts\reproduce_no_atobt_support_nodes.py") `
    --input $InputFiles `
    --output-dir $SourceOutputDir `
    --train-pattern "2026-03|2026-04" `
    --test-pattern "2026-05" `
    --period-label "2026-03_04_train_2026-05_test" `
    --rounds 260 `
    --catboost-iterations 500 `
    --seed 42

& $Python (Join-Path $Root "scripts\build_airline_best_outputs.py") `
    --source-dir $SourceOutputDir `
    --output-dir $BestOutputDir

Write-Host ""
Write-Host "Reproduction finished."
Write-Host "Source model outputs: $SourceOutputDir"
Write-Host "Per-airline best outputs: $BestOutputDir"
