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
$OutputDir = Join-Path $Root "calculation_details"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $Python (Join-Path $Root "scripts\build_airline_calculation_details.py") `
    --input $InputFiles `
    --output-dir $OutputDir `
    --train-pattern "2026-03|2026-04" `
    --test-pattern "2026-05" `
    --rounds 260 `
    --catboost-iterations 500 `
    --seed 42 `
    --top-k 10 `
    --surrogate-k 8

Write-Host ""
Write-Host "Calculation details finished."
Write-Host "Output directory: $OutputDir"
