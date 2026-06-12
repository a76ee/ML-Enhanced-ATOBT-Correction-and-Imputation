# ML-Enhanced ATOBT Correction and Imputation

This repository provides a local web app for ATOBT correction and missing-value imputation.

It is intended for colleagues and managers who need to download the tool, run it locally, upload Excel or CSV files, compare predictions with historical actual handover times, and export validation results.

No raw training data, test data, generated Excel outputs, or trained model artifacts are included in this repository.

## Quick Start

1. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

2. Put local private training files into:

```text
data/training/
```

3. Start the web app:

```powershell
.\launch_atobt_web.ps1
```

Or double-click:

```text
双击启动_ATOBT网页.bat
```

4. Upload an Excel or CSV file in the web page.

5. Review the historical error comparison and download the result workbook.

## What The App Does

- Predicts the final target time: `实际移交机坪管制时间`
- Uses `A-TOBT` as the primary correction anchor
- Uses `A-DOBT` as the fallback anchor when `A-TOBT` is missing
- Compares predictions against historical `实际移交机坪管制`
- Exports Chinese result sheets:
  - `汇总`
  - `分航司对比`
  - `预测明细`

The actual handover time is used only as the training and evaluation label. It is not used as an input feature during prediction.

## Algorithm Scope

The web app trains two correction stacks:

- `A-TOBT修正`: predicts `实际移交机坪管制 - A-TOBT`
- `A-DOBT兜底`: predicts `实际移交机坪管制 - A-DOBT` when `A-TOBT` is missing

The machine-learning models include:

- XGBoost
- LightGBM
- CatBoost
- Equal-weight ensemble of XGBoost, LightGBM, and CatBoost

The reproducible scripts also include lightweight Random Forest and Gradient Boosting implementations for comparison.

## Input Columns

The app accepts `.csv`, `.xlsx`, and `.xlsm` files.

Important columns include:

- `A-TOBT`
- `A-DOBT`
- `实际移交机坪管制`
- `CTOT`
- `IFC`
- `CLA`
- `TAR`
- `ITY`
- `RWYA`
- `RWYD`

See [docs/input_columns.md](docs/input_columns.md) for the full input-field description.

## Data Policy

Do not commit original ACDM/flight data or generated outputs.

The `.gitignore` intentionally excludes:

- `*.csv`, `*.xlsx`, `*.xls`, `*.xlsm`
- `outputs/`, `source_outputs/`, `reproduced_outputs/`, `calculation_details/`
- `web_app/outputs/`
- model files such as `*.cbm`, `*.pkl`, `*.joblib`

## Files Included

- `web_app/`: local browser app for upload, prediction, comparison, and export
- `scripts/`: reproducible algorithm and calculation-detail scripts
- `launch_atobt_web.ps1`: PowerShell web-app launcher
- `双击启动_ATOBT网页.bat`: double-click Windows launcher
- `requirements.txt`: Python dependency versions
- `docs/input_columns.md`: input column guide
- `data/training/README.md`: placeholder for private local training files

## Reproduce Algorithm Outputs

```powershell
.\run_reproduction.ps1
```

The script reads local private files from `data/training/` and writes generated outputs into ignored folders.

To export per-airline calculation processes, parameters, SHAP contributions, surrogate coefficients, and model files:

```powershell
.\run_calculation_details.ps1
```

Those exports are generated locally and ignored by Git.

