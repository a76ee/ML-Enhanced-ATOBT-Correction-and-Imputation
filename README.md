# ML-Enhanced ATOBT Correction and Imputation

This repository contains the algorithm code and local web app for the thesis project:

**基于保障节点的离港航班 ATOBT 估计与缺失补全研究**

No raw training data, test data, generated Excel outputs, or trained model artifacts are included.

## Algorithm Target

The final prediction target is **实际移交机坪管制时间**.

- Primary anchor: `A-TOBT`
- Fallback anchor: `A-DOBT`
- Ground truth for evaluation: `实际移交机坪管制`
- The actual handover time is used only as the training/evaluation label, not as an input feature during prediction.

The web app trains two correction stacks:

- `A-TOBT修正`: predicts `实际移交机坪管制 - A-TOBT`
- `A-DOBT兜底`: predicts `实际移交机坪管制 - A-DOBT` when `A-TOBT` is missing

## Files Included

- `web_app/`: local browser app for uploading Excel/CSV and exporting Chinese result sheets.
- `scripts/`: reproducible algorithm scripts and per-airline calculation-detail exporters.
- `launch_atobt_web.ps1` and `双击启动_ATOBT网页.bat`: double-click launcher for Windows.
- `requirements.txt`: Python package versions used in this project.
- `docs/input_columns.md`: input field description.
- `data/training/README.md`: where to put local private training files.

## Data Policy

Do not commit original ACDM/flight data or generated outputs.

The `.gitignore` intentionally excludes:

- `*.csv`, `*.xlsx`, `*.xls`, `*.xlsm`
- `outputs/`, `source_outputs/`, `reproduced_outputs/`, `calculation_details/`
- `web_app/outputs/`
- model files such as `*.cbm`, `*.pkl`, `*.joblib`

## Run Locally

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Put local private training files under:

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

You can also specify training files explicitly:

```powershell
python .\web_app\app.py --train-input .\data\training\train_1.csv .\data\training\train_2.csv
```

## Reproduce Algorithm Outputs

```powershell
.\run_reproduction.ps1
```

The script reads local files from `data/training/` and writes generated outputs into ignored folders.

To export per-airline calculation processes, parameters, SHAP contributions, surrogate coefficients, and model files:

```powershell
.\run_calculation_details.ps1
```

Those exports are generated locally and ignored by Git.
