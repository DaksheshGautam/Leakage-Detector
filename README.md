## Overview

**LeakGuard** is a lightweight inspection tool designed to detect potential data leakage risks in machine learning datasets **before model training**.

The system acts as a **pre-model safety scanner**, analyzing features and identifying suspicious columns that may artificially inflate model performance.

LeakGuard does **not** perform data cleaning, feature engineering, or modeling. Its sole responsibility is to detect leakage patterns and generate warnings to support safer ML workflows.

---

## Features (Version 0.8.0)

*   **Dataset Loading & Profiling**: Ingests a CSV and profiles its structure.
*   **Leakage Detectors**:
    *   Identifier Column Detection
    *   Duplicate Row Detection
    *   Group Leakage Detection
    *   Temporal Leakage Detection
    *   High Correlation Detection with Target
    *   High Feature Importance Detection
*   **Rich Reporting**: Color-coded findings summary and dashboard.
*   **Analysis Confidence**: High/Medium/Low score with percentage.
*   **Analysis Stability**: Warns for very small datasets and high-dimensional instability.
*   **Validation Advisory**: Recommends split strategy (`TimeSeriesSplit`, `GroupKFold`, or standard split).
*   **Next Actions Checklist**: Deduplicated, priority-based (`P1/P2/P3`) checklist with reasons.
*   **JSON Export**: Export report + checklist to stdout, file, or Python payload.
*   **Notebook Export Button (Optional)**: In-notebook button to export JSON.

---

## Installation

```bash
pip install -r requirements.txt
```

Optional notebook UI dependencies (only needed for `show_export_button=True`):

```bash
pip install ipywidgets ipython
```

---

## Usage

```python
from leakguard import run_leakguard

run_leakguard("dataset.csv", target_column="TargetColumn")

# Print JSON to stdout
run_leakguard("dataset.csv", target_column="TargetColumn", json_stdout=True)

# Write JSON to file
run_leakguard("dataset.csv", target_column="TargetColumn", json_output_path="leakguard_report.json")

# Return payload as dict
payload = run_leakguard("dataset.csv", target_column="TargetColumn", return_payload=True)

# Show export button in a notebook
run_leakguard(
    "dataset.csv",
    target_column="TargetColumn",
    show_export_button=True,
    export_button_path="leakguard_report.json"
)
```

CLI usage:

```bash
python leakguard.py --file dataset.csv --target TargetColumn --json
python leakguard.py --file dataset.csv --target TargetColumn --json-path leakguard_report.json
```

---

## `run_leakguard` Parameters

* `file_path` *(str, required)*: CSV path.
* `target_column` *(str, required)*: target column name.
* `json_output_path` *(str, optional)*: write JSON report to file.
* `json_stdout` *(bool, optional)*: print JSON report to stdout.
* `return_payload` *(bool, optional)*: return JSON payload as a Python dict.
* `show_export_button` *(bool, optional)*: show Jupyter export button under output.
* `export_button_path` *(str, optional)*: output file path used by export button.

---

## Project Objective

LeakGuard demonstrates understanding of:

* Data leakage failure modes in machine learning.
* Structural and statistical dataset analysis.
* Data-centric ML safety practices.
* Modular and clear engineering design.
