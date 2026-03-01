## Overview

**LeakProfiler** is a lightweight inspection tool designed to detect potential data leakage risks in machine learning datasets **before model training**.

The system acts as a **pre-model safety scanner**, analyzing features and identifying suspicious columns that may artificially inflate model performance.

LeakProfiler does **not** perform data cleaning, feature engineering, or modeling. Its sole responsibility is to detect leakage patterns and generate warnings to support safer ML workflows.

---

## Features (Version 1.0.0a1 - Alpha)

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
*   **Cross-Detector Reasoning**: Adds composite findings by combining evidence across detectors (reported as `Cross-Detector` category).
*   **Benign Pattern Detection**: Adds conservative, explainable low-risk contextual findings (`Benign-Pattern`) for likely non-leakage signals.
*   **Principled Advisory Engine**: Uses calibrated evidence scoring, overlap de-duplication, uncertainty estimation, and explainable rationale.
*   **Config-Driven Advisory Weights**: Advisory scoring uses explicit, tunable config for severity/category weights, thresholds, bonuses, and penalties.
*   **Per-Finding Confidence**: Each finding gets a confidence estimate that influences contribution to overall risk.
*   **JSON Export**: Export report + checklist to stdout, file, or Python payload.
*   **Notebook Export Button (Optional)**: In-notebook button to export JSON.

---

## Cross-Detector Reasoning

After individual detectors run, LeakProfiler performs a second-pass inference step to identify multi-signal risks.

Current cross-detector rules include:

* Correlation + Feature Importance overlap on the same feature → **Cross-detector proxy leakage consensus**.
* Identifier + Group Leakage overlap on the same column/entity key → **Cross-detector entity memorization risk**.
* Temporal signals + highly predictive feature overlap → **Cross-detector temporal proxy risk**.
* Unstable analysis conditions + strong statistical findings → **Cross-detector confidence caution**.

These composite findings are integrated into:

* Findings summary table
* Risk score/dashboard
* Validation advisory and checklist
* JSON export payload

---

## Benign Pattern Detection

After base detectors and cross-detector reasoning, LeakProfiler runs a conservative benign-pattern pass to identify signals that may look suspicious statistically but are often operationally normal.

Design principles:

* **Do not suppress risk findings**: benign findings are additive context, not replacements.
* **Exclude from risk score**: `Benign-Pattern` findings are not counted in leakage severity totals.
* **Conservative edge-case handling**: benign tags are blocked when strong corroborating risk signals exist.

Current benign rules (tuned):

* **Sparse duplicate noise**: duplicate ratio must be extremely small (adaptive threshold by dataset size).
* **Isolated strong predictor**: allowed only when no corroborating correlation/group/temporal/identifier risk exists, with stable + high-confidence analysis and no HIGH-severity findings.
* **Weak temporal structure**: only moderate temporal signal without high autocorrelation, regular spacing, high timestamp uniqueness, group risk, or temporal cross-detector overlap.

Output behavior:

* Dashboard now includes a `Benign Findings` count.
* Benign findings appear in the findings table as category `Benign-Pattern`.
* JSON summary includes `benign_findings`.

---

## Principled Advisory Engine

LeakProfiler separates risk estimation from action recommendation:

* **Risk estimation** uses weighted evidence from findings (severity + category + per-finding confidence), then applies:
    * corroboration bonus when multiple independent signals align,
    * overlap penalties to reduce double-counting between correlated detectors.
* **Configuration-driven policy** keeps thresholds and weights explicit and tunable.
* **Strict HIGH gate**: `HIGH` risk is only emitted when corroboration and confidence gates both pass; otherwise it is downgraded to `MODERATE`.
* **Uncertainty estimation** combines confidence level, analysis stability, and evidence sufficiency.
* **Action recommendation** builds a deduplicated checklist from actionable findings only.

Explainability outputs:

* Dashboard includes `Risk Level` and `Advisory Uncertainty`.
* Advisory panel includes an `Advisory Basis` section with top contributors, bonuses/penalties, and calibrated score.
* JSON summary includes `risk_level`, `uncertainty`, and `risk_rationale`.

---

## Installation

```bash
pip install -r requirements.txt
```

Optional notebook UI dependencies (only needed for `show_export_button=True`):

```bash
pip install ipywidgets ipython
```

Install from PyPI (after publish):

```bash
pip install leakprofiler
```

---

## Distribution (PyPI)

Build artifacts:

```bash
python -m pip install --upgrade build twine
python -m build
```

Validate package metadata:

```bash
python -m twine check dist/*
```

Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```

The package exposes:

* CLI command: `leakprofiler`
* Python API: `from LeakProfiler import run_leakprofiler`

---

## Usage

```python
from LeakProfiler import run_leakprofiler

run_leakprofiler("dataset.csv", target_column="TargetColumn")

# Print JSON to stdout
run_leakprofiler("dataset.csv", target_column="TargetColumn", json_stdout=True)

# Write JSON to file
run_leakprofiler("dataset.csv", target_column="TargetColumn", json_output_path="leakprofiler_report.json")

# Return payload as dict
payload = run_leakprofiler("dataset.csv", target_column="TargetColumn", return_payload=True)

# Show export button in a notebook
run_leakprofiler(
    "dataset.csv",
    target_column="TargetColumn",
    show_export_button=True,
    export_button_path="leakprofiler_report.json"
)

# Backward compatibility: importing from `leakguard` and calling `run_leakguard(...)` is still supported.
```

CLI usage:

```bash
python LeakProfiler.py --file dataset.csv --target TargetColumn --json
python LeakProfiler.py --file dataset.csv --target TargetColumn --json-path leakprofiler_report.json
```

---

## `run_leakprofiler` Parameters

* `file_path` *(str, required)*: CSV path.
* `target_column` *(str, required)*: target column name.
* `json_output_path` *(str, optional)*: write JSON report to file.
* `json_stdout` *(bool, optional)*: print JSON report to stdout.
* `return_payload` *(bool, optional)*: return JSON payload as a Python dict.
* `show_export_button` *(bool, optional)*: show Jupyter export button under output.
* `export_button_path` *(str, optional)*: output file path used by export button.

---

## Project Objective

LeakProfiler demonstrates understanding of:

* Data leakage failure modes in machine learning.
* Structural and statistical dataset analysis.
* Data-centric ML safety practices.
* Modular and clear engineering design.
