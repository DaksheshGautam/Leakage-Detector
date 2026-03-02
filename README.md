# LeakProfiler

![PyPI version](https://img.shields.io/pypi/v/leakprofiler)
![Python](https://img.shields.io/pypi/pyversions/leakprofiler)
![License](https://img.shields.io/github/license/DaksheshGautam/LeakProfiler)


LeakProfiler is a Python package for automated data leakage detection and machine learning validation strategy analysis.

It inspects tabular datasets *before training* and flags structural, statistical, and temporal leakage patterns that can inflate model performance.

LeakProfiler focuses on:
- Leakage risk discovery
- Explainable risk scoring
- Validation strategy guidance (TimeSeriesSplit, GroupKFold, etc.)

---

## Why Data Leakage Matters

Data leakage can artificially inflate model accuracy and lead to unreliable production deployment. 
LeakProfiler helps detect hidden leakage patterns before training, ensuring trustworthy evaluation results.



## Release Status

**Current release:** `1.0.2` (**Stable**)

This stable release is validated for broad dataset inspection workflows and packaged for standard PyPI installation.

### What changed in 1.0.2
- Added robust target-missing handling: rows with missing target values are automatically dropped and reported in dataset tips.
- Added explicit validation error when the target column is entirely missing after filtering.
- Added a conservative risk floor so strong group/proxy leakage signals are not under-classified as `LOW`.
- Fixed correlation threshold calibration to reliably catch perfect proxy features.
- Expanded validation with diverse column-type scenarios and semantic expectation checks.

---

## What It Detects

### Core detectors
- Identifier column leakage
- Duplicate row leakage risk
- Group leakage risk
- Temporal leakage risk
- High feature-target correlation risk
- High feature importance leakage signals

### Reasoning layers
- **Cross-detector reasoning** for multi-signal leakage consensus
- **Benign-pattern detection** for conservative low-risk context (additive only)

### Advisory engine
- Calibrated risk scoring (weighted by severity/category/confidence)
- Overlap de-duplication to reduce double-counting
- Advisory uncertainty estimation
- **Strict HIGH gate**: `HIGH` is emitted only when corroboration + confidence both pass
- Priority-based next-actions checklist (`P1/P2/P3`)

---

## Installation

### Option A: Install from PyPI (recommended)

```bash
pip install leakprofiler
```

For reproducible environments, pin an exact version:

```bash
pip install leakprofiler==1.0.2
```

### Option B: Local development install

```bash
pip install -e .
```

Optional notebook extras:

```bash
pip install "leakprofiler[notebook]"
```

---

## Quick Start

### CLI (installed package)

```bash
leakprofiler --file "dataset.csv" --target "TargetColumn"
leakprofiler --file "dataset.csv" --target "TargetColumn" --json
leakprofiler --file "dataset.csv" --target "TargetColumn" --json-path "leakprofiler_report.json"

# Positional shorthand (also supported)
leakprofiler "dataset.csv" "TargetColumn"
leakprofiler "dataset.csv" "TargetColumn" --json
```

### CLI (from source tree)

```bash
python src/LeakProfiler.py --file "dataset.csv" --target "TargetColumn" --json
python src/LeakProfiler.py "dataset.csv" "TargetColumn" --json
```

### Python API

```python
from LeakProfiler import run_leakprofiler

run_leakprofiler("dataset.csv", target_column="TargetColumn")

# JSON to stdout
run_leakprofiler("dataset.csv", target_column="TargetColumn", json_stdout=True)

# JSON to file
run_leakprofiler(
    "dataset.csv",
    target_column="TargetColumn",
    json_output_path="leakprofiler_report.json"
)

# Return payload dict
payload = run_leakprofiler(
    "dataset.csv",
    target_column="TargetColumn",
    return_payload=True
)

# Notebook export button
run_leakprofiler(
    "dataset.csv",
    target_column="TargetColumn",
    show_export_button=True,
    export_button_path="leakprofiler_report.json"
)
```

Backward-compatible alias is still available:

```python
from LeakProfiler import run_leakguard
```

---

## Output Summary

LeakProfiler renders:
- Findings summary table
- Dashboard (risk score, risk level, confidence, stability, uncertainty)
- Validation advisory panel
- Advisory basis rationale (top contributors, penalties/bonuses, gate status)
- Next-actions checklist

JSON export includes:
- Input metadata
- Findings list
- Severity counts and benign count
- Risk score, risk level, uncertainty
- Risk rationale and next actions

---

## Detection Design Notes

### Cross-detector reasoning
Combines independent signals to produce composite findings such as:
- Correlation + importance overlap → proxy leakage consensus
- Identifier + group overlap → entity memorization risk
- Temporal + predictive overlap → temporal proxy risk
- Unstable analysis + strong statistical findings → confidence caution

### Benign-pattern detection
- Adds context; **does not suppress** risk findings
- Excluded from risk severity totals
- Conservatively blocked when stronger corroborating risk signals exist

---

## `run_leakprofiler` Parameters

- `file_path` *(str, required)*: CSV path.
- `target_column` *(str, required)*: target column name.
- Missing target values in `target_column` are auto-dropped; if all target values are missing, LeakProfiler raises a clear `ValueError`.
- `json_output_path` *(str, optional)*: write JSON to file.
- `json_stdout` *(bool, optional)*: print JSON to stdout.
- `return_payload` *(bool, optional)*: return JSON payload as `dict`.
- `show_export_button` *(bool, optional)*: show notebook export button.
- `export_button_path` *(str, optional)*: export path used by notebook button.

---

## Development

### Run tests

```bash
pytest -q src/test_advisory_engine.py
```

### Stability sweep (100+ datasets)

Run the automated edge-case sweep:

```bash
python scripts/stability_sweep.py
```

Current sweep baseline:
- total datasets: 208
- successful runs: 208
- runtime errors: 0

Results are written to `stability_sweep_report.json`.

### Semantic expectation sweep

Run expected-outcome matching against generated datasets:

```bash
python scripts/semantic_expectation_sweep.py
```

Current semantic baseline:
- total datasets: 208
- passed expectations: 208
- failed expectations: 0
- runtime errors: 0

Results are written to `semantic_expectation_report.json`.

### Build package

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

---

## Maintainer: Publish Flow

### TestPyPI

```bash
python -m twine upload --repository testpypi dist/leakprofiler-1.0.2*
```

### PyPI

```bash
python -m twine upload dist/leakprofiler-1.0.2*
```

### Tag suggestion

```bash
git tag -a v1.0.2 -m "LeakProfiler 1.0.2 stable"
git push origin v1.0.2
```

---

## Project Objective

LeakProfiler demonstrates practical, data-centric ML safety engineering:
- leakage-first dataset inspection,
- explainable risk diagnostics,
- conservative false-positive controls,
- and actionable validation guidance before model training.
