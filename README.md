# LeakProfiler

Pre-model data leakage scanner for machine learning datasets.

LeakProfiler inspects a dataset before training and flags structural/statistical leakage patterns that can inflate model performance. It does not train models, clean data, or do feature engineering—it focuses on leakage risk discovery, explainable risk scoring, and actionable validation guidance.

---

## Release Status

**Current release:** `1.0.0a1` (**Alpha**)

This alpha is production-installable from PyPI, but scoring policy and heuristics may still evolve based on early feedback.

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
pip install leakprofiler==1.0.0a1
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
leakprofiler --file dataset.csv --target TargetColumn
leakprofiler --file dataset.csv --target TargetColumn --json
leakprofiler --file dataset.csv --target TargetColumn --json-path leakprofiler_report.json
```

### CLI (from source tree)

```bash
python src/LeakProfiler.py --file dataset.csv --target TargetColumn --json
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
python -m twine upload --repository testpypi dist/leakprofiler-1.0.0a1*
```

### PyPI

```bash
python -m twine upload dist/leakprofiler-1.0.0a1*
```

### Tag suggestion

```bash
git tag -a v1.0.0-alpha.1 -m "LeakProfiler 1.0.0 alpha 1"
git push origin v1.0.0-alpha.1
```

---

## Project Objective

LeakProfiler demonstrates practical, data-centric ML safety engineering:
- leakage-first dataset inspection,
- explainable risk diagnostics,
- conservative false-positive controls,
- and actionable validation guidance before model training.
