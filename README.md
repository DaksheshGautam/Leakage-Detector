
## Overview

**LeakGuard** is a lightweight inspection tool designed to detect potential data leakage risks in machine learning datasets **before model training**.

The system acts as a **pre-model safety scanner**, analyzing features and identifying suspicious columns that may artificially inflate model performance.

LeakGuard does **not** perform data cleaning, feature engineering, or modeling. Its sole responsibility is to detect leakage patterns and generate warnings to support safer ML workflows.

---

## Problem Statement

Machine learning models often achieve unrealistically high performance due to **data leakage**, where features contain information unavailable at prediction time.

Common leakage sources include:

* Post-outcome variables
* Identifier columns
* Duplicate samples
* Strong proxy features correlated with target
* Overly predictive engineered attributes

These issues are frequently overlooked and lead to models that fail in production.

LeakGuard aims to automatically surface statistical signals of such risks.

---

## Position in ML Pipeline

LeakGuard operates after basic dataset cleaning but before modeling.

```
Raw Data
   ↓
Basic Cleaning (User)
   ↓
🚨 LeakGuard Scan
   ↓
Train-Test Split
   ↓
Feature Engineering
   ↓
Model Training
```

LeakGuard functions as a **safety gate** within the ML pipeline.

---

## Features (Version 1)

LeakGuard V1 provides:

* Dataset loading
* Target separation
* Column structural profiling
* Automatic column type detection
* Identifier column detection
* Duplicate row detection
* High correlation detection with target
* Feature importance–based leakage detection
* Temporal leakage detection
* Structured console reporting

The tool **flags suspicious features only** and does not automatically modify datasets.

---

## Input Design

User provides:

* Dataset file path
* Target column name

LeakGuard internally:

* Separates features and target
* Performs dataset analysis
* Executes a lightweight internal train-test split only for feature importance inspection

---

## System Architecture

LeakGuard follows a modular inspection pipeline:

### Input & Preparation

* Load dataset
* Separate target

### Dataset Understanding

* Column profiling
* Column type detection

### Detection Engines

* Identifier detector
* Duplicate detector
* Correlation detector
* Feature importance detector

### Reporting

* Aggregate findings
* Generate readable report

---

## Detection Logic

### Identifier Detection

Columns with extremely high uniqueness ratios are flagged as potential identifiers that may cause memorization.

### Duplicate Detection

Duplicate rows are detected as they can artificially inflate model performance.

### Correlation Detection

Features exhibiting unusually high correlation with the target are flagged as possible leakage proxies.

### Feature Importance Detection

A lightweight RandomForest model is trained on internally split data. Features with unusually high importance are flagged as suspicious.

### Temporal Leakage Detection

Identifies potential datetime columns and checks if the dataset is sorted by time. A sorted dataset poses a risk for random train-test splits, which could cause the model to train on future data to predict the past.

---

## Example Output

```
========== LeakGuard Report ==========

Dataset shape: (10000, 18)

Identifier Risk:
['transaction_id']

Duplicates:
12

High Correlation:
['missed_payments']

High Importance:
['final_status']

Temporal Leakage Risks:
- Dataset is sorted by 'transaction_date' (Ascending) - Random splits may leak future info
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```python
from leakguard import run_leakguard

run_leakguard("dataset.csv", target="TargetColumn")
```

Or run directly from notebook using the provided functions.

---

## Design Principles

* Minimalistic scope
* Modular functions
* Dataset-agnostic logic
* Inspection over transformation
* Human-in-the-loop decision making
* Avoid overengineering

---

## Future Work

* Risk scoring
* Recommendations
* UI interface
* Automated feature removal
* HTML reporting

---

## Project Objective

LeakGuard demonstrates understanding of:

* Data leakage failure modes
* Dataset structural analysis
* ML pipeline safety practices
* Modular engineering design

The project serves as a portfolio artifact showcasing **data-centric machine learning awareness**.
