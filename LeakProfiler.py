__version__ = "0.10.0"
FUNC_NAME = "LeakProfiler"

import argparse
import importlib
import json
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import List, Any
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
from rich.rule import Rule


@dataclass
class Finding:
    title: str
    category: str
    severity: str
    description: str
    evidence: Any
    recommendation: List[str]


def run_leakprofiler(
    file_path,
    target_column,
    json_output_path=None,
    json_stdout=False,
    return_payload=False,
    show_export_button=False,
    export_button_path="leakprofiler_report.json"
):
    findings = []

    header_df = pd.read_csv(file_path, nrows=0)
    parse_date_cols = ['timestamp'] if 'timestamp' in header_df.columns else []
    df = pd.read_csv(file_path, parse_dates=parse_date_cols)

    confidence = calculate_confidence_score(df, target_column)
    stability = check_analysis_stability(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    while target_column in X.columns:
        X = X.drop(columns=[target_column])

    detectors = [
        ("Identifier Detection", detect_identifiers, (X,)),
        ("Duplicate Row Detection", detect_duplicates, (df,)),
        ("Group Leakage Detection", detect_group_leakage, (df, target_column)),
        ("High Correlation Detection", detect_high_correlation, (X, y, target_column)),
        ("Feature Importance Leakage Detection", detect_feature_importance_leakage, (X, y, target_column)),
        ("Temporal Leakage Detection", detect_temporal_leakage, (df, target_column)),
    ]

    with Progress(transient=True) as progress:
        task = progress.add_task(f"[cyan]Running {FUNC_NAME} detectors...", total=len(detectors))

        for name, func, args in detectors:
            progress.update(task, description=f"[cyan]Running: {name}")
            if f := func(*args):
                findings.append(f)
            progress.advance(task)

    cross_findings = infer_cross_detector_findings(findings, stability)
    findings.extend(cross_findings)

    benign_findings = infer_benign_pattern_findings(findings, df, confidence, stability)
    findings.extend(benign_findings)

    console = Console()
    advice = advisory_logic(findings, df, confidence, stability)
    report_renderable = render_report(findings, df.shape)
    advice_renderable = render_advice(advice)

    console.print(Group(report_renderable, advice_renderable))

    payload = None
    if json_output_path or json_stdout or return_payload or show_export_button:
        payload = build_json_export_payload(file_path, target_column, df.shape, findings, advice)

    if json_output_path:
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2, ensure_ascii=False)

    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    if show_export_button:
        _render_notebook_export_button(payload, export_button_path, console)

    if return_payload:
        return payload


def run_leakguard(
    file_path,
    target_column,
    json_output_path=None,
    json_stdout=False,
    return_payload=False,
    show_export_button=False,
    export_button_path="leakprofiler_report.json"
):
    return run_leakprofiler(
        file_path=file_path,
        target_column=target_column,
        json_output_path=json_output_path,
        json_stdout=json_stdout,
        return_payload=return_payload,
        show_export_button=show_export_button,
        export_button_path=export_button_path,
    )


def _render_notebook_export_button(payload, export_button_path, console):
    try:
        widgets = importlib.import_module("ipywidgets")
        ipy_display = importlib.import_module("IPython.display")
        display = ipy_display.display
    except Exception:
        console.print("[yellow]Export button is only available in notebook environments with ipywidgets installed.[/yellow]")
        return

    title = widgets.HTML(value="<b>Export Report</b>")
    subtitle = widgets.HTML(value="<span style='color:#666;'>Save findings and checklist as JSON</span>")

    button = widgets.Button(
        description="Export JSON",
        button_style="success",
        icon="download",
        tooltip="Export report and checklist to JSON",
        layout=widgets.Layout(width="140px", height="34px")
    )

    path_label = widgets.HTML(
        value=f"<span style='font-family:monospace;color:#444;'>{export_button_path}</span>",
        layout=widgets.Layout(margin="0 0 0 8px")
    )

    status = widgets.HTML(value="")

    row = widgets.HBox([button, path_label], layout=widgets.Layout(align_items="center"))
    card = widgets.VBox(
        [title, subtitle, row, status],
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="10px",
            margin="8px 0 0 0",
            width="100%"
        )
    )

    def _on_click(_):
        with open(export_button_path, "w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2, ensure_ascii=False)
        status.value = f"<span style='color:#1a7f37;'>✓ Saved: {export_button_path}</span>"

    button.on_click(_on_click)
    display(card)


def get_adaptive_weights(dataframe):
    n_samples = len(dataframe)
    weights = {"LOW": 1, "MEDIUM": 3, "HIGH": 5}

    if n_samples > 50000:
        weights["LOW"] = 2
        weights["MEDIUM"] = 4

    if n_samples < 1000:
        weights["HIGH"] = 7

    return weights


def calculate_confidence_score(df, target_column):
    score = 100
    n_samples, n_features = df.shape

    if n_samples < 500:
        score -= 30
    elif n_samples < 5000:
        score -= 10

    if n_features < 5:
        score -= 20

    missing_percentage = df.isnull().sum().sum() / (n_samples * n_features)
    if missing_percentage > 0.2:
        score -= 30
    elif missing_percentage > 0.05:
        score -= 10

    is_classification = df[target_column].dtype == 'object' or df[target_column].nunique() < 20
    if is_classification:
        imbalance = df[target_column].value_counts(normalize=True).max()
        if imbalance > 0.95:
            score -= 20
        elif imbalance > 0.80:
            score -= 10

    if score >= 80:
        level = "High"
    elif score >= 50:
        level = "Medium"
    else:
        level = "Low"

    return {"score": score, "level": level}


def check_analysis_stability(df):
    n_samples, n_features = df.shape

    warnings = []

    if n_samples < 200:
        warnings.append(f"Dataset has very few samples ({n_samples}). Statistical results may be highly unstable.")
    elif n_samples < 1000:
        warnings.append(f"Dataset has a small number of samples ({n_samples}). Results may be sensitive to data changes.")

    if n_features > n_samples:
        warnings.append("Dataset has more features than samples, which can lead to unstable feature importance and correlation results.")

    if not warnings:
        return {"level": "Stable", "message": None}
    return {"level": "Warning", "message": " ".join(warnings)}


def advisory_logic(findings, df, confidence, stability):
    severity_weights = get_adaptive_weights(df)
    risk_findings = [f for f in findings if f.category != "Benign-Pattern"]
    benign_findings = [f for f in findings if f.category == "Benign-Pattern"]

    total_score = 0
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for finding in risk_findings:
        total_score += severity_weights.get(finding.severity, 0)
        if finding.severity in severity_counts:
            severity_counts[finding.severity] += 1

    advice = {
        "splitting_strategy": "Standard (e.g., StratifiedKFold)",
        "dataset_tips": [],
        "leakage_score": total_score,
        "severity_counts": severity_counts,
        "total_findings": len(findings),
        "benign_findings": len(benign_findings),
        "confidence": confidence,
        "stability": stability
    }

    has_temporal = any(f.title == "Temporal leakage risk" for f in risk_findings)
    has_group = any(f.title == "Group leakage risk detected" for f in risk_findings)

    if has_temporal:
        advice["splitting_strategy"] = "TimeSeriesSplit"
    elif has_group:
        advice["splitting_strategy"] = "GroupKFold"

    if total_score > 10:
        advice["dataset_tips"].append("High risk of data leakage. Manual inspection of features is highly recommended.")
    elif total_score >= 5:
        advice["dataset_tips"].append("Moderate risk of data leakage. Review the findings and apply recommended actions.")
    else:
        advice["dataset_tips"].append("Low risk of data leakage, but it's good practice to review the findings.")

    if not risk_findings:
        advice["dataset_tips"] = ["No leakage risks detected. Dataset looks safe for standard modeling."]

    if advice["stability"]["message"]:
        advice["dataset_tips"].append(advice["stability"]["message"])

    advice["next_actions"] = build_next_actions(findings, advice["splitting_strategy"], advice["stability"])

    return advice


def build_next_actions(findings, splitting_strategy, stability):
    severity_priority = {"HIGH": "P1", "MEDIUM": "P2", "LOW": "P3"}
    priority_rank = {"P1": 1, "P2": 2, "P3": 3}
    action_map = {}

    def add_action(priority, action, why):
        if not action:
            return

        action_text = str(action).strip()
        if not action_text:
            return

        key = action_text.lower()
        why_text = str(why).strip() if why else ""

        if key in action_map:
            existing = action_map[key]
            if priority_rank[priority] < priority_rank[existing["priority"]]:
                existing["priority"] = priority
            if why_text and why_text not in existing["why"]:
                existing["why"].append(why_text)
        else:
            action_map[key] = {
                "priority": priority,
                "action": action_text,
                "why": [why_text] if why_text else []
            }

    for finding in findings:
        if finding.category == "Benign-Pattern":
            continue
        priority = severity_priority.get(finding.severity, "P3")
        why = f"{finding.title} ({finding.severity})"
        for rec in finding.recommendation:
            add_action(priority, rec, why)

    if splitting_strategy == "TimeSeriesSplit":
        add_action("P1", "Use TimeSeriesSplit for validation", "Temporal leakage signal detected")
    elif splitting_strategy == "GroupKFold":
        add_action("P1", "Use GroupKFold with detected grouping columns", "Group leakage risk detected")

    if stability.get("level") != "Stable":
        add_action("P2", "Treat metric estimates as unstable until data size/shape is improved", "Analysis stability warning")

    if not findings:
        add_action("P3", "Proceed with standard leakage-safe modeling workflow", "No leakage findings were detected")

    actions = []
    for item in action_map.values():
        why_text = "; ".join(item["why"]) if item["why"] else "General best practice"
        actions.append({
            "priority": item["priority"],
            "action": item["action"],
            "why": why_text
        })

    actions.sort(key=lambda x: (priority_rank.get(x["priority"], 99), x["action"].lower()))
    return actions


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _extract_quoted_tokens(text_items):
    tokens = set()
    for item in text_items:
        for match in re.findall(r"'([^']+)'", str(item)):
            token = match.strip()
            if token:
                tokens.add(token)
    return tokens


def _get_finding_by_title(findings, title):
    return next((f for f in findings if f.title == title), None)


def infer_cross_detector_findings(findings, stability):
    composite_findings = []

    corr_finding = _get_finding_by_title(findings, "High correlation with target")
    importance_finding = _get_finding_by_title(findings, "High feature importance detected")
    group_finding = _get_finding_by_title(findings, "Group leakage risk detected")
    identifier_finding = _get_finding_by_title(findings, "Identifier columns detected")
    temporal_finding = _get_finding_by_title(findings, "Temporal leakage risk")

    corr_features = set(_as_list(corr_finding.evidence)) if corr_finding else set()
    importance_features = set(_as_list(importance_finding.evidence)) if importance_finding else set()
    identifier_cols = set(_as_list(identifier_finding.evidence)) if identifier_finding else set()

    group_cols = set()
    if group_finding:
        group_cols = _extract_quoted_tokens(_as_list(group_finding.evidence))

    temporal_cols = set()
    if temporal_finding:
        temporal_cols = _extract_quoted_tokens(_as_list(temporal_finding.evidence))

    overlap_proxy = sorted(corr_features.intersection(importance_features))
    if overlap_proxy:
        composite_findings.append(
            Finding(
                title="Cross-detector proxy leakage consensus",
                category="Cross-Detector",
                severity="HIGH",
                description="The same features were flagged by both correlation and feature-importance detectors, indicating strong proxy leakage risk.",
                evidence=overlap_proxy,
                recommendation=[
                    "Prioritize manual audit of these overlapping features",
                    "Remove features that are unavailable at prediction time"
                ]
            )
        )

    overlap_entity = sorted(identifier_cols.intersection(group_cols))
    if overlap_entity:
        composite_findings.append(
            Finding(
                title="Cross-detector entity memorization risk",
                category="Cross-Detector",
                severity="HIGH",
                description="Identifier-like columns overlap with grouping columns, increasing the chance that the model memorizes entities instead of learning general patterns.",
                evidence=overlap_entity,
                recommendation=[
                    "Exclude overlapping identifier/group columns from training features",
                    "Use strict group-aware validation with non-overlapping entities"
                ]
            )
        )

    temporal_overlap = sorted(temporal_cols.intersection(corr_features.union(importance_features)))
    if temporal_overlap:
        composite_findings.append(
            Finding(
                title="Cross-detector temporal proxy risk",
                category="Cross-Detector",
                severity="HIGH",
                description="Time-related columns also appear among highly predictive features, suggesting potential future-information leakage.",
                evidence=temporal_overlap,
                recommendation=[
                    "Validate with strict chronological splits",
                    "Drop or lag time-derived features that leak future information"
                ]
            )
        )

    if stability.get("level") != "Stable" and (corr_finding or importance_finding):
        composite_findings.append(
            Finding(
                title="Cross-detector confidence caution",
                category="Cross-Detector",
                severity="MEDIUM",
                description="Strong statistical leakage signals were found under unstable data conditions; rankings may be sensitive to small data changes.",
                evidence=stability.get("message") or "Analysis instability detected",
                recommendation=[
                    "Re-run leakage analysis after increasing sample size or reducing dimensionality",
                    "Confirm suspicious features across multiple data slices or folds"
                ]
            )
        )

    return composite_findings


def _extract_row_count(evidence):
    match = re.search(r"(\d+)\s+rows", str(evidence))
    if not match:
        return None
    return int(match.group(1))


def infer_benign_pattern_findings(findings, df, confidence, stability):
    benign_findings = []

    corr_finding = _get_finding_by_title(findings, "High correlation with target")
    importance_finding = _get_finding_by_title(findings, "High feature importance detected")
    group_finding = _get_finding_by_title(findings, "Group leakage risk detected")
    temporal_finding = _get_finding_by_title(findings, "Temporal leakage risk")
    duplicates_finding = _get_finding_by_title(findings, "Duplicate rows detected")
    cross_temporal = _get_finding_by_title(findings, "Cross-detector temporal proxy risk")
    cross_proxy = _get_finding_by_title(findings, "Cross-detector proxy leakage consensus")
    identifier_finding = _get_finding_by_title(findings, "Identifier columns detected")
    high_risk_findings = [f for f in findings if f.severity == "HIGH" and f.category != "Benign-Pattern"]

    if duplicates_finding and len(df) > 0:
        duplicate_rows = _extract_row_count(duplicates_finding.evidence)
        duplicate_ratio = (duplicate_rows / len(df)) if duplicate_rows is not None else None
        duplicate_ratio_threshold = 0.002 if len(df) >= 5000 else 0.001
        if duplicate_ratio is not None and duplicate_ratio <= duplicate_ratio_threshold:
            benign_findings.append(
                Finding(
                    title="Benign pattern: sparse duplicate noise",
                    category="Benign-Pattern",
                    severity="LOW",
                    description="A very small proportion of duplicates was detected and may represent benign ingestion noise rather than systematic leakage.",
                    evidence=f"{duplicate_rows} duplicate rows out of {len(df)} ({duplicate_ratio:.2%})",
                    recommendation=[
                        "Remove duplicates as hygiene, but treat as low-priority leakage concern",
                        "Monitor duplicate rate in future data refreshes"
                    ]
                )
            )

    if (
        importance_finding
        and not corr_finding
        and not group_finding
        and not temporal_finding
        and not identifier_finding
        and not cross_proxy
        and stability.get("level") == "Stable"
        and confidence.get("level") == "High"
        and len(high_risk_findings) == 0
    ):
        benign_findings.append(
            Finding(
                title="Benign pattern: isolated strong predictor",
                category="Benign-Pattern",
                severity="LOW",
                description="A strong feature-importance signal appears without corroborating leakage signals, which can indicate a genuinely informative pre-outcome feature.",
                evidence=_as_list(importance_finding.evidence),
                recommendation=[
                    "Keep feature under review, but treat as potentially legitimate signal",
                    "Validate consistency across fresh holdout data"
                ]
            )
        )

    if temporal_finding and not corr_finding and not cross_temporal and not group_finding:
        temporal_evidence = _as_list(temporal_finding.evidence)
        has_high = any("High Target Autocorrelation" in e for e in temporal_evidence)
        has_moderate = any("Moderate Target Autocorrelation" in e for e in temporal_evidence)
        has_regular = any("Regular Time Spacing" in e for e in temporal_evidence)
        has_high_uniqueness = any("High Timestamp Uniqueness" in e for e in temporal_evidence)

        if (
            temporal_evidence
            and has_moderate
            and not has_high
            and not has_regular
            and not has_high_uniqueness
            and stability.get("level") == "Stable"
            and confidence.get("level") in {"High", "Medium"}
            and len(high_risk_findings) == 0
        ):
            benign_findings.append(
                Finding(
                    title="Benign pattern: weak temporal structure",
                    category="Benign-Pattern",
                    severity="LOW",
                    description="Temporal structure is present but only moderate and not corroborated by other predictive leakage signals.",
                    evidence=temporal_evidence,
                    recommendation=[
                        "Prefer chronological validation as precaution",
                        "Do not treat this temporal signal alone as definitive leakage"
                    ]
                )
            )

    return benign_findings


def detect_identifiers(df, threshold=None):
    if len(df) == 0:
        return None

    if threshold is None:
        threshold = 1 - (1 / np.sqrt(len(df)))

    identifier_cols = []
    for col in df.columns:
        if df[col].nunique() / len(df) > threshold:
            if pd.api.types.is_float_dtype(df[col]):
                continue
            identifier_cols.append(col)

    if identifier_cols:
        return Finding(
            title="Identifier columns detected",
            category="Structural",
            severity="MEDIUM",
            description="Columns with near-unique values allow row memorization.",
            evidence=identifier_cols,
            recommendation=[
                "Drop identifier columns before training",
                "Keep only if needed for joins"
            ]
        )
    return None


def detect_duplicates(df):
    count = df.duplicated().sum()
    if count > 0:
        return Finding(
            title="Duplicate rows detected",
            category="Hygiene",
            severity="LOW",
            description="Duplicate rows can artificially inflate model performance if they appear in both train and test sets.",
            evidence=f"{count} rows",
            recommendation=[
                "Remove duplicate rows",
                "Check data collection pipeline"
            ]
        )
    return None


def detect_group_leakage(df, target_column, uniqueness_min=0.01, uniqueness_max=0.8, group_purity_threshold=0.9, feature_constancy_threshold=0.9):
    leakage_evidence = {}
    feature_cols = [col for col in df.columns if col != target_column]

    for col in feature_cols:
        if pd.api.types.is_float_dtype(df[col]):
            continue

        n_unique = df[col].nunique()
        n_total = len(df)

        if n_unique <= 1 or n_total == 0:
            continue

        uniqueness_ratio = n_unique / n_total

        if not (uniqueness_min < uniqueness_ratio < uniqueness_max):
            continue

        group_sizes = df.groupby(col).size()
        multi_member_group_ids = group_sizes[group_sizes > 1].index

        if len(multi_member_group_ids) == 0:
            continue

        multi_member_df = df[df[col].isin(multi_member_group_ids)].copy()

        is_categorical_target = df[target_column].dtype == 'object' or df[target_column].nunique() < 20

        if is_categorical_target:
            def get_purity(group):
                if group.empty or group.value_counts().empty:
                    return 0.0
                return group.value_counts().iloc[0] / len(group)

            group_purity = multi_member_df.groupby(col)[target_column].apply(get_purity)

            if not group_purity.empty and (avg_purity := group_purity.mean()) > group_purity_threshold:
                leakage_evidence.setdefault(col, []).append(f"High target consistency (avg. purity: {avg_purity:.2f})")
        else:
            intra_group_std = multi_member_df.groupby(col)[target_column].std().fillna(0)
            avg_intra_group_std = intra_group_std.mean()
            overall_std = multi_member_df[target_column].std()

            if overall_std > 1e-6 and (avg_intra_group_std / overall_std) < 0.1:
                leakage_evidence.setdefault(col, []).append("Low target variance within groups")

        constant_features = []
        other_features = [c for c in feature_cols if c != col]

        if other_features:
            grouped_by_col = multi_member_df.groupby(col)

            for feature in other_features:
                feature_uniqueness_per_group = grouped_by_col[feature].nunique()
                num_constant_groups = (feature_uniqueness_per_group == 1).sum()
                proportion_constant = num_constant_groups / len(multi_member_group_ids)

                if proportion_constant > feature_constancy_threshold:
                    constant_features.append(feature)

        if constant_features:
            leakage_evidence.setdefault(col, []).append(f"Constant entity features: {', '.join(constant_features)}")

    if not leakage_evidence:
        return None

    final_evidence = []
    for group_col, reasons in leakage_evidence.items():
        final_evidence.append(f"Grouping column '{group_col}': {'; '.join(reasons)}")

    if final_evidence:
        return Finding(
            title="Group leakage risk detected",
            category="Structural",
            severity="HIGH",
            description="Columns were found that group data points (e.g., user_id, session_id). This can cause leakage if groups are split across train/test sets.",
            evidence=final_evidence,
            recommendation=[
                "Use GroupKFold or a similar group-aware splitting strategy, using the identified grouping column(s).",
                "Ensure that all data for a given group ID is in the same split (train or test)."
            ]
        )
    return None


def detect_high_correlation(X, y, target_name=None, threshold=None):
    y_numeric = y
    if y.dtype == 'object':
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)

    df_corr = X.copy()

    cols_to_drop = set()
    if target_name:
        cols_to_drop.add(target_name)
    if hasattr(y, 'name') and y.name:
        cols_to_drop.add(y.name)

    df_corr = df_corr.drop(columns=[c for c in cols_to_drop if c in df_corr.columns], errors='ignore')

    for col in df_corr.columns:
        if df_corr[col].dtype == 'object':
            df_corr[col] = df_corr[col].astype('category').cat.codes
        if df_corr[col].isnull().any():
            df_corr[col] = df_corr[col].fillna(df_corr[col].median())

    df_corr['target'] = y_numeric

    numeric_df_corr = df_corr.select_dtypes(include=np.number)
    correlations = numeric_df_corr.corr()['target'].abs().sort_values(ascending=False)

    if threshold is None:
        corr_mean = correlations.mean()
        corr_std = correlations.std() if len(correlations) > 1 else 0
        threshold = max(corr_mean + 3 * corr_std, 0.75)

    high_corr_features = correlations[correlations > threshold]
    evidence = high_corr_features.index.tolist()
    evidence = [f for f in evidence if f != 'target' and f not in cols_to_drop]

    if evidence:
        return Finding(
            title="High correlation with target",
            category="Statistical",
            severity="HIGH",
            description="Features with extremely high correlation to the target may be proxies for the target itself (leakage).",
            evidence=evidence,
            recommendation=[
                "Inspect these features manually",
                "Remove if they are post-outcome variables"
            ]
        )
    return None


def detect_feature_importance_leakage(X, y, target_name=None, threshold=None):
    X_processed = X.copy()

    cols_to_drop = set()
    if target_name:
        cols_to_drop.add(target_name)
    if hasattr(y, 'name') and y.name:
        cols_to_drop.add(y.name)

    X_processed = X_processed.drop(columns=[c for c in cols_to_drop if c in X_processed.columns], errors='ignore')

    for col in X_processed.columns:
        if not pd.api.types.is_numeric_dtype(X_processed[col]):
            X_processed[col] = X_processed[col].astype('category').cat.codes
        if X_processed[col].isnull().any():
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    y_processed = y
    if y_processed.dtype == 'object':
        le = LabelEncoder()
        y_processed = le.fit_transform(y_processed)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

    is_classification = y.dtype == 'object' or y.nunique() < 20

    if is_classification:
        model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X_processed.columns)

    if threshold is None:
        median_imp = importances.median()
        mask = ((importances > 2 * median_imp) & (importances > 0.10)) | (importances > 0.40)
        high_importance_features = importances[mask].index.tolist()
    else:
        high_importance_features = importances[importances > threshold].index.tolist()

    high_importance_features = [f for f in high_importance_features if f not in cols_to_drop]

    if high_importance_features:
        return Finding(
            title="High feature importance detected",
            category="Statistical",
            severity="MEDIUM",
            description="A simple model found these features to be overwhelmingly predictive, suggesting potential leakage.",
            evidence=high_importance_features,
            recommendation=[
                "Verify if these features are available at prediction time",
                "Check for target leakage"
            ]
        )
    return None


def detect_temporal_leakage(df, target_col, threshold=0.4):
    temporal_warnings = []

    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

    candidate_cols = [
        c for c in df.select_dtypes(include=['object', 'string']).columns
        if any(x in c.lower() for x in ['date', 'time', 'year', 'month', 'day'])
    ]

    for col in candidate_cols:
        try:
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() > len(sample) * 0.5:
                    date_cols.append(col)
        except Exception:
            pass

    for col in set(date_cols):
        temp_df = df[[col, target_col]].copy().dropna()

        try:
            temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce')
            temp_df = temp_df.dropna(subset=[col])
        except Exception:
            continue

        if len(temp_df) < 10:
            continue

        if temp_df[target_col].dtype == 'object':
            try:
                temp_df[target_col] = pd.to_numeric(temp_df[target_col])
            except Exception:
                le = LabelEncoder()
                temp_df[target_col] = le.fit_transform(temp_df[target_col].astype(str))

        temp_df = temp_df.sort_values(by=col)

        autocorr = temp_df[target_col].autocorr(lag=1)
        if pd.isna(autocorr):
            autocorr = 0.0

        is_regular = False
        time_diffs = temp_df[col].diff().dropna()
        if len(time_diffs) > 0:
            mode_freq = time_diffs.value_counts(normalize=True).iloc[0]
            if mode_freq > 0.8:
                is_regular = True

        n_unique = temp_df[col].nunique()
        uniqueness_ratio = n_unique / len(temp_df)
        is_unique = uniqueness_ratio > 0.95

        detected_signals = []

        if abs(autocorr) > threshold:
            detected_signals.append(f"High Target Autocorrelation ({autocorr:.2f})")
        elif abs(autocorr) > 0.1:
            detected_signals.append(f"Moderate Target Autocorrelation ({autocorr:.2f})")

        if is_regular:
            detected_signals.append("Regular Time Spacing")

        if is_unique:
            detected_signals.append("High Timestamp Uniqueness")

        if (abs(autocorr) > threshold) or (len(detected_signals) >= 2):
            temporal_warnings.append(
                f"Temporal Leakage Risk in '{col}': {', '.join(detected_signals)}. "
                "Data appears to be a time-series; use TimeSeriesSplit."
            )

    if temporal_warnings:
        return Finding(
            title="Temporal leakage risk",
            category="Structural",
            severity="HIGH",
            description="Data exhibits strong time-dependence. Random splits will cause future-to-past leakage.",
            evidence=temporal_warnings,
            recommendation=[
                "Use TimeSeriesSplit for validation",
                "Do not use random K-Fold or train_test_split"
            ]
        )
    return None


def render_report(findings, shape):
    title = Text(f"{FUNC_NAME} v{__version__}", style="bold cyan", justify="center")
    shape_text = Text(f"Dataset Shape: {shape}", justify="center")

    if not findings:
        return Group(
            title,
            Rule(),
            shape_text,
            "\n",
            Text("✅ No leakage risks detected.", style="green", justify="center")
        )

    severity_colors = {
        "HIGH": "bold red",
        "MEDIUM": "yellow",
        "LOW": "cyan"
    }

    category_order = {"Cross-Detector": 4, "Structural": 3, "Statistical": 2, "Hygiene": 1, "Benign-Pattern": 0}
    severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

    findings = sorted(findings, key=lambda f: (category_order.get(f.category, 0), severity_order.get(f.severity, 0)), reverse=True)

    table = Table(title="Findings Summary", show_lines=True)
    table.add_column("ID", justify="center")
    table.add_column("Finding", style="bold")
    table.add_column("Category", style="bold")
    table.add_column("Severity", justify="center")
    table.add_column("Key Evidence", overflow="fold")

    for idx, finding in enumerate(findings, 1):
        evidence = finding.evidence
        if isinstance(evidence, list):
            evidence = "\n".join(evidence) if evidence else "-"
        evidence = str(evidence)
        color = severity_colors.get(finding.severity, "white")
        table.add_row(str(idx), finding.title, finding.category, f"[{color}]{finding.severity}[/{color}]", evidence)

    return Group(title, Rule(), shape_text, table)


def render_advice(advice):
    risk_score = advice["leakage_score"]
    if risk_score > 10:
        risk_level = "[bold red]HIGH[/bold red]"
    elif risk_score >= 5:
        risk_level = "[yellow]MODERATE[/yellow]"
    else:
        risk_level = "[green]LOW[/green]"

    stability_level = advice["stability"]["level"]
    stability_message = advice["stability"]["message"]

    if stability_level == "Stable":
        stability_display_text = f"[green]{stability_level}[/green]"
    else:
        reasons = []
        if "samples" in (stability_message or ""):
            reasons.append("small dataset")
        if "features" in (stability_message or ""):
            reasons.append("high dimensionality")

        reason_text = " & ".join(reasons) if reasons else "instability detected"
        stability_display_text = f"[yellow]{stability_level} (Reason: {reason_text})[/yellow]"

    dashboard_text = f"""
Total Findings      : {advice['total_findings']}
Benign Findings     : {advice.get('benign_findings', 0)}
High Severity       : {advice['severity_counts']['HIGH']}
Medium Severity     : {advice['severity_counts']['MEDIUM']}
Low Severity        : {advice['severity_counts']['LOW']}
Overall Risk Score  : {risk_score}
Risk Level          : {risk_level}
Analysis Confidence : {advice['confidence']['level']} ({advice['confidence']['score']}%)
Analysis Stability  : {stability_display_text}
"""

    dashboard = Panel(dashboard_text, title=f"{FUNC_NAME.upper()} DASHBOARD", border_style="green", title_align="center")

    advice_text = f"""
Recommended Split  : [bold]{advice['splitting_strategy']}[/bold]

Notes:
"""
    for tip in advice["dataset_tips"]:
        advice_text += f"• {tip}\n"

    advice_panel = Panel(advice_text.strip(), title="Validation Advisory", border_style="blue")

    checklist_text = "Checklist:\n"
    for action in advice.get("next_actions", []):
        checklist_text += f"[ ] {action['priority']} | {action['action']}\n"
        checklist_text += f"    Why: {action['why']}\n"

    checklist_panel = Panel(checklist_text.strip(), title="Next Actions", border_style="magenta")

    return Group(dashboard, advice_panel, checklist_panel)


def _to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Series)):
        return [_to_json_safe(v) for v in value.tolist()]
    return value


def build_json_export_payload(file_path, target_column, shape, findings, advice):
    findings_payload = []
    for finding in findings:
        findings_payload.append({
            "title": finding.title,
            "category": finding.category,
            "severity": finding.severity,
            "description": finding.description,
            "evidence": _to_json_safe(finding.evidence),
            "recommendation": _to_json_safe(finding.recommendation)
        })

    return {
        "version": __version__,
        "input": {
            "file_path": file_path,
            "target_column": target_column,
            "shape": list(shape)
        },
        "summary": {
            "total_findings": advice.get("total_findings"),
            "benign_findings": advice.get("benign_findings", 0),
            "severity_counts": _to_json_safe(advice.get("severity_counts", {})),
            "risk_score": advice.get("leakage_score"),
            "splitting_strategy": advice.get("splitting_strategy"),
            "confidence": _to_json_safe(advice.get("confidence", {})),
            "stability": _to_json_safe(advice.get("stability", {}))
        },
        "dataset_tips": _to_json_safe(advice.get("dataset_tips", [])),
        "next_actions": _to_json_safe(advice.get("next_actions", [])),
        "findings": findings_payload
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=f"{FUNC_NAME} data leakage scanner")
    parser.add_argument("--file", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--json", action="store_true", help="Print JSON export to stdout")
    parser.add_argument("--json-path", default=None, help="Write JSON export to this path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_leakprofiler(
        file_path=args.file,
        target_column=args.target,
        json_output_path=args.json_path,
        json_stdout=args.json
    )
