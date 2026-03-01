import contextlib
import io

import pandas as pd

from LeakProfiler import (
    ADVISORY_CONFIG,
    Finding,
    _parse_args,
    advisory_logic,
    detect_high_correlation,
    determine_splitting_strategy,
    estimate_finding_confidence,
    estimate_risk_profile,
    run_leakprofiler,
)


def _mk_finding(title, category, severity, evidence=None):
    return Finding(
        title=title,
        category=category,
        severity=severity,
        description="d",
        evidence=evidence if evidence is not None else [],
        recommendation=["r"],
    )


def test_determine_splitting_strategy_temporal_precedence():
    findings = [
        _mk_finding("Group leakage risk detected", "Structural", "HIGH"),
        _mk_finding("Temporal leakage risk", "Structural", "HIGH"),
    ]
    assert determine_splitting_strategy(findings) == "TimeSeriesSplit"


def test_estimate_risk_profile_applies_overlap_penalty():
    findings = [
        _mk_finding("High correlation with target", "Statistical", "HIGH", ["f1"]),
        _mk_finding("High feature importance detected", "Statistical", "MEDIUM", ["f1"]),
        _mk_finding("Cross-detector proxy leakage consensus", "Cross-Detector", "HIGH", ["f1"]),
    ]
    confidence = {"level": "High", "score": 90}
    stability = {"level": "Stable", "message": None}

    profile = estimate_risk_profile(findings, confidence, stability)

    assert profile["score"] > 0
    assert any("proxy overlap de-duplication" in reason for reason in profile["rationale"])


def test_advisory_excludes_benign_from_severity_counts_and_keeps_count():
    findings = [
        _mk_finding("Group leakage risk detected", "Structural", "HIGH", ["g1"]),
        _mk_finding("Benign pattern: sparse duplicate noise", "Benign-Pattern", "LOW", "1 duplicate"),
    ]
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    confidence = {"level": "High", "score": 90}
    stability = {"level": "Stable", "message": None}

    advice = advisory_logic(findings, df, confidence, stability)

    assert advice["benign_findings"] == 1
    assert advice["severity_counts"]["LOW"] == 0
    assert advice["severity_counts"]["HIGH"] == 1


def test_finding_confidence_drops_for_statistical_when_unstable_and_low_confidence():
    finding = _mk_finding("High feature importance detected", "Statistical", "MEDIUM", ["f1"])
    risk_findings = [finding]

    low_conf = {"level": "Low", "score": 40}
    unstable = {"level": "Warning", "message": "few samples"}
    high_conf = {"level": "High", "score": 90}
    stable = {"level": "Stable", "message": None}

    low_case = estimate_finding_confidence(finding, risk_findings, low_conf, unstable)
    high_case = estimate_finding_confidence(finding, risk_findings, high_conf, stable)

    assert low_case["score"] < high_case["score"]


def test_high_risk_gate_downgrades_without_corroboration():
    findings = [
        _mk_finding("Group leakage risk detected", "Structural", "HIGH", ["g1"]),
    ]
    confidence = {"level": "High", "score": 90}
    stability = {"level": "Stable", "message": None}

    original_high = ADVISORY_CONFIG["risk_thresholds"]["HIGH"]
    ADVISORY_CONFIG["risk_thresholds"]["HIGH"] = 5
    try:
        profile = estimate_risk_profile(findings, confidence, stability)
    finally:
        ADVISORY_CONFIG["risk_thresholds"]["HIGH"] = original_high

    assert profile["level"] == "MODERATE"
    assert any("HIGH gate applied" in reason for reason in profile["rationale"])


def test_high_risk_gate_allows_high_with_corroboration_and_confidence():
    findings = [
        _mk_finding("Group leakage risk detected", "Structural", "HIGH", ["g1"]),
        _mk_finding("Cross-detector temporal consensus", "Cross-Detector", "HIGH", ["g1"]),
    ]
    confidence = {"level": "High", "score": 90}
    stability = {"level": "Stable", "message": None}

    original_high = ADVISORY_CONFIG["risk_thresholds"]["HIGH"]
    ADVISORY_CONFIG["risk_thresholds"]["HIGH"] = 10
    try:
        profile = estimate_risk_profile(findings, confidence, stability)
    finally:
        ADVISORY_CONFIG["risk_thresholds"]["HIGH"] = original_high

    assert profile["level"] == "HIGH"
    assert any("HIGH gate status" in reason for reason in profile["rationale"])


def test_parse_args_accepts_unicode_dash_for_target(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "leakprofiler",
            "--file",
            "dataset.csv",
            "—target",
            "label",
        ],
    )

    args = _parse_args()

    assert args.file == "dataset.csv"
    assert args.target == "label"


def test_parse_args_accepts_positional_file_and_target(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "leakprofiler",
            "dataset.csv",
            "target_col",
        ],
    )

    args = _parse_args()

    assert args.file == "dataset.csv"
    assert args.target == "target_col"


def test_parse_args_accepts_mixed_positional_file_and_target_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "leakprofiler",
            "dataset.csv",
            "--target",
            "label",
        ],
    )

    args = _parse_args()

    assert args.file == "dataset.csv"
    assert args.target == "label"


def test_risk_floor_promotes_group_signal_to_moderate():
    findings = [
        _mk_finding("Group leakage risk detected", "Structural", "HIGH", ["g1"]),
    ]
    confidence = {"level": "Low", "score": 40}
    stability = {"level": "Warning", "message": "few samples"}

    profile = estimate_risk_profile(findings, confidence, stability)

    assert profile["level"] == "MODERATE"
    assert any("Risk floor applied" in reason for reason in profile["rationale"])


def test_risk_floor_promotes_proxy_overlap_to_moderate():
    findings = [
        _mk_finding("High correlation with target", "Statistical", "MEDIUM", ["proxy_target"]),
        _mk_finding("High feature importance detected", "Statistical", "MEDIUM", ["proxy_target"]),
    ]
    confidence = {"level": "Low", "score": 45}
    stability = {"level": "Warning", "message": "few samples"}

    profile = estimate_risk_profile(findings, confidence, stability)

    assert profile["level"] == "MODERATE"
    assert any("Risk floor applied" in reason for reason in profile["rationale"])


def test_detect_high_correlation_catches_perfect_proxy():
    df = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "proxy_target": [0, 1, 0, 1, 0],
            "target": [0, 1, 0, 1, 0],
        }
    )

    finding = detect_high_correlation(df.drop(columns=["target"]), df["target"], target_name="target")

    assert finding is not None
    assert "proxy_target" in finding.evidence


def test_run_leakprofiler_drops_missing_target_rows(tmp_path):
    df = pd.DataFrame(
        {
            "feature_a": [0.2, 0.3, 0.4, 0.5],
            "feature_b": [1, 0, 1, 0],
            "target": [1, 0, None, 1],
        }
    )
    csv_path = tmp_path / "missing_target.csv"
    df.to_csv(csv_path, index=False)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        payload = run_leakprofiler(str(csv_path), target_column="target", return_payload=True)

    assert payload is not None
    assert any("Dropped 1 row(s) with missing target values" in tip for tip in payload.get("dataset_tips", []))


def test_run_leakprofiler_raises_when_target_all_missing(tmp_path):
    df = pd.DataFrame(
        {
            "feature_a": [0.2, 0.3, 0.4],
            "target": [None, None, None],
        }
    )
    csv_path = tmp_path / "all_missing_target.csv"
    df.to_csv(csv_path, index=False)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            run_leakprofiler(str(csv_path), target_column="target", return_payload=True)
            assert False, "Expected ValueError for all-missing target"
        except ValueError as exc:
            assert "contains only missing values" in str(exc)
