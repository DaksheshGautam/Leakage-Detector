import pandas as pd

from LeakProfiler import (
    ADVISORY_CONFIG,
    Finding,
    advisory_logic,
    determine_splitting_strategy,
    estimate_finding_confidence,
    estimate_risk_profile,
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
