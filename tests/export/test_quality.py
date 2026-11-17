"""Threshold re-derivation and quality gates (synthetic scores only; no Core ML needed)."""

import numpy as np
import pytest

from veto.evaluation.threshold import select_threshold
from veto.export import quality
from veto.export.parity import compare


def scores():
    y = np.array([0] * 10 + [1] * 4)
    p = np.array([0.05, 0.1, 0.12, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 0.85, 0.95])
    return y, p


def test_rederived_threshold_gives_same_validation_decisions_with_margin():
    y, p = scores()
    on_score = select_threshold(y, p)["threshold"]
    new = quality.rederive_threshold(y, p)
    assert new["threshold"] < on_score and new["threshold_on_score"] == on_score
    assert new["next_lower_score"] < new["threshold"] < on_score
    assert new["margin"] == pytest.approx(on_score - new["threshold"])
    assert ((p >= new["threshold"]) == (p >= on_score)).all()
    assert "midpoint" in new["reason"]


def test_rederive_falls_back_cleanly_without_both_classes():
    new = quality.rederive_threshold(np.zeros(5, dtype=int), np.linspace(0.1, 0.9, 5))
    assert new["margin"] == 0.0 and 0.0 < new["threshold"] < 1.0


def test_quality_gates_report_each_model_at_its_own_threshold():
    y, p = scores()
    shifted = np.clip(p - 0.1, 0, 1)  # compression lowered every score
    out = quality.quality_gates(p, shifted, y, select_threshold(y, p)["threshold"])
    assert out["candidate"]["threshold"] == out["threshold"]["threshold"]
    assert out["candidate_at_fp16_threshold"]["recall"] <= out["candidate"]["recall"]
    assert out["passes"] is all(out["gates"].values())
    assert out["score_deviation_vs_fp16"]["max_abs_diff"] == pytest.approx(0.1)


def test_gates_fail_when_precision_drops():
    y = np.array([0] * 6 + [1] * 2)
    good = np.array([0.1] * 6 + [0.9, 0.95])
    bad = np.array([0.9, 0.92, 0.94, 0.1, 0.1, 0.1, 0.5, 0.6])  # negatives outrank positives
    out = quality.quality_gates(good, bad, y, 0.9)
    assert not out["passes"]


def test_compare_supports_a_separate_reference_threshold():
    py = np.array([0.80, 0.90])
    ml = np.array([0.70, 0.90])
    same = compare(py, ml, 0.85)
    assert same["classification_disagreements"] == 0
    split = compare(py, ml, 0.65, reference_threshold=0.85)  # compressed model, lower threshold
    assert split["classification_disagreements"] == 1 and split["disagreement_indices"] == [0]
