"""Quality gates for a (possibly compressed) Core ML export, plus threshold re-derivation.

Scores come from the private labeled corpus (local only). Reports contain aggregate numbers
only; message text is never written to them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from veto.evaluation.metrics import compute_metrics
from veto.evaluation.threshold import select_threshold
from veto.export import tokenizer as tokmod
from veto.export.parity import coreml_probabilities, load_mlmodel, python_probabilities

TARGET_PRECISION = 0.95
#: Recall may drop at most this much (absolute) versus the FP16 export.
MAX_RECALL_DROP = 0.02


def split_texts(labeled_path: Path, split: str) -> tuple[list[str], np.ndarray]:
    import pandas as pd

    df = pd.read_parquet(labeled_path)
    df = df[(df["split"] == split) & df["label"].notna()]
    return df["normalized_text"].tolist(), df["label"].astype(int).to_numpy()


def score_split(package: Path, tokenizer_dir: Path, seq_len: int, texts: list[str],
                compute_units: str = "CPU_ONLY") -> np.ndarray:
    return coreml_probabilities(load_mlmodel(package, compute_units),
                                tokmod.load_tokenizer(tokenizer_dir), texts, seq_len)


def rederive_threshold(labels: np.ndarray, probs: np.ndarray,
                       target_precision: float = TARGET_PRECISION) -> dict:
    """Apply the project's rule, then move the threshold to the middle of the score gap.

    ``select_threshold`` returns a threshold that sits exactly on one validation score, so a
    reduced-precision model can flip that message. Any threshold in ``(next lower score, score]``
    gives identical validation decisions; the midpoint of that interval has the largest margin.
    """
    chosen = select_threshold(labels, probs, target_precision)
    on_score = chosen["threshold"]
    lower = np.unique(probs[probs < on_score])
    if not len(lower) or "no threshold" in chosen["reason"] or "lacks" in chosen["reason"]:
        return {"threshold": on_score, "threshold_on_score": on_score, "margin": 0.0,
                "reason": chosen["reason"]}
    prev = float(lower.max())
    mid = (prev + on_score) / 2
    return {"threshold": float(mid), "threshold_on_score": float(on_score),
            "next_lower_score": prev, "margin": float(on_score - mid),
            "reason": f"{chosen['reason']}; moved to the midpoint of the gap to the next lower "
                      f"validation score so a small score shift cannot flip the boundary case"}


def evaluate_scores(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    m = compute_metrics(labels, probs, threshold)
    return {k: m[k] for k in ("n", "positives", "threshold", "precision", "recall", "pr_auc")} | {
        "false_positives": m["false_positive_ids_count"],
        "false_negatives": m["false_negative_ids_count"]}


def delta_report(reference: np.ndarray, candidate: np.ndarray, threshold: float) -> dict:
    diff = np.abs(reference - candidate)
    flips = np.flatnonzero((reference >= threshold) != (candidate >= threshold))
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "p95_abs_diff": float(np.percentile(diff, 95)),
            "decision_disagreements": int(len(flips))}


def quality_gates(fp16_val: np.ndarray, cand_val: np.ndarray, labels: np.ndarray,
                  fp16_threshold: float) -> dict:
    """Compare a compressed model against the FP16 export on the validation set.

    Each model is evaluated at its own threshold (FP16: the shipped one; candidate: re-derived
    from its own validation scores), since compression shifts probabilities.
    """
    new = rederive_threshold(labels, cand_val)
    ref_eval = evaluate_scores(labels, fp16_val, fp16_threshold)
    cand_eval = evaluate_scores(labels, cand_val, new["threshold"])
    cand_at_old = evaluate_scores(labels, cand_val, fp16_threshold)
    gates = {
        "precision_ge_target": cand_eval["precision"] >= TARGET_PRECISION,
        "recall_within_2pts_of_fp16": cand_eval["recall"] >= ref_eval["recall"] - MAX_RECALL_DROP,
    }
    return {"threshold": new, "fp16_reference": ref_eval, "candidate": cand_eval,
            "candidate_at_fp16_threshold": cand_at_old,
            "score_deviation_vs_fp16": delta_report(fp16_val, cand_val, fp16_threshold),
            "gates": gates, "passes": all(gates.values())}


def fp32_python_scores(model_dir: Path, texts: list[str]) -> np.ndarray:
    return python_probabilities(model_dir, texts)
