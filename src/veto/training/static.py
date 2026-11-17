"""Static-embedding classifier distilled from MiniLM, small enough for the iOS extension.

The iOS Message Filter extension is killed above 24 MB, and the MiniLM encoder cannot load under
that limit on a device even when its weights are 4-bit. This model drops the transformer:

* every vocabulary token gets a fixed vector: MiniLM's mean-pooled output for ``[CLS] t [SEP]``,
  reduced with PCA;
* only tokens that occur in the training split are kept; all other tokens (and the special
  tokens) map to a zero row, so they are ignored;
* a message vector is the L2-normalized sum of its token vectors (equal to the normalized mean),
  and a logistic regression on it gives the probability.

Token ids are unchanged: the vocabulary pruning is an id remap stored inside the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

#: Tokenizer files copied unchanged from the source encoder (token ids must not change).
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                   "special_tokens_map.json")


@dataclass
class StaticConfig:
    dims: int = 256
    C: float = 10.0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def ignored_token_ids(tokenizer) -> list[int]:
    """[PAD], [CLS], [SEP] and [MASK] never contribute ([UNK] is kept, like any token)."""
    return [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id,
            tokenizer.mask_token_id]


def distill_token_vectors(encoder_dir, cls_id: int = 101, sep_id: int = 102,
                          batch_size: int = 1024) -> np.ndarray:
    """MiniLM's mean-pooled output for ``[CLS] t [SEP]``, for every token id ``t``."""
    import torch
    from transformers import AutoModel

    enc = AutoModel.from_pretrained(encoder_dir).eval()
    vocab = enc.config.vocab_size
    out = np.zeros((vocab, enc.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, vocab, batch_size):
            ids = torch.arange(start, min(start + batch_size, vocab))
            x = torch.stack([torch.full_like(ids, cls_id), ids, torch.full_like(ids, sep_id)], 1)
            h = enc(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
            out[start:start + len(ids)] = h.mean(1).numpy()
    return out


@dataclass
class StaticModel:
    """``remap[token_id]`` is a row of ``table`` (row 0 is all zeros, for ignored tokens)."""

    remap: np.ndarray  # int32 [vocab_size]
    table: np.ndarray  # float32 [kept + 1, dims]
    coef: np.ndarray  # float32 [dims]
    intercept: float

    def features(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        rows = self.table[self.remap[input_ids]] * attention_mask[..., None]
        x = rows.sum(1)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def predict_proba(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        z = self.features(input_ids, attention_mask) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path) -> None:
        np.savez(path, remap=self.remap, table=self.table, coef=self.coef,
                 intercept=np.float32(self.intercept))

    @classmethod
    def load(cls, path) -> StaticModel:
        d = np.load(path)
        return cls(d["remap"], d["table"], d["coef"], float(d["intercept"]))

    @property
    def parameter_count(self) -> int:
        return int(self.table.size + self.coef.size + 1)


def kept_token_ids(train_ids: np.ndarray, train_mask: np.ndarray, vocab_size: int,
                   ignored: list[int]) -> np.ndarray:
    """Token ids that occur in the training split, minus the ignored special tokens."""
    seen = np.bincount(train_ids[train_mask.astype(bool)], minlength=vocab_size) > 0
    seen[list(ignored)] = False
    return np.flatnonzero(seen)


def fit_static(token_vectors: np.ndarray, train_ids: np.ndarray, train_mask: np.ndarray,
               y: np.ndarray, cfg: StaticConfig, ignored: list[int]) -> StaticModel:
    """PCA the kept token vectors to ``cfg.dims``, then fit the logistic regression."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    vocab = len(token_vectors)
    kept = kept_token_ids(train_ids, train_mask, vocab, ignored)
    reduced = PCA(cfg.dims, random_state=cfg.seed).fit_transform(token_vectors[kept])
    table = np.zeros((len(kept) + 1, cfg.dims), dtype=np.float32)
    table[1:] = reduced
    # Store the table at the precision Core ML will use, so Python and Core ML share weights.
    table = table.astype(np.float16).astype(np.float32)
    remap = np.zeros(vocab, dtype=np.int32)
    remap[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)

    model = StaticModel(remap, table, np.zeros(cfg.dims, np.float32), 0.0)
    clf = LogisticRegression(C=cfg.C, class_weight="balanced", max_iter=3000,
                             random_state=cfg.seed)
    clf.fit(model.features(train_ids, train_mask), y)
    model.coef = clf.coef_[0].astype(np.float32)
    model.intercept = float(clf.intercept_[0])
    return model


def train_static(source_dir, labeled_path, output_dir, cfg: StaticConfig | None = None) -> dict:
    """Distill, fit and save the static model; compare it with the source MiniLM model.

    The threshold uses the project's rule on validation scores (max recall subject to
    precision >= 0.95), moved to the midpoint of the gap to the next lower score.
    """
    import json
    import shutil
    from datetime import UTC, datetime
    from pathlib import Path

    import pandas as pd

    from veto.export import tokenizer as tokmod
    from veto.export.parity import python_probabilities
    from veto.export.quality import evaluate_scores, rederive_threshold

    cfg = cfg or StaticConfig()
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    source_meta = json.loads((source_dir / "metadata.json").read_text())
    if source_meta["model_type"] != "baseline":
        raise ValueError("distillation expects the MiniLM baseline model as the source")
    enc_dir = source_dir / "model" / "encoder"
    seq_len = int(source_meta["hyperparameters"]["max_length"])
    tok = tokmod.load_tokenizer(enc_dir)

    df = pd.read_parquet(labeled_path)
    df = df[df["label"].notna()]
    splits = {}
    for name in ("train", "validation", "test"):
        part = df[df["split"] == name]
        ids, mask = tokmod.encode_fixed(tok, part["normalized_text"].tolist(), seq_len)
        splits[name] = (part["normalized_text"].tolist(), ids, mask,
                        part["label"].astype(int).to_numpy())

    _, tr_ids, tr_mask, tr_y = splits["train"]
    vectors = distill_token_vectors(enc_dir, tok.cls_token_id, tok.sep_token_id)
    model = fit_static(vectors, tr_ids, tr_mask, tr_y, cfg, ignored_token_ids(tok))

    _, va_ids, va_mask, va_y = splits["validation"]
    threshold = rederive_threshold(va_y, model.predict_proba(va_ids, va_mask))
    comparison = {}
    for name in ("validation", "test"):
        texts, ids, mask, y = splits[name]
        comparison[name] = {
            "static": evaluate_scores(y, model.predict_proba(ids, mask), threshold["threshold"]),
            "minilm_baseline": evaluate_scores(y, python_probabilities(source_dir, texts),
                                               source_meta["threshold"]),
        }

    (output_dir / "model").mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model" / "static.npz")
    (output_dir / "tokenizer").mkdir(exist_ok=True)
    for name in TOKENIZER_FILES:
        if (enc_dir / name).exists():
            shutil.copy2(enc_dir / name, output_dir / "tokenizer" / name)
    meta = {
        "saved_at": datetime.now(UTC).isoformat(),
        "model_type": "static",
        "approach": "static_embedding_logreg",
        "model_version": f"{source_meta.get('model_version', 'v1')}-static",
        "base_model": source_meta["base_model"],
        "distilled_from": {"model_version": source_meta.get("model_version"),
                           "approach": source_meta.get("approach"),
                           "threshold": source_meta["threshold"]},
        "threshold": threshold["threshold"],
        "threshold_reason": threshold["reason"],
        "threshold_margin": threshold["margin"],
        "hyperparameters": {"max_length": seq_len, **cfg.to_dict(),
                            "vocabulary": "tokens seen in the training split",
                            "kept_tokens": int(len(model.table) - 1),
                            "token_vector": "MiniLM mean-pooled output of [CLS] t [SEP], PCA"},
        "parameter_count": model.parameter_count,
        "training_dataset_size": int(len(tr_y)),
        "validation_metrics": comparison["validation"]["static"],
        "test_metrics": comparison["test"]["static"],
        "comparison_with_minilm_baseline": comparison,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    return meta
