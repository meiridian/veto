"""Tiny offline models for export tests (no downloads, no private data)."""

import json

import joblib
import numpy as np
import pytest
import torch
from export_tiny import SEQ, tiny_config, tiny_tokenizer
from sklearn.linear_model import LogisticRegression
from transformers import BertForSequenceClassification, BertModel


@pytest.fixture
def tokenizer(tmp_path):
    return tiny_tokenizer(tmp_path)


@pytest.fixture
def tiny_bert():
    torch.manual_seed(0)
    return BertModel(tiny_config(), add_pooling_layer=True).eval()


def _write_meta(model_dir, model_type):
    (model_dir / "metadata.json").write_text(json.dumps({
        "model_type": model_type, "approach": f"tiny_{model_type}", "base_model": "tiny-test-model",
        "model_version": "v-test", "threshold": 0.42, "saved_at": "2026-01-01T00:00:00+00:00",
        "threshold_reason": "test", "hyperparameters": {"max_length": SEQ},
    }))


@pytest.fixture
def saved_baseline(tmp_path):
    """A saved baseline model dir: tiny encoder + a fitted logistic regression."""
    torch.manual_seed(1)
    d = tmp_path / "baseline"
    enc = d / "model" / "encoder"
    bert = BertModel(tiny_config(), add_pooling_layer=False).eval()
    bert.save_pretrained(enc)
    tiny_tokenizer(tmp_path).save_pretrained(enc)
    rng = np.random.default_rng(0)
    clf = LogisticRegression(class_weight="balanced").fit(
        rng.normal(size=(40, 32)), np.array([0, 1] * 20))
    joblib.dump(clf, d / "model" / "logreg.joblib")
    _write_meta(d, "baseline")
    return d


@pytest.fixture
def saved_finetuned(tmp_path):
    torch.manual_seed(2)
    d = tmp_path / "finetuned"
    model = BertForSequenceClassification(tiny_config()).eval()
    model.save_pretrained(d / "model")
    tiny_tokenizer(tmp_path).save_pretrained(d / "tokenizer")
    _write_meta(d, "finetuned")
    return d


STATIC_TRAIN = ["vote early now", "donate now to the campaign", "chip in to the campaign",
                "vote and donate", "hello friend", "dinner tonight lol", "you and me tonight",
                "hello hello friend"]
STATIC_LABELS = [1, 1, 1, 1, 0, 0, 0, 0]


@pytest.fixture
def saved_static(tmp_path):
    """A saved static-embedding model dir built from random token vectors (no MiniLM)."""
    from export_tiny import WORDS

    from veto.export.tokenizer import encode_fixed
    from veto.training.static import StaticConfig, fit_static, ignored_token_ids

    d = tmp_path / "static"
    tok = tiny_tokenizer(tmp_path)
    vectors = np.random.default_rng(3).normal(size=(5 + len(WORDS), 16)).astype(np.float32)
    ids, mask = encode_fixed(tok, STATIC_TRAIN, SEQ)
    model = fit_static(vectors, ids, mask, np.array(STATIC_LABELS), StaticConfig(dims=6, C=1.0),
                       ignored_token_ids(tok))
    (d / "model").mkdir(parents=True)
    model.save(d / "model" / "static.npz")
    tok.save_pretrained(d / "tokenizer")
    _write_meta(d, "static")
    return d
