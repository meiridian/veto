"""Static-embedding model: fitting, id remap, persistence and the export wrapper.

Uses random token vectors and a tiny tokenizer; no MiniLM download and no message data.
"""

import numpy as np
import pytest
import torch
from export_tiny import SEQ

from conftest import STATIC_TRAIN
from veto.export.tokenizer import encode_fixed, load_tokenizer
from veto.export.wrapper import VetoStaticCoreMLModel, load_source_model
from veto.inference import VetoClassifier
from veto.training.static import StaticModel, ignored_token_ids, kept_token_ids

UNSEEN = "stop reply a"  # words never used in STATIC_TRAIN


def load(saved_static):
    return (StaticModel.load(saved_static / "model" / "static.npz"),
            load_tokenizer(saved_static / "tokenizer"))


def test_only_training_tokens_are_kept_and_specials_are_ignored(saved_static):
    model, tok = load(saved_static)
    ids, mask = encode_fixed(tok, STATIC_TRAIN, SEQ)
    kept = kept_token_ids(ids, mask, len(model.remap), ignored_token_ids(tok))
    assert (model.remap[kept] > 0).all()
    assert len(model.table) == len(kept) + 1 and not model.table[0].any()
    for special in ignored_token_ids(tok):
        assert model.remap[special] == 0
    unseen_ids, unseen_mask = encode_fixed(tok, [UNSEEN], SEQ)
    assert (model.remap[unseen_ids[unseen_mask.astype(bool)]] == 0).all()


def test_message_of_unseen_tokens_scores_the_intercept(saved_static):
    model, tok = load(saved_static)
    p = model.predict_proba(*encode_fixed(tok, [UNSEEN], SEQ))[0]
    assert p == pytest.approx(1 / (1 + np.exp(-model.intercept)))


def test_scores_are_probabilities_and_padding_invariant(saved_static):
    model, tok = load(saved_static)
    short = model.predict_proba(*encode_fixed(tok, STATIC_TRAIN, SEQ))
    long = model.predict_proba(*encode_fixed(tok, STATIC_TRAIN, 2 * SEQ))
    assert ((short > 0) & (short < 1)).all()
    assert np.allclose(short, long, atol=1e-6)
    assert short[:4].mean() > short[4:].mean()  # learned the toy labels


def test_save_load_roundtrip(saved_static, tmp_path):
    model, _ = load(saved_static)
    model.save(tmp_path / "copy.npz")
    again = StaticModel.load(tmp_path / "copy.npz")
    assert np.array_equal(model.remap, again.remap) and np.array_equal(model.table, again.table)
    assert model.parameter_count == model.table.size + model.coef.size + 1


def test_table_is_stored_at_fp16_precision(saved_static):
    model, _ = load(saved_static)
    assert np.array_equal(model.table, model.table.astype(np.float16).astype(np.float32))


def test_python_inference_api_loads_static_model(saved_static):
    model, tok = load(saved_static)
    clf = VetoClassifier.load(saved_static, device="cpu")
    expected = model.predict_proba(*encode_fixed(tok, STATIC_TRAIN, SEQ))
    assert np.allclose(clf.predict_proba(STATIC_TRAIN), expected)
    assert clf.threshold == 0.42


def test_export_wrapper_matches_numpy_model(saved_static):
    model, tok = load(saved_static)
    src = load_source_model(saved_static)
    assert isinstance(src.wrapper, VetoStaticCoreMLModel) and src.model_type == "static"
    assert src.wrapper.seq_len == SEQ and src.tokenizer_dir == saved_static / "tokenizer"
    ids, mask = encode_fixed(tok, STATIC_TRAIN + [UNSEEN], SEQ)
    expected = model.predict_proba(ids, mask)
    for i in range(len(ids)):
        with torch.no_grad():
            got = src.wrapper(torch.tensor(ids[i:i + 1]), torch.tensor(mask[i:i + 1])).item()
        assert got == pytest.approx(expected[i], abs=1e-6)
