"""Core ML conversion tests. Skipped unless coremltools is installed and the OS is macOS."""

import json
import sys

import numpy as np
import pytest
import torch

ct = pytest.importorskip("coremltools")
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Core ML prediction needs macOS")

from export_tiny import SEQ  # noqa: E402

from veto.export import parity  # noqa: E402
from veto.export import tokenizer as tokmod  # noqa: E402
from veto.export.coreml import (  # noqa: E402
    INPUT_NAMES,
    OUTPUT_NAME,
    build_metadata,
    convert_wrapper,
    export_coreml,
)
from veto.export.wrapper import load_source_model  # noqa: E402

TEXTS = ["vote early now", "hello friend dinner tonight lol", "chip in to the campaign and stop"]


@pytest.mark.parametrize("fixture,tol", [("saved_baseline", 2e-3), ("saved_finetuned", 2e-3)])
def test_fp16_coreml_matches_torch_wrapper(request, fixture, tol):
    src = load_source_model(request.getfixturevalue(fixture))
    ml = convert_wrapper(src.wrapper, "fp16")
    tok = tokmod.load_tokenizer(src.tokenizer_dir)
    ids, mask = tokmod.encode_fixed(tok, TEXTS, SEQ)
    for i in range(len(TEXTS)):
        with torch.no_grad():
            ref = src.wrapper(torch.tensor(ids[i:i + 1]).long(), torch.tensor(mask[i:i + 1]).long())
        out = ml.predict({INPUT_NAMES[0]: ids[i:i + 1], INPUT_NAMES[1]: mask[i:i + 1]})
        assert abs(float(np.asarray(out[OUTPUT_NAME]).reshape(-1)[0]) - ref.item()) < tol


def test_public_input_and_output_contract(saved_baseline):
    ml = convert_wrapper(load_source_model(saved_baseline).wrapper, "fp16")
    spec = ml.get_spec()
    assert [i.name for i in spec.description.input] == list(INPUT_NAMES)
    assert [o.name for o in spec.description.output] == [OUTPUT_NAME]
    for i in spec.description.input:
        assert list(i.type.multiArrayType.shape) == [1, SEQ]
    assert spec.WhichOneof("Type") == "mlProgram"


def test_export_writes_artifact_and_metadata(saved_baseline, tmp_path):
    out = tmp_path / "dist" / "coreml"
    res = export_coreml(saved_baseline, out, "fp16")
    assert (out / "veto.mlpackage").exists() and (out / "metadata.json").exists()
    assert sorted(p.name for p in (out / "tokenizer").iterdir()) == [
        "golden_tokenization.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    golden = json.loads((out / "tokenizer" / "golden_tokenization.json").read_text())
    tok = tokmod.load_tokenizer(out / "tokenizer")
    for ex in golden["examples"][:10]:
        ids, mask = tokmod.encode_fixed(tok, [ex["text"]], SEQ)
        assert ids[0, :ex["length"]].tolist() == ex["input_ids"][:SEQ]
        assert mask.sum() == ex["length"]
    meta = json.loads((out / "metadata.json").read_text())
    required = ["model_name", "model_version", "base_model", "model_type", "coreml_format",
                "precision", "threshold", "positive_label", "negative_label",
                "max_sequence_length", "tokenizer", "input_names", "output_names",
                "source_model_version", "export_timestamp", "coremltools_version",
                "torch_version", "transformers_version"]
    assert all(k in meta for k in required)
    # Values come from the source model's metadata, not constants.
    assert meta["threshold"] == 0.42 and meta["model_version"] == "v-test"
    assert meta["base_model"] == "tiny-test-model" and meta["max_sequence_length"] == SEQ
    assert meta["coreml_format"] == "mlprogram" and meta["precision"] == "fp16"
    assert meta["positive_label"] == "political_solicitation"
    assert meta["artifact_sizes_bytes"]["total"] == res["sizes"]["total"] > 0
    # The threshold is not baked into the graph: the output is a raw probability.
    ml = parity.load_mlmodel(out / "veto.mlpackage", "CPU_ONLY")
    tok = tokmod.load_tokenizer(out / "tokenizer")
    p = parity.coreml_probabilities(ml, tok, TEXTS, SEQ)
    assert ((p >= 0) & (p <= 1)).all() and not np.isin(p, [0.0, 1.0]).all()


def test_two_exports_of_the_same_model_agree(saved_baseline, tmp_path):
    outs = [tmp_path / n for n in ("a", "b")]
    probs = []
    for out in outs:
        export_coreml(saved_baseline, out, "fp16")
        ml = parity.load_mlmodel(out / "veto.mlpackage", "CPU_ONLY")
        tok = tokmod.load_tokenizer(out / "tokenizer")
        probs.append(parity.coreml_probabilities(ml, tok, TEXTS, SEQ))
    assert np.array_equal(probs[0], probs[1])


def test_run_parity_end_to_end_on_tiny_model(saved_baseline, tmp_path):
    """Fixtures + bands run through the real parity machinery (FP32 should be near-exact)."""
    out = tmp_path / "fp32"
    export_coreml(saved_baseline, out, "fp32")
    rep = parity.run_parity(saved_baseline, out / "veto.mlpackage", out / "tokenizer", SEQ,
                            threshold=0.42, tolerance=parity.PROB_TOLERANCE_FP32,
                            compute_units="CPU_ONLY")
    assert rep["fixtures"]["n"] >= 60
    assert rep["fixtures"]["classification_disagreements"] == 0
    assert rep["fixtures"]["max_abs_diff"] < parity.PROB_TOLERANCE_FP32
    assert parity.verdict(rep) in ("PASS", "PASS_WITH_NOTES")


def test_build_metadata_uses_source_threshold(saved_baseline):
    src = load_source_model(saved_baseline)
    meta = build_metadata(src, "fp16", "iOS17", {"mlpackage": 1, "tokenizer": 1, "metadata": 1})
    assert meta["threshold"] == src.metadata["threshold"]
    assert meta["minimum_deployment_target"] == "iOS17"


@pytest.mark.parametrize("compression", ["palettize4", "int8"])
def test_compressed_export_keeps_contract_and_records_method(saved_baseline, tmp_path, compression):
    plain, small = tmp_path / "plain", tmp_path / compression
    export_coreml(saved_baseline, plain, "fp16")
    export_coreml(saved_baseline, small, "fp16", compression=compression)

    spec = parity.load_mlmodel(small / "veto.mlpackage", "CPU_ONLY").get_spec()
    assert [i.name for i in spec.description.input] == list(INPUT_NAMES)
    assert [o.name for o in spec.description.output] == [OUTPUT_NAME]
    for i in spec.description.input:
        assert list(i.type.multiArrayType.shape) == [1, SEQ]

    meta = json.loads((small / "metadata.json").read_text())
    assert meta["precision"] == compression and meta["compute_precision"] == "fp16"
    assert meta["compression"]["method"] in ("palettize_weights", "linear_quantize_weights")
    assert meta["compression"]["nbits"] == (4 if compression == "palettize4" else 8)
    assert 0.0 < meta["threshold"] < 1.0
    assert meta["input_dtype"] == "int32" and meta["max_sequence_length"] == SEQ
    assert json.loads((plain / "metadata.json").read_text())["compression"] == {"method": "none"}

    ml = parity.load_mlmodel(small / "veto.mlpackage", "CPU_ONLY")
    p = parity.coreml_probabilities(ml, tokmod.load_tokenizer(small / "tokenizer"), TEXTS, SEQ)
    assert ((p >= 0) & (p <= 1)).all()


def test_unknown_compression_is_rejected(saved_baseline, tmp_path):
    with pytest.raises(ValueError, match="compression"):
        export_coreml(saved_baseline, tmp_path / "x", "fp16", compression="palettize3")


def test_generated_bundle_fits_the_extension_budget():
    """If a local ``dist/coreml`` bundle exists and is compressed, it must stay under budget.

    The Message Filter extension is killed at 24 MB, so the model file has a hard ceiling.
    """
    from pathlib import Path

    meta_path = Path("dist/coreml/metadata.json")
    if not meta_path.exists():
        pytest.skip("no local dist/coreml bundle")
    meta = json.loads(meta_path.read_text())
    static = meta["model_type"].startswith("static")
    if meta.get("compression", {}).get("method", "none") == "none" and not static:
        pytest.skip("local bundle is the uncompressed MiniLM FP16 reference")
    assert meta["artifact_sizes_bytes"]["mlpackage"] <= (5_000_000 if static else 12_000_000)
    assert 0.0 < meta["threshold"] < 1.0
    assert meta["input_names"] == list(INPUT_NAMES) and meta["output_names"] == [OUTPUT_NAME]
    assert meta["input_dtype"] == "int32"
    assert meta["input_shapes"] == {n: [1, meta["max_sequence_length"]] for n in INPUT_NAMES}


def test_static_model_exports_with_same_contract_and_exact_parity(saved_static, tmp_path):
    out = tmp_path / "static_export"
    export_coreml(saved_static, out, "fp16")
    spec = parity.load_mlmodel(out / "veto.mlpackage", "CPU_ONLY").get_spec()
    assert [i.name for i in spec.description.input] == list(INPUT_NAMES)
    assert [o.name for o in spec.description.output] == [OUTPUT_NAME]
    for i in spec.description.input:
        assert list(i.type.multiArrayType.shape) == [1, SEQ]
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["model_type"].startswith("static token embeddings")
    assert meta["compute_precision"] == "fp16 table lookup, fp32 arithmetic"
    rep = parity.run_parity(saved_static, out / "veto.mlpackage", out / "tokenizer", SEQ,
                            threshold=0.42, tolerance=1e-3, compute_units="CPU_ONLY")
    assert rep["fixtures"]["classification_disagreements"] == 0
    assert rep["fixtures"]["max_abs_diff"] < 1e-3
