"""Convert the Veto export wrapper to a Core ML ML Program and write the dist/ artifact."""

from __future__ import annotations

import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from veto.export import tokenizer as tokmod
from veto.export.fixtures import load_fixtures
from veto.export.wrapper import SourceModel, load_source_model, sha256_of

INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAME = "political_solicitation_probability"
POSITIVE_LABEL = "political_solicitation"
NEGATIVE_LABEL = "not_political_solicitation"
DEFAULT_TARGET = "iOS17"
PRECISIONS = ("fp16", "fp32")
#: Post-conversion weight compression. ``none`` keeps the plain FP16/FP32 export (the reference).
COMPRESSIONS = ("none", "int8", "palettize4", "palettize6")


def _import_ct():
    try:
        import coremltools as ct
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "coremltools is required for Core ML export. Install it with: uv sync --extra coreml"
        ) from exc
    import logging

    logging.getLogger("coremltools").setLevel(logging.WARNING)  # drop per-op INFO noise
    return ct


def convert_wrapper(wrapper, precision: str = "fp16", target: str = DEFAULT_TARGET):
    """Trace ``wrapper`` and convert it to a Core ML ML Program (fixed shape ``[1, L]``)."""
    import torch

    ct = _import_ct()
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    seq = wrapper.seq_len
    ids = torch.zeros(1, seq, dtype=torch.long)
    mask = torch.ones(1, seq, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ids, mask), strict=False)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=INPUT_NAMES[0], shape=(1, seq), dtype=np.int32),
                ct.TensorType(name=INPUT_NAMES[1], shape=(1, seq), dtype=np.int32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=getattr(ct.target, target),
    )


def compress_mlmodel(mlmodel, compression: str):
    """Apply weight compression to a converted ML Program.

    ``palettizeN`` replaces every weight tensor with an N-bit k-means lookup table (the large
    embedding table benefits most); ``int8`` is symmetric linear quantization. Activations stay
    FP16, and input/output names, shapes and dtypes are untouched.
    """
    if compression == "none":
        return mlmodel
    if compression not in COMPRESSIONS:
        raise ValueError(f"compression must be one of {COMPRESSIONS}")
    _import_ct()
    import coremltools.optimize.coreml as cto

    if compression == "int8":
        config = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=512))
        return cto.linear_quantize_weights(mlmodel, config)
    nbits = int(compression.removeprefix("palettize"))
    config = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
        mode="kmeans", nbits=nbits))
    return cto.palettize_weights(mlmodel, config)


def compression_info(compression: str) -> dict:
    """The ``compression`` block recorded in ``metadata.json``."""
    if compression == "none":
        return {"method": "none"}
    if compression == "int8":
        return {"method": "linear_quantize_weights", "mode": "linear_symmetric", "nbits": 8,
                "scope": "all weight tensors with at least 512 elements"}
    return {"method": "palettize_weights", "mode": "kmeans",
            "nbits": int(compression.removeprefix("palettize")), "granularity": "per_tensor",
            "scope": "all weight tensors"}


def dir_size(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def library_versions() -> dict:
    import coremltools
    import torch
    import transformers

    return {"coremltools_version": coremltools.__version__, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version()}


def build_metadata(src: SourceModel, precision: str, target: str, sizes: dict,
                   compression: str = "none") -> dict:
    """Describe the artifact. Every value comes from the source model or the export itself."""
    meta = src.metadata
    seq = src.wrapper.seq_len
    kind = src.model_type
    return {
        "model_name": "veto",
        "model_version": meta.get("model_version", "unknown"),
        "base_model": meta["base_model"],
        "model_type": ("MiniLM encoder + mean pooling + L2 norm + logistic regression"
                       if kind == "baseline"
                       else "fine-tuned MiniLM sequence classifier (pooler + 2-class head)"),
        "source_approach": meta.get("approach"),
        "coreml_format": "mlprogram",
        "precision": precision if compression == "none" else compression,
        "compute_precision": precision,
        "compression": compression_info(compression),
        "minimum_deployment_target": target,
        "compute_note": "Fixed input shape; Core ML chooses CPU/GPU/Neural Engine at runtime.",
        "threshold": meta["threshold"],
        "threshold_rule": "should_filter = probability >= threshold (applied by the app, "
                          "not part of the model graph)",
        "threshold_selection": meta.get("threshold_reason"),
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "max_sequence_length": seq,
        "tokenizer": tokmod.describe_tokenizer(src.tokenizer_dir, seq),
        "tokenizer_assets": "tokenizer/ (tokenizer.json, tokenizer_config.json, vocab.txt, "
                            "golden_tokenization.json)",
        "input_names": list(INPUT_NAMES),
        "input_shapes": {n: [1, seq] for n in INPUT_NAMES},
        "input_dtype": "int32",
        "output_names": [OUTPUT_NAME],
        "output_description": "Probability in [0, 1] that the message is political solicitation.",
        "source_model_version": meta.get("model_version", "unknown"),
        "source_training_timestamp": meta.get("saved_at"),
        "source_weights_sha256": sha256_of(src.weight_files),
        "export_timestamp": datetime.now(UTC).isoformat(),
        "artifact_sizes_bytes": sizes,
        **library_versions(),
    }


def export_coreml(model_dir: Path, output: Path, precision: str = "fp16",
                  target: str = DEFAULT_TARGET, compression: str = "none") -> dict:
    """Export the saved Veto model to ``output`` and return a summary.

    Writes ``veto.mlpackage``, ``tokenizer/`` and ``metadata.json``; an existing ``output`` is
    replaced so the directory always matches exactly what would be handed to a consumer.
    """
    output = Path(output)
    src = load_source_model(model_dir)
    mlmodel = compress_mlmodel(convert_wrapper(src.wrapper, precision, target), compression)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    pkg = output / "veto.mlpackage"
    mlmodel.save(str(pkg))
    tokmod.export_tokenizer_assets(src.tokenizer_dir, output / "tokenizer")
    tokmod.export_golden_tokenization(
        tokmod.load_tokenizer(src.tokenizer_dir), [f["text"] for f in load_fixtures()],
        src.wrapper.seq_len, output / "tokenizer")

    sizes = {"mlpackage": dir_size(pkg), "tokenizer": dir_size(output / "tokenizer")}
    sizes["metadata"] = 0  # filled in after writing
    meta = build_metadata(src, precision, target, sizes, compression)
    meta_path = output / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    sizes["metadata"] = meta_path.stat().st_size
    sizes["total"] = sizes["mlpackage"] + sizes["tokenizer"] + sizes["metadata"]
    meta["artifact_sizes_bytes"] = sizes
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"output": output, "package": pkg, "metadata": meta, "sizes": sizes, "source": src}
