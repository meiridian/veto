"""``veto export coreml``: export, verify against Python inference, report."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from veto.export import parity, quality
from veto.export.benchmark import benchmark
from veto.export.coreml import export_coreml
from veto.export.memory import measure_memory

REPORT_PATH = Path("artifacts/coreml_export_report.json")
#: Compressed models deviate from the Python reference far more than FP16 does (observed
#: maxima up to ~0.4 for 4-bit); the real acceptance checks are the decision-based quality
#: gates. This bound only catches a corrupted model.
PROB_TOLERANCE_COMPRESSED = 0.5


def _mb(n: int) -> str:
    return f"{n / 1e6:.2f} MB"


def _summary(label: str, s: dict) -> None:
    print(f"    {label:<14} n={s['n']:<5} max={s['max_abs_diff']:.5f} "
          f"mean={s['mean_abs_diff']:.5f} p95={s['p95_abs_diff']:.5f} "
          f"disagreements={s['classification_disagreements']}"
          + (f" (within precision noise: {s['disagreements_within_precision_noise']})"
             if s["classification_disagreements"] else ""))


def _verify_compressed(args, result: dict, meta: dict) -> tuple[str, dict, dict]:
    """Quality gates for a compressed export; re-derives and records its threshold.

    Compares against the uncompressed FP16 export (built in a temp dir) on the full validation
    set, then reports the held-out test split with the same thresholds.
    """
    ref_threshold = meta["threshold"]
    seq = meta["max_sequence_length"]
    tok_dir = args.output / "tokenizer"
    val_texts, val_y = quality.split_texts(args.labeled, "validation")
    test_texts, test_y = quality.split_texts(args.labeled, "test")
    with tempfile.TemporaryDirectory() as tmp:
        ref = export_coreml(args.model_dir, Path(tmp) / "fp16", "fp16", args.deployment_target)
        fp16_val = quality.score_split(ref["package"], tok_dir, seq, val_texts)
        fp16_test = quality.score_split(ref["package"], tok_dir, seq, test_texts)
        ref_sizes = ref["sizes"]
    cand_val = quality.score_split(result["package"], tok_dir, seq, val_texts)
    cand_test = quality.score_split(result["package"], tok_dir, seq, test_texts)

    gates = quality.quality_gates(fp16_val, cand_val, val_y, ref_threshold)
    new_threshold = gates["threshold"]["threshold"]
    gates["test_split"] = {
        "fp16_reference": quality.evaluate_scores(test_y, fp16_test, ref_threshold),
        "candidate": quality.evaluate_scores(test_y, cand_test, new_threshold),
        "note": "held-out: the threshold was chosen on validation only",
    }
    gates["fp16_reference_sizes_bytes"] = ref_sizes

    report = {"compressed": parity.run_parity(
        args.model_dir, result["package"], tok_dir, seq, new_threshold,
        PROB_TOLERANCE_COMPRESSED, local_labeled=None, compute_units="CPU_ONLY",
        reference_threshold=ref_threshold)}
    fx = report["compressed"]["fixtures"]
    flips = fx["classification_disagreements"]
    if not gates["passes"] or fx["max_abs_diff"] > PROB_TOLERANCE_COMPRESSED:
        status = "FAIL"
    else:
        status = "PASS" if flips == 0 else "PASS_WITH_EXCEPTIONS"

    meta["threshold_fp16_reference"] = ref_threshold
    meta["threshold"] = new_threshold
    meta["threshold_selection"] = gates["threshold"]["reason"]
    meta["threshold_margin"] = gates["threshold"]["margin"]
    return status, report, gates


def run_export_coreml(args) -> int:
    compressed = args.compression != "none"
    if compressed and args.precision != "fp16":
        print("Compression is applied on top of the FP16 export; use --precision fp16.")
        return 2
    if compressed and not args.no_verify and not args.labeled.exists():
        print(f"A compressed model needs validation data to re-derive its threshold, but "
              f"{args.labeled} does not exist. Run the corpus pipeline first (or pass --no-verify "
              f"to keep the FP16 threshold, unvalidated).")
        return 2
    print("Loading selected Veto model...")
    result = export_coreml(args.model_dir, args.output, args.precision, args.deployment_target,
                           args.compression)
    src, meta, sizes = result["source"], result["metadata"], result["sizes"]
    print(f"  Source model:  {meta['source_approach']} ({src.model_type}), {meta['model_version']}")
    print(f"  Base model:    {meta['base_model']}")
    print(f"  Format:        Core ML {meta['coreml_format']}, {args.precision.upper()}, "
          f"{args.deployment_target}+, input [1, {meta['max_sequence_length']}]"
          + (f", compression {args.compression}" if compressed else ""))
    print(f"  Threshold:     {meta['threshold']:.4f} (stored in metadata; not in the graph)")

    status, report = "NOT_VERIFIED", {}
    tol = parity.PROB_TOLERANCE_FP16 if args.precision == "fp16" else parity.PROB_TOLERANCE_FP32
    gates: dict = {}
    if not args.no_verify and compressed:
        print("\nValidating the compressed export against the FP16 export and Python inference...")
        status, report, gates = _verify_compressed(args, result, meta)
        tol = PROB_TOLERANCE_COMPRESSED
        for name, key in (("validation, FP16 export", "fp16_reference"),
                          ("validation, compressed", "candidate")):
            m = gates[key]
            print(f"  {name:<26} precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"pr_auc={m['pr_auc']:.4f} fp={m['false_positives']} fn={m['false_negatives']}")
        for name, key in (("test (held-out), FP16", "fp16_reference"),
                          ("test (held-out), compressed", "candidate")):
            m = gates["test_split"][key]
            print(f"  {name:<26} precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"pr_auc={m['pr_auc']:.4f} fp={m['false_positives']} fn={m['false_negatives']}")
        d = gates["score_deviation_vs_fp16"]
        print(f"  score deviation vs FP16: max={d['max_abs_diff']:.4f} "
              f"mean={d['mean_abs_diff']:.5f} p95={d['p95_abs_diff']:.4f}")
        print(f"  threshold: {meta['threshold_fp16_reference']:.4f} -> {meta['threshold']:.4f} "
              f"(margin {meta['threshold_margin']:.4f})")
        _summary("fixtures", report["compressed"]["fixtures"])
        print(f"\nCompressed export: {status}")
    elif not args.no_verify:
        print("\nValidating export against Python inference...")
        local = args.labeled if args.labeled.exists() else None
        if local is None:
            print(f"  (no local corpus at {args.labeled}; skipping the large local comparison)")
        kwargs = dict(model_dir=args.model_dir, package=result["package"],
                      tokenizer_dir=args.output / "tokenizer", seq_len=meta["max_sequence_length"],
                      threshold=meta["threshold"], tolerance=tol, local_labeled=local,
                      local_n=args.local_sample)
        report[args.precision] = parity.run_parity(**kwargs, compute_units="ALL")
        report[f"{args.precision}_cpu_only"] = parity.run_parity(**kwargs, compute_units="CPU_ONLY")
        if args.precision == "fp16":  # FP32 Core ML reference, to separate precision from bugs
            with tempfile.TemporaryDirectory() as tmp:
                ref = export_coreml(args.model_dir, Path(tmp) / "fp32", "fp32",
                                    args.deployment_target)
                kwargs.update(package=ref["package"], tolerance=parity.PROB_TOLERANCE_FP32)
                report["fp32_reference"] = parity.run_parity(**kwargs, compute_units="ALL")
        verdicts = {k: parity.verdict(v) for k, v in report.items()}
        order = ["FAIL", "PASS_WITH_NOTES", "PASS"]
        status = min(verdicts.values(), key=order.index)
        for name, rep in report.items():
            print(f"  [{name}] compute units: {rep['compute_units']}  verdict: {verdicts[name]}")
            _summary("fixtures", rep["fixtures"])
            for band, s in rep["threshold_bands"].items():
                if s.get("n"):
                    _summary(f"band:{band}", s)
            if "local_corpus" in rep:
                _summary("local corpus", rep["local_corpus"])
        print(f"\nPython/Core ML parity: {status}")
        if status == "PASS_WITH_NOTES":
            print("  Flips occur only for messages whose Python probability is within the "
                  "FP16 deviation of the threshold.")

    bench = None
    if args.benchmark:
        print("\nBenchmarking (development Mac; not an iPhone measurement)...")
        bench = benchmark(result["package"], args.output / "tokenizer", meta["max_sequence_length"])
        print(f"  Machine: {bench['machine']}")
        print("  Measuring memory (compiled model, CPU only, fresh process)...")
        bench["memory"] = measure_memory(result["package"], meta["max_sequence_length"])
        print(f"  Memory growth (peak, load + {bench['memory']['predictions_per_run']} predictions)"
              f": RSS {bench['memory']['rss_growth_peak_mb']} MB, footprint "
              f"{bench['memory'].get('footprint_growth_peak_mb', 'n/a')} MB")
        print(f"  Tokenization: median {bench['tokenization']['median_ms']} ms, "
              f"p95 {bench['tokenization']['p95_ms']} ms")
        for k, v in bench.items():
            if k.startswith("coreml_"):
                print(f"  {k}: inference median {v['inference']['median_ms']} ms "
                      f"(p95 {v['inference']['p95_ms']}); with tokenization "
                      f"{v['total_with_tokenization']['median_ms']} ms")

    # Record the validation in metadata.json so the artifact is self-describing.
    meta_path = args.output / "metadata.json"
    fx = report.get("compressed" if compressed else args.precision, {}).get("fixtures", {})
    meta["validation"] = {
        "status": status, "tolerance": tol,
        "fixture_examples": fx.get("n"),
        "fixtures_max_abs_diff": fx.get("max_abs_diff"),
        "fixtures_classification_disagreements": fx.get("classification_disagreements"),
        "reference": "Veto Python inference (veto.inference.VetoClassifier)",
    }
    if gates:
        fx_flips = fx.get("classification_disagreements")
        meta["validation"]["quality_gates"] = {
            k: gates[k] for k in ("threshold", "fp16_reference", "candidate", "test_split",
                                  "score_deviation_vs_fp16", "gates", "fp16_reference_sizes_bytes")}
        meta["validation"]["fixture_decision_exceptions_vs_fp16_reference"] = fx_flips
    if bench:
        meta["benchmark"] = bench
    for _ in range(3):  # the recorded size includes the file itself; iterate until stable
        meta["artifact_sizes_bytes"] = sizes
        meta_path.write_text(json.dumps(meta, indent=2))
        new = meta_path.stat().st_size
        if new == sizes["metadata"]:
            break
        sizes["metadata"] = new
        sizes["total"] = sizes["mlpackage"] + sizes["tokenizer"] + new

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({"parity": report, "quality": gates, "benchmark": bench,
                                       "sizes": sizes, "metadata": meta}, indent=2, default=float))

    print("\nArtifact:")
    print(f"  {result['package']}")
    print("Size:")
    print(f"  veto.mlpackage   {_mb(sizes['mlpackage'])}")
    print(f"  tokenizer/       {_mb(sizes['tokenizer'])}")
    print(f"  metadata.json    {sizes['metadata']} bytes")
    print(f"  total            {_mb(sizes['total'])}")
    print(f"Metadata:\n  {meta_path}\nReport:\n  {REPORT_PATH} (aggregate numbers only)")
    return 1 if status == "FAIL" else 0
