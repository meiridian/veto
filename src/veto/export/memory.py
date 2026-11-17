"""Memory probe for a Core ML export, run in a fresh process (development Mac only).

The iOS Message Filter extension loads a *compiled* model with ``.cpuOnly`` and is killed at
24 MB. This probe mirrors that load on macOS: compile the package, load it CPU-only, run 50
predictions and report resident memory (``ps`` RSS) and ``phys_footprint`` (``footprint``, the
figure jetsam enforces) before and after. It is a proxy: Core ML memory-maps weights, so the
macOS numbers do not reproduce the iOS limit and must be confirmed on a device.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTIONS = 50


def _rss_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                         text=True).stdout
    return int(out) / 1024


def _footprint_mb() -> float | None:
    try:
        out = subprocess.run(["footprint", "-p", str(os.getpid()), "--noCategories"],
                             capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)", out)
    if not m:
        return None
    return float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[m.group(2)]


def _child(compiled: str, seq_len: int) -> dict:
    import coremltools as ct
    import numpy as np

    rss0, fp0 = _rss_mb(), _footprint_mb()
    model = ct.models.CompiledMLModel(compiled, compute_units=ct.ComputeUnit.CPU_ONLY)
    rss1, fp1 = _rss_mb(), _footprint_mb()
    ids = np.zeros((1, seq_len), dtype=np.int32)
    ids[0, :5] = [101, 2000, 3000, 4000, 102]
    mask = np.zeros((1, seq_len), dtype=np.int32)
    mask[0, :5] = 1
    rss_peak, fp_peak = rss1, fp1
    for _ in range(PREDICTIONS):
        model.predict({"input_ids": ids, "attention_mask": mask})
        rss_peak = max(rss_peak, _rss_mb())
        if fp_peak is not None:
            fp_peak = max(fp_peak, _footprint_mb() or fp_peak)
    result = {"rss_growth_after_load_mb": rss1 - rss0, "rss_growth_peak_mb": rss_peak - rss0}
    if fp0 is not None and fp_peak is not None:
        result |= {"footprint_growth_after_load_mb": fp1 - fp0,
                   "footprint_growth_peak_mb": fp_peak - fp0}
    return {k: round(v, 1) for k, v in result.items()}


def measure_memory(package: Path, seq_len: int, runs: int = 3) -> dict:
    """Median of ``runs`` fresh-process measurements for the compiled, CPU-only model."""
    import coremltools as ct

    with tempfile.TemporaryDirectory() as tmp:
        compiled = Path(ct.utils.compile_model(str(package)))
        copy = Path(tmp) / "veto.mlmodelc"
        shutil.copytree(compiled, copy)
        samples = []
        for _ in range(runs):
            out = subprocess.run([sys.executable, "-m", "veto.export.memory", str(copy),
                                  str(seq_len)], capture_output=True, text=True, check=True)
            samples.append(json.loads(out.stdout.strip().splitlines()[-1]))
    keys = samples[0].keys()
    median = {k: sorted(s[k] for s in samples)[len(samples) // 2] for k in keys}
    return {"compute_units": "CPU_ONLY", "model": "compiled .mlmodelc",
            "predictions_per_run": PREDICTIONS, "runs": runs, **median,
            "note": "macOS proxy (Core ML memory-maps weights); the iOS 24 MB extension limit "
                    "must be confirmed on a device"}


if __name__ == "__main__":
    print(json.dumps(_child(sys.argv[1], int(sys.argv[2]))))
