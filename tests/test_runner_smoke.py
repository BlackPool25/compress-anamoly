"""Smoke test: --smoke exits 0 with the exact 14-col CSV and >= 4 rows."""

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "harness" / "results" / "baseline_run.csv"

EXPECTED_HEADER = [
    "dataset", "codec", "payload_bytes", "realized_ratio", "detector",
    "path", "seed", "point_f1", "event_f1", "pr_auc", "vus_pr", "tau",
    "rmse", "gpu_model",
]


def test_runner_smoke() -> None:
    """Run --smoke and assert header == 14 expected names + row count >= 4."""
    r = subprocess.run(
        [sys.executable, "harness/run_harness.py", "--smoke"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"--smoke failed: {r.stderr[-2000:]}"
    assert "Spike" in r.stdout  # per-dataset Markdown table, never pooled
    with OUT.open(newline="") as f:
        rows = list(csv.DictReader(f))
        assert rows and list(rows[0].keys()) == EXPECTED_HEADER
    assert len(rows) >= 4  # spike x R0->[PCA,IF] + Q8->[PCA,IF]
