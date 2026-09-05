"""Subprocess tests for eval/check_expected.py over all five fixtures."""

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "eval" / "check_expected.py"
FIX = ROOT / "eval" / "fixtures"
HEADER = ["dataset", "codec", "payload_bytes", "realized_ratio", "detector",
          "path", "seed", "point_f1", "event_f1", "pr_auc", "vus_pr", "tau",
          "rmse", "gpu_model"]


def run(csv_path: Path) -> subprocess.CompletedProcess:
    """Run the gate script against one CSV fixture."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--csv", str(csv_path)],
        capture_output=True, text=True, cwd=str(ROOT))


def test_header_schema():
    """All five fixtures carry the exact 14-col harness HEADER."""
    for name in ("pass.csv", "fail_a_collapse.csv", "fail_b_nocliff.csv",
                 "fail_c_symbolic.csv", "fail_d_rhythm.csv"):
        with (FIX / name).open(newline="") as f:
            assert csv.DictReader(f).fieldnames == HEADER, name


def test_pass():
    """pass.csv exits 0."""
    r = run(FIX / "pass.csv")
    assert r.returncode == 0, r.stdout + r.stderr


def test_fail_a_collapse_names_gate():
    """Spike Q8 collapse fails gate (a) loudly."""
    r = run(FIX / "fail_a_collapse.csv")
    assert r.returncode != 0
    assert "gate (a) NON-COLLAPSE" in (r.stdout + r.stderr)


def test_fail_b_nocliff_names_gate():
    """Drift Q4 without cliff fails gate (b)."""
    r = run(FIX / "fail_b_nocliff.csv")
    assert r.returncode != 0
    assert "gate (b) CLIFF" in (r.stdout + r.stderr)


def test_fail_c_symbolic_names_gate():
    """Drift R4-decode symbolic rescue fails gate (c)."""
    r = run(FIX / "fail_c_symbolic.csv")
    assert r.returncode != 0
    assert "gate (c) CLIFF" in (r.stdout + r.stderr)


def test_fail_d_rhythm_names_gate():
    """Rhythm R4/D4 grammar break fails gate (d)."""
    r = run(FIX / "fail_d_rhythm.csv")
    assert r.returncode != 0
    assert "gate (d) GRAMMAR" in (r.stdout + r.stderr)


def test_honest_no_knee(tmp_path):
    """Flat past R3 on every stratum prints HONEST NO-KNEE and exits 0."""
    rows = [
        {"dataset": "Spike", "codec": "R0", "payload_bytes": "1200",
         "realized_ratio": "1.0", "detector": "IF", "path": "A", "seed": "42",
         "point_f1": "0.42", "event_f1": "0.40", "pr_auc": "0.55",
         "vus_pr": "0.50", "tau": "0.81", "rmse": "0.001", "gpu_model": "cpu"},
        {"dataset": "Spike", "codec": "Q8", "payload_bytes": "302",
         "realized_ratio": "3.98", "detector": "IF", "path": "A", "seed": "42",
         "point_f1": "0.42", "event_f1": "0.40", "pr_auc": "0.55",
         "vus_pr": "0.50", "tau": "0.81", "rmse": "0.02", "gpu_model": "cpu"},
        {"dataset": "Drift", "codec": "R0", "payload_bytes": "1200",
         "realized_ratio": "1.0", "detector": "IF", "path": "A", "seed": "42",
         "point_f1": "0.60", "event_f1": "0.55", "pr_auc": "0.70",
         "vus_pr": "0.65", "tau": "0.75", "rmse": "0.001", "gpu_model": "cpu"},
        {"dataset": "Drift", "codec": "Q4", "payload_bytes": "152",
         "realized_ratio": "7.9", "detector": "IF", "path": "A", "seed": "42",
         "point_f1": "0.58", "event_f1": "0.55", "pr_auc": "0.68",
         "vus_pr": "0.63", "tau": "0.75", "rmse": "0.05", "gpu_model": "cpu"},
        {"dataset": "Drift", "codec": "R4-decode", "payload_bytes": "250",
         "realized_ratio": "19.2", "detector": "PCA", "path": "A",
         "seed": "42", "point_f1": "0.55", "event_f1": "0.50",
         "pr_auc": "0.60", "vus_pr": "0.55", "tau": "0.70", "rmse": "0.08",
         "gpu_model": "cpu"},
        {"dataset": "Rhythm", "codec": "R0", "payload_bytes": "1200",
         "realized_ratio": "1.0", "detector": "PCA", "path": "A", "seed": "42",
         "point_f1": "0.50", "event_f1": "0.48", "pr_auc": "0.62",
         "vus_pr": "0.58", "tau": "0.68", "rmse": "0.001", "gpu_model": "cpu"},
        {"dataset": "Rhythm", "codec": "R4", "payload_bytes": "250",
         "realized_ratio": "19.2", "detector": "D4", "path": "B", "seed": "42",
         "point_f1": "0.49", "event_f1": "0.47", "pr_auc": "0.61",
         "vus_pr": "0.56", "tau": "0.60", "rmse": "", "gpu_model": "cpu"},
    ]
    p = tmp_path / "flat.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    r = run(p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "HONEST NO-KNEE" in (r.stdout + r.stderr)


def test_freeze_audit_never_gates_and_never_written_for_fixtures():
    """Fail fixtures exit nonzero for their gate, and write no audit file."""
    audit = ROOT / "eval" / "freeze_audit.md"
    if audit.is_file():
        audit.unlink()
    r = run(FIX / "fail_a_collapse.csv")
    assert r.returncode != 0
    assert not audit.is_file()
