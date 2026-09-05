"""Subprocess tests for eval/check_phase2.py (amended A2(a)+B(a) semantics)."""

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "eval" / "check_phase2.py"
FIX = ROOT / "eval" / "fixtures"
HEADER = ["dataset", "codec", "payload_bytes", "realized_ratio", "detector",
          "path", "seed", "point_f1", "event_f1", "pr_auc", "vus_pr", "tau",
          "rmse", "gpu_model"]
NAMES = ("phase2_pass.csv", "phase2_fail_a1.csv", "phase2_fail_a2.csv",
         "phase2_fail_b.csv", "phase2_fail_c.csv", "phase2_fail_d.csv",
         "phase2_floor.csv", "phase2_no_knee.csv")


def run(csv_path: Path) -> subprocess.CompletedProcess:
    """Run the Phase-2 gate script against one CSV fixture."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--csv", str(csv_path)],
        capture_output=True, text=True, cwd=str(ROOT))


def test_header_schema():
    """All eight fixtures carry the exact 14-col harness HEADER."""
    for name in NAMES:
        with (FIX / name).open(newline="") as f:
            assert csv.DictReader(f).fieldnames == HEADER, name


def test_pass():
    """phase2_pass.csv exits 0 with all gates logged."""
    r = run(FIX / "phase2_pass.csv")
    assert r.returncode == 0, r.stdout + r.stderr
    for gate in ("Gate-A1", "Gate-A2(a)", "Gate-B(a)", "Gate-C", "Gate-D"):
        assert gate in r.stdout, gate
    assert "H2_FALSIFIED" in r.stdout


def _assert_only_gate_tripped(output: str, gate: str):
    """Named gate fails; no other gate fails."""
    assert f"FAIL {gate}" in output, output
    for other in ("Gate-A1", "Gate-A2(a)", "Gate-B(a)", "Gate-C", "Gate-D"):
        if other != gate:
            assert f"FAIL {other}" not in output, other


def test_fail_a1_names_only_gate_a1():
    """R1 parity break fails ONLY Gate-A1."""
    r = run(FIX / "phase2_fail_a1.csv")
    assert r.returncode != 0
    _assert_only_gate_tripped(r.stdout + r.stderr, "Gate-A1")


def test_fail_a2_names_only_gate_a2_plus_carve_note():
    """R2a band breach on non-032 fails ONLY Gate-A2(a); 032 reported, never gated."""
    r = run(FIX / "phase2_fail_a2.csv")
    assert r.returncode != 0
    _assert_only_gate_tripped(r.stdout + r.stderr, "Gate-A2(a)")
    assert "DOCUMENTED-BOUNDARY" in r.stdout
    assert "032" in r.stdout


def test_fail_b_names_only_gate_b():
    """IF<=TCN drops on both Drift legs fail ONLY Gate-B(a)."""
    r = run(FIX / "phase2_fail_b.csv")
    assert r.returncode != 0
    _assert_only_gate_tripped(r.stdout + r.stderr, "Gate-B(a)")


def test_fail_c_names_only_gate_c():
    """Spike bypass collapse fails ONLY Gate-C."""
    r = run(FIX / "phase2_fail_c.csv")
    assert r.returncode != 0
    _assert_only_gate_tripped(r.stdout + r.stderr, "Gate-C")


def test_fail_d_names_only_gate_d():
    """Missing seed-46 coverage fails ONLY Gate-D."""
    r = run(FIX / "phase2_fail_d.csv")
    assert r.returncode != 0
    _assert_only_gate_tripped(r.stdout + r.stderr, "Gate-D")


def test_floor_excluded_named_never_gating():
    """Floor series is FLOOR-EXCLUDED by name; run still exits 0."""
    r = run(FIX / "phase2_floor.csv")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FLOOR-EXCLUDED" in r.stdout
    assert "UCR-F" in r.stdout


def test_honest_no_knee_note_without_masking():
    """Flat ladder prints HONEST NO-KNEE but Gate-B(a) still fails honestly."""
    r = run(FIX / "phase2_no_knee.csv")
    assert "HONEST NO-KNEE" in (r.stdout + r.stderr)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "FAIL Gate-B(a)" in (r.stdout + r.stderr)


def test_freeze_audit_never_written_for_fixtures():
    """Fixture runs exit without writing eval/freeze_audit.md."""
    audit = ROOT / "eval" / "freeze_audit.md"
    if audit.is_file():
        audit.unlink()
    r = run(FIX / "phase2_fail_a1.csv")
    assert r.returncode != 0
    assert not audit.is_file()
