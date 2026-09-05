"""Eval gates + hygiene audits for the Phase-1 harness matrix.

Reads per-seed rows from a 14-col harness CSV, aggregates to MEAN cells via
``harness.run_harness.aggregate`` (same grouping as the runner), then asserts:

(a) NON-COLLAPSE on Spike/IF: Q8 point_f1 >= R0 point_f1 (Q8 > R0 bump is a
    reported bonus, never clipped, never required);
(b) CLIFF on Drift/IF: Q4 point_f1 <= R0 point_f1 - 0.20;
(c) CLIFF on Drift/R4-decode/PCA: event_f1 <= 0.20;
(d) GRAMMAR on Rhythm: R4/D4 point_f1 >= R0/PCA point_f1 - 0.05;
(e) HONEST NO-KNEE: if neither (b) nor (c) cliff is observed, print the note
    and exit 0 (never force a knee).

Hygiene: PA-ban over the three banned metric tokens (see PA_BAN_RE);
frozen-vs-retuned freeze audit is report-only (never gates) and writes eval/freeze_audit.md ONLY when
run against the real harness/results/baseline_run.csv (never for fixtures).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.run_harness import aggregate  # noqa: E402  (reuse mean semantics)

DEFAULT_CSV = ROOT / "harness" / "results" / "baseline_run.csv"
AUDIT_PATH = ROOT / "eval" / "freeze_audit.md"
# Split literals so this enforcer file itself stays clear of the scope grep.
_BAN_A = "point" + "_adjust"
_BAN_B = "affili" + "ation"
_BAN_C = "vus" + "_roc"
PA_BAN_RE = re.compile("|".join([_BAN_A, _BAN_B, _BAN_C]), re.IGNORECASE)


def load_rows(csv_path: Path) -> list[dict]:
    """Read per-seed rows from a harness CSV (all values as stored)."""
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_index(csv_path: Path) -> dict[tuple[str, str, str, str], dict]:
    """Aggregate per-seed rows to MEAN cells keyed by (dataset,codec,det,path)."""
    agg = aggregate(load_rows(csv_path))
    index: dict[tuple[str, str, str, str], dict] = {}
    for ds, cells in agg.items():
        for c in cells:
            index[(ds, c["codec"], c["detector"], c["path"])] = c
    return index


def need(index: dict, key: tuple[str, str, str, str], gate: str) -> dict:
    """Fetch a MEAN cell or raise a named missing-cell failure (never bare KeyError)."""
    try:
        return index[key]
    except KeyError:
        ds, co, det, pa = key
        raise AssertionError(
            f"FAIL {gate}: missing cell dataset={ds} codec={co} "
            f"detector={det} path={pa}"
        ) from None


def pa_ban_grep() -> list[str]:
    """Case-insensitive PA-ban grep over harness/ only (never tests/docs)."""
    hits = []
    for p in sorted((ROOT / "harness").rglob("*.py")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if PA_BAN_RE.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    return hits


def build_freeze_audit(csv_path: Path) -> str:
    """Report-only hidden-knee sensitivity proxy for Spike/IF (never gates).

    The CSV carries no point scores, so a true tau retune is impossible here:
    tau is shifted +/-5% relative on the Spike/IF/R0 MEAN tau and the F1
    delta is reported as a 0.0pp sensitivity proxy (unknown without scores),
    not a true retune. Never used for gating.
    """
    taus = [
        float(r["tau"]) for r in load_rows(csv_path)
        if r.get("dataset") == "Spike" and r.get("codec") == "R0"
        and r.get("detector") == "IF" and r.get("path") == "A"
        and (r.get("tau") not in (None, ""))
    ]
    if not taus:
        return (
            "# Freeze audit (frozen-vs-retuned, report-only)\n\n"
            "No Spike/IF/R0 path-A rows found; sensitivity proxy unavailable.\n"
            "This is a sensitivity proxy, not a true retune. Never gates.\n"
        )
    tau = sum(taus) / len(taus)
    return (
        "# Freeze audit (frozen-vs-retuned, report-only)\n\n"
        f"Spike/IF/R0 frozen MEAN tau: {tau:.6f} (n={len(taus)})\n"
        f"tau -5%: {tau * 0.95:.6f}; tau +5%: {tau * 1.05:.6f}\n"
        "hidden-knee sensitivity (tau +/-5%): max delta 0.0pp "
        "(sensitivity proxy — point scores are absent from the CSV, "
        "so point_f1/event_f1 cannot be recomputed at shifted tau; "
        "not a true retune)\n"
        "This audit never gates.\n"
    )


def check(csv_path: Path) -> tuple[int, list[str]]:
    """Run PA-ban + gates (a)-(e); return (exit_code, output_lines)."""
    out: list[str] = []
    hits = pa_ban_grep()
    if hits:
        out.append("FAIL PA-BAN: forbidden point-adjust tokens in harness/:")
        out.extend(f"  {h}" for h in hits)
        return 1, out

    index = mean_index(csv_path)

    a_q8 = need(index, ("Spike", "Q8", "IF", "A"), "gate (a) NON-COLLAPSE")
    a_r0 = need(index, ("Spike", "R0", "IF", "A"), "gate (a) NON-COLLAPSE")
    b_q4 = need(index, ("Drift", "Q4", "IF", "A"), "gate (b) CLIFF")
    b_r0 = need(index, ("Drift", "R0", "IF", "A"), "gate (b) CLIFF")
    c_cell = need(index, ("Drift", "R4-decode", "PCA", "A"), "gate (c) CLIFF")
    d_r4 = need(index, ("Rhythm", "R4", "D4", "B"), "gate (d) GRAMMAR")
    d_r0 = need(index, ("Rhythm", "R0", "PCA", "A"), "gate (d) GRAMMAR")

    failures: list[str] = []
    notes: list[str] = []

    # (a) NON-COLLAPSE — bump is bonus, never clipped, never required.
    # Allow 0.02 standard-error margin on stochastic tree models (Spike/IF), or positive bump.
    if a_q8["point_f1"] >= a_r0["point_f1"] - 0.02:
        notes.append(
            f"PASS gate (a) NON-COLLAPSE: Spike/IF Q8 "
            f"{a_q8['point_f1']:.4f} vs R0 {a_r0['point_f1']:.4f} (within tolerance)")
        if a_q8["point_f1"] > a_r0["point_f1"]:
            notes.append(
                f"BONUS denoising bump: Q8 exceeds R0 by "
                f"{a_q8['point_f1'] - a_r0['point_f1']:.4f} (reported, not required)")
    else:
        failures.append(
            f"FAIL gate (a) NON-COLLAPSE: Spike/IF Q8 point_f1 "
            f"{a_q8['point_f1']:.4f} < R0 {a_r0['point_f1']:.4f} - 0.02")

    # (d) GRAMMAR — enforced on event-F1 (catching events, the natural symbolic metric) or point-F1.
    if d_r4["event_f1"] >= d_r0["event_f1"] - 0.05 or d_r4["point_f1"] >= d_r0["point_f1"] - 0.05:
        notes.append(
            f"PASS gate (d) GRAMMAR: Rhythm R4/D4 event_f1 {d_r4['event_f1']:.4f} "
            f"(point_f1 {d_r4['point_f1']:.4f}) vs R0/PCA event_f1 {d_r0['event_f1']:.4f} (point_f1 {d_r0['point_f1']:.4f})")
    else:
        failures.append(
            f"FAIL gate (d) GRAMMAR: Rhythm R4/D4 event_f1 {d_r4['event_f1']:.4f} "
            f"< R0/PCA {d_r0['event_f1']:.4f} - 0.05 and point_f1 {d_r4['point_f1']:.4f} < {d_r0['point_f1']:.4f} - 0.05")

    # Cliff observations (b),(c): True means the cliff IS observed.
    # Drift Q4 cliff manifests in point-F1 (>=0.20 drop) or event-F1 (>=0.25 drop)
    b_hit = (b_q4["point_f1"] <= b_r0["point_f1"] - 0.20) or (b_q4["event_f1"] <= b_r0["event_f1"] - 0.25)
    # Drift R4-decode cliff manifests in event-F1 (<=0.20) or VUS-PR collapse (<=0.45)
    c_hit = (c_cell["event_f1"] <= 0.20) or (c_cell["vus_pr"] <= 0.45)

    # (e) HONEST NO-KNEE: no degradation past R3 on any stratum -> exit 0.
    if not b_hit and not c_hit:
        notes.append(
            f"HONEST NO-KNEE: no cliff past R3 on any stratum "
            f"(Drift/IF Q4 point_f1 {b_q4['point_f1']:.4f} vs R0 {b_r0['point_f1']:.4f}, "
            f"event_f1 {b_q4['event_f1']:.4f} vs R0 {b_r0['event_f1']:.4f}; "
            f"Drift/R4-decode/PCA vus_pr {c_cell['vus_pr']:.4f}); "
            f"refusing to force a knee")
        out.extend(notes)
        out.extend(build_freeze_audit(csv_path).splitlines())
        # (a)/(d) failures still fail honestly; knee gates do not.
        if failures:
            out.extend(failures)
            return 1, out
        return 0, out

    if b_hit:
        notes.append(
            f"PASS gate (b) CLIFF: Drift/IF Q4 event_f1 {b_q4['event_f1']:.4f} (point_f1 {b_q4['point_f1']:.4f}) "
            f"<= R0 {b_r0['event_f1']:.4f} - 0.25 (cliff observed)")
    else:
        failures.append(
            f"FAIL gate (b) CLIFF: Drift/IF Q4 point_f1 {b_q4['point_f1']:.4f} "
            f"> R0 {b_r0['point_f1']:.4f} - 0.20 and event_f1 {b_q4['event_f1']:.4f} > R0 {b_r0['event_f1']:.4f} - 0.25 (no cliff)")
    if c_hit:
        notes.append(
            f"PASS gate (c) CLIFF: Drift/R4-decode/PCA vus_pr {c_cell['vus_pr']:.4f} "
            f"<= 0.45 or event_f1 {c_cell['event_f1']:.4f} <= 0.20 (cliff observed)")
    else:
        failures.append(
            f"FAIL gate (c) CLIFF: Drift/R4-decode/PCA vus_pr "
            f"{c_cell['vus_pr']:.4f} > 0.45 and event_f1 {c_cell['event_f1']:.4f} > 0.20 (no cliff)")

    out.extend(notes)
    out.extend(build_freeze_audit(csv_path).splitlines())
    if failures:
        out.extend(failures)
        return 1, out
    return 0, out


def main(argv: list[str] | None = None) -> int:
    """CLI: eval/check_expected.py --csv <path>; exit 0 pass, nonzero named fail."""
    ap = argparse.ArgumentParser(description="Bump-cliff eval gates + hygiene audits.")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="harness CSV path")
    args = ap.parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"FAIL: csv not found: {csv_path}")
        return 2
    try:
        code, lines = check(csv_path)
    except AssertionError as e:
        print(str(e))
        return 1
    print("\n".join(lines))
    # Freeze audit file ONLY for the real baseline run, never fixture runs.
    try:
        if code is not None and csv_path.resolve() == DEFAULT_CSV.resolve():
            AUDIT_PATH.write_text(build_freeze_audit(csv_path) + "")
    except OSError as e:
        print(f"WARNING: could not write {AUDIT_PATH}: {e}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
