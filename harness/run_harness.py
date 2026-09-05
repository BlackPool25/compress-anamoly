"""Phase-1 matrix runner: full dataset x codec x detector x path harness.

CELL MATRIX (dataset x codec x detector x path) — every dataset in
[Spike, Rhythm, Drift, Chaos, UCR] runs::

    R0  -> PCA(A), IF(A)
    Q8  -> PCA(A), IF(A)
    Q4  -> PCA(A), IF(A)
    R4  -> D4(B)
    R4-decode -> PCA(A) on Drift only (R4-decode->IF is NOT run)

FROZEN TAU RULE: tau is calibrated per (seed, dataset, detector) on
R0-TRAIN point scores only and frozen across codecs. Concretely::

    tau = 99th percentile of detector.score(X_train)

where ``X_train`` is the R0 train slice (raw float32) and ``score()``
already returns point-level scores via trailing alignment with the
detector's train-median fill. Each codec's decoded TEST series is then
scored and evaluated against the test labels with that SAME tau.
DEVIATION (documented): D4 rejects float input (dtype gate), so D4 tau is
the 99th percentile of ``D4.score(train_tokens)`` — R4-train token scores,
still train-only per (seed, dataset, detector); D4 has a single codec so
"frozen across codecs" is vacuous for it.

FROZEN INTEGRATION RULES (do not change without a plan amendment):
(a) Canonical raw dtype is float32: generator and ucr_loader emit float64,
    so EVERY raw series is cast to float32 immediately after load, BEFORE
    any encode/get_ratio call (else R0 ratio=2.0, SAX get_ratio 64x raises).
(b) SAXCodec.encode raises unless len%8==0: for the R4 path ONLY, series +
    labels are truncated to floor(N/8)*8 (tail points dropped). Synthetic
    slices (800/1200) are already multiples of 8 (truncated length recorded
    anyway); UCR N=79795 -> 79792, train 31918 -> 31912, test 47877 -> 47872.
(c) Q4 odd-N decode returns N+1: EVERY decode output is trimmed/padded to
    the raw length N before scoring (applies uniformly to Q8/Q4/R0-Path-A;
    assert len(decoded)==N after alignment, raise otherwise).
(d) D4 Path-B: fit on R4-train TOKENS (decode_tokens of the encoded train
    slice), score on R4-test tokens (x8 repeat resample is inside D4.score),
    then trim/pad to the (possibly truncated) test length before metrics.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from pathlib import Path

import numpy as np
from tabulate import tabulate

from harness.codecs.lossless import LosslessCodec
from harness.codecs.quantization import Q4Codec, Q8Codec
from harness.codecs.symbolic import PAA_WINDOW, SAXCodec
from harness.datasets import generator
from harness.datasets.ucr_loader import SERIES_NAME, cache_path, load_series
from harness.detectors.direct_symbolic import DirectSymbolicDetector
from harness.detectors.isolation_forest import IsolationForestDetector
from harness.detectors.pca_detector import PCADetector
from harness.metrics.frozen_evaluator import (
    calibrate_frozen_threshold,
    evaluate_event_f1,
    evaluate_point_f1,
    evaluate_pr_auc,
)
from harness.metrics.vus import vus_pr

SEEDS: list[int] = [42, 43, 44, 45, 46]
DATASETS: list[str] = ["Spike", "Rhythm", "Drift", "Chaos", "UCR"]

HEADER: list[str] = [
    "dataset", "codec", "payload_bytes", "realized_ratio", "detector",
    "path", "seed", "point_f1", "event_f1", "pr_auc", "vus_pr", "tau",
    "rmse", "gpu_model",
]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "harness" / "results" / "baseline_run.csv"

_GPU_MODEL = "cpu"


def _load_dataset(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Load (x_raw_float32, y, train_end) for a dataset; UCR ignores seed."""
    if name == "Spike":
        s = generator.make_spike(seed)
        train_end = generator.TRAIN_END
    elif name == "Rhythm":
        s = generator.make_rhythm(seed)
        train_end = generator.TRAIN_END
    elif name == "Drift":
        s = generator.make_drift(seed)
        train_end = generator.TRAIN_END
    elif name == "Chaos":
        s = generator.make_chaos(seed)
        train_end = generator.TRAIN_END
    elif name == "UCR":
        s = load_series()
        train_end = int(s.meta["train_end"])
    else:  # pragma: no cover - CLI-validated
        raise ValueError(f"unknown dataset: {name}")
    x = np.ascontiguousarray(s.x, dtype=np.float32).ravel()
    y = np.asarray(s.y, dtype=int).ravel()
    assert x.shape[0] == y.shape[0]
    return x, y, train_end


def _align_len(decoded: np.ndarray, n: int) -> np.ndarray:
    """Trim/pad a decode output to raw length N; raise if still mismatched."""
    d = np.asarray(decoded).ravel()
    if d.size > n:
        d = d[:n]
    elif d.size < n:
        d = np.pad(d, (0, n - d.size), mode="edge")
    if d.size != n:
        raise ValueError(f"decode alignment failed: {d.size} != raw N={n}")
    return d


def _truncate8(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R4-only: truncate series+labels to floor(N/8)*8 (tail dropped)."""
    n_trunc = (x.shape[0] // PAA_WINDOW) * PAA_WINDOW
    return x[:n_trunc], y[:n_trunc]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _row(dataset: str, codec: str, payload: bytes, ratio: float, detector: str,
         path: str, seed: int, scores: np.ndarray, y_test: np.ndarray,
         tau: float, rmse: str) -> dict:
    return {
        "dataset": dataset, "codec": codec, "payload_bytes": len(payload),
        "realized_ratio": ratio, "detector": detector, "path": path,
        "seed": seed, "point_f1": evaluate_point_f1(y_test, scores, tau),
        "event_f1": evaluate_event_f1(y_test, scores, tau),
        "pr_auc": evaluate_pr_auc(y_test, scores),
        "vus_pr": vus_pr(y_test, scores), "tau": tau, "rmse": rmse,
        "gpu_model": _GPU_MODEL,
    }


def run_cell(dataset: str, seed: int, codecs: list[str]) -> list[dict]:
    """Run all matrix cells for one (dataset, seed); return per-seed rows."""
    x_raw, y, train_end = _load_dataset(dataset, seed)
    x_train, y_train = x_raw[:train_end], y[:train_end]
    x_test, y_test = x_raw[train_end:], y[train_end:]
    rows: list[dict] = []

    # Fit classical detectors on the R0 train slice; freeze tau per detector.
    pca = PCADetector(seed)
    pca.fit(x_train)
    tau_pca = calibrate_frozen_threshold(pca.score(x_train))
    iff = IsolationForestDetector(seed)
    iff.fit(x_train)
    tau_if = calibrate_frozen_threshold(iff.score(x_train))
    taus = {"PCA": (pca, tau_pca), "IF": (iff, tau_if)}

    r0 = LosslessCodec()

    def path_a(codec_name: str, payload: bytes, decoded: np.ndarray,
               raw_test: np.ndarray, labels: np.ndarray, ratio: float) -> None:
        dec = _align_len(decoded, raw_test.shape[0])
        rmse = _rmse(dec, raw_test)
        for det_name in ("PCA", "IF"):
            if codec_name == "R4-decode" and det_name == "IF":
                continue  # R4-decode->IF is NOT run
            det, tau = taus[det_name]
            rows.append(_row(dataset, codec_name, payload, ratio, det_name,
                             "A", seed, det.score(dec), labels, tau, rmse))

    if "R0" in codecs:
        p = r0.encode(x_test)
        path_a("R0", p, r0.decode(p), x_test, y_test, r0.get_ratio(x_test, p))
    if "Q8" in codecs:
        c = Q8Codec()
        p = c.encode(x_test)
        path_a("Q8", p, c.decode(p), x_test, y_test, c.get_ratio(x_test, p))
    if "Q4" in codecs:
        c = Q4Codec()
        p = c.encode(x_test)
        path_a("Q4", p, c.decode(p), x_test, y_test, c.get_ratio(x_test, p))
    if "R4" in codecs:
        sax = SAXCodec()
        xtr_tr, _ = _truncate8(x_train, y_train)
        sax.fit(xtr_tr)  # same truncation rule as the test slice
        xte_tr, yte_tr = _truncate8(x_test, y_test)
        # Path B (D4 direct): fit on R4-train tokens, score R4-test tokens.
        d4 = DirectSymbolicDetector()
        d4.fit(sax.decode_tokens(sax.encode(xtr_tr)))
        tok_test = sax.decode_tokens(sax.encode(xte_tr))
        s_b = _align_len(d4.score(tok_test), xte_tr.shape[0])
        tau_d4 = calibrate_frozen_threshold(
            d4.score(sax.decode_tokens(sax.encode(xtr_tr))))
        p_r4 = sax.encode(xte_tr)
        rows.append(_row(dataset, "R4", p_r4,
                         sax.get_ratio(xte_tr, p_r4),
                         "D4", "B", seed, s_b, yte_tr, tau_d4, ""))
        # R4-decode -> PCA(A) on Drift only.
        if dataset == "Drift":
            p = sax.encode(xte_tr)
            path_a("R4-decode", p, sax.decode(p), xte_tr, yte_tr,
                   float(xte_tr.nbytes / max(len(p), 1)))
    return rows


def aggregate(rows: list[dict]) -> dict[str, list[dict]]:
    """Group per-seed rows by (dataset,codec,detector,path): mean + sample std."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(
            (r["dataset"], r["codec"], r["detector"], r["path"]), []).append(r)
    out: dict[str, list[dict]] = {}
    for (ds, co, det, pa), rs in sorted(groups.items()):
        n = len(rs)
        cell: dict = {"codec": co, "detector": det, "path": pa, "n": n}
        for m in ("point_f1", "event_f1", "pr_auc", "vus_pr"):
            v = np.array([r[m] for r in rs], dtype=float)
            cell[m] = float(np.mean(v))
            cell[m + "_std"] = float(np.std(v, ddof=1)) if n > 1 else 0.0
        out.setdefault(ds, []).append(cell)
    return out


def print_tables(agg: dict[str, list[dict]]) -> None:
    """Print one Markdown table per dataset (never pooled)."""
    for ds in sorted(agg):
        print(f"\n## {ds}")
        print(tabulate(
            [[c["codec"], c["detector"], c["path"], c["n"],
              f"{c['point_f1']:.4f} ± {c['point_f1_std']:.4f}",
              f"{c['event_f1']:.4f} ± {c['event_f1_std']:.4f}",
              f"{c['pr_auc']:.4f} ± {c['pr_auc_std']:.4f}",
              f"{c['vus_pr']:.4f} ± {c['vus_pr_std']:.4f}"]
             for c in agg[ds]],
            headers=["codec", "detector", "path", "n", "point_f1",
                     "event_f1", "pr_auc", "vus_pr"],
            tablefmt="github"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase-1 compression matrix runner.")
    ap.add_argument("--smoke", action="store_true",
                    help="fast check: seed 42, Spike only, R0+Q8, PCA+IF")
    ap.add_argument("--seeds", default="",
                    help="subset override, e.g. --seeds 42,43")
    ap.add_argument("--offline", action="store_true",
                    help="skip UCR; fail loudly if UCR cache is missing")
    ap.add_argument("--output", default=str(DEFAULT_OUT),
                    help="CSV output path")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        seeds = [42]
        datasets = ["Spike"]
        codecs = ["R0", "Q8"]
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or SEEDS
        datasets = list(DATASETS)
        codecs = ["R0", "Q8", "Q4", "R4"]
    if args.offline and "UCR" in datasets:
        if not cache_path(SERIES_NAME).is_file():
            raise RuntimeError(
                f"offline and no UCR cache at {cache_path(SERIES_NAME)}; "
                "refusing to substitute synthetic data")
        print("OFFLINE: skipping UCR dataset (no network fetch)", file=sys.stderr)
        datasets = [d for d in datasets if d != "UCR"]
    rows: list[dict] = []
    for seed in seeds:
        for ds in datasets:
            rows.extend(run_cell(ds, seed, codecs))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    print_tables(aggregate(rows))
    print(f"\nwrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
