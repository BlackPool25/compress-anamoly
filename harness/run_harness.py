"""Phase-2 matrix runner: full dataset x codec x detector x path harness.

CELL MATRIX (Phase-2: 16 datasets x 9 codec-rows, routed):

- 4 synth (seeded): Spike, Rhythm, Drift, Chaos.
- 10 UCR (registry, seed-ignored): 001/019/032/044/043/048/012/078/045/089
  short IDs (full ``<id>_<span>_<start>_<stop>`` names in ``UCR_SERIES``).
- 2 NASA (registry, seed-ignored): SMAP-P-1, MSL-T-4.

Every dataset runs::

    R0, R1, R2a, R2b, Q8, Q4, Q2 -> PCA(A), IF(A), TCN(A)   (Path-A, 21 rows)
    R4        -> D4(B)                                      (Path-B, 1 row)
    R4-bypass -> D4-hybrid(B)                               (Path-B, 1 row)
    R4-decode -> PCA(A) on Drift only (R4-decode->IF is NOT run)

Per (dataset, seed): 23 rows, 24 on Drift. Full matrix
(15*23+24)*5 = 1845 rows -> ``harness/results/phase2_pareto.csv``.

FROZEN TAU RULE: tau is calibrated per (seed, dataset, detector) on
R0-TRAIN point scores only and frozen across codecs. Concretely::

    tau = 99th percentile of detector.score(X_train)

where ``X_train`` is the R0 train slice (raw float32) and ``score()``
already returns point-level scores via trailing alignment with the
detector's train-median fill. Each codec's decoded TEST series is then
scored and evaluated against the test labels with that SAME tau.
DEVIATIONS (documented): D4 rejects float input (dtype gate), so D4 tau is
the 99th percentile of ``D4.score(train_tokens)`` — R4-train token scores,
still train-only per (seed, dataset, detector); D4 has a single codec so
"frozen across codecs" is vacuous for it. The R4-bypass hybrid tau is the
MAX (100th percentile) of the train hybrid point scores, per the T6
handoff (p99 would admit novel-bigram FPs); still train-only.

FROZEN INTEGRATION RULES (do not change without a plan amendment):
(a) Canonical raw dtype is float32: generator and loaders emit float64,
    so EVERY raw series is cast to float32 immediately after load, BEFORE
    any encode/get_ratio call (else R0 ratio=2.0, SAX get_ratio 64x raises).
(b) SAXCodec.encode raises unless len%8==0: for the R4 paths ONLY (plain
    R4 AND R4-bypass), series + labels are truncated to floor(N/8)*8
    (tail points dropped). Synthetic slices (800/1200) are already
    multiples of 8 (truncated length recorded anyway); UCR N=79795 ->
    79792, train 31918 -> 31912, test 47877 -> 47872.
(c) Q4/Q2 odd-N decode returns N+1: EVERY decode output is trimmed/padded
    to the raw length N before scoring (applies uniformly to all Path-A
    decodes; assert len(decoded)==N after alignment, raise otherwise).
(d) D4 Path-B: fit on R4-train TOKENS (decode_tokens of the encoded train
    slice), score on R4-test tokens (x8 repeat resample is inside D4.score),
    then trim/pad to the (possibly truncated) test length before metrics.
(e) Bypass Path-B (RUNNER-HANDOFF CONTRACT, T6): D4 8x8 UNCHANGED
    (``direct_symbolic.py`` untouched); fit on NON-BYPASS train tokens
    ``toks_tr[~mask_tr]``; flagged test chunks score
    ``MODEL-MAX + BYPASS_MARGIN`` (0.5) via ``hybrid_token_surprisals``
    (takes NO train-token arg); resample ``np.repeat(tok, 8)`` +
    ``_align_len``; tau = MAX of the train hybrid point scores.
    Bypass consumes raw floats only (fit/encode reject int dtype, so
    bypass+Q2 stacking fails loudly).

TIME GATE (frozen fallback ladder): the full 1845-row run must finish in
<= 360 s on 8-core CPU with peak RSS <= 2 GB. Every run logs its
wall-clock (total + per-dataset). If the budget is blown, step down ONE
rung at a time and re-measure — cells are NEVER dropped silently and
intermediate rungs are NEVER invented::

    Rung0: TCN epochs=10, MAX_TRAIN_WINDOWS=4000 (default)
    Rung1: TCN epochs=5,  MAX_TRAIN_WINDOWS=4000
    Rung2: TCN epochs=5,  MAX_TRAIN_WINDOWS=2000
    Rung3: TCN epochs=2,  MAX_TRAIN_WINDOWS=2000 (--fast profile)

CLI profiles: ``--smoke`` (Spike/R0+Q8/seed42, seconds; writes
``baseline_run.csv`` by default — Phase-1 path preserved);
``--fast`` (1 seed=42, synth-only, full codec rows, TCN epochs=2);
default (no flags) = full Phase-2 matrix; ``--seeds`` subset-override
preserved (used by the seed-42 determinism cmp); ``--offline`` preserved.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np
from tabulate import tabulate

from harness.codecs.bypass_sax import (
    BYPASS_MARGIN,
    BypassSAXCodec,
    hybrid_token_surprisals,
)
from harness.codecs.deadband import DeadbandCodec
from harness.codecs.gorilla import GorillaCodec
from harness.codecs.lossless import LosslessCodec
from harness.codecs.quantization import Q2Codec, Q4Codec, Q8Codec
from harness.codecs.symbolic import PAA_WINDOW, SAXCodec
from harness.datasets import generator
from harness.datasets.nasa_loader import NASA_SERIES
from harness.datasets.nasa_loader import load_series as nasa_load_series
from harness.datasets.ucr_loader import SERIES_NAME, cache_path
from harness.datasets.ucr_loader import load_series as ucr_load_series
from harness.detectors.direct_symbolic import DirectSymbolicDetector
from harness.detectors.isolation_forest import IsolationForestDetector
from harness.detectors.pca_detector import PCADetector
from harness.detectors.tcn_autoencoder import TCNAutoencoder
from harness.metrics.frozen_evaluator import (
    calibrate_frozen_threshold,
    evaluate_event_f1,
    evaluate_point_f1,
    evaluate_pr_auc,
)
from harness.metrics.vus import vus_pr

SEEDS: list[int] = [42, 43, 44, 45, 46]
SYNTH_DATASETS: list[str] = ["Spike", "Rhythm", "Drift", "Chaos"]
UCR_SERIES: list[str] = [
    "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620",
    "019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212",
    "032_UCR_Anomaly_DISTORTEDInternalBleeding4_1000_4675_5033",
    "044_UCR_Anomaly_DISTORTEDPowerDemand1_9000_18485_18821",
    "043_UCR_Anomaly_DISTORTEDMesoplodonDensirostris_10000_19280_19440",
    "048_UCR_Anomaly_DISTORTEDTkeepFifthMARS_3500_5988_6085",
    "012_UCR_Anomaly_DISTORTEDECG2_15000_16000_16100",
    "078_UCR_Anomaly_DISTORTEDresperation1_100000_110260_110412",
    "045_UCR_Anomaly_DISTORTEDPowerDemand2_14000_23357_23717",
    "089_UCR_Anomaly_DISTORTEDtiltAPB1_100000_114283_114350",
]
DATASETS: list[str] = SYNTH_DATASETS + UCR_SERIES + list(NASA_SERIES)

#: User-requestable codec rows. R4-decode is derived (R4 on Drift), never requested.
KNOWN_CODECS: tuple[str, ...] = (
    "R0", "R1", "R2a", "R2b", "Q8", "Q4", "Q2", "R4", "R4-bypass",
)
FULL_CODECS: list[str] = list(KNOWN_CODECS)

HEADER: list[str] = [
    "dataset", "codec", "payload_bytes", "realized_ratio", "detector",
    "path", "seed", "point_f1", "event_f1", "pr_auc", "vus_pr", "tau",
    "rmse", "gpu_model",
]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "harness" / "results" / "baseline_run.csv"

_GPU_MODEL = "cpu"


def _load_dataset(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Load (x_raw_float32, y, train_end) for a dataset; registry ignores seed."""
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
    elif name in NASA_SERIES:
        s = nasa_load_series(name)
        train_end = int(s.meta["train_end"])
    else:
        # Registry-driven UCR (10 series); unknown names raise ValueError
        # naming the series — never KeyError, never synthetic substitution.
        s = ucr_load_series(name)
        train_end = int(s.meta["train_end"])
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


def run_cell(dataset: str, seed: int, codecs: list[str],
             tcn_epochs: int = 10) -> list[dict]:
    """Run all matrix cells for one (dataset, seed); return per-seed rows.

    23 rows (24 on Drift): 7 Path-A codecs x PCA/IF/TCN, R4+D4 and
    R4-bypass+D4-hybrid on Path-B, plus R4-decode+PCA on Drift only.
    TCN trains ONCE on the R0 train slice; unknown codec names raise
    ValueError naming the codec (never a silent skip).
    """
    unknown = [c for c in codecs if c not in KNOWN_CODECS]
    if unknown:
        raise ValueError(f"unknown codec(s) {unknown}: expected subset of {list(KNOWN_CODECS)}")
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
    # TCN (Todo 8): trained ONCE on the R0 train slice; tau_TCN = p99 of
    # TCN.score(R0-train), the SAME object reused across all codecs below.
    tcn = TCNAutoencoder(seed, epochs=tcn_epochs)
    tcn.fit(x_train)
    tau_tcn = calibrate_frozen_threshold(tcn.score(x_train))
    taus = {"PCA": (pca, tau_pca), "IF": (iff, tau_if), "TCN": (tcn, tau_tcn)}

    r0 = LosslessCodec()

    def path_a(codec_name: str, payload: bytes, decoded: np.ndarray,
               raw_test: np.ndarray, labels: np.ndarray, ratio: float) -> None:
        """Score one decode with PCA/IF/TCN under the frozen taus; append path-A rows."""
        dec = _align_len(decoded, raw_test.shape[0])
        rmse = _rmse(dec, raw_test)
        for det_name in ("PCA", "IF", "TCN"):
            if codec_name == "R4-decode" and det_name in ("IF", "TCN"):
                continue  # R4-decode runs PCA only, Drift only
            det, tau = taus[det_name]
            rows.append(_row(dataset, codec_name, payload, ratio, det_name,
                             "A", seed, det.score(dec), labels, tau, rmse))

    if "R0" in codecs:
        p = r0.encode(x_test)
        path_a("R0", p, r0.decode(p), x_test, y_test, r0.get_ratio(x_test, p))
    if "R1" in codecs:
        c = GorillaCodec()
        p = c.encode(x_test)
        path_a("R1", p, c.decode(p), x_test, y_test, c.get_ratio(x_test, p))
    if "R2a" in codecs:
        c = DeadbandCodec(0.01)
        p = c.encode(x_test)
        path_a("R2a", p, c.decode(p), x_test, y_test, c.get_ratio(x_test, p))
    if "R2b" in codecs:
        c = DeadbandCodec(0.02)
        p = c.encode(x_test)
        path_a("R2b", p, c.decode(p), x_test, y_test, c.get_ratio(x_test, p))
    if "Q8" in codecs:
        c = Q8Codec()
        p = c.encode(x_test)
        path_a("Q8", p, c.decode(p), x_test, y_test, c.get_ratio(x_test, p))
    if "Q4" in codecs:
        c = Q4Codec()
        p = c.encode(x_test)
        path_a("Q4", p, c.decode(p), x_test, y_test, c.get_ratio(x_test, p))
    if "Q2" in codecs:
        c = Q2Codec()
        p = c.encode(x_test)
        path_a("Q2", p, c.decode(p), x_test, y_test, c.get_ratio(x_test, p))
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
    if "R4-bypass" in codecs:
        bcodec = BypassSAXCodec()
        xtr_tr, _ = _truncate8(x_train, y_train)
        bcodec.fit(xtr_tr)  # same floor(N/8)*8 truncation rule as SAX fit
        xte_tr, yte_tr = _truncate8(x_test, y_test)
        # Path B hybrid (T6 handoff): D4 8x8 fit on NON-BYPASS train tokens;
        # flagged test chunks score MODEL-MAX + BYPASS_MARGIN; tau = MAX of
        # the train hybrid point scores (not p99).
        d4b = DirectSymbolicDetector()
        toks_tr, mask_tr = bcodec.decode_tokens(bcodec.encode(xtr_tr))
        d4b.fit(toks_tr[~mask_tr])
        toks_te, mask_te = bcodec.decode_tokens(bcodec.encode(xte_tr))
        s_b = _align_len(
            np.repeat(hybrid_token_surprisals(d4b.transitions, toks_te, mask_te), 8),
            xte_tr.shape[0])
        tau_b = calibrate_frozen_threshold(
            _align_len(
                np.repeat(hybrid_token_surprisals(d4b.transitions, toks_tr, mask_tr), 8),
                xtr_tr.shape[0]),
            100.0)
        p_b = bcodec.encode(xte_tr)
        rows.append(_row(dataset, "R4-bypass", p_b,
                         bcodec.get_ratio(xte_tr, p_b),
                         "D4", "B", seed, s_b, yte_tr, tau_b, ""))
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
    """Parse CLI flags (--smoke, --fast, --seeds, --offline, --output)."""
    ap = argparse.ArgumentParser(description="Phase-2 compression matrix runner.")
    ap.add_argument("--smoke", action="store_true",
                    help="fast check: seed 42, Spike only, R0+Q8, PCA+IF+TCN")
    ap.add_argument("--fast", action="store_true",
                    help="1 seed (42), synth-only, full codec rows, TCN epochs=2")
    ap.add_argument("--seeds", default="",
                    help="subset override, e.g. --seeds 42,43")
    ap.add_argument("--offline", action="store_true",
                    help="skip UCR; fail loudly if UCR cache is missing")
    ap.add_argument("--output", default=str(DEFAULT_OUT),
                    help="CSV output path")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the matrix, write the 14-col CSV, print per-dataset tables."""
    args = parse_args(argv)
    tcn_epochs = 10
    if args.smoke:
        seeds = [42]
        datasets = ["Spike"]
        codecs = ["R0", "Q8"]
    elif args.fast:
        seeds = [42]
        datasets = list(SYNTH_DATASETS)
        codecs = list(FULL_CODECS)
        tcn_epochs = 2
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or SEEDS
        datasets = list(DATASETS)
        codecs = list(FULL_CODECS)
    if args.offline and any(d in UCR_SERIES or d in NASA_SERIES for d in datasets):
        from harness.datasets.nasa_loader import train_cache_path
        from harness.datasets.ucr_loader import read_freeze_rows

        rows = read_freeze_rows()
        missing = [
            r["series"]
            for r in rows
            if not cache_path(r["series"]).is_file()
            or (
                r["series"] in ("SMAP-P-1", "MSL-T-4")
                and not train_cache_path(r["series"]).is_file()
            )
        ]
        for r in rows:
            extra = (
                f" + {train_cache_path(r['series'])}"
                if r["series"] in ("SMAP-P-1", "MSL-T-4")
                else ""
            )
            print(
                f"OFFLINE: {'CACHED' if r['series'] not in missing else 'MISSING'} "
                f"{r['series']}{extra}",
                file=sys.stderr,
            )
        if missing:
            print(
                f"OFFLINE: {len(missing)} uncached series, refusing to "
                f"substitute synthetic data: {missing}",
                file=sys.stderr,
            )
        if any(
            m not in ("SMAP-P-1", "MSL-T-4") for m in missing
        ) or not cache_path(SERIES_NAME).is_file():
            print("OFFLINE: skipping uncached registry datasets (no network fetch)",
                  file=sys.stderr)
            datasets = [d for d in datasets if d not in missing]
    rows: list[dict] = []
    t_start = time.perf_counter()
    per_ds: dict[str, float] = {}
    for seed in seeds:
        for ds in datasets:
            t_cell = time.perf_counter()
            rows.extend(run_cell(ds, seed, codecs, tcn_epochs=tcn_epochs))
            per_ds[ds] = per_ds.get(ds, 0.0) + time.perf_counter() - t_cell
    wall = time.perf_counter() - t_start
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    print_tables(aggregate(rows))
    print(f"\nwrote {len(rows)} rows -> {out}")
    print(f"wall-clock {wall:.1f}s total; " +
          ", ".join(f"{d} {v:.1f}s" for d, v in sorted(per_ds.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
