"""Pinned NASA (telemanom) channel loader with cache and freeze record.

Channels (Todo 1 pin, evidence ``task-1-downloads.log`` section 4)::

    SMAP-P-1  test (8505, 25) float64, 3 contextual events
              [[2149, 2349], [3539, 3779], [4536, 4844]] (test-row offsets)
    MSL-T-4   test (2217, 55) float64, 1 point event [[1172, 1240]]

FORMAT PARSER: each ``.npy`` is a 2-D float64 matrix (timesteps x features).
Column 0 is the ONLY continuous telemetry column (P-1: 4750 unique values,
5-7 MAD excursions at all 3 events; T-4: ternary {-1, 0, 1} where 1.0 occurs
only at test rows 1212-1215, inside the labeled event; train col 0 is
nominal {-1, 0}). All other columns are binary/one-hot command features and
are NOT loaded. The sample ``x`` is ``concat(train_col0, test_col0)``
float64 with ``train_end = len(train)``; labels come from
``labeled_anomalies.csv`` event intervals bound here (no label column is
embedded in the files).

NaN POLICY: any non-finite value in either signal column raises ValueError
naming the series and the file (test/train). No imputation, no dropping.

RESAMPLING: forbidden. Train and test arrays are concatenated verbatim; the
naive 40/60 cut is REJECTED (P-1's first event starts at 25.3% < 40%, so a
cut would poison train). Train is the sibling train ``.npy`` (all-nominal
by telemanom construction); the split check (train slice anomaly-free,
every event fully inside the test slice) runs BEFORE the sample is built.

Contract (identical to ucr_loader): registry-driven ``load_series(name)``
for the two freeze rows; per-series cache key ``data/cache/<name>``
(verbatim test ``.npy`` bytes, freeze-hash-verified) plus the train sidecar
``data/cache/<name>.train`` (verified against the TRAIN_* table below,
frozen from the Todo 1 evidence log -- the six-column freeze schema has no
train column); size-then-hash verify with mismatch deleting the bad file
and raising ValueError naming the series; offline-without-cache raising
RuntimeError naming the series; synthetic data never substituted; float64
parse with the runner casting float32 immediately after load.
"""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path

import numpy as np

from harness.datasets.ucr_loader import (
    TimeSeriesSample,
    download_bytes,
    download_file,  # noqa: F401  (shared resume fetcher; same contract)
    read_freeze_rows,
    verify_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
FREEZE_PATH = ROOT / "dataset-freeze.csv"

NASA_SERIES = ("SMAP-P-1", "MSL-T-4")

# Signal column: the only continuous telemetry column (see module docstring).
SIGNAL_COLUMN = 0

# Labeled event intervals as (start, stop) test-row offsets, sorted, from
# labeled_anomalies.csv via the Todo 1 evidence log.
EVENTS: dict[str, list[tuple[int, int]]] = {
    "SMAP-P-1": [(2149, 2349), (3539, 3779), (4536, 4844)],
    "MSL-T-4": [(1172, 1240)],
}

# Expected row counts (== freeze num_values for test; evidence log for train).
TEST_LEN: dict[str, int] = {"SMAP-P-1": 8505, "MSL-T-4": 2217}
TRAIN_LEN: dict[str, int] = {"SMAP-P-1": 2872, "MSL-T-4": 2272}

# Frozen sibling-train truth (Todo 1 evidence log, section 4).
TRAIN_SHA256: dict[str, str] = {
    "SMAP-P-1": "78b6eec2b3cb1e6711d07ce994c7375de94b036d839c0f06fe258f446dff667a",
    "MSL-T-4": "d137cceeedfd2c0759b9217ccb15cb29d7c8c0458d2d2515f84f54d5f0655c4c",
}
TRAIN_BYTES: dict[str, int] = {"SMAP-P-1": 574528, "MSL-T-4": 999808}

SPLIT_RULE = (
    "train=sibling train npy all-nominal by construction; "
    "test=test npy holds all labeled events"
)


def cache_path(name: str = "SMAP-P-1") -> Path:
    """Return the cache file for a pinned channel: ``data/cache/<name>``."""
    return CACHE_DIR / name


def train_cache_path(name: str = "SMAP-P-1") -> Path:
    """Return the train-sidecar cache file ``data/cache/<name>.train``."""
    return CACHE_DIR / (name + ".train")


def train_url(test_url: str) -> str:
    """Derive the sibling train URL from a test URL (``/test/`` -> ``/train/``)."""
    out = test_url.replace("/data/test/", "/data/train/")
    if out == test_url:
        raise ValueError(f"cannot derive train URL from {test_url!r}")
    return out


def _row_for(name: str) -> dict[str, str]:
    """Return the ACTIVE freeze row for a NASA series; ValueError names unknowns."""
    if name not in NASA_SERIES:
        raise ValueError(
            f"unknown NASA series {name!r}: expected one of {list(NASA_SERIES)}; "
            "refusing to substitute synthetic data"
        )
    for row in read_freeze_rows():
        if row["series"] == name:
            return row
    raise ValueError(
        f"NASA series {name!r} has no ACTIVE row in {FREEZE_PATH}; "
        "reselect via supersede, never substitute"
    )


def _checked(raw: bytes, sha256: str, nbytes: int, name: str, which: str) -> bytes:
    """Size-then-hash verify; ValueError names the series and file on mismatch."""
    if len(raw) != nbytes:
        raise ValueError(
            f"size mismatch for series {name!r} ({which}): "
            f"expected {nbytes} bytes, got {len(raw)}"
        )
    try:
        verify_sha256(raw, sha256)
    except ValueError as exc:
        raise ValueError(f"series {name!r} ({which}): {exc}") from None
    return raw


def _signal_column(raw: bytes, expected_rows: int, name: str, which: str) -> np.ndarray:
    """Decode a cached/downloaded ``.npy`` to its float64 signal column."""
    try:
        arr = np.load(io.BytesIO(raw))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"cannot decode {which} npy for series {name!r}: {exc}"
        ) from exc
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(
            f"expected a 2-D (n, k) matrix for series {name!r} ({which}), "
            f"got shape {arr.shape}"
        )
    if arr.shape[0] != expected_rows:
        raise ValueError(
            f"row-count mismatch for series {name!r} ({which}): "
            f"expected {expected_rows} rows, got {arr.shape[0]}"
        )
    col = np.asarray(arr[:, SIGNAL_COLUMN], dtype=np.float64)
    if not np.isfinite(col).all():
        raise ValueError(
            f"non-finite values in series {name!r} ({which} col {SIGNAL_COLUMN}); "
            "NaN policy is reject, never impute"
        )
    return col


def parse_nasa(test_raw: bytes, train_raw: bytes, name: str) -> TimeSeriesSample:
    """Build a TimeSeriesSample from verified test+train ``.npy`` bytes.

    The split check runs BEFORE the sample is built: every labeled event
    must sit fully inside the test slice (which holds all of them) and the
    train slice -- all-nominal by telemanom construction -- must hold zero
    labeled anomalies; otherwise ValueError naming the series is raised.
    """
    if name not in NASA_SERIES:
        raise ValueError(
            f"unknown NASA series {name!r}: expected one of {list(NASA_SERIES)}"
        )
    events = EVENTS[name]
    n_train, n_test = TRAIN_LEN[name], TEST_LEN[name]
    x_test = _signal_column(test_raw, n_test, name, "test")
    x_train = _signal_column(train_raw, n_train, name, "train")
    for start, stop in events:
        if not (0 <= start < stop <= n_test):
            raise ValueError(
                f"test-slice violation for series {name!r}: "
                f"event [{start}, {stop}) outside test bounds [0, {n_test})"
            )
    if not events:
        raise ValueError(f"no labeled events frozen for series {name!r}")
    x = np.concatenate([x_train, x_test])
    y = np.zeros(n_train + n_test, dtype=int)
    for start, stop in events:
        y[n_train + start : n_train + stop] = 1
    if int(y[:n_train].sum()) != 0:
        raise ValueError(
            f"train-slice violation for series {name!r}: "
            f"first {n_train} rows hold {int(y[:n_train].sum())} labeled anomalies"
        )
    return TimeSeriesSample(
        x=np.asarray(x, dtype=np.float64),
        y=y,
        meta={
            "series": name,
            "seed": 0,
            "split": SPLIT_RULE,
            "train_end": n_train,
            "anomaly": tuple(events),
        },
    )


def load_series(name: str = "SMAP-P-1") -> TimeSeriesSample:
    """Load a NASA channel, cache-first at ``data/cache/<name>`` + sidecar.

    Cached files are size-then-hash verified (mismatch deletes the bad file
    and raises ValueError naming the series). On cache miss the missing
    file(s) are downloaded (test URL from the freeze row, train URL from
    the sibling convention), verified, then cached. Offline without cache
    raises RuntimeError naming the series; synthetic data is never
    substituted.
    """
    row = _row_for(name)
    tpath, trpath = cache_path(name), train_cache_path(name)
    tsha, tn = row["sha256"], int(row["bytes"])
    trsha, trn = TRAIN_SHA256[name], TRAIN_BYTES[name]
    t_raw = tr_raw = None
    if tpath.is_file():
        try:
            t_raw = _checked(tpath.read_bytes(), tsha, tn, name, "test")
        except ValueError:
            tpath.unlink()
            raise
    if trpath.is_file():
        try:
            tr_raw = _checked(trpath.read_bytes(), trsha, trn, name, "train")
        except ValueError:
            trpath.unlink()
            raise
    if t_raw is None or tr_raw is None:
        try:
            if t_raw is None:
                t_raw = _checked(download_bytes(row["url"]), tsha, tn, name, "test")
            if tr_raw is None:
                tr_raw = _checked(
                    download_bytes(train_url(row["url"])), trsha, trn, name, "train"
                )
        except (ConnectionError, OSError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"offline and no cache: cannot fetch series {name!r} "
                f"(cache miss at {tpath} / {trpath}); "
                f"refusing to substitute synthetic data: {exc}"
            ) from exc
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if t_raw is not None and not tpath.is_file():
            tpath.write_bytes(t_raw)
        if tr_raw is not None and not trpath.is_file():
            trpath.write_bytes(tr_raw)
    assert t_raw is not None and tr_raw is not None
    return parse_nasa(t_raw, tr_raw, name)
