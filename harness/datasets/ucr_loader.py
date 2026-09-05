"""Pinned UCR time-series-anomaly series loader with cache and freeze record.

Phase A pin (verified 2026-09-05 by re-download + re-hash + byte-compare)::

    series = 001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620
    url    = https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/...
    sha256 = 4a9c39e6...9973bca3 (see SERIES_SHA256)
    bytes  = 1356515

The series file is one float per line; the single labeled anomaly interval
``[ANOMALY_START, ANOMALY_END)`` is encoded in the file name, following the
UCR Time Series Anomaly Archive convention (Wu & Keogh 2021). The bytes are
mirrored verbatim from the archive by the ML-KULeuven/dtaianomaly repo; the
URL pins an immutable commit so the hash cannot drift.
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # Sibling worker owns harness.datasets.generator concurrently; it may
    # not exist yet, so fall back to the frozen contract definition below.
    from harness.datasets.generator import TimeSeriesSample  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on sibling worker timing

    @dataclass(frozen=True)
    class TimeSeriesSample:
        """Frozen contract: x float series, y binary labels, meta mapping."""

        x: np.ndarray
        y: np.ndarray
        meta: dict[str, Any] = field(default_factory=dict)


TIMEOUT_S = 30
MAX_RETRIES = 3

SERIES_NAME = "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620"
SERIES_URL = (
    "https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/"
    "5d94855fdd5d6b4803797457886a1c498c171e16/"
    "data/UCR-time-series-anomaly-archive/"
    "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
)
SERIES_SHA256 = "4a9c39e6d2cbd819909df7d20cf553afa995daa3155611f9972960199973bca3"
SERIES_BYTES = 1356515
SPLIT_RULE = (
    "train=first 40% anomaly-free; "
    "test=last 60% holds exactly one contiguous labeled anomaly"
)
FREEZE_COLUMNS = ("series", "url", "sha256", "bytes", "license", "split_rule")

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
FREEZE_PATH = ROOT / "dataset-freeze.csv"

_NAME_RE = re.compile(r"_(\d+)_(\d+)_(\d+)\.txt$")


def cache_path(name: str = SERIES_NAME) -> Path:
    """Return the cache file for a pinned series: ``data/cache/<name>``."""
    return CACHE_DIR / name


def download_bytes(url: str = SERIES_URL) -> bytes:
    """Download URL with timeout=30s and up to 3 retries; return raw bytes."""
    last: Exception | None = None
    for _ in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
    raise ConnectionError(f"download failed after {MAX_RETRIES} retries: {url}: {last}")


def verify_sha256(data: bytes, expected: str = SERIES_SHA256) -> None:
    """Raise ValueError if the SHA256 of data differs from expected."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: expected {expected}, got {actual}")


def _anomaly_span(name: str) -> tuple[int, int]:
    """Parse the ``_<traintest>_<start>_<stop>.txt`` suffix of a series name."""
    m = _NAME_RE.search(name if name.endswith(".txt") else name + ".txt")
    if m is None:
        raise ValueError(f"cannot parse anomaly span from series name: {name!r}")
    return int(m.group(2)), int(m.group(3))


def parse_series(raw: bytes, series: str = SERIES_NAME) -> TimeSeriesSample:
    """Parse pinned-series bytes to a TimeSeriesSample.

    Labels come from the anomaly span encoded in the series name. The split
    check runs BEFORE the train/test cut: the first 40% must hold zero
    labeled anomalies and the full series must hold exactly one contiguous
    labeled region fully inside the last 60%; otherwise ValueError naming
    the violation is raised.
    """
    x = np.loadtxt(io.BytesIO(raw), dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"expected a non-empty 1-D series, got shape {x.shape}")
    n = int(x.size)
    cut = int(0.4 * n)
    start, stop = _anomaly_span(series)
    if not (0 <= start < stop <= n):
        raise ValueError(
            f"anomaly span [{start}, {stop}) out of bounds for series length {n}"
        )
    y = np.zeros(n, dtype=int)
    y[start:stop] = 1
    if int(y[:cut].sum()) != 0:
        raise ValueError(
            f"train-slice violation: first 40% [0, {cut}) holds "
            f"{int(y[:cut].sum())} labeled anomalies, expected 0"
        )
    (idx,) = np.nonzero(y)
    gaps = np.nonzero(np.diff(idx) > 1)[0]
    n_regions = int(gaps.size) + 1
    if n_regions != 1 or int(idx[0]) < cut:
        raise ValueError(
            "test-slice violation: expected exactly one contiguous labeled "
            f"anomaly region fully inside the last 60% [{cut}, {n}), got "
            f"{n_regions} region(s) starting at {int(idx[0])}"
        )
    return TimeSeriesSample(
        x=np.asarray(x, dtype=float),
        y=y,
        meta={
            "series": series,
            "seed": 0,
            "split": SPLIT_RULE,
            "train_end": cut,
            "anomaly": (start, stop),
        },
    )


def load_series(
    name: str = SERIES_NAME,
    url: str = SERIES_URL,
    sha256: str = SERIES_SHA256,
) -> TimeSeriesSample:
    """Load the pinned series, cache-first at ``data/cache/<name>``.

    A cached file is hash-verified (mismatch deletes the file and raises).
    Without cache the series is downloaded, verified, then cached. Offline
    without cache raises RuntimeError loudly; synthetic data is never
    substituted.
    """
    path = cache_path(name)
    if path.is_file():
        raw = path.read_bytes()
        try:
            verify_sha256(raw, sha256)
        except ValueError:
            path.unlink()
            raise
        if len(raw) != SERIES_BYTES and name == SERIES_NAME and url == SERIES_URL:
            path.unlink()
            raise ValueError(
                f"size mismatch: expected {SERIES_BYTES} bytes, got {len(raw)}"
            )
        return parse_series(raw, series=name)
    try:
        raw = download_bytes(url)
    except (ConnectionError, OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"offline and no cache: cannot fetch {url} "
            f"(cache miss at {path}); refusing to substitute synthetic data: {exc}"
        ) from exc
    verify_sha256(raw, sha256)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return parse_series(raw, series=name)


def read_freeze_rows(path: Path = FREEZE_PATH) -> list[dict[str, str]]:
    """Read and schema-validate ALL ACTIVE rows of the freeze file.

    ``# SUPERSEDED ...`` and other ``#`` comment lines are skipped; the
    header must carry EXACTLY the six columns
    ``series,url,sha256,bytes,license,split_rule`` and at least one live
    row must remain. UCR rows sourced from the official archive zip share
    one URL/SHA-256/bytes cell triple (the zip itself); the ``series`` cell
    names the zip member (per-member bytes/hashes/split verdicts live in
    the Todo 1 evidence log, since the six-column schema has no member
    column). NASA rows pin one HF test ``.npy`` per channel; the sibling
    train file is named in the row's ``split_rule`` for Todo 2's loader.
    """
    rows = [
        line.split(",")
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not rows:
        raise ValueError(f"freeze file is empty: {path}")
    if tuple(rows[0]) != FREEZE_COLUMNS:
        raise ValueError(f"freeze header mismatch: {rows[0]} != {list(FREEZE_COLUMNS)}")
    live = rows[1:]
    if not live:
        raise ValueError(f"freeze file holds no ACTIVE rows: {path}")
    out: list[dict[str, str]] = []
    for row in live:
        if len(row) != len(FREEZE_COLUMNS):
            raise ValueError(f"freeze row must have 6 columns, got {len(row)}: {row}")
        out.append(dict(zip(FREEZE_COLUMNS, row, strict=True)))
    return out


def read_freeze_row(path: Path = FREEZE_PATH) -> dict[str, str]:
    """Read the first ACTIVE UCR row of the freeze file (backward compat).

    Kept working after the 1→12 migration by returning the first live row
    instead of demanding exactly one; new code should prefer
    ``read_freeze_rows``.
    """
    return read_freeze_rows(path)[0]


def supersede_freeze_row(
    new_row: dict[str, str],
    path: Path = FREEZE_PATH,
    series: str | None = None,
) -> None:
    """Mark ACTIVE row(s) SUPERSEDED and pin a replacement ACTIVE row.

    RESELECTION RULE: if a pinned series violates the split check, comment
    the old row out as ``# SUPERSEDED <csv>`` and append the new row, so the
    file keeps EXACTLY the six schema columns. Pass ``series`` to supersede
    only that row (multi-row registry); omit it for the legacy behavior of
    superseding every ACTIVE row (single-row files).
    """
    if tuple(new_row.keys()) != FREEZE_COLUMNS:
        raise ValueError(f"new row must carry EXACTLY {FREEZE_COLUMNS}")
    lines = path.read_text().splitlines()
    out: list[str] = []
    superseded = False
    for line in lines:
        if line.strip() and not line.startswith("#") and not line.startswith("series,"):
            if series is None or line.split(",")[0] == series:
                out.append("# SUPERSEDED " + line)
                superseded = True
            else:
                out.append(line)
        else:
            out.append(line)
    if not superseded:
        raise ValueError(
            f"no ACTIVE row to supersede in {path}"
            + (f" for series {series!r}" if series is not None else "")
        )
    out.append(",".join(new_row[c] for c in FREEZE_COLUMNS))
    path.write_text("\n".join(out) + "\n")
