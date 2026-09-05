"""Pinned UCR time-series-anomaly series loader with cache and freeze record.

Phase A pin (verified 2026-09-05 by re-download + re-hash + byte-compare)::

    series = 001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620
    url    = https://raw.githubusercontent.com/ML-KULeuven/dtaianomaly/...
    sha256 = 4a9c39e6...9973bca3 (see SERIES_SHA256)
    bytes  = 1356515

Phase 2 registry (Todo 2): ``load_series(name)`` serves all 10 UCR rows of
``dataset-freeze.csv``. Row 1 is the mirror file above (LF bytes, fetched
directly). Rows 2-10 share the official UCR archive zip URL + zip
SHA-256/bytes; ``series`` names the zip member (``<series>.txt``, CRLF
verbatim). Per-series cache keys stay extensionless (``data/cache/<name>``);
the source zip is cached once at ``data/cache/<zip basename>``. Member
bytes/hashes below are frozen from the Todo 1 evidence log
(``task-1-downloads.log`` section 3); the six-column freeze schema has no
member column, so this table is the member-truth copy.
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.error
import urllib.request
import zipfile
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

# Frozen zip-member truth (Todo 1 evidence log, section 3): CRLF member
# bytes + SHA-256 for the 9 official-archive members. Row 1 (mirror, LF)
# verifies against its freeze-row cells instead, so it has no entry here.
MEMBER_SHA256: dict[str, str] = {
    "019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212":
        "d5ef58edd75915cd1cfb88368905186281e25eb7587e2c5cd28a44f4a3e2fb8f",
    "032_UCR_Anomaly_DISTORTEDInternalBleeding4_1000_4675_5033":
        "90bb58ed2d8c100486d6ab744eb385960583f882a0c185cdd86ba81a588a278e",
    "044_UCR_Anomaly_DISTORTEDPowerDemand1_9000_18485_18821":
        "a7ca3d07ad8d56e274197c5f617da49fb0176c438add92d58a1a97aaf0d8280d",
    "043_UCR_Anomaly_DISTORTEDMesoplodonDensirostris_10000_19280_19440":
        "241271e6b255d97184928478470632deaa4525a0bd005f4f4f2b87ed4860b103",
    "048_UCR_Anomaly_DISTORTEDTkeepFifthMARS_3500_5988_6085":
        "a93a5fa9a5c95f1a3e975f1bbc7cf8f6adac879b6d95d7605337b8e38798ae7e",
    "012_UCR_Anomaly_DISTORTEDECG2_15000_16000_16100":
        "918236c9ecd8e65f0e61df7209c5874b811657177de2ea982ef98c6f3ac6172f",
    "078_UCR_Anomaly_DISTORTEDresperation1_100000_110260_110412":
        "b2c72cf631268a469e90106a63efb911b499aceb7ec7149d48b692bb7f1e7a41",
    "045_UCR_Anomaly_DISTORTEDPowerDemand2_14000_23357_23717":
        "39e72cdcd0c18022f06012b49020bdc01bb0424a12a78f6fb1eeee7c4ff65e9e",
    "089_UCR_Anomaly_DISTORTEDtiltAPB1_100000_114283_114350":
        "4f4984e84ed00e12c9a6c4ad3561ce0eebaa864266fd9baab666745e1cdae71c",
}
MEMBER_BYTES: dict[str, int] = {
    "019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212": 216000,
    "032_UCR_Anomaly_DISTORTEDInternalBleeding4_1000_4675_5033": 131778,
    "044_UCR_Anomaly_DISTORTEDPowerDemand1_9000_18485_18821": 538758,
    "043_UCR_Anomaly_DISTORTEDMesoplodonDensirostris_10000_19280_19440": 444006,
    "048_UCR_Anomaly_DISTORTEDTkeepFifthMARS_3500_5988_6085": 204012,
    "012_UCR_Anomaly_DISTORTEDECG2_15000_16000_16100": 540000,
    "078_UCR_Anomaly_DISTORTEDresperation1_100000_110260_110412": 3600000,
    "045_UCR_Anomaly_DISTORTEDPowerDemand2_14000_23357_23717": 538758,
    "089_UCR_Anomaly_DISTORTEDtiltAPB1_100000_114283_114350": 2340018,
}

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


def download_file(url: str, dest: Path) -> Path:
    """Download URL to dest with Range-resume + retries; return dest.

    A partial dest is resumed via ``Range: bytes=<have>-``; if the server
    answers 200 instead of 206 the transfer restarts from zero. Shared by
    the UCR zip fetch and the NASA .npy fetches.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for _ in range(MAX_RETRIES + 1):
        have = dest.stat().st_size if dest.is_file() else 0
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={have}-"} if have else {}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                resume = have and getattr(resp, "status", 200) == 206
                mode = "ab" if resume else "wb"
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
    raise ConnectionError(
        f"download failed after {MAX_RETRIES} retries: {url}: {last}"
    )


def _row_for(name: str) -> dict[str, str]:
    """Return the ACTIVE freeze row for a series; ValueError names unknowns."""
    for row in read_freeze_rows():
        if row["series"] == name:
            return row
    raise ValueError(
        f"unknown series {name!r}: no ACTIVE row in {FREEZE_PATH}; "
        "refusing to substitute synthetic data"
    )


def _is_zip_row(row: dict[str, str]) -> bool:
    """True when the row pins the official UCR archive zip (member load)."""
    return row["url"].endswith(".zip")


def _expected(name: str, row: dict[str, str]) -> tuple[str, int]:
    """Return the (sha256, bytes) the per-series cache file must match.

    Mirror rows verify against their freeze-row cells; zip-member rows
    verify against the frozen MEMBER_* table (the row cells pin the zip).
    """
    if _is_zip_row(row):
        try:
            return MEMBER_SHA256[name], MEMBER_BYTES[name]
        except KeyError:
            raise ValueError(
                f"no frozen member hash for series {name!r}; "
                "reselect via supersede, never substitute"
            ) from None
    return row["sha256"], int(row["bytes"])


def _verify_size(raw: bytes, expected: int, name: str) -> None:
    """Raise ValueError naming the series when the byte count mismatches."""
    if len(raw) != expected:
        raise ValueError(
            f"size mismatch for series {name!r}: "
            f"expected {expected} bytes, got {len(raw)}"
        )


def _checked(raw: bytes, sha256: str, nbytes: int, name: str) -> bytes:
    """Size-then-hash verify; ValueError names the series on any mismatch."""
    _verify_size(raw, nbytes, name)
    try:
        verify_sha256(raw, sha256)
    except ValueError as exc:
        raise ValueError(f"series {name!r}: {exc}") from None
    return raw


def _extract_member(zip_path: Path, name: str) -> bytes:
    """Return the raw bytes of the archive member ending in ``<name>.txt``.

    Members live under ``.../UCR_Anomaly_FullData/<name>.txt`` inside the
    official zip, so the match is by path suffix; zero or 2+ hits raise
    ValueError naming the series (never a silent wrong-member read).
    """
    with zipfile.ZipFile(zip_path) as zf:
        hits = [n for n in zf.namelist() if n.endswith(name + ".txt")]
        if len(hits) != 1:
            raise ValueError(
                f"expected exactly 1 member ending in {(name + '.txt')!r} "
                f"in archive {zip_path}, found {len(hits)}"
            )
        return zf.read(hits[0])


def _ensure_zip(row: dict[str, str], name: str) -> Path:
    """Return the verified source-zip path, downloading with resume if needed.

    A cached zip that fails the row SHA-256/bytes check is deleted and
    fetched once more; a second failure raises ValueError naming the row.
    Offline without cache raises RuntimeError naming the series.
    """
    dest = CACHE_DIR / row["url"].rsplit("/", 1)[-1]
    sha, nbytes = row["sha256"], int(row["bytes"])
    if dest.is_file():
        try:
            _checked(dest.read_bytes(), sha, nbytes, row["url"])
            return dest
        except ValueError:
            dest.unlink()
    try:
        download_file(row["url"], dest)
    except (ConnectionError, OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"offline and no cache: cannot fetch series {name!r} "
            f"({row['url']}, cache miss at {dest}); "
            f"refusing to substitute synthetic data: {exc}"
        ) from exc
    raw = dest.read_bytes()
    try:
        _checked(raw, sha, nbytes, row["url"])
    except ValueError:
        dest.unlink()
        raise
    return dest


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
    if not np.isfinite(x).all():
        raise ValueError(
            f"non-finite values in series {series!r}; refusing to parse"
        )
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


def load_series(name: str = SERIES_NAME) -> TimeSeriesSample:
    """Load a registry series, cache-first at ``data/cache/<name>``.

    Mirror rows download the series file directly; zip rows fetch the
    source zip once (resumed) and extract the ``<name>.txt`` member. A
    cached file is size-then-hash verified (mismatch deletes the file and
    raises ValueError naming the series). Offline without cache raises
    RuntimeError loudly; synthetic data is never substituted. Parsed
    ``x`` is float64; the runner casts to float32 immediately after load.
    """
    row = _row_for(name)
    sha256, nbytes = _expected(name, row)
    path = cache_path(name)
    if path.is_file():
        try:
            raw = _checked(path.read_bytes(), sha256, nbytes, name)
        except ValueError:
            path.unlink()
            raise
        return parse_series(raw, series=name)
    if _is_zip_row(row):
        raw = _checked(_extract_member(_ensure_zip(row, name), name), sha256, nbytes, name)
    else:
        try:
            raw = _checked(download_bytes(row["url"]), sha256, nbytes, name)
        except (ConnectionError, OSError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"offline and no cache: cannot fetch {row['url']} "
                f"(cache miss at {path}); refusing to substitute synthetic data: {exc}"
            ) from exc
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
