"""Tests for the pinned UCR loader. All HTTP is mocked; no real network."""

from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from harness.datasets import ucr_loader as L

REAL_CACHE = L.cache_path()
REAL_RAW = REAL_CACHE.read_bytes()


def _tiny(n: int = 100, val: float = 1.0) -> bytes:
    return ("\n".join([str(val)] * n) + "\n").encode()


def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path / "cache")
    return L.CACHE_DIR


def test_cache_hit_returns_parsed_sample(tmp_path, monkeypatch):
    """Cache-hit path parses pinned bytes without touching the network."""
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    (cache / L.SERIES_NAME).write_bytes(REAL_RAW)
    with patch.object(L, "download_bytes", side_effect=AssertionError("no net")):
        s = L.load_series()
    assert s.x.shape == (79795,) and s.x.dtype == np.float64
    assert s.y.shape == (79795,) and set(np.unique(s.y)) <= {0, 1}
    assert int(s.y.sum()) == 620
    assert s.meta["series"] == L.SERIES_NAME and s.meta["seed"] == 0
    assert s.meta["split"] == L.SPLIT_RULE
    assert int(s.y[: s.meta["train_end"]].sum()) == 0


def test_checksum_mismatch_raises_and_deletes(tmp_path, monkeypatch):
    """Corrupting one cached byte raises and removes the bad file."""
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    bad = bytearray(REAL_RAW)
    bad[1000] ^= 0x01
    target = cache / L.SERIES_NAME
    target.write_bytes(bytes(bad))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        L.load_series()
    assert not target.exists()


def test_offline_no_cache_raises_loudly(tmp_path, monkeypatch):
    """Offline without cache raises a clear error, never synthetic data."""
    _isolate(tmp_path, monkeypatch)
    with patch.object(L, "download_bytes", side_effect=ConnectionError("down")):
        with pytest.raises(RuntimeError, match="offline and no cache"):
            L.load_series()


def test_download_then_caches_and_parses(tmp_path, monkeypatch):
    """Cache miss + mocked download verifies, writes cache, parses."""
    cache = _isolate(tmp_path, monkeypatch)
    with patch.object(L, "download_bytes", return_value=REAL_RAW) as dl:
        s = L.load_series()
    dl.assert_called_once()
    assert (cache / L.SERIES_NAME).read_bytes() == REAL_RAW
    assert int(s.y.sum()) == 620


def test_download_bytes_retries_then_raises(monkeypatch):
    """urlopen failures retry; timeout=30s; gives up after 3 retries."""
    calls = []

    def fail(url, timeout=None):
        calls.append(timeout)
        raise urllib.error.URLError("net down")

    monkeypatch.setattr(L.urllib.request, "urlopen", fail)
    with pytest.raises(ConnectionError, match="after 3 retries"):
        L.download_bytes("https://example.invalid/x.txt")
    assert len(calls) == L.MAX_RETRIES + 1
    assert all(t == L.TIMEOUT_S == 30 for t in calls)


def test_download_bytes_succeeds_after_retry(monkeypatch):
    """Two failures then success returns the body."""
    state = {"n": 0}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"1.0\n2.0\n"

    def flaky(url, timeout=None):
        state["n"] += 1
        if state["n"] < 3:
            raise urllib.error.URLError("flaky")
        return Resp()

    monkeypatch.setattr(L.urllib.request, "urlopen", flaky)
    assert L.download_bytes("https://example.invalid/x.txt") == b"1.0\n2.0\n"


def test_parse_rejects_train_anomaly():
    """Anomaly span inside the first 40% raises, naming the violation."""
    with pytest.raises(ValueError, match="train-slice violation"):
        L.parse_series(_tiny(), series="X_100_5_10")


def test_parse_rejects_straddling_span_first():
    """Span straddling the cut pollutes train: train check fires first."""
    with pytest.raises(ValueError, match="train-slice violation"):
        L.parse_series(_tiny(), series="X_100_30_50")


def test_parse_accepts_valid_span():
    """Span fully inside the last 60% parses to a binary-labeled sample."""
    s = L.parse_series(_tiny(), series="X_100_50_60")
    assert int(s.y.sum()) == 10 and int(s.y[:40].sum()) == 0


def test_freeze_row_schema_validated():
    """Freeze file keeps EXACTLY the six schema columns with a live row."""
    row = L.read_freeze_row()
    assert tuple(row.keys()) == L.FREEZE_COLUMNS
    assert len(hashlib.sha256(REAL_RAW).hexdigest()) == 64
    assert row["sha256"] == hashlib.sha256(REAL_RAW).hexdigest()
    assert row["bytes"] == str(len(REAL_RAW)) == str(L.SERIES_BYTES)
    assert row["split_rule"] == L.SPLIT_RULE
    assert row["url"].startswith("https://")


def test_supersede_keeps_one_active_row(tmp_path):
    """Reselection comments the old row SUPERSEDED and appends the new one."""
    fp = tmp_path / "dataset-freeze.csv"
    fp.write_text(
        "series,url,sha256,bytes,license,split_rule\n"
        "OLD,https://example.invalid/o,abc,3,lic,rule\n"
    )
    L.supersede_freeze_row(
        {
            "series": "NEW",
            "url": "https://example.invalid/n",
            "sha256": "def",
            "bytes": "4",
            "license": "lic",
            "split_rule": "rule",
        },
        path=fp,
    )
    text = fp.read_text()
    assert "# SUPERSEDED OLD,https://example.invalid/o,abc,3,lic,rule" in text
    assert L.read_freeze_row(path=fp)["series"] == "NEW"
