"""Tests for the NASA (telemanom) loader. All HTTP is mocked; no real network.

TDD anchor for Todo 2 (compress-phase2-ladder): written BEFORE
`harness/datasets/nasa_loader.py` exists, so every test here fails until
the loader implements the registry-driven contract (same as ucr_loader):
per-series cache keys `data/cache/<name>`, per-row SHA-256/bytes verify,
offline-without-cache raises RuntimeError, split-check before cut, never
substitute synthetic data.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from harness.datasets import nasa_loader as L


def _npy_bytes(n: int, ncols: int = 2, val: float = -1.0) -> bytes:
    """Encode a tiny (n, ncols) float64 .npy payload with col 0 = val."""
    a = np.full((n, ncols), 0.0, dtype=np.float64)
    a[:, 0] = val
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path / "cache")
    return L.CACHE_DIR


def _patch_meta(monkeypatch: pytest.MonkeyPatch, test_raw: bytes,
                train_raw: bytes, events: list[tuple[int, int]],
                test_n: int, train_n: int) -> None:
    """Point the loader's frozen metadata at tiny synthetic payloads.

    The test-file URL/hash/size come from a fake freeze row (the loader is
    registry-driven); the train sibling + events + lengths come from the
    in-module frozen dicts, patched here to match the tiny payloads.
    """
    row = {
        "series": "SMAP-P-1",
        "url": "https://example.invalid/data/test/P-1.npy",
        "sha256": hashlib.sha256(test_raw).hexdigest(),
        "bytes": str(len(test_raw)),
        "license": "lic",
        "split_rule": "rule",
    }
    monkeypatch.setattr(L, "_row_for", lambda name: row)
    monkeypatch.setitem(L.TRAIN_SHA256, "SMAP-P-1", hashlib.sha256(train_raw).hexdigest())
    monkeypatch.setitem(L.TRAIN_BYTES, "SMAP-P-1", len(train_raw))
    monkeypatch.setitem(L.EVENTS, "SMAP-P-1", list(events))
    monkeypatch.setitem(L.TEST_LEN, "SMAP-P-1", test_n)
    monkeypatch.setitem(L.TRAIN_LEN, "SMAP-P-1", train_n)


def test_cache_hit_returns_parsed_sample(tmp_path, monkeypatch):
    """Cache-hit path parses cached test+train .npy without network."""
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    test_raw = _npy_bytes(100)
    train_raw = _npy_bytes(40)
    (cache / "SMAP-P-1").write_bytes(test_raw)
    (cache / "SMAP-P-1.train").write_bytes(train_raw)
    _patch_meta(monkeypatch, test_raw, train_raw, [(90, 95)], 100, 40)
    with patch.object(L, "download_bytes", side_effect=AssertionError("no net")):
        s = L.load_series("SMAP-P-1")
    assert s.x.dtype == np.float64
    assert s.x.shape == (140,)
    assert s.y.shape == (140,)
    assert int(s.y.sum()) == 5
    assert s.meta["series"] == "SMAP-P-1"
    assert s.meta["train_end"] == 40
    assert int(s.y[:40].sum()) == 0  # train slice anomaly-free


def test_checksum_mismatch_raises_and_deletes(tmp_path, monkeypatch):
    """Corrupting one cached byte raises ValueError naming the series."""
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    test_raw = _npy_bytes(100)
    train_raw = _npy_bytes(40)
    bad = bytearray(test_raw)
    bad[100] ^= 0x01
    target = cache / "SMAP-P-1"
    target.write_bytes(bytes(bad))
    (cache / "SMAP-P-1.train").write_bytes(train_raw)
    _patch_meta(monkeypatch, test_raw, train_raw, [(90, 95)], 100, 40)
    with pytest.raises(ValueError, match="SMAP-P-1"):
        L.load_series("SMAP-P-1")
    assert not target.exists()


def test_bytes_mismatch_raises_and_deletes(tmp_path, monkeypatch):
    """Truncated cache raises ValueError naming the series and deletes."""
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    test_raw = _npy_bytes(100)
    train_raw = _npy_bytes(40)
    target = cache / "SMAP-P-1"
    target.write_bytes(test_raw[: len(test_raw) // 2])
    (cache / "SMAP-P-1.train").write_bytes(train_raw)
    _patch_meta(monkeypatch, test_raw, train_raw, [(90, 95)], 100, 40)
    with pytest.raises(ValueError, match="SMAP-P-1"):
        L.load_series("SMAP-P-1")
    assert not target.exists()


def test_offline_no_cache_raises_loudly(tmp_path, monkeypatch):
    """Offline without cache raises RuntimeError naming the series."""
    _isolate(tmp_path, monkeypatch)
    with patch.object(L, "download_bytes", side_effect=ConnectionError("down")):
        with pytest.raises(RuntimeError, match="SMAP-P-1"):
            L.load_series("SMAP-P-1")


def test_unknown_series_raises_naming_it(tmp_path, monkeypatch):
    """Unknown names raise ValueError naming the series, never synthetic."""
    _isolate(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="NOPE-NOT-A-SERIES"):
        L.load_series("NOPE-NOT-A-SERIES")


def test_download_then_caches_and_parses(tmp_path, monkeypatch):
    """Cache miss + mocked download verifies, writes both caches, parses."""
    cache = _isolate(tmp_path, monkeypatch)
    test_raw = _npy_bytes(100)
    train_raw = _npy_bytes(40)
    _patch_meta(monkeypatch, test_raw, train_raw, [(90, 95)], 100, 40)
    bodies = {"test": test_raw, "train": train_raw}

    def fake_download(url: str) -> bytes:
        return bodies["train"] if "/train/" in url else bodies["test"]

    with patch.object(L, "download_bytes", side_effect=fake_download) as dl:
        s = L.load_series("SMAP-P-1")
    assert dl.call_count == 2
    assert (cache / "SMAP-P-1").read_bytes() == test_raw
    assert (cache / "SMAP-P-1.train").read_bytes() == train_raw
    assert int(s.y.sum()) == 5


def test_train_url_is_test_url_sibling():
    """Train URL derives from the test URL by swapping /test/ -> /train/."""
    url = "https://huggingface.co/datasets/appleparan/telemanom/resolve/abc/data/data/test/P-1.npy"
    assert L.train_url(url) == url.replace("/data/test/", "/data/train/")


def test_event_outside_test_bounds_rejected(tmp_path, monkeypatch):
    """An event outside the test array raises before any cut is trusted."""
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    test_raw = _npy_bytes(100)
    train_raw = _npy_bytes(40)
    (cache / "SMAP-P-1").write_bytes(test_raw)
    (cache / "SMAP-P-1.train").write_bytes(train_raw)
    _patch_meta(monkeypatch, test_raw, train_raw, [(95, 105)], 100, 40)
    with pytest.raises(ValueError, match="SMAP-P-1"):
        L.load_series("SMAP-P-1")


def test_nan_in_signal_rejected(tmp_path, monkeypatch):
    """NaN in the signal column raises ValueError (NaN policy: reject)."""
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    a = np.full((100, 2), 0.0, dtype=np.float64)
    a[:, 0] = -1.0
    a[50, 0] = np.nan
    buf = io.BytesIO()
    np.save(buf, a)
    test_raw = buf.getvalue()
    train_raw = _npy_bytes(40)
    (cache / "SMAP-P-1").write_bytes(test_raw)
    (cache / "SMAP-P-1.train").write_bytes(train_raw)
    _patch_meta(monkeypatch, test_raw, train_raw, [(90, 95)], 100, 40)
    with pytest.raises(ValueError, match="SMAP-P-1"):
        L.load_series("SMAP-P-1")


def test_real_caches_parse_where_present():
    """Pinned NASA caches (if populated) parse with exact n/train/y sums."""
    from harness.datasets import ucr_loader as U

    rows = {r["series"]: r for r in U.read_freeze_rows()}
    for name in ("SMAP-P-1", "MSL-T-4"):
        if not L.cache_path(name).is_file():
            pytest.skip(f"real NASA cache missing for {name}")
    s = L.load_series("SMAP-P-1")
    assert s.x.shape == (2872 + 8505,) and s.x.dtype == np.float64
    assert s.meta["train_end"] == 2872
    assert int(s.y.sum()) == (2349 - 2149) + (4844 - 4536) + (3779 - 3539)
    assert int(s.y[:2872].sum()) == 0
    assert rows["SMAP-P-1"]["bytes"] == str(len(L.cache_path("SMAP-P-1").read_bytes()))
    t = L.load_series("MSL-T-4")
    assert t.x.shape == (2272 + 2217,) and t.x.dtype == np.float64
    assert t.meta["train_end"] == 2272
    assert int(t.y.sum()) == 1240 - 1172
    assert int(t.y[:2272].sum()) == 0
