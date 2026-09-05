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


# --- Todo 2: registry-driven multi-series coverage (TDD red-first) ---

import shutil
import zipfile

ALL_TEN = (
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
)


def test_registry_all_ten_load_offline_from_cache(tmp_path, monkeypatch):
    """All 10 registry series load cache-first with no network touch."""
    real = {name: (L.CACHE_DIR / name).read_bytes() for name in ALL_TEN}
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    for name, raw in real.items():
        (cache / name).write_bytes(raw)
    with patch.object(L, "download_bytes", side_effect=AssertionError("no net")):
        with patch.object(L, "download_file", side_effect=AssertionError("no net")):
            for name in ALL_TEN:
                s = L.load_series(name)
                assert s.x.dtype == np.float64 and s.y.shape == s.x.shape
                assert s.meta["series"] == name
                assert int(s.y.sum()) > 0
                assert int(s.y[: s.meta["train_end"]].sum()) == 0


def test_unknown_series_raises_naming_it(tmp_path, monkeypatch):
    """Unknown names raise ValueError naming the series, never synthetic."""
    _isolate(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="NOPE-NOT-A-SERIES"):
        L.load_series("NOPE-NOT-A-SERIES")


def test_bytes_mismatch_raises_and_deletes(tmp_path, monkeypatch):
    """Truncated cache raises ValueError naming the series and deletes."""
    cache = _isolate(tmp_path, monkeypatch)
    cache.mkdir(parents=True)
    target = cache / L.SERIES_NAME
    target.write_bytes(REAL_RAW[: len(REAL_RAW) // 2])
    with pytest.raises(ValueError, match=L.SERIES_NAME):
        L.load_series()
    assert not target.exists()


def test_extract_member_reads_zip_entry(tmp_path):
    """Zip-member helper returns the exact bytes of `<name>.txt`."""
    zp = tmp_path / "z.zip"
    body = b"1.0\r\n2.0\r\n"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("M_10_6_8.txt", body)
    assert L._extract_member(zp, "M_10_6_8") == body


def test_zip_row_cache_miss_extracts_and_caches(tmp_path, monkeypatch):
    """Zip-row miss downloads the zip once, extracts, verifies, caches."""
    cache = _isolate(tmp_path, monkeypatch)
    body = ("\n".join(["1.0"] * 100) + "\n").encode()
    member = "X_100_50_60"
    zp = tmp_path / "src.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr(member + ".txt", body)
    zip_raw = zp.read_bytes()
    row = {
        "series": member,
        "url": "https://example.invalid/z.zip",
        "sha256": hashlib.sha256(zip_raw).hexdigest(),
        "bytes": str(len(zip_raw)),
        "license": "lic",
        "split_rule": L.SPLIT_RULE,
    }
    monkeypatch.setattr(L, "_row_for", lambda name: row)
    monkeypatch.setitem(L.MEMBER_SHA256, member, hashlib.sha256(body).hexdigest())
    monkeypatch.setitem(L.MEMBER_BYTES, member, len(body))

    def fake_download(url: str, dest: Path) -> Path:
        assert url == row["url"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zip_raw)
        return dest

    with patch.object(L, "download_file", side_effect=fake_download) as dl:
        s = L.load_series(member)
    dl.assert_called_once()
    assert (cache / member).read_bytes() == body
    assert int(s.y.sum()) == 10 and int(s.y[:40].sum()) == 0
    # Second load is a pure cache hit (no download at all).
    with patch.object(L, "download_file", side_effect=AssertionError("no net")):
        assert int(L.load_series(member).y.sum()) == 10
