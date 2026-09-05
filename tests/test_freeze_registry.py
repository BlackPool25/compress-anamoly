"""Freeze-registry tests: 12 ACTIVE schema-exact rows + plural reader.

TDD anchor for Todo 1 (compress-phase2-ladder): written BEFORE the
12-row migration, so every test here fails until `read_freeze_rows`
exists and `dataset-freeze.csv` holds the full registry.

Pin-time resolution note (evidence:
`.omo/evidence/compress-phase2-ladder/task-1-downloads.log`): the
requested short IDs were resolved name-first against the official UCR
archive listing because the ML-KULeuven/dtaianomaly mirror vendors only
001/002 and four requested names (PackagingMachine, TiltECG, wafer,
Lab2Cmac011211Netflix) plus MSL-T-1 exist nowhere upstream. Supersessions
with logged evidence: 025->019 GP711, 067->043 Mesoplodon, 145->078
resperation1, 242->089 tiltAPB1 (name-exact, number corrected);
Lab2Cmac011211Netflix->044 PowerDemand1, PackagingMachine->048
TkeepFifthMARS, TiltECG->012 ECG2 (011 ECG1 split-rejected), wafer->045
PowerDemand2, MSL-T-1->MSL-T-4 (T-1 is SMAP per labels CSV).

All tests are offline-safe: no test touches the network. Split checks
run against `data/cache/<series>` only when member caches are present
(else skipped); registry shape/metadata assertions always run.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from harness.datasets import ucr_loader as L

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "dataset-freeze.csv"
SMOOTH_PATH = ROOT / "smooth_subset.json"

# The 10 real pinned UCR series (full member names, extensionless).
# Row 1 keeps the Phase-1 mirror pin (LF bytes); rows 2-10 are official
# archive zip members (CRLF bytes) sharing one zip URL/hash cell triple.
EXPECTED_UCR_SERIES = (
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

EXPECTED_NASA_SERIES = ("SMAP-P-1", "MSL-T-4")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_plural_reader_returns_twelve_active_rows():
    """Plural reader exists and returns exactly 12 ACTIVE rows."""
    rows = L.read_freeze_rows()
    assert len(rows) == 12


def test_registry_schema_exact_per_row():
    """Every row carries EXACTLY the six schema columns with sane cells."""
    rows = L.read_freeze_rows()
    for row in rows:
        assert tuple(row.keys()) == L.FREEZE_COLUMNS
        assert row["series"].strip()
        assert row["url"].startswith("https://")
        assert _HEX64.match(row["sha256"]), row["series"]
        assert int(row["bytes"]) > 0, row["series"]
        assert row["license"].strip()
        assert row["split_rule"].strip()
        assert "," not in "".join(row.values()), row["series"]


def test_ten_ucr_series_pinned():
    """Registry holds exactly the 10 real pinned UCR DISTORTED series."""
    rows = L.read_freeze_rows()
    ucr = sorted(r["series"] for r in rows if r["series"].startswith("0"))
    assert ucr == sorted(EXPECTED_UCR_SERIES)
    for name in ucr:
        L._anomaly_span(name)  # full name keeps a parseable span suffix


def test_nasa_channels_pinned():
    """SMAP-P-1 pinned; MSL row is the verified T-4 neighbor."""
    rows = L.read_freeze_rows()
    names = [r["series"] for r in rows]
    assert "SMAP-P-1" in names
    assert "MSL-T-4" in names
    assert not any(n.startswith("MSL-T-1") for n in names)


def test_smap_split_rule_adapted_multi_event():
    """SMAP-P-1 row documents the multi-event split adaptation."""
    rows = L.read_freeze_rows()
    smap = next(r for r in rows if r["series"] == "SMAP-P-1")
    rule = smap["split_rule"]
    assert "train" in rule and "test" in rule
    assert "2149" in rule  # first-event evidence that killed the naive cut


def test_ucr_split_check_green_where_cached():
    """Cached UCR members parse with train slice anomaly-free (pre-cut)."""
    rows = L.read_freeze_rows()
    cached = [
        r for r in rows if r["series"] in EXPECTED_UCR_SERIES
        if L.cache_path(r["series"]).is_file()
    ]
    if not cached:
        pytest.skip("no UCR member caches; pin-time workspace only")
    assert len(cached) == len(EXPECTED_UCR_SERIES)
    for row in cached:
        raw = L.cache_path(row["series"]).read_bytes()
        sample = L.parse_series(raw, series=row["series"])
        assert int(sample.y[: sample.meta["train_end"]].sum()) == 0
        assert int(sample.y.sum()) > 0


def test_zip_hash_verifies_where_cached():
    """Cached source zip re-hashes to the shared UCR row sha256."""
    zips = {r["url"]: r["sha256"] for r in L.read_freeze_rows()
            if r["series"] in EXPECTED_UCR_SERIES[1:]}
    assert len(zips) == 1
    (url, sha), = zips.items()
    zpath = L.CACHE_DIR / url.rsplit("/", 1)[-1]
    if not zpath.is_file():
        pytest.skip("source zip not cached; pin-time workspace only")
    assert hashlib.sha256(zpath.read_bytes()).hexdigest() == sha


def test_single_row_reader_backward_compat():
    """Old single-row reader keeps working: returns the first ACTIVE row."""
    first = L.read_freeze_rows()[0]
    assert L.read_freeze_row() == first
    assert first["series"] == EXPECTED_UCR_SERIES[0]


def test_hash_mismatch_raises_loudly():
    """Wrong hash raises ValueError naming the mismatch (no fallback)."""
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        L.verify_sha256(b"tampered-bytes", "0" * 64)


def test_smooth_subset_valid():
    """smooth_subset.json lists Drift + 3 registry members."""
    members = json.loads(SMOOTH_PATH.read_text())
    assert members[0] == "Drift" and len(members) == 4
    names = {r["series"] for r in L.read_freeze_rows()}
    assert set(members[1:]) <= names
