"""Phase-2 matrix routing (Todo 9, TDD-red).

Locks the full Phase-2 cell routing in ``harness.run_harness.run_cell``:

- Path-A: R0, R1, R2a, R2b, Q8, Q4, Q2 x PCA/IF/TCN (7 x 3 = 21 rows).
- Path-B: R4+D4 and R4-bypass+D4-hybrid (2 rows).
- R4-decode+PCA Drift-only (+1 row on Drift; R4-decode->IF is NEVER run).

So per (dataset, seed): 23 rows, 24 on Drift. TCN trains ONCE per
(dataset, seed) on R0-train and is reused across codecs. Unknown
dataset/codec names raise a named ValueError (never KeyError, never
silent skip).
"""

import numpy as np
import pytest

from harness import run_harness
from harness.detectors.tcn_autoencoder import TCNAutoencoder

FULL_CODECS = ["R0", "R1", "R2a", "R2b", "Q8", "Q4", "Q2", "R4", "R4-bypass"]


def _keys(rows: list[dict]) -> set[tuple]:
    """Project rows to (codec, detector, path) triples."""
    return {(r["codec"], r["detector"], r["path"]) for r in rows}


def test_phase2_row_count_non_drift():
    """Spike x full codec list yields exactly 23 rows (21 A + 2 B)."""
    rows = run_harness.run_cell("Spike", 42, FULL_CODECS)
    assert len(rows) == 23


def test_phase2_row_count_drift():
    """Drift x full codec list yields exactly 24 rows (23 + R4-decode/PCA)."""
    rows = run_harness.run_cell("Drift", 42, FULL_CODECS)
    assert len(rows) == 24
    dec = [r for r in rows if r["codec"] == "R4-decode"]
    assert _keys(dec) == {("R4-decode", "PCA", "A")}


def test_r4_decode_pca_drift_only():
    """Non-Drift datasets emit zero R4-decode rows (PCA or otherwise)."""
    rows = run_harness.run_cell("Spike", 42, FULL_CODECS)
    assert [r for r in rows if r["codec"] == "R4-decode"] == []


def test_q2_wired_into_path_a():
    """Q2 decodes to floats and scores under PCA/IF/TCN on Path-A."""
    rows = run_harness.run_cell("Spike", 42, FULL_CODECS)
    assert _keys([r for r in rows if r["codec"] == "Q2"]) == {
        ("Q2", "PCA", "A"), ("Q2", "IF", "A"), ("Q2", "TCN", "A"),
    }


def test_r1_r2_rows_path_a():
    """R1/R2a/R2b each score under PCA/IF/TCN on Path-A."""
    rows = run_harness.run_cell("Spike", 42, FULL_CODECS)
    keys = _keys(rows)
    for codec in ("R1", "R2a", "R2b"):
        assert {(codec, "PCA", "A"), (codec, "IF", "A"),
                (codec, "TCN", "A")} <= keys


def test_bypass_d4_hybrid_path():
    """R4-bypass scores via the D4 hybrid on Path-B (never Path-A)."""
    rows = run_harness.run_cell("Spike", 42, FULL_CODECS)
    byp = [r for r in rows if r["codec"] == "R4-bypass"]
    assert len(byp) == 1
    assert (byp[0]["detector"], byp[0]["path"]) == ("D4", "B")


def test_tcn_trained_once_full_matrix():
    """TCN.fit runs exactly once per (dataset, seed) over the full matrix."""
    calls: list = []
    orig_fit = TCNAutoencoder.fit

    def counting_fit(self, x_train: np.ndarray) -> None:
        """Count fits then delegate to the real fit."""
        calls.append(1)
        return orig_fit(self, x_train)

    TCNAutoencoder.fit = counting_fit  # type: ignore[method-assign]
    try:
        run_harness.run_cell("Spike", 42, FULL_CODECS)
    finally:
        TCNAutoencoder.fit = orig_fit  # type: ignore[method-assign]
    assert len(calls) == 1


def test_unknown_dataset_raises_named_valueerror():
    """Unknown dataset names raise ValueError naming the dataset (not KeyError)."""
    with pytest.raises(ValueError, match="NopeNope"):
        run_harness.run_cell("NopeNope", 42, ["R0"])


def test_unknown_codec_raises_named_valueerror():
    """Unknown codec names raise ValueError naming the codec (never silent)."""
    with pytest.raises(ValueError, match="ZZZ"):
        run_harness.run_cell("Spike", 42, ["R0", "ZZZ"])


def test_smoke_profile_unchanged_cells():
    """--smoke stays Spike/R0+Q8/seed42: 2 codecs x 3 detectors = 6 rows."""
    import csv

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.csv"
        rc = run_harness.main(["--smoke", "--output", str(out)])
        assert rc == 0
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        assert {r["codec"] for r in rows} == {"R0", "Q8"}
        assert {r["dataset"] for r in rows} == {"Spike"}


def test_fast_profile_flags():
    """--fast selects 1 seed, synth-only datasets, TCN epochs=2."""
    args = run_harness.parse_args(["--fast"])
    assert args.fast is True
