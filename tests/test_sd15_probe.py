"""Unit tests for comfy/sd15_probe.py.

All tensors are on CPU so no MPS device is required.
Tests manipulate the SD15_PROBE env var via monkeypatch and call
clear_snapshots() between cases to avoid cross-test state leakage.
"""

from __future__ import annotations

import importlib
import math
import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_probe(monkeypatch: pytest.MonkeyPatch, active: bool) -> object:
    """Re-import sd15_probe with SD15_PROBE set/unset.

    The module caches _ACTIVE at import time, so tests that need a different
    activation state must reload it after patching the env var.
    """
    if active:
        monkeypatch.setenv("SD15_PROBE", "1")
    else:
        monkeypatch.delenv("SD15_PROBE", raising=False)

    # Force fresh import
    if "comfy.sd15_probe" in sys.modules:
        del sys.modules["comfy.sd15_probe"]

    import comfy.sd15_probe as probe  # noqa: PLC0415
    return probe


# ---------------------------------------------------------------------------
# 1. maybe_snapshot does nothing when SD15_PROBE is NOT set
# ---------------------------------------------------------------------------

def test_maybe_snapshot_inactive_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=False)
    probe.clear_snapshots()

    t = torch.randn(4, 4)
    probe.maybe_snapshot("latent", t)

    assert probe.get_snapshots() == {}


# ---------------------------------------------------------------------------
# 2. maybe_snapshot captures correct stats when SD15_PROBE=1
# ---------------------------------------------------------------------------

def test_maybe_snapshot_captures_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)
    probe.clear_snapshots()

    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    probe.maybe_snapshot("test_tag", t)

    snaps = probe.get_snapshots()
    assert "test_tag" in snaps

    s = snaps["test_tag"]
    assert s["shape"] == (4,)
    assert s["dtype"] == "torch.float32"
    assert s["nan_count"] == 0
    assert s["inf_count"] == 0

    assert isinstance(s["mean"], float)
    assert abs(s["mean"] - 2.5) < 1e-5  # type: ignore[arg-type]

    assert isinstance(s["std"], float)
    assert s["std"] > 0  # type: ignore[operator]

    assert isinstance(s["max_abs"], float)
    assert abs(s["max_abs"] - 4.0) < 1e-5  # type: ignore[arg-type]


def test_maybe_snapshot_detects_nan_and_inf(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)
    probe.clear_snapshots()

    t = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])
    probe.maybe_snapshot("noisy", t)

    s = probe.get_snapshots()["noisy"]
    assert s["nan_count"] == 1
    assert s["inf_count"] == 2


def test_maybe_snapshot_integer_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integer tensors must not crash; floating stats are NaN."""
    probe = _reload_probe(monkeypatch, active=True)
    probe.clear_snapshots()

    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    probe.maybe_snapshot("int_tag", t)

    s = probe.get_snapshots()["int_tag"]
    assert s["shape"] == (3,)
    assert s["nan_count"] == 0
    assert s["inf_count"] == 0
    assert isinstance(s["mean"], float) and math.isnan(s["mean"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. clear_snapshots clears stored data
# ---------------------------------------------------------------------------

def test_clear_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)
    probe.clear_snapshots()

    probe.maybe_snapshot("a", torch.ones(3))
    assert "a" in probe.get_snapshots()

    probe.clear_snapshots()
    assert probe.get_snapshots() == {}


# ---------------------------------------------------------------------------
# 4. get_snapshots returns captured tags
# ---------------------------------------------------------------------------

def test_get_snapshots_returns_all_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)
    probe.clear_snapshots()

    probe.maybe_snapshot("x", torch.zeros(2))
    probe.maybe_snapshot("y", torch.ones(2))

    snaps = probe.get_snapshots()
    assert set(snaps.keys()) == {"x", "y"}


def test_get_snapshots_is_shallow_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating the returned dict must not affect internal state."""
    probe = _reload_probe(monkeypatch, active=True)
    probe.clear_snapshots()

    probe.maybe_snapshot("z", torch.zeros(1))
    copy = probe.get_snapshots()
    copy["injected"] = {}  # type: ignore[assignment]

    assert "injected" not in probe.get_snapshots()


# ---------------------------------------------------------------------------
# 5. compare_snapshots passes for identical stats
# ---------------------------------------------------------------------------

def test_compare_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)

    t = torch.randn(8, 8)
    probe.clear_snapshots()
    probe.maybe_snapshot("latent", t)
    run = probe.get_snapshots()

    results = probe.compare_snapshots(run, run, threshold=1e-3)
    assert len(results) == 1
    assert results[0]["diverged"] is False
    assert results[0]["tag"] == "latent"


# ---------------------------------------------------------------------------
# 6. compare_snapshots detects numeric divergence
# ---------------------------------------------------------------------------

def test_compare_detects_numeric_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)
    probe.clear_snapshots()

    # Build two snapshot dicts directly (no need to call maybe_snapshot)
    cpu_run: dict[str, dict[str, object]] = {
        "latent": {
            "shape": (1, 4, 64, 64),
            "dtype": "torch.float32",
            "device": "cpu",
            "mean": 0.0,
            "std": 1.0,
            "max_abs": 3.0,
            "min": -3.0,
            "max": 3.0,
            "nan_count": 0,
            "inf_count": 0,
        }
    }
    mps_run: dict[str, dict[str, object]] = {
        "latent": {
            "shape": (1, 4, 64, 64),
            "dtype": "torch.float32",
            "device": "mps:0",
            "mean": 0.5,   # differs by 0.5 > 1e-3
            "std": 1.0,
            "max_abs": 3.0,
            "min": -3.0,
            "max": 3.0,
            "nan_count": 0,
            "inf_count": 0,
        }
    }

    results = probe.compare_snapshots(cpu_run, mps_run, threshold=1e-3)
    assert len(results) == 1
    r = results[0]
    assert r["diverged"] is True
    assert isinstance(r["delta_mean"], float)
    assert abs(r["delta_mean"] - 0.5) < 1e-6  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. compare_snapshots reports missing tags
# ---------------------------------------------------------------------------

def test_compare_missing_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)

    cpu_run: dict[str, dict[str, object]] = {
        "latent": {
            "shape": (1,), "dtype": "torch.float32", "device": "cpu",
            "mean": 0.0, "std": 1.0, "max_abs": 1.0,
            "min": -1.0, "max": 1.0, "nan_count": 0, "inf_count": 0,
        },
        "extra_cpu_only": {
            "shape": (1,), "dtype": "torch.float32", "device": "cpu",
            "mean": 0.0, "std": 1.0, "max_abs": 1.0,
            "min": -1.0, "max": 1.0, "nan_count": 0, "inf_count": 0,
        },
    }
    mps_run: dict[str, dict[str, object]] = {
        "latent": {
            "shape": (1,), "dtype": "torch.float32", "device": "mps:0",
            "mean": 0.0, "std": 1.0, "max_abs": 1.0,
            "min": -1.0, "max": 1.0, "nan_count": 0, "inf_count": 0,
        }
    }

    results = probe.compare_snapshots(cpu_run, mps_run)
    by_tag = {r["tag"]: r for r in results}

    assert by_tag["extra_cpu_only"]["diverged"] is True
    assert "missing" in str(by_tag["extra_cpu_only"]["reason"])


# ---------------------------------------------------------------------------
# 8. report includes diverged tag names
# ---------------------------------------------------------------------------

def test_report_includes_diverged_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)

    comparisons: list[dict[str, object]] = [
        {
            "tag": "cond_latent",
            "diverged": True,
            "reason": "delta_mean=0.42",
            "delta_mean": 0.42,
            "delta_std": 0.0,
            "delta_max_abs": 0.0,
        },
        {
            "tag": "uncond_latent",
            "diverged": False,
            "reason": "ok",
            "delta_mean": 0.0,
            "delta_std": 0.0,
            "delta_max_abs": 0.0,
        },
    ]

    text = probe.report(comparisons)
    assert "cond_latent" in text
    assert "uncond_latent" not in text or "All diverged" in text


# ---------------------------------------------------------------------------
# 9. report includes first diverged tag
# ---------------------------------------------------------------------------

def test_report_first_diverged_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _reload_probe(monkeypatch, active=True)

    comparisons: list[dict[str, object]] = [
        {
            "tag": "step_000",
            "diverged": True,
            "reason": "delta_mean=0.1",
            "delta_mean": 0.1,
            "delta_std": 0.0,
            "delta_max_abs": 0.0,
        },
        {
            "tag": "step_001",
            "diverged": True,
            "reason": "delta_mean=0.5",
            "delta_mean": 0.5,
            "delta_std": 0.0,
            "delta_max_abs": 0.0,
        },
    ]

    text = probe.report(comparisons)
    # The first diverged tag must appear prominently
    assert "step_000" in text
    first_idx = text.index("step_000")
    second_idx = text.index("step_001")
    # "First diverged tag" line appears before the table listing step_001
    assert first_idx < second_idx
