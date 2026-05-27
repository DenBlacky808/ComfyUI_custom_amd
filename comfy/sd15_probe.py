"""
SD 1.5 tensor-stat probe for CPU-vs-MPS divergence debugging.

Inactive by default. Activate by setting SD15_PROBE=1 in the environment.
All state lives in this module and has zero impact on normal runtime paths.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

# Module-level storage. Populated only when SD15_PROBE=1.
_snapshots: dict[str, dict[str, object]] = {}

_ACTIVE = os.environ.get("SD15_PROBE") == "1"


def maybe_snapshot(tag: str, tensor: torch.Tensor) -> None:
    """Capture scalar stats for *tensor* under *tag*.

    No-op unless SD15_PROBE=1.  Never stores raw tensor values or writes
    files.  Safe to call from any device; stats are computed on a temporary
    CPU view.
    """
    if not _ACTIVE:
        return

    with torch.no_grad():
        t = tensor.detach()

        shape: tuple[int, ...] = tuple(t.shape)
        dtype: str = str(t.dtype)
        device: str = str(t.device)

        if t.is_floating_point() or t.is_complex():
            c = t.cpu().float()
            mean: float = c.mean().item()
            std: float = c.std().item() if c.numel() > 1 else 0.0
            max_abs: float = c.abs().max().item() if c.numel() > 0 else 0.0
            min_val: float = c.min().item() if c.numel() > 0 else 0.0
            max_val: float = c.max().item() if c.numel() > 0 else 0.0
            nan_count: int = int(torch.isnan(c).sum().item())
            inf_count: int = int(torch.isinf(c).sum().item())
        else:
            # Integer / bool tensors: skip floating stats
            c = t.cpu()
            mean = float("nan")
            std = float("nan")
            max_abs = float("nan")
            min_val = float("nan")
            max_val = float("nan")
            nan_count = 0
            inf_count = 0

        _snapshots[tag] = {
            "shape": shape,
            "dtype": dtype,
            "device": device,
            "mean": mean,
            "std": std,
            "max_abs": max_abs,
            "min": min_val,
            "max": max_val,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }


def get_snapshots() -> dict[str, dict[str, object]]:
    """Return a shallow copy of the collected snapshot dictionary."""
    return dict(_snapshots)


def clear_snapshots() -> None:
    """Clear all stored snapshots."""
    _snapshots.clear()


def compare_snapshots(
    cpu_run: dict[str, dict[str, object]],
    mps_run: dict[str, dict[str, object]],
    threshold: float = 1e-3,
) -> list[dict[str, object]]:
    """Compare per-tag stats from two runs.

    For each tag in the union of both dictionaries, report whether the runs
    diverge.  A tag is flagged as diverged if any numeric delta exceeds
    *threshold*, or if the tag is missing from one run, or if shape / dtype
    differ.
    """
    all_tags = sorted(set(cpu_run) | set(mps_run))
    results: list[dict[str, object]] = []

    for tag in all_tags:
        in_cpu = tag in cpu_run
        in_mps = tag in mps_run

        if not in_cpu or not in_mps:
            missing_in = "cpu" if not in_cpu else "mps"
            results.append(
                {
                    "tag": tag,
                    "diverged": True,
                    "reason": f"tag missing in {missing_in} run",
                    "delta_mean": None,
                    "delta_std": None,
                    "delta_max_abs": None,
                }
            )
            continue

        cs = cpu_run[tag]
        ms = mps_run[tag]

        if cs["shape"] != ms["shape"]:
            results.append(
                {
                    "tag": tag,
                    "diverged": True,
                    "reason": f"shape mismatch: cpu={cs['shape']} mps={ms['shape']}",
                    "delta_mean": None,
                    "delta_std": None,
                    "delta_max_abs": None,
                }
            )
            continue

        if cs["dtype"] != ms["dtype"]:
            results.append(
                {
                    "tag": tag,
                    "diverged": True,
                    "reason": f"dtype mismatch: cpu={cs['dtype']} mps={ms['dtype']}",
                    "delta_mean": None,
                    "delta_std": None,
                    "delta_max_abs": None,
                }
            )
            continue

        def _safe_delta(a: object, b: object) -> float:
            if not isinstance(a, float) or not isinstance(b, float):
                return 0.0
            if math.isnan(a) and math.isnan(b):
                return 0.0
            if math.isnan(a) or math.isnan(b):
                return float("inf")
            return abs(a - b)

        delta_mean = _safe_delta(cs["mean"], ms["mean"])
        delta_std = _safe_delta(cs["std"], ms["std"])
        delta_max_abs = _safe_delta(cs["max_abs"], ms["max_abs"])

        diverged = any(d > threshold for d in (delta_mean, delta_std, delta_max_abs))

        reasons: list[str] = []
        if delta_mean > threshold:
            reasons.append(f"delta_mean={delta_mean:.4g}")
        if delta_std > threshold:
            reasons.append(f"delta_std={delta_std:.4g}")
        if delta_max_abs > threshold:
            reasons.append(f"delta_max_abs={delta_max_abs:.4g}")

        results.append(
            {
                "tag": tag,
                "diverged": diverged,
                "reason": ", ".join(reasons) if reasons else "ok",
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "delta_max_abs": delta_max_abs,
            }
        )

    return results


def report(comparisons: list[dict[str, object]]) -> str:
    """Return a Markdown string suitable for a GitHub PR or issue comment."""
    diverged = [c for c in comparisons if c.get("diverged")]
    ok_count = len(comparisons) - len(diverged)

    lines: list[str] = [
        "## SD15 CPU-vs-MPS tensor-stat probe report",
        "",
        f"**Total tags:** {len(comparisons)}  "
        f"**Diverged:** {len(diverged)}  "
        f"**OK:** {ok_count}",
        "",
    ]

    if diverged:
        first = diverged[0]
        lines.append(f"**First diverged tag:** `{first['tag']}`  ")
        lines.append(f"Reason: {first['reason']}")
        lines.append("")
        lines.append("### All diverged tags")
        lines.append("")
        lines.append("| tag | reason | delta_mean | delta_std | delta_max_abs |")
        lines.append("| --- | ------ | ---------- | --------- | ------------- |")
        for c in diverged:
            dm = c["delta_mean"]
            ds = c["delta_std"]
            da = c["delta_max_abs"]
            lines.append(
                f"| `{c['tag']}` | {c['reason']} "
                f"| {_fmt(dm)} | {_fmt(ds)} | {_fmt(da)} |"
            )
        lines.append("")
    else:
        lines.append("No divergence detected above threshold.")
        lines.append("")

    # List tags that were missing from one run
    missing = [c for c in comparisons if "missing" in str(c.get("reason", ""))]
    if missing:
        lines.append("### Missing tags")
        lines.append("")
        for c in missing:
            lines.append(f"- `{c['tag']}`: {c['reason']}")
        lines.append("")

    return "\n".join(lines)


def _fmt(val: object) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if math.isinf(val):
            return "∞"
        return f"{val:.4g}"
    return str(val)
