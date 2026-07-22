"""Reproducible experiments used in the paper."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .model import Calibration, Outcome, compare_policies


def _linspace(start: float, stop: float, count: int) -> List[float]:
    if count < 2:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def phase_diagram(base: Calibration, size: int = 61) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ratio in _linspace(0.15, 3.5, size):
        for misuse in _linspace(0.0, 1.8, size):
            c = base.with_changes(
                lambda_adversary=ratio * base.lambda_defender,
                opportunistic_misuse=misuse,
            )
            ranked = list(compare_policies(c).values())
            best, runner_up = ranked[:2]
            rows.append(
                {
                    "adversary_defender_rate_ratio": ratio,
                    "opportunistic_misuse": misuse,
                    "policy": best.policy,
                    "welfare": best.welfare,
                    "runner_up": runner_up.policy,
                    "margin": best.welfare - runner_up.welfare,
                    "window": "" if best.window is None else best.window,
                }
            )
    return rows


def externality_diagram(base: Calibration, size: int = 61) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ratio in _linspace(0.15, 3.5, size):
        for eta in _linspace(0.0, 1.5, size):
            c = base.with_changes(
                lambda_adversary=ratio * base.lambda_defender,
                defensive_externality=eta,
            )
            ranked = list(compare_policies(c).values())
            rows.append(
                {
                    "adversary_defender_rate_ratio": ratio,
                    "defensive_externality": eta,
                    "policy": ranked[0].policy,
                    "welfare": ranked[0].welfare,
                    "runner_up": ranked[1].policy,
                    "margin": ranked[0].welfare - ranked[1].welfare,
                }
            )
    return rows


def cost_exchange_diagram(base: Calibration, size: int = 61) -> List[Dict[str, object]]:
    """Vary adversary substitution and offense/defense direct conversion."""
    rows: List[Dict[str, object]] = []
    for rate_ratio in _linspace(0.15, 3.5, size):
        for conversion_ratio in _linspace(0.40, 2.50, size):
            c = base.with_changes(
                lambda_adversary=rate_ratio * base.lambda_defender,
                adversary_uplift=conversion_ratio * base.defender_uplift,
            )
            ranked = list(compare_policies(c).values())
            rows.append(
                {
                    "adversary_defender_rate_ratio": rate_ratio,
                    "offense_defense_conversion_ratio": conversion_ratio,
                    "policy": ranked[0].policy,
                    "welfare": ranked[0].welfare,
                    "runner_up": ranked[1].policy,
                    "margin": ranked[0].welfare - ranked[1].welfare,
                }
            )
    return rows


def policy_slices(base: Calibration, points: int = 121) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for misuse in (0.20, 0.62, 1.10):
        for ratio in _linspace(0.15, 3.5, points):
            c = base.with_changes(
                lambda_adversary=ratio * base.lambda_defender,
                opportunistic_misuse=misuse,
            )
            for outcome in compare_policies(c).values():
                rows.append(
                    {
                        "adversary_defender_rate_ratio": ratio,
                        "opportunistic_misuse": misuse,
                        "policy": outcome.policy,
                        "welfare": outcome.welfare,
                        "window": "" if outcome.window is None else outcome.window,
                    }
                )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, size: int = 61) -> Dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    base = Calibration()
    files = {
        "phase": output / "phase_diagram.csv",
        "externality": output / "externality_diagram.csv",
        "cost_exchange": output / "cost_exchange_diagram.csv",
        "slices": output / "policy_slices.csv",
        "calibration": output / "calibration.csv",
        "summary": output / "policy_summary.csv",
    }
    write_csv(files["phase"], phase_diagram(base, size))
    write_csv(files["externality"], externality_diagram(base, size))
    write_csv(files["cost_exchange"], cost_exchange_diagram(base, size))
    write_csv(files["slices"], policy_slices(base))
    write_csv(files["calibration"], [{"parameter": k, "value": v} for k, v in asdict(base).items()])
    summary = []
    for rank, outcome in enumerate(compare_policies(base).values(), start=1):
        row = asdict(outcome)
        row["rank"] = rank
        summary.append(row)
    write_csv(files["summary"], summary)
    return files
