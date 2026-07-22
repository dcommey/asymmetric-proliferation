"""Reproducible experiments used in the paper."""

from __future__ import annotations

import csv
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from .model import Calibration, Outcome, compare_policies


EVIDENCE_DIR = Path(__file__).resolve().parent / "data"


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


def _van_der_corput(index: int, base: int) -> float:
    """Return one element of a base-``base`` van der Corput sequence."""
    value, denominator = 0.0, 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _scale(unit_value: float, lower: float, upper: float) -> float:
    return lower + unit_value * (upper - lower)


def robustness_scan(base: Calibration, samples: int = 2048) -> List[Dict[str, object]]:
    """Deterministic global sensitivity scan over a documented parameter box.

    The prime-base low-discrepancy design covers ten uncertain inputs without
    assigning a probability distribution to them.  Rows are design points, not
    estimates of how likely any calibration is in the world.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    rows: List[Dict[str, object]] = []
    for sample_id in range(1, samples + 1):
        u = [_van_der_corput(sample_id, prime) for prime in primes]
        lambda_defender = _scale(u[0], 0.35, 1.20)
        rate_ratio = _scale(u[1], 0.30, 3.00)
        conversion_ratio = _scale(u[4], 0.60, 1.80)
        c = base.with_changes(
            lambda_defender=lambda_defender,
            lambda_adversary=rate_ratio * lambda_defender,
            opportunistic_misuse=_scale(u[2], 0.10, 1.40),
            defensive_externality=_scale(u[3], 0.10, 1.20),
            adversary_uplift=conversion_ratio * base.defender_uplift,
            deploy_rate=_scale(u[5], 1.00, 4.00),
            guardrail_deterrence=_scale(u[6], 0.30, 0.85),
            guardrail_friction=_scale(u[7], 0.04, 0.25),
            irreversibility_guarded=_scale(u[8], 0.15, 0.65),
            prerelease_delay_cost=_scale(u[9], 0.03, 0.20),
        )
        ranked = list(compare_policies(c).values())
        rows.append(
            {
                "sample_id": sample_id,
                "lambda_defender": lambda_defender,
                "adversary_defender_rate_ratio": rate_ratio,
                "opportunistic_misuse": c.opportunistic_misuse,
                "defensive_externality": c.defensive_externality,
                "offense_defense_conversion_ratio": conversion_ratio,
                "deploy_rate": c.deploy_rate,
                "guardrail_deterrence": c.guardrail_deterrence,
                "guardrail_friction": c.guardrail_friction,
                "irreversibility_guarded": c.irreversibility_guarded,
                "prerelease_delay_cost": c.prerelease_delay_cost,
                "policy": ranked[0].policy,
                "runner_up": ranked[1].policy,
                "margin": ranked[0].welfare - ranked[1].welfare,
            }
        )
    return rows


def _quartile(value: float, lower: float, upper: float) -> int:
    position = min(0.999999, max(0.0, (value - lower) / (upper - lower)))
    return int(position * 4) + 1


def summarize_robustness(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate winning-policy shares by quartile for three focal inputs."""
    materialized = list(rows)
    specifications = (
        ("substitution ratio", "adversary_defender_rate_ratio", 0.30, 3.00),
        ("opportunistic misuse", "opportunistic_misuse", 0.10, 1.40),
        ("defensive externality", "defensive_externality", 0.10, 1.20),
    )
    output: List[Dict[str, object]] = []
    for label, key, lower, upper in specifications:
        for quartile in range(1, 5):
            subset = [
                row
                for row in materialized
                if _quartile(float(row[key]), lower, upper) == quartile
            ]
            for policy in ("controlled", "prerelease", "open_guarded", "open_minimal"):
                count = sum(row["policy"] == policy for row in subset)
                output.append(
                    {
                        "parameter": label,
                        "quartile": quartile,
                        "lower": lower + (quartile - 1) * (upper - lower) / 4,
                        "upper": lower + quartile * (upper - lower) / 4,
                        "policy": policy,
                        "count": count,
                        "share": count / len(subset),
                    }
                )
    return output


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
        "robustness": output / "robustness_scan.csv",
        "robustness_summary": output / "robustness_summary.csv",
        "release_evidence": output / "release_evidence.csv",
        "cyber_evidence": output / "cyber_evidence.csv",
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
    robustness = robustness_scan(base)
    write_csv(files["robustness"], robustness)
    write_csv(files["robustness_summary"], summarize_robustness(robustness))
    for name in ("release_evidence", "cyber_evidence"):
        source = EVIDENCE_DIR / f"{name}.csv"
        if not source.exists():
            raise FileNotFoundError(f"missing evidence file: {source}")
        shutil.copyfile(source, files[name])
    return files
