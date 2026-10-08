"""Experiments in the paper. All runs give the same results."""

from __future__ import annotations

import csv
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from .model import (
    POLICIES,
    Calibration,
    compare_policies,
    evaluate_policy,
    rank_distinct_outcomes,
)


EVIDENCE_DIR = Path(__file__).resolve().parent / "data"


ROBUSTNESS_BOXES = {
    "narrow": {
        "lambda_defender": (0.50, 1.00),
        "rate_ratio": (0.60, 2.20),
        "opportunistic_misuse": (0.25, 1.10),
        "defensive_externality": (0.25, 0.95),
        "conversion_ratio": (0.75, 1.50),
        "deploy_rate": (1.50, 3.50),
        "guardrail_deterrence": (0.40, 0.78),
        "guardrail_friction": (0.07, 0.20),
        "irreversibility_guarded": (0.25, 0.55),
        "controlled_tail_cost": (0.00, 0.45),
        "prerelease_delay_cost": (0.05, 0.16),
        "open_benefit_scale": (0.60, 1.40),
        "minimal_irreversibility_extra": (0.05, 0.25),
    },
    "reference": {
        "lambda_defender": (0.35, 1.20),
        "rate_ratio": (0.30, 3.00),
        "opportunistic_misuse": (0.10, 1.40),
        "defensive_externality": (0.10, 1.20),
        "conversion_ratio": (0.60, 1.80),
        "deploy_rate": (1.00, 4.00),
        "guardrail_deterrence": (0.30, 0.85),
        "guardrail_friction": (0.04, 0.25),
        "irreversibility_guarded": (0.15, 0.65),
        "controlled_tail_cost": (0.00, 0.65),
        "prerelease_delay_cost": (0.03, 0.20),
        "open_benefit_scale": (0.40, 1.60),
        "minimal_irreversibility_extra": (0.00, 0.30),
    },
    "wide": {
        "lambda_defender": (0.20, 1.50),
        "rate_ratio": (0.15, 4.00),
        "opportunistic_misuse": (0.00, 1.80),
        "defensive_externality": (0.00, 1.50),
        "conversion_ratio": (0.40, 2.20),
        "deploy_rate": (0.50, 5.00),
        "guardrail_deterrence": (0.10, 0.95),
        "guardrail_friction": (0.00, 0.35),
        "irreversibility_guarded": (0.05, 0.90),
        "controlled_tail_cost": (0.00, 0.90),
        "prerelease_delay_cost": (0.00, 0.30),
        "open_benefit_scale": (0.00, 2.00),
        "minimal_irreversibility_extra": (0.00, 0.45),
    },
}


def _linspace(start: float, stop: float, count: int) -> List[float]:
    if count < 2:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _region_row(c: Calibration) -> Dict[str, object]:
    """Get the winner, the second policy, and the welfare of each policy at one point.

    The figures use the welfare of each policy to draw smooth region boundaries.
    Each boundary is a zero contour of a welfare difference.
    """
    outcomes = {policy: evaluate_policy(c, policy) for policy in POLICIES}
    ranked = list(rank_distinct_outcomes(c))
    best, runner_up = ranked[:2]
    row: Dict[str, object] = {
        "policy": best.policy,
        "welfare": best.welfare,
        "runner_up": runner_up.policy,
        "margin": best.welfare - runner_up.welfare,
    }
    for policy, outcome in outcomes.items():
        row[f"welfare_{policy}"] = outcome.welfare
    row["window"] = outcomes["prerelease"].window
    return row


def phase_diagram(base: Calibration, size: int = 61) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ratio in _linspace(0.15, 3.5, size):
        for misuse in _linspace(0.0, 1.8, size):
            c = base.with_changes(
                lambda_adversary=ratio * base.lambda_defender,
                opportunistic_misuse=misuse,
            )
            rows.append(
                {
                    "adversary_defender_rate_ratio": ratio,
                    "opportunistic_misuse": misuse,
                    **_region_row(c),
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
            rows.append(
                {
                    "adversary_defender_rate_ratio": ratio,
                    "defensive_externality": eta,
                    **_region_row(c),
                }
            )
    return rows


def uplift_diagram(base: Calibration, size: int = 61) -> List[Dict[str, object]]:
    """Change the adversary substitution rate and the ratio of direct uplifts."""
    rows: List[Dict[str, object]] = []
    for rate_ratio in _linspace(0.15, 3.5, size):
        for conversion_ratio in _linspace(0.40, 2.50, size):
            c = base.with_changes(
                lambda_adversary=rate_ratio * base.lambda_defender,
                adversary_uplift=conversion_ratio * base.defender_uplift,
            )
            rows.append(
                {
                    "adversary_defender_rate_ratio": rate_ratio,
                    "adversary_defender_uplift_ratio": conversion_ratio,
                    **_region_row(c),
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
    """Return one element of a van der Corput sequence in base ``base``."""
    value, denominator = 0.0, 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _scale(unit_value: float, lower: float, upper: float) -> float:
    return lower + unit_value * (upper - lower)


def scan_calibration(base: Calibration, point: Dict[str, float]) -> Calibration:
    """Change one point of the sensitivity design into a calibration.

    ``open_benefit_scale`` multiplies the benefit of each policy above the
    benefit of controlled access. A value of one keeps the baseline order. A
    value of zero removes all benefit advantages of release.
    ``minimal_irreversibility_extra`` is the added one-time loss of minimally
    restricted release, compared with safeguarded release.
    """
    scale = point["open_benefit_scale"]
    premium = lambda value: base.benefit_controlled + scale * (value - base.benefit_controlled)
    return base.with_changes(
        lambda_defender=point["lambda_defender"],
        lambda_adversary=point["rate_ratio"] * point["lambda_defender"],
        opportunistic_misuse=point["opportunistic_misuse"],
        defensive_externality=point["defensive_externality"],
        adversary_uplift=point["conversion_ratio"] * base.defender_uplift,
        deploy_rate=point["deploy_rate"],
        guardrail_deterrence=point["guardrail_deterrence"],
        guardrail_friction=point["guardrail_friction"],
        irreversibility_guarded=point["irreversibility_guarded"],
        irreversibility_minimal=point["irreversibility_guarded"]
        + point["minimal_irreversibility_extra"],
        controlled_tail_cost=point["controlled_tail_cost"],
        prerelease_delay_cost=point["prerelease_delay_cost"],
        benefit_prerelease=premium(base.benefit_prerelease),
        benefit_open_guarded=premium(base.benefit_open_guarded),
        benefit_open_minimal=premium(base.benefit_open_minimal),
    )


SCAN_INPUTS = (
    "lambda_defender", "rate_ratio", "opportunistic_misuse", "defensive_externality",
    "conversion_ratio", "deploy_rate", "guardrail_deterrence", "guardrail_friction",
    "irreversibility_guarded", "controlled_tail_cost", "prerelease_delay_cost",
    "open_benefit_scale", "minimal_irreversibility_extra",
)


def scan_point(sample_id: int, box_name: str = "reference") -> Dict[str, float]:
    """Return design point ``sample_id`` (counted from 1) for a named parameter box."""
    bounds = ROBUSTNESS_BOXES[box_name]
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    return {
        key: _scale(_van_der_corput(sample_id, prime), *bounds[key])
        for key, prime in zip(SCAN_INPUTS, primes)
    }


def robustness_scan(
    base: Calibration, samples: int = 2048, box_name: str = "reference"
) -> List[Dict[str, object]]:
    """Do a deterministic global sensitivity scan over one parameter box.

    The Halton design covers 13 uncertain inputs. It does not give a probability
    distribution to the inputs. Each row is a design point. A row is not an
    estimate of the probability of a calibration.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    if box_name not in ROBUSTNESS_BOXES:
        raise ValueError(f"unknown robustness box: {box_name}")
    rows: List[Dict[str, object]] = []
    for sample_id in range(1, samples + 1):
        point = scan_point(sample_id, box_name)
        c = scan_calibration(base, point)
        ranked = list(rank_distinct_outcomes(c))
        rows.append(
            {
                "box": box_name,
                "sample_id": sample_id,
                "lambda_defender": point["lambda_defender"],
                "adversary_defender_rate_ratio": point["rate_ratio"],
                "opportunistic_misuse": c.opportunistic_misuse,
                "defensive_externality": c.defensive_externality,
                "adversary_defender_uplift_ratio": point["conversion_ratio"],
                "deploy_rate": c.deploy_rate,
                "guardrail_deterrence": c.guardrail_deterrence,
                "guardrail_friction": c.guardrail_friction,
                "irreversibility_guarded": c.irreversibility_guarded,
                "controlled_tail_cost": c.controlled_tail_cost,
                "prerelease_delay_cost": c.prerelease_delay_cost,
                "open_benefit_scale": point["open_benefit_scale"],
                "irreversibility_minimal": c.irreversibility_minimal,
                "minimal_irreversibility_extra": point["minimal_irreversibility_extra"],
                "policy": ranked[0].policy,
                "runner_up": ranked[1].policy,
                "margin": ranked[0].welfare - ranked[1].welfare,
            }
        )
    return rows


def robustness_box_scan(
    base: Calibration, samples: int = 2048
) -> List[Dict[str, object]]:
    """Run the same Halton design over the three nested parameter boxes."""
    rows: List[Dict[str, object]] = []
    for box_name in ("narrow", "reference", "wide"):
        rows.extend(robustness_scan(base, samples=samples, box_name=box_name))
    return rows


def _quantile(values: List[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_robustness_boxes(
    rows: Iterable[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Calculate the policy shares and winner margins in each parameter box."""
    materialized = list(rows)
    output: List[Dict[str, object]] = []
    for box_name in ("narrow", "reference", "wide"):
        subset = [row for row in materialized if row["box"] == box_name]
        margins = [float(row["margin"]) for row in subset]
        for policy in ("controlled", "prerelease", "open_guarded", "open_minimal"):
            count = sum(row["policy"] == policy for row in subset)
            output.append(
                {
                    "box": box_name,
                    "policy": policy,
                    "count": count,
                    "share": count / len(subset),
                    "median_winner_margin": _quantile(margins, 0.50),
                    "p10_winner_margin": _quantile(margins, 0.10),
                }
            )
    return output


def open_delay_diagram(
    base: Calibration, size: int = 41, maximum_delay: float = 0.75
) -> List[Dict[str, object]]:
    """Change the delays from weight release to effective use after open release."""
    rows: List[Dict[str, object]] = []
    for adversary_delay in _linspace(0.0, maximum_delay, size):
        for defender_delay in _linspace(0.0, maximum_delay, size):
            c = base.with_changes(
                open_adversary_delay=adversary_delay,
                open_defender_delay=defender_delay,
            )
            rows.append(
                {
                    "open_adversary_delay": adversary_delay,
                    "open_defender_delay": defender_delay,
                    **_region_row(c),
                }
            )
    return rows


def _quartile(value: float, lower: float, upper: float) -> int:
    position = min(0.999999, max(0.0, (value - lower) / (upper - lower)))
    return int(position * 4) + 1


def summarize_robustness(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Calculate the winning-policy shares in each quarter of the range of six inputs."""
    materialized = list(rows)
    specifications = (
        ("substitution ratio", "adversary_defender_rate_ratio", 0.30, 3.00),
        ("opportunistic misuse", "opportunistic_misuse", 0.10, 1.40),
        ("defensive externality", "defensive_externality", 0.10, 1.20),
        ("controlled tail cost", "controlled_tail_cost", 0.00, 0.65),
        ("open benefit scale", "open_benefit_scale", 0.40, 1.60),
        ("minimal-release loss increment", "minimal_irreversibility_extra", 0.00, 0.30),
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


def run(output: Path, size: int = 161) -> Dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    base = Calibration()
    files = {
        "phase": output / "phase_diagram.csv",
        "externality": output / "externality_diagram.csv",
        "uplift": output / "uplift_diagram.csv",
        "slices": output / "policy_slices.csv",
        "calibration": output / "calibration.csv",
        "summary": output / "policy_summary.csv",
        "robustness": output / "robustness_scan.csv",
        "robustness_summary": output / "robustness_summary.csv",
        "robustness_boxes": output / "robustness_boxes.csv",
        "robustness_box_summary": output / "robustness_box_summary.csv",
        "open_delays": output / "open_delay_diagram.csv",
        "release_evidence": output / "release_evidence.csv",
        "cyber_evidence": output / "cyber_evidence.csv",
        "incident_evidence": output / "incident_evidence.csv",
    }
    write_csv(files["phase"], phase_diagram(base, size))
    write_csv(files["externality"], externality_diagram(base, size))
    write_csv(files["uplift"], uplift_diagram(base, size))
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
    box_rows = robustness_box_scan(base)
    write_csv(files["robustness_boxes"], box_rows)
    write_csv(
        files["robustness_box_summary"], summarize_robustness_boxes(box_rows)
    )
    write_csv(files["open_delays"], open_delay_diagram(base, size=size))
    for name in ("release_evidence", "cyber_evidence", "incident_evidence"):
        source = EVIDENCE_DIR / f"{name}.csv"
        if not source.exists():
            raise FileNotFoundError(f"missing evidence file: {source}")
        shutil.copyfile(source, files[name])
    return files
