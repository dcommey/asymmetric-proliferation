"""Welfare model for dual-use model release under asymmetric proliferation.

Time is measured in years. Flow quantities are discounted continuously.  The
calibration is deliberately transparent and illustrative; it is not presented
as a point estimate of real-world welfare.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log1p
from typing import Dict, Iterable, Optional, Tuple


POLICIES: Tuple[str, ...] = ("controlled", "prerelease", "open_guarded", "open_minimal")


@dataclass(frozen=True)
class Calibration:
    """Parameters for the social-welfare comparison."""

    rho: float = 0.35
    lambda_defender: float = 0.70
    lambda_adversary: float = 0.95
    deploy_rate: float = 2.50
    baseline_gap: float = 0.05
    adversary_uplift: float = 1.10
    defender_uplift: float = 0.95
    defensive_externality: float = 0.55
    defender_reach_controlled: float = 0.18
    defender_reach_prerelease: float = 0.48
    defender_reach_open: float = 1.00
    opportunistic_misuse: float = 0.62
    controlled_misuse_share: float = 0.04
    guardrail_deterrence: float = 0.62
    guardrail_friction: float = 0.13
    harm_curvature: float = 1.60
    harm_scale: float = 1.00
    benefit_controlled: float = 0.08
    benefit_prerelease: float = 0.17
    benefit_open_guarded: float = 0.34
    benefit_open_minimal: float = 0.40
    controlled_tail_cost: float = 0.0
    irreversibility_guarded: float = 0.38
    irreversibility_minimal: float = 0.52
    open_adversary_delay: float = 0.0
    open_defender_delay: float = 0.0
    prerelease_window_max: float = 2.0
    prerelease_delay_cost: float = 0.10
    window_search_points: int = 81
    window_refinement_steps: int = 48

    def with_changes(self, **kwargs: float) -> "Calibration":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class Outcome:
    policy: str
    welfare: float
    discounted_harm: float
    discounted_benefit: float
    irreversibility: float
    window: Optional[float] = None


def _validate(c: Calibration) -> None:
    positive = {
        "rho": c.rho,
        "lambda_defender": c.lambda_defender,
        "lambda_adversary": c.lambda_adversary,
        "deploy_rate": c.deploy_rate,
        "harm_curvature": c.harm_curvature,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    shares = {
        "controlled_misuse_share": c.controlled_misuse_share,
        "guardrail_deterrence": c.guardrail_deterrence,
        "guardrail_friction": c.guardrail_friction,
    }
    for name, value in shares.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    nonnegative = {
        "controlled_tail_cost": c.controlled_tail_cost,
        "irreversibility_guarded": c.irreversibility_guarded,
        "irreversibility_minimal": c.irreversibility_minimal,
        "open_adversary_delay": c.open_adversary_delay,
        "open_defender_delay": c.open_defender_delay,
        "prerelease_window_max": c.prerelease_window_max,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if c.window_search_points < 3:
        raise ValueError("window_search_points must be at least 3")
    if c.window_refinement_steps < 1:
        raise ValueError("window_refinement_steps must be positive")


def access_exposure(rate: float, rho: float) -> float:
    """Expected discounted time with access after an exponential waiting time."""
    if rate < 0 or rho <= 0:
        raise ValueError("rate must be nonnegative and rho must be positive")
    return rate / (rho * (rate + rho))


def acquisition_best_response(
    base_rate: float,
    productivity: float,
    access_value: float,
    effort_cost: float,
    rho: float,
) -> Tuple[float, float]:
    """Unique costly-effort best response and the resulting acquisition rate.

    The actor maximizes ``access_value * access_exposure(rate, rho)`` minus
    ``effort_cost * effort**2 / 2``, with
    ``rate = base_rate + productivity * effort``.
    """
    if base_rate < 0 or min(productivity, access_value, effort_cost, rho) <= 0:
        raise ValueError("base_rate must be nonnegative and other inputs positive")

    def foc(effort: float) -> float:
        rate = base_rate + productivity * effort
        marginal_benefit = access_value * productivity / (rate + rho) ** 2
        return marginal_benefit - effort_cost * effort

    low, high = 0.0, 1.0
    while foc(high) > 0:
        high *= 2.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if foc(mid) > 0:
            low = mid
        else:
            high = mid
    effort = 0.5 * (low + high)
    return effort, base_rate + productivity * effort


def marginal_empowerment(usefulness: float, substitute_rate: float, horizon: float) -> float:
    """Capability added by immediate release at a finite policy horizon.

    Under restriction, the actor has obtained a substitute by ``horizon`` with
    probability ``1-exp(-substitute_rate*horizon)``. Immediate release closes
    the remaining access gap, so its marginal capability effect is the model's
    usefulness times the probability the actor would still lack a substitute.
    """
    if min(usefulness, substitute_rate, horizon) < 0:
        raise ValueError("inputs must be nonnegative")
    return usefulness * exp(-substitute_rate * horizon)


def capability_moat(
    privileged_rate: float,
    constrained_rate: float,
    usefulness: float,
    horizon: float,
) -> float:
    """Expected restricted-access capability gap at a finite horizon."""
    if min(privileged_rate, constrained_rate, usefulness, horizon) < 0:
        raise ValueError("inputs must be nonnegative")
    privileged = usefulness * (1.0 - exp(-privileged_rate * horizon))
    constrained = usefulness * (1.0 - exp(-constrained_rate * horizon))
    return privileged - constrained


def proliferation_threshold(
    psi_zero: float,
    offense_increment: float,
    rho: float,
) -> Optional[float]:
    """Closed-form adversary-substitution threshold in the linear benchmark.

    ``psi_zero`` is broad-release welfare minus controlled-access welfare when
    the sophisticated-adversary acquisition rate is zero.
    ``offense_increment`` is the product ``alpha*q_s`` from the paper. A
    threshold exists only when control wins at zero and broad release wins as
    the acquisition rate tends to infinity.
    """
    if offense_increment <= 0 or rho <= 0:
        raise ValueError("offense_increment and rho must be positive")
    psi_infinity = psi_zero + offense_increment / rho
    if psi_zero >= 0 or psi_infinity <= 0:
        return None
    theta = -rho * psi_zero / offense_increment
    return rho * theta / (1.0 - theta)


def defender_window_success(deploy_rate: float, adversary_rate: float, window: float) -> float:
    """P(T_deploy < min(T_adversary, window)) for independent exponentials."""
    if min(deploy_rate, adversary_rate, window) < 0:
        raise ValueError("rates and window must be nonnegative")
    total = deploy_rate + adversary_rate
    if total == 0:
        return 0.0
    return deploy_rate / total * (1.0 - exp(-total * window))


def _damage(gap: float, c: Calibration) -> float:
    """Increasing, convex softplus damage function."""
    z = c.harm_curvature * gap
    if z > 35:
        softplus = z
    elif z < -35:
        softplus = exp(z)
    else:
        softplus = log1p(exp(z))
    return c.harm_scale * softplus / c.harm_curvature


def _state_harm(p_s: float, p_d: float, defender_gain: float, c: Calibration) -> float:
    """Expected strategic harm for independent binary access/deployment states."""
    result = 0.0
    for s, ps in ((0, 1.0 - p_s), (1, p_s)):
        for d, pd in ((0, 1.0 - p_d), (1, p_d)):
            gap = c.baseline_gap + c.adversary_uplift * s - defender_gain * d
            result += ps * pd * _damage(gap, c)
    return result


def _controlled_harm(c: Calibration) -> float:
    ls, ld, rho = c.lambda_adversary, c.lambda_defender, c.rho
    v00 = 1.0 / (rho + ls + ld)
    v10 = 1.0 / (rho + ld) - v00
    v01 = 1.0 / (rho + ls) - v00
    v11 = 1.0 / rho - v00 - v10 - v01
    d_gain = c.defender_uplift + c.defensive_externality * c.defender_reach_controlled
    strategic = (
        v00 * _damage(c.baseline_gap, c)
        + v10 * _damage(c.baseline_gap + c.adversary_uplift, c)
        + v01 * _damage(c.baseline_gap - d_gain, c)
        + v11 * _damage(c.baseline_gap + c.adversary_uplift - d_gain, c)
    )
    misuse = c.controlled_misuse_share * c.opportunistic_misuse / rho
    return strategic + misuse


def _discounted_interval(rate: float, stop: float) -> float:
    """Integral of ``exp(-rate*t)`` from zero to ``stop``."""
    if stop <= 0:
        return 0.0
    return (1.0 - exp(-rate * stop)) / rate


def _discounted_open_strategic_harm(
    c: Calibration, defender_gain: float
) -> float:
    """Strategic harm after release with deterministic effective-use delays.

    Weight publication occurs at time zero.  The two delay parameters measure
    artifact-to-effective-use time for the sophisticated actor and defender.
    Zero delays recover the immediate-effective-access benchmark exactly.
    """
    delays = sorted({0.0, c.open_adversary_delay, c.open_defender_delay})
    total = 0.0
    for index, start in enumerate(delays):
        stop = delays[index + 1] if index + 1 < len(delays) else None
        adversary_has_access = start >= c.open_adversary_delay
        defender_has_access = start >= c.open_defender_delay
        gap = c.baseline_gap
        if adversary_has_access:
            gap += c.adversary_uplift
        if defender_has_access:
            gap -= defender_gain
        if stop is None:
            weight = exp(-c.rho * start) / c.rho
        else:
            weight = (exp(-c.rho * start) - exp(-c.rho * stop)) / c.rho
        total += weight * _damage(gap, c)
    return total


def _discounted_open_harm(c: Calibration, guarded: bool) -> float:
    friction = c.guardrail_friction if guarded else 0.0
    deterrence = c.guardrail_deterrence if guarded else 0.0
    defender_gain = (1.0 - friction) * (
        c.defender_uplift + c.defensive_externality * c.defender_reach_open
    )
    strategic = _discounted_open_strategic_harm(c, defender_gain)
    misuse = (1.0 - deterrence) * c.opportunistic_misuse / c.rho
    return strategic + misuse


def _open_outcome(c: Calibration, guarded: bool) -> Outcome:
    policy = "open_guarded" if guarded else "open_minimal"
    harm = _discounted_open_harm(c, guarded)
    benefit_flow = c.benefit_open_guarded if guarded else c.benefit_open_minimal
    benefit = benefit_flow / c.rho
    irreversibility = c.irreversibility_guarded if guarded else c.irreversibility_minimal
    return Outcome(policy, benefit - harm - irreversibility, harm, benefit, irreversibility)


def _prerelease_harm(c: Calibration, tau: float) -> float:
    """Exact pre-window harm plus the guarded-open continuation value."""
    pre_gain = c.defender_uplift + c.defensive_externality * c.defender_reach_prerelease
    h00 = _damage(c.baseline_gap, c)
    h10 = _damage(c.baseline_gap + c.adversary_uplift, c)
    h01 = _damage(c.baseline_gap - pre_gain, c)
    h11 = _damage(c.baseline_gap + c.adversary_uplift - pre_gain, c)
    pre = h11 * _discounted_interval(c.rho, tau)
    pre += (h01 - h11) * _discounted_interval(
        c.rho + c.lambda_adversary, tau
    )
    pre += (h10 - h11) * _discounted_interval(c.rho + c.deploy_rate, tau)
    pre += (h00 - h10 - h01 + h11) * _discounted_interval(
        c.rho + c.lambda_adversary + c.deploy_rate, tau
    )
    pre += (
        c.controlled_misuse_share
        * c.opportunistic_misuse
        * _discounted_interval(c.rho, tau)
    )
    post = exp(-c.rho * tau) * _discounted_open_harm(c, guarded=True)
    return pre + post


def prerelease_outcome(c: Calibration, tau: float) -> Outcome:
    """Evaluate a fixed defender-first window followed by guarded release."""
    _validate(c)
    if not 0 <= tau <= c.prerelease_window_max:
        raise ValueError("tau must lie in [0, prerelease_window_max]")
    harm = _prerelease_harm(c, tau)
    # Benefits accrue during selected access, then as guarded-open benefits.
    pre_benefit = c.benefit_prerelease * (1.0 - exp(-c.rho * tau)) / c.rho
    post_benefit = exp(-c.rho * tau) * c.benefit_open_guarded / c.rho
    delay_cost = c.prerelease_delay_cost * tau
    benefit = pre_benefit + post_benefit - delay_cost
    open_irreversibility = exp(-c.rho * tau) * c.irreversibility_guarded
    controlled_exposure = 1.0 - exp(-c.rho * tau)
    controlled_tail_cost = controlled_exposure * c.controlled_tail_cost
    one_time_cost = open_irreversibility + controlled_tail_cost
    welfare = benefit - harm - one_time_cost
    return Outcome("prerelease", welfare, harm, benefit, one_time_cost, tau)


def _best_prerelease_outcome(c: Calibration) -> Outcome:
    """Deterministically maximize window welfare on a bounded continuum.

    A uniform global screen locates the best basin; golden-section refinement
    then searches the adjacent interval.  Endpoints are always evaluated, so
    ``tau=0`` reproduces guarded open release and the maximum allowed window is
    not silently excluded.
    """
    maximum = c.prerelease_window_max
    if maximum == 0:
        return prerelease_outcome(c, 0.0)
    points = c.window_search_points
    step = maximum / (points - 1)
    screened = [prerelease_outcome(c, index * step) for index in range(points)]
    best_index = max(range(points), key=lambda index: screened[index].welfare)
    if best_index in (0, points - 1):
        return screened[best_index]

    low = (best_index - 1) * step
    high = (best_index + 1) * step
    ratio = (5.0 ** 0.5 - 1.0) / 2.0
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_outcome = prerelease_outcome(c, left)
    right_outcome = prerelease_outcome(c, right)
    for _ in range(c.window_refinement_steps):
        if left_outcome.welfare < right_outcome.welfare:
            low, left, left_outcome = left, right, right_outcome
            right = low + ratio * (high - low)
            right_outcome = prerelease_outcome(c, right)
        else:
            high, right, right_outcome = right, left, left_outcome
            left = high - ratio * (high - low)
            left_outcome = prerelease_outcome(c, left)
    candidates = (screened[best_index], left_outcome, right_outcome)
    return max(candidates, key=lambda outcome: outcome.welfare)


def evaluate_policy(c: Calibration, policy: str) -> Outcome:
    _validate(c)
    if policy == "controlled":
        harm = _controlled_harm(c)
        benefit = c.benefit_controlled / c.rho
        tail_cost = c.controlled_tail_cost
        return Outcome(policy, benefit - harm - tail_cost, harm, benefit, tail_cost)
    if policy == "open_guarded":
        return _open_outcome(c, guarded=True)
    if policy == "open_minimal":
        return _open_outcome(c, guarded=False)
    if policy == "prerelease":
        return _best_prerelease_outcome(c)
    raise ValueError(f"unknown policy: {policy}")


def compare_policies(c: Calibration, policies: Iterable[str] = POLICIES) -> Dict[str, Outcome]:
    outcomes = {policy: evaluate_policy(c, policy) for policy in policies}
    return dict(sorted(outcomes.items(), key=lambda item: item[1].welfare, reverse=True))


def rank_distinct_outcomes(c: Calibration) -> Tuple[Outcome, ...]:
    """Rank economically distinct policy outcomes.

    The boundary ``P(0)`` is exactly guarded open release.  It remains
    available through :func:`prerelease_outcome` for analytic checks, but is
    removed from policy counts and runner-up margins when the optimized window
    collapses to zero.
    """
    outcomes = {policy: evaluate_policy(c, policy) for policy in POLICIES}
    prerelease = outcomes["prerelease"]
    if prerelease.window is not None and prerelease.window <= 1e-10:
        outcomes.pop("prerelease")
    return tuple(
        sorted(outcomes.values(), key=lambda outcome: outcome.welfare, reverse=True)
    )
