import csv
import unittest
from datetime import date
from math import exp, log1p

from asymprolif.experiments import (
    EVIDENCE_DIR,
    ROBUSTNESS_BOXES,
    open_delay_diagram,
    robustness_box_scan,
    robustness_scan,
    summarize_robustness,
    summarize_robustness_boxes,
)
from asymprolif.model import (
    Calibration,
    POLICIES,
    acquisition_best_response,
    access_exposure,
    capability_moat,
    compare_policies,
    defender_window_success,
    evaluate_policy,
    marginal_empowerment,
    prerelease_outcome,
    proliferation_threshold,
    rank_distinct_outcomes,
)


class AnalyticResultsTest(unittest.TestCase):
    def test_access_inversion_has_expected_sign(self):
        rho = 0.3
        self.assertGreater(access_exposure(1.2, rho), access_exposure(0.6, rho))
        self.assertEqual(access_exposure(0.6, rho), access_exposure(0.6, rho))

    def test_window_success_matches_limits(self):
        self.assertAlmostEqual(defender_window_success(2.0, 1.0, 0.0), 0.0)
        self.assertAlmostEqual(defender_window_success(2.0, 1.0, 100.0), 2.0 / 3.0)
        self.assertGreater(
            defender_window_success(2.0, 0.4, 1.0),
            defender_window_success(2.0, 2.0, 1.0),
        )

    def test_acquisition_best_response_satisfies_foc(self):
        effort, rate = acquisition_best_response(0.2, 0.8, 1.4, 1.1, 0.3)
        residual = 1.4 * 0.8 / (rate + 0.3) ** 2 - 1.1 * effort
        self.assertAlmostEqual(residual, 0.0, places=10)
        _, higher_rate = acquisition_best_response(0.2, 0.8, 2.0, 1.1, 0.3)
        self.assertGreater(higher_rate, rate)

    def test_asymmetric_empowerment_favors_slower_substituter(self):
        horizon = 1.0
        defender_gain = marginal_empowerment(1.0, 0.5, horizon)
        sophisticated_gain = marginal_empowerment(1.0, 1.5, horizon)
        self.assertGreater(defender_gain, sophisticated_gain)
        self.assertAlmostEqual(marginal_empowerment(2.0, 0.7, 0.0), 2.0)

    def test_capability_moat_tracks_access_advantage(self):
        self.assertGreater(capability_moat(1.4, 0.5, 1.0, 1.0), 0.0)
        self.assertAlmostEqual(capability_moat(0.8, 0.8, 1.0, 2.0), 0.0)

    def test_closed_form_proliferation_threshold_is_a_root(self):
        psi_zero, offense_increment, rho = -1.0, 1.0, 0.5
        threshold = proliferation_threshold(psi_zero, offense_increment, rho)
        self.assertIsNotNone(threshold)
        psi_at_threshold = psi_zero + offense_increment / rho * (
            threshold / (threshold + rho)
        )
        self.assertAlmostEqual(psi_at_threshold, 0.0)

    def test_proliferation_threshold_requires_an_interior_crossing(self):
        self.assertIsNone(proliferation_threshold(0.1, 1.0, 0.5))
        self.assertIsNone(proliferation_threshold(-3.0, 1.0, 0.5))


class WelfareModelTest(unittest.TestCase):
    def test_all_policies_return_finite_outcomes(self):
        outcomes = compare_policies(Calibration())
        self.assertEqual(set(outcomes), set(POLICIES))
        for outcome in outcomes.values():
            self.assertTrue(abs(outcome.welfare) < 1e6)

    def test_prerelease_selects_allowed_window(self):
        c = Calibration()
        outcome = evaluate_policy(c, "prerelease")
        self.assertGreaterEqual(outcome.window, 0.0)
        self.assertLessEqual(outcome.window, c.prerelease_window_max)

    def test_zero_window_equals_guarded_open_release(self):
        c = Calibration(open_adversary_delay=0.2, open_defender_delay=0.35)
        immediate = prerelease_outcome(c, 0.0)
        guarded = evaluate_policy(c, "open_guarded")
        self.assertAlmostEqual(immediate.welfare, guarded.welfare)
        self.assertAlmostEqual(immediate.discounted_harm, guarded.discounted_harm)
        self.assertAlmostEqual(immediate.discounted_benefit, guarded.discounted_benefit)
        self.assertAlmostEqual(immediate.irreversibility, guarded.irreversibility)

    def test_zero_window_boundary_is_counted_once(self):
        c = Calibration(prerelease_delay_cost=10.0)
        ranked = rank_distinct_outcomes(c)
        self.assertEqual(sum(outcome.policy == "open_guarded" for outcome in ranked), 1)
        self.assertFalse(any(outcome.policy == "prerelease" for outcome in ranked))

    def test_open_effective_use_delays_move_welfare_in_expected_directions(self):
        base = Calibration()
        delayed_adversary = base.with_changes(open_adversary_delay=0.25)
        delayed_defender = base.with_changes(open_defender_delay=0.25)
        benchmark = evaluate_policy(base, "open_guarded").welfare
        self.assertGreater(
            evaluate_policy(delayed_adversary, "open_guarded").welfare,
            benchmark,
        )
        self.assertLess(
            evaluate_policy(delayed_defender, "open_guarded").welfare,
            benchmark,
        )

    def test_exact_prerelease_harm_matches_numerical_integration(self):
        c = Calibration()
        tau = 0.73
        steps = 10000
        dt = tau / steps
        defender_gain = c.defender_uplift + c.defensive_externality * c.defender_reach_prerelease

        def damage(gap):
            return c.harm_scale * log1p(exp(c.harm_curvature * gap)) / c.harm_curvature

        total = 0.0
        for index in range(steps + 1):
            time = index * dt
            p_s = 1.0 - exp(-c.lambda_adversary * time)
            p_d = 1.0 - exp(-c.deploy_rate * time)
            strategic = (
                (1 - p_s) * (1 - p_d) * damage(c.baseline_gap)
                + p_s * (1 - p_d) * damage(c.baseline_gap + c.adversary_uplift)
                + (1 - p_s) * p_d * damage(c.baseline_gap - defender_gain)
                + p_s * p_d * damage(c.baseline_gap + c.adversary_uplift - defender_gain)
            )
            flow = strategic + c.controlled_misuse_share * c.opportunistic_misuse
            total += (0.5 if index in (0, steps) else 1.0) * exp(-c.rho * time) * flow
        numerical = total * dt
        numerical += exp(-c.rho * tau) * evaluate_policy(c, "open_guarded").discounted_harm
        self.assertAlmostEqual(
            prerelease_outcome(c, tau).discounted_harm,
            numerical,
            places=7,
        )

    def test_more_adversary_substitution_hurts_controlled_access(self):
        base = Calibration(lambda_adversary=0.25)
        fast = base.with_changes(lambda_adversary=3.0)
        self.assertLess(
            evaluate_policy(fast, "controlled").welfare,
            evaluate_policy(base, "controlled").welfare,
        )

    def test_guardrails_gain_value_as_opportunistic_misuse_rises(self):
        low = Calibration(opportunistic_misuse=0.1)
        high = low.with_changes(opportunistic_misuse=1.5)
        low_gap = evaluate_policy(low, "open_guarded").welfare - evaluate_policy(low, "open_minimal").welfare
        high_gap = evaluate_policy(high, "open_guarded").welfare - evaluate_policy(high, "open_minimal").welfare
        self.assertGreater(high_gap, low_gap)

    def test_controlled_tail_cost_lowers_controlled_welfare(self):
        base = Calibration(controlled_tail_cost=0.0)
        costly = base.with_changes(controlled_tail_cost=0.4)
        welfare_loss = evaluate_policy(base, "controlled").welfare - evaluate_policy(
            costly, "controlled"
        ).welfare
        self.assertAlmostEqual(welfare_loss, 0.4)

    def test_prerelease_inherits_partial_controlled_tail_cost(self):
        base = Calibration(controlled_tail_cost=0.0)
        costly = base.with_changes(controlled_tail_cost=0.4)
        welfare_loss = prerelease_outcome(base, 1.0).welfare - prerelease_outcome(
            costly, 1.0
        ).welfare
        expected = 0.4 * (1.0 - exp(-base.rho))
        self.assertAlmostEqual(welfare_loss, expected)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_policy(Calibration(), "unknown")

    def test_robustness_design_is_deterministic_and_complete(self):
        first = robustness_scan(Calibration(), samples=16)
        second = robustness_scan(Calibration(), samples=16)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertTrue(all(row["policy"] in POLICIES for row in first))

    def test_robustness_summary_shares_sum_to_one(self):
        summary = summarize_robustness(robustness_scan(Calibration(), samples=32))
        for parameter in {row["parameter"] for row in summary}:
            for quartile in range(1, 5):
                share = sum(
                    row["share"]
                    for row in summary
                    if row["parameter"] == parameter and row["quartile"] == quartile
                )
                self.assertAlmostEqual(share, 1.0)

    def test_nested_robustness_boxes_are_complete_and_normalized(self):
        rows = robustness_box_scan(Calibration(), samples=32)
        self.assertEqual(len(rows), 32 * len(ROBUSTNESS_BOXES))
        summary = summarize_robustness_boxes(rows)
        for box_name in ROBUSTNESS_BOXES:
            share = sum(
                row["share"] for row in summary if row["box"] == box_name
            )
            self.assertAlmostEqual(share, 1.0)

    def test_open_delay_grid_is_complete(self):
        rows = open_delay_diagram(Calibration(), size=7, maximum_delay=0.5)
        self.assertEqual(len(rows), 49)
        self.assertTrue(all(row["policy"] in POLICIES for row in rows))

    def test_release_evidence_dates_match_reported_lags(self):
        with (EVIDENCE_DIR / "release_evidence.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        for row in rows:
            announced = date.fromisoformat(row["announcement_date"])
            weights = date.fromisoformat(row["weight_date"])
            self.assertEqual((weights - announced).days, int(row["weight_lag_days"]))
            self.assertTrue(row["source_url"].startswith("https://"))

    def test_cyber_evidence_has_valid_intervals_and_costs(self):
        with (EVIDENCE_DIR / "cyber_evidence.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["metric"] == "frontier_lag":
                self.assertLessEqual(float(row["low"]), float(row["high"]))
            else:
                self.assertGreater(float(row["value"]), 0.0)

    def test_incident_evidence_preserves_provider_uncertainty(self):
        with (EVIDENCE_DIR / "incident_evidence.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        hosted = next(row for row in rows if row["case_id"] == "hf-hosted-refusal")
        self.assertEqual(hosted["model_or_service"], "Unnamed frontier models")
        self.assertIn("did not name", hosted["evidence_limit"])
        self.assertTrue(all(row["source_url"].startswith("https://") for row in rows))


if __name__ == "__main__":
    unittest.main()
