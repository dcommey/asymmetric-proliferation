import unittest

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
    proliferation_threshold,
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
        self.assertIn(outcome.window, c.prerelease_windows)

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

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_policy(Calibration(), "unknown")


if __name__ == "__main__":
    unittest.main()
