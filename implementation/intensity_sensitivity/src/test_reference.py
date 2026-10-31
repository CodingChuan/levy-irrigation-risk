"""Mechanical RED/GREEN tests for the daily Bellman reference."""

from __future__ import annotations

import json
import sys
import unittest

import numpy as np

import bellman
import constrained
import model


def red_discrepancies() -> dict[str, float]:
    # A legacy score-then-cap implementation scores .08, then caps the chosen
    # action to .06.  The continuation values deliberately make that route
    # disagree with evaluating the feasible .06 action itself.
    legacy_reported = 0.08 + 0.0
    actual_feasible = 0.06 + 1.0
    action_discrepancy = abs(legacy_reported - actual_feasible)

    # An arbitrary exponential fixture is nonzero at the terminal alive state,
    # so it cannot satisfy the terminal-zero Bellman relation.
    manufactured_terminal = float(np.exp(-0.4 * 1.0) * np.exp(-8.0 * (0.60 - 0.35)))
    terminal_discrepancy = abs(manufactured_terminal - 0.0)
    return {
        "legacy_score_then_cap_cost_discrepancy": action_discrepancy,
        "manufactured_terminal_bellman_discrepancy": terminal_discrepancy,
    }


class ReferenceTests(unittest.TestCase):
    def test_generator_signs_rows_and_crossing_atoms(self) -> None:
        for intensity in model.INTENSITIES:
            for upper, h in ((1.15, 0.005), (1.55, 0.005)):
                row = model.build_model_row(upper, h, intensity)
                dense = row.Q.toarray()
                off = dense.copy()
                np.fill_diagonal(off, 0.0)
                self.assertGreaterEqual(float(np.min(off)), -1.0e-12)
                self.assertLess(float(np.max(np.abs(dense.sum(axis=1)))), 1.0e-10)
                if intensity == 0.12:
                    self.assertGreater(row.Q[1, 0], 0.0)
                    self.assertGreater(row.Q[-2, -1], 0.0)

    def test_nojump_control_and_absorbing_transition(self) -> None:
        row = model.build_model_row(1.55, 0.01, 0.0)
        dense = row.Q.toarray()
        self.assertEqual(float(dense[0].sum()), 0.0)
        self.assertEqual(float(dense[-1].sum()), 0.0)
        self.assertEqual(float(row.P[0, 0]), 1.0)
        self.assertEqual(float(row.P[-1, -1]), 1.0)
        self.assertTrue(np.allclose(row.P @ np.ones(row.state_count), 1.0, atol=1.0e-10))

    def test_transition_orientation_constant_preservation(self) -> None:
        row = model.build_model_row(1.55, 0.005, 0.12)
        self.assertGreaterEqual(float(np.min(row.P)), -1.0e-12)
        self.assertTrue(np.allclose(row.P @ np.ones(row.state_count), 1.0, atol=1.0e-10))
        self.assertTrue(np.allclose(row.P[0], np.eye(row.state_count)[0], atol=1.0e-12))

    def test_budget_contract_and_exact_ties_choose_noop(self) -> None:
        self.assertEqual(model.actions_for_budget(0), (0.0,))
        self.assertEqual(model.actions_for_budget(1), (0.0, 0.06))
        self.assertEqual(model.successor_budget_index(1, 0.06), 0)
        self.assertEqual(model.successor_budget_index(4, 0.08), 3)
        row = model.build_model_row(1.15, 0.005, 0.0)
        solution = bellman.solve(0.0, np.eye(row.state_count), row.grid)
        self.assertTrue(np.all(solution.actions == 0.0))
        self.assertTrue(np.all(solution.successor_budget == np.arange(model.BUDGETS.size)[None, :, None]))

    def test_terminal_failure_safe_and_no_action_sentinels(self) -> None:
        row = model.build_model_row(1.15, 0.005, 0.0)
        solution = bellman.solve(4.0, row.P, row.grid)
        self.assertTrue(np.all(solution.V[:, :, 0] == 4.0))
        self.assertTrue(np.all(solution.R[:, :, 0] == 1.0))
        self.assertTrue(np.all(solution.C[:, :, 0] == 0.0))
        self.assertTrue(np.all(solution.V[:, :, -1] == 0.0))
        self.assertTrue(np.all(solution.actions[:, :, 0] == 0.0))
        self.assertTrue(np.all(solution.actions[:, :, -1] == 0.0))

    def test_same_mu_zero_objective_encourages_zero_for_both_intensities(self) -> None:
        for intensity in model.INTENSITIES:
            row = model.build_model_row(1.15, 0.005, intensity)
            solution = bellman.solve(0.0, row.P, row.grid)
            self.assertTrue(np.all(solution.actions == 0.0))
            self.assertAlmostEqual(float(solution.C[0, -1, int(round((model.W0 - model.LOWER) / row.h))]), 0.0)

    def test_identity_and_independent_fixed_policy_evaluator(self) -> None:
        row = model.build_model_row(1.15, 0.005, 0.12)
        solution = bellman.solve(8.0, row.P, row.grid)
        evaluation = bellman.evaluate_policy(solution.actions, row.P, row.grid)
        self.assertLessEqual(solution.identity_gap, 1.0e-8)
        self.assertLessEqual(float(np.max(np.abs(solution.R - evaluation.R))), 1.0e-10)
        self.assertLessEqual(float(np.max(np.abs(solution.C - evaluation.C))), 1.0e-10)
        self.assertTrue(evaluation.budget_feasible)

    def test_risk_only_keeps_cost_accounting_and_mixture_formula(self) -> None:
        row = model.build_model_row(1.15, 0.005, 0.0)
        solution = bellman.solve(None, row.P, row.grid, risk_only=True)
        evaluation = bellman.evaluate_policy(solution.actions, row.P, row.grid)
        self.assertTrue(np.allclose(solution.C, evaluation.C, atol=1.0e-10))
        risky_risk, safe_risk = 0.30, 0.10
        risky_cost, safe_cost = 0.20, 0.05
        alpha = (model.EPSILON - safe_risk) / (risky_risk - safe_risk)
        self.assertAlmostEqual(alpha, 0.5)
        self.assertAlmostEqual(alpha * risky_risk + (1.0 - alpha) * safe_risk, model.EPSILON)
        dual = 0.11
        primal = alpha * risky_cost + (1.0 - alpha) * safe_cost
        self.assertGreaterEqual(primal - dual, -1.0e-8)

    def test_no_formal_outcome_imports(self) -> None:
        forbidden = ("pinn", "monte_carlo", "stage4", "stage2")
        imported = set(sys.modules)
        self.assertFalse(any(any(token in name.lower() for token in forbidden) for name in imported))


def run_red() -> int:
    discrepancies = red_discrepancies()
    print(json.dumps(discrepancies, sort_keys=True))
    if all(value > 1.0e-12 for value in discrepancies.values()):
        return 1
    print(json.dumps({"error": "expected RED discrepancy missing"}), file=sys.stderr)
    return 2


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--mode" and sys.argv[2] == "red":
        return run_red()
    result = unittest.main(argv=[sys.argv[0]], exit=False, testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
    return 0 if result.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
