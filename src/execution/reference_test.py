"""Regression test for the solver exercises, run against a small fixed instance (Debug_1) with
known-good expected results.

Compares the full output dict against a fixture (expected_results.json), so it catches not just
"did it serve requests" but "did it produce exactly the numbers it used to." mip_solver/greedy/
consensus are compared against the fixture under total_profit, and mip_solver/greedy again under
total_customers (a different objective expression, greedy score and calc_objective_value branch);
dqn is a structural-only check (a freshly-initialized, untrained network has no stable expected
output to pin).

Run with: python -m unittest src.execution.reference_test
If solver behavior legitimately changes (e.g. a mip_gap/Threads tuning change), regenerate the
fixture with scripts/generate_reference_fixture.py before re-running this.
"""
import json
import unittest

from src.execution.reference_test_config import CHECKS, DEBUG_INSTANCE_PATH, FIXTURE_PATH
from src.simulation.run_simulation import run_taxi_simulation
from src.utilities import result_columns as col


def run_check(name: str) -> dict:
    """Run one algorithm against Debug_1 and return its output dict, minus the non-deterministic
    "runtime (s)" key.
    """
    config = CHECKS[name]()
    _, output_dict = run_taxi_simulation(test_folder=DEBUG_INSTANCE_PATH, config=config)
    output_dict.pop(col.RUNTIME_S, None)
    return output_dict


class ReferenceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH) as f:
            cls.expected = json.load(f)

    def test_mip_solver(self):
        self.assertEqual(run_check("mip_solver"), self.expected["mip_solver"])

    def test_mip_solver_customers(self):
        self.assertEqual(run_check("mip_solver_customers"), self.expected["mip_solver_customers"])

    def test_greedy(self):
        self.assertEqual(run_check("greedy"), self.expected["greedy"])

    def test_greedy_customers(self):
        self.assertEqual(run_check("greedy_customers"), self.expected["greedy_customers"])

    def test_consensus(self):
        self.assertEqual(run_check("consensus"), self.expected["consensus"])

    def test_dqn_structural(self):
        """dqn's own network is freshly initialized every call (no checkpoint), so there's no
        stable expected output to pin -- just confirm a full run completes and produces a
        well-formed result.
        """
        output_dict = run_check("dqn")
        self.assertIn(col.NUM_SERVED, output_dict)
        served = output_dict[col.NUM_SERVED]
        total_requests = served + output_dict[col.NUM_REJECTED]
        self.assertTrue(0 <= served <= total_requests)


if __name__ == "__main__":
    unittest.main()
