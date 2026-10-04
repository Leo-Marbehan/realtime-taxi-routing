"""Shared SimulationConfig builders for reference_test.py and scripts/generate_reference_fixture.py.

Kept in one place so the test and the fixture generator can never drift apart -- both call the same
CHECKS[name]() builder to construct the exact SimulationConfig used for that algorithm.
"""
import os

from src.utilities.config import ConsensusConfig, DQNConfig, SimulationConfig
from src.utilities.config_tools import derive_consensus_params
from src.utilities.enums import Algorithm, ConsensusParams, Objectives, SolutionMode
from src.utilities.paths import DEBUG_INSTANCES_FOLDER

DEBUG_INSTANCE_PATH = os.path.join(DEBUG_INSTANCES_FOLDER, "Debug_1")
FIXTURE_PATH = os.path.join(DEBUG_INSTANCES_FOLDER, "expected_results.json")

# Debug_1's requests are generated on a 180s pickup window (due_time - ready_time == 180 for every
# request) -- match it here so every check reflects the instance's own intended difficulty.
TIME_WINDOW = 180

_nb_requests, _generation_sim_time = derive_consensus_params(DEBUG_INSTANCE_PATH)

# Fixture-compared algorithms: mip_solver/greedy/consensus, each under total_profit, plus
# mip_solver/greedy under total_customers -- that objective builds a different MIP expression,
# a different greedy score and a different calc_objective_value branch, and it is the one TP1's
# exercises target, so profit-only coverage would leave it untested. dqn is structural-only (see
# reference_test.py's test_dqn_structural) -- an untrained network has no stable "expected" output.
CHECKS = {
    "mip_solver": lambda: SimulationConfig(
        objective=Objectives.TOTAL_PROFIT, algorithm=Algorithm.MIP_SOLVER,
        solution_mode=SolutionMode.OFFLINE, time_window=TIME_WINDOW,
    ),
    "mip_solver_customers": lambda: SimulationConfig(
        objective=Objectives.TOTAL_CUSTOMERS, algorithm=Algorithm.MIP_SOLVER,
        solution_mode=SolutionMode.OFFLINE, time_window=TIME_WINDOW,
    ),
    "greedy": lambda: SimulationConfig(
        objective=Objectives.TOTAL_PROFIT, algorithm=Algorithm.GREEDY,
        solution_mode=SolutionMode.FULLY_ONLINE, time_window=TIME_WINDOW,
    ),
    "greedy_customers": lambda: SimulationConfig(
        objective=Objectives.TOTAL_CUSTOMERS, algorithm=Algorithm.GREEDY,
        solution_mode=SolutionMode.FULLY_ONLINE, time_window=TIME_WINDOW,
    ),
    "consensus": lambda: SimulationConfig(
        objective=Objectives.TOTAL_PROFIT, algorithm=Algorithm.CONSENSUS,
        solution_mode=SolutionMode.FULLY_ONLINE, time_window=TIME_WINDOW,
        consensus=ConsensusConfig(
            nb_scenario=5, nb_requests=_nb_requests, generation_sim_time=_generation_sim_time,
            consensus_param=ConsensusParams.QUALITATIVE, seed=0,
        ),
    ),
    "dqn": lambda: SimulationConfig(
        objective=Objectives.TOTAL_PROFIT, algorithm=Algorithm.DQN,
        solution_mode=SolutionMode.FULLY_ONLINE, time_window=TIME_WINDOW,
        dqn=DQNConfig(model_path="", training_mode=False, epsilon=0.0),
    ),
}
