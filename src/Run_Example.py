import os
import logging
import argparse
from src.utilities.reporting_tools import print_dict_as_table, print_result_as_table
from src.utilities.config_tools import match_enum, derive_consensus_params, resolve_release_params
from src.utilities.enums import (Algorithm, Objectives, DestroyMethod, ConsensusParams, SolutionMode)
from src.simulation.run_simulation import run_taxi_simulation
from src.utilities.config import SimulationConfig, ConsensusConfig, ReOptimizeConfig
from src.utilities.paths import REAL_INSTANCES_FOLDER


def run_example(test_folder: str, config: SimulationConfig, deep_copy: bool = False) -> None:
    """
    Runs a taxi dispatch simulation with the specified parameters.

    Parameters:
    ----------------------
    test_folder: str
        folder of the instance to test, under data/Real_Instances

    config: SimulationConfig
        Configuration object containing all simulation parameters.

    deep_copy: bool
        Whether multimodalsim should deep-copy the environment before each optimization step
        (state_deep_copy). False (the default) is faster; True is the safer/slower legacy mode.
    """
    logging.getLogger().setLevel(logging.WARN)  # INFO

    # Define base paths
    test_path = os.path.join(REAL_INSTANCES_FOLDER, test_folder)

    # Run the simulation
    info_dict, output_dict = run_taxi_simulation(
        test_path,
        config=config,
        deep_copy=deep_copy,
        require_competitive_ratio=True,
    )

    # Combine and print the results
    result = {**info_dict, **output_dict}
    print_result_as_table(result)


def parse_arguments() -> tuple[str, SimulationConfig]:
    """
    Parses command-line arguments and returns test_folder, a SimulationConfig object, and the
    deep_copy flag.

    Returns:
        test_folder: The directory path (under data/Real_Instances) where the test files are located.
        SimulationConfig: Configured simulation parameters.
        deep_copy: Whether multimodalsim should deep-copy the environment before each
            optimization step (state_deep_copy).
    """
    parser = argparse.ArgumentParser(
        description='Run a taxi simulation',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-i", "--instance", type=str, default="Instance_1",
                        help="Folder of the instance to test, under data/Real_Instances. Default: Instance_1")

    parser.add_argument("-o", "--objective", type=str, default="total_customers",
                        help=("The optimization objective to achieve:\n"
                              "- total_profit: Maximizes the total profit of served requests.\n"
                              "- waiting_time: Minimizes the total wait time of served requests.\n"
                              "- total_customers: Maximizes the total number of served customers.\n"
                              "- total_revenue: Maximizes the total fare of served requests.\n"
                              "- total_cost: Minimizes the total driving cost of served requests.\n"
                              "- total_empty_travel_time: Minimizes the total empty-leg travel time.\n"
                              "- multi_objective: Weighted combination of profit and wait time (use -wp for weight).\n"
                              "Default: total_customers"))

    parser.add_argument("-a", "--algorithm", type=str, default="mip_solver",
                        help=("Algorithm used to optimize the dispatch plan:\n"
                              "- mip_solver: Uses the Gurobi MIP solver to solve the problem.\n"
                              "- greedy: Greedy approach to assign requests to vehicles.\n"
                              "- random: Random approach to assign requests to vehicles.\n"
                              "- ranking: Ranking approach to assign requests to vehicles.\n"
                              "- consensus: Consensus approach to assign requests to vehicles.\n"
                              "- re_optimize: Re-optimize the solution based on destroy and repair.\n"
                              "Default: mip_solver"))
    parser.add_argument("-m", "--sol-mode", type=str, default="offline",
                        help="The mode of solution:\n"
                             "- offline : all the requests revealed at the start (release time = 0 for all requests).\n"
                             "- fully_online : release time is equal to the ready time for all requests.\n"
                             "- advance_notice : requests are known 30 minutes before the ready time.\n"
                             "- partial_online : a portion of requests are known in adavnce.\n"
                             "- custom_scenario : a mix of advance_notice and partial_online.\n"
                             "Default: offline")

    parser.add_argument("-kp", "--known-portion", type=float, default=100,
                        help="Portion of requests that are known in advance (0-100%%). Default: 100")

    parser.add_argument("-an", "--advance-notice", type=int, default=0,
                        help="Fixed amount of time (seconds) the requests are released before their ready time. Default: 0")

    parser.add_argument("-tw", "--time-window", type=int, default=180,
                        help="Size of the time window in seconds to serve a request. Default: 180")

    parser.add_argument("-ns", "--nb-scenario", type=int, default=5,
                        help="Total number of scenarios to be solved for consensus. Default: 20")

    parser.add_argument("-cr", "--cust-rate", type=float, default=0.35,
                        help=("The average rate of customers per node per hour:\n"
                              "- 0.2 for small-size tests.\n"
                              "- 0.3 for medium-size tests.\n"
                              "- 0.7 for large-size tests.\n"
                              "Default: 0.35"))

    parser.add_argument("-cp", "--consensus-params", type=str, default="quantitative",
                        help=("Type of consensus algorithm:\n"
                              "- qualitative: Increment a counter for the best request in each scenario.\n"
                              "- quantitative: Credit the best request with the optimal solution value.\n"
                              "Default: quantitative"))

    parser.add_argument("-dm", "--dest-method", type=str, default="default",
                        help=("Method used for destruction in re-optimizing:\n"
                              "- default: Default destruction method (Complete re-optimization)\n"
                              "- fix_variables: fix some of the variables in the model\n"
                              "- fix_arrivals: Fix a time window around the arrival time\n"
                              "- bonus: arbitrary destroy method as bonus\n"
                              "Default: default"))

    parser.add_argument("-wp", "--weight", type=float, default=1,
                        help=("Weight w in [0, 1] for profit in multi-objective. Default: 0.5"))

    parser.add_argument("-dc", "--deep-copy", action="store_true",
                        help="Have multimodalsim deep-copy the environment before each optimization "
                             "step (slower, legacy mode). Default: off (faster).")

    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Seed for the consensus algorithm's own scenario-sampling randomness. "
                             "Default: unseeded.")

    args = parser.parse_args()

    # Map string inputs to enums
    try:
        objective_enum = match_enum(args.objective, Objectives)
        algorithm_enum = match_enum(args.algorithm, Algorithm)
        dest_method_enum = match_enum(args.dest_method, DestroyMethod)
        consensus_param_enum = match_enum(args.consensus_params, ConsensusParams)
        solution_mode_enum = match_enum(args.sol_mode, SolutionMode)
        known_portion, advance_notice = resolve_release_params(solution_mode_enum, args.known_portion)
    except ValueError as e:
        parser.error(str(e))

    # Build the algorithm-specific config (if any) before constructing SimulationConfig, since
    # SimulationConfig validates that the right one is present for the chosen algorithm.
    consensus_config = None
    if algorithm_enum == Algorithm.CONSENSUS:
        test_path = os.path.join(REAL_INSTANCES_FOLDER, args.instance)
        nb_requests, generation_sim_time = derive_consensus_params(test_path)
        consensus_config = ConsensusConfig(
            nb_scenario=args.nb_scenario,
            nb_requests=nb_requests,
            generation_sim_time=generation_sim_time,
            consensus_param=consensus_param_enum,
            seed=args.seed,
        )

    re_optimize_config = None
    if algorithm_enum == Algorithm.RE_OPTIMIZE:
        re_optimize_config = ReOptimizeConfig(destroy_method=dest_method_enum)

    # Create the SimulationConfig object
    config = SimulationConfig(
        objective=objective_enum,
        algorithm=algorithm_enum,
        known_portion=known_portion,
        advance_notice=advance_notice,
        time_window=args.time_window,
        solution_mode=solution_mode_enum,
        weight=args.weight,
        consensus=consensus_config,
        re_optimize=re_optimize_config,
    )

    return args.instance, config, args.deep_copy


def main():
    """
    Main function to run the taxi simulation based on command-line arguments.
    """
    test_folder, config, deep_copy = parse_arguments()
    run_example(test_folder, config, deep_copy)

if __name__ == '__main__':
    main()