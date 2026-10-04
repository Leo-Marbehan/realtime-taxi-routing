import os
import pandas as pd
from multimodalsim.simulator.simulation import Simulation
from multimodalsim.optimization.optimization import Optimization
from multimodalsim.config.optimization_config import OptimizationConfig, NoDeepCopyOptimizationConfig

from src.simulation.data_reader import TaxiDataReader
from src.simulation.taxi_dispatcher import TaxiDispatcher
from src.utilities.enums import Algorithm, SolutionMode
from src.utilities.config import SimulationConfig
from src.utilities.config_tools import compute_instance_fingerprint
from src.utilities import result_columns as col


class CompetitiveRatioBaselineError(Exception):
    """Raised (only when require_competitive_ratio=True) when the TP1 offline baseline needed to
    compute Competitive Ratio is missing or can't be verified as matching the instance actually
    being run -- see run_taxi_simulation's competitive-ratio block.
    """


def run_taxi_simulation(
        test_folder: str,
        config: SimulationConfig,
        deep_copy: bool = False,
        return_dispatcher: bool = False,
        require_competitive_ratio: bool = False,
    ):

    """ Function: Conducts a simulation of taxi dispatching, based on specified parameters.
        Input:
        ------------
        test_folder : str
            The directory path where the test files are located. Its sibling `graph.json` (i.e.
            `os.path.dirname(test_folder)/graph.json`) is used as the network graph.

        config: SimulationConfig
            Configuration object containing all simulation parameters.

        deep_copy : bool
            Whether multimodalsim should deep-copy the environment before each optimization step
            (state_deep_copy). False (the default) is faster; True is the safer/slower legacy mode.
            See .claude/MULTIMODALSIM_MIGRATION_PLAN.md for the full background.

        return_dispatcher : bool
            If True, also return the TaxiDispatcher instance used for this run (e.g. so a training
            driver can reach dispatcher.solver -- a DQNSolver holds per-episode state like its
            replay buffer that isn't otherwise exposed). False (the default) keeps the original
            2-tuple return, so existing callers are unaffected.

        require_competitive_ratio : bool
            If True and config.solution_mode != OFFLINE, a missing or unverifiable TP1 baseline
            raises CompetitiveRatioBaselineError instead of silently omitting the 'Competitive
            Ratio' key. False (the default) keeps today's best-effort behavior: callers that don't
            care about this metric (e.g. train_dqn.py's training loop, or anything run on instances
            TP1 was never meant to cover) are unaffected. Only pass True from an entry point where
            a human is actually going to read/compare this result (run_tests.py's
            run_single_test/run_scenarios, Run_Example.py).

        Output:
        ------------
        info_dict : dict
            Summary of test inputs and simulation parameters.
        output_dict : dict
            Simulation output metrics and results.
        dispatcher : TaxiDispatcher, only when return_dispatcher is True
    """

    # Display simulation parameters
    print("==================================================")
    print("Run Taxi Simulation with the following settings:")
    print(f"  Instance: {test_folder}")
    print(f"  Objective: {config.objective.value}")
    print(f"  Algorithm: {config.algorithm.value}")
    print(f"  Solution Mode: {config.solution_mode.value}")
    print(f"  Percentage Known (%): {config.known_portion}")
    print(f"  Advance Notice (s): {config.advance_notice}")
    print(f"  Time Window (s): {config.time_window}")
    print(f"  Weight: {config.weight}")

    if config.algorithm == Algorithm.CONSENSUS:
        print(f'  Number of Scenarios: {config.consensus.nb_scenario}')
        print(f'  Requests instance was generated with: {config.consensus.nb_requests}')
        print(f'  Type of consensus algorithm:: {config.consensus.consensus_param.value}')

    if config.algorithm == Algorithm.RE_OPTIMIZE:
        print(f'  Re-optimizer Destroy Method: {config.re_optimize.destroy_method.value}')
    print("==================================================")

    # Run the simulation
    # Define file paths for requests, vehicles, and the network graph (a sibling of test_folder,
    # shared by every instance in the same instances folder)
    requests_file_path = f"{test_folder}/requests.csv"
    vehicles_file_path = f"{test_folder}/vehicles.csv"
    graph_file_path = os.path.join(os.path.dirname(test_folder), "graph.json")

    # Read and prepare data
    data_reader = TaxiDataReader(
        requests_file_path=requests_file_path,
        vehicles_file_path=vehicles_file_path,
    )
    network_graph = data_reader.load_graph(graph_file_path)
    #    ut.draw_network(network_graph, os.path.dirname(graph_file_path))

    # Trips read first so vehicles_end_time can be derived from the instance's own absolute time
    # range (requests may start well after t=0, e.g. a 7-9am real-trip-pool instance) instead of a
    # fixed constant that silently expires every vehicle before online dispatch ever gets a look in.
    trips = data_reader.get_trips(config)
    vehicles_end_time = max(
        (trip.latest_pickup + (trip.shortest_travel_time or 0) for trip in trips), default=100000)
    vehicles, routes_by_vehicle_id = data_reader.get_vehicles(vehicles_end_time=vehicles_end_time)

    # Initialize simulation components
    dispatcher = TaxiDispatcher(
        network=network_graph,
        vehicles=vehicles,
        simulation_config=config,
    )
    opt_config = OptimizationConfig() if deep_copy else NoDeepCopyOptimizationConfig()
    opt = Optimization(dispatcher, freeze_interval=10, config=opt_config)
    print(f"  multimodalsim state_deep_copy={opt.config.state_deep_copy}, asynchronous={opt.config.asynchronous}")

    # Initialize and run the simulation
    simulation = Simulation(
        optimization=opt,
        trips=trips,
        vehicles=vehicles,
        routes_by_vehicle_id=routes_by_vehicle_id,
        network=network_graph,
    )
    simulation.simulate()

    # Extract and process simulation output
    output_dict = dispatcher.extract_output()
    # Build a shorter, compact key: <instance>_<tw>_<obj_abbr>
    obj_type = str(output_dict[col.OBJECTIVE_TYPE])
    obj_abbr_map = {
        "total_profit": "tp",
        "total_revenue": "tr",
        "total_cost": "tc",
        "total_customers": "nc",
        "waiting_time": "wt",
        "total_empty_travel_time": "et",
        "multi_objective": "mo",
    }
    obj_abbr = obj_abbr_map.get(obj_type, obj_type[:3])
    unique_key = f"{os.path.basename(test_folder)}_{config.time_window}_{obj_abbr}"

    # Compile information about the test and results
    info_dict = {
        col.KEY: unique_key,
        col.TEST: os.path.basename(test_folder),
        col.NUM_TRIPS: len(trips),
        col.NUM_VEHICLES: len(vehicles),
        col.SOLUTION_MODE: config.solution_mode.value,
        col.TIME_WINDOW_S: config.time_window,
        # Content-based identity of the instance actually run, independent of the folder name --
        # lets a later run verify a saved baseline (e.g. TP1's) still matches what's on disk now,
        # rather than trusting the Key string alone (see compute_instance_fingerprint).
        col.INSTANCE_FINGERPRINT: compute_instance_fingerprint(test_folder),
    }
    if config.solution_mode == SolutionMode.PARTIAL_ONLINE:
        info_dict.update({col.KNOWN_PORTION_PCT: config.known_portion})

    if config.algorithm == Algorithm.CONSENSUS:
        info_dict.update({col.NUM_SCENARIOS: config.consensus.nb_scenario,
                          col.NB_REQUESTS: config.consensus.nb_requests,
                          col.CONSENSUS_TYPE: config.consensus.consensus_param.value})

    if config.algorithm == Algorithm.RE_OPTIMIZE:
        info_dict.update({col.DESTROY_METHOD: config.re_optimize.destroy_method.value})

    info_dict.update({col.WEIGHT: config.weight})

    if config.solution_mode != SolutionMode.OFFLINE:
        Results_folder = os.path.join(os.path.dirname(test_folder), "Results")
        Offline_Solution_path = os.path.join(Results_folder, "TP1_simulation_results.csv")

        if not os.path.exists(Offline_Solution_path):
            if require_competitive_ratio:
                raise CompetitiveRatioBaselineError(
                    f"Competitive Ratio requires TP1's offline results, but "
                    f"'{Offline_Solution_path}' does not exist. Run the TP1 scenario first "
                    f"(e.g. `python -m src.main -sn TP1`) to generate it.")
        else:
            offline_df = pd.read_csv(Offline_Solution_path, index_col=col.KEY)
            if unique_key not in offline_df.index:
                if require_competitive_ratio:
                    raise CompetitiveRatioBaselineError(
                        f"'{Offline_Solution_path}' exists but has no baseline for '{unique_key}' "
                        f"(instance={os.path.basename(test_folder)}, time_window={config.time_window}, "
                        f"objective={obj_type}). Run TP1 for this instance/time-window/objective "
                        f"combination before trusting Competitive Ratio here.")
            elif col.INSTANCE_FINGERPRINT not in offline_df.columns:
                if require_competitive_ratio:
                    raise CompetitiveRatioBaselineError(
                        f"'{Offline_Solution_path}' predates fingerprint verification, so it can't "
                        f"be confirmed to match the instance files on disk now. Re-run TP1 to "
                        f"regenerate it with a verifiable baseline.")
            elif offline_df.loc[unique_key, col.INSTANCE_FINGERPRINT] != info_dict[col.INSTANCE_FINGERPRINT]:
                if require_competitive_ratio:
                    raise CompetitiveRatioBaselineError(
                        f"TP1 baseline for '{unique_key}' is stale: it was computed on different "
                        f"instance file contents than what's on disk now for "
                        f"'{os.path.basename(test_folder)}'. Re-run TP1 for this instance before "
                        f"trusting Competitive Ratio here.")
            else:
                offline_value = offline_df.loc[unique_key, col.OBJECTIVE_VALUE]
                competitive_ratio = round(output_dict[col.OBJECTIVE_VALUE] / offline_value, 2) if\
                    offline_value != 0 else 0
                info_dict.update({col.COMPETITIVE_RATIO: competitive_ratio})



    if return_dispatcher:
        return info_dict, output_dict, dispatcher
    return info_dict, output_dict