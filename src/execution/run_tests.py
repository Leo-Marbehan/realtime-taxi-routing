import itertools
import os
import logging
from typing import Any, Dict, List
import pandas as pd
from src.utilities.enums import (Algorithm,Objectives,SolutionMode,DestroyMethod,ConsensusParams)
from src.simulation.run_simulation import run_taxi_simulation, CompetitiveRatioBaselineError
from src.utilities.config import SimulationConfig, ConsensusConfig, ReOptimizeConfig, DQNConfig
from src.utilities.reporting_tools import print_dict_as_table, print_result_as_table
from src.utilities.config_tools import derive_consensus_params, match_enum, resolve_release_params
from src.utilities.paths import REAL_INSTANCES_FOLDER, RESULTS_FOLDER
from src.utilities import result_columns

DEFAULT_NB_SCENARIO = 10
DEFAULT_DQN_MODEL_PATH = os.path.join("data", "Models", "dqn_checkpoint.pt")

def run_single_test(config_data: Dict[str, Any]):
    """Runs a single test based on configuration."""
    test_folder = config_data["instances"]
    config = create_simulation_config(config_data)

    logging.info(f"Running single test for instance '{test_folder}'.")

    test_path = os.path.join(REAL_INSTANCES_FOLDER, test_folder)
    info_dict, output_dict = run_taxi_simulation(
        test_folder=test_path,
        config=config,
        deep_copy=config_data.get("deep_copy", False),
        require_competitive_ratio=True,
    )

    combined_result = {**info_dict, **output_dict}
    print_result_as_table(combined_result)


def run_scenarios(part: str, SCENARIOS: Dict[str, Dict[str, Any]],
                   scenario_groups: Dict[str, List[str]] = None):
    """
    Runs scenarios based on the provided part.
    If part is a group (e.g., TP2), it runs all associated sub-scenarios.

    scenario_groups: mapping of group name -> list of sub-scenario names (e.g.
        {"TP2": ["TP2_1", "TP2_2"]}), read from inputs.json's "scenario_groups" key -- see main.py.
    """
    scenario_groups = scenario_groups or {}
    if part in scenario_groups:
        # Handle grouped scenarios (e.g., TP4, TP2)
        group_scenarios = scenario_groups[part]
        all_combinations = []
        for group_scenario in group_scenarios:
            scenario = SCENARIOS.get(group_scenario, {})
            if scenario:
                all_combinations.extend(generate_combinations(scenario))
    elif part in SCENARIOS:
        # Single scenario
        scenario = SCENARIOS[part]
        all_combinations = generate_combinations(scenario)
    else:
        logging.error(f"Scenario part '{part}' is not defined.")
        available = list(SCENARIOS.keys()) + list(scenario_groups.keys())
        print(f"Invalid scenario part. Please choose from: {', '.join(available)}")
        return

    logging.info(f"Running scenarios for {part}")
    logging.info(f"Total combinations to run for {part}: {len(all_combinations)}")

    results = []
    for idx, comb in enumerate(all_combinations, 1):
        logging.info(f"Running combination {idx}/{len(all_combinations)}")
        instance = comb["instances"]
        test_path = os.path.join(REAL_INSTANCES_FOLDER, instance)

        if not os.path.isdir(test_path):
            logging.warning(f"Instance folder '{test_path}' does not exist. Skipping.")
            continue

        config = create_simulation_config(comb)
        try:
            info_dict, output_dict = run_taxi_simulation(
                test_folder=test_path,
                config=config,
                deep_copy=comb.get("deep_copy", False),
                require_competitive_ratio=True,
            )

            combined_result = {**info_dict, **output_dict}
            results.append(combined_result)
            print_result_as_table(combined_result)

        except CompetitiveRatioBaselineError:
            # A missing/stale TP1 baseline is a setup problem affecting the whole batch, not a
            # one-off simulation failure -- abort immediately with one clear message instead of
            # repeating/burying it across every remaining combination.
            raise
        except Exception as e:
            logging.error(f"Error running simulation for {part} - Instance: {instance} - Error: {e}")
            continue

    # Save results to CSV
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    df = pd.DataFrame(results)
    csv_file_path = os.path.join(RESULTS_FOLDER, f"{part}_simulation_results.csv")
    df.to_csv(csv_file_path, index=False)
    logging.info(f"Results for '{part}' saved to {csv_file_path}.")

    # Print results
    cols_to_print = result_columns.DISPLAY_COLUMNS
    with pd.option_context('display.colheader_justify', 'center'):
        # to_markdown will print headers on top, values below
        print(df[cols_to_print].to_markdown(tablefmt="pipe", index=False))

def generate_combinations(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates all parameter combinations for scenarios."""
    keys = list(params.keys())
    values = list(params.values())

    if "algorithm_params" in params and params["algorithm_params"]:
        algorithm_param_keys = list(params["algorithm_params"].keys())
        algorithm_param_values = list(params["algorithm_params"].values())

        algo_param_combinations = list(itertools.product(*algorithm_param_values))
        keys_no_algo = [k for k in keys if k != "algorithm_params"]
        values_no_algo = [params[k] for k in keys_no_algo]
        base_combinations = list(itertools.product(*values_no_algo))

        all_combinations = []
        for base_comb in base_combinations:
            base_dict = dict(zip(keys_no_algo, base_comb))
            for algo_comb in algo_param_combinations:
                algo_dict = dict(zip(algorithm_param_keys, algo_comb))
                all_combinations.append({**base_dict, **algo_dict})
    else:
        # Simple case: generate all combinations without nested algorithm_params
        all_combinations = [
            dict(zip(keys, combination))
            for combination in itertools.product(*values)
        ]
    return all_combinations


def create_simulation_config(comb: Dict[str, Any]) -> SimulationConfig:
    """Creates a SimulationConfig object from a parameter combination."""
    # Map string inputs to enums
    try:
        objective_enum = match_enum(comb["objectives"], Objectives)
        algorithm_enum = match_enum(comb["algorithms"], Algorithm)
        solution_mode_enum = match_enum(comb["solution_mode"], SolutionMode)
        known_portion, advance_notice = resolve_release_params(solution_mode_enum, comb.get("known_portion"))
    except ValueError as e:
        logging.error(f"Error creating SimulationConfig object - {e}")
        raise

    # Build the algorithm-specific config (if any) before constructing SimulationConfig, since
    # SimulationConfig validates that the right one is present for the chosen algorithm.
    consensus_config = None
    if algorithm_enum == Algorithm.CONSENSUS:
        consensus_param_enum = match_enum(comb["consensus_params"], ConsensusParams)
        test_path = os.path.join(REAL_INSTANCES_FOLDER, comb["instances"])
        nb_requests, generation_sim_time = derive_consensus_params(test_path)
        consensus_config = ConsensusConfig(
            nb_scenario=comb.get("nb_scenario", DEFAULT_NB_SCENARIO),
            consensus_param=consensus_param_enum,
            nb_requests=nb_requests,
            generation_sim_time=generation_sim_time,
            seed=comb.get("seed"),
        )

    re_optimize_config = None
    if algorithm_enum == Algorithm.RE_OPTIMIZE:
        dest_method_enum = match_enum(comb["destroy_method"], DestroyMethod)
        re_optimize_config = ReOptimizeConfig(destroy_method=dest_method_enum)

    dqn_config = None
    if algorithm_enum == Algorithm.DQN:
        # Evaluation defaults: greedy argmax (epsilon=0), no training-mode side effects, loading
        # whatever checkpoint execution/train_dqn.py's driver last saved.
        dqn_config = DQNConfig(
            model_path=comb.get("model_path", DEFAULT_DQN_MODEL_PATH),
            training_mode=comb.get("training_mode", False),
            epsilon=comb.get("epsilon", 0.0),
            seed=comb.get("seed"),
        )

    config = SimulationConfig(
        objective=objective_enum,
        algorithm=algorithm_enum,
        known_portion=known_portion,
        advance_notice=advance_notice,
        time_window=comb["time_windows"],
        solution_mode=solution_mode_enum,
        # Weight for multi-objective (profit vs wait time); used when objective is MULTI_OBJECTIVE
        weight=comb.get("weight", 1),
        consensus=consensus_config,
        re_optimize=re_optimize_config,
        dqn=dqn_config,
    )

    return config

