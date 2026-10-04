import argparse
import os
import logging
import json
import sys
from pathlib import Path

from src.reporting.create_plots import handle_create_plot
from src.execution.run_tests import run_single_test, run_scenarios
from src.execution.run_viewer import run_viewer_simulation
from src.generation.create_instance import create_instances

# Instructor-only convenience: this repo pairs the public `main/` checkout with a private
# `src_solution/` reference-answers folder as a sibling of `main/` at the repo root. When present,
# prefer it over the public stub (mirrors the same pattern in simulation/taxi_dispatcher.py).
# Students never have `src_solution/`, so this always falls through to the stub for them.
_SRC_SOLUTION_DIR = Path(__file__).resolve().parents[2] / "src_solution"
if _SRC_SOLUTION_DIR.exists():
    sys.path.insert(0, str(_SRC_SOLUTION_DIR.parent))
try:
    from src_solution.train_dqn import train_from_config
except ImportError:
    from src.execution.train_dqn import train_from_config

logging.getLogger().setLevel(logging.WARN)

def main():
    """Main entry point of the program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run taxi simulation scenarios.")
    parser.add_argument(
        "-sn", "--scenario",
        type=str,
        help="Scenario to run (e.g., TP1, TP2, TP3, TP4_scenario, TP4).",
    )
    args = parser.parse_args()
    config_file = "src/execution/inputs.json"  # Replace with your JSON file path
    # Load configuration file
    if not os.path.isfile(config_file):
        logging.error(f"Configuration file '{config_file}' does not exist.")
        return

    with open(config_file, 'r') as file:
        config_data = json.load(file)

    task_type = config_data.get("task_type")

    if task_type == "scenarios":
        SCENARIOS = {scenario["scenario"]: scenario["parameters"] for scenario in config_data.get("scenarios", [])}
        run_scenarios(args.scenario, SCENARIOS, config_data.get("scenario_groups", {}))
    elif task_type == "single_test":
        run_single_test(config_data["single_test"])
    elif task_type == "viewer_simulate":
        run_viewer_simulation(config_data["single_test"])
    elif task_type == "create_plot":
        handle_create_plot(config_data["create_plot"], args.scenario)
    elif task_type == "create_instance":
        create_instances(config_data["create_instance"])
    elif task_type == "train_dqn":
        train_from_config(config_data["train_dqn"])
    else:
        logging.error(
            f"Invalid task type '{task_type}'. Must be one of 'single_test', 'scenarios', "
            "'viewer_simulate', 'create_plot', 'create_instance', 'train_dqn'.")


if __name__ == "__main__":
    main()
