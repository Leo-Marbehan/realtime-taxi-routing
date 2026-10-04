import logging
import os
from typing import Any, Callable, Dict, List

from src.reporting.visualization import (offline_plot, compare_algorithm_plot, compare_timeWindow_plot,
                                         number_scenarios, multi_plot)
from src.utilities.paths import RESULTS_FOLDER

# Constants
RESULTS_FILE_PATTERN = "{scenario}_simulation_results.csv"

# Plot name constants
PLOT_OFFLINE = "offline_plot"
PLOT_COMPARE_ALGORITHM = "compare_algorithm_plot"
PLOT_COMPARE_TIMEWINDOW = "compare_timeWindow_plot"
PLOT_NUMBER_SCENARIOS = "number_scenarios"
PLOT_MULTI = "multi_plot"

# Mapping of plot names to their corresponding functions
PLOT_FUNCTIONS: Dict[str, Callable[[str, List[str]], None]] = {
    PLOT_OFFLINE: offline_plot,
    PLOT_COMPARE_ALGORITHM: compare_algorithm_plot,
    PLOT_COMPARE_TIMEWINDOW: compare_timeWindow_plot,
    PLOT_NUMBER_SCENARIOS: number_scenarios,
    PLOT_MULTI: multi_plot,
}


def handle_create_plot(config_data: List[Dict[str, Any]], scenario: str) -> None:
    """
    Handle the create_plot task based on plot_name.
    
    Parameters:
    -----------
    config_data : List[Dict[str, Any]]
        List of scenario configurations containing plot definitions.
    scenario : str
        Name of the scenario to generate plots for.
    """
    # Find the configuration for the specified scenario
    scenario_entry = next((entry for entry in config_data if entry.get("scenario") == scenario), None)
    if not scenario_entry:
        logging.error(f"Scenario '{scenario}' not found in config data.")
        return

    plot_entries = scenario_entry.get("plots", [])
    file_path = os.path.join(RESULTS_FOLDER, RESULTS_FILE_PATTERN.format(scenario=scenario))

    if not os.path.isfile(file_path):
        logging.error(f"Result file '{file_path}' does not exist.")
        return

    # Iterate over each plot configuration for the scenario
    for plot in plot_entries:
        plot_name = plot.get("plot_name")
        metrics = plot.get("metrics", [])

        if not plot_name:
            logging.warning(f"Plot entry without a plot_name found in scenario '{scenario}'. Skipping.")
            continue

        plot_function = PLOT_FUNCTIONS.get(plot_name)
        if plot_function:
            plot_function(file_path, metrics)
        else:
            logging.error(f"Unknown plot name '{plot_name}' in scenario '{scenario}'. "
                         f"Available plots: {', '.join(PLOT_FUNCTIONS.keys())}")