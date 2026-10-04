import json
import logging
import os
from typing import Any, Dict

from src.simulation.data_reader import TaxiDataReader
from src.utilities.config import SimulationConfig, ConsensusConfig, ReOptimizeConfig, DQNConfig
from src.utilities.config_tools import match_enum, resolve_release_params, derive_consensus_params
from src.utilities.enums import Algorithm, Objectives, SolutionMode, ConsensusParams, DestroyMethod
from src.utilities.paths import REAL_INSTANCES_FOLDER

DEFAULT_NB_SCENARIO = 10
DEFAULT_DQN_MODEL_PATH = os.path.join("data", "Models", "dqn_checkpoint.pt")

VIEWER_FACTORY_TEMPLATE = "src/execution/viewer_factory_template.json"
VIEWER_INSTANCES_FOLDER = "data/Viewer_Instances"

OBJECTIVE_ABBREVIATIONS = {
    "total_profit": "tp",
    "total_revenue": "tr",
    "total_cost": "tc",
    "total_customers": "nc",
    "waiting_time": "wt",
    "total_empty_travel_time": "et",
    "multi_objective": "mo",
}


def run_viewer_simulation(config_data: Dict[str, Any]) -> None:
    """Generate a p-multimodal-viewer simulation-factory.json from an inputs.json single_test
    block and run it by calling multimodalsim_viewer's own run_simulation() directly in-process,
    so a test can be driven entirely from the terminal (config_data/inputs.json) instead of
    picking instance/algorithm in the web UI -- and without needing any change to the viewer
    package's own source (the factory template it needs is registered from here instead)."""
    instance = config_data["instances"]
    instance_path = os.path.abspath(os.path.join(REAL_INSTANCES_FOLDER, instance))
    graph_path = os.path.abspath(os.path.join(REAL_INSTANCES_FOLDER, "graph.json"))
    polylines_path = os.path.abspath(os.path.join(REAL_INSTANCES_FOLDER, "polylines.json"))

    objective_enum = match_enum(config_data["objectives"], Objectives)
    algorithm_enum = match_enum(config_data["algorithms"], Algorithm)
    solution_mode_enum = match_enum(config_data["solution_mode"], SolutionMode)
    known_portion, advance_notice = resolve_release_params(solution_mode_enum, config_data.get("known_portion"))
    time_window = config_data["time_windows"]
    weight = config_data.get("weight", 1)
    deep_copy = config_data.get("deep_copy", False)
    # When true, the visualization processor connects to a running viewer server (see `viewer start`)
    # and streams the simulation as it runs, instead of only writing it to disk for later replay.
    live = config_data.get("live", False)

    taxi_config = {
        "objective": {"value": objective_enum.value},
        "algorithm": {"value": algorithm_enum.value},
        "solution_mode": {"value": solution_mode_enum.value},
        "known_portion": known_portion,
        "advance_notice": advance_notice,
        "time_window": time_window,
        "weight": weight,
    }

    if algorithm_enum == Algorithm.CONSENSUS:
        consensus_param_enum = match_enum(config_data["consensus_params"], ConsensusParams)
        # Derived once here from the instance's own requests.csv, reused below for the real
        # ConsensusConfig object too, so the JSON factory config and the object never drift apart.
        consensus_nb_requests, consensus_generation_sim_time = derive_consensus_params(instance_path)
        taxi_config["consensus"] = {
            "nb_scenario": config_data.get("nb_scenario", DEFAULT_NB_SCENARIO),
            "nb_requests": consensus_nb_requests,
            "generation_sim_time": consensus_generation_sim_time,
            "consensus_param": {"value": consensus_param_enum.value},
        }

    if algorithm_enum == Algorithm.RE_OPTIMIZE:
        dest_method_enum = match_enum(config_data["destroy_method"], DestroyMethod)
        taxi_config["re_optimize"] = {
            "destroy_method": {"value": dest_method_enum.value},
        }

    if algorithm_enum == Algorithm.DQN:
        taxi_config["dqn"] = {
            "model_path": config_data.get("model_path", DEFAULT_DQN_MODEL_PATH),
            "training_mode": config_data.get("training_mode", False),
            "epsilon": config_data.get("epsilon", 0.0),
            "seed": config_data.get("seed"),
        }

    # Compute vehicles_end_time from the instance's own trips, mirroring run_simulation.py's fix for
    # the same issue: without an explicit value here, TaxiDataReader falls back to the factory
    # template's hardcoded `default: 10000` (viewer_factory_template.json), which silently completes
    # every vehicle before online dispatch (fully_online/advance_notice/partial_online) ever gets a
    # look in, for any instance whose absolute time scale exceeds 10000 (e.g. a 7am-start real-trip-
    # pool instance -- exactly what Instance_1-20 use).
    simulation_config_kwargs = dict(
        objective=objective_enum, algorithm=algorithm_enum, solution_mode=solution_mode_enum,
        known_portion=known_portion, advance_notice=advance_notice, time_window=time_window,
        weight=weight,
    )
    if algorithm_enum == Algorithm.CONSENSUS:
        simulation_config_kwargs["consensus"] = ConsensusConfig(
            nb_scenario=config_data.get("nb_scenario", DEFAULT_NB_SCENARIO),
            nb_requests=consensus_nb_requests,
            generation_sim_time=consensus_generation_sim_time,
            consensus_param=match_enum(config_data["consensus_params"], ConsensusParams),
        )
    if algorithm_enum == Algorithm.RE_OPTIMIZE:
        simulation_config_kwargs["re_optimize"] = ReOptimizeConfig(
            destroy_method=match_enum(config_data["destroy_method"], DestroyMethod))
    if algorithm_enum == Algorithm.DQN:
        simulation_config_kwargs["dqn"] = DQNConfig(
            model_path=config_data.get("model_path", DEFAULT_DQN_MODEL_PATH),
            training_mode=config_data.get("training_mode", False),
            epsilon=config_data.get("epsilon", 0.0),
            seed=config_data.get("seed"),
        )
    simulation_config = SimulationConfig(**simulation_config_kwargs)

    end_time_reader = TaxiDataReader(
        requests_file_path=os.path.join(instance_path, "requests.csv"),
        vehicles_file_path=os.path.join(instance_path, "vehicles.csv"),
    )
    end_time_reader.load_graph(graph_path)
    trips_for_end_time = end_time_reader.get_trips(simulation_config)
    vehicles_end_time = max(
        (trip.latest_pickup + (trip.shortest_travel_time or 0) for trip in trips_for_end_time),
        default=100000,
    )

    data_reader_value = {
        "vehicles_file": os.path.join(instance_path, "vehicles.csv"),
        "graph_file": graph_path,
        # vehicles_end_time is a primitive parameter on the inner TaxiDataReader object template
        # (viewer_factory_template.json), not a sibling field at this level -- it must be supplied
        # inside this nested "data_reader" value, or it silently falls back to the template's
        # hardcoded `default: 10000`.
        "data_reader": {"vehicles_end_time": vehicles_end_time},
    }

    factory_config = {
        "version": 3,
        "merge": "override",
        "root": {
            "default": {
                "taxi_config": taxi_config,
                "data_reader": {
                    "requests_file": os.path.join(instance_path, "requests.csv"),
                    "data_reader": {
                        "select": {"key": "taxi_data_reader"},
                        "value": data_reader_value,
                    },
                },
                "optimization": {
                    "freeze_interval": 10,
                    "optimization": {
                        "select": {"key": "optimization"},
                        "value": {
                            "config": {
                                "select": {
                                    "key": "optimization_config" if deep_copy else "no_deep_copy_optimization_config"
                                },
                                "value": {},
                            },
                            "dispatcher": {"select": {"key": "taxi_dispatcher"}, "value": {}},
                            "splitter": {"select": {"key": "one_leg_splitter"}, "value": {}},
                        },
                    },
                },
                "data_collection": {
                    "data_container": {"select": {"key": "standard_data_container"}, "value": {}},
                    "data_collectors": {"standard_data_collector": True},
                    "statistics_delta_time": 60,
                },
                "tag_processors": {},
                "visualization_processor": {"offline": not live},
            }
        },
    }

    # Only wire up road-following coordinates if Real_Instances actually ships a polylines.json --
    # the frontend falls back to straight lines gracefully when no coordinates provider is set.
    if os.path.isfile(polylines_path):
        factory_config["root"]["default"]["coordinates"] = {
            "select": {"key": "coordinates_from_polyline_file"},
            "value": {"file_path": polylines_path},
        }

    obj_abbr = OBJECTIVE_ABBREVIATIONS.get(objective_enum.value, objective_enum.value[:3])
    name = f"{instance}_{time_window}_{obj_abbr}_{algorithm_enum.value}"

    viewer_instance_dir = os.path.join(VIEWER_INSTANCES_FOLDER, name)
    os.makedirs(viewer_instance_dir, exist_ok=True)
    config_path = os.path.join(viewer_instance_dir, "simulation-factory.json")
    with open(config_path, "w") as config_file:
        json.dump(factory_config, config_file, indent=2)

    logging.info(f"Generated viewer instance config at {config_path}")

    # Imported lazily (rather than at module load time): only needed for this task type, and
    # importing multimodalsim_viewer triggers its own __init__.py registration + .env loading.
    import multimodalsim.factory.factory as factory_module
    import multimodalsim.factory.factory_reader as factory_reader_module
    import multimodalsim_viewer.common.utils as viewer_utils_module
    import multimodalsim_viewer.factory.factory as viewer_factory_module
    import multimodalsim_viewer.server.data_manager as data_manager_module
    import multimodalsim_viewer.viewer_configuration.viewer_configuration as viewer_configuration_module

    factory_reader_module.register_intermediate_factory_path(os.path.abspath(VIEWER_FACTORY_TEMPLATE))

    simulation_id = viewer_utils_module.build_simulation_id(name)
    logging.info(f"Running viewer simulation {simulation_id} (data={name})")

    # Mirrors multimodalsim_viewer.server.simulation._initialize_simulation()/run_simulation() --
    # replicated here (all public calls, no package source changes) instead of calling
    # run_simulation() directly, so that after sim.simulate() finishes we can do our own
    # polylines persistence pass below. p-multimodal-viewer's own process_polylines() (triggered
    # incrementally during the simulation) has a timing/caching gap that leaves polylines.jsonl
    # empty; doing this once, after the simulation is fully done and every route's stops are
    # final, sidesteps that gap entirely.
    data_manager_module.get_saved_simulation_directory_path(simulation_id, should_create=True)
    viewer_configuration_module.copy_viewer_configuration_from_instance(simulation_id=simulation_id, instance=name)
    viewer_factory_module.save_factory_value(simulation_id, None)

    def after_read_callback(factory_template):
        viewer_factory_module.save_factory_template(simulation_id, factory_template)

    factory_result = factory_module.read_and_build_factory(
        instance_folder_path=data_manager_module.get_input_data_directory_path(name) + "/",
        factory_value=None,
        inputs={"simulation_id": simulation_id, "max_duration": None, "input_data_description": name},
        after_read_callback=after_read_callback,
    )
    sim = factory_result["simulation"]
    sim.simulate()

    _persist_real_polylines(sim, simulation_id, data_manager_module)


def _persist_real_polylines(sim, simulation_id: str, data_manager_module) -> None:
    """Extract each vehicle's final stop-to-stop polylines directly from the completed
    simulation and persist them via data_manager.set_polylines() -- the same public function
    p-multimodal-viewer's own (buggy) process_polylines() would otherwise call. Coordinates keys
    are built as "{lat1},{lon1},{lat2},{lon2}", matching StopIdentity.coordinates_string exactly,
    since that's how the frontend looks up polylines for a given stop pair."""
    env = sim._Simulation__env
    if env.coordinates is None:
        return

    saved_pairs = set()
    polylines_to_save = {}

    for route in env.route_by_vehicle_id.values():
        all_stops = route.previous_stops + ([route.current_stop] if route.current_stop is not None else []) + route.next_stops
        if len(all_stops) < 2:
            continue

        polylines = env.coordinates.update_polylines(route)

        for index in range(len(all_stops) - 1):
            first_stop, second_stop = all_stops[index], all_stops[index + 1]
            first_location, second_location = first_stop.location, second_stop.location
            if first_location.lat is None or second_location.lat is None:
                continue

            pair_key = f"{first_location.lat},{first_location.lon},{second_location.lat},{second_location.lon}"
            if pair_key in saved_pairs:
                continue
            saved_pairs.add(pair_key)

            polyline_entry = polylines.get(str(index))
            if polyline_entry is not None:
                polylines_to_save[pair_key] = polyline_entry

    if polylines_to_save:
        data_manager_module.set_polylines(simulation_id, polylines_to_save)
        logging.info(f"Persisted {len(polylines_to_save)} real polylines for simulation {simulation_id}")
