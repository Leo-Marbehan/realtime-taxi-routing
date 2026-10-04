import csv
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from multimodalsim.simulator.vehicle import Vehicle
from multimodalsim.simulator.stop import LabelLocation, Stop

from src.utilities.create_scenario import create_requests_from_pool
from src.utilities.network_tools import load_network_graph, build_node_zone_mapping, has_valid_zones
from src.utilities.paths import REAL_INSTANCES_FOLDER, TRIP_POOL_PATH

def add_random_vehicles(
        network,
        start_ID,
        nb_vehicles,
        start_time=0,
        vehicles_end_time=100000,
        boarding_time=10,
        capacity=4,
        rng: Optional[np.random.Generator] = None):
    """ Function: Adds random vehicles (taxis) to the system with uniformly distributed initial positions.
        Input:
        ------------
        network : Network object
            The road network, including nodes representing stop points.
        start_ID : int
            Starting ID for the vehicles.
        nb_vehicles : int
            The number of vehicles (taxis) to add with uniform initial positions and specified availability times.
        start_time : float, optional
            The time in seconds after which vehicles become available. Defaults to 0.0.
        vehicles_end_time : float, optional
            The time in seconds after which vehicles are no longer available. Defaults to 100000.0.
        boarding_time : float, optional
            The time in seconds required for boarding passengers. Defaults to 10.0.
        capacity : int, optional
            The passenger capacity of each vehicle. Defaults to 4.
        rng : numpy.random.Generator, optional
            Random number generator to draw start stops from. Pass the same generator used for
            create_requests_from_pool to make a whole instance (requests + vehicles) reproducible
            from one seed via numpy.random.default_rng(seed). Defaults to an unseeded generator.

        Output:
        ------------
            taxis: List of generated Vehicle objects.
    """
    if rng is None:
        rng = np.random.default_rng()

    vehicles = []

    for i in range(nb_vehicles):
        vehicle_id = str(start_ID + i)

        # Randomly select a stop ID ensuring it's within the network nodes
        stop_id = str(rng.integers(0, len(network.nodes)))
        start_stop_location = LabelLocation(stop_id)

        # Calculate arrival and departure times
        arrival_time = start_time
        departure_time = start_time + boarding_time

        start_stop = Stop(
            arrival_time=arrival_time,
            departure_time=departure_time,
            location=start_stop_location
        )

        # Create the Vehicle object
        vehicle = Vehicle(vehicle_id,
                          start_time=start_time,
                          start_stop=start_stop,
                          capacity=capacity,
                          release_time=start_time,
                          end_time=vehicles_end_time)

        vehicles.append(vehicle)

    return vehicles

def save_trips_to_csv(trips, save_file_path, node_zone_mapping=None):
    """ Function: save list of requests to a CSV file
        Input:
        ------------
            trips: List of trip objects or dictionaries.
            save_file_path: String specifying the path to the output CSV file.
            node_zone_mapping: Optional dict mapping node id -> taxi zone LocationID (see
                network_tools.build_node_zone_mapping). Origin/destination zone columns are left
                blank when not provided (e.g. for a synthetic network with no real-world zones).
    """
    node_zone_mapping = node_zone_mapping or {}
    fieldnames = ["trip_id", "origin", "destination", "origin_zone", "destination_zone",
                  "release_time", "ready_time", "due_time", "fare"]
    with open(save_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for trip in trips:
            writer.writerow({
                "trip_id": trip.id,
                "origin": trip.origin.label,
                "destination": trip.destination.label,
                "origin_zone": node_zone_mapping.get(trip.origin.label, ""),
                "destination_zone": node_zone_mapping.get(trip.destination.label, ""),
                "release_time": trip.release_time,
                "ready_time": trip.ready_time,
                "due_time": trip.latest_pickup,
                "fare": trip.fare,
            })

def save_vehicles_to_csv(vehicles, save_file_path):
    """ Function: save list of vehicles to a CSV file
        Input:
        ------------
            vehicles: List of vehicle objects or dictionaries.
            save_file_path: String specifying the path to the output CSV file.
    """
    fieldnames = ["vehicle_id", "start_time", "start_stop", "capacity"]
    with open(save_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for vehicle in vehicles:
            writer.writerow({
                "vehicle_id": vehicle.id,
                "start_time": vehicle.start_time,
                "start_stop": vehicle.start_stop.location.label,
                "capacity": vehicle.capacity,
            })


def create_instances(config_data: Dict[str, Any]):

    network = load_network_graph(os.path.join(REAL_INSTANCES_FOLDER, "graph.json"))

    # Pool of real trip records to sample requests from (see clean_real_trips.build_trip_pool).
    trip_pool = pd.read_csv(TRIP_POOL_PATH)

    # Map each node to its real-world zone, from manhattan-geo.json alongside the network.
    zones_geojson_path = os.path.join(REAL_INSTANCES_FOLDER, "manhattan-geo.json")
    node_zone_mapping = build_node_zone_mapping(network, zones_geojson_path)

    # One rng for the whole run, threaded through every request/vehicle draw below, so a single
    # "seed" in config_data makes the entire run -- every instance -- reproducible end to end.
    # Omitting "seed" (the default) preserves unseeded, non-deterministic behavior.
    rng = np.random.default_rng(config_data.get("seed"))

    # Dynamically generate test names -- all instances in one call are the same calibrated size, no
    # tiering. start_index (default 1) lets a separate call generate a distinct, non-overlapping named
    # group (e.g. a denser experimental batch) without overwriting an earlier group's Instance_N files.
    start_index = config_data.get("start_index", 1)
    instances = [f"Instance_{start_index + i}" for i in range(config_data["num_instances"])]

    # Initialize table data
    table_data = []

    for test_name in instances:
        nb_requests = config_data["nb_requests"]
        # Sample requests from the real trip pool (zone-invalid trips are skipped and topped up
        # from the rest of the window automatically -- see create_requests_from_pool).
        requests = create_requests_from_pool(
            trip_pool=trip_pool,
            node_zone_mapping=node_zone_mapping,
            start_ID=0,
            start_time=config_data["start_time"],
            sim_time=config_data["sim_time"],
            nb_requests=nb_requests,
            time_window=180,  # seconds; calibrated via reachability analysis (see plan)
            alpha=config_data["alpha"],
            beta=config_data["beta"],
            advance_notice=config_data["advance_notice"],
            known_portion=config_data["known_portion"],
            rng=rng,
        )

        # Add random vehicles
        vehicles = add_random_vehicles(
            network=network,
            start_ID=0,
            nb_vehicles=config_data["nb_vehicles"],
            # Same clock as the requests: vehicles come on shift when the request window opens,
            # rather than at t=0, which otherwise leaves the simulation (and the viewer timeline)
            # with hours of dead air before the first request.
            start_time=config_data["start_time"],
            vehicles_end_time=100000,
            boarding_time=10,
            capacity=4,
            rng=rng,
        )

        # Save generated data to CSV
        test_path = os.path.join(REAL_INSTANCES_FOLDER, test_name)
        os.makedirs(test_path, exist_ok=True)
        requests_file_path = os.path.join(test_path, "requests.csv")
        vehicles_file_path = os.path.join(test_path, "vehicles.csv")
        save_trips_to_csv(requests, requests_file_path, node_zone_mapping)
        save_vehicles_to_csv(vehicles, vehicles_file_path)

        # Append to table data
        table_data.append({
            "Instance": test_name,
            "# Requests": len(requests),
            "# Vehicles": len(vehicles)
        })

    # Print results
    df = pd.DataFrame(table_data)
    with pd.option_context('display.colheader_justify', 'center'):
        print(df.to_markdown(tablefmt="pipe", headers="keys"))
