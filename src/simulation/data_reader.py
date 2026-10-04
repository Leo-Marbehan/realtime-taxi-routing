import csv
import random
import networkx as nx
from typing import Optional

from multimodalsim.reader.data_reader import DataReader
from multimodalsim.simulator.vehicle import Vehicle
from multimodalsim.simulator.stop import LabelLocation, Stop
from src.utilities.config import SimulationConfig

from src.simulation.ride_request import RideRequest
from src.utilities.enums import SolutionMode
from src.utilities.network_tools import load_network_graph


class TaxiDataReader(DataReader):
    """
    TaxiDataReader is responsible for reading and processing input data from files,
    including trip requests, vehicle information, and the transportation network.

    Additional Attributes:
    ------------
    requests_file_path : str
        File path to the CSV file containing trip requests.
    vehicles_file_path : str
        File path to the CSV file containing vehicle data.
    sim_end_time : int
        Simulation end time.
    vehicles_end_time : int
        End time for vehicles' operations.
    """

    def __init__(self, requests_file_path: str, vehicles_file_path: str,
                 sim_end_time: Optional[int] = None,
                 vehicles_end_time: Optional[int] = None) -> None:

        super().__init__()
        self.__network = None
        self.__requests_file_path = requests_file_path
        self.__vehicles_file_path = vehicles_file_path

        # The time difference between the arrival and the departure time.
        self.__boarding_time = 0
        self.__sim_end_time = sim_end_time
        self.__vehicles_end_time = vehicles_end_time
        self.__horizon_start: Optional[float] = None  # lazily derived, see _horizon_start()

    def _horizon_start(self) -> float:
        """Time the fleet comes on shift, i.e. the earliest vehicle start_time.

        This is the operating horizon's origin, so it -- not 0 -- is what "known from the start"
        means for a request's release_time. Read lazily and cached: get_trips() runs before
        get_vehicles(), so it cannot reuse anything the latter parsed.
        """
        if self.__horizon_start is None:
            with open(self.__vehicles_file_path, 'r', newline='') as vFile:
                self.__horizon_start = min(
                    float(entry['start_time']) for entry in csv.DictReader(vFile, delimiter=';'))
        return self.__horizon_start

    def get_trips(self, config: SimulationConfig):
        """ Function: read trip from a file
            Input:
            ------------
            solution_mode : The mode of solution (offline, fully online, etc.).
            time_window : Time window for requests pickup
            known_portion: percentage of requests that are known in advance

            Output:
            ------------
            trips : A list of RideRequest objects.
        """
        trips = []
        with open(self.__requests_file_path, 'r', newline='') as rFile:
            csv_data = list(csv.DictReader(rFile, delimiter=';'))
            nb_passengers = 1  # Each request corresponds to 1 passenger.
            # random.seed(42)
            # Shuffle the indices and select known_portion of them
            indices = list(range(len(csv_data)))
            random.shuffle(indices)
            known_trip_indices = indices[:int(len(csv_data) * config.known_portion/100)]

            for idx, entry in enumerate(csv_data):
                # Process trip data
                orig_id = str(int(entry['origin']))
                dest_id = str(int(entry['destination']))
                fare_value = float(entry['fare'])

                lon_orig = lat_orig = None
                lon_dest = lat_dest = None
                if self.__network is not None:
                    lon_orig, lat_orig = self.__network.nodes[orig_id]['pos']
                    lon_dest, lat_dest = self.__network.nodes[dest_id]['pos']

                orig_location = LabelLocation(orig_id, lon=lon_orig, lat=lat_orig)
                dest_location = LabelLocation(dest_id, lon=lon_dest, lat=lat_dest)

                travel_time = None
                if self.__network is not None:
                    path_info = self.__network.nodes[orig_id]['shortest_paths'][dest_id]
                    travel_time = path_info['total_duration']

                ready_time = round(float(entry['ready_time']), 3)
                # Determine release_time. "Known from the start" means the start of the operating
                # horizon (when the fleet comes on shift), not absolute time 0 -- a request revealed
                # before any vehicle exists is meaningless, and it would leave the simulation (and
                # the viewer timeline) with dead air ahead of the first vehicle.
                horizon_start = self._horizon_start()
                if config.solution_mode == SolutionMode.OFFLINE:
                    release_time = horizon_start
                elif config.solution_mode == SolutionMode.FULLY_ONLINE:
                    release_time = ready_time
                elif config.solution_mode == SolutionMode.PARTIAL_ONLINE:
                    if idx in known_trip_indices:
                        release_time = horizon_start
                    else:
                        release_time = ready_time
                elif config.solution_mode == SolutionMode.ADVANCE_NOTICE:
                    release_time = max(horizon_start, ready_time - config.advance_notice)
                elif config.solution_mode == SolutionMode.CUSTOM_SCENARIO:
                    if idx in known_trip_indices:
                        release_time = horizon_start
                    else:
                        release_time = max(horizon_start, ready_time - config.advance_notice)
                else:
                    raise ValueError(f"Unsupported solution mode: {config.solution_mode}")


                # Create and append RideRequest object
                trip = RideRequest(
                    id=entry['trip_id'],
                    origin=orig_location,
                    destination=dest_location,
                    nb_passengers=nb_passengers,
                    release_time=release_time,
                    ready_time=ready_time,
                    due_time=100000.0,  # Adjust as needed
                    latest_pickup=round(ready_time + config.time_window, 3),
                    fare=fare_value,
                    shortest_travel_time=travel_time
                )
                trips.append(trip)
        random.seed(None)
        return trips

    def get_vehicles(self, vehicles_end_time: Optional[float] = None):
        """ Function: read vehicles from a file
                Input:
                ------------
                vehicles_end_time : float, optional
                    Overrides the end_time every vehicle is given (e.g. derived from the actual trips'
                    latest_pickup/travel times by the caller, so it covers whatever absolute time range
                    this instance spans). Falls back to the constructor's vehicles_end_time if omitted.

                Output:
                ------------
                vehicles: a list of Vehicle objects
                routes_by_vehicle_id : a dictionary of routes by vehicle ID
        """
        end_time = vehicles_end_time if vehicles_end_time is not None else self.__vehicles_end_time
        vehicles = []
        routes_by_vehicle_id = {}  # Remains empty

        with open(self.__vehicles_file_path, 'r', newline='') as rFile:
            csv_data = csv.DictReader(rFile, delimiter=';')
            for entry in csv_data:
                # Process vehicle data
                vehicle_id = entry['vehicle_id']
                start_time = float(entry['start_time'])
                stop_departure_time = start_time + self.__boarding_time
                capacity = int(entry['capacity'])
                stop_id = str(int(entry['start_stop']))

                lon, lat, mode = None, None, None
                if self.__network is not None:
                    lon, lat = self.__network.nodes[stop_id]['pos']
                start_stop_location = LabelLocation(stop_id, lon=lon, lat=lat)

                start_stop = Stop(
                    arrival_time=start_time,
                    departure_time=stop_departure_time,
                    location=start_stop_location
                )

                # Create and append Vehicle object
                vehicle = Vehicle(
                    veh_id = vehicle_id,
                    start_time= start_time,
                    start_stop=start_stop,
                    capacity=capacity,
                    release_time= start_time,
                    end_time=end_time,
                    mode=mode)

                vehicles.append(vehicle)

        return vehicles, routes_by_vehicle_id

    def load_graph(self, file_path: str) -> nx.DiGraph:
        """Load the graph with all its data from a JSON file (networkx node-link format)."""
        self.__network = load_network_graph(file_path)
        return self.__network


