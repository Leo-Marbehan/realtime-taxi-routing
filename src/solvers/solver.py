import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utilities.enums import Objectives
from src.utilities.config import SimulationConfig
from src.utilities.types import TripLike, VehicleLike, TravelTimeMatrix

logger = logging.getLogger(__name__)


@dataclass
class VehicleState:
    """State information for a vehicle in the routing problem.

    Attributes:
        vehicle: The vehicle object (taxi) this state belongs to.
        assigned_requests: List of ride requests (trips) currently assigned to this vehicle.
        departure_stop: Label of the last stop from the previous iteration (starting point for the current plan).
        departure_time: Time at which the vehicle departs from departure_stop.
        last_stop: Label of the last stop in the current solution (end of assigned_requests).
        last_stop_time: The departure time from the last assigned stop in the current solution (used for reach-time and feasibility).
        assign_possible: A boolean value indicating whether this vehicle can feasibly serve a candidate trip (e.g. set in determine_available_vehicles).
        random_number: Optional score used by some online algorithms (e.g. ranking) to break ties.
        route_next_stops_count: len(route.next_stops) as of this round's update_vehicle_state --
            the raw fact behind RoutePlanBuilder.build_route_plans's outer per-vehicle gate
            (solution_mode == OFFLINE or len(route.next_stops) <= 1): a vehicle already mid-trip
            (>1) won't have a new route plan materialized for it this round in non-OFFLINE modes,
            no matter what a solver decides. Populated for every solver (shared base class); inert
            unless a solver's own candidate-selection logic reads it.
    """
    vehicle: VehicleLike
    assigned_requests: List[TripLike] = field(default_factory=list)
    departure_stop: Optional[str] = None
    departure_time: float = 0.0
    last_stop: Optional[str] = None
    last_stop_time: float = 0.0
    assign_possible: bool = False
    random_number: Optional[float] = None
    route_next_stops_count: int = 0


class Solver:
    """
    A class to solve the taxi routing problem using a Mixed Integer Programming (MIP) approach with Gurobi.

        Attributes:
        ------------
        vehicle_request_assign: dictionary
            a dictionary for each vehicle to keep track of various attributes associated with each vehicle
            This dictionary allows for saving the assignments of trips to vehicles which is used to create
            the route plan
        network: Any
            The road network, including nodes representing stop points.
        duration : dictionary
            travel time matrix between possible stop points
        costs: dictionary
            driving costs
        algorithm: Algorithm(Enum)
            The optimization algorithm utilized for planning and assigning trips to vehicles.
        objective: Objectives(Enum)
            The objective used to evaluate the effectiveness of the plan (e.g., maximizing profit or minimizing wait time).
        objective_value: float
            The objective value from served requests.

        X: Dict[int, Dict[int, bool]]
            Binary variables indicating if customer i is picked immediately after j by a taxi.
        Y: Dict[int, Dict[int, bool]]
            Binary variables indicating if customer i is picked up by vehicle k as the first customer.
        U: Dict[int, float]
            Pickup times for customers.
        Z: Dict[int, bool]
            Binary variables indicating if customer i is selected to be served.
        """

    def __init__(self,
                 network: Any,
                 vehicles: List[VehicleLike],
                 simulation_config: SimulationConfig,
                 durations: TravelTimeMatrix,
                 costs: TravelTimeMatrix) -> None:
        """
        Input:
        ------------
        network: The transport network over which the dispatching occurs.
        vehicles: Set of input vehicles
        simulation_config : Configuration object containing all simulation parameters.
        durations: Travel time matrix between stop points, computed once (e.g. by TaxiDispatcher)
            and shared across solver instances rather than recomputed here.
        costs: Driving cost matrix, computed once and shared the same way as durations.
        """
        self.network = network
        self.simulation_config = simulation_config
        self.algorithm = simulation_config.algorithm
        self.objective = simulation_config.objective
        self.objective_value = 0
        self.weight = simulation_config.weight
        self.durations: TravelTimeMatrix = durations
        self.costs: TravelTimeMatrix = costs
        self.vehicle_request_assign: Dict[int, VehicleState] = {}

        for vehicle in vehicles:
            self.vehicle_request_assign[vehicle.id] = VehicleState(vehicle=vehicle)

        self.X: Dict[int, Dict[int, bool]] = {}  # Decision variables for trip connections between customers
        self.Y: Dict[int, Dict[int, bool]] = {}  # Decision variables for assigning customers to vehicles
        self.Z: Dict[int, bool] = {}  # Decision variables for whether a customer is served
        self.U: Dict[int, float] = {}  # Decision variables for departure times from customer locations


    def update_vehicle_state(self, selected_routes: List[Any], current_time: float) -> None:
        """
            Function: Update departure time and point for the vehicles based on the current routes
            Input:
            ------------
                selected_routes : current vehicle routes
                current_time : current time of the simultaion
        """
        for route in selected_routes:
            vehicle_id = route.vehicle.id
            vehicle_state = self.vehicle_request_assign.get(vehicle_id)
            if vehicle_state is None:
                vehicle_state = VehicleState(vehicle=route.vehicle)
                self.vehicle_request_assign[vehicle_id] = vehicle_state

            vehicle_state.assigned_requests.clear()
            vehicle_state.route_next_stops_count = len(route.next_stops)

            if not route.next_stops:
                # vehicle route is empty
                last_stop = route.previous_stops[-1] if route.current_stop is None else route.current_stop
                departure_time = last_stop.departure_time if last_stop.departure_time != math.inf else last_stop.arrival_time
                if departure_time < current_time:
                    departure_time = current_time
                vehicle_state.departure_time = departure_time
                vehicle_state.departure_stop = last_stop.location.label
                vehicle_state.last_stop_time = departure_time
                vehicle_state.last_stop = last_stop.location.label
            else:
                last_stop = route.next_stops[-1]
                vehicle_state.departure_time = last_stop.arrival_time
                vehicle_state.departure_stop = last_stop.location.label
                vehicle_state.last_stop_time = last_stop.arrival_time
                vehicle_state.last_stop = last_stop.location.label

    def get_available_vehicles(self) -> Dict[int, VehicleState]:
        """Return the subset of vehicle_request_assign whose assign_possible flag is set."""
        return {
            vehicle_id: state
            for vehicle_id, state in self.vehicle_request_assign.items()
            if state.assign_possible
        }

    def print_vehicle_request_assign(self) -> None:
        """
        Function: Prints the current state of the vehicle_request_assign dictionary.
        """
        for vehicle_id, data in self.vehicle_request_assign.items():
            assigned_request_ids = [request.id for request in data.assigned_requests]
            if assigned_request_ids:
                logger.debug("  vehicle ID: %s", vehicle_id)
                logger.debug("  Assigned Requests (IDs): %s", assigned_request_ids)
                for key, value in vars(data).items():
                    if key not in ['assigned_requests', 'vehicle']:
                        logger.debug("  %s: %s", key, value)

    def calc_reach_time(self, vehicle_info: VehicleState, trip: TripLike) -> float:
        """Function: Calculate the time a vehicle can reach the trip's origin."""

        reach_time = vehicle_info.last_stop_time + self.durations[vehicle_info.last_stop][trip.origin.label]
        return max(reach_time, trip.ready_time)

    def assign_trip_to_vehicle(self, vehicle_info: VehicleState, trip: TripLike) -> None:
        """Function: Assign trip to a vehicle."""
        vehicle_info.assigned_requests.append(trip)
        reach_time_to_pickup = self.calc_reach_time(vehicle_info, trip)
        vehicle_info.last_stop_time = reach_time_to_pickup + trip.shortest_travel_time
        vehicle_info.last_stop = trip.destination.label

    def variables_declaration(self, K: List[VehicleLike], P: List[TripLike]) -> None:
        """
        Declares variables for online solvers.

        Parameters:
        ------------
        K: List of vehicles.
        P: List of customers to serve.

        """
        # initialized with False
        self.X = {i.id: {j.id: False for j in P} for i in P}
        self.Y = {k.id: {i.id: False for i in P} for k in K}
        self.U = {i.id: 0.0 for i in P}
        self.Z = {i.id: False for i in P}

    def create_online_solution(self):
        """
        Function: determine the value of the variables in the model based on vehicle_request_assign dictionary
                the function is used to check the feasibility of the solution
        """
        for vehicle_id, veh_info in self.vehicle_request_assign.items():
            if veh_info.assigned_requests:
                trip = veh_info.assigned_requests[0]
                self.Y[vehicle_id][trip.id] = True
                self.U[trip.id] = max(
                    trip.ready_time,
                    veh_info.departure_time + self.durations[veh_info.departure_stop][trip.origin.label],
                )
                self.Z[trip.id] = True

                for i in range(1, len(veh_info.assigned_requests)):
                    prev_trip = veh_info.assigned_requests[i - 1]
                    curr_trip = veh_info.assigned_requests[i]
                    self.X[prev_trip.id][curr_trip.id] = True
                    self.U[curr_trip.id] = max(
                        curr_trip.ready_time,
                        self.U[prev_trip.id] + prev_trip.shortest_travel_time
                        + self.durations[prev_trip.destination.label][curr_trip.origin.label],
                    )
                    self.Z[curr_trip.id] = True

    def verify_constraints(self, K: List[VehicleLike], P: List[TripLike]) -> bool:
        """
        Verifies all constraints.

        Parameters:
        ------------
        K: List of vehicles.
        P: List of customers to serve.

        Returns:
        --------
        bool: True if all constraints are satisfied, False otherwise.
        """
        return (
                self._verify_const_1(K, P) and
                self._verify_const_2(P) and
                self._verify_const_3(K, P) and
                self._verify_const_4(P) and
                self._verify_const_5(P) and
                self._verify_const_6(K, P)
        )

    def _verify_const_1(self, K: List[VehicleLike], P: List[TripLike]) -> bool:
        """
        Function to verify the validation of the Constraint (1)

        Parameters:
        ------------
        K: List of vehicles.
        P: List of customers to serve.

        Returns:
        --------
        bool: True if Constraint 1 is satisfied, False otherwise.
        """
        verified = True
        for trip_i in P:
            sum_x = 0
            sum_y = 0
            for trip_j in P:
                sum_x += (1 if self.X[trip_j.id][trip_i.id] else 0)

            for vehicle_k in K:
                sum_y += (1 if self.Y[vehicle_k.id][trip_i.id] else 0)

            z_i = 1 if self.Z[trip_i.id] else 0
            if z_i != sum_x + sum_y:
                verified = False
                break
        return verified

    def _verify_const_2(self, P: List[TripLike]) -> bool:
        """
        Function to verify the validation of the Constraint (2)

        Parameters:
        ------------
        P: List of customers to serve.

        Returns:
        --------
        bool: True if Constraint 2 is satisfied, False otherwise.
        """
        verified = True
        for trip_i in P:
            sum_x = 0
            for trip_j in P:
                sum_x += (1 if self.X[trip_i.id][trip_j.id] else 0)

            z_i = 1 if self.Z[trip_i.id] else 0
            if z_i < sum_x:
                verified = False
                break
        return verified

    def _verify_const_3(self, K: List[VehicleLike], P: List[TripLike]) -> bool:
        """
        Function to verify the validation of the Constraint (3)

        Parameters:
        ------------
        K: List of vehicles.
        P: List of customers to serve.

        Returns:
        --------
        bool: True if Constraint 3 is satisfied, False otherwise.
        """
        verified = True
        for vehicle_k in K:
            sum_y = 0
            for trip_i in P:
                sum_y += (1 if self.Y[vehicle_k.id][trip_i.id] else 0)

            if sum_y > 1:
                verified = False
                break
        return verified

    def _verify_const_4(self, P: List[TripLike]) -> bool:
        """
        Function to verify the validation of the Constraint (4)

        Parameters:
        ------------
        K: List of vehicles.
        P: List of customers to serve.

        Returns:
        --------
        bool: True if Constraint 4 is satisfied, False otherwise.
        """
        verified = True
        for trip_i in P:
            ready_time_trip_i = trip_i.ready_time
            latest_time_trip_i = trip_i.latest_pickup
            if self.U[trip_i.id] < ready_time_trip_i or self.U[trip_i.id] > latest_time_trip_i:
                verified = False
                break
        return verified

    def _verify_const_5(self, P: List[TripLike]) -> bool:
        """
        Function to verify the validation of the Constraint (5)

        Parameters:
        ------------
        K: List of vehicles.
        P: List of customers to serve.

        Returns:
        --------
        bool: True if Constraint 5 is satisfied, False otherwise.
        """
        verified = True
        for trip_i in P:
            for trip_j in P:
                if trip_i != trip_j:
                    T_ij = trip_i.shortest_travel_time + self.durations[trip_i.destination.label][trip_j.origin.label]
                    if (self.U[trip_j.id] - self.U[trip_i.id]).__round__(3) < (
                            T_ij if self.X[trip_i.id][trip_j.id] else (trip_j.ready_time - trip_i.latest_pickup)).__round__(3):
                        verified = False
                        break
            if not verified:
                break
        return verified

    def _verify_const_6(self, K: List[VehicleLike], P: List[TripLike]) -> bool:
        """
        Function to verify the validation of the Constraint (6)

        Parameters:
        ------------
        K: List of vehicles.
        P: List of customers to serve.

        Returns:
        --------
        bool: True if Constraint 6 is satisfied, False otherwise.
        """
        verified = True
        for trip_i in P:
            for vehicle_k in K:
                state = self.vehicle_request_assign[vehicle_k.id]
                T_ki = self.durations[state.departure_stop][trip_i.origin.label]
                if self.U[trip_i.id].__round__(3) < (
                    (state.departure_time + T_ki) if self.Y[vehicle_k.id][trip_i.id] else trip_i.ready_time
                ).__round__(3):
                    verified = False
                    break
            if not verified:
                break
        return verified


    def calc_objective_value(self, K: List[VehicleLike], P: List[TripLike]) -> None:
        """
        Function to calculate the objective value

        Input:
        ------------
            K : set of vehicles
            P : set of customers to serve

        Hint:
            - self.Z tells you which trips ended up served; TOTAL_CUSTOMERS below is worked as an
              example of reading it.
            - self.vehicle_request_assign holds each vehicle's assignments and where it departs from.
            - self.costs and self.durations give cost and travel time between two stop points.
            - Each objective sums a different per-trip quantity -- work out which one from the
              objective's name and what a served trip actually costs or earns.
        """
        value = 0
        if self.objective == Objectives.TOTAL_CUSTOMERS:
            value = sum(1 for trip_i in P if self.Z[trip_i.id])

        elif self.objective == Objectives.TOTAL_PROFIT:
            # your code here
            pass
        elif self.objective == Objectives.WAIT_TIME:
            # your code here
            pass
        elif self.objective == Objectives.MULTI_OBJECTIVE:
            # your code here
            pass
        elif self.objective == Objectives.TOTAL_REVENUE:
            # your code here
            pass
        elif self.objective == Objectives.TOTAL_COST:
            # your code here
            pass
        elif self.objective == Objectives.TOTAL_EMPTY_TRAVEL_TIME:
            # your code here
            pass

        self.objective_value = value


