import random
from typing import Any, List
import logging

from src.utilities.config import SimulationConfig
from src.utilities.enums import Algorithm, Objectives, SolutionMode
from src.solvers.solver import Solver
from src.utilities.types import TripLike, VehicleLike, TravelTimeMatrix

logger = logging.getLogger(__name__)


class OnlineSolver(Solver):
    """Provide online solution to optimize the vehicle routing and the trip-route assignment. This
    method includes:
        1. greedy solver
        2. random solver
        3. ranking solver

    Attributes:
    ------------
        vehicle_request_assign : Dict[int, VehicleState]
            Mapping vehicle id to VehicleState (inherited from Solver). Each state holds: vehicle,
            assigned_requests, departure_stop, departure_time, last_stop, last_stop_time, assign_possible,
            random_number; used to save assignments and build route plans.

        durations : dictionary
            travel time matrix between possible stop points
            example: for duration between destination of trip_i and the origin of trip_j use:
                     self.durations[trip_i.destination.label][trip_i.origin.label]

        costs: dictionary
            driving costs (it works based on location ids like durations)
        algorithm: Algorithm(Enum)
            The optimization algorithm utilized for planning and assigning trips to vehicles.
        objective: Objectives(Enum)
            The objective used to evaluate the effectiveness of the plan (e.g., maximizing profit or minimizing wait time).
        objective_value: float
            The objective value from served requests.
        total_customers_served: int
            The count of customers successfully served.

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
                 vehicles: List[Any],
                 simulation_config: SimulationConfig,
                 durations: TravelTimeMatrix,
                 costs: TravelTimeMatrix):
        super().__init__(network, vehicles, simulation_config, durations, costs)
        # Assign random numbers to each vehicle request assignment
        for vehicle_id, state in self.vehicle_request_assign.items():
            state.random_number = random.random()

    def solve(self, all_vehicles, active_vehicles, trips, rejected_trips, current_time=None) -> None:
        """Uniform entry point for TaxiDispatcher.optimize().

        `current_time` is accepted for interface uniformity with StochasticSolver.solve but unused
        by the online algorithms.
        """
        self.variables_declaration(all_vehicles, trips)
        self.online_solver(active_vehicles, trips, rejected_trips)

    def determine_available_vehicles(self, trip: TripLike) -> None:
        """Determine whether each vehicle can feasibly serve the trip (e.g. reach pickup by latest_pickup).

        Input:
        ------------
            trip : ride request to serve

        Hint:
            - Iterate over self.vehicle_request_assign and set assign_possible for each vehicle.
            - Use the parent class method calc_reach_time(veh_info, trip).
        """
        raise NotImplementedError("OnlineSolver.determine_available_vehicles() not implemented")


    def online_solver(self, K: List[VehicleLike], P_not_assigned: List[TripLike], rejected_trips: List[TripLike]) -> None:
        """Find a solution to assign ride requests to vehicles after arrival.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Steps:
            1. Assign requests to vehicles/routes based on the chosen algorithm.
            2. The three online methods differ only in how they choose among the available vehicles.
            3. Check the feasibility of the solution.
        """

        # Step 1: assign requests to the vehicles/ routes
        sorted_requests = sorted(P_not_assigned, key=lambda x: x.ready_time)

        if self.algorithm == Algorithm.GREEDY:
            assigned_requests = self.greedy_assign(sorted_requests, rejected_trips)
        elif self.algorithm == Algorithm.RANDOM:
            assigned_requests = self.random_assign(sorted_requests, rejected_trips)
        elif self.algorithm == Algorithm.RANKING:
            assigned_requests = self.ranking_assign(sorted_requests, rejected_trips)
        else:
            logger.error("Unsupported algorithm: %s", self.algorithm)
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Step 2: check the feasibility of the solution
        self.create_online_solution()
        if self.verify_constraints(K, assigned_requests):
            self.calc_objective_value(K, sorted_requests)
            if self.simulation_config.solution_mode == SolutionMode.OFFLINE:
                self.total_customers_served = sum(1 for f_i in P_not_assigned if self.Z[f_i.id])
            else:
                # Outside OFFLINE mode only the first trip per vehicle becomes a served leg
                # this round, so count vehicles with an assignment rather than the full chain.
                self.total_customers_served = sum(
                    1 for state in self.vehicle_request_assign.values() if state.assigned_requests
                )
            logger.info("Online solver solution is feasible. Served %d customers.", self.total_customers_served)

        else:
            logger.error("The solution is not feasible.")
            raise ValueError("The solution is not feasible")

    def greedy_assign(self, P_not_assigned: List[TripLike], rejected_trips: List[TripLike]) -> List[TripLike]:
        """Assign ride requests to vehicles using a greedy method (e.g. best objective per request).

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - Every trip ends up either assigned to a vehicle or appended to rejected_trips.
            - determine_available_vehicles decides which vehicles could serve a given trip;
              assign_trip_to_vehicle records the choice and updates that vehicle's state.
        """
        raise NotImplementedError("OnlineSolver.greedy_assign() not implemented")

    def random_assign(self, P_not_assigned: List[TripLike], rejected_trips: List[TripLike]) -> List[TripLike]:
        """Assign ride requests to vehicles based on random solution method

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - Every trip ends up either assigned to a vehicle or appended to rejected_trips.
            - determine_available_vehicles decides which vehicles could serve a given trip;
              assign_trip_to_vehicle records the choice and updates that vehicle's state.
        """
        raise NotImplementedError("OnlineSolver.random_assign() not implemented")

    def ranking_assign(self, P_not_assigned: List[TripLike], rejected_trips: List[TripLike]) -> List[TripLike]:
        """Assign ride requests to vehicles by ranking solution method.

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - Every trip ends up either assigned to a vehicle or appended to rejected_trips.
            - determine_available_vehicles decides which vehicles could serve a given trip;
              assign_trip_to_vehicle records the choice and updates that vehicle's state.

        """
        raise NotImplementedError("OnlineSolver.ranking_assign() not implemented")
