import numpy as np
import pandas as pd

from src.solvers.offline_solver import OfflineSolver
from src.solvers.solver import Solver
from src.utilities.enums import ConsensusParams
from src.utilities.create_scenario import create_requests_from_pool
from typing import Any, Dict, List
from src.utilities.config import SimulationConfig
from src.utilities.paths import TRIP_POOL_PATH
from src.utilities.types import TripLike, VehicleLike, TravelTimeMatrix



class StochasticSolver(Solver):
    """Online stochastic solution: scenario generation plus offline solve per scenario.

    Two types of consensus:
        - qualitative: aggregate which request is "best" across scenarios (e.g. by voting).
        - quantitative: use scenario objective values to score best request-vehicle assignments.

    Attributes:
    ------------
        consensus_type : ConsensusParams(Enum)
            Type of consensus algorithm.
        nb_scenario : int
            Number of scenarios to generate and solve at each step.
        scenario_param : Dict[str, Any]
            Parameters for scenario generation (time_window, nb_requests, generation_sim_time,
            known_portion, advance_notice).
        trip_pool : pandas.DataFrame
            Pool of real trip records (see clean_real_trips.build_trip_pool) that scenario requests
            are sampled from.
        rng : numpy.random.Generator
            Random number generator for scenario sampling, seeded from simulation_config.consensus.seed.

        vehicle_request_assign : Dict[int, VehicleState]
            Mapping vehicle id to VehicleState (inherited from Solver). Each state holds: vehicle,
            assigned_requests, departure_stop, departure_time, last_stop, last_stop_time, assign_possible,
            random_number; used to save assignments and build route plans.

        network : Any
            The road network, including nodes representing stop points.
        durations : Dict
            Travel time matrix between stop points (e.g. self.durations[from_label][to_label]).
        costs : Dict
            Driving cost matrix (same structure as durations).
        algorithm: Algorithm(Enum)
            The optimization algorithm utilized for planning and assigning trips to vehicles.
        objective: Objectives(Enum)
            The objective used to evaluate the effectiveness of the plan (e.g., maximizing profit or minimizing wait time).
        objective_value: float
            The objective value from served requests.
        total_customers_served: int
            The count of customers successfully served.
    """

    def __init__(self,
                 network: Any,
                 vehicles: List[Any],
                 simulation_config: SimulationConfig,
                 durations: TravelTimeMatrix,
                 costs: TravelTimeMatrix):
        super().__init__(network, vehicles, simulation_config, durations, costs)
        self.consensus_type = simulation_config.consensus.consensus_param
        self.nb_scenario = simulation_config.consensus.nb_scenario

        self.scenario_param = {
            'time_window': simulation_config.time_window,       # Time window for picking up the requests
            'nb_requests': simulation_config.consensus.nb_requests,  # requests this instance was generated with (see config_tools.derive_consensus_params)
            'generation_sim_time': simulation_config.consensus.generation_sim_time,  # window nb_requests was sampled over at generation time
            'known_portion': simulation_config.known_portion,    # percentage of requests that are known in advance
            'advance_notice': simulation_config.advance_notice,  # Fixed amount of time (in seconds) the requests are released before their ready time.
        }
        self.trip_pool = pd.read_csv(TRIP_POOL_PATH)
        self.rng = np.random.default_rng(simulation_config.consensus.seed)

    def solve(self, all_vehicles, active_vehicles, trips, rejected_trips, current_time=None) -> None:
        """Uniform entry point for TaxiDispatcher.optimize().

        `rejected_trips` is accepted for interface uniformity with OnlineSolver/ReOptimizer.solve
        but unused by the consensus algorithm.
        """
        self.variables_declaration(all_vehicles, trips)
        self.stochastic_solver(active_vehicles, trips, current_time)

    def stochastic_solver(self, K: List[VehicleLike], P_not_assigned: List[TripLike], current_time: float) -> None:
        """Assign ride requests to vehicles using scenario-based consensus.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            current_time : current time (used for scenario generation).
        """

        # Step 1: assign requests to the vehicles/ routes
        P_not_assigned = sorted(P_not_assigned, key=lambda x: x.ready_time)
        assigned_requests = []

        if self.consensus_type == ConsensusParams.QUANTITATIVE:
            assigned_requests = self.quantitative_consensus(K, P_not_assigned, current_time)

        elif self.consensus_type == ConsensusParams.QUALITATIVE:
            assigned_requests = self.qualitative_consensus(K, P_not_assigned, current_time)

        # Step 2: check the feasibility of the solution
        self.create_online_solution()
        if self.verify_constraints(K, assigned_requests):
            self.calc_objective_value(K, P_not_assigned)
            self.total_customers_served = sum(1 for f_i in P_not_assigned if self.Z[f_i.id])

        else:
            raise ValueError("The solution is not feasible")

    def qualitative_consensus(self, K: List[VehicleLike], P_not_assigned: List[TripLike], current_time: float) -> List[TripLike]:
        """Assign requests using qualitative consensus over scenario solutions.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            current_time : current time (for scenario generation).

        Output:
        ------------
            assigned_requests : list of assigned requests.

        Hint:
            - Generate multiple scenarios (use create_requests_from_pool from create_scenario.py) and solve each with OfflineSolver.
            - Use the scenario solutions to decide which request to assign to which vehicle (e.g. aggregate
              information across scenarios).
            - Assign one request at a time and update vehicle state before the next.
        """
        raise NotImplementedError("StochasticSolver.qualitative_consensus() not implemented")

        return assigned_requests


    def quantitative_consensus(self, K: List[VehicleLike], P_not_assigned: List[TripLike], current_time: float) -> List[TripLike]:
        """Assign requests using quantitative consensus (score by scenario objective values).

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            current_time : current time (for scenario generation).

        Output:
        ------------
            assigned_requests : list of assigned requests.

        Hint:
            - Generate multiple scenarios (use create_requests_from_pool from create_scenario.py) and solve each with OfflineSolver.
            - Use each scenario's objective value to score or rank request-vehicle options; then choose
              assignments.
            - Assign one request at a time and update vehicle state before the next.
        """
        raise NotImplementedError("StochasticSolver.quantitative_consensus() not implemented")
        return assigned_requests

