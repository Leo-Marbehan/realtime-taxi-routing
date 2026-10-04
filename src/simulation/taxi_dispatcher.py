import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

from multimodalsim.optimization.dispatcher import Dispatcher
from multimodalsim.optimization.optimized_route_plan import OptimizedRoutePlan
from src.utilities.config import SimulationConfig
from src.utilities.enums import Algorithm, Objectives, SolutionMode
from src.utilities.network_tools import get_durations, get_costs, compute_driving_cost
from src.utilities.timer import Timer
from src.utilities import result_columns as col

logger = logging.getLogger(__name__)

# Relative optimality gap for the exact offline MIP path (Algorithm.MIP_SOLVER). Named rather
# than inlined so the reference fixture can record the gap it was generated under.
OFFLINE_MIP_GAP = 0.01


# Instructor-only convenience: this repo pairs the public `main/` checkout with a private
# `src_solution/` reference-answers folder as a sibling of `main/` at the repo root. When present,
# prefer it over the public stub solvers. Students never have `src_solution/`, so this always falls
# through to the stub solvers below for them.
_SRC_SOLUTION_DIR = Path(__file__).resolve().parents[3] / "src_solution"
if _SRC_SOLUTION_DIR.exists():
    sys.path.insert(0, str(_SRC_SOLUTION_DIR.parent))

try:
    from src_solution.re_optimizer import ReOptimizer
    from src_solution.stochastic_solver import StochasticSolver
    from src_solution.online_solver import OnlineSolver
    from src_solution.offline_solver import OfflineSolver
    from src_solution.dqn_solver import DQNSolver
except ImportError:
    from src.solvers.re_optimizer import ReOptimizer
    from src.solvers.stochastic_solver import StochasticSolver
    from src.solvers.online_solver import OnlineSolver
    from src.solvers.offline_solver import OfflineSolver
    from src.solvers.dqn_solver import DQNSolver

from src.solvers.solver import Solver


class GurobiSolverAdapter(Solver):
    """Adapts `OfflineSolver` (the Gurobi MIP solver) to the same `solve(...)` entry point as
    `OnlineSolver`/`ReOptimizer`/`StochasticSolver`, so `TaxiDispatcher` doesn't need to special-case
    `Algorithm.MIP_SOLVER` separately from the other three algorithms (see architecture finding #1).
    Does not modify `OfflineSolver` itself -- its exercise-facing methods (`create_model`,
    `define_objective`, etc.) are untouched; this only wraps it from the dispatcher side.
    """

    def solve(self, all_vehicles, active_vehicles, trips, rejected_trips, current_time=None) -> None:
        """Build a fresh Gurobi model for this optimization step and solve it.

        `all_vehicles` and `current_time` are accepted for interface uniformity with the other
        solvers' `solve(...)` but are not needed by the MIP formulation.
        """
        offline_model = OfflineSolver(self.objective, durations=self.durations, costs=self.costs, weight=self.weight)
        offline_model.offline_solver(active_vehicles, trips, self.vehicle_request_assign, rejected_trips,
                                     mip_gap=OFFLINE_MIP_GAP)
        # A non-zero gap means Gurobi may stop short of proven optimal -- any "true optimal"
        # comparison downstream (Competitive Ratio, benchmark scripts) that assumed
        # this path was always a proven optimum no longer holds exactly; is_optimal tells you which.
        if not offline_model.is_optimal:
            logger.warning(
                "MIP_SOLVER result was not proven optimal (is_optimal=%s) -- any comparison treating "
                "this as the true optimum is now unreliable.", offline_model.is_optimal)


@dataclass
class ObjectiveMetrics:
    """
    Aggregated objective-related metrics for the dispatcher.

    Attributes:
        objective_value: Scalar value of the current optimization objective
            (computed according to config.objective).
        total_customers_served: Total number of customers successfully served.
        total_profit: Total profit (fare minus cost) of served trips
            under the profit-based objectives used in the dispatcher.
        total_waiting_time: Sum of waiting times over all trips (served + rejected),
            measured in seconds.
        total_revenue: Total fare (revenue) of served trips.
        total_cost: Total driving cost associated with served trips
            (e.g. using the same cost model as in profit calculations).
        total_empty_travel_time: Total travel time spent on empty legs
            (e.g. from vehicle position to pickup, and between trips).
        rejected_trips_count: Number of trips that could not be served.
    """
    objective_value: float = 0.0
    total_customers_served: int = 0
    total_profit: float = 0.0
    total_waiting_time: float = 0.0
    total_revenue: float = 0.0
    total_cost: float = 0.0
    total_empty_travel_time: float = 0.0
    rejected_trips_count: int = 0


class MetricsTracker:
    """Owns objective-value/metrics bookkeeping, kept separate from route-plan construction and
    solver selection (see architecture finding #7: TaxiDispatcher previously mixed all of these
    responsibilities together in one class).
    """

    def __init__(self) -> None:
        self.metrics = ObjectiveMetrics()

    def record_served_leg(
            self,
            objective: Objectives,
            fare: float,
            driving_cost: float,
            wait_time: float,
            empty_travel_time: float,
            weight: float,
    ) -> None:
        """Update aggregated metrics and objective_value for one newly-served leg.

        `wait_time` and `empty_travel_time` are both in seconds (matching the units used by
        `_compute_objective_delta`).
        """
        self.metrics.total_customers_served += 1
        self.metrics.total_revenue += fare
        self.metrics.total_cost += driving_cost
        self.metrics.total_profit += fare - driving_cost
        self.metrics.total_waiting_time += wait_time
        self.metrics.total_empty_travel_time += empty_travel_time / 60

        self.metrics.objective_value += self._compute_objective_delta(
            objective, served=True, fare=fare, driving_cost=driving_cost,
            wait_time=wait_time, empty_travel_time=empty_travel_time, weight=weight,
        )

    def record_rejected_trip(self, objective: Objectives, wait_penalty: float, weight: float) -> None:
        """Update aggregated metrics and objective_value for one rejected trip."""
        self.metrics.total_waiting_time += wait_penalty
        self.metrics.objective_value += self._compute_objective_delta(
            objective, served=False, wait_time=wait_penalty, weight=weight,
        )

    @staticmethod
    def _compute_objective_delta(
            objective: Objectives,
            *,
            served: bool,
            fare: float = 0.0,
            driving_cost: float = 0.0,
            wait_time: float = 0.0,
            empty_travel_time: float = 0.0,
            weight: float = 0.5,
    ) -> float:
        """Incremental contribution of a single trip (served or rejected) to the running objective_value.

        Note: this is one of three independent places in the codebase that compute an objective
        value for a given Objectives member, each over a different data representation -- the other
        two are Solver.calc_objective_value (decision-variable based, in solver.py) and
        OfflineSolver.define_objective (Gurobi LinExpr based, in offline_solver.py). They can't be
        unified (different computational models), but if you add/change an objective's semantics
        here, check whether the other two need the equivalent update.

        For a rejected trip (served=False), fare/driving_cost/empty_travel_time default to 0.0, so
        objectives that only make sense for served trips (TOTAL_CUSTOMERS, TOTAL_PROFIT,
        TOTAL_REVENUE, TOTAL_COST, TOTAL_EMPTY_TRAVEL_TIME) correctly contribute nothing.
        """
        if objective == Objectives.TOTAL_CUSTOMERS:
            return 1.0 if served else 0.0
        elif objective == Objectives.WAIT_TIME:
            return wait_time
        elif objective == Objectives.TOTAL_PROFIT:
            return (fare - driving_cost) if served else 0.0
        elif objective == Objectives.TOTAL_REVENUE:
            return fare if served else 0.0
        elif objective == Objectives.TOTAL_COST:
            return driving_cost if served else 0.0
        elif objective == Objectives.TOTAL_EMPTY_TRAVEL_TIME:
            return empty_travel_time if served else 0.0
        elif objective == Objectives.MULTI_OBJECTIVE:
            return weight * (fare - driving_cost) - (1 - weight) * wait_time
        return 0.0


class RoutePlanBuilder:
    """Translates solved vehicle-request assignments into `OptimizedRoutePlan` objects, kept
    separate from solver selection and metrics bookkeeping (see architecture finding #7).
    """

    def __init__(self, network: Any, simulation_config: SimulationConfig, metrics_tracker: MetricsTracker,
                 create_route_plan_fn) -> None:
        self.network = network
        self.simulation_config = simulation_config
        self.metrics_tracker = metrics_tracker
        # Bound Dispatcher._create_route_plan (inherited by TaxiDispatcher) -- constructing
        # OptimizedRoutePlan through it (rather than calling OptimizedRoutePlan(route) directly)
        # is what populates Dispatcher._routes_by_vehicle_id and respects the state_deep_copy/
        # copy_route_objects contract; see the multimodalsim 0.0.2 migration plan, item A2.
        self.create_route_plan_fn = create_route_plan_fn

    def build_route_plans(self, veh_trips_assignments_list, next_leg_by_trip_id, current_time, state):
        """
            Function: Constructs a list of optimized route plans based on vehicle assignments and current state.

                Input:
                ------------
                veh_trips_assignments_list: A list of VehicleState objects, each representing a
                    vehicle's assigned trips and its last stop.
                next_leg_by_trip_id: A dictionary mapping trip IDs to their corresponding next legs.
                current_time: The current time of the simulation.
                state: The current state of the environment, containing information about vehicles and routes.

                Output:
                ------------
                route_plans_list : A list of OptimizedRoutePlan instances, each representing an optimized route for a vehicle.
        """
        route_plans_list = []
        for veh_trips_assignment in veh_trips_assignments_list:
            trip_ids = [trip.id for trip in veh_trips_assignment.assigned_requests]

            route = state.route_by_vehicle_id[veh_trips_assignment.vehicle.id]
            if self.simulation_config.solution_mode == SolutionMode.OFFLINE or len(route.next_stops) <= 1:
                route_plan = self._build_route_plan(
                    route,
                    trip_ids,
                    veh_trips_assignment.departure_stop,
                    next_leg_by_trip_id, current_time)
                route_plans_list.append(route_plan)

        return route_plans_list

    def _build_route_plan(self, route, trip_ids, departure_stop_id,
                           next_leg_by_trip_id, current_time):
        """
            Function: Creates an optimized route plan for a vehicle based on assigned trips and current state.

                Input:
                ------------
                route: The current route of the vehicle.
                trip_ids: A list of trip IDs assigned to the vehicle.
                departure_stop_id: The ID of the location from which the vehicle will depart.
                next_leg_by_trip_id: A dictionary mapping trip IDs to their corresponding next legs.
                current_time: The current time of the simulation.

                Output:
                ------------
                OptimizedRoutePlan : An optimized route plan for the vehicle.
        """

        route_plan = self.create_route_plan_fn(route)
        # Read/mutate through route_plan.route (route_copy), not the raw `route` parameter: under
        # state_deep_copy=False, create_route_plan_fn deep-copies `route` internally, so `route_copy`
        # and `route` are different objects -- mutating `route` directly would silently bypass the
        # plan and write untracked to the live simulation route. Under state_deep_copy=True they're
        # the same object (no copy is made), so this is a no-op there. Mirrors
        # multimodalsim's own ShuttleGreedyDispatcher.__create_route_plan.
        route_copy = route_plan.route

        if len(route_copy.next_stops) == 0:
            # If the current route has no stops, update the departure time of the current stop to the current time.
            last_stop = route_copy.previous_stops[-1] if route_copy.current_stop is None else route_copy.current_stop
            if last_stop.departure_time < current_time or last_stop.departure_time == math.inf:
                last_stop.departure_time = current_time
            departure_time = last_stop.departure_time
            route_plan.update_current_stop_departure_time(departure_time)
        else:
            # If there are existing stops, set the departure time of the last stop to its arrival time.
            route_copy.next_stops[-1].departure_time = route_copy.next_stops[-1].arrival_time
            departure_time = route_copy.next_stops[-1].departure_time
            route_plan.copy_route_stops()


        for index, trip_id in enumerate(trip_ids):
            if self.simulation_config.solution_mode != SolutionMode.OFFLINE and len(route_plan.assigned_legs) > 0:
                break

            leg = next_leg_by_trip_id[trip_id]

            # Calculate and add pick-up stop.
            travel_time_to_pick = self.network.nodes[departure_stop_id]['shortest_paths'][
                leg.trip.origin.label]['total_duration']
            arrival_time = departure_time + travel_time_to_pick
            if arrival_time < leg.trip.ready_time:
                if self.simulation_config.solution_mode != SolutionMode.OFFLINE:
                    break
                # If the vehicle arrives earlier than the ready time, adjust departure to align with the ready time.
                if len(route_plan.next_stops) == 0:
                    route_plan.update_current_stop_departure_time(current_time + leg.trip.ready_time - arrival_time)
                else:
                    route_plan.next_stops[-1].departure_time += leg.trip.ready_time - arrival_time
                arrival_time = leg.trip.ready_time
            departure_time = arrival_time
            planned_stops = route_plan.next_stops
            last_planned_stop = planned_stops[-1] if planned_stops else None
            if (last_planned_stop is not None
                    and last_planned_stop.location.label == leg.trip.origin.label
                    and last_planned_stop.arrival_time == arrival_time):
                # The vehicle is already stopped at this node at this instant (the previous leg's
                # drop-off). Board here instead of appending a second, zero-duration stop at the same
                # node and time: the simulator collapses such a pair as the vehicle passes through,
                # which shifts every later stop by one position and breaks the viewer's positional
                # stop-list diffing.
                last_planned_stop.departure_time = departure_time
                route_plan.assign_legs_to_board_to_stop([leg], last_planned_stop)
            else:
                route_plan.append_next_stop(leg.trip.origin.label, arrival_time, departure_time,
                                             lon=leg.trip.origin.lon, lat=leg.trip.origin.lat,
                                             legs_to_board=[leg])
            route_plan.assign_leg(leg)
            # revenue and cost for this served trip
            driving_cost = compute_driving_cost(leg.trip.shortest_travel_time, travel_time_to_pick)
            # waiting time and empty travel time are both in seconds
            wait_time = (arrival_time - leg.trip.ready_time)

            w = self.simulation_config.weight
            self.metrics_tracker.record_served_leg(
                self.simulation_config.objective, fare=leg.trip.fare, driving_cost=driving_cost,
                wait_time=wait_time, empty_travel_time=travel_time_to_pick, weight=w,
            )

            # Calculate and add drop-off stop.
            arrival_time = departure_time + leg.trip.shortest_travel_time
            departure_time = arrival_time if index != len(trip_ids) - 1 else math.inf
            route_plan.append_next_stop(leg.trip.destination.label, arrival_time, departure_time,
                                         lon=leg.trip.destination.lon, lat=leg.trip.destination.lat, legs_to_alight=[leg])
            departure_stop_id = leg.trip.destination.label

        return route_plan


class TaxiDispatcher(Dispatcher):
    """Optimize the vehicle routing and the trip-route assignment. This method relies on three other methods:
            1. prepare_input
            2. optimize
            3. create_route_plans_list

        Attributes:
        ------------
        network: Network
            The transport network over which the dispatching occurs.
        simulation_config : SimulationConfig
            Configuration object containing all simulation parameters.
        rejected_trips: list
             an array of rideRequests that we are not able to serve them while meeting constraints
        metrics_tracker: MetricsTracker
            Collaborator owning objective-value/metrics bookkeeping (see architecture finding #7).
        route_plan_builder: RoutePlanBuilder
            Collaborator translating solved vehicle-request assignments into OptimizedRoutePlan objects.
        runtime : Timer
            Timer to keep track on the time to optimize the solution
        current_solution: Dict
            the current solution for re-optimization purpose
        solver: solver object
            the solver class including the optimizing functions


    """

    def __init__(self,
                 network: Any,
                 vehicles: List[Any],
                 simulation_config: SimulationConfig):
        """
        Call the constructor

        Input:
        ------------
        network: The transport network over which the dispatching occurs.
        vehicles: Set of input vehicles
        simulation_config : Configuration object containing all simulation parameters.
        """

        super().__init__()
        self.network = network
        self.simulation_config = simulation_config
        self.rejected_trips: List[Any] = []
        self.metrics_tracker = MetricsTracker()
        self.current_solution = None
        self.runtime = Timer()

        # Compute the network's travel-time/cost matrices once and share them with every solver
        # instance instead of letting each one recompute them independently (see architecture
        # finding #6: durations/costs were previously recomputed per Solver/OfflineSolver instance,
        # including once per scenario inside StochasticSolver/ReOptimizer).
        self.durations = get_durations(network)
        self.costs = get_costs(network)

        # Initialize the appropriate solver based on the algorithm
        self.solver = self._initialize_solver(vehicles)
        self.route_plan_builder = RoutePlanBuilder(network, simulation_config, self.metrics_tracker,
                                                    self._create_route_plan)

    def _initialize_solver(
            self,
            vehicles: List[Any],
    ) -> Any:
        """
        Factory method to initialize the appropriate solver based on the algorithm.

        Parameters:
        -----------
        vehicles: List[Any]
            List of input vehicles.

        Returns:
        --------
        Any
            An instance of the appropriate solver class.
        """
        algorithm = self.simulation_config.algorithm
        common_kwargs = dict(
            network=self.network,
            vehicles=vehicles,
            simulation_config=self.simulation_config,
            durations=self.durations,
            costs=self.costs,
        )

        if algorithm == Algorithm.MIP_SOLVER:
            return GurobiSolverAdapter(**common_kwargs)
        elif algorithm == Algorithm.CONSENSUS:
            return StochasticSolver(**common_kwargs)
        elif algorithm == Algorithm.RE_OPTIMIZE:
            return ReOptimizer(**common_kwargs)
        elif algorithm == Algorithm.DQN:
            return DQNSolver(**common_kwargs)
        else:
            return OnlineSolver(**common_kwargs)

    def __str__(self) -> str:
        """Provide a string representation of the TaxiDispatcher."""
        metrics = self.metrics_tracker.metrics
        return (
            f"\nNumber of Rejected Trips: {len(self.rejected_trips)}\n"
            f"Objective value: {metrics.objective_value}\n"
            f"Total Number of served customers: {metrics.total_customers_served}\n"
            f"Total Profit: {metrics.total_profit}\n"
            f"Total Waiting time: {metrics.total_waiting_time}\n"
        )


    def prepare_input(self, state: Any) -> tuple[List[Any], List[Any]]:
        """ Function: Extract from the state the next legs and the routes that are sent as
            input to the optimize method (i.e. the legs and the routes that
            you want to optimize).

            All next legs and all routes existing in the environment at
            the time of optimization will be optimized.

            Input:
            ------------
                state: An object of type State that corresponds to a partial deep
                    copy of the environment.

            Output:
            ------------
                selected_next_legs: A list of objects of type Trip that correspond
                    to the trips (i.e., passengers or requests) that should be
                    considered by the optimize method.

                selected_routes: A list of objects of type Route that correspond
                    to the routes associated with the vehicles that
                    should be considered by the optimize method.

                vehicle_request_assign: Dict mapping vehicle IDs to VehicleState objects.

            Note that if selected_next_legs or selected_routes is empty, no optimization will be done.
            """

        selected_route = []
        self.rejected_trips = [
            leg.trip for leg in state.non_assigned_next_legs
            if leg.trip.latest_pickup < state.current_time
        ]
        rejected_ids = {trip.id for trip in self.rejected_trips}

        # remove rejected trips from the list of non-assigned trips
        selected_next_legs = [
            leg for leg in state.non_assigned_next_legs
            if leg.trip.id not in rejected_ids
        ]

        if selected_next_legs:
            for vehicle in state.vehicles:
                route = state.route_by_vehicle_id[vehicle.id]
                if self.simulation_config.algorithm != Algorithm.CONSENSUS or len(route.next_stops) <= 1:
                    selected_route.append(route)
        current_routes = [state.route_by_vehicle_id[vehicle.id] for vehicle in state.vehicles]
        self.solver.update_vehicle_state(current_routes, state.current_time)
        return selected_next_legs, selected_route

    def optimize(
            self,
            selected_next_legs: List[Any],
            selected_routes: List[Any],
            current_time: int,
            state: Any
    ) -> tuple[List[OptimizedRoutePlan], None]:

        """
        Function: Determine the vehicle routing and the trip-route assignment
            according to an optimization algorithm. The optimization algorithm
            should be called in this method.

            Input:
            ------------
                selected_next_legs: List[Any]
                    List of the next legs to be optimized.
                selected_routes: List[Any]
                    List of the routes to be optimized.
                current_time: int
                    Current time of the simulation.
                state: Any
                An object of type State that corresponds to a partial deep copy of the environment.

            Output:
            ------------
                A tuple of (optimized_route_plans, additional_optimization_results). optimized_route_plans
                is a list of OptimizedRoutePlan objects; additional_optimization_results is always None
                here since we have no extra modified/new requests or vehicles beyond what the route plans
                themselves already report (required 2-tuple contract, see multimodalsim 0.0.2's
                Dispatcher.dispatch()).
        """
        self.runtime.start()

        vehicles = [route.vehicle for route in selected_routes]
        # non-assigned requests
        trips = [leg.trip for leg in selected_next_legs]
        trips = sorted(trips, key=lambda x: x.ready_time)
        next_leg_by_trip_id = {leg.trip.id: leg for leg in selected_next_legs}

        # Uniform entry point across all four algorithms (see architecture finding #1) -- each
        # solver's solve(...) internally does whatever variable-declaration/model-building it needs.
        all_vehicles = [vehicle_state.vehicle for vehicle_state in self.solver.vehicle_request_assign.values()]
        self.solver.solve(all_vehicles, vehicles, trips, self.rejected_trips, current_time)

        veh_trips_assignments_list = list(self.solver.vehicle_request_assign.values())
        # remove the vehicles without any changes in request-assign
        veh_trips_assignments_list = [
            state for state in veh_trips_assignments_list if state.assigned_requests
        ]
        route_plans_list = self.route_plan_builder.build_route_plans(
            veh_trips_assignments_list, next_leg_by_trip_id, current_time, state)
        self.runtime.stop()
        return route_plans_list, None

    def extract_output(self):
        """
            Function: Extracts and summarizes output information regarding the dispatch operation.

                Output:
                ------------
                output_dict: A dictionary containing details about the algorithm used, the optimization objective,
                    the objective value, the number of served customers, and the percentage of service.
                """
        metrics = self.metrics_tracker.metrics
        total_trips = metrics.total_customers_served + len(self.rejected_trips)
        for trip in self.rejected_trips:
            wait_penalty = (trip.latest_pickup - trip.ready_time)
            w = self.simulation_config.weight
            self.metrics_tracker.record_rejected_trip(self.simulation_config.objective, wait_penalty, w)

        percentage_service = (metrics.total_customers_served / total_trips * 100) if total_trips > 0 else 0
        percentage_service = round(percentage_service, 1)

        avg_profit = metrics.total_profit / metrics.total_customers_served if metrics.total_customers_served > 0 else 0.0
        avg_wait = metrics.total_waiting_time / total_trips if total_trips > 0 else 0.0
        avg_revenue = metrics.total_revenue / metrics.total_customers_served if metrics.total_customers_served > 0 else 0.0

        output_dict = {
            col.ALGORITHM: self.simulation_config.algorithm.value,
            col.OBJECTIVE_TYPE: self.simulation_config.objective.value,
            col.OBJECTIVE_VALUE: round(metrics.objective_value, 2),
            col.NUM_SERVED: metrics.total_customers_served,
            col.NUM_REJECTED: len(self.rejected_trips),
            # Profit-related metrics
            col.AVG_PROFIT: round(avg_profit, 2),
            col.TOTAL_PROFIT: round(metrics.total_profit, 2),
            # Revenue and cost metrics
            col.AVG_REVENUE: round(avg_revenue, 2),
            col.TOTAL_REVENUE: round(metrics.total_revenue, 2),
            col.TOTAL_COST: round(metrics.total_cost, 2),
            # Time metrics
            col.TOTAL_WAIT_TIME_S: round(metrics.total_waiting_time, 2),
            col.AVG_WAIT_TIME_S: round(avg_wait, 2),
            col.EMPTY_TRAVEL_TIME_MIN: round(metrics.total_empty_travel_time, 2),
            # Service and runtime
            col.PCT_SERVICE: percentage_service,
            col.RUNTIME_S: round(self.runtime.elapsed_since_init(), 3),
        }
        return output_dict
