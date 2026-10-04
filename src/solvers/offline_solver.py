import logging
from typing import Any, Dict, List, Optional, Tuple
import gurobipy as gp
from gurobipy import GRB

from src.utilities.enums import Objectives
from src.solvers.solver import VehicleState
from src.utilities.types import TripLike, VehicleLike, TravelTimeMatrix

GUROBI_BINARY_THRESHOLD = 0.5  # Threshold for reading a Gurobi binary variable's solved value as True/False


class OfflineSolver:
    """
    A class to solve the taxi routing problem using a MIP solver (Gurobi).

        Attributes:
        ------------
        objective: Objectives(Enum)
            The objective used to evaluate the effectiveness of the plan
        duration : dictionary
            travel time matrix between possible stop points
        costs: dictionary
            driving costs
        model: gp.Model
            The Gurobi model for optimization.
        X_var: Dict[Tuple[int, int], gp.Var]
            Decision variables for trip connections.
        Y_var: Dict[Tuple[int, int], gp.Var]
            Decision variables for assigning trips to vehicles.
        Z_var: Dict[int, gp.Var]
            Decision variables indicating if a trip is served.
        U_var: Dict[int, gp.Var]
            Decision variables for departure times from locations.
        """

    def __init__(self,
                 objective: Objectives,
                 durations: TravelTimeMatrix,
                 costs: TravelTimeMatrix,
                 weight: float = 0.5
                 ) -> None:
        """
        Initialize the OfflineSolver.

        Input:
        ------------
        objective: Objectives
            The objective used to evaluate the effectiveness of the plan.
        durations: Travel time matrix between stop points, computed once (e.g. by TaxiDispatcher)
            and shared across solver instances rather than recomputed here.
        costs: Driving cost matrix, computed once and shared the same way as durations.
        weight: float, optional
            Used only for MULTI_OBJECTIVE objective (ignored otherwise). Default 0.5.
        """
        self.objective = objective
        self.weight = weight

        self.objective_value = 0
        self.is_optimal: Optional[bool] = None  # set by solve(): True if proven optimal, False if
                                                 # accepted from a time/gap-limited incumbent instead
        self.durations: TravelTimeMatrix = durations
        self.costs: TravelTimeMatrix = costs

        self.model = gp.Model("TaxiRoutingModel")
        # Set OutputFlag based on the logging level
        if logging.getLogger().getEffectiveLevel() > logging.INFO:
            self.model.setParam('OutputFlag', 0)  # Disable solver output
        else:
            self.model.setParam('OutputFlag', 1)  # Enable solver output
        self.model.setParam('Threads', 1)
        self.X_var: Dict[Tuple[int, int], gp.Var] = {}  
        self.Y_var: Dict[Tuple[int, int], gp.Var] = {}  
        self.Z_var: Dict[int, gp.Var] = {}  
        self.U_var: Dict[int, gp.Var] = {}  

    def define_objective(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        """
        Define the objective function based on the selected objective.

        Note: this is one of three independent places in the codebase that compute an objective
        value for a given Objectives member, each over a different data representation -- the other
        two are Solver.calc_objective_value (decision-variable based, in solver.py) and
        TaxiDispatcher._compute_objective_delta (route/leg based, in taxi_dispatcher.py). They can't
        be unified (different computational models), but if you add/change an objective's semantics
        here, check whether the other two need the equivalent update.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.
        """
        if self.objective == Objectives.TOTAL_PROFIT:
            self.define_total_profit_objective(K, P, vehicle_request_assign)
        elif self.objective == Objectives.TOTAL_CUSTOMERS:
            self.define_total_customers_objective(P)
        elif self.objective == Objectives.WAIT_TIME:
            self.define_total_wait_time_objective(P)
        elif self.objective == Objectives.MULTI_OBJECTIVE:
            self.define_multi_objective(K, P, vehicle_request_assign)
        elif self.objective == Objectives.TOTAL_REVENUE:
            self.define_total_revenue_objective(P)
        elif self.objective == Objectives.TOTAL_COST:
            self.define_total_cost_objective(K, P, vehicle_request_assign)
        elif self.objective == Objectives.TOTAL_EMPTY_TRAVEL_TIME:
            self.define_total_empty_objective(K, P, vehicle_request_assign)
        else:
            raise ValueError(f"Objective {self.objective} not recognized.")

    def define_total_customers_objective(self, P: List[TripLike]) -> None:
        """
        Define objective of maximizing the total number of served customers and add it to the model.

        Input:
        ------------
        P: List[Any]
            Set of customers to serve.
        """
        self.model.setObjective(
            sum(self.Z_var[f_i.id] for f_i in P),
            sense=GRB.MAXIMIZE
        )

    def define_total_profit_objective(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        """
        Define objective of maximizing the total profit and add it to the model.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.

        Hint:
        ------------
        - Access costs via self.costs[from_location][to_location]
        - Access vehicle departure stop via vehicle_request_assign[vehicle_id].departure_stop

        """

        raise NotImplementedError("OfflineSolver.define_total_profit_objective() not implemented")


    def define_total_wait_time_objective(self, P: List[TripLike]) -> None:
        """
        Define objective of minimizing the total wait time and add it to the model.

        Input:
        ------------
        P: List[Any]
            Set of customers to serve.

        Hint:
        ------------
        - Wait time is in seconds already -- no conversion needed.

        """

        raise NotImplementedError("OfflineSolver.define_total_wait_time_objective() not implemented")

    def define_multi_objective(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        """
        Define weighted combined objective: maximize total profit and minimize wait time.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.

        Hint:
        ------------
        - Access costs via self.costs[from_location][to_location]
        - Access vehicle departure stop via vehicle_request_assign[vehicle_id].departure_stop
        - Uses single weight w: objective = w * (total profit) - (1 - w) * (total wait time).
        - Wait time is in seconds already -- no conversion needed.

        """

        raise NotImplementedError("OfflineSolver.define_multi_objective() not implemented")

    def define_total_revenue_objective(self, P: List[TripLike]) -> None:
        """
        Define objective of maximizing the total revenue (fares of served trips) and add it to the model.

        Input:
        ------------
        P: List[Any]
            Set of customers to serve.

        """

        raise NotImplementedError("OfflineSolver.define_total_revenue_objective() not implemented")

    def define_total_cost_objective(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        """
        Define objective of minimizing the total driving cost and add it to the model.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.

        Hint:
        ------------
        - Access costs via self.costs[from_location][to_location]
        - Access vehicle departure stop via vehicle_request_assign[vehicle_id].departure_stop
        """

        raise NotImplementedError("OfflineSolver.define_total_cost_objective() not implemented")

    def define_total_empty_objective(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        """
        Define objective of minimizing total empty travel time (empty-leg durations) and add it to the model.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.

        Hint:
        ------------
        - Access durations via self.durations[from_location][to_location]
        - Access vehicle departure stop via vehicle_request_assign[vehicle_id].departure_stop

        """

        raise NotImplementedError("OfflineSolver.define_total_empty_objective() not implemented")


    def create_model(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        """
        Create model to solve with Gurobi Solver.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.
        """
        self._declare_variables(K, P, vehicle_request_assign)
        self.model.update()

        self._add_constraint_1(K, P)
        self._add_constraint_2(P)
        self._add_constraint_3(K, P)
        self._add_constraint_4(P)
        self._add_constraint_5(P)
        self._add_constraint_6(K, P, vehicle_request_assign)
        self.model.update()

    def _declare_variables(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        """Declare the U, Z, X, and Y Gurobi decision variables.
        """
        for f_i in P:
            self.U_var[f_i.id] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f'U_{f_i.id}')
            self.Z_var[f_i.id] = self.model.addVar(vtype=GRB.BINARY, obj=0, name=f'Z_{f_i.id}')
            for f_j in P:
                if f_i == f_j:
                    continue
                self.X_var[f_i.id, f_j.id] = self.model.addVar(vtype=GRB.BINARY, name=f'X_{f_i.id}_{f_j.id}')

        for f_i in P:
            for f_k in K:
                self.Y_var[f_k.id, f_i.id] = self.model.addVar(vtype=GRB.BINARY, name=f'Y_{f_k.id}_{f_i.id}')

    def _add_constraint_1(self, K: List[VehicleLike], P: List[TripLike]) -> None:
        for f_i in P:
            self.model.addConstr(
                self.Z_var[f_i.id] == sum(self.Y_var[f_k.id, f_i.id] for f_k in K)
                + sum(self.X_var[f_j.id, f_i.id] for f_j in P if f_j != f_i),
                name=f"Constraint1_{f_i.id}"
            )

    def _add_constraint_2(self, P: List[TripLike]) -> None:
        for f_i in P:
            self.model.addConstr(
                self.Z_var[f_i.id] >= sum(self.X_var[f_i.id, f_j.id] for f_j in P if f_j != f_i),
                name=f"Constraint2_{f_i.id}"
            )

    def _add_constraint_3(self, K: List[VehicleLike], P: List[TripLike]) -> None:
        for f_k in K:
            self.model.addConstr(
                sum(self.Y_var[f_k.id, f_i.id] for f_i in P) <= 1,
                name=f"Constraint3_{f_k.id}"
            )

    def _add_constraint_4(self, P: List[TripLike]) -> None:
        for f_i in P:
            self.model.addConstr(self.U_var[f_i.id] >= f_i.ready_time, name=f"Constraint4a_{f_i.id}")
            self.model.addConstr(self.U_var[f_i.id] <= f_i.latest_pickup, name=f"Constraint4b_{f_i.id}")

    def _add_constraint_5(self, P: List[TripLike]) -> None:
        for f_i in P:
            for f_j in P:
                if f_i == f_j:
                    continue
                T_ij = f_i.shortest_travel_time + self.durations[f_i.destination.label][f_j.origin.label]
                delta = f_j.ready_time - f_i.latest_pickup
                self.model.addConstr(
                    self.U_var[f_j.id] - self.U_var[f_i.id] >= delta + self.X_var[f_i.id, f_j.id] * (T_ij - delta),
                    name=f"Constraint5_{f_i.id}_{f_j.id}"
                )

    def _add_constraint_6(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        for f_i in P:
            for f_k in K:
                state = vehicle_request_assign[f_k.id]
                T_ki = self.durations[state.departure_stop][f_i.origin.label]
                delta = state.departure_time + T_ki - f_i.ready_time
                self.model.addConstr(
                    self.U_var[f_i.id] >= f_i.ready_time + delta * self.Y_var[f_k.id, f_i.id],
                    name=f"Constraint6_{f_i.id}_{f_k.id}"
                )

    def solve(self, time_limit: Optional[float] = None, mip_gap: Optional[float] = None) -> None:
        """
        Optimize the model using Gurobi.

        Input:
        ------------
        time_limit: float, optional
            Wall-clock seconds to allow Gurobi to search before stopping, even without proof of
            optimality. None (default) means no limit -- run to a proven optimum (or infeasibility).
        mip_gap: float, optional
            Relative optimality gap (e.g. 0.05 for 5%) at which Gurobi may stop early even without a
            time limit. None (default) uses Gurobi's own default gap tolerance (effectively proof of
            optimality).

        Note:
        ------------
        Updates self.objective_value and self.is_optimal with the result. Accepts the best incumbent
        found whenever one exists (self.model.SolCount > 0), regardless of whether it's a proven
        optimum or a time/gap-limited stop -- check self.is_optimal to tell which. Only raises if no
        feasible solution was found at all.
        """
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit)
        if mip_gap is not None:
            self.model.setParam('MIPGap', mip_gap)

        self.model.optimize()

        if self.model.SolCount == 0:
            raise RuntimeError(
                f"Gurobi solver found no feasible solution (status={self.model.status})."
            )
        # GRB.OPTIMAL only means Gurobi met its termination criteria, which includes stopping
        # on MIPGap -- so the achieved bound has to be checked too for this to mean "proven optimal".
        self.is_optimal = (
            self.model.status == GRB.OPTIMAL
            and abs(self.model.ObjBound - self.model.objVal) <= 1e-6 * max(1.0, abs(self.model.objVal))
        )
        self.objective_value = round(self.model.objVal, 3)

    def extract_solution(self, K: List[VehicleLike], P: List[TripLike], rejected_trips: List[TripLike], vehicle_request_assign: Dict[int, VehicleState]) -> None:
        """
        Extract the solution from the optimized model and convert it to vehicle_request_assign format.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        rejected_trips: List[Any]
            List of trips that are rejected in the optimization process (will be populated by this method).
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects (will be populated with assignments by this method).

        Note:
        ------------
        - Adds trips to vehicle_request_assign[vehicle_id].assigned_requests
        - Adds unserved trips (Z_var < 0.5) to rejected_trips
        """

        # Extract the solution and populate the vehicle_request_assign and rejected_trips
        for f_k in K:
            for trip in P:
                if self.Y_var[f_k.id, trip.id].X > GUROBI_BINARY_THRESHOLD:
                    state = vehicle_request_assign[f_k.id]
                    state.assigned_requests.append(trip)
                    current_trip = trip
                    while True:
                        next_trip_found = False
                        for f_j in P:
                            if f_j == current_trip:
                                continue
                            if self.X_var[current_trip.id, f_j.id].X > GUROBI_BINARY_THRESHOLD:
                                state.assigned_requests.append(f_j)
                                current_trip = f_j
                                next_trip_found = True
                                break
                        if not next_trip_found:
                            break

        for trip in P:
            if self.Z_var[trip.id].X < GUROBI_BINARY_THRESHOLD:
                rejected_trips.append(trip)

    def offline_solver(self, K: List[VehicleLike], P: List[TripLike], vehicle_request_assign: Dict[int, VehicleState], rejected_trips: List[TripLike], time_limit: Optional[float] = None, mip_gap: Optional[float] = None) -> None:
        """
        Solve the taxi routing problem using a MIP solver (Gurobi).

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.
        rejected_trips: List[Any]
            List of trips that are rejected in the optimization process (will be populated by this method).
        time_limit: float, optional
            Passed straight through to solve() -- see its docstring. None (default) means unlimited.
        mip_gap: float, optional
            Passed straight through to solve() -- see its docstring. None (default) means unlimited.
        """
        self.create_model(K, P, vehicle_request_assign)
        self.define_objective(K, P, vehicle_request_assign)
        self.solve(time_limit=time_limit, mip_gap=mip_gap)
        self.extract_solution(K, P, rejected_trips, vehicle_request_assign)

