import logging
import os
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utilities.config import SimulationConfig
from src.utilities.enums import Objectives, SolutionMode
from src.utilities.network_tools import build_node_zone_mapping, compute_leg_profit
from src.utilities.paths import REAL_INSTANCES_FOLDER
from src.solvers.solver import Solver, VehicleState
from src.utilities.types import TripLike, VehicleLike, TravelTimeMatrix

logger = logging.getLogger(__name__)

# Fixed normalization constants -- NOT derived per-instance, so feature scale stays reproducible
# across episodes/instances. Chosen to squash this project's calibrated fare/duration ranges into
# roughly [-2, 2]; adjust here if alpha/beta/COST_PER_MINUTE/time_window are retuned.
TIME_NORM = 1800.0  # seconds
FARE_NORM = 20.0    # dollars
COST_NORM = 10.0    # dollars
FLEET_COUNT_NORM = 100.0  # divides n_pending/n_available -- squashes fleet-wide counts toward O(1)
ZONE_COUNT_NORM = 20.0    # divides per-zone pending/available counts -- same purpose, smaller
                          # typical magnitude per zone

# Per-candidate local feature count -- set this to however many values _local_features returns.
# _feature_dim() derives the network's input_dim from it, so this is the single place to change it.
_N_LOCAL_FEATURES = 0  # TODO


class QNetwork(nn.Module):
    """Scores one candidate (vehicle, trip) pair -- or the DEFER pseudo-candidate -- from its
    feature vector (local features concatenated with the round's shared context block, see
    DQNSolver._local_features / _build_context_features).

    Contract:
    ------------
        __init__(input_dim) : input_dim is DQNSolver._feature_dim(), i.e. the width of one row of
            the feature matrix.
        forward(x) : x has shape (n_rows, input_dim); the return must have shape (n_rows,), one
            scalar score per row.

    Design it yourself: the architecture is up to you.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        raise NotImplementedError("QNetwork.__init__() not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QNetwork.forward() not implemented")


class ReplayBuffer:
    """Fixed-capacity FIFO buffer of (features, reward, next_candidate_features, done) transitions.

    `next_candidate_features` is the FULL list of feature vectors available at the next decision
    point (not just the one that gets chosen there) -- training takes max over the target network's
    evaluation of all of them, standard DQN Bellman-optimality bootstrapping.
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, features: List[float], reward: float,
             next_candidate_features: Optional[List[List[float]]], done: bool) -> None:
        self.buffer.append((features, reward, next_candidate_features, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNSolver(Solver):
    """Online solver that scores every currently-feasible (vehicle, trip) pair with a learned
    Q-network, commits the single best-scoring pair (or defers), then rescores the shrunken
    remaining set and repeats -- all within one solve() call, since the simulation engine cannot
    pause mid-round.

    One decision = one pass of dqn_assign's loop, split into five steps, one method each:

        _build_candidates       which (vehicle, trip) pairs are eligible right now
        _build_feature_matrix   what the network sees: context + per-candidate locals + DEFER
        _select_action          epsilon-greedy over those scores, DEFER included
        _snapshot_last_seen     per-trip bookkeeping for the optional expiry penalty
        _apply_expiry_penalties penalty transitions for trips that expired since last round

    Attributes:
    ------------
        vehicle_request_assign : Dict[int, VehicleState]
            Mapping vehicle id to VehicleState (inherited from Solver) -- same fields as
            OnlineSolver's: vehicle, assigned_requests, departure_stop, departure_time, last_stop,
            last_stop_time, assign_possible, random_number (unused here).
        durations, costs : dictionary
            Travel time / driving cost matrices between stop points (inherited from Solver).
        online_net, target_net : QNetwork
            The network being trained/used for action selection, and its periodically-synced target
            copy (standard DQN stabilization).
        replay_buffer : ReplayBuffer
            Transitions collected during training_mode, consumed by gradient steps as the episode
            progresses (see _maybe_train_step). Also receives a penalty transition when a trip
            expires unserved, if expiry_penalty_lambda is non-zero (see _apply_expiry_penalties).
            Persists across episodes when simulation_config.dqn.reuse_solver is set.
        _last_seen_features, _last_seen_reward : Dict[str, Any]
            Per-trip snapshot of its best-scoring pair from the last round it was a live candidate
            (see _snapshot_last_seen) -- what an expiry penalty is attributed to. Reset each episode.
        _previously_rejected_ids : set
            Trip ids already seen in a prior round's rejected_trips, diffed each round to isolate
            the ones that newly expired. Reset each episode.
    """

    def __init__(self,
                 network: Any,
                 vehicles: List[VehicleLike],
                 simulation_config: SimulationConfig,
                 durations: TravelTimeMatrix,
                 costs: TravelTimeMatrix):
        super().__init__(network, vehicles, simulation_config, durations, costs)
        dqn_config = simulation_config.dqn

        self._training = dqn_config.training_mode
        self._epsilon = dqn_config.epsilon
        self._defer_explore_prob = dqn_config.defer_explore_prob
        self._gamma = dqn_config.gamma
        self._target_sync_every = dqn_config.target_sync_every
        self._batch_size = dqn_config.batch_size
        self._model_path = dqn_config.model_path
        self._rng = random.Random(dqn_config.seed)
        self._expiry_penalty_lambda = dqn_config.expiry_penalty_lambda

        _SUPPORTED_OBJECTIVES = (Objectives.TOTAL_PROFIT, Objectives.TOTAL_CUSTOMERS, Objectives.WAIT_TIME)
        if self.objective not in _SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"DQNSolver does not support objective={self.objective!r} -- only "
                f"{[o.value for o in _SUPPORTED_OBJECTIVES]} are implemented in _compute_commit_reward.")

        # Zone membership isn't retained on TripLike/VehicleLike at runtime (only baked into the
        # generation-time CSV) -- derive it the same way create_instance.py does at generation time,
        # from each node's (lon, lat) against the same real zone geometry.
        self._node_zone_mapping: Dict[str, Any] = build_node_zone_mapping(
            network, os.path.join(REAL_INSTANCES_FOLDER, "manhattan-geo.json"))
        self._zone_ids = sorted({z for z in self._node_zone_mapping.values() if z is not None})
        self._zone_index = {zone_id: i for i, zone_id in enumerate(self._zone_ids)}

        # Fixed episode horizon, for the time-to-go local feature (see _local_features).
        # Recomputed fresh every episode from *this* episode's own vehicles -- not reused via
        # reuse_solver, since different instances could in principle have different horizons.
        self._episode_end_time = max((v.end_time for v in vehicles if v.end_time is not None), default=None)

        # The network's input width is whatever the feature functions produce -- _feature_dim() is
        # the single declaration of that, so changing the features never means also hunting down an
        # arithmetic expression here. _build_feature_matrix cross-checks it against reality.
        input_dim = self._feature_dim()

        reuse_solver = dqn_config.reuse_solver
        if reuse_solver is not None:
            # Reuse the trainable state as-is (same objects, not copies) so the replay buffer,
            # optimizer momentum, and gradient-step count persist across episodes.
            assert reuse_solver.online_net.net[0].in_features == input_dim, (
                f"reuse_solver has input_dim={reuse_solver.online_net.net[0].in_features}, but this "
                f"episode's network needs input_dim={input_dim} -- instances must share the same "
                f"graph/zone geometry to reuse a solver across episodes.")
            self.online_net = reuse_solver.online_net
            self.target_net = reuse_solver.target_net
            self.optimizer = reuse_solver.optimizer
            self.replay_buffer = reuse_solver.replay_buffer
            self._gradient_steps = reuse_solver._gradient_steps
            self._episode_count = reuse_solver._episode_count
            # Sever the link once its pieces are borrowed, so past episodes' solvers can be freed.
            dqn_config.reuse_solver = None
        else:
            self.online_net = QNetwork(input_dim)
            self.target_net = QNetwork(input_dim)
            if self._model_path and os.path.isfile(self._model_path):
                self.online_net.load_state_dict(torch.load(self._model_path))
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.online_net.parameters(), lr=dqn_config.learning_rate)
            self.replay_buffer = ReplayBuffer(dqn_config.replay_buffer_size)
            self._gradient_steps = 0
            self._episode_count = 0

        # Per-episode bookkeeping for the in-band expiry penalty -- trip ids are only meaningful
        # within one episode's instance, so these are always fresh here, never carried across
        # episodes via reuse_solver (unlike online_net/replay_buffer/etc. above).
        self._last_seen_features: Dict[str, Any] = {}
        self._last_seen_reward: Dict[str, float] = {}
        self._previously_rejected_ids: set = set()

        # The one transition still waiting to learn its "next state" (the next decision point's
        # candidate set) -- finalized and pushed to the replay buffer as soon as that next decision
        # point is computed, whether it's later in the same round or in a later solve() call this
        # episode. See dqn_assign / finalize_episode.
        self._pending_transition: Optional[Dict[str, Any]] = None

    def solve(self, all_vehicles, active_vehicles, trips, rejected_trips, current_time=None) -> None:
        """Uniform entry point for TaxiDispatcher.optimize()."""
        self.variables_declaration(all_vehicles, trips)
        self.dqn_solver(active_vehicles, trips, rejected_trips, current_time or 0.0)

    def determine_available_vehicles(self, trip: TripLike) -> None:
        """Determine whether each vehicle can feasibly serve the trip (reach pickup by
                latest_pickup). Identical to OnlineSolver's version.
        """
        for vehicle_id, veh_info in self.vehicle_request_assign.items():
            reach_time = self.calc_reach_time(veh_info, trip)
            veh_info.assign_possible = reach_time <= trip.latest_pickup

    def dqn_solver(self, K: List[VehicleLike], P_not_assigned: List[TripLike],
                   rejected_trips: List[TripLike], current_time: float) -> None:
        """Round wrapper mirroring OnlineSolver.online_solver's two-step shape: assign, then verify.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers not yet assigned to be served
            rejected_trips : list of trips rejected in the optimization process
            current_time : current simulation time (used for the time-to-go feature and pending-pool
                           slack)
        """
        assigned_requests = self.dqn_assign(P_not_assigned, rejected_trips, current_time)

        self.create_online_solution()
        if self.verify_constraints(K, assigned_requests):
            self.calc_objective_value(K, P_not_assigned)
            self.total_customers_served = sum(1 for f_i in P_not_assigned if self.Z[f_i.id])
            logger.info("DQN solver solution is feasible. Served %d customers.", self.total_customers_served)
        else:
            logger.error("The solution is not feasible.")
            raise ValueError("The solution is not feasible")

    def dqn_assign(self, P_not_assigned: List[TripLike], rejected_trips: List[TripLike],
                   current_time: float) -> List[TripLike]:
        """Global-pair selection loop: score all feasible (vehicle, trip) pairs plus DEFER, commit
        the single best, rescore the shrunken remaining set, repeat.

        Input:
        ------------
            P_not_assigned : set of customers not yet assigned to be served
            rejected_trips : list of trips rejected in the optimization process
            current_time : current simulation time

        Output:
        ------------
            assigned_requests : list of assigned requests
        """
        self._apply_expiry_penalties(rejected_trips)

        assigned_requests: List[TripLike] = []
        remaining_trips = list(P_not_assigned)

        while remaining_trips:
            candidates = self._build_candidates(remaining_trips)
            if not candidates:
                break  # no feasible pairs at all this round -- round ends, done=False

            all_features, pair_features, defer_features = self._build_feature_matrix(
                remaining_trips, candidates, current_time)

            # This decision point's candidate features are the "next state" for whatever transition
            # was left pending from the previous decision (same round or a prior solve() call this
            # episode) -- finalize and push it now that we know it.
            self._finalize_pending_transition(all_features, done=False)

            with torch.no_grad():
                q_values = self.online_net(torch.tensor(all_features, dtype=torch.float32)).tolist()

            if self._expiry_penalty_lambda:
                self._snapshot_last_seen(candidates, pair_features, q_values)

            chosen_idx = self._select_action(q_values, len(candidates))

            if chosen_idx == len(candidates):  # DEFER chosen -- round ends here
                self._pending_transition = {"features": defer_features, "reward": 0.0}
                break

            trip, vehicle_id, veh_info = candidates[chosen_idx]
            reach_time = self.calc_reach_time(veh_info, trip)
            reward = self._compute_commit_reward(trip, veh_info, reach_time)
            self.assign_trip_to_vehicle(veh_info, trip)
            assigned_requests.append(trip)
            remaining_trips.remove(trip)

            self._pending_transition = {"features": pair_features[chosen_idx], "reward": reward}

            if self._training:
                self._maybe_train_step()

        for trip in remaining_trips:
            rejected_trips.append(trip)

        return assigned_requests

    def _build_candidates(self, remaining_trips: List[TripLike]
                          ) -> List[Tuple[TripLike, Any, VehicleState]]:
        """Every (trip, vehicle_id, veh_info) pair eligible for a commit right now.

        Feasibility is always rebuilt fresh, never cached -- a vehicle's reduced reachability after
        a commit is respected automatically on the next pass of dqn_assign's loop.
        """
        candidates: List[Tuple[TripLike, Any, VehicleState]] = []
        for trip in remaining_trips:
            self.determine_available_vehicles(trip)
            for vehicle_id, veh_info in self.get_available_vehicles().items():
                # At most one commit per vehicle per round: a second one would never be served,
                # so rewarding it would corrupt the training signal.
                if veh_info.assigned_requests:
                    continue
                # Already mid-trip from an earlier round -- the route-builder will skip it this
                # round regardless of what is decided here.
                if (self.simulation_config.solution_mode != SolutionMode.OFFLINE
                        and veh_info.route_next_stops_count > 1):
                    continue
                candidates.append((trip, vehicle_id, veh_info))
        return candidates

    def _build_feature_matrix(self, remaining_trips: List[TripLike],
                              candidates: List[Tuple[TripLike, Any, VehicleState]],
                              current_time: float
                              ) -> Tuple[np.ndarray, List[List[float]], List[float]]:
        """Everything the network sees at this decision point.

        Builds the round's shared context once, then one row per candidate (local features +
        context) and one final row for DEFER, in that order -- so row i is candidates[i] and the
        last row is always DEFER, which is what dqn_assign's `chosen_idx == len(candidates)` test
        relies on.

        Output:
        ------------
            all_features : (len(candidates) + 1, _feature_dim()) float32 array, network input
            pair_features : the per-candidate rows as Python lists, for the replay buffer
            defer_features : the DEFER row as a Python list, same

        """
        context = self._build_context_features(remaining_trips, candidates, current_time)
        pair_features = [
            self._local_features(veh_info, trip, current_time) + context
            for trip, vehicle_id, veh_info in candidates
        ]
        defer_features = self._local_features(None, None, current_time) + context
        all_features = np.asarray(pair_features + [defer_features], dtype=np.float32)

        # Catch a features/_feature_dim() mismatch here, with a message that says what went wrong,
        # rather than letting it surface as a torch shape error inside the first forward pass.
        expected_dim = self._feature_dim()
        if all_features.shape[1] != expected_dim:
            raise ValueError(
                f"feature vectors are {all_features.shape[1]} long but _feature_dim() declares "
                f"{expected_dim} -- _local_features returns {len(defer_features) - len(context)} "
                f"local values and _build_context_features returns {len(context)} context values.")
        return all_features, pair_features, defer_features

    def _snapshot_last_seen(self, candidates: List[Tuple[TripLike, Any, VehicleState]],
                            pair_features: List[List[float]], q_values: List[float]) -> None:
        """Record each pending trip's best-scoring candidate this round, so that if the trip later
        expires, its penalty can be attributed to that state. Only used when
        expiry_penalty_lambda is non-zero.
        """
        best_q_by_trip: Dict[str, float] = {}
        for i, (trip, vehicle_id, veh_info) in enumerate(candidates):
            if trip.id not in best_q_by_trip or q_values[i] > best_q_by_trip[trip.id]:
                best_q_by_trip[trip.id] = q_values[i]
                self._last_seen_features[trip.id] = pair_features[i]
                reach_time = self.calc_reach_time(veh_info, trip)
                self._last_seen_reward[trip.id] = self._compute_commit_reward(trip, veh_info, reach_time)

    def _apply_expiry_penalties(self, rejected_trips: List[TripLike]) -> None:
        """Penalize trips that expired unserved since the previous round, once each.

        rejected_trips is cumulative, so it is diffed against what was seen last round. Skipped
        entirely when expiry_penalty_lambda is 0.0 (the default).
        """
        if not self._expiry_penalty_lambda:
            return

        incoming_rejected_ids = {trip.id for trip in rejected_trips}
        newly_expired_ids = incoming_rejected_ids - self._previously_rejected_ids
        for trip in rejected_trips:
            if trip.id in newly_expired_ids:
                reference_features = self._last_seen_features.get(trip.id)
                reference_reward = self._last_seen_reward.get(trip.id)
                if reference_features is not None and reference_reward is not None:
                    penalty = -self._expiry_penalty_lambda * reference_reward
                    self.replay_buffer.push(reference_features, penalty, None, True)
                    if self._training:
                        self._maybe_train_step()
        self._previously_rejected_ids = incoming_rejected_ids

    def finalize_episode(self) -> None:
        """Call once after the episode's run_taxi_simulation() call returns (see
        execution/train_dqn.py) -- pushes whatever transition was left pending as a true terminal
        transition (done=True), since there is no next decision point once the simulation has ended.
        Also where the target network syncs: every target_sync_every episodes, at this clean episode
        boundary, rather than on a mid-episode gradient-step count.
        """
        self._finalize_pending_transition(next_candidate_features=None, done=True)
        self._episode_count += 1
        if self._episode_count % self._target_sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def save_checkpoint(self) -> None:
        """Persist the online network's weights to model_path, so the next episode's fresh
        DQNSolver instance (see class docstring: constructed per TaxiDispatcher) picks up where
        this one left off. Called by the training driver after each episode's gradient steps.
        """
        if self._model_path:
            torch.save(self.online_net.state_dict(), self._model_path)

    def _finalize_pending_transition(self, next_candidate_features: Optional[List[List[float]]],
                                      done: bool) -> None:
        if self._pending_transition is None:
            return
        self.replay_buffer.push(
            self._pending_transition["features"], self._pending_transition["reward"],
            next_candidate_features, done,
        )
        self._pending_transition = None

    def _maybe_train_step(self) -> None:
        if len(self.replay_buffer) < self._batch_size:
            return

        batch = self.replay_buffer.sample(self._batch_size)
        features = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        rewards = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        dones = torch.tensor([b[3] for b in batch], dtype=torch.float32)

        with torch.no_grad():
            next_q_max = torch.zeros(len(batch))
            for i, (_, _, next_candidates, is_done) in enumerate(batch):
                if not is_done and next_candidates is not None:
                    next_values = self.target_net(torch.tensor(next_candidates, dtype=torch.float32))
                    next_q_max[i] = next_values.max()
            targets = rewards + self._gamma * (1.0 - dones) * next_q_max

        predicted = self.online_net(features)
        loss = nn.functional.mse_loss(predicted, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._gradient_steps += 1

    def _zone_of(self, node_label: Optional[str]) -> Optional[Any]:
        return self._node_zone_mapping.get(node_label) if node_label is not None else None

    def _time_to_go(self, current_time: float) -> float:
        """Fraction of the episode horizon still ahead, in [0, 1]. 0.0 when no horizon is known,
        which makes the feature a harmless constant rather than an error.
        """
        if self._episode_end_time and self._episode_end_time > 0:
            return max(0.0, (self._episode_end_time - current_time) / self._episode_end_time)
        return 0.0

    def _feature_dim(self) -> int:
        """Total width of one row of the feature matrix: the local features for one candidate,
        plus the shared context block appended to it.

        This is what the network's input_dim is built from, so it must equal
        _N_LOCAL_FEATURES + len(_build_context_features(...)). It is checked every round in
        _build_feature_matrix.
        """
        raise NotImplementedError("DQNSolver._feature_dim() not implemented")

    def _build_context_features(self, remaining_trips: List[TripLike],
                                 candidates: List[Tuple[TripLike, Any, VehicleState]],
                                 current_time: float) -> List[float]:
        """Fixed-size fleet-wide + per-zone context, shared across every candidate this iteration --
        this is what lets the policy see competition/spatial imbalance that a pair-local view
        structurally cannot.

        Output:
        ------------
            A fixed-length list of scalars. Its length must not vary with len(remaining_trips) or
            len(candidates), since it is part of _feature_dim(). How many values, and what each
            one measures, is yours to decide.

        Design it yourself: what should this round's shared context capture about the fleet and the
        pending pool that a single candidate pair's own local features (see _local_features)
        cannot? What's available to build it from:
            - remaining_trips: every trip not yet assigned this round (always non-empty here --
              only ever called from inside dqn_assign's `while remaining_trips:` loop body).
            - candidates: the (trip, vehicle_id, veh_info) tuples currently feasible.
            - trip.latest_pickup, current_time -- per-trip urgency.
            - self._zone_of(node_label), self._zone_index, self._zone_ids -- map a stop to a zone
              and a zone to its position in a fixed-length per-zone vector.
            - TIME_NORM, FLEET_COUNT_NORM, ZONE_COUNT_NORM -- normalization constants already
              scaled for this project's magnitudes, available if you find you need count- or
              time-valued features on a comparable scale to the rest of the input.
        """
        raise NotImplementedError("DQNSolver._build_context_features() not implemented")

    def _local_features(self, veh_info: Optional[VehicleState], trip: Optional[TripLike],
                        current_time: float) -> List[float]:
        """The _N_LOCAL_FEATURES values describing one candidate on its own, before the round's
        shared context is appended (see _build_feature_matrix).

        Called two ways, and must return the same length either way, since DEFER competes against
        every real pair for the same argmax and is scored by the same network:
            - a real (vehicle, trip) pair: veh_info and trip both given.
            - DEFER: both None. Decide what "no pair" should look like to the network -- it needs
              to be recognizably not a real pair, but you choose whether any of it still varies
              with how much episode time is left.

        Set _N_LOCAL_FEATURES (top of this file) to however many values you return.

        Design it yourself: what about this specific (vehicle, trip) pair tells the network whether
        committing to it now is a good idea? What's available to compute from:
            - self.calc_reach_time(veh_info, trip) -- when this vehicle would reach the pickup.
            - trip.ready_time, trip.latest_pickup, trip.fare, trip.shortest_travel_time,
              trip.origin.label, trip.destination.label.
            - veh_info.last_stop.
            - self.durations[a][b] / self.costs[a][b] -- travel time / driving cost between any two
              stop labels (e.g. veh_info.last_stop -> trip.origin.label is the empty leg; note cost
              is just duration times a fixed rate, so it carries no information duration does not
              already).
            - self._time_to_go(current_time) -- how much of the episode horizon is left.
            - TIME_NORM, FARE_NORM, COST_NORM -- normalization constants already scaled for this
              project's magnitudes, available if you find you need them.
        """
        raise NotImplementedError("DQNSolver._local_features() not implemented")

    def _select_action(self, q_values: List[float], n_candidates: int) -> int:
            """Pick one index into the scored rows: 0..n_candidates-1 is a real pair, n_candidates is
            DEFER (the last row -- see _build_feature_matrix).
    
            Design it yourself: this is where exploration meets exploitation. What's available:
                - self._training -- False during evaluation, where you should never explore.
                - self._epsilon -- how often to explore while training.
                - self._defer_explore_prob -- see the question below.
                - self._rng -- use this, not the random module, so runs stay reproducible.
                - q_values -- one score per row, in the same order as the rows.
    
            Think about DEFER's share of exploration. If every action were equally likely to be
            explored, how would DEFER's chances change as the pending pool grows -- and is that what
            you want in exactly the busy rounds where waiting might be the right call?
            """
            raise NotImplementedError("DQNSolver._select_action() not implemented")

    def _compute_commit_reward(self, trip: TripLike, veh_info: VehicleState, reach_time: float) -> float:
        """Reward for committing this (vehicle, trip) pair, shaped by simulation_config.objective.

        TOTAL_CUSTOMERS is given as a worked example. Design the other two yourself, and keep in
        mind that DEFER earns exactly 0.0 -- whatever you return is what a real commit is worth
        relative to doing nothing.

        Available: trip.fare, trip.shortest_travel_time, trip.ready_time, trip.latest_pickup,
        reach_time, self.durations[veh_info.last_stop][trip.origin.label], and
        compute_leg_profit(fare, trip_duration, empty_duration) from network_tools.
        """
        if self.objective == Objectives.TOTAL_CUSTOMERS:
            return 1.0
        if self.objective == Objectives.WAIT_TIME:
            raise NotImplementedError("_compute_commit_reward: WAIT_TIME not implemented")
        raise NotImplementedError("_compute_commit_reward: TOTAL_PROFIT not implemented")
