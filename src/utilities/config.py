from dataclasses import dataclass
from typing import Any, Optional


from src.utilities.enums import (
    Algorithm,
    Objectives,
    SolutionMode,
    ConsensusParams,
    DestroyMethod,
)


@dataclass
class ConsensusConfig:
    """Parameters specific to Algorithm.CONSENSUS (the online stochastic solver).

    Attributes:
        nb_scenario: int
            Total number of scenarios to be solved for consensus.
        nb_requests: int
            Number of requests the instance actually being run was generated with -- used to
            rate-match the consensus algorithm's own scenario draws to this instance's sparsity,
            rather than assuming the real trip pool's full density. Derive this from the instance's
            own requests.csv via config_tools.derive_consensus_params(instance_folder) rather than a
            hardcoded constant, which can silently drift from what's actually on disk.
        generation_sim_time: float
            The window (in seconds) the instance's requests were actually sampled over at generation
            time. Together with nb_requests, this gives the instance's own arrival rate, which
            scenario generation scales down to its own (shorter) look-ahead window. Also derived via
            config_tools.derive_consensus_params(instance_folder) -- the default below only applies to
            the helper's own degenerate-instance fallback, not a real per-instance value.
        seed: Optional[int]
            Seed for the consensus algorithm's own scenario-sampling randomness, independent of any
            seed used at instance-generation time. None (default) is unseeded/non-reproducible.
        consensus_param: ConsensusParams(Enum)
            The type of consensus algorithm:
                - QUALITATIVE : A counter is incremented for the best request to assign at each scenario.
                - QUANTITATIVE : The best request to assign is credited by the optimal solution value,
                                 rather than merely incrementing a counter.
    """
    nb_scenario: int
    nb_requests: int
    consensus_param: ConsensusParams
    generation_sim_time: float = 7200.0
    seed: Optional[int] = None


@dataclass
class ReOptimizeConfig:
    """Parameters specific to Algorithm.RE_OPTIMIZE.

    Attributes:
        destroy_method: DestroyMethod(Enum)
            Method used for destruction in re-optimizing.
                - DEFAULT: Default destruction method (Complete re-optimization)
                - FIX_VARIABLES: fix some of the variables in the model
                - FIX_ARRIVALS: fix a time window around the arrival time
                - BONUS: arbitrary destroy method as bonus
    """
    destroy_method: DestroyMethod


@dataclass
class DQNConfig:
    """Parameters specific to Algorithm.DQN (learned Q-value based global-pair selection).

    Attributes:
        model_path: str
            Path to the checkpoint file the solver loads its network weights from at construction
            (if it exists) and that a training driver (see execution/train_dqn.py) overwrites after
            each episode. Weights are NOT persisted by the solver itself across separate
            run_taxi_simulation() calls -- this file is the only thing that carries them between
            episodes/processes.
        training_mode: bool
            If True, actions are epsilon-greedy (see epsilon) and every transition is pushed into an
            in-memory replay buffer for a training driver to consume after the episode ends. If
            False (evaluation), actions are always the greedy argmax over Q-values (epsilon is
            ignored) and no replay data is collected.
        epsilon: float
            Exploration rate for epsilon-greedy action selection during training. Ignored when
            training_mode is False.
        defer_explore_prob: float
            Given exploration has already fired (the epsilon roll), the probability the explored
            action is DEFER rather than a uniformly random real (vehicle, trip) pair. Without this,
            DEFER is just one more slot in a uniform draw over all candidates, so it gets explored
            less and less as the candidate pool grows -- this keeps its exploration rate fixed
            regardless of how many pairs are feasible that round.
        gamma: float
            Discount factor for the Bellman target used when training on replayed transitions.
        target_sync_every: int
            Number of episodes between copying the online network's weights into the target network
            (standard DQN stabilization), checked at the clean episode boundary in finalize_episode()
            -- not a mid-episode gradient-step count, so a sync always lands between decisions, never
            partway through one.
        replay_buffer_size: int
            Maximum number of transitions kept in the replay buffer before the oldest are evicted.
            Each transition stores the full candidate feature matrix at its next decision point
            (~50KB each, numpy float32, measured at this problem's scale) -- the default here is sized
            for that, not a generic DQN default; raising it substantially increases memory use
            (transitions * ~50KB), which matters once reuse_solver keeps the buffer alive for an
            entire training run instead of discarding it every episode.
        batch_size: int
            Minibatch size sampled from the replay buffer per gradient step.
        learning_rate: float
            Learning rate for the optimizer used to train the online network.
        seed: Optional[int]
            Seed for the epsilon-greedy exploration draws. None (default) is unseeded/non-reproducible.
        expiry_penalty_lambda: float
            Scales the in-band penalty pushed when a trip expires unserved: penalty =
            -expiry_penalty_lambda * _compute_commit_reward(trip, ...), using that trip's own
            last-seen best candidate -- so the penalty is automatically in the right units and scale
            for whichever objective is active (profit / total_customers / waiting_time), with no
            separate per-objective penalty logic needed. 0.0 (default) disables the penalty entirely.
        reuse_solver: Optional[Any]
            An existing DQNSolver instance from a prior episode this training run, whose network,
            optimizer, replay buffer, and gradient-step counter should be reused in place of
            constructing new ones -- set by a training driver across a run (see
            execution/train_dqn.py's train()) so the replay buffer and optimizer state actually
            persist across episodes instead of resetting every time. None (default) for every one-off
            eval/viewer/single_test call, which always gets a freshly-constructed solver as before.
    """
    model_path: str
    training_mode: bool = False
    epsilon: float = 0.0
    defer_explore_prob: float = 0.1
    gamma: float = 0.99
    target_sync_every: int = 4
    replay_buffer_size: int = 20_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: Optional[int] = None
    expiry_penalty_lambda: float = 0.0
    reuse_solver: Optional[Any] = None


@dataclass
class SimulationConfig:
    """
    objective : Objectives(Enum)
            The optimization objective to achieve (e.g., profit maximization).
                - total_profit: total profit of served requests (fare minus cost)
                - total_revenue: total fare (revenue) of served requests
                - total_cost: total driving cost of all routes
                - total_customers: total number of served customers
                - waiting_time: total wait time of customers
                - total_empty_travel_time: total travel time on empty legs only
                - multi_objective: weighted combination of multiple objectives (see weight)

        algorithm : Algorithm(Enum)
            The optimization algorithm to use for routing and assignment.
                - MIP_SOLVER : using the Gurobi MIP solver to solve the problem
                - GREEDY : greedy approach to assign requests to vehicles
                - RANDOM : random algorithm to assign arrival requests to vehicles
                - RANKING : ranking method to assign arrival requests to vehicles
                - CONSENSUS : consensus online stochastic algorithm to assign arrival requests to vehicles
                - RE_OPTIMIZE: Algorithm to re-optimize the solution based on destroy and repair
                - DQN : learned Q-value based global-pair selection (see solvers/dqn_solver.py)
        solution_modes: SolutionMode(Enum)
            The optimization mode based on the availability of request information
                - offline : all the requests revealed (known) at the start (release time = 0 for all requests)
                - fully_online : All requests are released exactly at their ready time
                - advance_notice : All requests are known a fixed amount of time before their ready time.
                - partial : a random percentage of requests are known at the start and for the rest release time is equal to the ready time

        known_portion: int
            percentage of requests that are known in advance
        advance_notice: int
            Fixed amount of time (in seconds) the requests are released before their ready time.
        time_window : int
            Time window (in seconds) for picking up the requests
        weight: float
            Weight w in [0, 1] for profit in the combined objective (cost weight is 1-w).
            Only used when objective is multi_objective (objective-specific, not algorithm-specific,
            so it lives here rather than in ConsensusConfig/ReOptimizeConfig). Default 0.5.
        consensus: Optional[ConsensusConfig]
            Required when algorithm is CONSENSUS; must be None otherwise.
        re_optimize: Optional[ReOptimizeConfig]
            Required when algorithm is RE_OPTIMIZE; must be None otherwise.
        dqn: Optional[DQNConfig]
            Required when algorithm is DQN; must be None otherwise.
    """
    objective: Objectives = Objectives.TOTAL_CUSTOMERS
    algorithm: Algorithm = Algorithm.MIP_SOLVER
    solution_mode: SolutionMode = SolutionMode.OFFLINE
    known_portion: int = 100
    advance_notice: int = 0
    time_window: int = 3
    weight: float = 0.5
    consensus: Optional[ConsensusConfig] = None
    re_optimize: Optional[ReOptimizeConfig] = None
    dqn: Optional[DQNConfig] = None

    def __post_init__(self):
        # Validate common parameters if necessary
        if not (0.0 <= self.known_portion <= 100.0):
            raise ValueError("known_portion must be between 0 and 100.")
        if self.advance_notice < 0:
            raise ValueError("advance_notice cannot be negative.")
        if self.time_window <= 0:
            raise ValueError("time_window must be positive.")
        # Fail fast on a missing/misconfigured algorithm-specific config, instead of a KeyError
        # deep inside whichever solver reads it (see architecture finding #5).
        if self.algorithm == Algorithm.CONSENSUS and self.consensus is None:
            raise ValueError("SimulationConfig.consensus must be set when algorithm is CONSENSUS.")
        if self.algorithm == Algorithm.RE_OPTIMIZE and self.re_optimize is None:
            raise ValueError("SimulationConfig.re_optimize must be set when algorithm is RE_OPTIMIZE.")
        if self.algorithm == Algorithm.DQN and self.dqn is None:
            raise ValueError("SimulationConfig.dqn must be set when algorithm is DQN.")
