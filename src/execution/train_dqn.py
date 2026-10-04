"""Training driver for Algorithm.DQN (see solvers/dqn_solver.py).

One run_taxi_simulation() call = one training episode. The DQNSolver instance lives only for that
one call (constructed fresh by TaxiDispatcher._initialize_solver each time), but DQNConfig.reuse_solver
lets a new instance borrow the previous episode's network, optimizer, and replay buffer by reference
instead of starting over -- see train()'s hints below. Network weights also persist across separate
processes via a checkpoint file (DQNConfig.model_path) this driver loads from / saves to between
episodes; the replay buffer and optimizer momentum do not survive a process restart, only in-memory
reuse within one continuous run of this loop.

Config-driven, like every other task type: train/held-out instances, checkpoint path, episode count,
and epsilon schedule all come from inputs.json's "train_dqn" block, read via train_from_config() --
run with `python -m src.main` after setting task_type to "train_dqn". train() itself still takes plain
parameters (so it's also usable directly, e.g. from a notebook), but nothing in this file hardcodes an
instance list or checkpoint path for any one experiment.
"""
import logging
import os
import random
from typing import Any, Dict, List, Optional, Union

from src.simulation.run_simulation import run_taxi_simulation
from src.utilities.config import DQNConfig, SimulationConfig
from src.utilities.config_tools import match_enum, resolve_release_params
from src.utilities.enums import Algorithm, Objectives, SolutionMode
from src.utilities.paths import REAL_INSTANCES_FOLDER
from src.utilities import result_columns as col

logger = logging.getLogger(__name__)


def train(train_instances: List[str], model_path: str, num_episodes: int,
          epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.99,
          time_window: int = 180, objective: Objectives = Objectives.TOTAL_PROFIT,
          solution_mode: SolutionMode = SolutionMode.FULLY_ONLINE, weight: float = 0,
          known_portion: Optional[float] = None, instance_selection_seed: int = 0,
          expiry_penalty_lambda: float = 0.0) -> None:
    """Run num_episodes training episodes, shuffling across train_instances each pass, decaying
    epsilon, and checkpointing after every episode.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    epsilon = epsilon_start
    resolved_known_portion, resolved_advance_notice = resolve_release_params(solution_mode, known_portion)
    # Carried across iterations so the replay buffer, optimizer momentum, and gradient-step count
    # persist across episodes instead of resetting every time a fresh DQNSolver is constructed --
    # None on the first episode (no prior solver to reuse yet).
    reuse_solver = None
    # Shuffled-per-epoch instance selection: every instance is seen exactly once per full pass
    # through train_instances, in a freshly randomized order each pass -- balanced exposure, but no
    # fixed sequence for the agent to pick up on. Refilled and reshuffled whenever exhausted.
    instance_rng = random.Random(instance_selection_seed)
    instance_queue: List[str] = []

    for episode in range(num_episodes):
        if not instance_queue:
            instance_queue = list(train_instances)
            instance_rng.shuffle(instance_queue)
        instance = instance_queue.pop()
        test_folder = os.path.join(REAL_INSTANCES_FOLDER, instance)

        dqn_config = DQNConfig(model_path=model_path, training_mode=True, epsilon=epsilon, seed=episode,
                                reuse_solver=reuse_solver, expiry_penalty_lambda=expiry_penalty_lambda)
        config = SimulationConfig(
            objective=objective, algorithm=Algorithm.DQN, solution_mode=solution_mode,
            known_portion=resolved_known_portion, advance_notice=resolved_advance_notice,
            time_window=time_window, weight=weight, dqn=dqn_config,
        )

        _, output_dict, dispatcher = run_taxi_simulation(
            test_folder=test_folder, config=config, return_dispatcher=True)

        solver = dispatcher.solver
        solver.finalize_episode()
        solver.save_checkpoint()
        reuse_solver = solver

        print(f"Episode {episode + 1}/{num_episodes}: instance={instance}, "
              f"served={output_dict[col.NUM_SERVED]}, profit={output_dict.get(col.TOTAL_PROFIT)}, "
              f"epsilon={epsilon:.3f}")

        epsilon = max(epsilon_end, epsilon * epsilon_decay)


def train_from_config(config_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """Entry point for inputs.json's "train_dqn" block. Accepts either a single training-run dict
    (today's format -- trains one model) or a list of them (e.g. one entry per objective, each
    with its own "objective"/"model_path") -- trains each in turn, in order, in one invocation.
    An entry's optional "name" key is log-only metadata, not read by _train_one.
    """
    if isinstance(config_data, list):
        for i, entry in enumerate(config_data, 1):
            name = entry.get("name", f"run {i}")
            logger.info("train_dqn: starting '%s' (%d/%d)", name, i, len(config_data))
            _train_one(entry)
        return
    _train_one(config_data)


def _train_one(config_data: Dict[str, Any]) -> None:
    """Build train()'s arguments from one training-run config dict. train_instances/model_path/
    num_episodes are required -- no fallback default, since the whole point is that these no longer
    live in code. held_out_instances is optional and purely for record-keeping/validation; the
    training loop itself only ever iterates train_instances.
    """
    train_instances = config_data["train_instances"]
    held_out = config_data.get("held_out_instances", [])
    overlap = set(train_instances) & set(held_out)
    if overlap:
        logger.warning("train_instances and held_out_instances overlap: %s", sorted(overlap))

    num_episodes = config_data["num_episodes"]
    epsilon_start = config_data.get("epsilon_start", 1.0)
    epsilon_end = config_data.get("epsilon_end", 0.05)
    # If not given explicitly, anneal so epsilon actually reaches epsilon_end by the last episode,
    # regardless of how many episodes are configured -- avoids hand-computing decay per experiment.
    epsilon_decay = config_data.get(
        "epsilon_decay", (epsilon_end / epsilon_start) ** (1 / num_episodes))

    train(
        train_instances=train_instances,
        model_path=config_data["model_path"],
        num_episodes=num_episodes,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        time_window=config_data.get("time_window", 180),
        objective=match_enum(config_data.get("objective", "total_profit"), Objectives),
        solution_mode=match_enum(config_data.get("solution_mode", "fully_online"), SolutionMode),
        weight=config_data.get("weight", 0),
        known_portion=config_data.get("known_portion"),
        instance_selection_seed=config_data.get("instance_selection_seed", 0),
        expiry_penalty_lambda=config_data.get("expiry_penalty_lambda", 0.0),
    )
