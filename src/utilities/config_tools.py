import csv
import difflib
import hashlib
import math
import os
from typing import Optional, Tuple

from src.utilities.enums import SolutionMode

# Constants for solution mode defaults
KNOWN_PORTION_OFFLINE = 100
KNOWN_PORTION_ONLINE = 0
ADVANCE_NOTICE_DEFAULT = 1800  # seconds (30 minutes)
ADVANCE_NOTICE_NONE = 0


def resolve_release_params(solution_mode: SolutionMode, requested_known_portion: Optional[float]) -> Tuple[float, float]:
    """
    Resolve the (known_portion, advance_notice) pair implied by a solution mode.

    Parameters:
    -----------------
    solution_mode : SolutionMode
        The selected solution mode.
    requested_known_portion : float
        The known_portion requested by the caller (used only for PARTIAL_ONLINE/CUSTOM_SCENARIO,
        where the mode itself doesn't dictate a fixed value).

    Returns:
    -----------------
    Tuple[float, float]
        (known_portion, advance_notice) to use for the SimulationConfig.
    """
    if solution_mode == SolutionMode.OFFLINE:
        return KNOWN_PORTION_OFFLINE, ADVANCE_NOTICE_NONE
    elif solution_mode == SolutionMode.FULLY_ONLINE:
        return KNOWN_PORTION_ONLINE, ADVANCE_NOTICE_NONE
    elif solution_mode == SolutionMode.PARTIAL_ONLINE:
        return requested_known_portion, ADVANCE_NOTICE_NONE
    elif solution_mode == SolutionMode.ADVANCE_NOTICE:
        return KNOWN_PORTION_ONLINE, ADVANCE_NOTICE_DEFAULT
    else:
        return requested_known_portion, ADVANCE_NOTICE_DEFAULT


def get_solution_mode(known_portion, advance_notice):
    """
    Determines the operational scenario (solution mode) based on
    known_portion and advance_notice.

    Parameters:
    -----------------
    known_portion : int
        Percentage of requests known in advance (0 to 100).
    advance_notice : int
        Advance notice in seconds for request release.

    Returns:
    -----------------
    SolutionMode
        The corresponding solution mode.
    """
    if known_portion == 100:
        return SolutionMode.OFFLINE
    elif known_portion == 0 and advance_notice == 0:
        return SolutionMode.FULLY_ONLINE
    elif known_portion == 0 and 1 <= advance_notice <= 30:
        return SolutionMode.ADVANCE_NOTICE
    elif 100 > known_portion > 0 == advance_notice:
        return SolutionMode.PARTIAL_ONLINE
    else:
        return SolutionMode.CUSTOM_SCENARIO


# seconds (30 min) -- instances are always generated with start_time/sim_time on clean 30-minute
# boundaries (e.g. 7:00/8:00/9:00), so rounding the observed ready_time range out to this boundary
# recovers the true generation window exactly (see derive_consensus_params).
WINDOW_BOUNDARY = 1800


def derive_consensus_params(instance_folder: str) -> Tuple[int, float]:
    """Derive (nb_requests, generation_sim_time) from an instance's own requests.csv, so consensus's
    scenario generation (stochastic_solver.py) rate-matches whatever instance is actually run, instead
    of assuming the density of one reference instance group. Reads the file directly rather than
    trusting a config value that could drift from what's actually on disk -- that drift is exactly how
    a previous hardcoded-constant version of this went stale.

    generation_sim_time is NOT simply max(ready_time) - min(ready_time): the first request essentially
    never arrives exactly at the window's start, nor the last exactly at its end, so the raw span is
    systematically short -- and increasingly so for sparser instances. Instead, min/max are rounded
    outward to the nearest WINDOW_BOUNDARY, recovering the true generation window exactly for instances
    that were generated on clean boundaries (as this project's always are). Floored at WINDOW_BOUNDARY
    itself so a very sparse instance (e.g. all requests falling inside one 30-minute bucket) can't
    round down to a zero-width window, which would divide by zero downstream.
    """
    ready_times = []
    with open(os.path.join(instance_folder, "requests.csv"), newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            ready_times.append(float(row["ready_time"]))

    nb_requests = len(ready_times)
    if len(ready_times) >= 2:
        window_start = math.floor(min(ready_times) / WINDOW_BOUNDARY) * WINDOW_BOUNDARY
        window_end = math.ceil(max(ready_times) / WINDOW_BOUNDARY) * WINDOW_BOUNDARY
        generation_sim_time = max(window_end - window_start, WINDOW_BOUNDARY)
    else:
        generation_sim_time = 7200.0
    return nb_requests, generation_sim_time


def compute_instance_fingerprint(instance_folder: str) -> str:
    """Hash of an instance's requests.csv + vehicles.csv content -- a verifiable identity for
    "is this the same instance data a baseline result was computed on", independent of the
    instance folder's name (which can be reused for regenerated/different data). Used to detect a
    stale competitive-ratio baseline (see run_simulation.py) rather than trusting a Key string that
    only encodes the folder name, not its content.

    Truncated to 16 hex chars (64 bits) -- collision-resistant enough for this non-adversarial use
    (detecting accidental staleness, not tampering) and short enough to read in a CSV column.
    """
    hasher = hashlib.sha256()
    for filename in ("requests.csv", "vehicles.csv"):
        with open(os.path.join(instance_folder, filename), "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()[:16]


def match_enum(arg, enum):
    enum_values = {e.value.lower(): e for e in enum}
    match = difflib.get_close_matches(arg.lower(), enum_values.keys(), n=1, cutoff=.2)
    if match:
        return enum_values[match[0]]

    raise ValueError("Not found in enum: " + arg)
