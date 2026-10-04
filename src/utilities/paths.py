"""Shared filesystem path constants, single source of truth for all entry points."""
import os

BASE_FOLDER = "data/Instances"
GRAPH_FILE_PATH = os.path.join(BASE_FOLDER, "graph.json")

# The one instance data source the active pipeline uses (synthetic generation has been removed).
REAL_INSTANCES_FOLDER = "data/Real_Instances"
RESULTS_FOLDER = os.path.join(REAL_INSTANCES_FOLDER, "Results")

# Raw NYC TLC trip records, used to build real (non-synthetic) request instances.
REAL_TRIPS_FOLDER = "data/Real_Trips"

# Merged, time-sorted pool of all cleaned Real_Trips records, sampled from when building instances.
TRIP_POOL_PATH = os.path.join(REAL_TRIPS_FOLDER, "trip_pool.csv")

# Small, hand-picked instances used by reference_test.py -- kept separate from Real_Instances so a
# fast regression check never accidentally picks up one of the full-size analysis instances.
DEBUG_INSTANCES_FOLDER = "data/Debug_Instances"
