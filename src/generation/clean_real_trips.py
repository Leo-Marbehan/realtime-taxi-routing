"""One-time data-prep: drop trip rows whose pickup/dropoff node isn't one of the nodes kept in
graph.json, add a trip_duration column from graph.json's precomputed shortest-path durations, drop
rows with trip_duration < 120 seconds, and merge everything into one time-sorted pool. Overwrites
each file in data/Real_Trips/ in place and (re)writes trip_pool.csv. Run manually:

    python -m src.generation.clean_real_trips
"""
import glob
import os

import pandas as pd

from src.utilities.network_tools import get_durations, load_network_graph
from src.utilities.paths import REAL_INSTANCES_FOLDER, REAL_TRIPS_FOLDER, TRIP_POOL_PATH

GRAPH_FILE_PATH = os.path.join(REAL_INSTANCES_FOLDER, "graph.json")


def _day_files(trips_folder: str) -> list:
    """Every per-day trip CSV in trips_folder, excluding the merged pool file itself."""
    return sorted(f for f in glob.glob(os.path.join(trips_folder, "*.csv"))
                  if os.path.basename(f) != os.path.basename(TRIP_POOL_PATH))


def clean_trip_file(csv_path: str, valid_node_ids: set, durations: dict) -> tuple:
    """Drop rows whose pickup_ID/dropoff_ID isn't in valid_node_ids, add a trip_duration column
    looked up from durations, and overwrite csv_path in place. Returns (rows_kept, rows_total)."""
    df = pd.read_csv(csv_path)
    total = len(df)
    mask = df["pickup_ID"].isin(valid_node_ids) & df["dropoff_ID"].isin(valid_node_ids)
    df = df[mask].copy()
    df["trip_duration"] = df.apply(
        lambda row: durations[str(row["pickup_ID"])][str(row["dropoff_ID"])], axis=1)
    df = df[df["trip_duration"] >= 120]
    df.to_csv(csv_path, index=False)
    return len(df), total


def clean_all_real_trips(trips_folder: str = REAL_TRIPS_FOLDER, graph_path: str = GRAPH_FILE_PATH) -> None:
    network = load_network_graph(graph_path)
    valid_node_ids = {int(node_id) for node_id in network.nodes}
    durations = get_durations(network)

    table_data = []
    for csv_path in _day_files(trips_folder):
        kept, total = clean_trip_file(csv_path, valid_node_ids, durations)
        table_data.append({
            "File": os.path.basename(csv_path),
            "Kept": kept,
            "Total": total,
            "% Kept": round(100 * kept / total, 1) if total else 0.0,
        })

    df = pd.DataFrame(table_data)
    with pd.option_context('display.colheader_justify', 'center'):
        print(df.to_markdown(tablefmt="pipe", index=False))


def build_trip_pool(trips_folder: str = REAL_TRIPS_FOLDER, pool_path: str = TRIP_POOL_PATH) -> None:
    """Merge every cleaned per-day file into one pool, sorted by request_time_sec, for
    create_instances() (and the consensus solver) to sample from."""
    frames = [pd.read_csv(f) for f in _day_files(trips_folder)]
    pool = pd.concat(frames, ignore_index=True).sort_values("request_time_sec").reset_index(drop=True)
    pool.to_csv(pool_path, index=False)
    print(f"Wrote {len(pool)} rows to {pool_path}")


if __name__ == "__main__":
    clean_all_real_trips()
    build_trip_pool()
