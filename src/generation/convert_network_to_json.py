"""One-time helper to convert the shared network graph from the old pickle format (network.pkl) to
JSON (graph.json), using networkx's standard node-link serialization. Run manually:

    python -m src.generation.convert_network_to_json
"""
import json
import os
import pickle

import networkx as nx

from src.utilities.paths import BASE_FOLDER, GRAPH_FILE_PATH

PICKLE_PATH = os.path.join(BASE_FOLDER, "network.pkl")
JSON_PATH = GRAPH_FILE_PATH


def convert_network_to_json(pickle_path: str = PICKLE_PATH, json_path: str = JSON_PATH) -> None:
    with open(pickle_path, "rb") as f:
        network = pickle.load(f)
    with open(json_path, "w") as f:
        json.dump(nx.node_link_data(network, edges="links"), f)


def main():
    convert_network_to_json()
    print(f"Converted {PICKLE_PATH} -> {JSON_PATH}")


if __name__ == "__main__":
    main()
