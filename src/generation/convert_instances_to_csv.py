"""One-time helper to convert instance data from the old JSON format (customers.json/taxis.json) to
the new CSV format (requests.csv/vehicles.csv). Run manually:

    python -m src.generation.convert_instances_to_csv
"""
import csv
import json
import os

from src.utilities.paths import BASE_FOLDER

DEFAULT_CAPACITY = 4  # matches the hardcoded default used everywhere before capacity was persisted

REQUEST_FIELDS = ["trip_id", "origin", "destination", "release_time", "ready_time", "due_time", "fare"]
VEHICLE_FIELDS = ["vehicle_id", "start_time", "start_stop", "capacity"]


def convert_instance_folder(folder_path: str) -> None:
    """Convert one instance folder's customers.json/taxis.json into requests.csv/vehicles.csv."""
    with open(os.path.join(folder_path, "customers.json"), "r") as f:
        customers = json.load(f)
    with open(os.path.join(folder_path, "requests.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUEST_FIELDS, delimiter=";")
        writer.writeheader()
        for entry in customers:
            writer.writerow({
                "trip_id": entry["id"],
                "origin": entry["orig"],
                "destination": entry["dest"],
                "release_time": entry["tcall"],
                "ready_time": entry["tmin"],
                "due_time": entry["tmax"],
                "fare": entry["fare"],
            })

    with open(os.path.join(folder_path, "taxis.json"), "r") as f:
        taxis = json.load(f)
    with open(os.path.join(folder_path, "vehicles.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VEHICLE_FIELDS, delimiter=";")
        writer.writeheader()
        for entry in taxis:
            writer.writerow({
                "vehicle_id": entry["id"],
                "start_time": entry["initTime"],
                "start_stop": entry["initPos"],
                "capacity": entry.get("capacity", DEFAULT_CAPACITY),
            })


def main():
    for name in sorted(os.listdir(BASE_FOLDER)):
        folder_path = os.path.join(BASE_FOLDER, name)
        if os.path.isfile(os.path.join(folder_path, "customers.json")):
            convert_instance_folder(folder_path)
            print(f"Converted {name}")


if __name__ == "__main__":
    main()
