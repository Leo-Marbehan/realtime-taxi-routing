import json
import os

import networkx as nx
import matplotlib.pyplot as plt


def load_network_graph(file_path):
    """Load a network graph from a JSON file (networkx node-link format, see save_network_graph)."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data, directed=True, multigraph=False, edges='links')


def build_node_zone_mapping(network, zones_geojson_path):
    """Map each network node id to the zone (cartodb_id) whose polygon contains its (lon, lat)
    position, via point-in-polygon containment against the real zone geometry. Returns {} if the
    file doesn't exist (e.g. for a synthetic network with no real-world zone correspondence)."""
    if not os.path.isfile(zones_geojson_path):
        return {}

    from shapely.geometry import Point, shape

    with open(zones_geojson_path) as f:
        zones_geojson = json.load(f)
    zones = [(shape(feature["geometry"]), feature["properties"]["cartodb_id"])
             for feature in zones_geojson["features"]]

    mapping = {}
    for node_id, data in network.nodes(data=True):
        lon, lat = data["pos"]
        point = Point(lon, lat)
        # Point-in-polygon test: which zone's actual polygon geometry contains this point.
        mapping[node_id] = next((zone_id for polygon, zone_id in zones if polygon.contains(point)), None)
    return mapping


def has_valid_zones(trip, node_zone_mapping):
    """Whether both the origin and destination of a trip fall inside a known zone."""
    return (node_zone_mapping.get(trip.origin.label) is not None
            and node_zone_mapping.get(trip.destination.label) is not None)


def get_distances(network):
    """
    Function: calculate the shortest distance between each pair of stop nodes in the network graph
        network : routing network graph
    """
    distances = {}
    for node1, data in network.nodes(data=True):
        if node1 not in distances:
            distances[node1] = {}
        for node2 in network.nodes():
            distances[node1][node2] = round(data['shortest_paths'][node2]['total_distance'],0)

    return distances


def get_durations(network):
    """ Function: calculate the shortest travel time between each pair of stop nodes in the network graph
        network : routing network graph
    """
    durations = {}
    for node1, data in network.nodes(data=True):
        if node1 not in durations:
            durations[node1] = {}
        for node2 in network.nodes():
            durations[node1][node2] = round(data['shortest_paths'][node2]['total_duration'],0)

    return durations


# Driving cost, expressed per minute ($11/hour, a realistic vehicle operating-cost estimate):
# durations are in seconds, so this is applied as duration_seconds / 60 * COST_PER_MINUTE. Shared
# with taxi_dispatcher.py's live driving_cost computation, which must use the exact same rate.
COST_PER_MINUTE = 11.00 / 60


def compute_driving_cost(shortest_travel_time, travel_time_to_pick, cost_per_minute=COST_PER_MINUTE):
    """Driving cost of one served leg: the empty approach to pickup plus the loaded trip itself, both
    in seconds. Shared by taxi_dispatcher.py's RoutePlanBuilder._build_route_plan (which still needs
    driving_cost on its own, for metrics_tracker.record_served_leg) and compute_leg_profit below --
    single source of truth for this arithmetic, no duplicated formula between callers.
    """
    return (shortest_travel_time + travel_time_to_pick) / 60 * cost_per_minute


def compute_leg_profit(fare, shortest_travel_time, travel_time_to_pick, cost_per_minute=COST_PER_MINUTE):
    """Profit realized by serving one leg: fare minus compute_driving_cost(...). `shortest_travel_time`/
    `travel_time_to_pick` are both in seconds (durations, not costs -- do not pass a `costs[...]`
    lookup here, the units would be dollars, not seconds).

    Lets a solver compute a candidate assignment's profit directly, without the metrics-tracker side
    effect that RoutePlanBuilder._build_route_plan's call site also has.
    """
    return fare - compute_driving_cost(shortest_travel_time, travel_time_to_pick, cost_per_minute)


def get_costs(network):
    """ Function: calculate the cost of driving between each pair of stop nodes in the network graph
        here the cost is COST_PER_MINUTE dollars per minute of driving
        network : routing network graph
    """
    costs = {}
    for node1, data in network.nodes(data=True):
        if node1 not in costs:
            costs[node1] = {}
        for node2 in network.nodes():
            costs[node1][node2] = round(round(data['shortest_paths'][node2]['total_duration'],0) / 60 * COST_PER_MINUTE,2)
    return costs


def draw_network(network, save_path):
    """Function : Draw the network graph.
    """
    plt.figure(figsize=(10, 10))

    # Extract positions from node attributes
    pos = nx.get_node_attributes(network, 'pos')
    road_types = nx.get_edge_attributes(network, 'roadType')
    color_map = {1: 'red', 2: 'blue', 3: 'gray'}
    edge_colors = [color_map.get(road_types[edge], 'black') for edge in network.edges()]

    nx.draw_networkx(
        network,
        pos=pos,
        with_labels=True,
        node_size=80,
        width=1,
        font_size=6,
        node_color='yellowgreen',
        edge_color=edge_colors,
        arrows=True
    )

    # Add edge labels for edges with length > 1000
    labels = {
        (u, v): f"{attrs['length']:.0f}"
        for (u, v, attrs) in network.edges(data=True)
        if attrs['length'] > 1000
    }
    nx.draw_networkx_edge_labels(network, pos=pos, edge_labels=labels, font_size=7)


    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path + '/Network.png', dpi=1000)
    plt.show(block=False)  # Show plot without blocking the script
    plt.pause(2)  # Pause for 2 seconds
    plt.close()
