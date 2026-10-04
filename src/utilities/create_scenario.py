from types import SimpleNamespace
from typing import Optional

import numpy as np
from scipy.stats import expon
from src.simulation.ride_request import RideRequest
from src.utilities.network_tools import has_valid_zones
from multimodalsim.simulator.stop import LabelLocation


def compute_release_time(t_ready, start_time, known_portion, advance_notice):
    """The release_time implied by known_portion/advance_notice for a request ready at t_ready.
    advance_notice is in seconds, same clock as t_ready/start_time."""
    if known_portion == 100:
        return start_time
    elif known_portion == 0 and advance_notice == 0:
        return t_ready
    else:
        return max(start_time, t_ready - advance_notice)


def create_random_requests(
        network,
        cust_node_hour,
        start_ID,
        start_time,
        durations,
        time_window,
        sim_time=1800,
        alpha=3.00,
        beta=0.03,
        advance_notice=1800,
        known_portion=100,
        nb_requests=None,
        rng: Optional[np.random.Generator] = None):
    """ Function: Generate random customer ride requests based on a Poisson process.
        Input:
        ------------
            network : Network object
                The road network, including nodes representing stop points.
            cust_node_hour : float
                Average number of customers per node per hour (mean of the Poisson process).
            start_ID : int
                Starting ID for the generated trip requests.
            start_time : float
                The time in seconds after which requests are received.
            durations : dict
                A nested dictionary containing travel times between nodes, with outer keys as origin node IDs
                and inner keys as destination node IDs.
            sim_time : float, optional
                The total time in seconds for receiving requests. Defaults to 1800 (30 minutes).
            alpha : float, optional
                Base fare ($), independent of travel time. Defaults to 3.00, matching a real NYC taxi
                flag-drop; calibrated jointly with beta and network_tools.COST_PER_MINUTE so that
                driving cost is a meaningful (though not dominant) share of a typical trip's fare.
            beta : float, optional
                Fare rate in $/second of travel_time. Defaults to 0.03 ($1.80/min); see alpha above for
                calibration context.
            time_window : int, optional
                The time window in seconds within which customers are willing to be picked up after
                their ready time.
            advance_notice : int, optional
                The time in seconds before the ready time that customers call to make a request.
                Defaults to 1800 (30 minutes).
            known_portion : int, optional
                The percentage of requests that are known in advance (between 0 and 100). Defaults to 0.
            nb_requests : int, optional
                Maximum number of requests to generate. If None, generates requests up to sim_time.
            rng : numpy.random.Generator, optional
                Random number generator to draw from. Pass the same generator across multiple calls
                (e.g. across instances/resampling rounds) to make a whole generation run reproducible
                from one seed via numpy.random.default_rng(seed). Defaults to an unseeded generator.

        Output:
        ------------
            trips : list
                List of generated RideRequest objects.
    """
    if rng is None:
        rng = np.random.default_rng()

    request_id = start_ID
    trips = []

    # Calculate the mean inter-arrival time based on the customer rate
    mean_interarrival_time = 3600.0 / (cust_node_hour * len(network.nodes))
    interarrival_distribution = expon(scale=mean_interarrival_time)

    # Initialize the time of the first request
    t = start_time + round(interarrival_distribution.rvs(random_state=rng), 1)

    while t <= start_time + sim_time:
        # Ensure dest_id != orig_id (uniform over every other node: shift draws >= orig_id up by one)
        orig_id = rng.integers(0, len(network.nodes))
        dest_id = rng.integers(0, len(network.nodes) - 1)
        if dest_id >= orig_id:
            dest_id += 1

        orig_location = LabelLocation(str(orig_id))
        dest_location = LabelLocation(str(dest_id))

        # Calculate travel time and fare
        try:
            travel_time = durations[str(orig_id)][str(dest_id)]
        except KeyError:
            # If travel time is not available, skip this iteration
            t += interarrival_distribution.rvs(random_state=rng)
            continue

        # calculate fare based on time (travel_time is in seconds)
        fare_value = alpha + beta * travel_time

        t_ready = round(t, 3)

        # Determine the release time based on known_portion and advance_notice
        t_release = compute_release_time(t_ready, start_time, known_portion, advance_notice)

        # Generate number of passengers (between 1 and 4)
        nb_passengers = rng.integers(1, 5)

        # Create a new RideRequest object
        trip = RideRequest(str(int(request_id)),
                           origin=orig_location,
                           destination=dest_location,
                           nb_passengers=nb_passengers,
                           release_time=round(t_release, 3),
                           ready_time=t_ready,
                           due_time=100000,
                           latest_pickup=round(t_ready + time_window, 3),
                           fare=round(fare_value, 3),
                           shortest_travel_time=travel_time)

        trips.append(trip)
        request_id += 1

        # Check if the maximum number of requests has been reached
        if nb_requests is not None and len(trips) >= nb_requests:
            break

        # Generate the time for the next request
        t += round(interarrival_distribution.rvs(random_state=rng), 1)

    return trips


def create_requests_from_pool(
        trip_pool,
        node_zone_mapping,
        start_ID,
        start_time,
        sim_time,
        nb_requests,
        time_window,
        alpha=3.00,
        beta=0.03,
        advance_notice=1800,
        known_portion=100,
        rng: Optional[np.random.Generator] = None):
    """ Function: Sample real trip records into customer ride requests, instead of generating
        synthetic ones.
        Input:
        ------------
            trip_pool : pandas.DataFrame
                Pool of real trip records (see clean_real_trips.build_trip_pool), with columns
                pickup_ID, dropoff_ID, request_time_sec, trip_duration, passenger_count.
            node_zone_mapping : dict or None
                Node id -> taxi zone mapping (see network_tools.build_node_zone_mapping). Trips whose
                origin/destination fall outside every zone are skipped and topped up from the
                remaining pool. Pass None/{} to skip zone filtering entirely.
            start_ID : int
                Starting ID for the generated trip requests.
            start_time : float
                Start of the window (in seconds) to sample trips from (inclusive).
            sim_time : float
                Length of the window (in seconds) to sample trips from; trips are drawn from those
                with request_time_sec in [start_time, start_time + sim_time).
            nb_requests : int
                Number of requests to sample (without replacement) from the window.
            time_window : int
                The time window in seconds within which customers are willing to be picked up after
                their ready time.
            alpha : float, optional
                Base fare ($), independent of travel time. Defaults to 3.00, matching a real NYC taxi
                flag-drop; calibrated jointly with beta and network_tools.COST_PER_MINUTE so that
                driving cost is a meaningful (though not dominant) share of a typical trip's fare.
            beta : float, optional
                Fare rate in $/second of travel_time. Defaults to 0.03 ($1.80/min); see alpha above for
                calibration context.
            advance_notice : int, optional
                The time in seconds before the ready time that customers call to make a request.
                Defaults to 1800 (30 minutes).
            known_portion : int, optional
                The percentage of requests that are known in advance (between 0 and 100). Defaults to 100.
            rng : numpy.random.Generator, optional
                Random number generator to draw from. Defaults to an unseeded generator.

        Output:
        ------------
            trips : list
                List of generated RideRequest objects (up to nb_requests, fewer if the window's pool
                is exhausted first), sorted by ready_time with trip_id re-indexed sequentially from
                start_ID in that sorted order (not the random sampling order).
    """
    if rng is None:
        rng = np.random.default_rng()

    window_pool = trip_pool[
        (trip_pool["request_time_sec"] >= start_time) &
        (trip_pool["request_time_sec"] < start_time + sim_time)
    ].reset_index(drop=True)

    shuffled_order = rng.permutation(len(window_pool))
    candidates = []  # sampled trip fields, before ready_time sorting / trip_id assignment
    pos = 0
    while len(candidates) < nb_requests and pos < len(shuffled_order):
        row = window_pool.iloc[shuffled_order[pos]]
        pos += 1

        t_ready = round(float(row["request_time_sec"]), 3)
        t_release = compute_release_time(t_ready, start_time, known_portion, advance_notice)
        travel_time = row["trip_duration"]
        origin = LabelLocation(str(int(row["pickup_ID"])))
        destination = LabelLocation(str(int(row["dropoff_ID"])))

        if node_zone_mapping and not has_valid_zones(
                SimpleNamespace(origin=origin, destination=destination), node_zone_mapping):
            continue

        candidates.append({
            "origin": origin,
            "destination": destination,
            "nb_passengers": int(row["passenger_count"]),
            "release_time": round(t_release, 3),
            "ready_time": t_ready,
            "latest_pickup": round(t_ready + time_window, 3),
            "fare": round(alpha + beta * travel_time, 3),
            "shortest_travel_time": travel_time,
        })

    # Sort by ready_time and re-index trip_id sequentially from start_ID in that order -- ids reflect
    # chronological order rather than the random sampling order used to draw them.
    candidates.sort(key=lambda c: c["ready_time"])

    trips = []
    for offset, c in enumerate(candidates):
        trips.append(RideRequest(
            str(int(start_ID + offset)),
            origin=c["origin"],
            destination=c["destination"],
            nb_passengers=c["nb_passengers"],
            release_time=c["release_time"],
            ready_time=c["ready_time"],
            due_time=100000,
            latest_pickup=c["latest_pickup"],
            fare=c["fare"],
            shortest_travel_time=c["shortest_travel_time"]))

    return trips
