"""Shared structural types for the solver classes.

`TripLike`/`VehicleLike` describe the duck-typed attributes solvers actually access on
`RideRequest`/`Vehicle` instances (and on the scenario-generated trips used by the stochastic
solver), without depending on the concrete `multimodalsim` classes.
"""
from typing import Dict, Protocol, runtime_checkable


@runtime_checkable
class LocationLike(Protocol):
    label: str


@runtime_checkable
class TripLike(Protocol):
    id: str
    origin: LocationLike
    destination: LocationLike
    ready_time: float
    latest_pickup: float
    fare: float
    shortest_travel_time: float


@runtime_checkable
class VehicleLike(Protocol):
    id: str


TravelTimeMatrix = Dict[str, Dict[str, float]]
