from multimodalsim.simulator.request import Trip
from typing import Optional
from multimodalsim.simulator.stop import Location


class RideRequest(Trip):
    """
    RideRequest extends the Trip class with additional attributes specific to ride requests.
        Additional Attributes:
        ----------------------
        fare: float
           fare paid for serving request.
        latest_pickup: float
           The latest possible time for passenger pickup.
        shortest_travel_time: float
           travel time between its origin and destination
    """

    def __init__(self,
                 id: str,
                 origin: Location,
                 destination: Location,
                 nb_passengers: int,
                 release_time: float,
                 ready_time: float,
                 due_time: float,
                 name: Optional[str] = None,
                 latest_pickup : Optional[float] = None,
                 fare: Optional[float] = None,
                 shortest_travel_time: Optional[float] = None):

        # Call the constructor of the parent class (Request)
        super().__init__(id, origin, destination, nb_passengers, release_time, ready_time, due_time, name)

        # Add the additional attribute
        self.__fare = fare
        self.__latest_pickup = latest_pickup
        self.__shortest_travel_time = shortest_travel_time



    @property
    def latest_pickup(self) -> float:
        """Getter for the latest_pickup attribute."""
        return self.__latest_pickup

    @property
    def fare(self) -> float:
        """Getter for the fare attribute."""
        return self.__fare


    @property
    def shortest_travel_time(self) -> float:
        """Getter for the shortest_travel_time attribute."""
        return self.__shortest_travel_time


    @latest_pickup.setter
    def latest_pickup(self, latest_pickup: float) -> None:
        """Setter for the latest_pickup_time attribute."""
        self.__latest_pickup = latest_pickup



