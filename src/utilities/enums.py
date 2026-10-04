from enum import Enum


class Objectives(Enum):
    """ objective functions:
        - total_Profit: total profit of served requests
        - waiting_time: total wait time of served requests
        - total_customers: total number of served customers
        - multi_objective: weighted sum of total profit and negative total wait time
    """
    TOTAL_PROFIT = "total_profit"
    TOTAL_REVENUE = "total_revenue"
    TOTAL_COST = "total_cost"
    TOTAL_CUSTOMERS = "total_customers"
    WAIT_TIME = "waiting_time"
    TOTAL_EMPTY_TRAVEL_TIME = "total_empty_travel_time"
    MULTI_OBJECTIVE = "multi_objective"


class SolutionMode(Enum):
    """ solution modes:
        - offline : all the requests revealed (known) at the start (release time = 0 for all requests)
        - fully_online : All requests are released exactly at their ready time
        - advance_notice : All requests are known a fixed amount of time before their ready time.
        - partial_online : a random percentage of requests are known at the start and for the rest release time is equal to the ready time
        - custom_scenario : others
    """
    OFFLINE = "offline"
    FULLY_ONLINE = "fully_online"
    ADVANCE_NOTICE = "advance_notice"
    PARTIAL_ONLINE = "partial_online"
    CUSTOM_SCENARIO = "custom_scenario"


class Algorithm(Enum):
    """ Algorithm used to optimize the plan:
        - MIP_SOLVER : using the Gurobi MIP solver to solve the problem
        - GREEDY : greedy approach to assign arrival requests to vehicles
        - RANDOM : random algorithm to assign arrival requests to vehicles
        - RANKING : ranking method to assign arrival requests to vehicles
        - CONSENSUS : consensus online stochastic algorithm to assign arrival requests to vehicles
        - RE_OPTIMIZE: Algorithm to re-optimize the solution based on destroy and repair
        - DQN : learned Q-value based global-pair selection (see solvers/dqn_solver.py)
    """
    MIP_SOLVER = "MIP_Solver"
    GREEDY = "Greedy"
    RANDOM = "Random"
    RANKING = "Ranking"
    CONSENSUS = "Consensus"
    RE_OPTIMIZE = "Re_Optimize"
    DQN = "DQN"

class ConsensusParams(Enum):
    """ The type of consensus algorithm:
        - QUALITATIVE : A counter is incremented for the best request to assign at each scenario.
        - QUANTITATIVE : The best request to assign is credited by the optimal solution value, rather than merely incrementing a counter.
    """
    QUALITATIVE = "Qualitative"
    QUANTITATIVE = "Quantitative"


class DestroyMethod(Enum):
    """ Method used for destruction in RE_OPTIMIZE algorithm
        - DEFAULT: Default destruction method (Complete re-optimization)
        - FIX_VARIABLES: fix some of the variables in the model
        - FIX_ARRIVALS: fix a time window around the arrival time
        - BONUS: arbitrary destroy method as bonus
    """
    DEFAULT = "default"
    FIX_ARRIVALS = "fix_arrivals"
    FIX_VARIABLES = "fix_variables"
    BONUS = "bonus"