"""Single source of truth for every result/metric column-name string used across the results and
plotting pipeline: TaxiDispatcher.extract_output() (simulation/taxi_dispatcher.py), run_simulation.py's
info_dict, run_tests.py's display table, reporting_tools.py's table/description helpers, and every
plot function in reporting/visualization.py.

Every one of those imports from here instead of retyping the literal strings -- a rename becomes
one edit here instead of a hunt across ~8 files, and a typo in any one of them becomes an import
error or a NameError instead of a column that silently never matches.

Values are the *exact current* strings already written to results CSVs -- this module only
centralizes them, it does not change what gets written. Two known internal inconsistencies
(TOTAL_COST has no space before "($)" unlike its AVG_PROFIT/AVG_REVENUE siblings;
EMPTY_TRAVEL_TIME_MIN uses "min" while its wait-time siblings use "(s)") are preserved as-is
deliberately -- "fixing" either would change the column header written to every future results
CSV, a real behavior change beyond centralizing names.

Legacy CSVs under data/Instances/Results_main/ use a different, older naming scheme entirely
(e.g. "Time window (min)", "Weight profit") and are NOT covered by these constants -- they predate
this module and remain unreadable by the current plot functions, exactly as before this refactor.
"""

# -- Written by TaxiDispatcher.extract_output() --
ALGORITHM = 'Algorithm'
OBJECTIVE_TYPE = 'Objective type'
OBJECTIVE_VALUE = 'Objective value'
NUM_SERVED = '# Served customers'
NUM_REJECTED = '# Rejected customers'
AVG_PROFIT = 'Average profit ($)'
TOTAL_PROFIT = 'Total profit'
AVG_REVENUE = 'Average revenue ($)'
TOTAL_REVENUE = 'Total revenue'
TOTAL_COST = 'Total cost($)'  # no space before "($)" -- inconsistent with AVG_PROFIT/AVG_REVENUE, preserved as-is
TOTAL_WAIT_TIME_S = 'Total wait time (s)'
AVG_WAIT_TIME_S = 'Avg. wait time (s)'
EMPTY_TRAVEL_TIME_MIN = 'Empty travel time (min)'  # unit is "min", unlike its "(s)" wait-time siblings, preserved as-is
PCT_SERVICE = '% of Service'
RUNTIME_S = 'runtime (s)'

# -- Written by run_simulation.py's info_dict --
KEY = 'Key'
TEST = 'Test'
NUM_TRIPS = '# Trips'
NUM_TRIPS_LEGACY = 'Trips'  # older fallback name print_result_as_table still checks for
NUM_VEHICLES = '# Vehicles'
NUM_VEHICLES_LEGACY = 'Vehicles'  # older fallback name print_result_as_table still checks for
SOLUTION_MODE = 'Solution Mode'
TIME_WINDOW_S = 'Time window (s)'
INSTANCE_FINGERPRINT = 'Instance Fingerprint'
KNOWN_PORTION_PCT = 'Known portion (%)'
NUM_SCENARIOS = '# Scenarios'
NB_REQUESTS = 'nb_requests'
CONSENSUS_TYPE = 'Consensus type'
DESTROY_METHOD = 'Destroy Method'
WEIGHT = 'weight'
WEIGHT_LEGACY = 'Weight profit'  # older fallback name multi_plot still checks for
COMPETITIVE_RATIO = 'Competitive Ratio'

# -- Derived display-only columns, computed in reporting_tools.py, consumed in visualization.py --
SOLUTION_DESCRIPTION = 'Solution Description'
ALGORITHMS = 'Algorithms'

# The fixed 11-column order shared by run_tests.py's scenario-batch table and
# reporting_tools.print_result_as_table's single-result table -- previously two independently
# hand-typed copies of the same list.
DISPLAY_COLUMNS = [
    TEST, NUM_TRIPS, NUM_VEHICLES, SOLUTION_MODE, TIME_WINDOW_S, WEIGHT,
    ALGORITHM, OBJECTIVE_TYPE, OBJECTIVE_VALUE, PCT_SERVICE, RUNTIME_S,
]
