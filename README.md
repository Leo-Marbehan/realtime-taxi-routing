# Real-time Taxi Routing

A simulator and optimization testbed for taxi dispatching. Ride requests arrive over time, a fleet
of vehicles must be assigned to serve them, and different algorithms are compared on how well they
do it under varying amounts of advance knowledge.

The project builds on two separate modules, each installed on its own:

- **`p-multimodal-simulator`** -- provides the `multimodalsim` package, the discrete-event engine
  that drives the simulation (environment, events, vehicles, trips, routes).
- **`p-multimodal-viewer`** -- provides `multimodalsim_viewer` and the `viewer` command, a web
  interface that replays a saved simulation or shows one live as it runs.

## Key Features

- Offline, online, and online-stochastic dispatching, plus a learned (reinforcement learning) policy.
- Re-optimization with destroy-and-repair methods.
- Instance generation, scenario batches, plotting, and a browser-based viewer.

## Project Layout

All source lives under `src/`.

### simulation
- **data_reader.py**: Reads requests, vehicles and the network graph, and applies the solution mode.
- **ride_request.py**: Defines the `RideRequest` class and its attributes.
- **run_simulation.py**: Builds and runs one simulation end to end.
- **taxi_dispatcher.py**: Dispatch entry point; selects the solver and turns assignments into routes.

### solvers
- **solver.py**: Base class shared by all solvers (vehicle state, feasibility, objective value).
- **offline_solver.py**: Gurobi MIP model for the offline case, where all requests are known upfront.
- **online_solver.py**: Greedy, random and ranking policies for requests arriving over time.
- **stochastic_solver.py**: Consensus algorithms, which sample future scenarios before deciding.
- **re_optimizer.py**: Destroy-and-repair re-optimization around an existing solution.
- **dqn_solver.py**: Deep Q-Network policy that scores every (vehicle, request) pair and commits the
  best, or defers.

### utilities
- **config.py**: `SimulationConfig` and the per-algorithm config objects.
- **config_tools.py**: Helpers for building configs and identifying instances.
- **enums.py**: `Objectives`, `Algorithm`, `SolutionMode`, and the other enumerations.
- **network_tools.py**: Network distances, durations, driving costs and related helpers.
- **create_scenario.py**: Samples synthetic requests, used by the consensus algorithms.
- **paths.py**: Canonical data/result directory paths.
- **types.py**: Shared type aliases.
- **timer.py**: Runtime measurement.
- **result_columns.py**, **reporting_tools.py**: Result column names and table formatting.

### execution
- **run_tests.py**: Runs a single test or a batch of scenario combinations.
- **run_viewer.py**: Runs a simulation through the viewer (saved for replay, or streamed live).
- **train_dqn.py**: Training loop for the DQN policy.
- **reference_test.py**: Regression checks against a small fixed instance.
- **inputs.json**: The configuration file that drives everything below.

### generation
- **create_instance.py**: Generates instances from the real trip pool.
- **clean_real_trips.py**, **convert_instances_to_csv.py**, **convert_network_to_json.py**: Data
  preparation utilities.

### reporting
- **create_plots.py**: Plot generation from saved results.
- **visualization.py**: Network and route drawing.

## The multimodalsim Engine

`multimodalsim` simulates multi-modal discrete-event transportation systems, focusing on the
interaction between trips (passengers) and vehicles on a network.

- **Agents**: `Trip` objects (origin, destination, timing constraints) and `Vehicle` objects.
- **Environment**: Where agents live and events are processed.
- **Events**: The unit of progress. Optimization, passenger and vehicle events are processed from a
  priority queue ordered by time and priority. Nothing changes between two consecutive events.
- **Optimization / Dispatcher**: The dispatcher is invoked at optimization events; it assigns trips
  to vehicles and decides each vehicle's next stops.

The flow is: read and prepare the data, build the components, then run.

```python
# 1. Read data
data_reader = TaxiDataReader(requests_file_path, vehicles_file_path)
data_reader.load_graph(graph_file_path)
trips = data_reader.get_trips(config)
vehicles, routes_by_vehicle_id = data_reader.get_vehicles(vehicles_end_time)

# 2. Build the dispatcher and wrap it in an Optimization object
dispatcher = TaxiDispatcher(network_graph, config)
opt = Optimization(dispatcher)

# 3. Run
simulation = Simulation(opt, trips, vehicles, routes_by_vehicle_id, network=network_graph)
simulation.simulate()
```

`run_simulation.py` does all of this for you -- the snippet is only to show the shape.

## The Viewer

`p-multimodal-viewer` serves a browser interface showing vehicles moving on the map, their routes,
and each request's state over time. It can replay a saved run or display one live while it is still
running, in which case you can also pause and resume the simulation from the browser.

The viewer reads its settings from a `.env` file **in the directory you run from**. The one shipped
in the project root sets the input/output folders and enables route reordering support:

```
INPUT_DATA_DIRECTORY_PATH=data/Viewer_Instances
OUTPUT_DATA_DIRECTORY_PATH=data/Viewer_Output
ENABLE_STOP_REORDERING=true
ENABLE_LEG_REORDERING=true
```

Keep those reordering flags set to `true`: re-optimizing algorithms legitimately change a vehicle's
planned stops, and with the flags off the viewer treats that as an error and stops.

# Installation and Setup

## Prerequisites
- **Python 3.x**: Download it from [the official Python website](https://www.python.org/).
- **Git**: Needed to clone the repository and its submodules.

## Setting Up the Environment

1. **Clone the repository with its submodules**. The simulator and the viewer are both submodules,
   so `--recursive` matters:

    ```bash
    git clone --recursive https://github.com/RTOpt/realtime-taxi-routing.git
    cd realtime-taxi-routing
    ```

    If you already cloned without `--recursive`:

    ```bash
    git submodule update --init --recursive
    ```

2. **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install the simulator, the viewer, and the project**:

    ```bash
    pip install ./p-multimodal-simulator/python    # the multimodalsim engine
    pip install ./p-multimodal-viewer/python       # the viewer and its `viewer` command
    pip install -e .                               # this project and its dependencies
    ```

    Install this project with `-e` (editable). A plain `pip install .` copies `src/` into the
    environment, and that copy will then shadow your own edits.

4. **Point your IDE at the virtual environment** so it uses the same interpreter.

#### PyCharm
1. `File` > `Settings` > `Project: your-project-name` > `Python Interpreter`.
2. `Add Interpreter`, then choose the one in your virtual environment (`venv/bin/python`).

#### Visual Studio Code
1. Press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS).
2. Run `Python: Select Interpreter` and choose the one in `venv`.

## Installing Gurobi and Obtaining a Student License

Gurobi solves the MIP models used by `mip_solver`, `re_optimize` and `consensus`.

1. **Register and download**: [Gurobi Download Center](https://www.gurobi.com/downloads/gurobi-software/).
   Use your academic email to qualify for the free academic license.
2. **Install**: follow the [Software Installation Guide](https://support.gurobi.com/hc/en-us/articles/14799677517585)
   for your operating system.
3. **Get a free academic license**: [Gurobi License Center](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
   Apply, verify your academic status, then activate it as instructed.
4. **Install the Python bindings** in your environment:

    ```bash
    python -m pip install gurobipy
    ```

See the [official Gurobi documentation](https://www.gurobi.com/documentation/) for the current
instructions. The licensing terms are Gurobi's and may change.

# Running the Simulator

Activate your virtual environment first, and run every command from the project root:

```bash
cd [project-directory]/realtime-taxi-routing
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

## Configuration File

Everything is driven by `src/execution/inputs.json`. Its `task_type` key selects what happens:

| `task_type` | What it does |
|---|---|
| `single_test` | Runs one instance with one configuration. |
| `scenarios` | Runs a batch of parameter combinations (needs `-sn <SCENARIO_NAME>`). |
| `create_plot` | Plots saved results for a scenario (needs `-sn <SCENARIO_NAME>`). |
| `viewer_simulate` | Runs one simulation for the viewer. |
| `create_instance` | Generates new instances. |
| `train_dqn` | Trains the DQN policy. |

## Quick Sanity Check

Before running the full instances, you can check a small fixed instance
(`data/Debug_Instances/Debug_1`) against known-good results:

```bash
python -m unittest src.execution.reference_test.ReferenceTestCase.test_mip_solver
python -m unittest src.execution.reference_test.ReferenceTestCase.test_greedy
python -m unittest src.execution.reference_test.ReferenceTestCase.test_dqn_structural
```

These compare against reference numbers, so they are useful for catching obviously broken logic
early. `mip_solver` is the most meaningful of the three, since the offline optimum is well defined.
For `greedy`, a different but still valid way of breaking ties between equally good vehicles can
produce different numbers without being wrong.

## Running a Single Test

1. Set `task_type` to `"single_test"` and fill in the `"single_test"` block of `inputs.json`.
2. Run:

    ```bash
    python -m src.main
    ```

A single test can also be run from the command line:

```bash
python -m src.Run_Example -i <INSTANCE> -o <OBJECTIVE> -a <ALGORITHM> -m <SOLUTION_MODE> -tw <TIME_WINDOW>
```

- `-i, --instance`: Instance folder (e.g. `Instance_1`).
- `-o, --objective`: `total_profit`, `waiting_time`, `total_customers`, `multi_objective`.
- `-a, --algorithm`: `mip_solver`, `greedy`, `random`, `ranking`, `consensus`, `re_optimize`, `dqn`.
- `-m, --sol-mode`: `offline`, `fully_online`, `advance_notice`, `partial_online`, `custom_scenario`.
- `-tw, --time-window`: Time window **in seconds** (e.g. `240`).

Algorithm-specific options:

- `-kp, --known-portion`: Percentage of requests known in advance (0-100).
- `-an, --advance-notice`: How long before its ready time a request is revealed, in seconds.
- `-ns, --nb-scenario`: Number of scenarios for consensus.
- `-cp, --consensus-params`: `qualitative` or `quantitative`.
- `-dm, --dest-method`: `default`, `fix_variables`, `fix_arrivals`, `bonus`.
- `-wp, --weight`: Weight of the first objective under `multi_objective` (0-1).
- `-s, --seed`: Random seed, where the algorithm uses one.

See `Run_Example.py` for the full list.

## Running Scenarios

1. Set `task_type` to `"scenarios"`.
2. Run:

    ```bash
    python -m src.main -sn <SCENARIO_NAME>
    ```

Available scenarios: `initial_test`, `TP1`, `TP1_Multi`, `TP2`, `TP3`, `TP4`, `TP4_scenario`, `TP5`.

`TP2`, `TP4` and `TP5` are *groups*: running one runs all of its sub-scenarios and combines the
results into a single `<GROUP>_simulation_results.csv`. The mapping lives in `inputs.json` under
`"scenario_groups"`, so it is visible in the configuration rather than hidden in code.

Results are written to `data/Real_Instances/Results`.

**Run `TP1` first.** The other scenarios report a `Competitive Ratio` against TP1's offline-optimal
results. If TP1 has not been run for the instance, time window and objective being tested -- or its
saved results no longer match the instance files on disk -- the run stops with an error telling you
to re-run TP1, rather than quietly comparing against the wrong baseline.

## Creating Plots

1. Set `task_type` to `"create_plot"`.
2. Make sure `<SCENARIO_NAME>_simulation_results.csv` exists in `data/Real_Instances/Results`; do
   not rename or move it.
3. Run:

    ```bash
    python -m src.main -sn <SCENARIO_NAME>
    ```

## Using the Viewer

The viewer runs as a separate process and the simulation connects to it.

1. Start the viewer (opens your browser):

    ```bash
    viewer start
    ```

2. In another terminal, set `task_type` to `"viewer_simulate"`, configure the `"single_test"` block,
   and run:

    ```bash
    python -m src.main
    ```

Both commands must run from the project root so they share the same `.env`.

By default the run is saved and can be replayed from the viewer afterwards. Adding `"live": true` to
the `"single_test"` block streams it to the viewer while it runs, and lets you pause and resume it
from the browser. If the viewer is not running, the simulation prints a connection warning and
continues normally.

Use `viewer stop` to shut it down.

## Generating Instances

Set `task_type` to `"create_instance"`, configure the `"create_instance"` block, and run
`python -m src.main`. Parameters:

- **num_instances**: How many instances to generate.
- **nb_requests**: Requests per instance.
- **start_time**: Start of the request window, in seconds from midnight (e.g. `25200` = 07:00).
- **sim_time**: Length of the request window, in seconds.
- **alpha**, **beta**: Fare parameters (base fare and per-unit rate).
- **known_portion**: Percentage of requests known in advance at generation time.
- **advance_notice**: How far ahead requests are revealed, in seconds.
- **nb_vehicles**: Fleet size.
- **seed**: Random seed, for reproducible instances.

## Training the DQN

Set `task_type` to `"train_dqn"` and run `python -m src.main`. The `"train_dqn"` block lists one
entry per model, each with its own objective, training instances and output checkpoint path.
Checkpoints are written to `data/Models` and are loaded automatically when you later run the `dqn`
algorithm with the matching `model_path`.

# Configuration Parameters

### instances
Folder name of the instance to run, e.g. `"Instance_1"`. The real instances are `Instance_1` to
`Instance_20`; `Tiny_1` and `Debug_1` are small instances useful for quick checks.

### objectives
- `"total_profit"`: Maximize total profit of served requests.
- `"waiting_time"`: Minimize total waiting time.
- `"total_customers"`: Maximize the number of served customers.
- `"multi_objective"`: Weighted combination, controlled by `weight`.

### algorithms
- `"mip_solver"`: Gurobi MIP model (offline).
- `"greedy"`, `"random"`, `"ranking"`: Online heuristics.
- `"consensus"`: Online stochastic, sampling future scenarios.
- `"re_optimize"`: Destroy-and-repair re-optimization.
- `"dqn"`: Learned policy; requires a trained checkpoint.

### solution_mode
- `"offline"`: All requests known at the start.
- `"fully_online"`: Each request is revealed at its ready time.
- `"advance_notice"`: Requests are revealed a fixed time (default 1800 s = 30 min) before ready time.
- `"partial_online"`: A percentage is known in advance, the rest at their ready time.
- `"custom_scenario"`: A mix of advance notice and partial knowledge.

### known_portion
Percentage of requests known in advance (0-100). Applies to `"partial_online"` and
`"custom_scenario"`.

### time_window
Time window **in seconds** within which a request must be picked up (e.g. `240`).

### nb_scenario
Number of scenarios sampled by `"consensus"` (default `10`).

### consensus_params
- `"qualitative"`: Count votes for the best request in each scenario.
- `"quantitative"`: Weight each vote by the scenario's objective value.

### destroy_method
Applies to `"re_optimize"`:
- `"default"`: Complete re-optimization.
- `"fix_variables"`: Fix some model variables.
- `"fix_arrivals"`: Fix a time window around arrival times.
- `"bonus"`: Your own destroy method.

### model_path
Checkpoint file used by `"dqn"`, e.g. `"data/Models/dqn_model_profit.pt"`.
