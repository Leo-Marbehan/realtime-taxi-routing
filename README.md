# Realtime Taxi routing Overview

The Taxi routing System is a solution designed to simulate and optimize taxi dispatching operations. This project requires the `multimodalsim` package for simulation and optimization of multimodal transportation systems. 

## Key Features

- Handles offline, online, and stochastic optimization scenarios for taxi dispatching.
- Supports re-optimization with destruction and repair methods.
- Includes utilities for request generation, vehicle management, and runtime measurements.

## Modules
### simulation
- **data_reader.py**: Manages input data reading and preprocessing.
- **RideRequest.py**: Defines the `RideRequest` class and associated methods.
- **run_simulation.py**: Facilitates running simulations of the taxi dispatching system under various conditions.
- **taxi_dispatcher.py**: Core module where the dispatching logic and algorithms are implemented.
### utilities
- **config.py**: Defines a configuration class (`SimulationConfig`) for customizing the simulation setup.
- **enums.py**: Enumerates objectives, solution modes, algorithms, and other constants used across the system.
- **tools.py**: Includes functions for calculating network distances, durations, costs, and visualization utilities.
- **Timer.py**: Provide the possibility to calculate runtimes.
- **test_generator.py**: Generates synthetic requests and vehicles for testing purposes.
### solvers
- **solver.py**: Baseline solver class that provides a foundation for other solver implementations.
- **Offline_solver.py**: Contains an MIP solver for solving dispatch scenarios in offline mode, when all the requests are known in advance.
- **Online_solver.py**: Contains the algorithms for solving dispatch scenarios in online mode, when all the requests are not known in advance and are received online.
- **stochastic_solver.py**: Contains the algorithms for solving dispatch scenarios using online stochastic methods.
- **Re_optimizer.py**: Contains the algorithms for re-optimizing the problem by first destroying the current solution and then repairing it.
### run_test
- **create_instances.py**: Contains functions to generate instances.
- **create_plots.py**: Contains functions for plot generation based on user-specified metrics.
- **run_test.py**: Contains functions to run a single test or multiple scenario-based tests.

## multimodalsim Package Overview

The `multimodalsim` package is a Python library for simulating multi-modal discrete event transportation systems, focusing on the dynamics between trips (passengers) and vehicles within a network. It enables the comprehensive setup, execution, and analysis of simulations to evaluate transportation strategies.

### Key Components
- **Agents**: Simulates two primary agents: trips (passengers) and vehicles. Trips are modeled as `Trip` objects with detailed attributes, including origin, destination, and timing constraints. Vehicles modeled as `Vehicle` objects, transport passengers based on the simulation's dynamics.

- **Environment**: The simulation environment where all agents operate, and events are processed.

- **Events**: Fundamental units driving the simulation's progress. Events are categorized into optimization, passenger, and vehicle events. The handling of events follows a priority queue mechanism, ensuring timely and orderly processing.

### Simulation Flow
The simulation process is categorized into three main phases: Data Preparation, Initialization of Simulation Components, and Execution of the Simulation.

### 1. Read and Prepare Data
This phase involves reading input data, visualizing the network, and preparing vehicles and trips for the simulation.

- **Data Reading**:
  Utilize a `TaxiDataReader` to read in taxi requests, vehicle information, and the network graph.
    ```python
    data_reader = TaxiDataReader(requests_file_path, vehicles_file_path, graph_file_path, vehicles_end_time=100000)
    vehicles, routes_by_vehicle_id = data_reader.get_json_vehicles()
    trips = data_reader.get_json_trips(solution_mode, time_window)
    ```

- **Network Visualization**:
  Extract and draw the network graph for a visual representation of the network's structure.
    ```python
    network_graph = data_reader.get_json_graph()
    draw_network(network_graph, graph_file_path)
    ```
    
### 2. Initialize Simulation Components
Set up the core components of the simulation, including the dispatcher, optimization model, and environment observer. Insidee Dispatcher class there are three main methods for prearing inputs, optimizing and creating route plans.

- **Dispatcher Setup**:
  Initialize a `TaxiDispatcher` with the network graph, chosen algorithm, and objective.
    ```python
    dispatcher = TaxiDispatcher(network_graph, algorithm, objective)
    ```

- **Optimization Setup**:
  Create an `Optimization` object with the dispatcher to manage trip splitting and route assignment.
    ```python
    opt = Optimization(dispatcher)
    ```

- **Environment Observer**:
  Initialize a `StandardEnvironmentObserver` for simulation monitoring and visualization.
    ```python
    environment_observer = StandardEnvironmentObserver()
    ```

### 3. Initialize and Run the Simulation
Set up the simulation with all components and execute it.

- **Simulation Initialization**:
  Create a `Simulation` object with the optimization model, trips, vehicles, routes, network graph, and environment observer. 
    ```python
    simulation = Simulation(opt, trips, vehicles, routes_by_vehicle_id, network=network_graph, environment_observer=environment_observer)
    ```

- **Simulation Execution**:
  Start the simulation process by calling the `simulate` method, managing the state and utilizing the environment observer.
    ```python
    simulation.simulate()
    ```

# Installation and Setup

## Prerequisites
- **Python 3.x**: Ensure you have Python 3.x installed on your system. You can download it from [the official Python website](https://www.python.org/).
- **Git**: Make sure Git is installed to clone the repository and initialize submodules.

## Setting Up the Environment
1. **Clone the Repository with Submodules**: Start by cloning the repository to your local machine and initializing its submodules. Use the following command:

    ```bash
    git clone --recurse-submodules https://github.com/lab-core/realtime-taxi-routing.git
    cd realtime_taxi_routing
    ```

    If you forgot to use `--recurse-submodules`, you can initialize the submodules later:

    ```bash
    git submodule update --init --recursive
    ```

2. **Create a Virtual Environment (Optional but Recommended)**:
It's a best practice to create a virtual environment for your project to avoid conflicts with system-wide Python packages. Use the following commands to navigate to the project directory and create the environment. Replace [project-directory] with the address of the place you have saved the project.
    ```bash
    cd [project-directory]/realtime-taxi-routing
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate` 

3. **Install the Multimodal Simulator**: Use the following command to install the `multimodal-simulator`:

    ```bash
    pip install ./multimodal-simulator/python
    ```
4. **Instal Required Dependencies**: Install the dependencies for the project, including the `multimodalsim` submodule:
    ```bash
    pip install .
    ```

    This will automatically install all required Python packages and the`multimodal-simulator`, submodule.
     
5. **Using the Virtual Environment as the Python Interpreter**: After setting up the virtual environment, it's essential to ensure that your IDE or code editor is configured to use the Python interpreter from the virtual environment. Follow the steps below to select the Python interpreter from the virtual environment

#### PyCharm:
1. Open your project in PyCharm.
2. Go to `File` > `Settings`.
3. Navigate to `Project: your-project-name` > `Python Interpreter`.
4. Click on `Add Interpreter`, and choose the Python interpreter located in your virtual environment (typically under the `venv/bin/python` path).

#### Visual Studio Code:
1. Open your project folder in VS Code.
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the command palette.
3. Type `Python: Select Interpreter` and select the command.
4. Choose the interpreter from your virtual environment (usually found in the `.venv` or `venv` folder).


## Installing Gurobi and Obtaining a Student License

Gurobi is an optimization solver used in various industries and academic research. Follow the steps below to install Gurobi and obtain a student license:

1. **Register and Download Gurobi**:
   - Visit the [Gurobi Download Center](https://www.gurobi.com/downloads/gurobi-software/).
   - Register for an account if you don't already have one. Ensure you use your academic email address to qualify for the free academic license.
   - Download the appropriate version of Gurobi for your operating system.

2. **Install Gurobi**:
   - Follow the [Software Installation Guide](https://support.gurobi.com/hc/en-us/articles/14799677517585) specific to your operating system to install Gurobi.

3. **Obtain a Free Academic License**:
   - Visit the [Gurobi License Center](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
   - Apply for a free academic license. You will need to provide your university email address and verify your academic status.
   - Follow the instructions to activate your license. Typically, this involves running a command in your terminal or command prompt.

4. **Set Up the Gurobi Environment**:
   - Ensure that your Python environment is set up to recognize Gurobi. If you're using a virtual environment for your project, you may need to update it with Gurobi's Python bindings.
   ```bash
    python -m pip install gurobipy

For the most up-to-date and detailed installation instructions, please refer to the [official Gurobi documentation](https://www.gurobi.com/documentation/).

**Note**: The process for obtaining a Gurobi license may change, and the terms of use for Gurobi software are subject to Gurobi's licensing agreement. Ensure you comply with all license terms and conditions.

## Testing the simulator
### Configuration File
To run the simulations, the inputs are provided in a JSON file(`inputs.json`). This file specifies whether to run a single test or multiple scenario-based tests. The `task_type` key determines the mode:

- **single_test:** Executes one test instance.
- **scenarios:** Runs a series of predefined scenario sets.
    
### Running a Single Test
To run a single test:

1. Configure your `inputs.json` file by specifying the `task_type` as `"single_test"` and providing the required parameters within the `"single_test"` block.
2. Navigate to the root project directory and run:
   ```bash
   cd [project-directory]/realtime-taxi-routing
   python main.py

This executes a single test as defined in your `inputs.json`.

### Running Scenarios
1. Set `task_type` to `"scenarios"` in the `inputs.json` file.
2. Determine a scenario to run 
3. Execute the scenario(s) using:
   ```bash
   python main.py -sn <SCENARIO_NAME>
   ```
    Replace `<SCENARIO_NAME>` with the name of the scenario you wish to run, for example:
   ```bash
   python main.py -sn TP2
   ```
    This will run all parameter combinations defined for the given scenario and save the results to a CSV file in the `data/Instances/Results` directory.
### Creating Plots
1. Set `task_type` to `"create_plot"` in the `inputs.json` file.
2. Determine the scenario to create the plot(s), ensure results are available and run: 
   ```bash
   python main.py -sn <SCENARIO_NAME>
   ```
### Generating Instances
Configure parameters in `inputs.json` and run:
```bash
   python main.py
   ```
## Configuration of Parameters
The following parameters can be specified in `inputs.json`:
### task_type
- **Description:** Determines the type of execution.
- **Options:** `single_test`, `scenarios`

### instances
- **Description:** Folder name of the instance to test.
- **Example:** `"Med_1"`

### objectives
- **Description:** Defines the optimization objective to achieve (Default: `"total_customers"`).
- **Options:**
  - `"total_profit"`: Maximizes the total profit of served requests.
  - `"waiting_time"`: Minimizes the total wait time of served requests.
  - `"total_customers"`: Maximizes the total number of served customers.


### algorithms
- **Description:** Selects the algorithm to optimize the dispatch plan (Default: `"mip_solver"`).
- **Options:**
  - `"mip_solver"`: Uses the Gurobi MIP solver to solve the problem.
  - `"greedy"`: Greedy approach to assign requests to vehicles.
  - `"random"`: Random approach to assign requests to vehicles.
  - `"ranking"`: Ranking approach to assign requests to vehicles.
  - `"consensus"`: Consensus approach to assign requests to vehicles.
  - `"re_optimize"`: Re-optimize the solution based on destroy and repair.

### solution_mode
- **Description:** Sets how requests become available (Default: `"offline"`).
- **Options:**
  - `"offline"`: All requests are known at the start.
  - `"fully_online"`: Release time is equal to the ready time for all requests.
  - `"advance_notice"`: Requests become known 30 minutes before the ready time.
  - `"partial_online"`: A percentage of requests are known in adavnce.
  - `"custom_scenario"`: A mix of advance_notice and partial_online.

### known-portion
- **Description:** Percentage of requests that are known in advance (0-100%).
- **Applicable if:** `solution_mode` is `"partial_online"` or `"custom_scenario"`.

### time_window
- **Description:** Time window (in minutes) to serve each request (Default: `3`).

### nb_scenario
- **Description:** Number of scenarios for the consensus algorithm (Default: `10`).
- **Applicable if:** `algorithms` is `"consensus"`.

### consensus_params
- **Description:** Type of consensus approach (Default: `"quantitative"`).
- **Applicable if:** `algorithms` is `"consensus"`.
- **Options:**
  - `"qualitative"`: Increment a counter for the best request in each scenario.
  - `"quantitative"`: Credit the best request with the optimal solution value.

### dest_method
- **Description:** Destruction method for re-optimization (Default: `"default"`).
- **Applicable if:** `algorithms` is `"re_optimize"`.
- **Options:**
  - `"default"`: Default destruction method (Complete re-optimization).
  - `"fix_variables"`: Fix some of the variables in the model
  - `"fix_arrivals"`: Fix a time window around the arrival time
  - `"bonus"`: Arbitrary destroy method as bonus

With default parameters you should obtain the following results (except for the optimization_time):
  ```console
    Attribute              | Value
    ----------------------+---------------------------
    Key                    | Med_1__3__total_customers
    Test                   | Med_1
    # Trips                | 72
    # Vehicles             | 20
    Solution Mode          | offline
    Known portion (%)      | 100
    Time window (min)      | 3
    Algorithm              | MIP_Solver
    Objective type         | total_customers
    Objective value        | 61.0
    # Served customers     | 61
    Average profit         | 1017.79
    Average waiting time   | 56.46
    % of Service           | 84.7
    optimization_time      | 0.251
  ```


Below is the explanation for the parameters used to generate instances:

- **num_suburbs (int)**:
  Determines the number of suburban areas surrounding the main city.

- **suburb_width (int)**:
  Indicates the size of each suburban area as a square grid. For example, `suburb_width = 4` creates a 4x4 grid of nodes for each suburb.

- **city_width (int)**:
  Defines the size of the central city as a square grid. For example, `city_width = 8` results in an 8x8 grid for the main city.

- **block_distance (float)**:
  Specifies the distance (in meters) between adjacent nodes within the city and the suburbs. Larger values produce a more spatially extensive network, increasing travel times between nodes.
