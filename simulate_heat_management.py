"""
Simulation of Server Load and Log Replica Management

This simulation models the distribution of log replicas across a set of servers under varying load conditions,
aiming to evaluate the efficiency of different server selection algorithms. The simulation environment is built
using the simpy library to mimic real-world server load fluctuations where replicas move from one server
to another or load increases / decreases over time and log replica distribution over time.

Servers and Log Replicas:
- The simulation considers N servers, each initially with zero load and zero log replicas.
- Log replicas, each with a variable write per second (WPS) rate and size, are distributed across these servers.
- The goal is to manage server load and log replica distribution efficiently, minimizing load imbalance.

Algorithms Evaluated:
1. Random Choice: Selects a random server for each new log replica.
2. Best of Two Choice: Selects the better of two randomly chosen servers based on a combined metric of load and replica count.
3. Best of Three Choice: Extends the Best of Two Choice algorithm by selecting the best server out of three randomly chosen ones.
4. Best Choice: Selects the server with the lowest combined metric of load and replica count across all servers. This is computationally expensive

Key Functions:
- `add_log_replica`: Adds a new log replica to a server, updating its load and replica count.
- `get_wps_for_time`: Determines the WPS for a new log replica based on the current simulation time, simulating varying load conditions.
- `simulate_dynamic_load`: Simulates dynamic changes in server load and replica count over time.
- `update_cache`: Periodically updates a cache with the current state of each server to simulate realistic information delay.
- `combined_metric_cached`: Calculates a combined metric for server selection using cached server states.
- `plot_grouped_comparison`: Plots a grouped bar chart to compare metrics (e.g., load or replica count) across different algorithms.
- `plot_stats_comparison`: Plots a bar chart to compare statistical metrics (e.g., standard deviation of load) across algorithms.
- `run_simulation`: Runs the simulation for a specified server selection algorithm and stores the results for comparison.

The simulation tracks the final load and log replica count for each server, along with the standard deviation of these metrics,
to evaluate and compare the performance of the different server selection algorithms.
"""

import random
import simpy
import matplotlib.pyplot as plt
import numpy as np

# Constants
N = 120  # Number of servers
TOTAL_LOG_REPLICAS = 20000  # Total number of log replicas to be added
SIM_TIME = 10000  # Simulation time, adjust as needed
WPS_MIN = 0  # Minimum writes per second for a log replica
WPS_MAX = 1000  # Maximum writes per second for a log replica

# Initialize servers and tracking variables
servers = [0] * N # Initialize server load to 0 for all servers
log_replicas_per_server = [0] * N # Initialize log replica count to 0 for all servers
load_history = [[] for _ in range(N)] # Track load history for each server
log_replicas_history = [[] for _ in range(N)]  # Track log replica history for each server
server_state_cache = [{'load': 0, 'replicas': 0} for _ in range(N)]

# Global dictionaries to store final results for comparison
final_loads = {}
final_replicas = {}
std_dev_loads = {}
std_dev_replicas = {}

def add_log_replica(env, server_index, wps, size_in_kb):
    """
    Add a log replica with variable traffic to a server and update history.
    This function checks if the server can accept more replicas and if the new load is within the limit.
    """

    # Convert size from KB to load units
    load_adjustment = size_in_kb / 1024  # Assuming 1KB = 1 load unit, 1MB = 1024 load units,
    new_load = servers[server_index] + (wps * load_adjustment) # Calculate new load after adding the log replica

    # Check if the server can accept more replicas and if new load is within the limit
    if log_replicas_per_server[server_index] < 200 and new_load <= 200000:
        servers[server_index] = new_load  # Update server load
        log_replicas_per_server[server_index] += 1 # Increment log replica count
        # Update histories
        load_history[server_index].append(servers[server_index])
        log_replicas_history[server_index].append(log_replicas_per_server[server_index])
    else:
        # Optionally handle the case where a server cannot accept more replicas or would exceed load limit
        pass


def random_choice(env):
    """
    Random choice algorithm for server selection.
    This function selects a random server for adding a new log replica.
    """
    for _ in range(TOTAL_LOG_REPLICAS):
        yield env.timeout(1)  # Simulate time between log replica additions
        wps = get_wps_for_time(env.now)  # Determine WPS for the new log replica
        size_in_kb = random.choice([1024, 1])  # Random size between 1 KB to 1 MB
        chosen_index = random.randint(0, N-1)  # Select a random server
        add_log_replica(env, chosen_index, wps, size_in_kb)

def best_of_two_choice(env):
    """
    Best of two choice algorithm for server selection.
    This function selects the best server out of two randomly chosen servers based on a combined metric.
    """
    for _ in range(TOTAL_LOG_REPLICAS):
        yield env.timeout(1)  # Simulate time between log replica additions
        wps = get_wps_for_time(env.now)  # Determine WPS for the new log replica
        size_in_kb = random.choice([1024, 1]) # Random size between 1 KB to 1 MB
        a, b = random.sample(range(N), 2) # Select two random servers
        chosen_index = a if combined_metric_cached(a) < combined_metric_cached(b) else b
        add_log_replica(env, chosen_index, wps, size_in_kb)

def best_of_three_choice(env):
    """Best of three choice algorithm for server selection."""
    for _ in range(TOTAL_LOG_REPLICAS):
        yield env.timeout(1)  # Simulate time between log replica additions
        wps = get_wps_for_time(env.now)  # Determine WPS for the new log replica
        size_in_kb = random.choice([1024, 1])  # Random size between 1 KB to 1 MB
        a, b, c = random.sample(range(N), 3)  # Select three random servers
        # Choose the server with the best (lowest) combined metric
        chosen_index = min([a, b, c], key=combined_metric_cached)
        add_log_replica(env, chosen_index, wps, size_in_kb)

def best_choice(env):
    """Best choice algorithm for server selection."""
    for _ in range(TOTAL_LOG_REPLICAS):
        yield env.timeout(1)  # Simulate time between log replica additions
        wps = get_wps_for_time(env.now)   # Determine WPS for the new log replica
        size_in_kb = random.choice([1024, 1])
        chosen_index = min(range(N), key=combined_metric_cached)
        add_log_replica(env, chosen_index, wps, size_in_kb)

def get_wps_for_time(env_time):
    """
    Determine the Writes Per Second (WPS) based on the current simulation time.

    This function dynamically adjusts the WPS value based on predefined time intervals
    within the simulation environment. It is designed to simulate varying load conditions
    that a server might experience throughout a day, including peak hours, off-peak hours,
    and sudden spikes in traffic.

    Parameters:
    - env_time: The current time within the simulation environment.

    Returns:
    - An integer representing the WPS value for the given time.

    The function defines three main periods:
    - Peak hours (1000-2000 simulation time units) with higher WPS values to simulate increased load.
    - Off-peak hours (0-1000 and 2000-10000 simulation time units) with lower WPS values to simulate decreased load.
    - Specific moments (e.g., 1500, 5500, 7500 simulation time units) to simulate sudden spikes in traffic.

    For times not covered by the above conditions, a default WPS range is provided.
    """
    # Define peak hours (e.g., 1000-2000 simulation time units)
    if 1000 <= env_time <= 2000:
        return random.randint(800, 1000)  # Higher WPS for peak hours
    # Define off-peak hours (e.g., 0-1000 and 2000-10000 simulation time units)
    elif 0 <= env_time < 1000 or 2000 < env_time <= 10000:
        return random.randint(100, 300)  # Lower WPS for off-peak hours
    # Optionally, handle sudden spikes
    elif env_time in [1500, 5500, 7500]:  # Example spike moments
        return random.randint(1000, 1200)  # Sudden spike WPS
    else:
        return random.randint(300, 500)  # Default WPS if not in any specific interval


def simulate_dynamic_load(env):
    """
    Simulate dynamic changes in server load and replica count.

    This function periodically adjusts the load and replica count for each server
    to simulate real-world fluctuations in server usage. It's designed to run
    continuously within a simulation environment, making adjustments at regular intervals.

    Parameters:
    - env: The simulation environment from simpy. This allows the function to be
           integrated into the simpy event loop and execute over time.

    The function loops indefinitely, adjusting server loads and replica counts at
    predefined intervals. Adjustments are made by applying a random change to the
    load and a random increment or decrement to the replica count, within specified
    limits to ensure values remain realistic.
    """
    while True:
        for i in range(N): #Iterate over all servers
            # Randomly adjust load by a percentage of the current load
            load_change = random.uniform(-0.05, 0.05) * servers[i]
            # Randomly adjust replica count by -1, 0, or 1
            replica_change = random.randint(-1, 1)
            # Ensure load and replica count are non-negative and within realistic limits
            servers[i] = max(0, servers[i] + load_change)
            # Apply replica change ensuring it stays within the [0,200] limit
            log_replicas_per_server[i] = max(0, min(200, log_replicas_per_server[i] + replica_change))
        # Wait for 10 simulation units before making the next adjustment
        yield env.timeout(10)  # Adjust every 10 simulation units

def update_cache(env):
    """
    Update the cache with a delay to simulate real-world update delays.

    This function periodically updates the server state cache to reflect the current
    state of each server in terms of load and replica count. It's designed to mimic
    the delay that might occur in a real system where state information is not updated
    instantaneously. In real system the metadata is stored on a distribute key value
    store and the cache is maintained either locally on each server or a L2 cache.
    The cache is updated at regular intervals to ensure that the server selection
    algorithms have access to the most recent data, albeit with a realistic delay.

    Parameters:
    - env: The simulation environment from simpy, which allows this function to be
           scheduled and executed at regular intervals within the simulation.

    The function loops indefinitely, ensuring that the cache is updated at regular
    intervals throughout the simulation. This helps in keeping the server selection
    algorithms informed with the most recent data, albeit with a realistic delay.

    The update interval is set to 50 simulation units, representing the frequency
    at which the cache is refreshed. This value can be adjusted to simulate different
    cache update frequencies.
    """
    while True:
        for i in range(N): # Iterate over all servers to update their current state in the cache
            server_state_cache[i]['load'] = servers[i] # Update the load for server i in the cache
            server_state_cache[i]['replicas'] = log_replicas_per_server[i] # Update the replica count in cache
        yield env.timeout(50)  # Wait for 50 simulation units before updating the cache again


def combined_metric_cached(server_index, load_weight=0.5, replica_weight=0.5):
    """
    Calculate a combined metric for server selection using cached server states.

    This function computes a weighted sum of the normalized load and replica count for a given server,
    using cached data. The purpose is to evaluate servers based on both their current load and the number
    of replicas they hold, allowing for a balanced decision in server selection processes.

    Parameters:
    - server_index: The index of the server in the server_state_cache list.
    - load_weight: The weight assigned to the server's load in the combined metric calculation.
                   Defaults to 0.5, indicating an equal balance with replica count by default.
    - replica_weight: The weight assigned to the server's replica count in the combined metric calculation.
                      Defaults to 0.5, indicating an equal balance with load by default.

    Returns:
    - combined: The combined metric for the server, calculated as a weighted sum of the normalized load
                and replica count. This metric is used to compare servers for selection.

    Note:
    The function ensures that division by zero is avoided by setting a minimum value of 1 for both
    max_load and max_replicas, in case all servers have zero load or replicas. This is crucial for
    maintaining stability in the server selection algorithm.
    """
    # Ensure max_load and max_replicas are at least 1 to avoid division by zero
    max_load = max([s['load'] for s in server_state_cache]) if max([s['load'] for s in server_state_cache]) > 0 else 1
    max_replicas = max([s['replicas'] for s in server_state_cache]) if max(
        [s['replicas'] for s in server_state_cache]) > 0 else 1

    # Normalize load and replica count for the server
    normalized_load = server_state_cache[server_index]['load'] / max_load
    normalized_replicas = server_state_cache[server_index]['replicas'] / max_replicas

    # Calculate combined metric using weighted sum of normalized load and replica count
    combined = (normalized_load * load_weight) + (normalized_replicas * replica_weight)
    return combined


def plot_grouped_comparison(metric_dict, title, ylabel):
    """
    Plot a grouped bar chart to compare metrics across different algorithms.

    This function takes a dictionary of metrics, where each key is an algorithm name and its value is a list
    of metrics corresponding to each server. It plots a grouped bar chart to visually compare these metrics
    across servers for different algorithms.

    Parameters:
    - metric_dict: A dictionary where keys are algorithm names and values are lists of metrics (e.g., load or replica count).
    - title: The title of the plot, indicating what is being compared.
    - ylabel: The label for the y-axis, indicating the metric being compared.

    The function creates a grouped bar chart with one group per server. Each group contains bars for each algorithm,
    allowing for easy comparison of the specified metric across algorithms for each server. The x-axis represents
    server indices, and the y-axis represents the metric value. Each algorithm is assigned a distinct color for clarity.

    Note:
    - The function assumes that all lists in metric_dict are of the same length, corresponding to the number of servers.
    - The width of the bars and the offsets for each group are calculated to neatly align the bars within each group.
    - Optional text annotations can be added above bars to display the metric value, enhancing readability.
    """

    labels = range(N)  # Server indices
    width = 0.2  # Narrower width for grouped bars
    algorithms = list(metric_dict.keys())
    n_algorithms = len(algorithms)
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange']  # Distinct colors for each algorithm

    fig, ax = plt.subplots(figsize=(15, 8))
    for i, algorithm in enumerate(algorithms):
        values = metric_dict[algorithm]
        offsets = [x + i * width for x in labels]  # Calculate offset for each group
        ax.bar(offsets, values, width, label=algorithm, color=colors[i])

        # Optional: Add text annotations above bars
        for x, y in zip(offsets, values):
            ax.text(x, y, f'{y:.2f}', ha='center', va='bottom')

    ax.set_xlabel('Server Index')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([x + width * (n_algorithms / 2 - 0.5) for x in labels])  # Center x-tick labels
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_stats_comparison(stats_dict, title, ylabel):
    """
    Plot a bar chart to compare statistical metrics across different algorithms.

    This function visualizes the comparison of statistical metrics (e.g., standard deviation of load or log replicas)
    for different server selection algorithms. It uses a bar chart format where each bar represents a metric value
    for a specific algorithm, facilitating easy comparison across algorithms.

    Parameters:
    - stats_dict: A dictionary where keys are algorithm names and values are the statistical metrics to be compared.
                  The metrics could represent any numerical value associated with the performance or outcome of
                  the algorithms, such as standard deviation.
    - title: The title of the plot, providing context about what statistical metric is being compared.
    - ylabel: The label for the y-axis, indicating the type of metric being compared (e.g., "Standard Deviation").

    The function assigns distinct colors to each algorithm for clarity and adds a legend that combines the algorithm
    name with its corresponding metric value for enhanced readability. This visualization aids in identifying which
    algorithms perform better or worse according to the chosen metric.

    Note:
    - The function assumes that `matplotlib.pyplot` has been imported as `plt`.
    - The colors array is predefined and will cycle through 'skyblue', 'lightgreen', 'lightcoral', 'orange' for the bars.
      If there are more algorithms than colors, colors will repeat.
    - Custom labels in the legend show both the algorithm name and its metric value, providing a quick summary without
      needing to read the bar values directly.
    """
    labels = list(stats_dict.keys()) # Extract algorithm names
    values = list(stats_dict.values())
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange']  # Distinct colors for each algorithm

    plt.figure(figsize=(10, 6)) # Set the figure size for the plot
    bars = plt.bar(labels, values, color=colors)
    plt.xlabel('Algorithm')
    plt.ylabel(ylabel)
    plt.title(title)

    # Generate custom labels for the legend, combining algorithm names with their metric values
    custom_labels = [f"{label} - {value:.2f}" for label, value in zip(labels, values)]
    plt.legend(bars, custom_labels)

    plt.tight_layout()  # Adjust the layout to make sure everything fits without overlapping
    plt.show()


def run_simulation(algorithm):
    """
    Run the simulation for a specified server selection algorithm and store the results.

    This function initializes the simulation environment, resets server loads and log replica counts,
    and runs the simulation using the provided server selection algorithm. It stores the final state
    of each server's load and log replica count for comparison. Additionally, it calculates and prints
    the standard deviation of the final loads and log replica counts to evaluate the distribution of
    load and replicas across servers.

    Parameters:
    - algorithm: A function representing the server selection algorithm to be simulated. This function
                 should accept a simpy.Environment instance as its only argument.

    The function performs the following steps:
    1. Resets the global servers and log_replicas_per_server lists to their initial state.
    2. Creates a new simpy.Environment instance for the simulation.
    3. Schedules the simulate_dynamic_load, update_cache, and the provided algorithm functions within
       the simulation environment.
    4. Runs the simulation for a predefined duration (SIM_TIME).
    5. Stores the final load and log replica count of each server for comparison.
    6. Calculates and prints the standard deviation of the final loads and log replica counts for each
       algorithm, providing insights into the load and replica distribution.
    """

    global servers, log_replicas_per_server
    servers = [0] * N  # Reset servers for each simulation
    log_replicas_per_server = [0] * N  # Reset log replica counts
    env = simpy.Environment()

    env.process(simulate_dynamic_load(env))
    env.process(update_cache(env))

    env.process(algorithm(env))
    env.run(until=SIM_TIME)
    # Store the final state for comparison
    final_loads[algorithm.__name__] = servers[:]
    final_replicas[algorithm.__name__] = log_replicas_per_server[:]

    # Calculate mean and standard deviation for final loads

    std_dev_loads[algorithm.__name__] = np.std(servers)

    std_dev_replicas[algorithm.__name__] = np.std(log_replicas_per_server)

    print(f"{algorithm.__name__} -  Standard Deviation Load: {std_dev_loads[algorithm.__name__]:.2f}")
    print(f"{algorithm.__name__} -  Standard Deviation Log Replicas: {std_dev_replicas[algorithm.__name__]:.2f}")



# Run simulations
for algorithm in [best_of_two_choice, best_choice, random_choice, best_of_three_choice]:
    run_simulation(algorithm)

# Plotting
plot_grouped_comparison(final_loads, 'Final Load Comparison', 'Total Traffic (WPS)')
plot_grouped_comparison(final_replicas, 'Final Log Replicas Comparison', 'Number of Log Replicas')
plot_stats_comparison(std_dev_loads, 'Load Standard Deviation Comparison', 'Standard Deviation (WPS)')
plot_stats_comparison(std_dev_replicas, 'Log Replicas Standard Deviation Comparison', 'Standard Deviation of Log Replicas')