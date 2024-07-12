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
servers = [0] * N
log_replicas_per_server = [0] * N
load_history = [[] for _ in range(N)]
log_replicas_history = [[] for _ in range(N)]

# Global dictionaries to store final results for comparison
final_loads = {}
final_replicas = {}
std_dev_loads = {}
std_dev_replicas = {}

def add_log_replica(env, server_index, wps, size_in_kb):
    """Add a log replica with variable traffic to a server and update history."""

    load_adjustment = size_in_kb / 1024  # Assuming 1KB = 1 load unit, 1MB = 1024 load units
    new_load = servers[server_index] + (wps * load_adjustment)

    # Check if the server can accept more replicas and if new load is within the limit
    if log_replicas_per_server[server_index] < 200 and new_load <= 200000:
        servers[server_index] = new_load  # Update server load
        log_replicas_per_server[server_index] += 1
        # Update histories
        load_history[server_index].append(servers[server_index])
        log_replicas_history[server_index].append(log_replicas_per_server[server_index])
    else:
        # Optionally handle the case where a server cannot accept more replicas or would exceed load limit
        pass


def random_choice(env):
    """Random choice algorithm for server selection."""
    for _ in range(TOTAL_LOG_REPLICAS):
        yield env.timeout(1)  # Simulate time between log replica additions
        wps = get_wps_for_time(env.now)  # Determine WPS for the new log replica
        size_in_kb = 1 #random.choice([1024, 1])
        chosen_index = random.randint(0, N-1)  # Select a random server
        add_log_replica(env, chosen_index, wps, size_in_kb)

def best_of_two_choice(env):
    """Best of two choice algorithm for server selection."""
    for _ in range(TOTAL_LOG_REPLICAS):
        yield env.timeout(1)  # Simulate time between log replica additions
        wps = get_wps_for_time(env.now)  # Determine WPS for the new log replica
        size_in_kb = 1 #random.choice([1024, 1])
        a, b = random.sample(range(N), 2)
        chosen_index = a if combined_metric_cached(a) < combined_metric_cached(b) else b
        add_log_replica(env, chosen_index, wps, size_in_kb)

def best_of_three_choice(env):
    """Best of three choice algorithm for server selection."""
    for _ in range(TOTAL_LOG_REPLICAS):
        yield env.timeout(1)  # Simulate time between log replica additions
        wps = get_wps_for_time(env.now)  # Determine WPS for the new log replica
        size_in_kb = 1  # Assuming a fixed size for simplicity, can be randomized or adjusted as needed
        a, b, c = random.sample(range(N), 3)  # Select three random servers
        # Choose the server with the best (lowest) combined metric
        chosen_index = min([a, b, c], key=combined_metric_cached)
        add_log_replica(env, chosen_index, wps, size_in_kb)

def best_choice(env):
    """Best choice algorithm for server selection."""
    for _ in range(TOTAL_LOG_REPLICAS):
        yield env.timeout(1)  # Simulate time between log replica additions
        wps = get_wps_for_time(env.now)   # Determine WPS for the new log replica
        size_in_kb = 1 #random.choice([1024, 1])
        chosen_index = min(range(N), key=combined_metric_cached)
        add_log_replica(env, chosen_index, wps, size_in_kb)

# Cache for server states
server_state_cache = [{'load': 0, 'replicas': 0} for _ in range(N)]

def simulate_dynamic_load(env):
    """Simulate dynamic changes in server load and replica count."""
    while True:
        for i in range(N):
            # Randomly adjust load and replica count
            load_change = random.uniform(-0.05, 0.05) * servers[i]
            replica_change = random.randint(-1, 1)
            servers[i] = max(0, servers[i] + load_change)
            log_replicas_per_server[i] = max(0, min(200, log_replicas_per_server[i] + replica_change))
        yield env.timeout(10)  # Adjust every 10 simulation units

def update_cache(env):
    """Update the cache with a delay to simulate real-world update delays."""
    while True:
        for i in range(N):
            server_state_cache[i]['load'] = servers[i]
            server_state_cache[i]['replicas'] = log_replicas_per_server[i]
        yield env.timeout(50)  # Cache updates every 5 simulation units


def combined_metric_cached(server_index, load_weight=0.5, replica_weight=0.5):
    """
    Calculate a combined metric using cached server states, ensuring no division by zero.
    """
    # Ensure max_load and max_replicas are at least 1 to avoid division by zero
    max_load = max([s['load'] for s in server_state_cache]) if max([s['load'] for s in server_state_cache]) > 0 else 1
    max_replicas = max([s['replicas'] for s in server_state_cache]) if max(
        [s['replicas'] for s in server_state_cache]) > 0 else 1

    normalized_load = server_state_cache[server_index]['load'] / max_load
    normalized_replicas = server_state_cache[server_index]['replicas'] / max_replicas

    combined = (normalized_load * load_weight) + (normalized_replicas * replica_weight)
    return combined


def plot_grouped_comparison(metric_dict, title, ylabel):
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

def get_wps_for_time(env_time):
    """Determine WPS based on the current simulation time."""
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


def plot_stats_comparison(stats_dict, title, ylabel):
    labels = list(stats_dict.keys())
    values = list(stats_dict.values())
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange']  # Distinct colors for each algorithm

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.xlabel('Algorithm')
    plt.ylabel(ylabel)
    plt.title(title)
    custom_labels = [f"{label} - {value:.2f}" for label, value in zip(labels, values)]
    plt.legend(bars, custom_labels)

    plt.tight_layout()
    plt.show()


def run_simulation(algorithm):
    """Run the simulation for a given server selection algorithm and store results."""
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