import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Define parameters
np.random.seed(42)
num_queries = 1000  # Total number of queries
lambda_rate = 5     # Poisson arrival rate
resource_limit = 100  # Resource limit (CPU or memory)

# Query types and cost mappings
query_types = {
    "Point Lookup": {"Cost": 2, "CPU": 5, "Memory": 2},
    "Aggregation": {"Cost": 15, "CPU": 30, "Memory": 20},
    "Full Scan": {"Cost": 100, "CPU": 70, "Memory": 50}
}

# Generate Poisson arrival times
arrival_times = np.cumsum(np.random.exponential(1 / lambda_rate, num_queries))

# Assign query types randomly
query_data = pd.DataFrame({
    "QueryID": range(1, num_queries + 1),
    "ArrivalTime": arrival_times,
    "Type": np.random.choice(list(query_types.keys()), num_queries, p=[0.5, 0.3, 0.2])
})

# Map query types to their respective cost and resource usage
query_data["Cost"] = query_data["Type"].apply(lambda x: query_types[x]["Cost"])
query_data["CPU"] = query_data["Type"].apply(lambda x: query_types[x]["CPU"])
query_data["Memory"] = query_data["Type"].apply(lambda x: query_types[x]["Memory"])

# Simulation functions for scheduling algorithms
def simulate_fcfs(query_data):
    query_data = query_data.sort_values("ArrivalTime").reset_index(drop=True)
    completion_times = []
    current_time = 0
    for _, query in query_data.iterrows():
        current_time = max(current_time, query["ArrivalTime"]) + query["Cost"]
        completion_times.append(current_time)
    query_data["CompletionTime"] = completion_times
    return query_data

def simulate_weighted_fair_scheduling(query_data):
    weights = {"Point Lookup": 10, "Aggregation": 5, "Full Scan": 1}
    query_data["Weight"] = query_data["Type"].map(weights)
    query_data["Priority"] = query_data["Cost"] / query_data["Weight"]
    query_data = query_data.sort_values(["ArrivalTime", "Priority"]).reset_index(drop=True)
    completion_times = []
    current_time = 0
    for _, query in query_data.iterrows():
        current_time = max(current_time, query["ArrivalTime"]) + query["Cost"] / query["Weight"]
        completion_times.append(current_time)
    query_data["CompletionTime"] = completion_times
    return query_data

def simulate_round_robin(query_data, time_slice=10):
    query_data = query_data.sort_values("ArrivalTime").reset_index(drop=True)
    completion_times = np.zeros(len(query_data))
    remaining_cost = query_data["Cost"].values.copy()
    current_time = 0
    while remaining_cost.sum() > 0:
        for i in range(len(query_data)):
            if remaining_cost[i] > 0:
                execution_time = min(time_slice, remaining_cost[i])
                current_time = max(current_time, query_data.iloc[i]["ArrivalTime"]) + execution_time
                remaining_cost[i] -= execution_time
                if remaining_cost[i] <= 0:
                    completion_times[i] = current_time
    query_data["CompletionTime"] = completion_times
    return query_data

# Simulate the algorithms
fcfs_result = simulate_fcfs(query_data.copy())
weighted_result = simulate_weighted_fair_scheduling(query_data.copy())
rr_result = simulate_round_robin(query_data.copy())

# Combine results
all_results = pd.concat([
    fcfs_result[["QueryID", "CompletionTime"]].assign(Algorithm="FCFS"),
    weighted_result[["QueryID", "CompletionTime"]].assign(Algorithm="Weighted Fair Scheduling"),
    rr_result[["QueryID", "CompletionTime"]].assign(Algorithm="Round Robin")
])

# Plot the results
plt.figure(figsize=(12, 6))
for algorithm, data in all_results.groupby("Algorithm"):
    data = data.sort_values("QueryID")
    plt.plot(data["QueryID"], data["CompletionTime"], label=algorithm)

plt.title("Comparison of Scheduling Algorithms")
plt.xlabel("Query ID")
plt.ylabel("Completion Time")
plt.legend(title="Algorithm")
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()
