import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Define parameters
np.random.seed(42)
num_queries = 1000  # Total number of queries
lambda_rate = 5     # Poisson arrival rate

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

def simulate_sjf(query_data):
    query_data = query_data.sort_values(["ArrivalTime", "Cost"]).reset_index(drop=True)
    completion_times = []
    current_time = 0
    for _, query in query_data.iterrows():
        current_time = max(current_time, query["ArrivalTime"]) + query["Cost"]
        completion_times.append(current_time)
    query_data["CompletionTime"] = completion_times
    return query_data

def simulate_edf(query_data):
    # Assign deadlines inversely proportional to query cost for simulation
    query_data["Deadline"] = query_data["ArrivalTime"] + (100 / query_data["Cost"])
    query_data = query_data.sort_values(["ArrivalTime", "Deadline"]).reset_index(drop=True)
    completion_times = []
    current_time = 0
    for _, query in query_data.iterrows():
        current_time = max(current_time, query["ArrivalTime"]) + query["Cost"]
        completion_times.append(current_time)
    query_data["CompletionTime"] = completion_times
    return query_data

# Simulate the algorithms
fcfs_result = simulate_fcfs(query_data.copy())
weighted_result = simulate_weighted_fair_scheduling(query_data.copy())
sjf_result = simulate_sjf(query_data.copy())
edf_result = simulate_edf(query_data.copy())

# Combine results
all_results = pd.concat([
    fcfs_result[["QueryID", "CompletionTime"]].assign(Algorithm="FCFS"),
    weighted_result[["QueryID", "CompletionTime"]].assign(Algorithm="Weighted Fair Scheduling"),
    sjf_result[["QueryID", "CompletionTime"]].assign(Algorithm="Shortest Job First"),
    edf_result[["QueryID", "CompletionTime"]].assign(Algorithm="Earliest Deadline First")
])

print(all_results.groupby("Algorithm").mean())



styles = {
    "FCFS": "dashed",
    "Weighted Fair Scheduling": "solid",
    "Shortest Job First": "dotted",
    "Earliest Deadline First": "dashdot"
}

plt.figure(figsize=(12, 6))
for algorithm, data in all_results.groupby("Algorithm"):
    data = data.sort_values("QueryID")
    plt.plot(data["QueryID"], data["CompletionTime"], label=algorithm,
             linestyle=styles[algorithm], linewidth=2)

plt.title("Comparison of Scheduling Algorithms")
plt.xlabel("Query ID")
plt.ylabel("Completion Time")
plt.legend(title="Algorithm")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
