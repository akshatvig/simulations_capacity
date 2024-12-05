import numpy as np

# Example constants
num_simulations = 10000  # Number of Monte Carlo trials
time_period_days = 365  # Simulating for one year
storage_cost_per_gb = 0.023  # S3 storage cost per GB
compaction_threshold_gb = 100  # When to trigger compaction
compaction_cost_mean = 100  # Mean compaction cost
compaction_cost_stddev = 10  # Standard deviation of compaction cost
write_frequency_lambda = 50  # Poisson rate (average writes per day)

# Initialize arrays to store the results from each simulation
total_storage_costs = np.zeros(num_simulations)
total_compaction_costs = np.zeros(num_simulations)



# Run Monte Carlo Simulation for multiple trials
for sim in range(num_simulations):
    total_dead_data = 0
    storage_cost = 0
    compaction_cost = 0

    # Simulate over the time period (e.g., 365 days)
    for day in range(time_period_days):
        # Generate random number of writes for today using Poisson distribution
        writes_today = np.random.poisson(write_frequency_lambda)
        # Assume some percentage of writes generate dead data (for example, 10%)
        dead_data_today = writes_today * 0.1
        total_dead_data += dead_data_today

        # If we hit the compaction threshold, compact the data
        if total_dead_data >= compaction_threshold_gb:
            # Generate a random compaction cost using normal distribution
            compaction_cost += np.random.normal(compaction_cost_mean, compaction_cost_stddev)
            total_dead_data = 0  # Reset dead data after compaction

        # Calculate the storage cost for today's dead data
        storage_cost += total_dead_data * storage_cost_per_gb

    # Store results for this simulation
    total_storage_costs[sim] = storage_cost
    total_compaction_costs[sim] = compaction_cost

# Analyze the results
average_storage_cost = np.mean(total_storage_costs)
average_compaction_cost = np.mean(total_compaction_costs)
total_average_cost = average_storage_cost + average_compaction_cost

print(f"Average Storage Cost: ${average_storage_cost:.2f}")
print(f"Average Compaction Cost: ${average_compaction_cost:.2f}")
print(f"Average Total Cost: ${total_average_cost:.2f}")
