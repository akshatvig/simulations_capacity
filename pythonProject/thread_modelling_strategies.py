import simpy
import matplotlib.pyplot as plt
import numpy as np
import random

# Constants
SIMULATION_DURATION = 100  # Adjust as needed
CPU_RANGE = (10, 70)
IOPS_RANGE = (0.1, 0.9)
REQUESTS_RANGE = (1, 100000)
CONNECTIONS_RANGE = (1, 10000)
CONNECTION_LIFETIME_RANGE = (10, 3600)

# Data collection
throughput_data = {'RPT': [], 'CPT': []}
response_time_data = {'RPT': [], 'CPT': []}

def simulate_request(env, cpu_load, io_intensity):
    start = env.now
    yield env.timeout(random.uniform(0, cpu_load / 1000.0))  # Simulate CPU load
    yield env.timeout(random.uniform(0, io_intensity / 1000.0))  # Simulate I/O load
    return env.now - start  # Response time

def rpt_process(env, cpu_load, io_intensity, concurrent_requests):
    requests = 0
    total_response_time = 0
    while True:
        if env.now <= SIMULATION_DURATION:
            response_time = yield env.process(simulate_request(env, cpu_load, io_intensity))
            total_response_time += response_time
            requests += 1
        else:
            break
    throughput_data['RPT'].append(requests / SIMULATION_DURATION)
    response_time_data['RPT'].append(total_response_time / requests)

def cpt_process(env, cpu_load, io_intensity, connections, connection_lifetime):
    requests = 0
    total_response_time = 0
    for _ in range(connections):
        start_time = env.now
        while env.now - start_time < connection_lifetime:
            response_time = yield env.process(simulate_request(env, cpu_load, io_intensity))
            total_response_time += response_time
            requests += 1
    throughput_data['CPT'].append(requests / SIMULATION_DURATION)
    response_time_data['CPT'].append(total_response_time / requests)

# Run simulation
env = simpy.Environment()
cpu_load = random.randint(*CPU_RANGE)
io_intensity = random.uniform(*IOPS_RANGE)
concurrent_requests = random.randint(*REQUESTS_RANGE)
connections = random.randint(*CONNECTIONS_RANGE)
connection_lifetime = random.randint(*CONNECTION_LIFETIME_RANGE)

env.process(rpt_process(env, cpu_load, io_intensity, concurrent_requests))
env.process(cpt_process(env, cpu_load, io_intensity, connections, connection_lifetime))
env.run(until=SIMULATION_DURATION + 1)

# Generate graphs
plt.figure(figsize=(12, 6))

# Throughput graph
plt.subplot(1, 2, 1)
plt.bar(['RPT', 'CPT'], [np.mean(throughput_data['RPT']), np.mean(throughput_data['CPT'])])
plt.title('Throughput Comparison')
plt.ylabel('Requests per Second')

# Response time graph
plt.subplot(1, 2, 2)
plt.bar(['RPT', 'CPT'], [np.mean(response_time_data['RPT']), np.mean(response_time_data['CPT'])])
plt.title('Average Response Time Comparison')
plt.ylabel('Seconds')

plt.tight_layout()
plt.show()