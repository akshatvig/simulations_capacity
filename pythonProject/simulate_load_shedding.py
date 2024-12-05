import random
from collections import namedtuple, deque
import numpy as np
import simpy

LatencyDatum = namedtuple('LatencyDatum', ('t_queued', 't_processing', 't_total', 'dropped'))

class RequestSimulator:
    def __init__(self, admission_cap, execution_cap, latency_fn, num_requests, request_per_s, latency_threshold, strategy):
        self.admission_cap = admission_cap
        self.execution_cap = execution_cap
        self.latency_fn = latency_fn
        self.num_requests = int(num_requests)
        self.request_interval_ms = 1. / (request_per_s / 1000.)
        self.data = []
        self.dropped_requests = 0
        self.processed_requests = 0
        self.latency_history = deque(maxlen=100)
        self.strategy = strategy
        self.latency_threshold = latency_threshold

    def simulate(self):
        random.seed(1)
        np.random.seed(1)

        self.env = simpy.Environment()
        self.admission_queue = simpy.Store(self.env, capacity=self.admission_cap)
        self.execution_queue = simpy.Resource(self.env, capacity=self.execution_cap)
        self.env.process(self.generate_requests())
        self.env.process(self.process_requests())
        self.env.run()

    def get_average_latency(self):
        if len(self.latency_history) == 0:
            return 0
        return np.mean(self.latency_history)

    def generate_requests(self):
        for i in range(self.num_requests):
            t_arrive = self.env.now

            if self.strategy == "latency-based":
                # Calculate the current average latency
                avg_latency = self.get_average_latency()

                # Apply latency-based load shedding
                if avg_latency > self.latency_threshold:
                    print(f"Shedding load due to high latency: {avg_latency:.2f} ms")
                    self.dropped_requests += 1
                    self.data.append(LatencyDatum(0, 0, 0, True))
                    continue

            if len(self.admission_queue.items) >= self.admission_cap:
                # Admission queue is full, drop the request (load shedding)
                print(f"Shedding load: {len(self.admission_queue.items):.2f}")
                self.dropped_requests += 1
                self.data.append(LatencyDatum(0, 0, 0, True))
            else:
                # Add request to admission queue
                request = (i, t_arrive)
                yield self.admission_queue.put(request)
                print(f"Accepted load: {len(self.admission_queue.items):.2f}")

            # Exponential inter-arrival times (Poisson process)
            arrival_interval = random.expovariate(1.0 / self.request_interval_ms)
            yield self.env.timeout(arrival_interval)

    def process_requests(self):
        while self.processed_requests + self.dropped_requests < self.num_requests:
            print(f"Processed Request Count {self.processed_requests:.2f}")
            if len(self.admission_queue.items) > 0:
                request_id, t_arrive = yield self.admission_queue.get()

                with self.execution_queue.request() as req:
                    yield req
                    t_start = self.env.now
                    t_queued = t_start - t_arrive

                    # Determine processing time using latency function
                    processing_time = self.latency_fn(request_id)
                    yield self.env.timeout(processing_time)

                    t_done = self.env.now
                    t_processing = t_done - t_start
                    t_total_response = t_done - t_arrive

                    # Store latency data
                    datum = LatencyDatum(t_queued, t_processing, t_total_response, False)
                    self.data.append(datum)
                    self.latency_history.append(t_total_response)
                    self.processed_requests += 1

            else:
                # No requests to process, yield for a small interval

                yield self.env.timeout(1)

def run_simulation(admission_cap, execution_cap, num_requests, request_per_s, latency_fn, latency_threshold, strategy):
    simulator = RequestSimulator(admission_cap, execution_cap, latency_fn, num_requests, request_per_s, latency_threshold, strategy)
    simulator.simulate()
    return simulator.data, simulator.dropped_requests, num_requests

# Example Latency Function
def example_latency_fn(request_id):
    return random.uniform(5, 20)  # Random latency between 5ms and 20ms

# Run the simulation with an admission queue of 50, execution capacity of 10
data_queue_based, dropped_queue_based, num_requests_queue_based = run_simulation(
    admission_cap=50,
    execution_cap=10,
    num_requests=1000,
    request_per_s=100,
    latency_fn=example_latency_fn,
    latency_threshold=50,
    strategy="queue-based"
)

data_latency_based, dropped_latency_based, num_requests_latency_based = run_simulation(
    admission_cap=50,
    execution_cap=10,
    num_requests=1000,
    request_per_s=100,
    latency_fn=example_latency_fn,
    latency_threshold=50,
    strategy="latency-based"
)

# Analyze Results for Queue-Based Load Shedding
avg_latency_queue_based = np.mean([d.t_total for d in data_queue_based if not d.dropped])
drop_rate_queue_based = (dropped_queue_based / num_requests_queue_based) * 100

# Analyze Results for Latency-Based Load Shedding
avg_latency_latency_based = np.mean([d.t_total for d in data_latency_based if not d.dropped])
drop_rate_latency_based = (dropped_latency_based / num_requests_latency_based) * 100

# Print Comparison Results
print("\n=== Queue-Based Load Shedding ===")
print(f"Average Latency: {avg_latency_queue_based:.2f} ms")
print(f"Dropped Requests: {dropped_queue_based}")
print(f"Drop Rate: {drop_rate_queue_based:.2f}%")

print("\n=== Latency-Based Load Shedding ===")
print(f"Average Latency: {avg_latency_latency_based:.2f} ms")
print(f"Dropped Requests: {dropped_latency_based}")
print(f"Drop Rate: {drop_rate_latency_based:.2f}%")