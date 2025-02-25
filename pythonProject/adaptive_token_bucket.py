import csv
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import numpy as np

# -----------------------------
# Simulation Parameters
# -----------------------------
SIM_TIME = 3000_000  # total simulation time in milliseconds
TIME_STEP = 10  # each iteration is 10ms
NUM_CLIENTS = 100
BASE_ARRIVAL_RATE = 100.0  # average requests per 10ms step across all clients
# (equivalent to 10,000 req/sec if we scale properly)

# Lets assume we have 80 / 20 ratio of fast and slow requests
FAST_RATIO = 0.8  # 80% of requests are fast
SLOW_RATIO = 0.2  # 20% of requests are slow

# Service times (in ms) - simplified constant or random distribution
# The assumption here is that fast requests are 10X faster than slow requests
# Another assumption here is that once a fast request is admitted and actively processed
# it takes 5 ms to finish processing the request. A slow request on the other hand
# takes 50 ms to finish processing.
FAST_SERVICE_TIME = 5
SLOW_SERVICE_TIME = 50

# Queue capacities - The implementation here assumes a bounded queue model
# upto 100 fast requests can be enqueued. If a new fast request arrives and the queue
# is at 100, request is rejected. Similarly, upto 100 slow requests can be enqueued.
FAST_QUEUE_CAPACITY = 100
SLOW_QUEUE_CAPACITY = 100

# The idea here is that each request must acquire a token from the global token bucket
# before getting accepted. This lets us put an overall cap on the rate at which the node
# can accept new requests and also ensuring overload prevention.

# 1) GLOBAL_TOKEN_CAPACITY:
#    - This is the maximum number of tokens the bucket can hold at one time.
#    - By setting it very large (e.g., 100,000), we ensure that the refill rate
#      (see below) is the main factor limiting throughput, rather than the bucket
#      overflowing.
#    - If this capacity were small, then even a short burst of requests could overflow
#      the bucket (wasting refill tokens that can’t be stored), reducing throughput.
#
# 2) INITIAL_REFILL_RATE:
#    - This is the *starting* rate at which tokens are added into the bucket each second.
#    - In the simulation, we  adjust (increase or decrease) this rate in response to
#      runtime signals (e.g., queue dwell time, CPU usage, concurrency limits).
#    - For instance, if the system appears underutilized, we might raise the refill rate
#      to allow more requests; if overloaded, we lower it to throttle incoming requests.
#    - We start at a moderately conservative 300 tokens/second and adapt from there.
#
# 3) MAX_REFILL_RATE:
#    - This is the upper limit on how quickly we can add tokens (i.e., how many requests
#      per second we can admit) if the system seems to be running smoothly.
#    - If we find that requests are being handled quickly and we’re not hitting any
#      overload conditions, we might push the refill rate up to this maximum (e.g., 1000
#      tokens/sec) to increase throughput.
#
# 4) MIN_REFILL_RATE:
#    - This is the lower bound on the token add rate. If the system is under heavy load
#      or dwell times are high, the adaptive logic may reduce the rate—but never below
#      this number (50 tokens/sec).
#    - This ensures we always allow at least some incoming traffic, unless we’re rejecting
#      because of other constraints (like a full queue). It prevents the system from
#      dropping to zero admissions and causing full starvation.

GLOBAL_TOKEN_CAPACITY = 100_000  # large capacity so we only focus on refill rate
INITIAL_REFILL_RATE = 300.0  # tokens per second
MAX_REFILL_RATE = 1000.0  # upper bound if system is underutilized
MIN_REFILL_RATE = 50.0  # lower bound if system is overloaded

# Dwell time thresholds (in milliseconds):
# These define how long a request can wait in its queue before we consider it "too long,"
# triggering a potential throttle if we consistently exceed these times.
#
#  - FAST_DWELL_THRESHOLD: The dwell threshold for the "fast" queue (lightweight queries).
#    If requests in the fast queue often wait more than FAST_DWELL_THRESHOLD before starting service,
#    we consider the system overloaded for fast operations.
#  - SLOW_DWELL_THRESHOLD: Similarly, the threshold for the "slow" queue (heavy queries)
#    is SLOW_DWELL_THRESHOLD. If slow requests sit in the queue longer than this, it's a sign that
#    the system is overloaded for slow operations.
FAST_DWELL_THRESHOLD = 100
SLOW_DWELL_THRESHOLD = 500

# We'll re-check dwell time every 1000ms for adaptive logic:
#  - ADAPT_INTERVAL = 1000 means every 1 second of simulation time, we measure system
#    indicators (like average dwell time or CPU usage) and decide whether to adjust our
#    token-bucket refill rate up or down.
#  - This prevents thrashing by not recalculating too frequently (e.g., every 10ms).
#  - In a real system, we might adjust this interval based on the expected rate of change and
#    the cost of recalculating.
ADAPT_INTERVAL = 1000

# Per-client buckets (optional advanced approach):
#  - In addition to the global bucket, each client can have its own token bucket.
#    This ensures no single client can consume all tokens in bursts.
#  - PER_CLIENT_CAPACITY: maximum number of tokens each client’s bucket can hold.
#  - PER_CLIENT_REFILL: how many tokens per second are refilled into each client's bucket
#    (independent of the global token bucket rate). If both the global and the per-client
#    bucket have tokens, the request is admitted. If either is empty, the request is rejected.
#  - In a real system, we might adjust these values based on client importance, SLAs, etc.
PER_CLIENT_CAPACITY = 1000
PER_CLIENT_REFILL = 300.0  # tokens/sec per client for simplicity

# Simulation modes (which signal do we use to adapt?):
#  1) MODE_DWELL_TIME = "dwell":
#     - We observe how long requests sit in the queue (dwell time). If average dwell
#       time is too high, we slow down the refill rate; if it’s low, we speed up.
#  2) MODE_CPU_BASED = "cpu":
#     - We (naively in this example) measure some "CPU usage" metric. If CPU is too high,
#       we reduce the rate; if low, we increase it. In a real system, we would connect this
#       to actual CPU usage measurements or concurrency-based CPU estimates.
#  3) MODE_CONCURRENCY = "concurrency":
#     - We cap how many requests are "in service" at once. If we often hit that limit,
#       we reduce admission; if we rarely hit it, we increase. This ensures concurrency
#       (and hence resource usage) stays within bounds at the cost of potentially rejecting
#       many requests once the limit is reached.
MODE_DWELL_TIME = "dwell"
MODE_CPU_BASED = "cpu"
MODE_CONCURRENCY = "concurrency"

# -----------------------------
# Data Structures
# -----------------------------
class TokenBucket:
    """
    Tracks a global token bucket or per-client bucket.

    Attributes of the token bucket class:

    capacity : float,
        maximum number of tokens the bucket can hold at once. If the bucket is "full,"
        adding more tokens won't exceed this capacity.

    tokens : float,
        Current number of tokens available in the bucket. Initialized to 'capacity',
        meaning it's "full" at the start.

    refill_rate : float
        Number of tokens added to the bucket *per second*.
        In the simulation, this can be changed adaptively (e.g., if the system is overloaded).

    name : str
        A label to help identify this token bucket (e.g., "GLOBAL" or a specific client with a "CLIENT-ID" ).
    """

    def __init__(self, capacity, refill_rate_per_sec, name="GLOBAL"):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate_per_sec
        self.name = name

    def refill(self, dt_sec):
        """
        Add tokens based on current refill rate and elapsed time in seconds.
        The amount of tokens added is (refill_rate * dt_sec). We then clamp 'tokens'
        so it doesn't exceed 'capacity'.
        """

        added = self.refill_rate * dt_sec
        self.tokens = min(self.capacity, self.tokens + added)

    def consume(self, amount=1):
        """
        Consume tokens if available; return True if successful.
        The logic here is to attempt to remove 'amount' tokens from the bucket for a single incoming request.

        Returns True and reduces 'self.tokens' if enough tokens are available;
        returns False if there aren't enough tokens i.e. the admission is denied.
        """
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class Request:
    """
    Represents a single incoming request with tracking data:
      - creation time
      - queue entry time
      - start service time
      - completion time
      - whether it's fast or slow
      - which client sent it
    """

    def __init__(self, t_create, is_fast=True, client_id=0):
        self.t_create = t_create  # Timestamp when the request was generated/created.
        self.queue_enter_time = None  # when request actually enqueued
        self.start_service_time = None  # When request begins processing (after waiting in queue).
        self.end_time = None  # When the request finishes processing.
        self.is_fast = is_fast # Boolean flag to classify request type (fast vs. slow).
        self.client_id = client_id # Identifier for which client issued this request.

    @property
    def dwell_time(self):
        """
            Calculate how long the request sat in the queue:
            dwell_time = (start_service_time - queue_enter_time).
            If the request hasn't started service or hasn't been queued, return 0.
        """

        if self.start_service_time is not None and self.queue_enter_time is not None:
            return self.start_service_time - self.queue_enter_time
        return 0


# We'll track stats
class Stats:
    """
        Collects counters/metrics on simulation outcomes:
          - admitted: number of requests that passed token-bucket checks & queue capacity
          - rejected: number of requests denied at admission (no tokens, queue full, etc.)
          - completed: number of requests that fully finished service
          - total_dwell_fast / total_dwell_slow: sum of queue-wait times for fast/slow requests
          - count_fast / count_slow: how many fast/slow requests we've observed (to calculate average dwell)
    """
    def __init__(self):
        self.admitted = 0
        self.rejected = 0
        self.completed = 0
        self.total_dwell_fast = 0
        self.total_dwell_slow = 0
        self.count_fast = 0
        self.count_slow = 0


# -----------------------------
# Simulator Class
# -----------------------------
class AdmissionSimulator:

    """
    Main class orchestrating the simulation of admission control.

    mode : str
        Determines which adaptive signal is used ("dwell", "cpu", or "concurrency").

    global_bucket : TokenBucket
        A single global token bucket that limits overall request admission rate.

    client_buckets : list[TokenBucket]
        A token bucket for each client (optional). Prevents a single client
        from hogging all tokens if they spam requests.

    fast_queue : deque
        Queue storing fast requests that have been admitted but haven't started service.
        Each element is (Request object, remaining_service_time).

    slow_queue : deque
        Queue for slow/heavy requests waiting to start service.

    stats : Stats
        Tracks statistics about admitted, rejected, and completed requests,
        plus dwell times, etc.

    estimated_cpu_usage : float
        Placeholder metric for CPU-based adaptation. In a real system, you'd
        tie this to actual concurrency or completion rates.

    fast_concurrency_limit : int
        Maximum number of fast requests that can be in-service simultaneously
        when using concurrency-based adaptation.

    slow_concurrency_limit : int
        Similar concurrency cap for slow requests.
    """
    def __init__(self, mode=MODE_DWELL_TIME):
        self.mode = mode

        # 1) Create the GLOBAL token bucket, which starts at a certain capacity
        #    and an initial refill rate. This bucket applies to *all* incoming requests.
        self.global_bucket = TokenBucket(GLOBAL_TOKEN_CAPACITY, INITIAL_REFILL_RATE, name="GLOBAL")

        # 2) Create a per-client bucket for each of the 100 clients.
        #    This is optional and can be used to ensure no single client monopolizes traffic.
        self.client_buckets = [
            TokenBucket(PER_CLIENT_CAPACITY, PER_CLIENT_REFILL, name=f"Client_{i}")
            for i in range(NUM_CLIENTS)
        ]

        # 3) Prepare two queues for admitted requests:
        #    fast_queue (for lighter/faster ops) and slow_queue (for heavier ops).
        #    Each queue holds tuples of (Request, remaining_service_time).
        self.fast_queue = deque()  # store (Request, remaining_service_time)
        self.slow_queue = deque()

        # Initialize the Stats object
        self.stats = Stats()

        # For CPU-based or concurrency-based modes, we need placeholders
        self.estimated_cpu_usage = 0.0  # simplified metric
        self.fast_concurrency_limit = 50
        self.slow_concurrency_limit = 30

    def run(self):

        """
            Main simulation loop. Steps in increments of TIME_STEP ms until we reach SIM_TIME.
            1) Refill token buckets.
            2) Generate new requests from clients.
            3) Attempt to admit those requests via token buckets + queue capacity.
            4) Update/service in-flight requests (decrement service times).
            5) Dispatch from queue to 'in service' if there's capacity (especially important for concurrency-based mode).
            6) Periodically adapt (raise/lower) refill rates based on dwell time/CPU/concurrency signals.
            7) Advance time and repeat.
        """
        current_time = 0

        # The next time we adapt the token-bucket refill rate (every ADAPT_INTERVAL ms).
        next_adapt_time = ADAPT_INTERVAL

        # We'll track how many requests are in service at a time
        fast_in_service = []
        slow_in_service = []

        # Run the simulation until we hit the total SIM_TIME
        while current_time < SIM_TIME:
            dt_sec = TIME_STEP / 1000.0  # Convert ms to seconds for token-bucket refill calculations.

            # 1. Refill global tokens (and per-client tokens) for this timestep
            self.global_bucket.refill(dt_sec)
            for cb in self.client_buckets:
                cb.refill(dt_sec)

            # Step 2: Generate new requests (Poisson arrivals, etc.) for the current timestep.
            self.generate_requests(current_time)

            # Step 3: Attempt to admit these newly generated requests using token buckets
            #         and queue capacity checks. Admitted requests go into fast_queue or slow_queue.
            self.admit_requests(current_time)

            # Step 4: Update any requests currently being processed by decrementing their remaining service time.
            #         If a request finishes, we record stats; otherwise it remains in processing.
            self.service_requests(fast_in_service, slow_in_service, dt_sec, current_time)

            # Step 5: Move requests from the fast/slow queues into in_service lists if there's concurrency space
            #         (particularly important in concurrency-based mode). If not concurrency-based, we might
            #         just move all queued requests to in service immediately.
            self.dispatch_requests_to_service(fast_in_service, slow_in_service, current_time)

            # Step 6: Adapt the token-bucket refill rate if it's time (i.e., if we passed next_adapt_time).
            if current_time >= next_adapt_time:
                self.adapt_refill_rate(current_time)
                next_adapt_time += ADAPT_INTERVAL

            # Step 7: Advance the simulation clock by TIME_STEP ms.
            current_time += TIME_STEP

        # Once the loop finishes, we have final statistics to return.
        return self.gather_results()

    def generate_requests(self, current_time):
        """
            Generate new incoming requests during this timestep.
            We assume:
              1) A Poisson arrival process with mean BASE_ARRIVAL_RATE (i.e., on average
                 BASE_ARRIVAL_RATE requests per time step).
              2) Each new request is randomly assigned:
                 - A client_id (from 0 to NUM_CLIENTS-1)
                 - A "fast" vs. "slow" classification based on percentages defined in the constants

            Implementation details:
              - We call 'self.poisson(avg_requests)' to get a random integer count
                for how many requests arrive in this time step.
              - For each of those requests, we create a Request object at 'current_time'.
              - We store all these newly created Request objects in 'self.new_requests',
                a temporary list that gets handled by 'admit_requests()' next.
        """
        # 1. Calculate how many requests arrive this timestep (Poisson distributed).
        avg_requests = BASE_ARRIVAL_RATE
        num_new = self.poisson(avg_requests)

        # 2. For each new request, assign:
        #    - client_id (random)
        #    - whether it's fast or slow (based on FAST_RATIO).
        for _ in range(num_new):
            client_id = random.randint(0, NUM_CLIENTS - 1)
            is_fast = (random.random() < FAST_RATIO)

            # Create the Request object with the current_time as creation time.
            req = Request(current_time, is_fast=is_fast, client_id=client_id)

            # We'll attempt to admit it in the "admit_requests" step
            # So we store it temporarily in a "new_requests" list
            # For simplicity, let's place them in a single staging area
            # We'll attach it to self.new_requests
            # But let's just store them directly in a temporary list to handle
            if not hasattr(self, "new_requests"):
                self.new_requests = []
            self.new_requests.append(req)

    def admit_requests(self, current_time):
        """
            Decide which newly generated requests are admitted into the system.
            Steps:
              1) We check if there's a 'self.new_requests' list. If none exist, there's
                 nothing to admit at this moment.
              2) For each request in 'self.new_requests':
                 a) We first try to consume 1 token from the *global* bucket.
                    - If not available, we mark the request as rejected (stats.rejected++)
                      and move on.
                 b) If the global bucket had a token, we then check the per-client
                    bucket for the request's client_id.
                    - If that also lacks tokens, we reject and increment stats.rejected.
                      (Optionally, we could "return" the global token to the bucket, but
                      here we skip that for simplicity.)
                 c) If both token checks pass, we place the request into either fast_queue
                    or slow_queue (depending on req.is_fast), as long as that queue isn't full.
                    - If the queue is at capacity, we reject.
                    - Otherwise, we set 'req.queue_enter_time = current_time' (indicating
                      the request is now queued) and append it with its respective
                      service time (FAST_SERVICE_TIME or SLOW_SERVICE_TIME).

            Each admitted request increments 'stats.admitted'. Rejected requests increment
            'stats.rejected'. Then we clear 'self.new_requests' because we've processed them.
        """

        # If no new requests were generated in this step, nothing to do.
        if not hasattr(self, "new_requests"):
            return

        for req in self.new_requests:
            # 1. Check global token bucket
            if not self.global_bucket.consume(1):
                # Reject
                self.stats.rejected += 1
                continue

            # 2. Check per-client bucket
            cbucket = self.client_buckets[req.client_id]
            if not cbucket.consume(1):
                # Reject
                self.stats.rejected += 1
                # Return global token? (In practice you might or might not.)
                # skip for simplicity and accounting of failed requests to the total cost.
                continue

            # 3. Enqueue
            if req.is_fast:
                if len(self.fast_queue) < FAST_QUEUE_CAPACITY:
                    req.queue_enter_time = current_time
                    self.fast_queue.append((req, FAST_SERVICE_TIME))
                    self.stats.admitted += 1
                else:
                    self.stats.rejected += 1
            else:
                if len(self.slow_queue) < SLOW_QUEUE_CAPACITY:
                    req.queue_enter_time = current_time
                    self.slow_queue.append((req, SLOW_SERVICE_TIME))
                    self.stats.admitted += 1
                else:
                    self.stats.rejected += 1

        # Clear the new_requests list so we don't re-check them next iteration.
        self.new_requests.clear()

    def service_requests(self, fast_in_service, slow_in_service, dt_sec, current_time):

        """
            Update the status of requests currently being processed (“in service”).

            We iterate over both 'fast_in_service' and 'slow_in_service' lists, which store:
                (request_object, remaining_service_time)

            For each request, we decrement its 'remaining_service_time' by the simulation step
            (TIME_STEP, in ms). If the request finishes (remaining_time <= 0), we:
              - Mark 'req.end_time = current_time'
              - Ensure 'req.start_service_time' is set (if not already set)
              - Increment 'stats.completed'
              - Add to 'stats.total_dwell_fast' or 'stats.total_dwell_slow' based on request type
                (so we can compute average dwell later)
              - Increment 'stats.count_fast' or 'stats.count_slow'

            Any request that still has remaining time after decrementing stays in the respective
            "still_busy" list. At the end, we overwrite the old list with the updated "still_busy"
            list.

            Parameters:
            -----------
            fast_in_service : list of (Request, float)
                Active fast requests, each with a request object and its remaining service time.
            slow_in_service : list of (Request, float)
                Active slow requests, each with a request object and its remaining service time.
            dt_sec : float
                Elapsed time in seconds for this step (unused here since we directly subtract
                TIME_STEP in ms; but dt_sec could be used if we want to subtract dt_sec * 1000).
            current_time : float
                Current simulation timestamp in ms.
        """

        # Process FAST requests
        still_busy_fast = []
        for (req, remaining) in fast_in_service:
            # Decrement the service time by TIME_STEP ms
            new_remaining = remaining - (TIME_STEP)
            if new_remaining <= 0:
                # This request has completed in this timestep
                req.end_time = current_time
                # If start_service_time wasn’t set, assume it started exactly TIME_STEP ms ago
                req.start_service_time = req.start_service_time or (current_time - TIME_STEP)
                self.stats.completed += 1
                # track dwell time
                # dwell time is defined as the time a request spends waiting in the queue before it begins execution
                # (service). We calculate this as the difference between the start of service and the time the request
                # entered the queue.
                self.stats.total_dwell_fast += (req.start_service_time - req.queue_enter_time)
                self.stats.count_fast += 1
            else:
                # Still needs more service time; keep it in the list
                still_busy_fast.append((req, new_remaining))

        # slow
        still_busy_slow = []
        for (req, remaining) in slow_in_service:
            new_remaining = remaining - (TIME_STEP)
            if new_remaining <= 0:
                # completed
                req.end_time = current_time
                req.start_service_time = req.start_service_time or (current_time - TIME_STEP)
                self.stats.completed += 1
                # track dwell time
                self.stats.total_dwell_slow += (req.start_service_time - req.queue_enter_time)
                self.stats.count_slow += 1
            else:
                still_busy_slow.append((req, new_remaining))

        # Update the original lists in-place
        fast_in_service[:] = still_busy_fast
        slow_in_service[:] = still_busy_slow

    def dispatch_requests_to_service(self, fast_in_service, slow_in_service, current_time):
        """
            Transfer requests from the waiting queues (fast_queue / slow_queue) into the
            lists of actively "in service" requests (fast_in_service / slow_in_service).

            In the concurrency-based mode, we enforce a maximum concurrency limit:
              - Only allow up to self.fast_concurrency_limit fast requests to be in service at once.
              - Only allow up to self.slow_concurrency_limit slow requests in service at once.
              - If we're below the limit and there's a request in the queue, we move one from
                the queue to the "in service" list, marking its 'start_service_time'.

            In dwell-time or CPU-based modes (i.e., not MODE_CONCURRENCY), we ignore concurrency
            limits and immediately move *all* queued requests into service.

            Parameters:
            -----------
            fast_in_service  : list of (Request, float)
                Requests actively being processed (fast). Each tuple is (request, remaining_service_time).

            slow_in_service  : list of (Request, float)
                Requests actively being processed (slow).

            current_time     : float
                The current simulation timestamp in ms, used to set request.start_service_time.
        """

        # For concurrency-based approach, we limit concurrency
        # For dwell-time mode, we don't strictly limit concurrency but let's do a simplified approach:

        if self.mode == MODE_CONCURRENCY:
            # We'll only allow up to fast_concurrency_limit in service
            while len(fast_in_service) < self.fast_concurrency_limit and self.fast_queue:
                req, remaining = self.fast_queue.popleft()
                req.start_service_time = current_time
                fast_in_service.append((req, remaining))
            while len(slow_in_service) < self.slow_concurrency_limit and self.slow_queue:
                req, remaining = self.slow_queue.popleft()
                req.start_service_time = current_time
                slow_in_service.append((req, remaining))

        else:
            # Let's assume no explicit concurrency limit, just move everything to service right away
            # (in a real system you'd have a worker-thread limit, etc.)
            while self.fast_queue:
                req, remaining = self.fast_queue.popleft()
                req.start_service_time = current_time
                fast_in_service.append((req, remaining))
            while self.slow_queue:
                req, remaining = self.slow_queue.popleft()
                req.start_service_time = current_time
                slow_in_service.append((req, remaining))

    def adapt_refill_rate(self, current_time):
        """
        Decide how (and whether) to change the global token-bucket refill rate
        (and per-client rates) based on our simulation mode.

        In this simplified demo:
          - MODE_DWELL_TIME:   We call 'adapt_dwell_time(current_time)' which checks the average
                              dwell time for fast/slow queues and adjusts rates up or down.
          - MODE_CPU_BASED:    We call 'adapt_cpu_based(current_time)', which looks at a placeholder
                              CPU usage metric (random in this demo) to decide if we scale up/down.
          - MODE_CONCURRENCY:  We call 'adapt_concurrency_based(current_time)', which might lower
                              the refill rate if we've been hitting concurrency limits too often.

        Note: In the concurrency-based approach, the main concurrency enforcement is done in
          'dispatch_requests_to_service()'. However, we can still optionally adjust token-bucket
          rates if we detect constant concurrency saturation.

        """
        if self.mode == MODE_DWELL_TIME:
            self.adapt_dwell_time(current_time)
        elif self.mode == MODE_CPU_BASED:
            self.adapt_cpu_based(current_time)
        elif self.mode == MODE_CONCURRENCY:
            # concurrency approach is basically enforced in dispatch_requests_to_service
            # might adjust token-bucket rate if concurrency was near limits
            self.adapt_concurrency_based(current_time)

    def adapt_dwell_time(self, current_time):
        """
            Adapt the token-bucket refill rate based on observed dwell times for fast and slow requests.

            How It Works:
              1) We compute the *average dwell time* (queue wait) for fast and slow requests since
                 the last adaptation point by looking at:
                    avg_fast_dwell = total_dwell_fast / count_fast (if count_fast > 0)
                    avg_slow_dwell = total_dwell_slow / count_slow (if count_slow > 0)
              2) We compare these averages to our thresholds (FAST_DWELL_THRESHOLD, SLOW_DWELL_THRESHOLD).
                 - If either avg_fast_dwell or avg_slow_dwell exceeds its threshold, we mark the system
                   as 'overloaded'.
                 - If both are below half their thresholds, we mark the system as 'underutilized'.
              3) If 'overloaded':
                 - We cut the global refill rate to 80% of its current value (but never below MIN_REFILL_RATE).
                 - We optionally do the same for each per-client bucket as well, applying a 20% reduction.
              4) If 'underutilized':
                 - We increase the global refill rate by 20%, clamped at MAX_REFILL_RATE.
                 - We do the same for each client bucket.
              5) After adjusting rates, we reset the dwell counters (total_dwell_fast, total_dwell_slow,
                 count_fast, count_slow) so that the next adaptation interval starts fresh.

        """
        # We'll do a naive approach: use the ratio of total dwell to count
        avg_fast_dwell = 0
        if self.stats.count_fast > 0:
            avg_fast_dwell = self.stats.total_dwell_fast / self.stats.count_fast
        avg_slow_dwell = 0
        if self.stats.count_slow > 0:
            avg_slow_dwell = self.stats.total_dwell_slow / self.stats.count_slow

        # Check thresholds
        overloaded = False
        underutilized = False

        if avg_fast_dwell > FAST_DWELL_THRESHOLD or avg_slow_dwell > SLOW_DWELL_THRESHOLD:
            overloaded = True
        # If both avg dwells are well below half the threshold => underutilized
        if avg_fast_dwell < (FAST_DWELL_THRESHOLD * 0.5) and avg_slow_dwell < (SLOW_DWELL_THRESHOLD * 0.5):
            underutilized = True

        if overloaded:
            # reduce global refill rate
            old_rate = self.global_bucket.refill_rate
            new_rate = max(MIN_REFILL_RATE, old_rate * 0.8)  # reduce by 20%
            self.global_bucket.refill_rate = new_rate

            # also reduce per-client
            for cb in self.client_buckets:
                cb.refill_rate = max(MIN_REFILL_RATE, cb.refill_rate * 0.8)

        elif underutilized:
            # raise global refill rate
            old_rate = self.global_bucket.refill_rate
            new_rate = min(MAX_REFILL_RATE, old_rate * 1.2)  # increase by 20%
            self.global_bucket.refill_rate = new_rate
            for cb in self.client_buckets:
                cb.refill_rate = min(MAX_REFILL_RATE, cb.refill_rate * 1.2)

        #  Reset dwell stats so next adapt is for fresh data
        self.stats.total_dwell_fast = 0
        self.stats.total_dwell_slow = 0
        self.stats.count_fast = 0
        self.stats.count_slow = 0

    def adapt_cpu_based(self, current_time):
        """
            Adjust the global/per-client token-bucket refill rates based on a mock CPU usage.

             Pseudocode:
                - If cpu_usage > 80%, we assume we're overloaded and cut the refill rate to 70%
                  of the current value (capped at MIN_REFILL_RATE).
                - If cpu_usage < 40%, we assume there's plenty of CPU headroom, so we raise the
                  refill rate by 10% (up to MAX_REFILL_RATE).
                - Otherwise (between 40% and 80%), we leave it unchanged.

            For a more realistic approach, compute CPU usage from concurrency or from the
            rate of completed requests times a cost factor per request, etc.
        """
        # We'll just randomly vary CPU usage for demonstration
        cpu_usage = random.uniform(0, 100)  # placeholder

        if cpu_usage > 80:
            old_rate = self.global_bucket.refill_rate
            new_rate = max(MIN_REFILL_RATE, old_rate * 0.7)
            self.global_bucket.refill_rate = new_rate
            for cb in self.client_buckets:
                cb.refill_rate = max(MIN_REFILL_RATE, cb.refill_rate * 0.7)
        elif cpu_usage < 40:
            old_rate = self.global_bucket.refill_rate
            new_rate = min(MAX_REFILL_RATE, old_rate * 1.1)
            self.global_bucket.refill_rate = new_rate
            for cb in self.client_buckets:
                cb.refill_rate = min(MAX_REFILL_RATE, cb.refill_rate * 1.1)

    def adapt_concurrency_based(self, current_time):
        """
        Adjust the global/per-client token-bucket refill rates based on whether we're
        frequently hitting concurrency limits (fast_concurrency_limit / slow_concurrency_limit).

        In a REAL concurrency-based approach, you'd track how often or how many times
        you reached 'fast_concurrency_limit' or 'slow_concurrency_limit' in the last
        adapt interval. For instance, if dispatch_requests_to_service() saw we were
        at capacity most of the time, it indicates we're saturating concurrency =>
        we should throttle the admission rate.

        Here, we do a simple 'was_at_limit' = random check (30% chance) to illustrate
        the logic:
        - If 'was_at_limit' = True => we assume we're hitting concurrency caps,
            so we reduce refill rates by 20%.
        - Otherwise, we assume we have spare concurrency => raise refill rates by 10%.

        """
        # For a real approach, you'd track how often we were at concurrency capacity.
        # We'll do a random approach:
        was_at_limit = (random.random() < 0.3)  # 30% chance
        if was_at_limit:
            old_rate = self.global_bucket.refill_rate
            self.global_bucket.refill_rate = max(MIN_REFILL_RATE, old_rate * 0.8)
            for cb in self.client_buckets:
                cb.refill_rate = max(MIN_REFILL_RATE, cb.refill_rate * 0.8)
        else:
            old_rate = self.global_bucket.refill_rate
            self.global_bucket.refill_rate = min(MAX_REFILL_RATE, old_rate * 1.1)
            for cb in self.client_buckets:
                cb.refill_rate = min(MAX_REFILL_RATE, cb.refill_rate * 1.1)

    def gather_results(self):

        """
            Collect final statistics from the simulation and return them in a dictionary.

            This typically includes:
              - mode: which adaptive approach we used ("dwell", "cpu", or "concurrency")
              - admitted: total number of requests that passed admission checks
              - rejected: total number of requests that were turned away
              - completed: total number of requests that actually finished execution
              - final_global_refill: the final refill rate of the global token bucket
                at the end of the simulation, indicating how the adaptive logic settled

            Returns:
            --------
            dict : {
              "mode": str,
              "admitted": int,
              "rejected": int,
              "completed": int,
              "final_global_refill": float
            }
            """
        return {
            "mode": self.mode,
            "admitted": self.stats.admitted,
            "rejected": self.stats.rejected,
            "completed": self.stats.completed,
            "final_global_refill": self.global_bucket.refill_rate
        }

    @staticmethod
    def poisson(lmbda):
        """
            Generate a Poisson-distributed random integer with mean 'lmbda'.

            The Poisson distribution is commonly used to model the number of arrivals
            or events in a fixed interval, given an average rate 'λ' (lmbda).

            Implementation Notes:
              - This method uses a simplified version of Knuth's algorithm for generating
                Poisson random variables. It works by multiplying random numbers until
                the product drops below 'e^(-λ)'.
              - k ends up as the number of increments before p < e^(-lmbda). The final result
                (k - 1) is the Poisson random variable.
              - If k is 0 at the end (which would be unusual), we just return 0.
              - This is enough for demonstration, but for high λ or more precision, you'd
                likely use a more optimized method or a library function (e.g., NumPy).

            Steps (Knuth’s Algorithm, simplified):
              1. Let p = 1.0
              2. Let k = 0
              3. Let e_lmbda = exp(-lmbda)
              4. While p > e_lmbda:
                   k += 1
                   p *= random.random()
              5. Return k - 1 (since we overshoot by 1 in the loop)
        """
        # Simplified approach: convert to a small integer for arrivals in a step
        # More accurately we'd use Knuth or other algorithms. But this is enough for demonstration.
        p = 1.0
        k = 0
        e_lmbda = math.exp(-lmbda)
        while p > e_lmbda:
            k += 1
            p *= random.random()
        return k - 1 if k > 0 else 0


def run_one_hour(mode, seed):
    random.seed(seed)
    sim = AdmissionSimulator(mode=mode)
    return sim.run()

# -----------------------------
# Main: Run and Compare
# -----------------------------
if __name__ == "__main__":
    """
        Entry point for running the simulation in its three different modes:
          1) Dwell-Time–Based
          2) CPU-Based
          3) Concurrency-Based

        We set a fixed random seed (42) for reproducibility. Then we instantiate a
        AdmissionSimulator in each mode, run it, and print the resulting metrics.
    """
    # random.seed(42)  # for reproducibility
    #
    # # 1. Dwell-Time–Based
    # sim_dwell = AdmissionSimulator(mode=MODE_DWELL_TIME)
    # result_dwell = sim_dwell.run()
    # print("Dwell-Time Approach:", result_dwell)
    #
    # # 2. CPU-Based
    # sim_cpu = AdmissionSimulator(mode=MODE_CPU_BASED)
    # result_cpu = sim_cpu.run()
    # print("CPU-Based Approach:", result_cpu)
    #
    # # 3. Concurrency-Based
    # sim_conc = AdmissionSimulator(mode=MODE_CONCURRENCY)
    # result_conc = sim_conc.run()
    # print("Concurrency-Based Approach:", result_conc)

    all_results = []

    with open("one_hour_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mode", "Seed", "Admitted", "Rejected", "Completed", "FinalRefill"])

        for mode in [MODE_DWELL_TIME, MODE_CPU_BASED, MODE_CONCURRENCY]:
            for seed in [42, 43, 44]:  # run 3 seeds per mode
                result = run_one_hour(mode, seed)
                result["Seed"] = seed
                all_results.append(result)

                writer.writerow([
                    mode,
                    seed,
                    result["admitted"],
                    result["rejected"],
                    result["completed"],
                    result["final_global_refill"]
                ])
                print(f"Finished: mode={mode}, seed={seed}", result)

        # Let's separate out the results by mode:
        dwell_data = [r for r in all_results if r["mode"] == MODE_DWELL_TIME]
        cpu_data = [r for r in all_results if r["mode"] == MODE_CPU_BASED]
        conc_data = [r for r in all_results if r["mode"] == MODE_CONCURRENCY]

        # Sort each list by seed (just to have consistent ordering)
        dwell_data.sort(key=lambda x: x["Seed"])
        cpu_data.sort(key=lambda x: x["Seed"])
        conc_data.sort(key=lambda x: x["Seed"])

        # We'll create a bar chart showing (admitted, rejected, completed) for each run
        # grouped by (mode, seed).

        # Combine all into one list again, but in a sorted manner:
        sorted_results = dwell_data + cpu_data + conc_data

        labels = [f"{res['mode']}-seed{res['Seed']}" for res in sorted_results]
        admitted_vals = [res["admitted"] for res in sorted_results]
        rejected_vals = [res["rejected"] for res in sorted_results]
        completed_vals = [res["completed"] for res in sorted_results]

        x = np.arange(len(labels))  # indices
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        # Position each bar: we do x - width, x, x + width for the 3 categories
        bar_adm = ax.bar(x - width, admitted_vals, width, label='Admitted')
        bar_rej = ax.bar(x, rejected_vals, width, label='Rejected')
        bar_cmp = ax.bar(x + width, completed_vals, width, label='Completed')

        ax.set_ylabel("Number of Requests")
        ax.set_title("Admitted / Rejected / Completed by Mode & Seed")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        plt.savefig("bar_admitted_rejected_completed.png")
        plt.show()

        # (3) Plot final refill as a line chart
        # We'll group each mode and seeds for a line plot
        import collections

        # dict: mode -> list of (seed, final_refill)
        mode_refill_map = collections.defaultdict(list)

        for r in all_results:
            mode_refill_map[r["mode"]].append((r["Seed"], r["final_global_refill"]))

        # sort each list by seed
        for m in mode_refill_map:
            mode_refill_map[m].sort(key=lambda x: x[0])

        fig2, ax2 = plt.subplots()
        for m in mode_refill_map:
            seeds_ = [item[0] for item in mode_refill_map[m]]
            refills_ = [item[1] for item in mode_refill_map[m]]
            ax2.plot(seeds_, refills_, marker='o', label=m)

        ax2.set_xlabel("Seed")
        ax2.set_ylabel("Final Global Refill Rate")
        ax2.set_title("Final Refill Rate vs. Seed (by Mode)")
        ax2.legend()
        plt.tight_layout()
        plt.savefig("line_final_refill.png")
        plt.show()