import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Configuration
PORT = 8888
WORKER_POOL_SIZE = 8
FAST_REJECT = False # Toggle between fast reject and tar pit mode
TAR_PIT_DELAY = (5, 10) # Delay range for tar pit mode (seconds)
SIMULATION_DURATION = 120 # Total simulation time (seconds)
QUEUE_THRESHOLD = 5 # Reject requests when queue exceeds this size
PROCESSING_TIME_RANGE = (0.004, 0.01) # Processing time range (4ms to 10ms)

# Metrics
metrics = {
    "latencies": [], # Track sojourn times (latency) - for normal completed requests
    "tarpit_latencies": [], # Track latencies for tar pit responses
    "concurrency": [], # Track queue length over time - sampled periodically
    "timestamps": [], # Track timestamps for concurrency measurements
    "rejected_count": 0, # Count of rejected requests (fast reject mode only)
    "completed_count": 0, # Count of completed requests (normal processing)
    "tarpit_count": 0, # Count of tar pit responses
    "request_start_times": {}, # Map request_id -> start_time for latency calculation
}

# Request counter for unique IDs
request_counter = 0

# Semaphore for worker pool
worker_pool = asyncio.Semaphore(WORKER_POOL_SIZE)

# Thread pool for handling responses
response_executor = ThreadPoolExecutor(max_workers=WORKER_POOL_SIZE)

# Logging helper
def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Response logic (runs in thread pool)
def handle_response(client_address, tar_pit=False):
    if tar_pit:
        delay = random.uniform(*TAR_PIT_DELAY)
        log(f"Applying tar pit delay of {delay:.2f} seconds for client {client_address}")
        time.sleep(delay)
    else:
        log(f"Responding normally to client {client_address}")

# Worker logic
async def handle_client(reader, writer):
    global request_counter
    request_counter += 1
    request_id = request_counter
    
    client_address = writer.get_extra_info('peername')
    log(f"Handling client {client_address} (Request #{request_id})")
    
    # Record the start time immediately when request arrives
    start_time = time.time()
    metrics["request_start_times"][request_id] = start_time
    
    # Check queue status BEFORE acquiring semaphore to get true queue length
    active_workers = WORKER_POOL_SIZE - worker_pool._value
    queued_requests = len(worker_pool._waiters)
    current_load = active_workers + queued_requests
    
    # Record concurrency measurement
    metrics["concurrency"].append(current_load)
    metrics["timestamps"].append(start_time)
    
    # Apply queue threshold check
    if queued_requests > QUEUE_THRESHOLD:
        if FAST_REJECT:
            # Fast reject mode - drop the connection immediately
            log(f"Rejecting client {client_address} (Request #{request_id}) - queue length: {queued_requests}")
            metrics["rejected_count"] += 1
            # Clean up the start time tracking
            del metrics["request_start_times"][request_id]
            writer.close()
            await writer.wait_closed()
            return
        else:
            # Tar pit mode - give a slow response instead of rejecting
            log(f"Tar pit mode: slow response for client {client_address} (Request #{request_id}) - queue length: {queued_requests}")
            
            # Use our dedicated thread pool for tar pit response
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                response_executor, 
                handle_response, 
                client_address, 
                True  # Enable tar pit delay
            )
            
            # Record tar pit completion metrics
            end_time = time.time()
            request_start_time = metrics["request_start_times"][request_id]
            latency = end_time - request_start_time
            
            metrics["tarpit_latencies"].append(latency)
            metrics["tarpit_count"] += 1
            
            # Clean up tracking
            del metrics["request_start_times"][request_id]
            
            writer.close()
            await writer.wait_closed()
            log(f"Finished tar pit response for client {client_address} (Request #{request_id})")
            return
    
    # Normal processing path (queue is not full)
    async with worker_pool:        
        # Simulate normal work using configurable range
        processing_time = random.uniform(*PROCESSING_TIME_RANGE)
        log(f"Simulating {processing_time * 1000:.2f} ms of work for client {client_address} (Request #{request_id})")
        await asyncio.sleep(processing_time) # Simulate async work
        
        # Use our dedicated thread pool for normal response handling
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            response_executor, 
            handle_response, 
            client_address, 
            False  # Normal response, no tar pit delay
        )
        
        # Record normal completion metrics
        end_time = time.time()
        request_start_time = metrics["request_start_times"][request_id]
        latency = end_time - request_start_time
        
        metrics["latencies"].append(latency)
        metrics["completed_count"] += 1
        
        # Clean up tracking
        del metrics["request_start_times"][request_id]
        
    writer.close()
    await writer.wait_closed()
    log(f"Finished handling client {client_address} (Request #{request_id})")

# Main server logic
async def main():
    server = await asyncio.start_server(handle_client, '127.0.0.1', PORT)
    log(f"Server started on port {PORT}")
    log(f"Configuration: FAST_REJECT={FAST_REJECT}, WORKER_POOL_SIZE={WORKER_POOL_SIZE}, QUEUE_THRESHOLD={QUEUE_THRESHOLD}")
    
    try:
        async with server:
            await asyncio.sleep(SIMULATION_DURATION)
            server.close()
            await server.wait_closed()
    finally:
        # Properly shutdown the thread pool
        response_executor.shutdown(wait=True)
        log("Thread pool shut down")
    
    log("Simulation ended")
    plot_metrics()

# Plotting function
def plot_metrics():
    # Print summary stats
    total_requests = metrics['completed_count'] + metrics['rejected_count'] + metrics['tarpit_count']
    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Total requests: {total_requests}")
    print(f"Normal completed requests: {metrics['completed_count']}")
    print(f"Tar pit responses: {metrics['tarpit_count']}")
    print(f"Rejected requests: {metrics['rejected_count']}")
    
    if total_requests > 0:
        print(f"Normal completion rate: {metrics['completed_count'] / total_requests * 100:.1f}%")
        print(f"Tar pit response rate: {metrics['tarpit_count'] / total_requests * 100:.1f}%")
        print(f"Rejection rate: {metrics['rejected_count'] / total_requests * 100:.1f}%")
    
    if metrics["latencies"]:
        print(f"Normal request - Average latency: {sum(metrics['latencies']) / len(metrics['latencies']):.3f}s")
        print(f"Normal request - Max latency: {max(metrics['latencies']):.3f}s")
        print(f"Normal request - Min latency: {min(metrics['latencies']):.3f}s")
    
    if metrics["tarpit_latencies"]:
        print(f"Tar pit request - Average latency: {sum(metrics['tarpit_latencies']) / len(metrics['tarpit_latencies']):.3f}s")
        print(f"Tar pit request - Max latency: {max(metrics['tarpit_latencies']):.3f}s")
        print(f"Tar pit request - Min latency: {min(metrics['tarpit_latencies']):.3f}s")
        
        if metrics["latencies"]:
            normal_avg = sum(metrics['latencies']) / len(metrics['latencies'])
            tarpit_avg = sum(metrics['tarpit_latencies']) / len(metrics['tarpit_latencies'])
            print(f"Tar pit impact: {tarpit_avg / normal_avg:.1f}x slower than normal requests")
    
    # Create plots
    plt.figure(figsize=(15, 12))
    
    # Plot latency comparison
    plt.subplot(4, 1, 1)
    if metrics["latencies"] or metrics["tarpit_latencies"]:
        if metrics["latencies"]:
            plt.plot(range(len(metrics["latencies"])), metrics["latencies"], 
                    marker="o", linestyle="-", label=f"Normal Requests ({len(metrics['latencies'])})", 
                    markersize=3, alpha=0.7)
        if metrics["tarpit_latencies"]:
            plt.plot(range(len(metrics["tarpit_latencies"])), metrics["tarpit_latencies"], 
                    marker="s", linestyle="-", label=f"Tar Pit Requests ({len(metrics['tarpit_latencies'])})", 
                    markersize=3, alpha=0.7, color='red')
        plt.title("Request Latency Comparison: Normal vs Tar Pit")
        plt.xlabel("Request Index")
        plt.ylabel("Latency (seconds)")
        plt.grid(True)
        plt.legend()
        plt.yscale('log')  # Log scale to better show the difference
    else:
        plt.text(0.5, 0.5, 'No completed requests', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Request Latency (No Data)")
    
    # Plot concurrency over time
    plt.subplot(4, 1, 2)
    if metrics["timestamps"] and metrics["concurrency"]:
        start_time = min(metrics["timestamps"])
        relative_times = [(t - start_time) for t in metrics["timestamps"]]
        plt.plot(relative_times, metrics["concurrency"], marker="o", linestyle="-", 
                label="Concurrency (Queue Length)", markersize=2)
        plt.axhline(y=QUEUE_THRESHOLD, color='r', linestyle='--', alpha=0.7, 
                   label=f'Queue Threshold ({QUEUE_THRESHOLD})')
        plt.title("Concurrency (Queue Length) Over Time")
        plt.xlabel("Time (seconds from start)")
        plt.ylabel("Concurrency (Active + Queued)")
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No concurrency data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Concurrency Over Time (No Data)")
    
    # Plot latency histogram comparison
    plt.subplot(4, 1, 3)
    if metrics["latencies"] or metrics["tarpit_latencies"]:
        bins = 30
        if metrics["latencies"]:
            plt.hist(metrics["latencies"], bins=bins, alpha=0.7, label=f"Normal ({len(metrics['latencies'])})", 
                    color='blue', edgecolor='black', linewidth=0.5)
        if metrics["tarpit_latencies"]:
            plt.hist(metrics["tarpit_latencies"], bins=bins, alpha=0.7, label=f"Tar Pit ({len(metrics['tarpit_latencies'])})", 
                    color='red', edgecolor='black', linewidth=0.5)
        plt.title("Latency Distribution: Normal vs Tar Pit")
        plt.xlabel("Latency (seconds)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No latency data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Latency Distribution (No Data)")
    
    # Plot request type breakdown
    plt.subplot(4, 1, 4)
    categories = ['Normal\nCompleted', 'Tar Pit\nResponses', 'Rejected']
    counts = [metrics['completed_count'], metrics['tarpit_count'], metrics['rejected_count']]
    colors = ['green', 'orange', 'red']
    
    if sum(counts) > 0:
        bars = plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        plt.title("Request Handling Breakdown")
        plt.ylabel("Number of Requests")
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No request data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Request Handling Breakdown (No Data)")
    
    # Save and show the plots
    plt.tight_layout()
    plt.savefig("metrics_plots.png", dpi=150, bbox_inches='tight')
    plt.show()

# Run the server
if __name__ == "__main__":
    asyncio.run(main())