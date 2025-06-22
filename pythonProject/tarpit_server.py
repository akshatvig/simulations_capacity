import asyncio
import time
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import threading

should_close_connection = False

# Configuration
PORT = 8888
WORKER_POOL_SIZE = 8
FAST_REJECT = True # Toggle between fast reject and tar pit mode
TAR_PIT_DELAY = (0.5, 1) # Delay range for tar pit mode (seconds)
SIMULATION_DURATION = 10 # Total simulation time (seconds)
QUEUE_THRESHOLD = 5 # Reject requests when queue exceeds this size
PROCESSING_TIME_RANGE = (0.004, 0.01) # Processing time range (4ms to 10ms)

# Load generation configuration
LOAD_GENERATION_ENABLED = True # Whether to generate client load
CLIENT_ARRIVAL_RATE = 12.0 # Average requests per second (Poisson distribution)
BURST_MODE = True # If True, sends bursts of requests to test queue limits
BURST_SIZE = 100 # Number of requests in each burst
BURST_INTERVAL = 1 # Seconds between bursts

# Global metrics tracking
metrics = {
    "timestamps": [],
    "latencies": [],
    "tarpit_latencies": [],
    "concurrency": [],
    "completed_count": 0,
    "tarpit_count": 0,
    "rejected_count": 0
}

# Track actual request queue depth safely
current_queue_depth = 0
queue_depth_lock = threading.Lock()

# Thread-safe logging
log_lock = threading.Lock()

def log(message):
    with log_lock:
        timestamp = time.strftime('%H:%M:%S', time.localtime())
        print(f"[{timestamp}] {message}")

# Server state
active_connections = 0
connection_lock = threading.Lock()
worker_pool = asyncio.Semaphore(WORKER_POOL_SIZE)

# Shared thread pool for ALL responses (normal AND tar pit)
# This creates realistic resource contention - tar pit hurts normal users
response_executor = ThreadPoolExecutor(max_workers=WORKER_POOL_SIZE)

def update_concurrency():
    """Track actual request queue depth safely"""
    with queue_depth_lock:
        metrics["timestamps"].append(time.time())
        metrics["concurrency"].append(current_queue_depth)

# Response logic (runs in thread pool)
def handle_response(client_address, tar_pit=False):
    if tar_pit:
        delay = random.uniform(*TAR_PIT_DELAY)
        log(f"Applying tar pit delay of {delay:.2f} seconds for client {client_address}")
        time.sleep(delay)

async def handle_request(reader, writer):
    global active_connections, current_queue_depth
    
    client_address = writer.get_extra_info('peername')
    start_time = time.time()
    
    with connection_lock:
        active_connections += 1
    
    # Track queue depth - increment when request starts processing
    with queue_depth_lock:
        current_queue_depth += 1
    
    update_concurrency()
    
    try:
        # Read request (simplified)
        request_data = await reader.read(1024)
        
        # Check actual request queue depth for throttling decision
        with queue_depth_lock:
            queued_requests = current_queue_depth
        
        # Determine if request should be tar pitted based on ACTUAL queue depth
        should_tar_pit = False
        if not FAST_REJECT and queued_requests > QUEUE_THRESHOLD:
            should_tar_pit = True
        elif FAST_REJECT and queued_requests > QUEUE_THRESHOLD:
            # Fast reject - send rejection response but keep connection open
            response = b"HTTP/1.1 503 Service Unavailable\r\nConnection: keep-alive\r\nContent-Length: 16\r\n\r\nService Overloaded"
            writer.write(response)
            await writer.drain()
            
            # Count rejection but don't modify connection tracking
            metrics["rejected_count"] += 1
            return  # finally block handles connection cleanup
        
        if should_tar_pit:
            # Tar pit mode - delay response significantly
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(response_executor, handle_response, client_address, True)
            
            response = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nTar pit delay"
            writer.write(response)
            await writer.drain()
            
            end_time = time.time()
            latency = end_time - start_time
            metrics["tarpit_latencies"].append(latency)
            metrics["tarpit_count"] += 1
        else:
            # Normal processing - acquire worker semaphore
            async with worker_pool:
                # Simulate request processing time
                processing_time = random.uniform(*PROCESSING_TIME_RANGE)
                await asyncio.sleep(processing_time)
                
                # Use thread pool for response (shared with tar pit)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(response_executor, handle_response, client_address, False)
                
                # Send response
                response = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\nHello World"
                writer.write(response)
                await writer.drain()
                
                end_time = time.time()
                latency = end_time - start_time
                metrics["latencies"].append(latency)
                metrics["completed_count"] += 1
                return 
    
    except Exception as e:
        log(f"Error handling request from {client_address}: {e}")
    finally:
        # Decrement queue depth when request completes
        with queue_depth_lock:
            current_queue_depth -= 1
                
        update_concurrency()
        
        if should_close_connection:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
            
            with connection_lock:
                active_connections -= 1
            update_concurrency()
            

async def start_server():
    server = await asyncio.start_server(handle_request, 'localhost', PORT)
    log(f"Server started on port {PORT}")
    log(f"Worker pool size: {WORKER_POOL_SIZE}")
    log(f"Queue threshold: {QUEUE_THRESHOLD}")
    log(f"Fast reject mode: {FAST_REJECT}")
    
    async with server:
        await server.serve_forever()

async def send_request():
    try:
        reader, writer = await asyncio.open_connection('localhost', PORT)
        
        # Send a simple HTTP GET request
        request = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        writer.write(request)
        await writer.drain()
        
        # Read response (with timeout)
        try:
            response = await asyncio.wait_for(reader.read(1024), timeout=30.0)
        except asyncio.TimeoutError:
            log("Request timed out")
        
        writer.close()
        await writer.wait_closed()
    except Exception as e:
        # Connection refused or other error - happens when fast reject is enabled
        pass

async def generate_load():
    log(f"Starting load generation for {SIMULATION_DURATION} seconds")
    
    if BURST_MODE:
        log(f"Burst mode: {BURST_SIZE} requests every {BURST_INTERVAL} seconds")
        bursts = int(SIMULATION_DURATION / BURST_INTERVAL)
        
        for burst in range(bursts):
            burst_start = time.time()
            log(f"Sending burst {burst + 1}/{bursts} ({BURST_SIZE} requests)")
            
            # Send burst of requests
            tasks = []
            for _ in range(BURST_SIZE):
                task = asyncio.create_task(send_request())
                tasks.append(task)
            
            # Wait for all requests in burst to complete (with reasonable timeout)
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=60.0)
            except asyncio.TimeoutError:
                log(f"Burst {burst + 1} timed out")
            
            # Wait for next burst interval
            elapsed = time.time() - burst_start
            wait_time = max(0, BURST_INTERVAL - elapsed)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
    else:
        # Poisson arrival process
        total_requests = int(CLIENT_ARRIVAL_RATE * SIMULATION_DURATION)
        log(f"Generating {total_requests} requests over {SIMULATION_DURATION} seconds")
        
        tasks = []
        for i in range(total_requests):
            # Calculate next arrival time using exponential distribution
            inter_arrival_time = random.expovariate(CLIENT_ARRIVAL_RATE)
            await asyncio.sleep(inter_arrival_time)
            
            task = asyncio.create_task(send_request())
            tasks.append(task)
        
        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    log("Load generation completed")

async def main():
    global start_time
    start_time = time.time()
    
    # Start the server
    server_task = asyncio.create_task(start_server())
    
    # Wait a moment for server to start
    await asyncio.sleep(0.1)
    
    if LOAD_GENERATION_ENABLED:
        # Generate load
        await generate_load()
        
        # Give some time for remaining responses to complete
        log("Waiting for remaining responses to complete...")
        await asyncio.sleep(5)
    else:
        # Just run the server
        await server_task

    # Calculate resource utilization timeline
    simulation_end_time = start_time + SIMULATION_DURATION
    actual_end_time = time.time()
    total_runtime = actual_end_time - start_time
    wasted_time = total_runtime - SIMULATION_DURATION
    waste_percentage = (wasted_time / total_runtime) * 100
    
    # Analysis and plotting
    total_requests = metrics["completed_count"] + metrics["tarpit_count"] + metrics["rejected_count"]
    
    print(f"\n{'='*60}")
    print(f"SERVER PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total Requests: {total_requests}")
    print(f"Completed (Normal): {metrics['completed_count']}")
    print(f"Tar Pit Responses: {metrics['tarpit_count']}")
    print(f"Rejected: {metrics['rejected_count']}")
    
    if metrics["latencies"]:
        avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"])
        print(f"Average Normal Latency: {avg_latency*1000:.2f} ms")
    
    if metrics["tarpit_latencies"]:
        avg_tarpit_latency = sum(metrics["tarpit_latencies"]) / len(metrics["tarpit_latencies"])
        print(f"Average Tar Pit Latency: {avg_tarpit_latency*1000:.2f} ms")
    
    print(f"\n{'='*60}")
    print(f"RESOURCE WASTE ANALYSIS")
    print(f"{'='*60}")
    print(f"Load Generation Duration: {SIMULATION_DURATION} seconds")
    print(f"Total Server Runtime: {total_runtime:.1f} seconds")
    print(f"Wasted Time (post-load): {wasted_time:.1f} seconds")
    print(f"Resource Waste: {waste_percentage:.1f}% of total runtime")
    
    if not FAST_REJECT and wasted_time > 0:
        print(f"\n⚠️  TAR PIT IMPACT:")
        print(f"   Server spent {waste_percentage:.1f}% of time on useless work!")
        print(f"   Real clients could not be served for {wasted_time:.1f} seconds")

    # Create plots with 6 subplots to include waste analysis
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Server Performance Analysis: Tar Pit vs Fast Reject Impact', fontsize=16)
    
    # Plot 1: Latency comparison (top left)
    if metrics["latencies"] or metrics["tarpit_latencies"]:
        if metrics["latencies"]:
            ax1.plot(range(len(metrics["latencies"])), metrics["latencies"], 
                    marker="o", linestyle="-", label=f"Normal Requests ({len(metrics['latencies'])})", 
                    markersize=3, alpha=0.7)
        if metrics["tarpit_latencies"]:
            ax1.plot(range(len(metrics["tarpit_latencies"])), metrics["tarpit_latencies"], 
                    marker="s", linestyle="-", label=f"Tar Pit Requests ({len(metrics['tarpit_latencies'])})", 
                    markersize=3, alpha=0.7, color='red')
        ax1.set_title("Request Latency Comparison: Normal vs Tar Pit")
        ax1.set_xlabel("Request Index")
        ax1.set_ylabel("Latency (seconds)")
        ax1.grid(True)
        ax1.legend()
        ax1.set_yscale('log')  # Log scale to better show the difference
    else:
        ax1.text(0.5, 0.5, 'No completed requests', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Request Latency (No Data)")
    
    # Plot 2: Concurrency over time (top right)
    if metrics["timestamps"] and metrics["concurrency"]:
        start_time_plot = min(metrics["timestamps"])
        relative_times = [(t - start_time_plot) for t in metrics["timestamps"]]
        ax2.plot(relative_times, metrics["concurrency"], marker="o", linestyle="-", 
                label="Request Queue Depth", markersize=2)
        ax2.axhline(y=QUEUE_THRESHOLD, color='r', linestyle='--', alpha=0.7, 
                   label=f'Queue Threshold ({QUEUE_THRESHOLD})')
        ax2.set_title("Request Queue Depth Over Time")
        ax2.set_xlabel("Time (seconds from start)")
        ax2.set_ylabel("Queued Requests (Waiting for Workers)")
        ax2.grid(True)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No queue depth data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Request Queue Depth (No Data)")
    
    # Plot 3: Latency histogram comparison (middle left)
    if metrics["latencies"] or metrics["tarpit_latencies"]:
        bins = 30
        if metrics["latencies"]:
            ax3.hist(metrics["latencies"], bins=bins, alpha=0.7, label=f"Normal ({len(metrics['latencies'])})", 
                    color='blue', edgecolor='black', linewidth=0.5)
        if metrics["tarpit_latencies"]:
            ax3.hist(metrics["tarpit_latencies"], bins=bins, alpha=0.7, label=f"Tar Pit ({len(metrics['tarpit_latencies'])})", 
                    color='red', edgecolor='black', linewidth=0.5)
        ax3.set_title("Latency Distribution: Normal vs Tar Pit")
        ax3.set_xlabel("Latency (seconds)")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No latency data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Latency Distribution (No Data)")
    
    # Plot 4: Request type breakdown (middle right)
    categories = ['Normal\nCompleted', 'Tar Pit\nResponses', 'Rejected']
    counts = [metrics['completed_count'], metrics['tarpit_count'], metrics['rejected_count']]
    colors = ['green', 'orange', 'red']
    
    if sum(counts) > 0:
        bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax4.set_title("Request Handling Breakdown")
        ax4.set_ylabel("Number of Requests")
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No request data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Request Handling Breakdown (No Data)")
    
    # Plot 5: Resource utilization timeline (bottom left) - NEW WASTE GRAPH
    ax5.axvspan(0, SIMULATION_DURATION, alpha=0.3, color='green', label=f'Load Generation ({SIMULATION_DURATION}s)')
    if wasted_time > 0:
        ax5.axvspan(SIMULATION_DURATION, total_runtime, alpha=0.3, color='red', label=f'Wasted Resources ({wasted_time:.1f}s)')
    ax5.set_xlim(0, max(total_runtime, SIMULATION_DURATION + 1))
    ax5.set_ylim(0, 1)
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Resource Usage')
    ax5.set_title(f'Server Resource Timeline\nWaste: {waste_percentage:.1f}% of total runtime')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add text annotation for waste
    if wasted_time > 0:
        ax5.text(SIMULATION_DURATION + wasted_time/2, 0.5, f'{waste_percentage:.1f}%\nWASTED', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    
    # Plot 6: Resource efficiency comparison (bottom right) - NEW WASTE GRAPH
    waste_categories = ['Useful Work', 'Wasted Work']
    useful_time = SIMULATION_DURATION
    waste_values = [useful_time, max(0, wasted_time)]  # Ensure no negative values
    waste_colors = ['green', 'red']
    
    bars = ax6.bar(waste_categories, waste_values, color=waste_colors, alpha=0.7)
    ax6.set_ylabel('Time (seconds)')
    ax6.set_title('Resource Efficiency Analysis')
    ax6.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, v) in enumerate(zip(bars, waste_values)):
        if v > 0:
            percentage = (v / total_runtime) * 100
            ax6.text(i, v/2, f'{percentage:.1f}%\n({v:.1f}s)', 
                    ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Save and show the plots
    plt.tight_layout()
    plt.savefig("metrics_plots_with_waste.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Simulation interrupted by user")
    finally:
        log("Shutting down simulation")
        response_executor.shutdown(wait=False)