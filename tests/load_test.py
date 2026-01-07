## Multithreaded Load Tester for API Endpoints
# This script simulates concurrent load on a prediction endpoint using multiple threads.
# It measures latency, success rates, and handles timing to mimic real-world traffic patterns.

from typing import Tuple, List, Dict
import httpx
import numpy as np
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event, Thread, current_thread
import random 
from queue import Queue

# --- TEST CONFIGURATION ---
# Target endpoint for load testing (adjust as needed for your service)
LOAD_HOST = "http://localhost:8005/predict_batched/RawLogClassifier_10"
DIM = 128  # Dimensionality of input features (matches model expectations)

# Per-worker/thread configuration
# OK what actually we are doing here is to use blocking ops with threads to simulate load
# we are by definition: dead born to have some limitations, but this is a simple load tester after all
EXPECTED_TIME_ELAPSED_BETWEEN_REQUESTS = 50  # Average ms delay between requests per worker (randomized for realism)
AT_MOST_REQUESTS_PER_WORKER = 1500  # Max requests per worker (to prevent infinite loops)
TIME_TO_RUN_TEST_SECONDS = 30  # Total test duration in seconds (workers stop after this)
NUM_WORKERS = 50  # Number of concurrent threads (simulates user load)
RAMP_UP_DELAY = 0.5  # Seconds to stagger thread starts (avoids instant spike)

# Thread-safe storage for results (using a lock to prevent race conditions)
results_lock = Lock()
results: List[Dict] = []

def send_request_sync(request_id: int = None) -> Dict:
    """Send a single synchronous HTTP request and measure its latency."""
    # Generate random input features (simulate real data)
    data = {
        "features": (np.random.randn(DIM)).tolist()
    }
    start_time = time.perf_counter()
    try:
        # Send POST request with timeout
        response = httpx.post(LOAD_HOST, json=data, timeout=30)
        success = response.is_success
        status_code = response.status_code
        error = None
    except Exception as e:
        # Handle failures (e.g., timeouts, network errors)
        success = False
        status_code = 0
        error = str(e)
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Return structured result for analysis
    result = {
        "latency_ms": latency_ms,
        "success": success,
        "status_code": status_code,
        "error": error
    }
    return result

def worker_thread(worker_id: int) -> List[Dict]:
    """Worker thread that sends requests with randomized delays to simulate load."""
    worker_results = []
    total_time = 0.0
    
    # Run for up to the test duration or max requests, whichever comes first
    while len(worker_results) < AT_MOST_REQUESTS_PER_WORKER and total_time < TIME_TO_RUN_TEST_SECONDS:
        # Randomize wait time around the expected interval (adds realism, prevents thundering herd)
        wait_time = random.uniform(0, 2 * EXPECTED_TIME_ELAPSED_BETWEEN_REQUESTS) / 1000.0  # Convert ms to seconds
        total_time += wait_time
        time.sleep(wait_time)  # Sleep to control request rate
        
        # Send request and collect result
        result = send_request_sync()
        result["worker_id"] = worker_id
        worker_results.append(result)
    
    return worker_results

def run_multithreaded_load_test():
    """Orchestrate the multithreaded load test and collect results."""
    print(f"Starting load test with {NUM_WORKERS} workers...")
    print("-" * 50)
    
    all_results = []
    
    # Use ThreadPoolExecutor for managing concurrent threads
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        
        # Submit worker tasks with staggered starts to simulate gradual load increase
        for worker_id in range(NUM_WORKERS):
            future = executor.submit(worker_thread, worker_id)
            futures.append(future)
            time.sleep(RAMP_UP_DELAY)  # Ramp up delay between thread starts
        
        # Collect results as threads complete (non-blocking)
        for future in as_completed(futures):
            worker_results = future.result()
            all_results.extend(worker_results)
            print(f"Worker completed with {len(worker_results)} requests.")
    
    # Collect and compute detailed statistics for successful requests
    successful_latencies = [r["latency_ms"] for r in all_results if r["success"]]
    total_requests = len(all_results)
    successful_requests = sum(1 for r in all_results if r["success"])
 
    if successful_latencies:
        min_latency = min(successful_latencies)
        max_latency = max(successful_latencies)
        median_latency = statistics.median(successful_latencies)
        p95_latency = np.percentile(successful_latencies, 95)
        p99_latency = np.percentile(successful_latencies, 99)
        avg_latency = statistics.mean(successful_latencies)
    else:
        min_latency = max_latency = median_latency = p95_latency = p99_latency = avg_latency = 0
    
    success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
    throughput = total_requests / TIME_TO_RUN_TEST_SECONDS if TIME_TO_RUN_TEST_SECONDS > 0 else 0
    
    print(f"Test complete: {total_requests} requests, {successful_requests} successful ({success_rate:.1f}%), throughput: {throughput:.2f} req/s")
    print(f"Latency stats (ms) - Min: {min_latency:.2f}, Max: {max_latency:.2f}, Avg: {avg_latency:.2f}, Median (P50): {median_latency:.2f}, P95: {p95_latency:.2f}, P99: {p99_latency:.2f}")

if __name__ == "__main__":
    run_multithreaded_load_test()