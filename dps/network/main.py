import asyncio
import time
import argparse # For handling command-line arguments
import csv  # Import CSV module for file output

# Import the TrafficGenerator class from our module
from traffic_generator import TrafficGenerator
# Import shared config to display ports being used
import shared_config

# --- Client-Specific Configuration (Defaults/Base) ---
# These parameters configure the client's behavior.
# SERVER_HOST is now provided via command-line argument.
client_config_base = {
    'CLIENT_TIMEOUT_SECONDS': 10.0,         # Timeout for network operations (connect, send, receive ACK)
    'HEARTBEAT_INTERVAL_SECONDS': 0.01,      # Frequency of sending PING messages
    'RTT_EMA_ALPHA': 0.5,                 # Smoothing factor for RTT estimate (lower = smoother, slower reaction)
    'RTT_WINDOW_SIZE': 5,                  # Number of recent RTT samples to store in memory
    'seed': 42,                             # Random seed for SMMPP reproducibility (set to None for non-reproducible runs)
    # --- SMMPP State Definitions ---
    # These identifiers must match the keys used in LAMBDAS, DURATION_PARAMS, TRANSITION_PROBS
    'STATE_OFF': 0,
    'STATE_LOW': 1,
    'STATE_HIGH': 2,
    # --- SMMPP Parameters ---
    # LAMBDAS: Average number of flow start attempts per second for each state
    'LAMBDAS': {0: 10, 1: 20, 2: 50},
    # DURATION_PARAMS: Pareto distribution parameters (shape 'a', scale 'm' in seconds) for state durations
    'DURATION_PARAMS': {0: (1.3, 0.2), 1: (1.5, 0.6), 2: (1.1, 0.3)},
    # TRANSITION_PROBS: Probability of transitioning from State_Row to State_Col after duration ends
    'TRANSITION_PROBS': {
        0: {1: 0.4, 2: 0.6},                # From OFF state
        1: {0: 0.1, 1: 0.3, 2: 0.6},        # From LOW state
        2: {0: 0.1, 1: 0.6, 2: 0.3}         # From HIGH state
    }
}

# --- Main Application Async Function ---
async def run_main_app(args):
    """
    Initializes, runs, monitors, and stops the TrafficGenerator.

    Args:
        target_server_host (str): The IP address of the server to connect to.
    """
    total_runtime = 20.0 # Total duration (seconds) the main script lets the generator run
    check_interval = 0.1 # How often (seconds) the main script queries the generator status

    target_server_host = args.server_host

    # --- Create final client config dictionary ---
    client_config = client_config_base.copy() # Start with base defaults
    # Add the server host provided via command line
    client_config['SERVER_HOST'] = target_server_host

    client_config['seed'] = args.seed

    # --- Instantiate the Traffic Generator ---
    generator = TrafficGenerator(client_config)
    
    # Create CSV file for logging data
    csv_filename = f"flow_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Write header row
    csv_writer.writerow(['Elapsed Time (s)', 'Ongoing Flows', 'RTT (ms)'])

    try:
        # --- Start Generator ---
        print("Main App: Starting traffic generator...")
        print(f"Main App: Target Server Host: {target_server_host}")
        print(f"Main App: Target Flow Port: {shared_config.FLOW_PORT}")
        print(f"Main App: Target Heartbeat Port: {shared_config.HEARTBEAT_PORT}")
        print(f"Main App: Logging data to {csv_filename}")
        if client_config.get('seed') is not None:
             print(f"Main App: Using random seed: {client_config['seed']}")

        # Start the generator's background tasks (SMMPP and Heartbeat)
        await generator.start()
        print("Main App: Generator started.")

        # --- Monitoring Loop ---
        start_time = time.perf_counter()
        # Run the monitoring loop for the specified total_runtime
        while time.perf_counter() < start_time + total_runtime:
            # Wait for the check interval before querying status
            await asyncio.sleep(check_interval)

            # Query the generator for current status
            count = generator.get_ongoing_flow_count()
            estimated_rtt = generator.get_estimated_rtt()
            elapsed_time = time.perf_counter() - start_time

            # Format RTT for display (handle initial -1 case)
            rtt_ms_str = f"{estimated_rtt * 1000:.1f} ms" if estimated_rtt >= 0 else "N/A"
            rtt_ms = estimated_rtt * 1000 if estimated_rtt >= 0 else None
            
            # Write data to CSV file
            csv_writer.writerow([f"{elapsed_time:.3f}", count, rtt_ms])
            
            # Print the periodic status update
            # print(f"--- [{elapsed_time:.3f}] MAIN APP CHECK: "
            #       f"{count} flows ongoing | Est. RTT: {rtt_ms_str} ---")

        print(f"Main App: Desired runtime ({total_runtime}s) finished.")

    except asyncio.CancelledError:
        # Handle if the main application task is cancelled (e.g., by Ctrl+C)
        print("Main App: Cancelled.")
    except Exception as e:
        # Catch any other unexpected errors in the main application logic
        print(f"Main App: An error occurred: {e}")
        # Ensure cleanup happens even if the main loop errors out
    finally:
        # Close the CSV file
        csv_file.close()
        print(f"Main App: Data logged to {csv_filename}")
        
        # --- Stop Generator and Cleanup ---
        print("Main App: Stopping traffic generator...")
        # Determine timeout for waiting for active flows during stop
        wait_timeout = client_config.get('CLIENT_TIMEOUT_SECONDS', 5.0) / 2
        # Call the generator's stop method to halt background tasks gracefully
        await generator.stop(wait_for_flows_timeout=wait_timeout)
        print("Main App: Generator stopped.")

        # --- Final Summary ---
        # Get final statistics from the generator instance
        summary = generator.get_flow_summary()
        final_rtt = generator.get_estimated_rtt()
        rtt_ms_str = f"{final_rtt * 1000:.1f} ms" if final_rtt >= 0 else "N/A"

        print("\n--- Generator Final Summary ---")
        print(f"Total flows initiated: {summary['total']}")
        print(f"Completed successfully: {summary['completed']}")
        print(f"Failed (Timeout/Error): {summary['failed']}")
        # These should ideally be 0 after stop() finishes cleanly
        if summary['running'] > 0: print(f"Still running (unexpected): {summary['running']}")
        if summary['pending'] > 0: print(f"Still pending (unexpected): {summary['pending']}")
        print(f"Final Estimated RTT: {rtt_ms_str}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run SMMPP Traffic Generator Client")
    # Define a required positional argument for the server's IP address
    parser.add_argument(
        'server_host', # Argument name
        help='IP address of the target SMMPP server'
    )
    # parser.add_argument('--runtime', type=float, default=30.0, help='Total runtime in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Parse the arguments provided when running the script
    args = parser.parse_args()
    # --- End Argument Parsing ---

    print(f"Starting Main Application, targeting server: {args.server_host}")
    try:
        # Run the main asynchronous function, passing the parsed server host
        asyncio.run(run_main_app(args))
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully during execution
        print("\nMain Application stopped manually.")
    print("Main Application finished.")

