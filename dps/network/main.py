import asyncio
import time
import argparse # For handling command-line arguments
import sys # For exiting on error

# Import the TrafficGenerator class from our module
from traffic_generator import TrafficGenerator
# Import shared config to display ports being used
import shared_config

# --- Client-Specific Configuration (Defaults/Base) ---
# These parameters configure the client's behavior.
# SERVER_HOST is now provided via command-line argument.
client_config_base = {
    'CLIENT_TIMEOUT_SECONDS': 10.0,         # Timeout for network operations (connect, send, receive ACK)
    'HEARTBEAT_INTERVAL_SECONDS': 0.1,      # Frequency of sending PING messages
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
    'LAMBDAS': {0: 0, 1: 5, 2: 50},
    # DURATION_PARAMS: Pareto distribution parameters (shape 'a', scale 'm' in seconds) for state durations
    'DURATION_PARAMS': {0: (1.3, 0.5), 1: (1.5, 0.2), 2: (1.1, 0.05)},
    # TRANSITION_PROBS: Probability of transitioning from State_Row to State_Col after duration ends
    'TRANSITION_PROBS': {
        0: {1: 0.4, 2: 0.6},                # From OFF state
        1: {0: 0.1, 1: 0.3, 2: 0.6},        # From LOW state
        2: {0: 0.1, 1: 0.7, 2: 0.2}         # From HIGH state
    }
}

# --- Main Application Async Function ---
async def run_main_app(target_server_host):
    """
    Initializes, runs, monitors, and stops the TrafficGenerator.

    Args:
        target_server_host (str): The IP address of the server to connect to.
    """
    total_runtime = 10.0 # Total duration (seconds) the main script lets the generator run
    check_interval = 0.5 # How often (seconds) the main script queries the generator status

    # --- Create final client config dictionary ---
    client_config = client_config_base.copy() # Start with base defaults
    # Add the server host provided via command line
    client_config['SERVER_HOST'] = target_server_host

    # --- Instantiate the Traffic Generator ---
    generator = TrafficGenerator(client_config)

    try:
        # --- Start Generator ---
        print("Main App: Starting traffic generator...")
        # Display connection info being used
        print(f"Main App: Target Server Host: {target_server_host}")
        print(f"Main App: Target Flow Port: {shared_config.FLOW_PORT}")
        print(f"Main App: Target Heartbeat Port: {shared_config.HEARTBEAT_PORT}")
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

            # Format RTT for display (handle initial -1 case)
            rtt_ms_str = f"{estimated_rtt * 1000:.1f} ms" if estimated_rtt >= 0 else "N/A"
            # Print the periodic status update
            print(f"--- [{time.perf_counter():.3f}] MAIN APP CHECK: "
                  f"{count} flows ongoing | Est. RTT: {rtt_ms_str} ---")

        print(f"Main App: Desired runtime ({total_runtime}s) finished.")

    except asyncio.CancelledError:
        # Handle if the main application task is cancelled (e.g., by Ctrl+C)
        print("Main App: Cancelled.")
    except Exception as e:
        # Catch any other unexpected errors in the main application logic
        print(f"Main App: An error occurred: {e}")
        # Ensure cleanup happens even if the main loop errors out
    finally:
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

        # Print the summary report
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
    # Example of adding more optional arguments:
    # parser.add_argument('--runtime', type=float, default=30.0, help='Total runtime in seconds')
    # parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    # Parse the arguments provided when running the script
    args = parser.parse_args()
    # --- End Argument Parsing ---

    print(f"Starting Main Application, targeting server: {args.server_host}")
    try:
        # Run the main asynchronous function, passing the parsed server host
        asyncio.run(run_main_app(args.server_host))
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully during execution
        print("\nMain Application stopped manually.")
    print("Main Application finished.")

