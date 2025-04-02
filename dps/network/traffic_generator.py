import asyncio
import random
import numpy as np
import time
import socket
import datetime
import pytz # Optional, for timezone output
import collections # For deque

# Import shared configuration for ports, messages, flow size
import shared_config

# --- Flow Class ---
class Flow:
    """Represents a single network flow attempt and its status."""
    _counter = 0 # Class variable to assign unique IDs
    def __init__(self, size_bytes):
        Flow._counter += 1
        self.id = Flow._counter
        self.size_bytes = size_bytes
        self.start_time = -1.0 # Flow start time (using time.perf_counter())
        self.end_time = -1.0   # Flow end time (using time.perf_counter())
        self.duration = -1.0   # Calculated duration (end_time - start_time)
        # Status tracks the lifecycle: PENDING -> RUNNING -> COMPLETED/FAILED
        self.status = "PENDING"
        self.error_info = None # Stores error message if status is FAILED

    def record_start(self):
        """Records the start time and sets status to RUNNING."""
        self.start_time = time.perf_counter()
        self.status = "RUNNING"

    def record_end(self, success=True, error=None):
        """Records the end time, duration, status, and any error."""
        self.end_time = time.perf_counter()
        # Calculate duration only if start time was recorded
        if self.start_time > 0:
            self.duration = self.end_time - self.start_time
        self.status = "COMPLETED" if success else "FAILED"
        self.error_info = error

    def is_active_realtime(self):
        """Checks if the flow is currently considered running (status is RUNNING)."""
        return self.status == "RUNNING"

    def was_active_at(self, time_t_perf_counter):
        """
        Checks if the flow was active at a specific past perf_counter time 't'.
        Useful for historical analysis after the run.
        """
        # Cannot be active if it never started
        if self.status == "PENDING" or self.start_time < 0:
            return False
        # If completed or failed, check if 't' falls within its run duration
        if self.status in ["COMPLETED", "FAILED"] and self.end_time > 0:
             # Active if t is between start (inclusive) and end (exclusive)
             return self.start_time <= time_t_perf_counter < self.end_time
        elif self.status == "RUNNING":
             # If still running, it was active from start_time up to the query time 't'
             return self.start_time <= time_t_perf_counter
        # Should not reach here in normal operation
        return False

    def __repr__(self):
        """String representation of the Flow object for logging/debugging."""
        err_str = f", error={self.error_info}" if self.error_info else ""
        return (f"Flow(id={self.id}, status={self.status}, "
                f"start={self.start_time:.3f}, end={self.end_time:.3f}, "
                f"dur={self.duration:.3f}, size={self.size_bytes}{err_str})")


# --- Traffic Generator Class ---
class TrafficGenerator:
    """
    Manages SMMPP-based traffic generation, network clients, heartbeats,
    and RTT estimation using asyncio.
    """
    def __init__(self, client_config):
        """
        Initializes the Traffic Generator using shared config and client-specific config.

        Args:
            client_config (dict): Dictionary containing CLIENT-SPECIFIC parameters like:
                SERVER_HOST (target server IP), CLIENT_TIMEOUT_SECONDS,
                LAMBDAS (dict of {state: rate}),
                DURATION_PARAMS (dict of {state: (shape_a, scale_m)}),
                TRANSITION_PROBS (dict of {state: {next_state: prob}}),
                STATE_OFF, STATE_LOW, STATE_HIGH (state identifiers),
                HEARTBEAT_INTERVAL_SECONDS, RTT_EMA_ALPHA, RTT_WINDOW_SIZE,
                seed (optional, int or None for reproducible randomness)
                (Ports, flow size, messages are taken from shared_config)
        """
        self.client_config = client_config
        # Stores all flow objects created during the run, keyed by flow.id
        self.all_flows = {}
        self._generator_task = None # asyncio task for the SMMPP generator loop
        self._heartbeat_task = None # asyncio task for the heartbeat manager loop
        # Set to keep track of active client network tasks (_run_flow_client)
        self._active_client_tasks = set()
        self._is_running = False # Flag to control background task loops

        # --- RTT Estimation State ---
        # Max number of recent RTT samples to store in the deque
        self.rtt_window_size = client_config.get('RTT_WINDOW_SIZE', 20)
        # Smoothing factor for Exponential Moving Average (0 < alpha <= 1)
        self.rtt_ema_alpha = client_config.get('RTT_EMA_ALPHA', 0.125)
        # Deque stores the last N RTT measurements
        self._rtt_measurements = collections.deque(maxlen=self.rtt_window_size)
        # Current estimated RTT, initialized to -1 (unknown)
        self._estimated_rtt = -1.0

        # --- Random Seed ---
        # Seed for random number generation (optional)
        self.seed = client_config.get('seed', None)

        # --- Unpack client-specific config for easier access ---
        self.server_host = client_config['SERVER_HOST'] # Target server IP
        self.heartbeat_interval = client_config['HEARTBEAT_INTERVAL_SECONDS']
        self.timeout = client_config['CLIENT_TIMEOUT_SECONDS'] # Network operation timeout
        self.lambdas = client_config['LAMBDAS'] # Flow arrival rates per state
        self.duration_params = client_config['DURATION_PARAMS'] # State duration params (Pareto)
        self.transition_probs = client_config['TRANSITION_PROBS'] # State transition matrix
        # State identifiers (ensure these keys exist in LAMBDAS, DURATION_PARAMS, TRANSITION_PROBS)
        self.state_off = client_config['STATE_OFF']
        self.state_low = client_config['STATE_LOW']
        self.state_high = client_config['STATE_HIGH']

        # --- Use imported shared config values ---
        self.server_port = shared_config.FLOW_PORT # Target port for flows
        self.heartbeat_port = shared_config.HEARTBEAT_PORT # Target port for heartbeats
        self.flow_size = shared_config.CONSTANT_FLOW_SIZE_BYTES # Fixed flow size
        self.ack_msg = shared_config.FLOW_ACKNOWLEDGEMENT_MSG # Expected flow ACK
        self.ack_msg_len = shared_config.FLOW_ACK_MSG_LEN # Length of flow ACK
        self.ping_msg = shared_config.PING_MSG # Heartbeat request
        self.pong_msg = shared_config.PONG_MSG # Expected heartbeat reply


    async def _run_flow_client(self, flow):
        """Internal async method handling network communication for a single flow."""
        self.all_flows[flow.id] = flow # Track the flow object
        flow.record_start() # Mark flow as running and record start time
        reader, writer = None, None # Initialize stream reader/writer
        try:
            # 1. Establish Connection to the flow port
            open_connection = asyncio.open_connection(self.server_host, self.server_port)
            reader, writer = await asyncio.wait_for(open_connection, timeout=self.timeout)

            # 2. Send Fixed-Size Data
            # Create dummy payload; replace with actual data if needed
            data_to_send = b'\0' * flow.size_bytes
            writer.write(data_to_send)
            # Wait until the OS buffer has accepted all data
            await asyncio.wait_for(writer.drain(), timeout=self.timeout)

            # 3. Signal End of Sending Data (optional but good practice)
            if writer.can_write_eof(): # Check if EOF is supported
                 writer.write_eof()

            # 4. Wait for Acknowledgement (Read Exactly Expected Length)
            ack = await asyncio.wait_for(
                reader.readexactly(self.ack_msg_len), # Use readexactly for precision
                timeout=self.timeout
            )

            # 5. Verify Acknowledgement
            if ack == self.ack_msg:
                 flow.record_end(success=True) # Mark flow as completed successfully
            else:
                 # Log and mark flow as failed if ACK is incorrect
                 print(f"[{time.perf_counter():.3f}] Flow {flow.id}: ERROR - Invalid ACK received: {ack!r}")
                 flow.record_end(success=False, error="Invalid ACK")

        # --- Exception Handling for Network Operations ---
        except asyncio.TimeoutError:
            # Mark flow as failed on timeout
            flow.record_end(success=False, error="Timeout")
        except asyncio.IncompleteReadError as e:
             # Handle case where server sends less than expected ACK bytes
             print(f"[{time.perf_counter():.3f}] Flow {flow.id}: ERROR - Incomplete ACK received (got {len(e.partial)} bytes)")
             flow.record_end(success=False, error=f"Incomplete ACK ({len(e.partial)} bytes)")
        except ConnectionRefusedError:
            # Mark flow as failed if connection is refused
            flow.record_end(success=False, error="Connection Refused")
        except OSError as e: # Catches ConnectionResetError, BrokenPipeError, etc.
            # Mark flow as failed on other OS-level network errors
            flow.record_end(success=False, error=f"OSError: {e}")
        except Exception as e: # Catch any other unexpected errors during the process
            # Mark flow as failed on generic exceptions
            flow.record_end(success=False, error=f"Exception: {e}")
        finally:
            # --- Cleanup: Ensure Connection is Closed ---
            if writer:
                try:
                     # Close the writer stream
                     writer.close()
                     # Wait for the underlying connection to close
                     await writer.wait_closed()
                except Exception:
                    pass # Ignore potential errors during cleanup


    async def _smmpp_generator(self):
        """Internal async method running the SMMPP state logic and launching flow tasks."""
        print(f"[{time.perf_counter():.3f}] SMMPP Generator task started.")
        current_state = self.state_off # Start in the OFF state
        state_start_time = time.perf_counter() # Time when the current state began

        # --- Helper Functions (defined locally for encapsulation) ---
        def choose_next_state(current, probs):
            """Selects the next state based on transition probabilities."""
            targets = list(probs[current].keys()) # Possible next states
            probabilities = list(probs[current].values()) # Corresponding probabilities
            # Choose one state based on the weights (probabilities)
            return random.choices(targets, weights=probabilities, k=1)[0]

        def get_state_duration(state, params):
            """Calculates state duration using Pareto distribution."""
            # Unpack Pareto parameters (shape 'a', scale 'm') for the state
            shape_a, scale_m = params[state]
            # Generate Pareto sample and ensure a minimum positive duration
            # Formula from numpy docs: (np.random.pareto(a) + 1) * m
            return max(1e-9, (np.random.pareto(shape_a) + 1) * scale_m)
        # --- End Helper Functions ---

        # Calculate duration and end time for the initial state
        current_duration = get_state_duration(current_state, self.duration_params)
        time_state_ends = state_start_time + current_duration

        # Main generator loop - continues as long as the generator is running
        while self._is_running:
            current_real_time = time.perf_counter()

            # --- State Transition Logic ---
            # Check if the current state's duration has elapsed
            if current_real_time >= time_state_ends:
                # Choose the next state based on transition probabilities
                current_state = choose_next_state(current_state, self.transition_probs)
                # Calculate the duration for the new state
                current_duration = get_state_duration(current_state, self.duration_params)
                # Record the time the new state started
                state_start_time = current_real_time
                # Calculate when the new state will end
                time_state_ends = state_start_time + current_duration
                # print(f"[{current_real_time:.3f}] Entering State {current_state} for {current_duration:.3f}s") # Debug

            # --- Flow Generation Logic ---
            # Get the Poisson arrival rate (lambda) for the current state
            current_lambda = self.lambdas[current_state]
            time_to_next_flow = float('inf') # Assume no flow if lambda is 0
            if current_lambda > 0:
                # Calculate time until next flow arrival (exponential distribution)
                time_to_next_flow = random.expovariate(current_lambda)

            # --- Determine Sleep Duration ---
            # Calculate time remaining in the current state
            time_until_state_end = max(0, time_state_ends - current_real_time)
            # Sleep until the next flow is due OR the state ends, whichever is sooner
            sleep_duration = min(time_to_next_flow, time_until_state_end)
            # Prevent potential busy-waiting with a small minimum sleep time
            sleep_duration = max(sleep_duration, 0.001)

            try:
                # Asynchronously wait for the calculated duration
                await asyncio.sleep(sleep_duration)
            except asyncio.CancelledError:
                # Handle cancellation request cleanly
                print(f"[{time.perf_counter():.3f}] SMMPP Generator task cancelled.")
                break # Exit the loop

            # --- Trigger Flow if Applicable ---
            # Check if the sleep completed because it was time for a flow
            # (use tolerance for float comparison) and ensure generator wasn't stopped during sleep
            if self._is_running and current_lambda > 0 and abs(sleep_duration - time_to_next_flow) < 1e-9 :
                 # Create a new Flow object
                 flow = Flow(size_bytes=self.flow_size)
                 # Launch the client task for this flow non-blockingly
                 task = asyncio.create_task(self._run_flow_client(flow))
                 # Add the task to the tracking set
                 self._active_client_tasks.add(task)
                 # Add a callback to automatically remove the task from the set when done
                 task.add_done_callback(self._active_client_tasks.discard)
                 # print(f"[{time.perf_counter():.3f}] Triggered Flow {flow.id}") # Debug

        print(f"[{time.perf_counter():.3f}] SMMPP Generator task finished.")


    async def _heartbeat_manager(self):
        """Internal async method managing the heartbeat connection and RTT estimation."""
        print(f"[{time.perf_counter():.3f}] Heartbeat manager started.")
        reader, writer = None, None # Initialize stream reader/writer
        # Main heartbeat loop - continues as long as the generator is running
        while self._is_running:
            try:
                # --- Ensure Connection ---
                # Check if connection is down or needs establishing
                if writer is None or writer.is_closing():
                    # print(f"[{time.perf_counter():.3f}] Heartbeat: Attempting connection...") # Debug
                    try:
                        # Connect to the dedicated Heartbeat Port
                        open_conn = asyncio.open_connection(self.server_host, self.heartbeat_port)
                        # Use a shorter timeout for the connection attempt
                        reader, writer = await asyncio.wait_for(open_conn, timeout=self.timeout/2)
                        # print(f"[{time.perf_counter():.3f}] Heartbeat: Connection established.") # Debug
                    except (OSError, asyncio.TimeoutError) as e:
                        # print(f"[{time.perf_counter():.3f}] Heartbeat: Connection failed: {e}. Retrying...") # Debug
                        reader, writer = None, None # Reset connection objects
                        # Wait before retrying connection in the next loop iteration
                        await asyncio.sleep(self.heartbeat_interval)
                        continue # Go to start of loop to retry connection

                # --- Send PING and Measure RTT ---
                t_send = time.perf_counter() # Record time before sending
                writer.write(self.ping_msg) # Send PING message
                # Wait for write buffer to drain (use short timeout)
                await asyncio.wait_for(writer.drain(), timeout=self.timeout/2)

                # Wait for PONG reply (use full timeout)
                # Read until newline, as PONG includes \n
                response = await asyncio.wait_for(reader.readline(), timeout=self.timeout)

                if response == self.pong_msg: # Check if reply is correct
                    t_recv = time.perf_counter() # Record time upon receiving reply
                    rtt_sample = t_recv - t_send # Calculate sample RTT
                    # Basic validation: RTT should be positive and less than timeout
                    if 0 <= rtt_sample < self.timeout:
                        self._rtt_measurements.append(rtt_sample) # Store the sample
                        # Update EMA (Exponential Moving Average)
                        if self._estimated_rtt < 0: # Initialize with the first valid sample
                            self._estimated_rtt = rtt_sample
                        else:
                            # Apply EMA formula
                            self._estimated_rtt = (self.rtt_ema_alpha * rtt_sample) + \
                                                  ((1 - self.rtt_ema_alpha) * self._estimated_rtt)
                    # else: print(f"[{time.perf_counter():.3f}] Heartbeat: Ignoring invalid RTT sample: {rtt_sample:.4f}s") # Debug
                else:
                    # Handle unexpected data received on heartbeat connection
                    print(f"[{time.perf_counter():.3f}] Heartbeat: Received unexpected data: {response!r}")
                    # Close and force reconnect might be safest here
                    if writer: writer.close(); writer = None; reader = None
                    continue # Skip sleep, try reconnecting immediately

                # --- Wait for next heartbeat interval ---
                await asyncio.sleep(self.heartbeat_interval)

            # --- Exception Handling for Heartbeat Loop ---
            except asyncio.TimeoutError:
                print(f"[{time.perf_counter():.3f}] Heartbeat: Timeout waiting for PONG.")
                # Assume connection is dead, force reconnect in the next iteration
                if writer: writer.close(); writer = None; reader = None
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                 # Handle common connection errors
                 print(f"[{time.perf_counter():.3f}] Heartbeat: Connection error: {e}. Reconnecting...")
                 if writer: writer.close(); writer = None; reader = None
                 # Wait a short time before attempting reconnect in the loop
                 await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                 # Handle cancellation request cleanly
                 print(f"[{time.perf_counter():.3f}] Heartbeat manager cancelled.")
                 break # Exit the loop
            except Exception as e:
                 # Catch any other unexpected errors
                 print(f"[{time.perf_counter():.3f}] Heartbeat: Unexpected error: {e}. Retrying...")
                 if writer: writer.close(); writer = None; reader = None
                 # Wait before retrying the loop
                 await asyncio.sleep(self.heartbeat_interval)

        # --- Cleanup after loop exit ---
        if writer and not writer.is_closing():
             print(f"[{time.perf_counter():.3f}] Heartbeat manager closing final connection.")
             writer.close()
             try:
                 # Wait for the connection to close fully
                 await writer.wait_closed()
             except Exception:
                 pass # Ignore errors during final close
        print(f"[{time.perf_counter():.3f}] Heartbeat manager finished.")


    async def start(self):
        """Starts the traffic generator and heartbeat background tasks. Applies seed if set."""
        # Prevent starting if already running
        if self._is_running:
            print("Generator is already running.")
            return

        # --- Apply Seed ---
        # Seed random number generators for reproducibility if a seed is provided
        if self.seed is not None:
            print(f"Applying random seed: {self.seed}")
            random.seed(self.seed)
            np.random.seed(self.seed)
        else:
            print("Running with default random seed.")
        # --------------------

        print("Starting Traffic Generator and Heartbeat Manager...")
        self._is_running = True # Set running flag
        # Reset state variables
        self.all_flows.clear()
        self._active_client_tasks.clear()
        self._rtt_measurements.clear()
        self._estimated_rtt = -1.0
        # Start background tasks using asyncio.create_task
        self._generator_task = asyncio.create_task(self._smmpp_generator())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_manager())
        print("Traffic Generator and Heartbeat Manager started.")

    async def stop(self, wait_for_flows_timeout=5.0):
        """Stops the background tasks and waits briefly for active flows."""
        # Prevent stopping if not running
        if not self._is_running:
            print("Generator is not running.")
            return
        print("Stopping Traffic Generator and Heartbeat Manager...")
        self._is_running = False # Signal background loops to stop

        # --- Stop Heartbeat Task ---
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel() # Request cancellation
            try:
                # Wait briefly for the task to finish its cleanup
                await asyncio.wait_for(self._heartbeat_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass # Ignore if it times out or was already cancelled
        print("Heartbeat task stopped.")

        # --- Stop Generator Task ---
        if self._generator_task and not self._generator_task.done():
            self._generator_task.cancel() # Request cancellation
            try:
                # Wait briefly for the task to finish
                await asyncio.wait_for(self._generator_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        print("Generator task stopped.")

        # --- Wait for Active Flow Client Tasks (Optional) ---
        if self._active_client_tasks:
            print(f"Waiting up to {wait_for_flows_timeout}s for {len(self._active_client_tasks)} active flows...")
            try:
                # Use the client timeout from config, ensure it's positive
                actual_wait_timeout = max(0.1, self.client_config.get('CLIENT_TIMEOUT_SECONDS', 5.0))
                # Wait for all tasks in the set to complete or the timeout occurs
                done, pending = await asyncio.wait(
                    # Create a list copy of the set to avoid issues if callbacks modify it during wait
                    list(self._active_client_tasks),
                    timeout=actual_wait_timeout
                )
                if pending:
                    # Log if some tasks didn't finish in time
                    print(f"Warning: {len(pending)} flow tasks did not complete within timeout.")
                    # Optionally, explicitly cancel pending tasks if required:
                    # for task in pending: task.cancel()
            except asyncio.TimeoutError:
                 # This condition is less likely with asyncio.wait timeout parameter
                 print(f"Warning: Timeout exceeded while waiting for flow tasks.")
            except Exception as e:
                # Catch other potential errors during the wait process
                print(f"Error during waiting for active flows: {e}")

        # Reset task handles
        self._generator_task = None
        self._heartbeat_task = None
        print("Traffic Generator stopped.")


    def get_ongoing_flow_count(self):
        """Returns the number of flows currently in the RUNNING state."""
        count = 0
        try:
            # Iterate over a list copy of values for safety against dict changes
            for flow in list(self.all_flows.values()):
                if flow.is_active_realtime():
                    count += 1
        except RuntimeError: # Handle rare dict size changed during iteration
             # Retry the count if a runtime error occurred
             count = sum(1 for f in list(self.all_flows.values()) if f.is_active_realtime())
        return count

    def get_estimated_rtt(self):
        """Returns the current estimated RTT in seconds, or -1 if unknown."""
        return self._estimated_rtt

    def get_last_rtt_sample(self):
        """Returns the most recent RTT sample in seconds, or None if no samples yet."""
        try:
            # Access deque safely using index -1 for the last element
            if self._rtt_measurements:
                return self._rtt_measurements[-1]
        except IndexError:
            pass # Deque might be empty
        return None

    def get_flow_summary(self):
        """Returns a dictionary summarizing the status of all initiated flows."""
        # Create a list copy for safe iteration
        flows_list = list(self.all_flows.values())
        summary = {
            "total": len(flows_list), # Total flows created
            "completed": sum(1 for f in flows_list if f.status == "COMPLETED"), # Count successful
            "failed": sum(1 for f in flows_list if f.status == "FAILED"), # Count failed
            "running": sum(1 for f in flows_list if f.status == "RUNNING"), # Count still running
            "pending": sum(1 for f in flows_list if f.status == "PENDING"), # Count not yet started (should be 0 after run)
        }
        return summary

