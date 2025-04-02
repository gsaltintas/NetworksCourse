# shared_config.py
# Configuration parameters shared between the SMMPP client and server.

# --- Network Ports ---
# Ports clients connect to AND server listens on.
FLOW_PORT = 65432           # Port for main data flows
HEARTBEAT_PORT = 65434      # Port for heartbeats

# --- Flow Parameters ---
CONSTANT_FLOW_SIZE_BYTES = 500_000 # Expected flow size

# --- Protocol Messages ---
FLOW_ACKNOWLEDGEMENT_MSG = b"OK"    # ACK message for successful flow transfer
PING_MSG = b"PING\n"                # Heartbeat request message
PONG_MSG = b"PONG\n"                # Heartbeat reply message

# --- Calculated Values (derived from messages) ---
FLOW_ACK_MSG_LEN = len(FLOW_ACKNOWLEDGEMENT_MSG)
PONG_MSG_LEN = len(PONG_MSG) # Although server reads line, client might use fixed read

# --- Add any other parameters truly shared by both client and server logic ---
# Note: SERVER_HOST is now a command-line argument for both client and server.

