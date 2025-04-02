import asyncio
import socket
import time
import argparse # Import argparse

# Import shared configuration for ports, messages, flow size
import shared_config

# --- Server-Specific Configuration ---
# HOST is now determined by command-line argument, defaulting to '0.0.0.0'

# --- Handler for Data Flows ---
async def handle_flow_client(reader, writer):
    """Handles connection for a data flow client."""
    addr = writer.get_extra_info('peername')
    total_received = 0
    try:
        while total_received < shared_config.CONSTANT_FLOW_SIZE_BYTES:
            bytes_to_read = shared_config.CONSTANT_FLOW_SIZE_BYTES - total_received
            chunk_size = min(bytes_to_read, 4096)
            data = await reader.read(chunk_size)
            if not data:
                print(f"WARN: Flow connection from {addr} closed prematurely after {total_received} bytes.")
                break
            total_received += len(data)

        if total_received == shared_config.CONSTANT_FLOW_SIZE_BYTES:
            writer.write(shared_config.FLOW_ACKNOWLEDGEMENT_MSG)
            await writer.drain()

    except asyncio.CancelledError: raise
    except ConnectionResetError: print(f"WARN: Flow connection reset by {addr} after {total_received} bytes.")
    except Exception as e: print(f"ERROR handling flow client {addr}: {e}")
    finally:
        try: writer.close(); await writer.wait_closed()
        except Exception: pass

# --- Handler for Heartbeats ---
async def handle_heartbeat_client(reader, writer):
    """Handles connection for a heartbeat client."""
    addr = writer.get_extra_info('peername')
    print(f"Heartbeat connection from {addr}")
    try:
        while True:
            message = await reader.readline()
            if not message:
                 print(f"Heartbeat client {addr} disconnected.")
                 break
            elif message == shared_config.PING_MSG:
                 writer.write(shared_config.PONG_MSG)
                 await writer.drain()
            else:
                 print(f"WARN: Received unexpected heartbeat data from {addr}: {message!r}")

    except asyncio.CancelledError: print(f"Heartbeat handler for {addr} cancelled."); raise
    except ConnectionResetError: print(f"WARN: Heartbeat connection reset by {addr}.")
    except Exception as e: print(f"ERROR handling heartbeat client {addr}: {e}")
    finally:
        print(f"Closing heartbeat connection to {addr}")
        try: writer.close(); await writer.wait_closed()
        except Exception: pass


# --- Main Server Function ---
async def main(listen_host):
    """Starts and runs both flow and heartbeat servers."""
    # Start server for data flows
    try:
        flow_server = await asyncio.start_server(
            handle_flow_client, listen_host, shared_config.FLOW_PORT)
        flow_addrs = ', '.join(str(sock.getsockname()) for sock in flow_server.sockets)
        print(f'Flow server listening on {flow_addrs} | Flow Size: {shared_config.CONSTANT_FLOW_SIZE_BYTES} bytes')
    except OSError as e:
        print(f"\nERROR starting flow server on {listen_host}:{shared_config.FLOW_PORT}: {e}")
        print("Check if the IP address is valid for this machine and the port is free.")
        return

    # Start server for heartbeats
    try:
        heartbeat_server = await asyncio.start_server(
            handle_heartbeat_client, listen_host, shared_config.HEARTBEAT_PORT)
        hb_addrs = ', '.join(str(sock.getsockname()) for sock in heartbeat_server.sockets)
        print(f'Heartbeat server listening on {hb_addrs}')
    except OSError as e:
        print(f"\nERROR starting heartbeat server on {listen_host}:{shared_config.HEARTBEAT_PORT}: {e}")
        print("Check if the IP address is valid for this machine and the port is free.")
        flow_server.close()
        await flow_server.wait_closed()
        return

    # Run both servers concurrently
    async with flow_server, heartbeat_server:
        await asyncio.gather(
            flow_server.serve_forever(),
            heartbeat_server.serve_forever()
        )

if __name__ == "__main__":
    # --- Argument Parsing --- * NEW *
    parser = argparse.ArgumentParser(description="SMMPP Echo and Heartbeat Server")
    parser.add_argument(
        '--host',
        default='0.0.0.0', # Default to listen on all interfaces
        help='Host IP address to bind the server sockets to (default: 0.0.0.0)'
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    print(f"Starting SMMPP Flow and Heartbeat Servers (Binding to {args.host})...")
    try:
        # Pass the parsed host to the main function
        asyncio.run(main(args.host))
    except KeyboardInterrupt:
        print("\nServers stopped manually.")
    # OSError related to binding is now handled within main()

