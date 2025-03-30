#!/usr/bin/env python3
import subprocess
import ipaddress
import time
import sys

def run_cmd(cmd, check=True):
    """Run a command and return the result"""
    return subprocess.run(cmd, check=check, capture_output=True, text=True)

def create_flat_network(num_aggregates, num_nodes_per_aggregate, agg_bandwidth=None, node_bandwidth=None):
    """
    Creates a 3-layer network tree using bridges and virtual interfaces in the default namespace.
    
    Args:
        num_aggregates: Number of aggregate bridges to create
        num_nodes_per_aggregate: Number of nodes per aggregate bridge
        agg_bandwidth: Bandwidth limit for core-aggregate links (e.g., "100mbit")
        node_bandwidth: Bandwidth limit for aggregate-node links (e.g., "10mbit")
    """
    print("Creating flat network topology without namespaces...")

    # Create core bridge
    core_bridge = "brcore"
    run_cmd(["sudo", "ip", "link", "add", "name", core_bridge, "type", "bridge"])
    run_cmd(["sudo", "ip", "addr", "add", "10.0.254.1/24", "dev", core_bridge])
    run_cmd(["sudo", "ip", "link", "set", "dev", core_bridge, "up"])

    # Enable IP forwarding globally
    run_cmd(["sudo", "sysctl", "-w", "net.ipv4.ip_forward=1"])

    # Create aggregate bridges and connect to core
    for agg_num in range(num_aggregates):
        # Create aggregate bridge
        agg_bridge = f"bragg{agg_num}"
        run_cmd(["sudo", "ip", "link", "add", "name", agg_bridge, "type", "bridge"])
        run_cmd(["sudo", "ip", "addr", "add", f"10.{agg_num}.254.1/24", "dev", agg_bridge])
        run_cmd(["sudo", "ip", "link", "set", "dev", agg_bridge, "up"])

        # Create veth pair to connect core to aggregate
        veth_core_agg = f"veth-core-agg{agg_num}"
        veth_agg_core = f"veth-agg{agg_num}-core"
        run_cmd(["sudo", "ip", "link", "add", veth_core_agg, "type", "veth", "peer", "name", veth_agg_core])

        # Configure the veth interfaces
        run_cmd(["sudo", "ip", "addr", "add", f"10.0.{agg_num}.1/24", "dev", veth_core_agg])
        run_cmd(["sudo", "ip", "addr", "add", f"10.0.{agg_num}.2/24", "dev", veth_agg_core])

        # Connect veth interfaces to their respective bridges
        run_cmd(["sudo", "ip", "link", "set", "dev", veth_core_agg, "master", core_bridge, "up"])
        run_cmd(["sudo", "ip", "link", "set", "dev", veth_agg_core, "master", agg_bridge, "up"])

        # Apply bandwidth limits if specified
        if agg_bandwidth:
            # Bandwidth limit on core side
            run_cmd([
                "sudo", "tc", "qdisc", "add", "dev", veth_core_agg, 
                "root", "tbf", "rate", agg_bandwidth, "burst", "32kbit", "latency", "400ms"
            ])
            # Bandwidth limit on aggregate side
            run_cmd([
                "sudo", "tc", "qdisc", "add", "dev", veth_agg_core,
                "root", "tbf", "rate", agg_bandwidth, "burst", "32kbit", "latency", "400ms"
            ])

        # Add specific routes for network (not default routes)
        # Route from core to this aggregate's network
        run_cmd(["sudo", "ip", "route", "add", f"10.{agg_num}.0.0/16", "via", f"10.0.{agg_num}.2"])

        # Create nodes for this aggregate
        for node_num in range(num_nodes_per_aggregate):
            # Create veth pair for node
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            veth_agg_node = f"veth-agg{agg_num}-node{node_num}"
            veth_node = f"veth-node{global_node_num}"  # Use global node number for uniqueness

            run_cmd(["sudo", "ip", "link", "add", veth_agg_node, "type", "veth", "peer", "name", veth_node])

            # Configure the node interface
            node_ip = f"10.{agg_num}.{node_num}.2"
            agg_node_ip = f"10.{agg_num}.{node_num}.1"

            run_cmd(["sudo", "ip", "addr", "add", agg_node_ip + "/24", "dev", veth_agg_node])
            run_cmd(["sudo", "ip", "addr", "add", node_ip + "/24", "dev", veth_node])

            # Connect aggregate side to the bridge
            run_cmd(["sudo", "ip", "link", "set", "dev", veth_agg_node, "master", agg_bridge, "up"])
            run_cmd(["sudo", "ip", "link", "set", "dev", veth_node, "up"])

            # Apply bandwidth limits if specified
            if node_bandwidth:
                # Bandwidth limit on aggregate side
                run_cmd([
                    "sudo", "tc", "qdisc", "add", "dev", veth_agg_node,
                    "root", "tbf", "rate", node_bandwidth, "burst", "32kbit", "latency", "400ms"
                ])
                # Bandwidth limit on node side
                run_cmd([
                    "sudo", "tc", "qdisc", "add", "dev", veth_node,
                    "root", "tbf", "rate", node_bandwidth, "burst", "32kbit", "latency", "400ms"
                ])

            # Create a policy routing table for this node interface
            table_id = 100 + global_node_num

            # Add policy routing for the node's traffic
            # 1. Add a separate routing table for this interface
            # 2. Add a rule to use this table for packets from this node's IP
            # 3. Add a default route in this table pointing to the aggregate router

            # First, add to /etc/iproute2/rt_tables if not already there
            rt_tables_entry = f"{table_id} node{global_node_num}"
            run_cmd(["sudo", "bash", "-c", f"grep -q '{rt_tables_entry}' /etc/iproute2/rt_tables || echo '{rt_tables_entry}' | sudo tee -a /etc/iproute2/rt_tables"])

            # Add route to the specific table
            run_cmd(["sudo", "ip", "route", "add", "default", "via", agg_node_ip, "dev", veth_node, "table", f"node{global_node_num}"])

            # Add rule to use this table for traffic from this node's IP
            run_cmd(["sudo", "ip", "rule", "add", "from", node_ip, "table", f"node{global_node_num}"])

            print(f"Created node{global_node_num} interface: {veth_node} with IP {node_ip}")

    print("\nNetwork setup complete.")
    print("To access a specific node, use its interface directly.")
    print(f"Core bridge: {core_bridge}")
    print(f"Aggregate bridges: " + ", ".join([f"bragg{i}" for i in range(num_aggregates)]))

    # Display network information
    print("\nNode information:")
    for agg_num in range(num_aggregates):
        for node_num in range(num_nodes_per_aggregate):
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            node_ip = f"10.{agg_num}.{node_num}.2"
            node_interface = f"veth-node{global_node_num}"
            print(f"  node{global_node_num}: Interface={node_interface}, IP={node_ip}")


def remove_flat_network(num_aggregates, num_nodes_per_aggregate):
    """Remove the network topology created without namespaces"""
    print("Removing flat network topology...")

    # First, remove the policy routing rules and tables
    for agg_num in range(num_aggregates):
        for node_num in range(num_nodes_per_aggregate):
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            node_ip = f"10.{agg_num}.{node_num}.2"

            # Remove the rule
            try:
                run_cmd(["sudo", "ip", "rule", "del", "from", node_ip, "table", f"node{global_node_num}"], check=False)
            except:
                pass

            # We can't easily remove entries from rt_tables, but that's not critical

    # Remove all node veth pairs
    for agg_num in range(num_aggregates):
        for node_num in range(num_nodes_per_aggregate):
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            veth_node = f"veth-node{global_node_num}"
            try:
                run_cmd(["sudo", "ip", "link", "del", veth_node], check=False)
            except:
                pass

    # Remove all aggregate-core veth pairs
    for agg_num in range(num_aggregates):
        try:
            run_cmd(["sudo", "ip", "link", "del", f"veth-core-agg{agg_num}"], check=False)
        except:
            pass

    # Remove all aggregate bridges
    for agg_num in range(num_aggregates):
        try:
            run_cmd(["sudo", "ip", "link", "del", f"bragg{agg_num}"], check=False)
        except:
            pass

    # Remove core bridge
    try:
        run_cmd(["sudo", "ip", "link", "del", "brcore"], check=False)
    except:
        pass

    print("Network topology removed.")


def get_node_ip(node_num, num_nodes_per_aggregate):
    """
    Get the IP address of a node based on its global node number.
    
    Args:
        node_num: Global node number (e.g., node0, node1, etc.)
        num_nodes_per_aggregate: Number of nodes per aggregate router
    
    Returns:
        A string with the IP address of the node (e.g., "10.0.1.2")
    """
    agg_num = node_num // num_nodes_per_aggregate
    local_node_num = node_num % num_nodes_per_aggregate
    return f"10.{agg_num}.{local_node_num}.2"


def test_connectivity():
    """Test connectivity in the network"""
    print("\nTesting network connectivity...")

    # Test from host to a node
    try:
        result = run_cmd(["ping", "-c", "1", "-W", "2", "10.0.0.2"])
        print("Ping to first node: Success" if result.returncode == 0 else "Ping to first node: Failed")
    except Exception as e:
        print(f"Ping test to first node failed: {e}")

    # Test from host to a node in another subnet
    try:
        result = run_cmd(["ping", "-c", "1", "-W", "2", "10.1.0.2"])
        print("Ping to node in another subnet: Success" if result.returncode == 0 else "Ping to node in another subnet: Failed")
    except Exception as e:
        print(f"Ping test to other subnet failed: {e}")

    # Test internet connectivity (google's DNS)
    try:
        result = run_cmd(["ping", "-c", "1", "-W", "2", "8.8.8.8"])
        print("Internet connectivity: Success" if result.returncode == 0 else "Internet connectivity: Failed")
    except Exception as e:
        print(f"Internet connectivity test failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a network topology without using namespaces")
    parser.add_argument("--aggs", type=int, default=2, help="Number of aggregate bridges")
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes per aggregate")
    parser.add_argument("--cleanup", action="store_true", help="Remove the network topology")
    parser.add_argument("--agg-bandwidth", type=str, default=None, help="Bandwidth for aggregate links (e.g., '100mbit')")
    parser.add_argument("--node-bandwidth", type=str, default=None, help="Bandwidth for node links (e.g., '10mbit')")

    args = parser.parse_args()

    if args.cleanup:
        remove_flat_network(args.aggs, args.nodes)
    else:
        create_flat_network(
            args.aggs, 
            args.nodes, 
            args.agg_bandwidth,
            args.node_bandwidth
        )
        test_connectivity()
