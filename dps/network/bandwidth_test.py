#!/usr/bin/env python3
import subprocess
import argparse
import time
import sys

def install_iperf():
    """Check if iperf3 is installed, and install it if not"""
    try:
        subprocess.run(["which", "iperf3"], check=True, stdout=subprocess.DEVNULL)
        print("iperf3 is already installed.")
    except subprocess.CalledProcessError:
        print("Installing iperf3...")
        subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "iperf3"], check=True)

def run_iperf_server(namespace):
    """Start an iperf3 server in the given namespace"""
    print(f"Starting iperf3 server in {namespace}...")
    server_proc = subprocess.Popen(
        ["sudo", "ip", "netns", "exec", namespace, "iperf3", "-s"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Give server time to start
    time.sleep(2)
    return server_proc

def run_iperf_client(namespace, server_ip, duration=10):
    """Run iperf3 client in given namespace targeting the server_ip"""
    print(f"Running iperf3 client in {namespace} to {server_ip}...")
    try:
        result = subprocess.run(
            ["sudo", "ip", "netns", "exec", namespace, "iperf3", "-c", server_ip, "-t", str(duration), "-f", "m"],
            capture_output=True,
            text=True,
            check=True
        )
        # Extract and return bandwidth from result
        output = result.stdout
        print(f"\n--- Bandwidth Test Results ---\n{output}\n")
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error running iperf3 client: {e}")
        print(e.stderr)
        return None

def test_link_bandwidth(source_namespace, target_namespace, target_ip, duration=10):
    """Test bandwidth between two namespaces"""
    # Start server in target namespace
    server_proc = run_iperf_server(target_namespace)
    
    try:
        # Run client in source namespace
        result = run_iperf_client(source_namespace, target_ip, duration)
        return result
    finally:
        # Clean up server
        server_proc.terminate()
        server_proc.wait()

def test_tree_bandwidths(num_aggregates, num_nodes_per_aggregate):
    """Test bandwidths across the network tree"""
    # Ensure iperf3 is installed
    install_iperf()
    
    # Test bandwidth between core and one aggregate router
    core_to_agg_result = test_link_bandwidth(
        "core_router", 
        "agg_router0", 
        "10.0.0.2", 
        duration=5
    )
    
    # Test bandwidth between aggregate and one node
    agg_to_node_result = test_link_bandwidth(
        "agg_router0", 
        f"node0", 
        "10.0.0.2", 
        duration=5
    )
    
    # Test bandwidth between two nodes (across aggregates if there are multiple)
    if num_aggregates > 1:
        # Test between node0 (on agg0) and a node on another aggregate
        node_to_node_result = test_link_bandwidth(
            "node0", 
            f"node{num_nodes_per_aggregate}", 
            f"10.1.0.2", 
            duration=5
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test bandwidth in the network tree")
    parser.add_argument("--aggs", type=int, default=3, help="Number of aggregate routers")
    parser.add_argument("--nodes", type=int, default=4, help="Number of nodes per aggregate")
    args = parser.parse_args()
    
    test_tree_bandwidths(args.aggs, args.nodes)
