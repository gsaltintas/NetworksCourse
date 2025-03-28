#!/usr/bin/env python3
import subprocess
import sys

def ping_test(source_namespace, destination_ip, count=3):
    """Test connectivity between namespaces using ping"""
    try:
        result = subprocess.run([
            "sudo", "ip", "netns", "exec", source_namespace, 
            "ping", "-c", str(count), destination_ip
        ], capture_output=True, text=True, check=True)
        print(f"✅ {source_namespace} can reach {destination_ip}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {source_namespace} cannot reach {destination_ip}")
        print(f"  Error: {e.stderr.strip()}")
        return False

def test_full_connectivity(num_aggregates, num_nodes_per_aggregate):
    """Test connectivity between all nodes in the network"""
    all_tests_passed = True
    total_nodes = num_aggregates * num_nodes_per_aggregate
    
    # Test connectivity between nodes
    for i in range(total_nodes):  # Zero-based node indices
        source = f"node{i}"
        # Test connection to a few other nodes
        target_indices = [0, (total_nodes - 1) // 2, total_nodes - 1]  # Beginning, middle, end (zero-based)
        for j in target_indices:
            if i == j:
                continue  # Skip testing connectivity to self
                
            # Calculate target IP based on node number (zero-based)
            agg_num = j // num_nodes_per_aggregate
            node_num = j % num_nodes_per_aggregate
            target_ip = f"10.{agg_num}.{node_num}.2"
            
            if not ping_test(source, target_ip):
                all_tests_passed = False
    
    if all_tests_passed:
        print("\nAll connectivity tests passed! 🎉")
    else:
        print("\nSome connectivity tests failed. Check your network configuration.")
        
    return all_tests_passed

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_connectivity.py <num_aggregates> <nodes_per_aggregate>")
        sys.exit(1)
        
    num_aggs = int(sys.argv[1])
    nodes_per_agg = int(sys.argv[2])
    test_full_connectivity(num_aggs, nodes_per_agg)
