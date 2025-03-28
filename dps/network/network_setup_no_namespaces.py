#!/usr/bin/env python3
import subprocess
import ipaddress
import time
import sys

def run_cmd(cmd, check=True):
    """Run a command and return the result"""
    return subprocess.run(cmd, check=check, capture_output=True, text=True)

# New helper to enable IP forwarding on specific interface
def enable_interface_forwarding(interface):
    run_cmd(["sudo", "sysctl", "-w", f"net.ipv4.conf.{interface}.forwarding=1"])
    run_cmd(["sudo", "sysctl", "-w", f"net.ipv4.conf.{interface}.rp_filter=0"])
    run_cmd(["sudo", "sysctl", "-w", f"net.ipv4.conf.{interface}.proxy_arp=1"])

# def add_static_arp_entry(interface, ip_addr, mac_addr):
#     """Add a static ARP entry"""
#     run_cmd(["sudo", "ip", "neigh", "replace", ip_addr, "lladdr", mac_addr, "dev", interface])


def set_bandwidth(interface, rate=None, burst="32kbit", limit=None):
    """
    Set bandwidth parameters on an interface using tc
    Args:
        interface: The network interface to configure
        rate: Bandwidth rate (e.g., "100mbit")
        burst: Burst size (e.g., "32kbit"), mandatory with default of 32kbit
        latency: Queue latency (e.g., "400ms")
        limit: Queue size limit (e.g., "1000000") limit: Queue size in bytes
    """
    # Remove any existing qdisc
    run_cmd(["sudo", "tc", "qdisc", "del", "dev", interface, "root"], check=False)
    
    # Only apply new settings if at least rate is specified
    if rate:
        cmd = ["sudo", "tc", "qdisc", "add", "dev", interface, "root", "tbf", "rate", rate, "burst", burst]
        if limit:
            cmd.extend(["limit", limit])
        run_cmd(cmd)
    
    # Verify the settings
    result = run_cmd(["sudo", "tc", "qdisc", "show", "dev", interface])
    print(f"TC settings for {interface}: {result.stdout}")

def update_network_bandwidth(num_aggregates, num_nodes_per_aggregate, agg_bandwidth=None, node_bandwidth=None, 
                            agg_burst="32kbit", node_burst="32kbit", agg_limit="8kbit", node_limit="8kbit"):
    """
    Update bandwidth settings for all links in the network topology
    """
    print("Updating bandwidth settings for network links...")
    
    # Remove references to core bridge IP and routing table
    # Bridges should only operate at layer 2, not layer 3
    
    # Update core-to-aggregate links and their bandwidth settings
    for agg_num in range(num_aggregates):
        veth_core_agg = f"veth-core-agg{agg_num}"
        veth_agg_core = f"veth-agg{agg_num}-core"
        
        # Core-side and aggregate-side IPs
        core_agg_ip = f"10.255.{agg_num}.1"
        agg_core_ip = f"10.255.{agg_num}.2"
        
        # Create routing tables for veth interfaces if they don't exist
        create_routing_table(f"core-agg{agg_num}", 30 + agg_num*2)
        create_routing_table(f"agg{agg_num}-core", 31 + agg_num*2)
        
        # Set rules to use these tables if they don't exist
        run_cmd(["sudo", "ip", "rule", "add", "from", core_agg_ip, "table", f"core-agg{agg_num}"], check=False)
        run_cmd(["sudo", "ip", "rule", "add", "from", agg_core_ip, "table", f"agg{agg_num}-core"], check=False)
        
        if agg_bandwidth is not None and agg_burst is not None and agg_limit is not None:
            print(f"Setting bandwidth for {veth_core_agg} and {veth_agg_core}")
            set_bandwidth(veth_core_agg, agg_bandwidth, agg_burst, limit=agg_limit)
            set_bandwidth(veth_agg_core, agg_bandwidth, agg_burst, limit=agg_limit)
        
        # Setup routes in the proper routing tables, not the main table
        # Core to agg veth routes
        run_cmd(["sudo", "ip", "route", "replace", f"{agg_core_ip}/32", "dev", veth_core_agg, "table", f"core-agg{agg_num}"])
        run_cmd(["sudo", "ip", "route", "replace", f"10.{agg_num}.0.0/16", "via", agg_core_ip, "dev", veth_core_agg, "table", f"core-agg{agg_num}"])
        
        # Agg to core veth routes
        run_cmd(["sudo", "ip", "route", "replace", f"{core_agg_ip}/32", "dev", veth_agg_core, "table", f"agg{agg_num}-core"])
        
        # Update aggregate-to-node links
        for node_num in range(num_nodes_per_aggregate):
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            veth_agg_node = f"veth-agg{agg_num}-node{node_num}"
            veth_node = f"veth-node{global_node_num}"
            
            # Node-side and aggregate-side IPs
            node_ip = f"10.{agg_num}.{node_num}.2"
            agg_node_ip = f"10.{agg_num}.{node_num}.1"
            
            # Create routing tables for these veth interfaces if they don't exist
            create_routing_table(f"agg{agg_num}-node{node_num}", 50 + global_node_num*2)
            create_routing_table(f"node{global_node_num}", 51 + global_node_num*2)
            
            # Set rules to use these tables if they don't exist
            run_cmd(["sudo", "ip", "rule", "add", "from", agg_node_ip, "table", f"agg{agg_num}-node{node_num}"], check=False)
            run_cmd(["sudo", "ip", "rule", "add", "from", node_ip, "table", f"node{global_node_num}"], check=False)
            
            if node_bandwidth is not None and node_burst is not None and node_limit is not None:
                print(f"Setting bandwidth for {veth_agg_node} and {veth_node}")
                set_bandwidth(veth_agg_node, node_bandwidth, node_burst, limit=node_limit)
                set_bandwidth(veth_node, node_bandwidth, node_burst, limit=node_limit)
            
            # Setup routes in proper routing tables, not the main table
            # Node to agg veth
            run_cmd(["sudo", "ip", "route", "flush", "table", f"node{global_node_num}"], check=False)
            run_cmd(["sudo", "ip", "route", "replace", f"{agg_node_ip}/32", "dev", veth_node, "table", f"node{global_node_num}"])
            
            # Force all traffic to other nodes in same subnet through the aggregate
            for other_node_num in range(num_nodes_per_aggregate):
                if other_node_num != node_num:  # Skip self
                    other_node_ip = f"10.{agg_num}.{other_node_num}.2"
                    print(f"Adding route from node{global_node_num} to node{agg_num*num_nodes_per_aggregate+other_node_num} ({other_node_ip}) via {agg_node_ip}")
                    run_cmd(["sudo", "ip", "route", "replace", f"{other_node_ip}/32", "via", agg_node_ip, "dev", veth_node, "table", f"node{global_node_num}"])
            
            # Add routes to all other aggregate subnets through this node's aggregate
            for other_agg_num in range(num_aggregates):
                if other_agg_num != agg_num:
                    print(f"Adding route from node{global_node_num} to subnet 10.{other_agg_num}.0.0/16 via {agg_node_ip}")
                    run_cmd(["sudo", "ip", "route", "replace", f"10.{other_agg_num}.0.0/16", "via", agg_node_ip, "dev", veth_node, "table", f"node{global_node_num}"])
            
            # Agg to node veth
            run_cmd(["sudo", "ip", "route", "replace", f"{node_ip}/32", "dev", veth_agg_node, "table", f"agg{agg_num}-node{node_num}"])
        
        # Set up routes from this aggregate to other aggregates via core
        for other_agg_num in range(num_aggregates):
            if other_agg_num != agg_num:
                # Remove reference to non-existent aggregate bridge routing table
                # run_cmd(["sudo", "ip", "route", "replace", f"10.{other_agg_num}.0.0/16", "via", core_agg_ip, "dev", veth_agg_core, "table", f"agg{agg_num}"])
                
                # Only add route to the veth interface routing table
                run_cmd(["sudo", "ip", "route", "replace", f"10.{other_agg_num}.0.0/16", "via", core_agg_ip, "dev", veth_agg_core, "table", f"agg{agg_num}-core"])
                print(f"Updated route from agg{agg_num} to subnet 10.{other_agg_num}.0.0/16 via core")
    
    print("Bandwidth settings and routing updated.")

def create_routing_table(entity_name, table_id):
    """Create a routing table for an entity and set up routing rule"""
    # Add entry to rt_tables if not exists
    rt_tables_entry = f"{table_id} {entity_name}"
    run_cmd(["sudo", "bash", "-c", f"grep -q '{rt_tables_entry}' /etc/iproute2/rt_tables || echo '{rt_tables_entry}' | sudo tee -a /etc/iproute2/rt_tables"])
    
    # Flush any existing routes in the table to start clean
    run_cmd(["sudo", "ip", "route", "flush", "table", entity_name], check=False)

def create_flat_network(num_aggregates, num_nodes_per_aggregate, agg_bandwidth=None, node_bandwidth=None, 
                        agg_burst="32kbit", node_burst="32kbit", agg_limit=None, node_limit=None):
    """
    Creates a 3-layer network tree using bridges and virtual interfaces in the default namespace.
    """
    print("Creating flat network topology without namespaces...")

    # Create a dummy interface for the core instead of a bridge
    core_iface = "core0"
    run_cmd(["sudo", "ip", "link", "add", "name", core_iface, "type", "dummy"])
    run_cmd(["sudo", "ip", "addr", "add", "10.255.254.1/32", "dev", core_iface])
    run_cmd(["sudo", "ip", "link", "set", core_iface, "up"])
    enable_interface_forwarding(core_iface)

    # Enable IP forwarding and set up NAT for all 10.x.x.x traffic
    run_cmd(["sudo", "sysctl", "-w", "net.ipv4.ip_forward=1"])
    run_cmd(["sudo", "iptables", "-t", "nat", "-A", "POSTROUTING", "-s", "10.0.0.0/8", "-j", "MASQUERADE"])

    # First, configure a higher rule priority for our custom routing tables 
    # This ensures our tables are consulted before the local table
    run_cmd(["sudo", "sysctl", "-w", "net.ipv4.conf.all.rp_filter=0"], check=False)
    run_cmd(["sudo", "sysctl", "-w", "net.ipv4.conf.default.rp_filter=0"], check=False)
    
    # For each node, set a higher priority rule (lower number = higher priority)
    # The default local table rule has priority 0, so we'll use priority 1
    for agg_num in range(num_aggregates):
        for node_num in range(num_nodes_per_aggregate):
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            node_ip = f"10.{agg_num}.{node_num}.2"
            veth_node = f"veth-node{global_node_num}"
            
            # Remove any existing rules and add high-priority rule
            run_cmd(["sudo", "ip", "rule", "del", "from", node_ip], check=False)
            run_cmd(["sudo", "ip", "rule", "add", "from", node_ip, "lookup", f"node{global_node_num}", "pref", "1"])
            
            # Also add rule to ensure destinations to other nodes use our table first
            for other_agg_num in range(num_aggregates):
                for other_node_num in range(num_nodes_per_aggregate):
                    if agg_num != other_agg_num or node_num != other_node_num:
                        other_node_ip = f"10.{other_agg_num}.{other_node_num}.2"
                        run_cmd(["sudo", "ip", "rule", "add", "to", other_node_ip, "lookup", f"node{global_node_num}", "pref", "1"])

    # Create aggregate dummy interfaces and connect to core
    for agg_num in range(num_aggregates):
        agg_iface = f"agg{agg_num}"
        run_cmd(["sudo", "ip", "link", "add", "name", agg_iface, "type", "dummy"])
        run_cmd(["sudo", "ip", "addr", "add", f"10.255.{agg_num}.2/32", "dev", agg_iface])
        run_cmd(["sudo", "ip", "link", "set", agg_iface, "up"])
        enable_interface_forwarding(agg_iface)

        # Create veth pair to connect core to aggregate
        veth_core_agg = f"veth-core-agg{agg_num}"
        veth_agg_core = f"veth-agg{agg_num}-core"
        run_cmd(["sudo", "ip", "link", "add", veth_core_agg, "type", "veth", "peer", "name", veth_agg_core])

        # Define IP addresses for the ends of the veth pair
        core_agg_ip = f"10.255.{agg_num}.1"  # Core end
        agg_core_ip = f"10.255.{agg_num}.2"  # Aggregate end

        # Configure the veth interfaces with /32 masks
        run_cmd(["sudo", "ip", "addr", "add", f"{core_agg_ip}/32", "dev", veth_core_agg])
        run_cmd(["sudo", "ip", "addr", "add", f"{agg_core_ip}/32", "dev", veth_agg_core])
        
        # Create routing tables for veth interfaces
        create_routing_table(f"core-agg{agg_num}", 30 + agg_num*2)
        create_routing_table(f"agg{agg_num}-core", 31 + agg_num*2)
        
        # Set rules to use these tables
        run_cmd(["sudo", "ip", "rule", "add", "from", core_agg_ip, "table", f"core-agg{agg_num}"])
        run_cmd(["sudo", "ip", "rule", "add", "from", agg_core_ip, "table", f"agg{agg_num}-core"])

        # Connect veth interfaces to their respective bridges
        run_cmd(["sudo", "ip", "link", "set", "dev", veth_core_agg, "up"])
        enable_interface_forwarding(veth_core_agg)
        run_cmd(["sudo", "ip", "link", "set", "dev", veth_agg_core, "up"])
        enable_interface_forwarding(veth_agg_core)

        # Apply bandwidth limits if specified
        if agg_bandwidth:
            set_bandwidth(veth_core_agg, agg_bandwidth, agg_burst, agg_limit)
            set_bandwidth(veth_agg_core, agg_bandwidth, agg_burst, agg_limit)

        # Set up direct routes between veth peers in each veth's routing table
        # Core to aggregate veth
        run_cmd(["sudo", "ip", "route", "add", f"{agg_core_ip}/32", "dev", veth_core_agg, "table", f"core-agg{agg_num}"])
        # Route from core to this aggregate's subnet
        run_cmd(["sudo", "ip", "route", "add", f"10.{agg_num}.0.0/16", "via", agg_core_ip, "dev", veth_core_agg, "table", f"core-agg{agg_num}"])
        
        # Aggregate to core veth
        run_cmd(["sudo", "ip", "route", "add", f"{core_agg_ip}/32", "dev", veth_agg_core, "table", f"agg{agg_num}-core"])
        
        # Create nodes for this aggregate
        for node_num in range(num_nodes_per_aggregate):
            # Create veth pair for node
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            veth_agg_node = f"veth-agg{agg_num}-node{node_num}"
            veth_node = f"veth-node{global_node_num}"

            run_cmd(["sudo", "ip", "link", "add", veth_agg_node, "type", "veth", "peer", "name", veth_node])

            # Configure the node interface IPs
            node_ip = f"10.{agg_num}.{node_num}.2"  # Node end
            agg_node_ip = f"10.{agg_num}.{node_num}.1"  # Aggregate end

            # Use /32 addresses to avoid automatic subnet routes
            run_cmd(["sudo", "ip", "addr", "add", f"{agg_node_ip}/32", "dev", veth_agg_node])
            run_cmd(["sudo", "ip", "addr", "add", f"{node_ip}/32", "dev", veth_node])
            
            # Create routing tables for these veth interfaces
            create_routing_table(f"agg{agg_num}-node{node_num}", 50 + global_node_num*2)
            create_routing_table(f"node{global_node_num}", 51 + global_node_num*2)
            
            # Set rules to use these tables
            run_cmd(["sudo", "ip", "rule", "add", "from", agg_node_ip, "table", f"agg{agg_num}-node{node_num}"])
            run_cmd(["sudo", "ip", "rule", "add", "from", node_ip, "table", f"node{global_node_num}"])

            # Connect aggregate side to the bridge
            run_cmd(["sudo", "ip", "link", "set", "dev", veth_agg_node, "up"])
            run_cmd(["sudo", "ip", "link", "set", "dev", veth_node, "up"])
            enable_interface_forwarding(veth_node)

            # Apply bandwidth limits if specified
            if node_bandwidth:
                set_bandwidth(veth_agg_node, node_bandwidth, node_burst, node_limit)
                set_bandwidth(veth_node, node_bandwidth, node_burst, node_limit)

            # Set up the node's routing table
            
            # Direct route to aggregate end of veth pair
            run_cmd(["sudo", "ip", "route", "add", f"{agg_node_ip}/32", "dev", veth_node, "table", f"node{global_node_num}"])
            
            # Add specific host routes for same-subnet nodes via aggregate
            for other_node_num in range(num_nodes_per_aggregate):
                if other_node_num != node_num:  # Skip self
                    other_node_ip = f"10.{agg_num}.{other_node_num}.2"
                    run_cmd(["sudo", "ip", "route", "add", f"{other_node_ip}/32", "via", agg_node_ip, 
                           "dev", veth_node, "table", f"node{global_node_num}"])
            
            # Routes to other subnets via the aggregate
            for other_agg_num in range(num_aggregates):
                if other_agg_num != agg_num:
                    run_cmd(["sudo", "ip", "route", "add", f"10.{other_agg_num}.0.0/16", "via", agg_node_ip, 
                           "dev", veth_node, "table", f"node{global_node_num}"])
            
            # Set up the veth-agg-node routing table
            run_cmd(["sudo", "ip", "route", "add", f"{node_ip}/32", "dev", veth_agg_node, "table", f"agg{agg_num}-node{node_num}"])
            
            # REMOVE routes from main routing table - each interface should only use its own table
            # These routes were being incorrectly added to main table:
            
            # Remove direct route to aggregate end of veth pair in main table - keep only in node's table
            # run_cmd(["sudo", "ip", "route", "add", f"{agg_node_ip}/32", "dev", veth_node])
            
            # Remove routes to same-subnet nodes via aggregate in main table - keep only in node's table
            # for other_node_num in range(num_nodes_per_aggregate):
            #    if other_node_num != node_num:
            #        other_node_ip = f"10.{agg_num}.{other_node_num}.2"
            #        run_cmd(["sudo", "ip", "route", "add", f"{other_node_ip}/32", "via", agg_node_ip, "dev", veth_node])
            
            # Remove routes to other subnet nodes via aggregate in main table - keep only in node's table
            # for other_agg_num in range(num_aggregates):
            #    if other_agg_num != agg_num:
            #        run_cmd(["sudo", "ip", "route", "add", f"10.{other_agg_num}.0.0/16", "via", agg_node_ip, "dev", veth_node])
            
            print(f"Created node{global_node_num} interface: {veth_node} with IP {node_ip}")
        
        # Add routes in agg{agg_num}-core table for other aggregates
        for other_agg_num in range(num_aggregates):
            if other_agg_num != agg_num:
                run_cmd(["sudo", "ip", "route", "add", f"10.{other_agg_num}.0.0/16", "via", core_agg_ip, 
                         "dev", veth_agg_core, "table", f"agg{agg_num}-core"])
                print(f"Added route from agg{agg_num} to subnet 10.{other_agg_num}.0.0/16 via core")

    print("\nNetwork setup complete.")
    print("To access a specific node, use its interface directly.")
    print(f"Core interface: {core_iface}")

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
    
    # Remove all routing rules and flush routing tables
    print("Removing routing rules and tables...")
    
    # Remove core bridge routing rules and table - not needed anymore
    # try:
    #     run_cmd(["sudo", "ip", "rule", "del", "from", "10.255.254.1", "table", "core"], check=False)
    #     run_cmd(["sudo", "ip", "route", "flush", "table", "core"], check=False)
    # except:
    #     pass
    
    # Remove veth pair routing tables, no bridge tables anymore
    for agg_num in range(num_aggregates):
        try:
            # Remove aggregate bridge rule and table - not needed anymore
            # run_cmd(["sudo", "ip", "rule", "del", "from", f"10.{agg_num}.254.1", "table", f"agg{agg_num}"], check=False)
            # run_cmd(["sudo", "ip", "route", "flush", "table", f"agg{agg_num}"], check=False)
            
            # Remove core-agg veth rules and tables
            core_agg_ip = f"10.255.{agg_num}.1"
            agg_core_ip = f"10.255.{agg_num}.2"
            run_cmd(["sudo", "ip", "rule", "del", "from", core_agg_ip, "table", f"core-agg{agg_num}"], check=False)
            run_cmd(["sudo", "ip", "rule", "del", "from", agg_core_ip, "table", f"agg{agg_num}-core"], check=False)
            run_cmd(["sudo", "ip", "route", "flush", "table", f"core-agg{agg_num}"], check=False)
            run_cmd(["sudo", "ip", "route", "flush", "table", f"agg{agg_num}-core"], check=False)
            
            # Remove node-related rules and tables
            for node_num in range(num_nodes_per_aggregate):
                global_node_num = agg_num * num_nodes_per_aggregate + node_num
                node_ip = f"10.{agg_num}.{node_num}.2"
                agg_node_ip = f"10.{agg_num}.{node_num}.1"
                
                # Remove node and agg-node veth rules and tables
                run_cmd(["sudo", "ip", "rule", "del", "from", node_ip, "table", f"node{global_node_num}"], check=False)
                run_cmd(["sudo", "ip", "rule", "del", "from", agg_node_ip, "table", f"agg{agg_num}-node{node_num}"], check=False)
                run_cmd(["sudo", "ip", "route", "flush", "table", f"node{global_node_num}"], check=False)
                run_cmd(["sudo", "ip", "route", "flush", "table", f"agg{agg_num}-node{node_num}"], check=False)
        except:
            pass
    
    # Now we can remove all remaining routes from the main table, if any
    print("Removing any IP routes from main table...")
    
    for agg_num in range(num_aggregates):
        # Define veth pair names for core-agg links
        veth_core_agg = f"veth-core-agg{agg_num}"
        veth_agg_core = f"veth-agg{agg_num}-core"
        core_agg_ip = f"10.255.{agg_num}.1"
        agg_core_ip = f"10.255.{agg_num}.2"
        
        # Remove direct routes between veth pairs from main table if any
        try:
            run_cmd(["sudo", "ip", "route", "del", f"{agg_core_ip}/32", "dev", veth_core_agg], check=False)
            run_cmd(["sudo", "ip", "route", "del", f"{core_agg_ip}/32", "dev", veth_agg_core], check=False)
        except:
            pass
        
        # Remove routes from core to each aggregate's network from main table if any
        try:
            run_cmd(["sudo", "ip", "route", "del", f"10.{agg_num}.0.0/16"], check=False)
        except:
            pass
        
        # Remove routes between aggregates from main table if any
        for other_agg_num in range(num_aggregates):
            if other_agg_num != agg_num:
                try:
                    run_cmd(["sudo", "ip", "route", "del", f"10.{other_agg_num}.0.0/16", "dev", veth_agg_core], check=False)
                except:
                    pass

        # Remove node routes and veth pair routes from main table if any
        for node_num in range(num_nodes_per_aggregate):
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            veth_node = f"veth-node{global_node_num}"
            veth_agg_node = f"veth-agg{agg_num}-node{node_num}"
            node_ip = f"10.{agg_num}.{node_num}.2"
            agg_node_ip = f"10.{agg_num}.{node_num}.1"
            
            # Remove direct routes between node veth pairs from main table if any
            try:
                run_cmd(["sudo", "ip", "route", "del", f"{node_ip}/32", "dev", veth_agg_node], check=False)
                run_cmd(["sudo", "ip", "route", "del", f"{agg_node_ip}/32", "dev", veth_node], check=False)
            except:
                pass
            
            # Remove routes to other nodes in same subnet from main table if any
            for other_node_num in range(num_nodes_per_aggregate):
                if other_node_num != node_num:
                    other_node_ip = f"10.{agg_num}.{other_node_num}.2"
                    try:
                        run_cmd(["sudo", "ip", "route", "del", f"{other_node_ip}/32", "dev", veth_node], check=False)
                    except:
                        pass
            
            # Remove routes to other subnets from main table if any
            for other_agg_num in range(num_aggregates):
                if other_agg_num != agg_num:
                    try:
                        run_cmd(["sudo", "ip", "route", "del", f"10.{other_agg_num}.0.0/16", "dev", veth_node], check=False)
                    except:
                        pass

    # Simplify main table cleanup since we shouldn't have added anything there
    # Just do a general flush of main table routes for our networks to be safe
    print("Making sure main routing table is clean...")
    for agg_num in range(num_aggregates):
        # Remove subnet route if any
        try:
            run_cmd(["sudo", "ip", "route", "flush", f"10.{agg_num}.0.0/16"], check=False)
        except:
            pass

    # Clean up our high priority rules
    print("Removing high-priority IP rules...")
    for agg_num in range(num_aggregates):
        for node_num in range(num_nodes_per_aggregate):
            global_node_num = agg_num * num_nodes_per_aggregate + node_num
            node_ip = f"10.{agg_num}.{node_num}.2"
            
            # Remove high-priority rules
            run_cmd(["sudo", "ip", "rule", "del", "from", node_ip, "pref", "1"], check=False)
            
            # Remove destination rules
            for other_agg_num in range(num_aggregates):
                for other_node_num in range(num_nodes_per_aggregate):
                    if agg_num != other_agg_num or node_num != other_node_num:
                        other_node_ip = f"10.{other_agg_num}.{other_node_num}.2"
                        run_cmd(["sudo", "ip", "rule", "del", "to", other_node_ip, "pref", "1"], check=False)

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

    for agg_num in range(num_aggregates):
        agg_iface = f"agg{agg_num}"
        try:
            run_cmd(["sudo", "ip", "link", "del", agg_iface], check=False)
        except Exception as e:
            print(f"Failed to remove aggregate interface {agg_iface}: {e}")

    # Remove core interface
    try:
        run_cmd(["sudo", "ip", "link", "del", "core0"], check=False)
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


def start_iperf_server(ip_address):
    """Start an iperf server on the specified IP address"""
    # Kill any existing iperf3 processes
    run_cmd(["sudo", "pkill", "-f", "iperf3"], check=False)
    # Start iperf server in the background
    server_proc = subprocess.Popen(
        ["sudo", "iperf3", "-s", "-B", ip_address, "-1"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    # Give the server a moment to start
    time.sleep(1)
    return server_proc

def run_iperf_client(source_ip, destination_ip, duration=3):
    """Run iperf client and return the bandwidth measurement"""
    try:
        result = run_cmd([
            "sudo", "iperf3", "-c", destination_ip, 
            "-B", source_ip, "-t", str(duration),
            "-f", "m", "-J"  # JSON output format
        ])
        
        # Parse JSON output to get bandwidth
        import json
        data = json.loads(result.stdout)
        bandwidth_mbps = data["end"]["sum_received"]["bits_per_second"] / 1000000
        return bandwidth_mbps
    except Exception as e:
        print(f"  Error running iperf from {source_ip} to {destination_ip}: {e}")
        return None

def test_connectivity(num_aggregates, num_nodes_per_aggregate):
    """Test connectivity and measure bandwidth between all created nodes using iperf3"""
    print("\nTesting connectivity and measuring bandwidth between all nodes using iperf3...")
    
    # Check if iperf3 is installed
    try:
        run_cmd(["which", "iperf3"])
    except:
        print("Error: iperf3 not found. Please install it with 'sudo apt-get install iperf3'")
        return

    # Build a list of nodes as (global_node_number, IP address) pairs
    nodes = []
    for agg in range(num_aggregates):
        for node in range(num_nodes_per_aggregate):
            global_node = agg * num_nodes_per_aggregate + node
            ip = get_node_ip(global_node, num_nodes_per_aggregate)
            nodes.append((global_node, ip))
    
    # Create a results matrix
    results = []
    
    # For each pair of distinct nodes, measure bandwidth using iperf3
    for src in nodes:
        # Create a row for this source node
        row_results = []
        
        for dst in nodes:
            if src[0] == dst[0]:
                # Skip testing from a node to itself
                row_results.append("N/A")
                continue
                
            print(f"Testing bandwidth from node{src[0]} ({src[1]}) to node{dst[0]} ({dst[1]})...")
            
            # Start iperf server on destination node
            server_proc = start_iperf_server(dst[1])
            
            # Run iperf client on source node
            bandwidth = run_iperf_client(src[1], dst[1])
            
            # Terminate server
            server_proc.terminate()
            
            if bandwidth is not None:
                result_str = f"{bandwidth:.2f} Mbps"
                print(f"  Bandwidth: {result_str}")
                row_results.append(result_str)
            else:
                print(f"  Failed to measure bandwidth")
                row_results.append("Failed")
            
            # Wait a moment before the next test
            time.sleep(1)
        
        results.append(row_results)
    
    # Display results in a table format
    print("\nBandwidth Results (Mbps):")
    
    # Print header row with destination node numbers
    header = "Src\\Dst |"
    for dst in nodes:
        header += f" Node {dst[0]} |"
    print(header)
    print("-" * len(header))
    
    # Print each row with source node and bandwidth results
    for i, src in enumerate(nodes):
        row = f" Node {src[0]} |"
        for j, bw in enumerate(results[i]):
            row += f" {bw:^8} |"
        print(row)
    
    print("\nBandwidth test completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a network topology without using namespaces")
    parser.add_argument("--aggs", type=int, default=2, help="Number of aggregate bridges")
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes per aggregate")
    parser.add_argument("--cleanup", action="store_true", help="Remove the network topology")
    parser.add_argument("--agg-bandwidth", type=str, default=None, help="Bandwidth for aggregate links (e.g., '100mbit')")
    parser.add_argument("--node-bandwidth", type=str, default=None, help="Bandwidth for node links (e.g., '10mbit')")
    parser.add_argument("--test", action="store_true", help="Test connectivity between nodes after setup")
    # Add new arguments for bandwidth update function
    parser.add_argument("--update-bandwidth", action="store_true", help="Update bandwidth settings for existing network")
    parser.add_argument("--agg-burst", type=str, default="32kbit", help="Burst size for aggregate links (e.g., '32kbit')")
    parser.add_argument("--node-burst", type=str, default="32kbit", help="Burst size for node links (e.g., '32kbit')")
    # parser.add_argument("--agg-latency", type=str, default=None, help="Latency for aggregate links (e.g., '400ms')")
    # parser.add_argument("--node-latency", type=str, default=None, help="Latency for node links (e.g., '400ms')")
    parser.add_argument("--agg-limit", type=str, default="1000000", help="Queue size limit for aggregate links (e.g., '8kbit')")
    parser.add_argument("--node-limit", type=str, default="1000000", help="Queue size limit for node links (e.g., '8kbit')")

    args = parser.parse_args()

    if args.cleanup:
        remove_flat_network(args.aggs, args.nodes)
    elif args.update_bandwidth:
        update_network_bandwidth(
            args.aggs, 
            args.nodes,
            args.agg_bandwidth,
            args.node_bandwidth,
            args.agg_burst,
            args.node_burst,
            args.agg_limit,
            args.node_limit
        )
    elif args.test:
        test_connectivity(args.aggs, args.nodes)
    else:
        create_flat_network(
            args.aggs, 
            args.nodes, 
            args.agg_bandwidth,
            args.node_bandwidth,
            agg_limit=args.agg_limit,
            node_limit=args.node_limit,
        )
        test_connectivity(args.aggs, args.nodes)
