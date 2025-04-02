#!/usr/bin/env python3

import subprocess
import sys
import argparse
import time
import shlex
import collections # To store network info
import json        # To parse iperf3 output
import re          # For parsing bandwidth strings
import shutil      # To check if iperf3 exists

# --- Configuration ---
# Interface naming conventions
VETH_CORE_AGG_PREFIX = "c-a"  # e.g., c-a1 <=> a1-c
VETH_AGG_NODE_PREFIX = "a{}-e{}_{}" # e.g., a1-e1_1 <=> e1_1-a1
BRIDGE_NAME_PREFIX = "br-agg" # e.g., br-agg1

# iperf settings
IPERF_TEST_DURATION = 5
IPERF_OMIT_SECS = 1
BANDWIDTH_TOLERANCE = 0.20

# --- Global lists/dicts to track created resources ---
CREATED_NAMESPACES = []
CREATED_VETH_PAIRS_BASE = []
NETWORK_CONFIGURATION = collections.defaultdict(list)

# --- Regex for parsing names ---
CORE_PATTERN = re.compile(r"^core$")
AGG_PATTERN = re.compile(r"^agg(\d+)$")
END_PATTERN = re.compile(r"^end(\d+)-(\d+)$") # Match end{i}-{j}

# --- Helper Functions ---

def run_cmd(command, check=True, capture_output=False, text=True, shell=False, suppress_print=False):
    """Runs a shell command using subprocess."""
    if not shell:
        command_list = shlex.split(command)
        cmd_print = command
    else:
        command_list = command
        cmd_print = command

    if not suppress_print:
        print(f"Executing: {cmd_print}")
    try:
        result = subprocess.run(
            command_list,
            check=check,
            capture_output=capture_output,
            text=text,
            shell=shell
        )
        if result.stderr:
            stderr_lower = result.stderr.strip().lower()
            # Suppress common "not found" errors when deleting non-existent things
            if not (("no such file or directory" in stderr_lower or "cannot find device" in stderr_lower) and "del" in cmd_print.lower()):
                 print(stderr_lower, file=sys.stderr)

        if capture_output:
             return result.stdout.strip() if result.stdout else ""
        return result
    except subprocess.CalledProcessError as e:
        if not (check is False):
            print(f"Error executing command: {cmd_print}", file=sys.stderr)
            print(f"  Return code: {e.returncode}", file=sys.stderr)
            if hasattr(e, 'stdout') and e.stdout:
                print(f"  Stdout: {e.stdout.strip()}", file=sys.stderr)
            if hasattr(e, 'stderr') and e.stderr:
                print(f"  Stderr: {e.stderr.strip()}", file=sys.stderr)
            raise
        else:
            return e # Return error object if check=False
    except FileNotFoundError:
        cmd_name = command_list[0] if isinstance(command_list, list) else command_list.split()[0]
        print(f"Error: Command not found: '{cmd_name}'. Is it installed and in PATH?", file=sys.stderr)
        raise

def setup_tc(namespace, interface, bandwidth):
    """Applies HTB bandwidth limiting using tc, using delete then add strategy."""

    # Always try to delete existing root qdisc first to ensure a clean state
    # Use check=False as it's okay if no qdisc exists to be deleted
    del_qdisc_cmd = f"sudo ip netns exec {namespace} tc qdisc del dev {interface} root"
    print(f"Ensuring no existing root qdisc on {interface} in {namespace}...")
    run_cmd(del_qdisc_cmd, check=False, suppress_print=True) # Suppress expected errors

    if not bandwidth:
        print(f"Removed TC qdisc for {interface} in {namespace} (or confirmed none exists)")
        return # If bandwidth is empty, we just wanted to delete.

    # If bandwidth is specified, add the HTB qdisc and class
    print(f"Applying TC ({bandwidth}) to {interface} in namespace {namespace}")
    try:
        # Add HTB qdisc at root handle 1:
        # Use 'add' now since we deleted any prior qdisc
        add_qdisc_cmd = f"sudo ip netns exec {namespace} tc qdisc add dev {interface} root handle 1: htb default 1"
        run_cmd(add_qdisc_cmd) # Use check=True, add should succeed on clean slate

        # Add class 1:1 (no need to check/change/replace, qdisc is freshly added)
        add_class_cmd = f"sudo ip netns exec {namespace} tc class add dev {interface} parent 1: classid 1:1 htb rate {bandwidth}"
        print(f"  Adding new class 1:1 rate...")
        run_cmd(add_class_cmd) # Use check=True

    except Exception as e:
        print(f"Error during setup_tc for {namespace}/{interface}: {e}", file=sys.stderr)
        # Attempt to clean up the potentially partially added qdisc on error
        print(f"Attempting cleanup after error on {interface} in {namespace}...")
        run_cmd(f"sudo ip netns exec {namespace} tc qdisc del dev {interface} root", check=False, suppress_print=True)
        raise # Re-raise to indicate failure in the calling function (update_bandwidth)


def parse_bandwidth(bw_str):
    """Converts bandwidth string (e.g., 100mbit, 1gbit) to bits per second."""
    if not bw_str:
        return 0
    bw_str = bw_str.lower().strip()
    match = re.match(r'(\d+(\.\d+)?)\s*([kmg]?bit)', bw_str)
    if not match:
        raise ValueError(f"Invalid bandwidth format: {bw_str}")

    value = float(match.group(1))
    unit = match.group(3)

    if unit == 'kbit': return int(value * 1000)
    elif unit == 'mbit': return int(value * 1000 * 1000)
    elif unit == 'gbit': return int(value * 1000 * 1000 * 1000)
    elif unit == 'bit': return int(value)
    else: raise ValueError(f"Unknown bandwidth unit in: {bw_str}")

def check_iperf_installed():
    """Checks if iperf3 is installed on the host."""
    if shutil.which("iperf3") is None:
        print("Error: 'iperf3' command not found. Please install iperf3.", file=sys.stderr)
        return False
    print("iperf3 found.")
    return True

def get_link_details(name1, name2):
    """
    Parses two namespace names and returns the namespaces and the
    veth interface names connecting them according to the script's convention.
    Returns: (namespace1, interface1, namespace2, interface2)
    Raises: ValueError if names are invalid or the pair is not directly connected.
    """
    m1_core = CORE_PATTERN.match(name1)
    m1_agg = AGG_PATTERN.match(name1)
    m1_end = END_PATTERN.match(name1)

    m2_core = CORE_PATTERN.match(name2)
    m2_agg = AGG_PATTERN.match(name2)
    m2_end = END_PATTERN.match(name2)

    # Case 1: core <-> aggX
    if m1_core and m2_agg:
        agg_idx = m2_agg.group(1)
        ns1, if1 = name1, f"{VETH_CORE_AGG_PREFIX}{agg_idx}"
        ns2, if2 = name2, f"a{agg_idx}-c"
        return ns1, if1, ns2, if2
    elif m1_agg and m2_core:
        ns2, if2, ns1, if1 = get_link_details(name2, name1)
        return ns1, if1, ns2, if2

    # Case 2: aggX <-> endX-Y
    elif m1_agg and m2_end:
        agg_idx1 = m1_agg.group(1)
        end_idx_agg, end_idx_node = m2_end.group(1), m2_end.group(2)
        if agg_idx1 != end_idx_agg:
            raise ValueError(f"Mismatch between aggregate index in {name1} (agg{agg_idx1}) and {name2} (end{end_idx_agg}-...)")
        ns1, if1 = name1, VETH_AGG_NODE_PREFIX.format(agg_idx1, end_idx_agg, end_idx_node)
        ns2, if2 = name2, f"e{end_idx_agg}_{end_idx_node}-a{agg_idx1}"
        return ns1, if1, ns2, if2
    elif m1_end and m2_agg:
        ns2, if2, ns1, if1 = get_link_details(name2, name1)
        return ns1, if1, ns2, if2

    # Invalid or unsupported pair
    else:
        invalid_names = []
        if not (m1_core or m1_agg or m1_end): invalid_names.append(name1)
        if not (m2_core or m2_agg or m2_end): invalid_names.append(name2)
        if invalid_names:
             raise ValueError(f"Invalid device name(s): {', '.join(invalid_names)}")
        else:
             raise ValueError(f"Devices are not directly connected in this topology: {name1}, {name2}")


# --- Main Network Functions ---

def create_network(num_aggregates, num_nodes_per_agg, core_agg_bw, agg_node_bw):
    """Creates the entire 3-layer network topology using bridges in aggregates."""
    global CREATED_NAMESPACES, CREATED_VETH_PAIRS_BASE, NETWORK_CONFIGURATION
    CREATED_NAMESPACES = []
    CREATED_VETH_PAIRS_BASE = []
    NETWORK_CONFIGURATION = collections.defaultdict(list)

    try:
        # 1. Create Core Namespace
        core_name = "core"
        print(f"\n--- Creating {core_name} namespace ---")
        run_cmd(f"sudo ip netns add {core_name}")
        CREATED_NAMESPACES.append(core_name)
        run_cmd(f"sudo ip netns exec {core_name} ip link set dev lo up")
        NETWORK_CONFIGURATION[core_name].append("lo: 127.0.0.1/8 UP")
        run_cmd(f"sudo ip netns exec {core_name} sysctl -w net.ipv4.ip_forward=1")

        # 2. Create Aggregates and Connect to Core
        for i in range(1, num_aggregates + 1):
            agg_name = f"agg{i}"
            bridge_name = f"{BRIDGE_NAME_PREFIX}{i}"
            print(f"\n--- Creating {agg_name} namespace & connecting to {core_name} ---")

            run_cmd(f"sudo ip netns add {agg_name}")
            CREATED_NAMESPACES.append(agg_name)
            run_cmd(f"sudo ip netns exec {agg_name} ip link set dev lo up")
            NETWORK_CONFIGURATION[agg_name].append("lo: 127.0.0.1/8 UP")
            run_cmd(f"sudo ip netns exec {agg_name} sysctl -w net.ipv4.ip_forward=1")

            print(f"Creating bridge {bridge_name} in {agg_name}")
            run_cmd(f"sudo ip netns exec {agg_name} ip link add name {bridge_name} type bridge")
            run_cmd(f"sudo ip netns exec {agg_name} ip link set dev {bridge_name} up")
            agg_gateway_ip = f"10.{i}.0.1"
            agg_gateway_ip_cidr = f"{agg_gateway_ip}/16"
            run_cmd(f"sudo ip netns exec {agg_name} ip addr add {agg_gateway_ip_cidr} dev {bridge_name}")
            NETWORK_CONFIGURATION[agg_name].append(f"{bridge_name}: {agg_gateway_ip_cidr} UP")

            veth_core_end = f"{VETH_CORE_AGG_PREFIX}{i}"
            veth_agg_end_core = f"a{i}-c"
            core_agg_subnet_base = f"10.0.{i}"
            core_ip = f"{core_agg_subnet_base}.1/24"
            agg_ip_tocore = f"{core_agg_subnet_base}.2/24"

            print(f"Creating veth pair: {veth_core_end} <--> {veth_agg_end_core}")
            run_cmd(f"sudo ip link add {veth_core_end} type veth peer name {veth_agg_end_core}")
            CREATED_VETH_PAIRS_BASE.append(veth_core_end)

            run_cmd(f"sudo ip link set {veth_core_end} netns {core_name}")
            run_cmd(f"sudo ip link set {veth_agg_end_core} netns {agg_name}")

            run_cmd(f"sudo ip netns exec {core_name} ip addr add {core_ip} dev {veth_core_end}")
            run_cmd(f"sudo ip netns exec {agg_name} ip addr add {agg_ip_tocore} dev {veth_agg_end_core}")
            run_cmd(f"sudo ip netns exec {core_name} ip link set dev {veth_core_end} up")
            run_cmd(f"sudo ip netns exec {agg_name} ip link set dev {veth_agg_end_core} up")
            NETWORK_CONFIGURATION[core_name].append(f"{veth_core_end}: {core_ip} UP")
            NETWORK_CONFIGURATION[agg_name].append(f"{veth_agg_end_core}: {agg_ip_tocore} UP")

            # Apply initial TC settings using the robust setup_tc
            setup_tc(core_name, veth_core_end, core_agg_bw)
            setup_tc(agg_name, veth_agg_end_core, core_agg_bw)

            agg_downstream_subnet = f"10.{i}.0.0/16"
            run_cmd(f"sudo ip netns exec {core_name} ip route add {agg_downstream_subnet} via {core_agg_subnet_base}.2")
            run_cmd(f"sudo ip netns exec {agg_name} ip route add default via {core_agg_subnet_base}.1")

            # 3. Create End Nodes and Connect to this Aggregate's Bridge
            for j in range(1, num_nodes_per_agg + 1):
                node_name = f"end{i}-{j}"
                print(f"  Creating {node_name} namespace & connecting to {bridge_name} in {agg_name}")

                run_cmd(f"sudo ip netns add {node_name}")
                CREATED_NAMESPACES.append(node_name)
                run_cmd(f"sudo ip netns exec {node_name} ip link set dev lo up")
                NETWORK_CONFIGURATION[node_name].append("lo: 127.0.0.1/8 UP")

                veth_agg_end_node = VETH_AGG_NODE_PREFIX.format(i, i, j)
                veth_node_end_agg = f"e{i}_{j}-a{i}"
                node_ip = f"10.{i}.1.{j}"
                node_ip_cidr = f"{node_ip}/16"

                print(f"  Creating veth pair: {veth_agg_end_node} <--> {veth_node_end_agg}")
                run_cmd(f"sudo ip link add {veth_agg_end_node} type veth peer name {veth_node_end_agg}")
                CREATED_VETH_PAIRS_BASE.append(veth_agg_end_node)

                run_cmd(f"sudo ip link set {veth_agg_end_node} netns {agg_name}")
                run_cmd(f"sudo ip link set {veth_node_end_agg} netns {node_name}")

                run_cmd(f"sudo ip netns exec {node_name} ip addr add {node_ip_cidr} dev {veth_node_end_agg}")
                run_cmd(f"sudo ip netns exec {agg_name} ip link set dev {veth_agg_end_node} up")
                run_cmd(f"sudo ip netns exec {node_name} ip link set dev {veth_node_end_agg} up")

                print(f"  Attaching {veth_agg_end_node} to bridge {bridge_name} in {agg_name}")
                run_cmd(f"sudo ip netns exec {agg_name} ip link set dev {veth_agg_end_node} master {bridge_name}")

                NETWORK_CONFIGURATION[node_name].append(f"{veth_node_end_agg}: {node_ip_cidr} UP")
                NETWORK_CONFIGURATION[agg_name].append(f"{veth_agg_end_node}: BRIDGE PORT ({bridge_name}) UP")

                # Apply initial TC settings using the robust setup_tc
                setup_tc(agg_name, veth_agg_end_node, agg_node_bw)
                setup_tc(node_name, veth_node_end_agg, agg_node_bw)

                run_cmd(f"sudo ip netns exec {node_name} ip route add default via {agg_gateway_ip}")

        # 4. Final Summary Printout (Moved to end after potential tests)

    except Exception as e:
        print(f"\n---! ERROR during network creation: {e} !---", file=sys.stderr)
        print("---! Attempting emergency cleanup... !---", file=sys.stderr)
        cleanup_network(num_aggregates, num_nodes_per_agg)
        print("---! Emergency cleanup finished. Exiting. !---", file=sys.stderr)
        sys.exit(1)

def run_iperf_test(client_ns, server_ns, server_ip, expected_bw_bps):
    """Runs a single iperf3 test between two namespaces."""
    print(f"\n--- Testing {client_ns} -> {server_ns} ({server_ip}) ---")
    if expected_bw_bps == 0:
        expected_bw_mbit = 0.0
        print("Expected Bandwidth: UNLIMITED (0 indicates no limit set)")
    else:
        expected_bw_mbit = expected_bw_bps / 1e6
        print(f"Expected Bandwidth: {expected_bw_mbit:.2f} Mbit/s")

    server_cmd = f"sudo ip netns exec {server_ns} iperf3 -s -D"
    client_cmd = f"sudo ip netns exec {client_ns} iperf3 -c {server_ip} -t {IPERF_TEST_DURATION} -O {IPERF_OMIT_SECS} -J"

    try:
        print("Starting iperf3 server...")
        run_cmd(server_cmd)
        time.sleep(1)

        print("Starting iperf3 client...")
        json_output = run_cmd(client_cmd, capture_output=True, suppress_print=True)

        try:
            result = json.loads(json_output)
            if 'error' in result:
                 print(f"  iperf3 client error: {result['error']}", file=sys.stderr)
                 return False

            measured_bps = result.get('end', {}).get('sum_sent', {}).get('bits_per_second', 0)
            measured_mbit = measured_bps / 1e6
            print(f"  Measured Bandwidth: {measured_mbit:.2f} Mbit/s")

            if expected_bw_bps == 0:
                print("  Result: PASS (No bandwidth limit to check against)")
                return True

            lower_bound = expected_bw_bps * (1 - BANDWIDTH_TOLERANCE)
            upper_bound = expected_bw_bps * (1 + BANDWIDTH_TOLERANCE)

            if lower_bound <= measured_bps <= upper_bound:
                print(f"  Result: PASS (within +/- {BANDWIDTH_TOLERANCE*100:.0f}% tolerance)")
                return True
            else:
                if measured_bps < lower_bound: reason = "Lower than expected"
                elif measured_bps > upper_bound: reason = "Higher than expected"
                else: reason = "Comparison error"
                print(f"  Result: FAIL ({reason}. Expected {expected_bw_mbit:.2f}, Measured {measured_mbit:.2f} Mbit/s)")
                return False

        except json.JSONDecodeError:
            print("  Error: Failed to parse iperf3 JSON output.", file=sys.stderr)
            print(f"  Raw output: {json_output}", file=sys.stderr)
            return False
        except Exception as e:
             print(f"  Error processing iperf results: {e}", file=sys.stderr)
             return False

    except Exception as e:
        print(f"  Error running iperf command: {e}", file=sys.stderr)
        return False
    finally:
        print("Stopping iperf3 server...")
        kill_cmd = f"sudo ip netns exec {server_ns} pkill iperf3"
        run_cmd(kill_cmd, check=False, suppress_print=True)


def test_bandwidth(num_aggregates, num_nodes_per_agg, core_agg_bw_str, agg_node_bw_str):
    """Runs various iperf3 tests to verify bandwidth limits on an existing network."""
    print("\n" + "="*40)
    print(" Starting Bandwidth Tests")
    print("="*40)

    if not check_iperf_installed():
        return False

    try:
        core_bw_bps = parse_bandwidth(core_agg_bw_str)
        node_bw_bps = parse_bandwidth(agg_node_bw_str)
    except ValueError as e:
        print(f"Error parsing bandwidth: {e}", file=sys.stderr)
        return False

    results = {}
    overall_success = True

    # Test Case 1: Intra-Aggregate
    if num_aggregates >= 1 and num_nodes_per_agg >= 2:
        expected_bps = node_bw_bps
        test_name = "Intra-Agg (end1-1 -> end1-2)"
        passed = run_iperf_test("end1-1", "end1-2", "10.1.1.2", expected_bps)
        results[test_name] = passed
        if not passed: overall_success = False
    else:
        print("\nSkipping Intra-Aggregate test (requires >= 1 aggregate and >= 2 nodes per aggregate).")

    # Test Case 2: Inter-Aggregate
    if num_aggregates >= 2 and num_nodes_per_agg >= 1:
        if core_bw_bps > 0 and node_bw_bps > 0: expected_bps = min(core_bw_bps, node_bw_bps)
        elif core_bw_bps > 0: expected_bps = core_bw_bps
        elif node_bw_bps > 0: expected_bps = node_bw_bps
        else: expected_bps = 0
        test_name = "Inter-Agg (end1-1 -> end2-1)"
        passed = run_iperf_test("end1-1", "end2-1", "10.2.1.1", expected_bps)
        results[test_name] = passed
        if not passed: overall_success = False
    else:
        print("\nSkipping Inter-Aggregate test (requires >= 2 aggregates and >= 1 node per aggregate).")

    # Test Case 3: Aggregate to Core
    if num_aggregates >= 1:
        expected_bps = core_bw_bps
        test_name = "Agg-Core (agg1 -> core)"
        passed = run_iperf_test("agg1", "core", "10.0.1.1", expected_bps)
        results[test_name] = passed
        if not passed: overall_success = False
    else:
        print("\nSkipping Aggregate-Core test (requires >= 1 aggregate).")

    # Test Results Summary
    print("\n" + "="*40)
    print(" Bandwidth Test Summary")
    print("="*40)
    if not results:
         print("No tests were run.")
         pass
    else:
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  Test: {name:<30} Status: {status}")

    if not results:
        print("\nNo bandwidth tests were executed.")
    elif overall_success:
         print("\nAll executed bandwidth tests passed (within tolerance).")
    else:
         print("\nOne or more bandwidth tests failed.")
    print("="*40)
    return overall_success


def update_bandwidth(device1_name, device2_name, new_bandwidth_str):
    """Updates the bandwidth limit on the link between two devices."""
    print(f"\n--- Updating bandwidth between {device1_name} and {device2_name} ---")
    try:
        if not new_bandwidth_str:
            print("  New bandwidth is empty: removing existing rate limits.")
            # parse_bandwidth handles empty string -> 0
        else:
            # Validate format before proceeding
            parse_bandwidth(new_bandwidth_str) # Raises ValueError on invalid format

        ns1, if1, ns2, if2 = get_link_details(device1_name, device2_name)
        print(f"  Targeting: {ns1}/{if1} <--> {ns2}/{if2}")

        print(f"  Applying limit '{new_bandwidth_str}' to {ns1}/{if1}...")
        setup_tc(ns1, if1, new_bandwidth_str) # Use updated setup_tc

        print(f"  Applying limit '{new_bandwidth_str}' to {ns2}/{if2}...")
        setup_tc(ns2, if2, new_bandwidth_str) # Use updated setup_tc

        print(f"  Successfully updated bandwidth between {device1_name} and {device2_name}.")
        return True

    except ValueError as e:
        print(f"Error updating bandwidth: Invalid input - {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error applying tc commands during update: {e}", file=sys.stderr)
        return False


def cleanup_network(num_aggregates, num_nodes_per_agg):
    """
    Deletes network components based on the provided topology size.
    """
    print("\n--- Cleaning up network based on provided parameters ---")
    print(f"Targeting topology: {num_aggregates} aggregate(s), {num_nodes_per_agg} nodes/agg")

    namespaces_to_delete = ['core']
    for i in range(1, num_aggregates + 1):
        namespaces_to_delete.append(f"agg{i}")
        for j in range(1, num_nodes_per_agg + 1):
            namespaces_to_delete.append(f"end{i}-{j}")

    print(f"Namespaces targeted for deletion: {namespaces_to_delete}")

    print("Stopping any remaining iperf3 processes in target namespaces...")
    for ns in namespaces_to_delete:
         ns_check_cmd = f"sudo ip netns list | grep '^{re.escape(ns)} '"
         ns_exists_output = run_cmd(ns_check_cmd, shell=True, check=False, capture_output=True, suppress_print=True)
         ns_exists = isinstance(ns_exists_output, str) and ns_exists_output.strip()
         if ns_exists:
              run_cmd(f"sudo ip netns exec {ns} pkill iperf3", check=False, suppress_print=True)

    print("Deleting namespaces...")
    for ns in reversed(namespaces_to_delete):
        print(f"Attempting to delete namespace: {ns}")
        run_cmd(f"sudo ip netns del {ns}", check=False)

    CREATED_NAMESPACES.clear()
    CREATED_VETH_PAIRS_BASE.clear()
    NETWORK_CONFIGURATION.clear()
    print("Cleanup attempt complete.")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create, test, update, or cleanup a 3-layer network topology.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--create", action="store_true", help="Create the network topology.")
    action_group.add_argument("--cleanup", action="store_true", help="Perform cleanup for the specified topology size.")
    action_group.add_argument("--update-bw", nargs=3, metavar=('DEVICE1', 'DEVICE2', 'BANDWIDTH'),
                              help="Update bandwidth between two existing devices (e.g., core agg1 50mbit).")
    action_group.add_argument("--run-tests", action="store_true", help="Run bandwidth tests on an existing network.")

    parser.add_argument("-a", "--aggregates", type=int, help="Number of aggregate routers (Required for create/cleanup/run-tests).")
    parser.add_argument("-n", "--nodes", type=int, help="Number of end nodes PER aggregate router (Required for create/cleanup/run-tests).")

    parser.add_argument("--bw-core", default="100mbit", help="Core-aggregate bandwidth (Used by create/run-tests). Use '' for none.")
    parser.add_argument("--bw-node", default="10mbit", help="Aggregate-node bandwidth (Used by create/run-tests). Use '' for none.")

    args = parser.parse_args()

    # Check root privileges
    try:
         user_id = subprocess.run("id -u", shell=True, capture_output=True, text=True, check=True).stdout.strip()
         if user_id != "0":
             print("Error: This script requires root privileges. Please run with sudo.", file=sys.stderr)
             sys.exit(1)
    except Exception as e:
         print(f"Error checking user ID. Cannot continue. ({e})", file=sys.stderr)
         sys.exit(1)

    # --- Action Dispatch ---

    if args.update_bw:
        dev1, dev2, bw = args.update_bw
        update_bandwidth(dev1, dev2, bw)

    elif args.cleanup:
        if args.aggregates is None or args.nodes is None:
             parser.error("--cleanup requires -a/--aggregates and -n/--nodes to specify topology.")
        if args.aggregates <= 0 or args.nodes < 0:
             print("Error: Aggregates must be positive, nodes non-negative for cleanup.", file=sys.stderr)
             sys.exit(1)
        cleanup_network(args.aggregates, args.nodes)

    elif args.run_tests:
        if args.aggregates is None or args.nodes is None:
             parser.error("--run-tests requires -a/--aggregates and -n/--nodes to specify topology.")
        if args.aggregates <= 0 or args.nodes < 0:
             print("Error: Aggregates must be positive, nodes non-negative for testing.", file=sys.stderr)
             sys.exit(1)
        test_bandwidth(args.aggregates, args.nodes, args.bw_core, args.bw_node)

    elif args.create:
        if args.aggregates is None or args.nodes is None:
            parser.error("--create requires -a/--aggregates and -n/--nodes to specify topology.")
        if args.aggregates <= 0 or args.nodes < 0:
             print("Error: Aggregates must be positive, nodes non-negative for create.", file=sys.stderr)
             sys.exit(1)
        if args.nodes == 0:
            print("Warning: Creating network with 0 end nodes per aggregate.")

        try:
            create_network(args.aggregates, args.nodes, args.bw_core, args.bw_node)
        except Exception:
             sys.exit(1) # create_network handles cleanup on error

        # Print Final Network Info only on successful creation run
        print("\n" + "="*40)
        print(" Final Network Configuration")
        print("="*40)
        if not NETWORK_CONFIGURATION:
             print("No network configuration recorded.")
        else:
            if 'core' in NETWORK_CONFIGURATION:
                print("\nNamespace: core")
                for info in sorted(NETWORK_CONFIGURATION['core']): print(f"  {info}")
            agg_names = sorted([k for k in NETWORK_CONFIGURATION if k.startswith('agg')])
            for agg_name in agg_names:
                print(f"\nNamespace: {agg_name}")
                for info in sorted(NETWORK_CONFIGURATION[agg_name]): print(f"  {info}")
            end_node_names = sorted([k for k in NETWORK_CONFIGURATION if k.startswith('end')])
            for node_name in end_node_names:
                 print(f"\nNamespace: {node_name}")
                 for info in sorted(NETWORK_CONFIGURATION[node_name]): print(f"  {info}")
        print("="*40)

        # Instructions
        print("\n---")
        print("Network creation complete.")
        print("To run bandwidth tests on this network, use:")
        print(f"sudo {sys.argv[0]} --run-tests -a {args.aggregates} -n {args.nodes} --bw-core '{args.bw_core}' --bw-node '{args.bw_node}'")
        print("To cleanup the created network resources, run:")
        print(f"sudo {sys.argv[0]} --cleanup -a {args.aggregates} -n {args.nodes}")
        print("To update bandwidth (example):")
        print(f"sudo {sys.argv[0]} --update-bw core agg1 50mbit")
        print("---")

    else:
         parser.print_help()
         sys.exit(1)
