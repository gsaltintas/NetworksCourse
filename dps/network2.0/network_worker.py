#!/usr/bin/env python3
"""
Example client code for the Network Monitoring API
Shows how to query and interact with the API from Python
"""

import json
import os
import subprocess
import sys
import time

import requests

# Set your API endpoint

nodelist = os.environ.get("SLURM_JOB_NODELIST")
hostnames = (
    subprocess.check_output(f"scontrol show hostnames {nodelist}", shell=True)
    .decode()
    .splitlines()
)
MASTER_ADDR = hostnames[0]
MASTER_PORT = int(os.environ.get("SLURM_JOB_ID")) % 16384 + 49152
API_URL = f"http://{MASTER_ADDR}:{MASTER_PORT}/api"


def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))


def get_api_status():
    """Check API status"""
    response = requests.get(f"{API_URL}/status")
    return response.json()


def get_monitored_hosts():
    """Get list of monitored hosts"""
    response = requests.get(f"{API_URL}/hosts")
    return response.json()


def add_host(hostname):
    """Add a host to monitoring"""
    response = requests.post(f"{API_URL}/hosts", json={"hostname": hostname})
    return response.json()


def remove_host(hostname):
    """Remove a host from monitoring"""
    response = requests.delete(f"{API_URL}/hosts/{hostname}")
    return response.json()


def measure_ping_rtt(hostname):
    """Measure RTT to a host using ping"""
    response = requests.get(f"{API_URL}/ping/{hostname}")
    return response.json()


def measure_tcp_rtt(hostname, port=80):
    """Measure RTT to a host:port using TCP"""
    response = requests.get(f"{API_URL}/tcp/{hostname}/{port}")
    return response.json()


def get_flows():
    """Get current network flows"""
    response = requests.get(f"{API_URL}/flows")
    return response.json()


def get_metrics(host=None, metric_type=None, hours=24):
    """Get historical metrics with optional filtering"""
    params = {}
    if host:
        params["host"] = host
    if metric_type:
        params["type"] = metric_type
    if hours:
        params["hours"] = hours

    response = requests.get(f"{API_URL}/metrics", params=params)
    return response.json()


def main():
    """Main function to demonstrate API usage"""
    print("Network Monitoring API Client Example")
    print("====================================")

    # Check API status
    print("\nChecking API status...")
    status = get_api_status()
    print_json(status)

    # Get list of monitored hosts
    print("\nGetting monitored hosts...")
    hosts = get_monitored_hosts()
    print_json(hosts)

    # # Add example.com if not already monitored
    # if "example.com" not in hosts.get("hosts", []):
    #     print("\nAdding example.com to monitoring...")
    #     result = add_host("example.com")
    #     print_json(result)

    # Measure ping RTT to example.com
    print("\nMeasuring ping RTT to example.com...")
    ping_result = measure_ping_rtt("example.com")
    print_json(ping_result)

    # Measure TCP RTT to example.com:80
    print("\nMeasuring TCP RTT to example.com:80...")
    tcp_result = measure_tcp_rtt("example.com", 80)
    print_json(tcp_result)

    # Get current flows
    print("\nGetting current network flows...")
    flows = get_flows()
    print_json(flows)

    # Get ping RTT metrics for the last hour
    print("\nGetting ping RTT metrics for the last hour...")
    metrics = get_metrics(metric_type="ping_rtt", hours=1)
    print(f"Found {metrics.get('count', 0)} metrics")

    # If there are metrics, print the first 3
    if metrics.get("count", 0) > 0:
        print("Latest metrics:")
        for i, metric in enumerate(metrics.get("metrics", [])[:3]):
            print_json(metric)
            if i < 2:  # Don't print separator after the last item
                print("-" * 40)


if __name__ == "__main__":
    main()
    main()
