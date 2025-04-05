#!/usr/bin/env python3
"""
Network Monitoring API Service
Provides REST API access to RTT and flow data collected by the monitoring system
"""

import csv
import datetime
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
METRICS_FILE = DATA_DIR / "metrics.csv"
CONFIG_DIR = BASE_DIR / "configs"
HOSTS_FILE = CONFIG_DIR / "target-hosts.txt"


# Initialize metrics file if it doesn't exist
def init_metrics_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w") as f:
            f.write("timestamp,source,destination,measurement_type,value,unit,status\n")


# Load host list from file
def load_hosts():
    if not os.path.exists(HOSTS_FILE):
        os.makedirs(os.path.dirname(HOSTS_FILE), exist_ok=True)
        with open(HOSTS_FILE, "w") as f:
            f.write("# List of hosts to monitor (one per line)\n")
            f.write("localhost\n")
        return ["localhost"]

    hosts = []
    with open(HOSTS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                hosts.append(line)
    return hosts


# Add a host to monitoring
def add_host(hostname):
    hosts = load_hosts()
    if hostname not in hosts:
        with open(HOSTS_FILE, "a") as f:
            f.write(f"{hostname}\n")
        return True
    return False


# Remove a host from monitoring
def remove_host(hostname):
    hosts = load_hosts()
    if hostname in hosts:
        hosts.remove(hostname)
        with open(HOSTS_FILE, "w") as f:
            f.write("# List of hosts to monitor (one per line)\n")
            for host in hosts:
                f.write(f"{host}\n")
        return True
    return False


# Measure RTT using ping
def measure_ping_rtt(destination):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source = socket.gethostname()
        result = subprocess.run(
            ["ping", "-c", "5", "-q", destination],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            # Extract average RTT
            for line in result.stdout.splitlines():
                if "min/avg/max" in line:
                    parts = line.split("=")[1].strip().split("/")
                    rtt = float(parts[1])  # avg value
                    with open(METRICS_FILE, "a") as f:
                        f.write(
                            f"{timestamp},{source},{destination},ping_rtt,{rtt},ms,success\n"
                        )
                    return {
                        "source": source,
                        "destination": destination,
                        "rtt": rtt,
                        "status": "success",
                    }

        # If we get here, ping failed or parsing failed
        with open(METRICS_FILE, "a") as f:
            f.write(f"{timestamp},{source},{destination},ping_rtt,null,ms,failed\n")
        return {
            "source": source,
            "destination": destination,
            "rtt": None,
            "status": "failed",
        }

    except Exception as e:
        return {
            "source": source,
            "destination": destination,
            "rtt": None,
            "status": f"error: {str(e)}",
        }


# Measure TCP RTT
def measure_tcp_rtt(destination, port=80):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source = socket.gethostname()

        start_time = time.time()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex((destination, port))
        end_time = time.time()
        s.close()

        if result == 0:
            rtt = (end_time - start_time) * 1000  # Convert to milliseconds
            with open(METRICS_FILE, "a") as f:
                f.write(
                    f"{timestamp},{source},{destination},tcp_rtt_port{port},{rtt},ms,success\n"
                )
            return {
                "source": source,
                "destination": destination,
                "port": port,
                "rtt": rtt,
                "status": "success",
            }
        else:
            with open(METRICS_FILE, "a") as f:
                f.write(
                    f"{timestamp},{source},{destination},tcp_rtt_port{port},null,ms,failed\n"
                )
            return {
                "source": source,
                "destination": destination,
                "port": port,
                "rtt": None,
                "status": "failed",
            }

    except Exception as e:
        return {
            "source": source,
            "destination": destination,
            "port": port,
            "rtt": None,
            "status": f"error: {str(e)}",
        }


# Collect basic flow data
def collect_flow_data():
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source = socket.gethostname()

        # Get established connections using netstat
        result = subprocess.run(["netstat", "-tn"], capture_output=True, text=True)

        connections = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and "ESTABLISHED" in line:
                remote_addr = parts[4].split(":")[0]  # Extract IP without port
                connections[remote_addr] = connections.get(remote_addr, 0) + 1

        # Save to metrics file
        for dest, count in connections.items():
            with open(METRICS_FILE, "a") as f:
                f.write(
                    f"{timestamp},{source},{dest},active_connections,{count},count,success\n"
                )

        return {"source": source, "timestamp": timestamp, "connections": connections}

    except Exception as e:
        return {"status": f"error: {str(e)}"}


# Background monitoring thread
def background_monitor():
    while True:
        try:
            hosts = load_hosts()
            # Collect flow data
            collect_flow_data()

            # Measure RTT to each host
            for host in hosts:
                measure_ping_rtt(host)
                measure_tcp_rtt(host, 80)  # Web port
                measure_tcp_rtt(host, 22)  # SSH port

            # Sleep for 5 minutes
            time.sleep(300)
        except Exception as e:
            print(f"Error in background monitoring: {e}")
            time.sleep(60)  # Sleep shorter time on error


# API Routes
@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(
        {
            "status": "running",
            "timestamp": datetime.datetime.now().isoformat(),
            "monitored_hosts": load_hosts(),
            "data_file": METRICS_FILE.as_posix(),
        }
    )


@app.route("/api/hosts", methods=["GET"])
def get_hosts():
    return jsonify({"hosts": load_hosts()})


@app.route("/api/hosts", methods=["POST"])
def post_host():
    data = request.json
    if not data or "hostname" not in data:
        return jsonify({"error": "hostname field required"}), 400

    hostname = data["hostname"]
    result = add_host(hostname)

    if result:
        return jsonify(
            {"status": "success", "message": f"Added {hostname} to monitoring"}
        )
    else:
        return jsonify(
            {"status": "info", "message": f"{hostname} is already being monitored"}
        )


@app.route("/api/hosts/<hostname>", methods=["DELETE"])
def delete_host(hostname):
    result = remove_host(hostname)

    if result:
        return jsonify(
            {"status": "success", "message": f"Removed {hostname} from monitoring"}
        )
    else:
        return jsonify(
            {"status": "error", "message": f"{hostname} is not in the monitoring list"}
        )


@app.route("/api/ping/<hostname>", methods=["GET"])
def ping_host(hostname):
    result = measure_ping_rtt(hostname)
    return jsonify(result)


@app.route("/api/tcp/<hostname>/<int:port>", methods=["GET"])
def tcp_rtt(hostname, port):
    result = measure_tcp_rtt(hostname, port)
    return jsonify(result)


@app.route("/api/flows", methods=["GET"])
def get_flows():
    result = collect_flow_data()
    return jsonify(result)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    try:
        # Optional query parameters
        host = request.args.get("host")
        metric_type = request.args.get("type")
        hours = request.args.get("hours", default=24, type=int)

        # Time filter
        since = datetime.datetime.now() - datetime.timedelta(hours=hours)
        since_str = since.strftime("%Y-%m-%d %H:%M:%S")

        # Read metrics file into pandas DataFrame
        df = pd.read_csv(METRICS_FILE)

        # Convert timestamp column to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Apply filters
        df = df[df["timestamp"] >= since_str]

        if host:
            df = df[df["destination"] == host]

        if metric_type:
            df = df[df["measurement_type"] == metric_type]

        # Convert back to CSV format for API response
        records = df.to_dict("records")

        return jsonify(
            {
                "metrics": records,
                "count": len(records),
                "filters": {"since": since_str, "host": host, "type": metric_type},
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Initialize metrics file
    init_metrics_file()

    # Start background monitoring in a separate thread
    monitor_thread = threading.Thread(target=background_monitor, daemon=True)
    monitor_thread.start()

    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    hostnames = (
        subprocess.check_output(f"scontrol show hostnames {nodelist}", shell=True)
        .decode()
        .splitlines()
    )
    MASTER_ADDR = hostnames[0]
    MASTER_PORT = int(os.environ.get("SLURM_JOB_ID")) % 16384 + 49152
    print(f"Starting on {MASTER_ADDR} at port {MASTER_PORT}.")

    # Run the Flask app
    app.run(host=MASTER_ADDR, port=MASTER_PORT, debug=False)
    # app.run(host='0.0.0.0', port=5000, debug=False)    app.run(host='0.0.0.0', port=5000, debug=False)    # app.run(host='0.0.0.0', port=5000, debug=False)    app.run(host='0.0.0.0', port=5000, debug=False)
