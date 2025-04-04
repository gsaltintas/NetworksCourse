import time
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd


class Topology(Enum):
    FAT_TREE = "fat_tree"


class NetworkMonitorOld:
    def __init__(
        self,
        topology: Topology = Topology.FAT_TREE,
    ):
        self.topology: Topology = topology
        self.stats = {}  # Track per-link statistics
        self.last_update_time = time.time()

    def get_link_stats(self, src_device, dst_device):
        """
        Get current statistics for link between src and dst.

        Returns:
            Dict with metrics like:
            - utilization: Current link utilization (0-1)
            - congestion: Congestion level (e.g., ECN marks)
            - bandwidth: Available bandwidth (MB/s)
            - latency: Current latency (ms)
        """
        link_key = f"{src_device}-{dst_device}"
        if link_key not in self.stats:
            # Return default values if no measurements exist
            return {
                "utilization": 0.0,
                "congestion": 0.0,
                "bandwidth": 10000.0,  # 10 GB/s default
                "latency": 0.1,  # 0.1 ms default
                "last_updated": self.last_update_time,
            }
        return self.stats[link_key]

    def update(self, new_measurements):
        """Update monitored statistics with new data

        Args:
            new_measurements: Dict mapping link IDs to their metrics
        """
        self.last_update_time = time.time()
        for link_key, metrics in new_measurements.items():
            if link_key in self.stats:
                self.stats[link_key].update(metrics)
                self.stats[link_key]["last_updated"] = self.last_update_time
            else:
                metrics["last_updated"] = self.last_update_time
                self.stats[link_key] = metrics


class NetworkMonitor:
    def __init__(
        self,
        topology: Topology = Topology.FAT_TREE,
        is_offline: bool = False,
        offline_network_file: Optional[Path] = None,
    ):
        self.topology: Topology = topology
        self.stats = {}  # Track per-link statistics
        self.last_update_time = time.time()
        self.is_offline = is_offline
        if self.is_offline:
            self.offline_network_file = offline_network_file
            self.observations = pd.read_csv(offline_network_file)
            self.last_update_time = 0
            self.n_observations = len(self.observations)

    def get_link_stats(self, src_device, dst_device):
        """
        Get current statistics for link between src and dst.

        Returns:
            Dict with metrics like:
            - utilization: Current link utilization (0-1)
            - congestion: Congestion level (e.g., ECN marks)
            - bandwidth: Available bandwidth (MB/s)
            - latency: Current latency (ms)
        """
        link_key = f"{src_device}-{dst_device}"
        if self.is_offline:
            current_row = self.observations.iloc[self.last_update_time]
            return {
                "flows": current_row["Ongoing Flows"],
                "rtt": current_row["RTT (ms)"],
                "bw-core": current_row["bw-core (mbit)"],
                "bw-node": current_row["bw-node (mbit)"],  # local
                "hops": current_row["hops"],
            }

        ## not implemented anyways
        if link_key not in self.stats:
            # Return default values if no measurements exist
            return {
                "utilization": 0.0,
                "congestion": 0.0,
                "bandwidth": 10000.0,  # 10 GB/s default
                "latency": 0.1,  # 0.1 ms default
                "last_updated": self.last_update_time,
            }
        return self.stats[link_key]

    def update(self):
        """Update monitored statistics with new data

        Args:
            new_measurements: Dict mapping link IDs to their metrics
        """
        if self.is_offline:
            # roll to the beginning
            self.last_update_time = (self.last_update_time + 1) % self.n_observations
        else:
            self.last_update_time = time.time()
        # for link_key, metrics in new_measurements.items():
        #     if link_key in self.stats:
        #         self.stats[link_key].update(metrics)
        #         self.stats[link_key]["last_updated"] = self.last_update_time
        #     else:
        #         metrics["last_updated"] = self.last_update_time
        #         self.stats[link_key] = metrics
