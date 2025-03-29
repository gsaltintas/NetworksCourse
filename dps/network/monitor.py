import time
from enum import Enum


class Topology(Enum):
    FAT_TREE = "fat_tree"


class NetworkMonitor:
    def __init__(self, topology: Topology = Topology.FAT_TREE):
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
