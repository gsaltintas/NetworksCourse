from enum import Enum


class Topology(Enum):
    FAT_TREE = "fat_tree"


class NetworkMonitor:
    def __init__(self, topology: Topology):
        self.topology: Topology = topology
        self.stats = {}  # Track per-link statistics

    def get_link_stats(self, src_device, dst_device):
        """
        Get current statistics for link between src and dst.

        Returns:
            Dict with metrics like:
            - utilization: Current link utilization (0-1)
            - congestion: Congestion level (e.g., ECN marks)
            - bandwidth: Available bandwidth
            - latency: Current latency
        """
        pass

    def update(self, new_measurements):
        """Update monitored statistics with new data"""
        pass
