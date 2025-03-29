from typing import Dict

from dps.utils.precision import Precision

from .base import PrecisionPolicy

""" A heuristic based basic policy that based on a predefined threshold determines the precision to use """


class NetworkAwareHeuristicPolicy(PrecisionPolicy):
    def __init__(
        self, high_congestion_threshold: float, extreme_congestion_threshold: float
    ):
        # thresholds define when to switch precisions based on network conditions
        self.high_congestion_threshold = high_congestion_threshold
        self.extreme_congestion_threshold = extreme_congestion_threshold

    def select_precision(self, network_stats, model_context, src_dst_pair):
        # Determine precision based on network conditions
        congestion_level = network_stats["congestion"]

        # Consider if this is backward pass (may need higher precision)
        if model_context["is_backward"]:
            # Use more conservative thresholds for backward pass
            if congestion_level > self.high_congestion_threshold:
                return Precision.FP16
            return Precision.FP32
        else:
            # Forward pass can be more aggressive with low precision
            if congestion_level > self.extreme_congestion_threshold:
                return Precision.FP8
            elif congestion_level > self.high_congestion_threshold:
                return Precision.FP16
            return Precision.FP32
