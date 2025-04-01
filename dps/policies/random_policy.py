"""A random policy that selects the precisions randomly"""

import random
from typing import List

from dps.utils.precision import Precision

from .base import PrecisionPolicy


class RandomPolicy(PrecisionPolicy):
    def __init__(self, available_precisions: List[Precision] = None, seed=None):
        self.available_precisions = (
            available_precisions
            if available_precisions is not None
            else list(Precision)
        )
        self.rng = random.Random(seed)

    def select_precision(self, network_stats, model_context, src_dst_pair):
        # Randomly select precision regardless of input
        return self.rng.choice(self.available_precisions)
