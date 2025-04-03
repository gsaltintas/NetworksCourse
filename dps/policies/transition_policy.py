"""A random policy that selects the precisions randomly"""

import random
from typing import List

from dps.utils.precision import Precision
import numpy as np

from .base import PrecisionPolicy


class TransitionPolicy(PrecisionPolicy):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.current_precision = Precision.FP32
        self.precision_mapping = {
            Precision.FP32: 0,
            Precision.FP16: 1,
            Precision.BFLOAT16: 2,
            Precision.FP8: 3,
        }
        self.rev_lut = {v: k for k, v in self.precision_mapping.items()}
        self.transition_matrix = np.array([
            [70, 18, 18, 2],
            [25, 65, 5, 5],
            [25, 5, 64, 5],
            [10, 40, 40, 10],
        ])

    def select_precision(self, network_stats, model_context, src_dst_pair):
        state = self.transition_matrix[self.precision_mapping[self.current_precision]]
        precision = self.rng.choices(range(len(state)), weights=state, k=1)[0]
        self.current_precision = self.rev_lut[precision]
        return self.current_precision
