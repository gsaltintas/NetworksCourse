"""A random policy that selects the precisions randomly"""

import random
from typing import List

from dps.utils.precision import Precision

from .base import PrecisionPolicy


class ConstantPolicy(PrecisionPolicy):
    def __init__(
        self,
        base_precision: Precision,
    ):
        self.base_precision = base_precision

    def select_precision(self, network_stats, model_context, src_dst_pair):
        return self.base_precision
