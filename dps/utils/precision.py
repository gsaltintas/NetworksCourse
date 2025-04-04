from enum import Enum

import torch


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BFLOAT16 = "bfloat16"
    FP8 = "fp8"
    INT4 = "int4"
    INT2 = "int2"


class LinkType(Enum):
    INTRA = "intra"
    INTER = "inter"


PENALTIES = {
    Precision.FP32: 0.0,  # No penalty for full precision
    Precision.FP16: -0.01,  # Minimal penalty
    Precision.BFLOAT16: -0.01,  # Minimal penalty
    Precision.FP8: -0.05,  # Modest penalty
    Precision.INT4: -0.10,  # Significant penalty
    Precision.INT2: -0.20,  # Severe penalty
}


def map_to_dtype(precision: Precision):
    dtype = torch.float32
    if precision == Precision.FP16:
        dtype = torch.float16
    elif precision == Precision.BFLOAT16:
        dtype = torch.bfloat16
    elif precision == Precision.FP8:
        dtype = torch.int8
    elif precision == Precision.INT4:
        dtype = torch.int4
    elif precision == Precision.INT2:
        dtype = torch.int2
    return dtype
