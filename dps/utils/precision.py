from enum import Enum


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
