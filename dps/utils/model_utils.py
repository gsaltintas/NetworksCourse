from typing import Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .system import VECTOR_HF_MAPPING, Hosts, get_host


def load_tokenizer(model_name: str) -> AutoTokenizer:
    kwargs = dict()
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_model(model_name, dtype=None):
    kwargs = dict()
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        kwargs["local_files_only"] = True

    if dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model
