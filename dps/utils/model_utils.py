import logging
from typing import Literal, Optional

import contextual_logger
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .system import VECTOR_HF_MAPPING, Hosts, get_host

logger = logging.getLogger("dps")


def load_tokenizer(model_name: str) -> AutoTokenizer:
    kwargs = dict()
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        logger.info(
            "We are on the vector cluster, loading tokenizer for %s from %s",
            model_name,
            model_path,
        )
        kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_model(model_name, dtype=None, from_scratch: bool = False):
    kwargs = dict()
    model_path = model_name
    if get_host() == Hosts.vector:
        model_path = VECTOR_HF_MAPPING.get(model_name, model_path)
        logger.info(
            "We are on the vector cluster, loading weights for %s from %s",
            model_name,
            model_path,
        )
        kwargs["local_files_only"] = True

    if dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16

    if from_scratch:
        logging.info("Training the model from scratch")
        config = AutoConfig.from_pretrained(model_path)
        kwargs.pop("local_files_only")
        model = AutoModelForCausalLM.from_config(config=config, **kwargs)
    else:
        logging.info("Loading the model from pretrained at %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model
