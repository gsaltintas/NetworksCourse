#!/usr/bin/env python
# coding=utf-8

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import HfArgumentParser

from dps.utils.config import Config

logger = logging.getLogger(__name__)


def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=getattr(logging, config.log_level),
    )


def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_wandb(config):
    """Setup Weights & Biases logging if enabled"""
    if config.use_wandb:
        import wandb

        # Start a W&B run
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.experiment_name,
            config=vars(config),
        )
        return wandb
    return None


def main():
    # Parse arguments from command line
    parser = HfArgumentParser((Config,))

    if len(sys.argv) == 2:
        file = Path(sys.argv[1]).absolute()
        # If a file is provided, parse it to get arguments
        if sys.argv[1].endswith(".json"):
            (config,) = parser.parse_json_file(json_file=file)
        elif sys.argv[1].endswith(".yaml"):
            (config,) = parser.parse_yaml_file(yaml_file=file)
    else:
        # Otherwise parse command line arguments
        (config,) = parser.parse_args_into_dataclasses()

    # Setup
    setup_logging(config)
    set_seed(config.seed)
    wandb_run = setup_wandb(config)

    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Config: {config}")

    try:
        # Log configuration
        logger.info(f"Network topology: {config.network_topology}")
        logger.info(f"Model type: {config.model_type}, size: {config.model_size}")
        logger.info(f"Precision policy: {config.precision_policy}")
        logger.info(
            f"Training for {config.num_epochs} epochs with batch size {config.batch_size}"
        )

        # Placeholder for training loop
        for epoch in range(config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

            # Simulate epoch
            epoch_metrics = {
                "train_loss": 0.5 / (epoch + 1),  # Placeholder values
                "validation_accuracy": 0.7 + 0.02 * epoch,
                "communication_latency": 100 - epoch * 5,
                "avg_precision_level": 16
                - epoch,  # Starting high, decreasing as training progresses
            }

            # Log metrics
            logger.info(f"Epoch {epoch + 1} metrics: {epoch_metrics}")
            if wandb_run:
                wandb_run.log(epoch_metrics, step=epoch)

        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.exception(f"Error during experiment: {e}")
        if wandb_run:
            wandb_run.finish(exit_code=1)
        raise

    finally:
        # Cleanup
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
