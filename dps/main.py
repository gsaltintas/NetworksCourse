import logging
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import HfArgumentParser, default_data_collator, get_scheduler

from dps.utils.config import Config
from dps.utils.model_utils import load_model, load_tokenizer

logger = logging.getLogger(__name__)


def prepare_datasets(
    config,
    dp_size,
    dp_rank,
    per_device_eval_batch_size,
    per_device_train_batch_size,
    tokenizer,
):
    """Load and prepare the datasets."""
    logger.info(f"Loading dataset from {config.dataset}.")

    # Load the dataset
    datasets = load_dataset(config.dataset)
    if "test" not in datasets:
        split_dataset = datasets["train"].train_test_split(test_size=0.2, seed=42)
        datasets["train"] = split_dataset["train"]
        datasets["test"] = split_dataset["test"]
    # Preprocess the datasets
    text_column_name = config.text_column_name
    max_seq_length = config.max_seq_length

    def preprocess_function(examples):
        """Tokenize the examples and prepare them for training."""
        # Concatenate all texts
        text_column = examples[text_column_name]

        # Tokenize
        tokenized = tokenizer(
            text_column,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

        # Create labels (for causal LM, labels are the same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=[
            col
            for col in datasets["train"].column_names
            if col != config.text_column_name
        ],
        desc="Tokenizing datasets",
    )

    # Create samplers for train and eval
    # Important: for Tensor Parallelism, we shard the data based on DP rank, not global rank
    train_sampler = DistributedSampler(
        tokenized_datasets["train"],
        num_replicas=dp_size,
        rank=dp_rank,
    )

    eval_sampler = DistributedSampler(
        tokenized_datasets["validation"]
        if "validation" in tokenized_datasets
        else tokenized_datasets["test"],
        num_replicas=dp_size,
        rank=dp_rank,
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=default_data_collator,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"]
        if "validation" in tokenized_datasets
        else tokenized_datasets["test"],
        batch_size=per_device_eval_batch_size,
        sampler=eval_sampler,
        collate_fn=default_data_collator,
        pin_memory=True,
    )

    logger.info(
        f"Validation dataset size: {len(tokenized_datasets['test'])} (sharded across {dp_size} DP groups)"
    )
    logger.info(
        f"Train dataset size: {len(tokenized_datasets['train'])} (sharded across {dp_size} DP groups)"
    )

    return train_dataloader, eval_dataloader


# TODO: for some reason doesn't work
def parallelize_model_with_fsdp(model, config, local_rank, world_size, mesh):
    """Apply either tensor parallelism or FSDP, but not both"""

    # Initialize process group if not already done
    # if not dist.is_initialized():
    #     dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    # Calculate sizes
    tp_size = min(config.tensor_parallel_size, world_size)
    dp_size = min(config.data_parallel_size, world_size // tp_size)

    # Move model to device first
    model = model.to(local_rank)

    try:
        # Create a 1D mesh for tensor parallelism
        # device_mesh = dist.DeviceMesh("cuda", (tp_size,))
        # device_mesh = mesh._flatten("tp")
        device_mesh = mesh["tp"]._flatten()["tp"]
        # Define explicit parallelization plan for transformer models like LLaMA
        parallel_plan = {}

        # Track modules that require parallelization
        for name, module in model.named_modules():
            # LLaMA/transformer specific parallelization
            if isinstance(module, torch.nn.Linear):
                # QKV projections - Colwise parallelism (split output features)
                if any(
                    attn_name in name
                    for attn_name in [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "query",
                        "key",
                        "value",
                    ]
                ):
                    parallel_plan[name] = dist.tensor.parallel.ColwiseParallel()

                # Output projections - Rowwise parallelism (split input features)
                elif any(
                    out_name in name for out_name in ["o_proj", "out_proj", "output"]
                ):
                    parallel_plan[name] = dist.tensor.parallel.RowwiseParallel()

                # MLP layers
                elif any(
                    mlp_name in name
                    for mlp_name in ["up_proj", "gate_proj", "down_proj", "mlp"]
                ):
                    # Up/gate projections - Colwise (split output features)
                    if any(up_name in name for up_name in ["up_proj", "gate_proj"]):
                        parallel_plan[name] = dist.tensor.parallel.ColwiseParallel()
                    # Down projections - Rowwise (split input features)
                    elif "down_proj" in name:
                        parallel_plan[name] = dist.tensor.parallel.RowwiseParallel()

        logger.info(
            f"Applying tensor parallelism with plan for {len(parallel_plan)} modules"
        )

        # Apply tensor parallelism
        model = dist.tensor.parallel.parallelize_module(
            model, device_mesh, parallel_plan
        )
        # model = FSDP(
        #     model, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD
        # )
        model = dist.fsdp.fully_shard(model, mesh=mesh["dp"])
        logger.info("Model parallelized with tensor parallelism")

        # Return with TP rank info
        tp_rank = local_rank % tp_size
        dp_rank = local_rank // tp_size
        print(tp_rank, dp_rank)
        return model, dp_rank, tp_rank

    except Exception as e:
        raise RuntimeError(f"Error applying tensor parallelism: {e}")


def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=getattr(logging, config.log_level),
    )


def set_seed(seed, rank=0):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed + rank)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments from command line
    config: Config
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
    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    assert config.data_parallel_size * config.tensor_parallel_size == config.num_gpus, (
        f"Please check your distributed arguments: num_gpus {config.num_gpus} should be data_parallel_size {config.data_parallel_size} * tensor_parallel_size {config.tensor_parallel_size}"
    )
    # Setup
    setup_logging(config)
    set_seed(config.seed, local_rank)
    #####
    """Initialize distributed training with tensor parallelism."""
    # In a distributed environment, only log from process 0
    is_main_process = local_rank == 0

    # Calculate local ranks and groups
    tp_size = min(config.tensor_parallel_size, world_size)
    dp_size = min(config.data_parallel_size, world_size // tp_size)

    # Verify the configuration is valid
    if tp_size * dp_size > world_size:
        raise ValueError(
            f"Invalid parallel configuration: {tp_size} * {dp_size} > {world_size}"
        )

    mesh_2d = dist.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = mesh_2d["tp"]  # a submesh that connects intra-host devices
    dp_mesh = mesh_2d["dp"]  # a submesh that connects inter-host devices

    dp_rank = dp_mesh.get_local_rank()
    tp_rank = tp_mesh.get_local_rank()

    #  Set device
    device = torch.device("cuda", local_rank)
    # Initialize wandb if requested and this is the main process
    if config.use_wandb and is_main_process:
        wandb.init(
            project="networks-lm-finetuning",
            name=f"{config.model_name}-finetuning",
            config=asdict(config.config),
        )
    logger.info("Loading model and tokenizer from %s", config.model_name)
    tokenizer = load_tokenizer(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(config.model_name)
    # Apply tensor parallelism and FSDP from the same mesh
    model, dp_rank, tp_rank = parallelize_model_with_fsdp(
        model, config, local_rank, world_size, mesh_2d
    )
    logger.info("Model after parallelization %s", model)
    train_dataloader, eval_dataloader = prepare_datasets(
        config,
        dp_size,
        dp_rank,
        config.batch_size // config.tensor_parallel_size,
        config.batch_size // config.tensor_parallel_size,
        tokenizer,
    )
    total_steps = config.num_train_steps
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # TODO: check if foreach is necessary
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,  # foreach=True
    )
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    logger.info("Starting training")
    model.train()
    total_loss = 0
    step_count = 0
    # Get the number of training steps
    num_update_steps_per_epoch = (
        len(train_dataloader) // config.gradient_accumulation_steps
    )
    total_training_steps = total_steps * num_update_steps_per_epoch
    epochs = max(1, math.ceil(total_training_steps / len(train_dataloader)))
    # Setup progress bar
    progress_bar = tqdm(
        total=total_training_steps,
        disable=not is_main_process,
        desc="Training",
    )

    # Training loop
    for epoch in range(epochs):
        # Reset the sampler for each epoch
        train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            # TODO: which device
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            # TODO: modify allreduce
            # dist.all_reduce
            loss = outputs.loss

            # Normalize loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update parameters if we've accumulated enough gradients
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Log progress
                progress_bar.update(1)
                step_count += 1

                if is_main_process and config.use_wandb:
                    wandb.log(
                        {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                    )

                # Check if we've reached the total steps
                if step_count >= total_steps:
                    break

        # Evaluate at the end of each epoch
        model.eval()
        eval_loss = 0
        eval_steps = 0

        for eval_batch in eval_dataloader:
            eval_batch = {k: v.to(model.device) for k, v in eval_batch.items()}

            with torch.no_grad():
                outputs = model(**eval_batch)

            eval_loss += outputs.loss.item()
            eval_steps += 1

        eval_loss /= eval_steps

        if is_main_process:
            logger.info(f"Epoch {epoch}: Eval Loss: {eval_loss}")
            if config.use_wandb:
                wandb.log({"eval_loss": eval_loss, "epoch": epoch})

        model.train()

        # Break if we've reached total steps
        if step_count >= total_steps:
            break

    # Save final model if needed
    if is_main_process:
        # Here you would save the model
        logger.info("Training completed")


if __name__ == "__main__":
    main()
