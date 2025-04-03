import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union


@dataclass
class Config:
    master_addr: str = field(default="localhost")
    master_port: str = field(default="1111")
    # Basic settings
    model_name: str = field(
        default="resnet20", metadata={"help": "Base model used for generation"}
    )
    from_scratch: bool = field(
        default=False, metadata={"help": "Should we train the model from scratch?"}
    )
    dataset: str = field(
        default="mnist", metadata={"help": "Dataset to use for training"}
    )
    dataset_config: str | None = field(
        default=None, metadata={"help": "The dataset config to use for training."}
    )
    use_wandb: bool = field(
        default=False, metadata={"help": "Whether to use Weights & Biases for logging"}
    )
    wandb_project: str = field(default="dps", metadata={"help": "W&B project name"})
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "W&B entity name"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    save_dir: Path = field(
        default=Path("./results"), metadata={"help": "Directory to save results"}
    )
    resume_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to resume training from"}
    )
    output_dir: Optional[str] = field(
        default="./results", metadata={"help": "Directory to log training runs to"}
    )
    experiment_name: str = field(
        default="dps_default", metadata={"help": "Name for this experiment"}
    )
    debug: bool = field(default=False, metadata={"help": "Enable debug mode"})
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = field(
        default="INFO", metadata={"help": "Logging level"}
    )
    num_gpus: int = field(
        default=4, metadata={"help": "Total number of GPUs available to the job."}
    )
    # Network settings
    network_topology: Literal["flat", "tree", "fattree", "torus"] = field(
        default="fattree", metadata={"help": "Network topology type"}
    )
    num_devices: int = field(
        default=4, metadata={"help": "Total number of devices in the network"}
    )
    # todo: these are not used yet
    bandwidth_intra_node: float = field(
        default=100.0,
        metadata={"help": "Bandwidth between devices in the same node (Gbps)"},
    )
    bandwidth_inter_node: float = field(
        default=25.0, metadata={"help": "Bandwidth between different nodes (Gbps)"}
    )
    network_simulator: Literal["ns3", "simple"] = field(
        default="simple", metadata={"help": "Network simulator to use (ns3 or simple)"}
    )
    congestion_pattern: Literal["random", "periodic", "bursty"] = field(
        default="bursty", metadata={"help": "Pattern of network congestion to simulate"}
    )
    high_congestion_threshold: float = field(default=0.8)
    extreme_congestion_threshold: float = field(default=0.9)

    # Model settings
    model_type: Literal["cnn", "transformer", "mlp"] = field(
        default="transformer", metadata={"help": "Type of model architecture"}
    )
    model_size: Literal["tiny", "small", "medium", "large"] = field(
        default="medium", metadata={"help": "Size of the model"}
    )

    # Precision settings
    precision_policy: Literal["random", "constant", "heuristic", "transition", "rl"] = field(
        default="constant", metadata={"help": "Precision selection policy to use"}
    )
    precision_formats: str = field(
        default="FP32,FP16,FP8,INT4,INT2",
        metadata={"help": "Comma-separated list of available precision formats"},
    )
    high_congestion_threshold: float = field(
        default=0.7, metadata={"help": "Threshold for high congestion"}
    )
    medium_congestion_threshold: float = field(
        default=0.4, metadata={"help": "Threshold for medium congestion"}
    )
    backward_precision_boost: float = field(
        default=1.5,
        metadata={"help": "Factor to increase precision importance in backward pass"},
    )
    eval_dtype: Literal["bfloat16", "float16", "float32"] = field(
        default="bfloat16",
        metadata={"help": "Floating point number format used for evaluation"},
    )
    constant_dtype: Literal["bfloat16", "float16", "float32"] = field(
        default="bfloat16",
        metadata={
            "help": "Floating point number format used for constant policy, only takes effect if policy == constant."
        },
    )

    # Training settings
    seed: int = field(default=42)
    batch_size: int = field(default=32, metadata={"help": "Training batch size"})
    # Todo: change to num_steps
    num_train_steps: int = field(
        default=1000, metadata={"help": "Number of training steps"}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio"})
    lr_scheduler: str = field(default="linear")
    max_grad_norm: float = field(
        default=2.0, metadata={"help": "Maximum norm for gradient clipping"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Grad. accim. steps"}
    )
    optimizer: Literal["adam", "sgd"] = field(
        default="adam", metadata={"help": "Optimizer to use"}
    )
    parallelism: Literal["data", "model", "tensor", "pipeline", "hybrid"] = field(
        default="tensor", metadata={"help": "Type of parallelism to use"}
    )
    tensor_parallel_size: int = field(
        default=4, metadata={"help": "Number of devices for tensor parallelism"}
    )
    data_parallel_size: int = field(
        default=1, metadata={"help": "Number of devices for data parallelism"}
    )
    reward_latency_weight: float = field(
        default=0.7, metadata={"help": "Weight for latency in reward function"}
    )
    reward_accuracy_weight: float = field(
        default=0.3, metadata={"help": "Weight for accuracy in reward function"}
    )
    num_eval_samples: int = field(
        default=-1, metadata={"help": "Pass -1 to evaluate on all samples."}
    )

    def __post_init__(self):
        if self.wandb_entity is None:
            self.wandb_entity = os.environ.get("WANDB_ENTITY")

        # Convert string paths to Path objects
        self.save_dir = Path(self.save_dir)
        if self.resume_dir:
            self.resume_dir = Path(self.resume_dir)

        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique experiment path including timestamp
        if self.experiment_name == "dps_default":
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"dps_{self.model_name}_{timestamp}"

        self.experiment_dir = self.save_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Convert string list to actual list
        self.precision_formats = self.precision_formats.split(",")

        self.text_column_name = "text"
        self.max_seq_length = 512
