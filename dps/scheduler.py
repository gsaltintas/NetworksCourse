from typing import Literal

import torch

from dps.network.monitor import NetworkMonitor, Topology
from dps.policies.constant_policy import ConstantPolicy
from dps.policies.network_aware_policy import NetworkAwareHeuristicPolicy
from dps.policies.random_policy import RandomPolicy
from dps.policies.transition_policy import TransitionPolicy
from dps.utils import logs
from dps.utils.precision import Precision, map_to_dtype

logger = logs.get_logger("dps")


class DynamicPrecisionScheduler:
    def __init__(
        self,
        config,
        model_info,
        total_steps: int,
    ):
        self.config = config
        self.setup_policy(
            config.precision_policy, available_precisions=config.precision_formats
        )
        self.setup_network_monitor()
        self.model_info = model_info
        self.training_step = 0
        self.total_steps = total_steps

    def setup_policy(
        self,
        policy: Literal["random", "constant", "rl", "heuristic", "transition"],
        **kwargs,
    ):
        dps_policy = None
        if policy == "random":
            dps_policy = RandomPolicy(
                seed=self.config.seed,
                available_precisions=kwargs.get("available_precisions"),
            )
        elif policy == "constant":
            dps_policy = ConstantPolicy(self.config.constant_dtype)
        elif policy == "heuristic":
            dps_policy = NetworkAwareHeuristicPolicy(
                congestion_column=self.config.congestion_column,
                high_congestion_threshold=self.config.high_congestion_threshold,
                extreme_congestion_threshold=self.config.extreme_congestion_threshold,
            )
        elif policy == "transition":
            dps_policy = TransitionPolicy()
        elif policy == "rl":
            raise NotImplementedError("RL is under development")
        else:
            raise NotImplementedError(f"Invalid policy value {policy}.")
        logger.info("Making precision decisions based on: %s", dps_policy)
        self.policy = dps_policy

    def setup_network_monitor(
        self,
        topology: Topology = Topology.FAT_TREE,
    ):
        self.network_monitor = NetworkMonitor(
            topology=topology,
            is_offline=self.config.network_mode == "offline",
            offline_network_file=self.config.offline_network_file,
        )

    def get_precision(
        self,
        src_device,
        dst_device,
        tensor_info,
        is_backward=False,
        return_dtype: bool = True,
    ):
        """
        Determine the precision to use for communication.

        Args:
            src_device: Source device ID
            dst_device: Destination device ID
            tensor_info: Dict containing metadata about the tensor being communicated
                         (e.g., layer_name, layer_type, tensor_shape)
            is_backward: Whether this is during backward pass

        Returns:
            Precision enum (e.g., FP32, FP16, FP8)
        """
        # Collect relevant information
        self.network_monitor.update()
        network_stats = self.network_monitor.get_link_stats(src_device, dst_device)
        model_context = {
            "tensor_info": tensor_info,
            "is_backward": is_backward,
            "training_progress": self.training_step / self.total_steps,
        }

        # Get decision from policy
        precision = self.policy.select_precision(
            network_stats=network_stats,
            model_context=model_context,
            src_dst_pair=(src_device, dst_device),
        )
        logger.debug("Selected Precision: %s", precision)

        if return_dtype:
            return map_to_dtype(precision)
        return precision

    def update_step(self, update: int = 1):
        self.training_step += update
