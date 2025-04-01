import torch
import torch.distributed as dist

from dps.network.monitor import NetworkMonitor
from dps.scheduler import DynamicPrecisionScheduler
from dps.utils.precision import Precision, map_to_dtype


def dynamic_precision_allreduce(
    send, recv, tp_group, scheduler, tensor_info=None, is_backward=False
):
    """
    AllReduce with dynamic precision based on network conditions

    Args:
        send: Tensor to send
        recv: Tensor to receive result
        tp_group: Process group or device mesh
        scheduler: DynamicPrecisionScheduler instance
        tensor_info: Information about the tensor being communicated
        is_backward: Whether this is during backward pass
    """
    # Extract process group if needed
    if hasattr(tp_group, "get_process_group"):
        process_group = tp_group.get_process_group()
    else:
        process_group = tp_group

    # Get rank and world size
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    # Skip if singleton
    if world_size <= 1:
        recv.copy_(send)
        return

    # Get source and destination devices
    # In a standard TP scenario, communication happens with neighbors
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    # Prepare default tensor info if not provided
    if tensor_info is None:
        tensor_info = {
            "shape": list(send.shape),
            "size_bytes": send.element_size() * send.nelement(),
            "layer_type": "unknown",
        }

    # Determine precision for communication
    communication_dtype = scheduler.get_precision(
        src_device=rank,
        dst_device=next_rank,
        tensor_info=tensor_info,
        is_backward=is_backward,
        return_dtype=True,
    )

    # Convert to target precision for communication
    if communication_dtype != send.dtype:
        send_converted = send.to(communication_dtype)
    else:
        send_converted = send.clone()

    # Use standard all_reduce for reliability
    dist.all_reduce(send_converted, op=dist.ReduceOp.SUM, group=process_group)

    # Convert back to original precision if needed
    if communication_dtype != send.dtype:
        recv.copy_(send_converted.to(send.dtype))
    else:
        recv.copy_(send_converted)


class DynamicPrecisionTPLinearLayer(torch.nn.Module):
    """Tensor-Parallel Linear Layer with dynamic precision communication"""

    def __init__(
        self, linear_layer, tp_group, tp_size, tp_rank, split_dim=0, scheduler=None
    ):
        super().__init__()
        self.tp_group = tp_group
        self.split_dim = split_dim
        self.scheduler = scheduler
        self.layer_name = "unknown"  # Will be set during model parallelization

        # Get original dimensions
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # Shard the weight
        if split_dim == 0:  # Column-wise split (output dimension)
            shard_size = self.out_features // tp_size
            start_idx = tp_rank * shard_size
            end_idx = (
                start_idx + shard_size if tp_rank < tp_size - 1 else self.out_features
            )

            # Create weight parameter with sharded dimensions
            weight = linear_layer.weight[start_idx:end_idx, :].contiguous()
            self.weight = torch.nn.Parameter(weight)

            # Shard bias if present
            if linear_layer.bias is not None:
                bias = linear_layer.bias[start_idx:end_idx].contiguous()
                self.bias = torch.nn.Parameter(bias)
            else:
                self.bias = None

            # Update output dimension
            self.out_features = end_idx - start_idx

        elif split_dim == 1:  # Row-wise split (input dimension)
            shard_size = self.in_features // tp_size
            start_idx = tp_rank * shard_size
            end_idx = (
                start_idx + shard_size if tp_rank < tp_size - 1 else self.in_features
            )

            # Create weight parameter with sharded dimensions
            weight = linear_layer.weight[:, start_idx:end_idx].contiguous()
            self.weight = torch.nn.Parameter(weight)

            # Keep bias intact for row-wise
            if linear_layer.bias is not None:
                self.bias = torch.nn.Parameter(linear_layer.bias.clone().contiguous())
            else:
                self.bias = None

            # Update input dimension
            self.in_features = end_idx - start_idx

    def forward(self, x):
        """Forward pass with dynamic precision all-reduce for row-wise parallelism"""
        # Regular linear operation
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        # For row-wise parallelism, we need all-reduce
        if self.split_dim == 1 and self.scheduler is not None:
            tensor_info = {
                "name": self.layer_name,
                "shape": list(output.shape),
                "size_bytes": output.element_size() * output.nelement(),
                "layer_type": "linear_output",
            }

            # Create result tensor
            result = torch.empty_like(output)

            # Use dynamic precision all-reduce
            dynamic_precision_allreduce(
                output,
                result,
                self.tp_group,
                self.scheduler,
                tensor_info=tensor_info,
                is_backward=False,
            )
            return result
        elif self.split_dim == 1:
            # If no scheduler, use standard all_reduce
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)

        return output

    def set_name(self, name):
        """Set the layer name for better logging and tracking"""
        self.layer_name = name
        return self


def apply_dynamic_precision_tp(model, tp_group, tp_size, tp_rank, scheduler=None):
    """Apply tensor parallelism with dynamic precision to a transformer model"""
    # Track modules that require parallelization
    for name, module in model.named_modules():
        # Apply to linear layers based on their role
        if isinstance(module, torch.nn.Linear):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            attr_name = name.split(".")[-1]

            # Determine the parallelism strategy based on the module's role
            if any(
                pattern in name
                for pattern in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "query",
                    "key",
                    "value",
                    "up_proj",
                    "gate_proj",
                ]
            ):
                # Column-wise splitting (output dim)
                tp_layer = DynamicPrecisionTPLinearLayer(
                    module, tp_group, tp_size, tp_rank, split_dim=0, scheduler=scheduler
                ).set_name(name)
                setattr(parent, attr_name, tp_layer)

            elif any(
                pattern in name
                for pattern in ["o_proj", "out_proj", "output", "down_proj"]
            ):
                # Row-wise splitting (input dim)
                tp_layer = DynamicPrecisionTPLinearLayer(
                    module, tp_group, tp_size, tp_rank, split_dim=1, scheduler=scheduler
                ).set_name(name)
                setattr(parent, attr_name, tp_layer)

    return model


# Hook for backward pass to use dynamic precision
class DynamicPrecisionBackwardHook:
    def __init__(self, scheduler, tp_group):
        self.scheduler = scheduler
        self.tp_group = tp_group

    def __call__(self, grad):
        if grad is None:
            return None

        tensor_info = {
            "name": "grad_tensor",
            "shape": list(grad.shape),
            "size_bytes": grad.element_size() * grad.nelement(),
            "layer_type": "gradient",
        }

        result = torch.empty_like(grad)
        dynamic_precision_allreduce(
            grad,
            result,
            self.tp_group,
            self.scheduler,
            tensor_info=tensor_info,
            is_backward=True,
        )
        return result


def register_backward_hooks(model, scheduler, tp_group):
    """Register backward hooks for dynamic precision in backward pass"""
    hooks = []

    # Attach hooks to parameters that need all-reduce in backward
    for name, module in model.named_modules():
        if isinstance(module, DynamicPrecisionTPLinearLayer) and module.split_dim == 1:
            # For row-wise parallelism, we need to handle gradients in backward pass
            hook = module.weight.register_hook(
                DynamicPrecisionBackwardHook(scheduler, tp_group)
            )
            hooks.append(hook)

            if module.bias is not None:
                hook = module.bias.register_hook(
                    DynamicPrecisionBackwardHook(scheduler, tp_group)
                )
                hooks.append(hook)

    return hooks
