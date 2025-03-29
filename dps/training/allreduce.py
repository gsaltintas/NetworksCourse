import torch
import torch.distributed as dist


def mesh_allreduce(send, recv, device_mesh, precision=None):
    """
    Custom all-reduce implementation with device mesh support and optional precision control.

    Args:
        send: Tensor to send
        recv: Tensor to receive result
        device_mesh: DeviceMesh object containing device information
        precision: Optional precision to use during communication ("fp16", "fp32", etc.)
    """
    # Extract mesh info
    mesh_ranks = (
        device_mesh.mesh.tolist()
        if hasattr(device_mesh, "mesh")
        else device_mesh.get_all_ranks()
    )
    rank = dist.get_rank()
    size = len(mesh_ranks)

    # Skip if running on a single device
    if size <= 1:
        recv.copy_(send)
        return

    # Convert to target precision for communication if specified
    if precision is not None:
        if precision == "fp16" and send.dtype == torch.float32:
            send_buff = send.clone().half()
            recv_buff = send.clone().half()
            accum = send.clone().half()
        else:
            send_buff = send.clone()
            recv_buff = send.clone()
            accum = send.clone()
    else:
        send_buff = send.clone()
        recv_buff = send.clone()
        accum = send.clone()

    # Find position in mesh ranks
    my_index = mesh_ranks.index(rank) if rank in mesh_ranks else 0

    # Compute left and right neighbors in the ring
    left_index = (my_index - 1) % size
    right_index = (my_index + 1) % size
    left = mesh_ranks[left_index]
    right = mesh_ranks[right_index]

    # Perform ring allreduce
    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum.add_(recv_buff)
        else:
            # Send recv_buff
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum.add_(send_buff)
        send_req.wait()

    # Convert back to original precision if needed
    if precision is not None and precision == "fp16" and send.dtype == torch.float32:
        recv.copy_(accum.float())
    else:
        recv.copy_(accum)


# Integration with tensor parallel implementation
def tp_linear_forward(x, weight, bias, tp_group, split_dim=0, precision=None):
    """
    Forward pass for tensor-parallel linear layer with custom allreduce.

    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Bias tensor or None
        tp_group: Tensor parallel group or device mesh
        split_dim: 0 for column-wise, 1 for row-wise
        precision: Precision for communication ("fp16", "fp32", etc.)
    """
    output = torch.nn.functional.linear(x, weight, bias)

    # For row-wise parallelism, we need all-reduce
    if split_dim == 1:
        result = torch.empty_like(output)
        mesh_allreduce(output, result, tp_group, precision=precision)
        return result

    return output


class TPLinearLayer(torch.nn.Module):
    def __init__(
        self, linear_layer, tp_group, tp_size, tp_rank, split_dim=0, precision=None
    ):
        super().__init__()
        self.tp_group = tp_group
        self.split_dim = split_dim
        self.precision = precision

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
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        # For row-wise parallelism, we need all-reduce
        if self.split_dim == 1:
            # Use simple all_reduce without any custom handling
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)

        return output
