import os
import socket
import torch
import torch.distributed as dist


def run(rank, local_rank):
    print(f"Rank {rank}:{local_rank} Cuda? {torch.cuda.is_available()}")
    print(f"Rank {rank}:{local_rank} {torch.cuda.device_count()} cuda devices")
    print(f"Rank {rank}:{local_rank} {socket.gethostname()}")
    print(f"Rank {rank}:{local_rank} {os.environ['CUDA_VISIBLE_DEVICES']}")

    tensor = torch.ones((10,)) * rank
    if torch.cuda.is_available():
        tensor = tensor.to(f"cuda:0")
    print(f'Rank {rank}:{local_rank} generated {tensor}')

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=None)
    print(f'Rank {rank}:{local_rank} had {tensor}')


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size=int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        "nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    run(rank, local_rank)

    dist.destroy_process_group()
