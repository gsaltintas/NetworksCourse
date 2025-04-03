#!/bin/bash
#SBATCH --partition a40
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=tp-demo
#SBATCH --output=output/slurm_%j.out
#SBATCH --wait-all-nodes=1
pwd; hostname; date

source .venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo "Rank 0 node is at ${MASTER_ADDR}"

export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
echo "Using port ${MASTER_PORT} for communication"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export TORCH_NCCL_BLOCKING_WAIT=1

# export WORLD_SIZE="$(( ${SLURM_NNODES} * ${SLURM_GPUS_ON_NODE}))"
# echo "WORLD_SIZE = ${WORLD_SIZE}"

# RANK and LOCAL_RANK are automatically set by torchrun

echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"

for node_idx in $(seq 0 $((${SLURM_JOB_NUM_NODES} - 1))); do
  srun -n 1 torchrun \
    --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc-per-node="${SLURM_GPUS_ON_NODE}" \
    --node-rank="${node_idx}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    --rdzv-backend="c10d" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    demo.py &
done

wait
date
