#!/bin/bash
#SBATCH --partition a40
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=tp-precision
#SBATCH --output=output/slurm_%j.out
#SBATCH --wait-all-nodes=1
pwd; hostname; date

source .venv/bin/activate

# MODEL="meta-llama/Llama-3.2-1B-Instruct"
MODEL="/model-weights/Llama-3.2-1B-Instruct"

# Set the JOB_LABEL environment variable
echo "-------- Setting JOB_LABEL ---------------------------------------------"
echo ""
# Decide the name of the paths to use for saving this job
if [ "$SLURM_ARRAY_TASK_COUNT" != "" ] && [ "$SLURM_ARRAY_TASK_COUNT" -gt 1 ];
then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}";
else
    JOB_ID="${SLURM_JOB_ID}";
fi
# Decide the name of the paths to use for saving this job
JOB_LABEL="${SLURM_JOB_NAME}__${JOB_ID}";
echo "JOB_ID = $JOB_ID"
echo "JOB_LABEL = $JOB_LABEL"
echo ""

# -----------------------------------------------------------------------------

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Rank 0 node is at $MASTER_ADDR"
# Dynamic Ports, also known as Private Ports.
# MASTER_PORT="$(( $SLURM_JOB_ID % 16384 + 49152 ))"
# export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500  # Choose an open port
if ss -tulpn | grep -q ":$MASTER_PORT ";
then
    # The port we selected is in use, so we'll get a random available port instead.
    echo "Finding a free port to use for $SLURM_NNODES node training"
    MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')";
fi
export MASTER_PORT;
echo "Will use port $MASTER_PORT for c10d communication"
export WORLD_SIZE="$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))"
echo "WORLD_SIZE = $WORLD_SIZE"



# NCCL options ----------------------------------------------------------------

# This is needed to print debug info from NCCL, can be removed if all goes well
export NCCL_DEBUG=INFO

# This is needed to avoid NCCL to use ifiniband, which the cluster does not have
export NCCL_IB_DISABLE=1

# This is to tell NCCL to use bond interface for network communication
if [[ "${SLURM_JOB_PARTITION}" == "t4v2" ]] || [[ "${SLURM_JOB_PARTITION}" == "rtx6000" ]];
then
    echo "Using NCCL_SOCKET_IFNAME=bond0 on ${SLURM_JOB_PARTITION}"
    export NCCL_SOCKET_IFNAME=bond0
fi

# Set this when using the NCCL backend for inter-GPU communication.
export TORCH_NCCL_BLOCKING_WAIT=1

export OMP_NUM_THREADS=1
# -----------------------------------------------------------------------------
SLURM_JOB_NUM_NODES=$SLURM_NNODES
# Multi-GPU configuration
echo ""
echo "Main script begins via torchrun with host tcp://${MASTER_ADDR}:$MASTER_PORT with backend NCCL"
if [[ "$SLURM_JOB_NUM_NODES" == "1" ]];
then
    echo "Single ($SLURM_JOB_NUM_NODES) node training ($SLURM_GPUS_ON_NODE GPUs)"
else
    echo "Multiple ($SLURM_JOB_NUM_NODES) node training (x$SLURM_GPUS_ON_NODE GPUs per node)"
fi
echo ""
NUM_GPUS=$((SLURM_JOB_NUM_NODES*SLURM_GPUS_ON_NODE))
echo "endpoint $MASTER_ADDR:$MASTER_PORT"
# -----------------------------------------------------------------------------

for node_idx in $(seq 0 $((${SLURM_JOB_NUM_NODES} - 1))); do
  srun -n 1 torchrun \
    --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend="c10d" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --node_rank="${node_idx}" \
    --max-restarts=0 \
  dps/main.py \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --model_name=${MODEL} \
  --dataset="wikitext" \
  --dataset_config="wikitext-2-raw-v1" \
  --output_dir="./output" \
  --batch_size=8 \
  --tensor_parallel_size=${SLURM_JOB_NUM_NODES} \
  --num_eval_samples=800 \
  --tensor_parallel_size=2 \
  --data_parallel_size=1 \
  --num_gpus=${SLURM_JOB_NUM_NODES} \
  --log_level="INFO" \
  --learning_rate=2e-5 \
  --weight_decay=0.01 \
  --num_train_steps=10000 \
  --lr_scheduler="cosine" \
  --warmup_ratio=0.1 \
  --use_wandb=True \
  --gradient_accumulation_steps=1 \
    "${@}" &
done

wait
date
