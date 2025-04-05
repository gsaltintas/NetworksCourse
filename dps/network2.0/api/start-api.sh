
#!/bin/bash
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"  && cd .. &>/dev/null && pwd)"
API_DIR=$BASE_DIR/api
LOG_DIR=$BASE_DIR/logs

# Check if the API is already running
if pgrep -f "python.*api.py" > /dev/null; then
    echo "API is already running"
    exit 0
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="$(( $SLURM_JOB_ID % 16384 + 49152 ))"

# Start the API server
echo "Starting Network Monitoring API server..."
nohup python $API_DIR/api.py > $LOG_DIR/api.log 2>&1 &

echo "API started on port $MASTER_PORT"
echo "Logs available at $LOG_DIR/api.log"
