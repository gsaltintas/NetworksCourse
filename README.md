# NetworksCourse

## Set-up
```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
```


## Dynamic Precision Scheduler (DPS)
In this work, we implemented DPS, a dynamic precision scheduler, that determines what numerical precision to use during multi-gpu communication.

### Overview
The Dynamic Precision Scheduler (DPS) is responsible for deciding what numerical precision should be used during communication steps in distributed training. It takes into account network utilization, model layer characteristics, training progress, and network topology to dynamically adjust precision levels, optimizing the trade-off between communication latency and model accuracy.


1. Random Scheduler
We first implement a basic framework and a random policy to validate that dynamic precision changes don't significantly harm training. This step doesn't require any network monitoring but can be considered a baseline against the mixed-precision training.

2. Network-aware Policies
2.a. Heuristic Policy

2.b. RL-based Policy


## Usage
### Basic Usage
[WIP] Not implemented fully
```bash
# Run with default configuration
python main.py

# Run with custom configuration file
python main.py config.json

# Run with command-line parameters
python main.py  --seed 123 --network_topology tree --precision_policy random
```

See `dps/utils/config.py` for the full configuration options.
