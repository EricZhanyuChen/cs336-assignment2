#!/bin/bash
#SBATCH --job-name=s5_dist_comm
#SBATCH --partition=gpumedium
#SBATCH --output=outputs/s5_ddp/dist_comm_%j.out
#SBATCH --gres=gpu:rtx_pro_6000:6
#SBATCH --time=01:00:00

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s5_ddp

# Section 5.1: all-reduce bandwidth benchmark
# Sweep over world_size (2,4,6) and data_size (1,10,100,1000 MB)
# RTX Pro 6000 nodes have 8 GPUs → 6 GPUs available for WS=6
for WS in 2 4 6; do
    for MB in 1 10 100 1000; do
        echo "=== world_size=$WS, data_size=${MB}MB ==="
        uv run python cs336_systems/distributed_communication_single_node.py \
            --world_size $WS \
            --data_size_mb $MB \
            --backend nccl \
            --output_dir outputs/s5_ddp
    done
done
