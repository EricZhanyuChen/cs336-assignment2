#!/bin/bash
#SBATCH --job-name=s5_naive_ddp
#SBATCH --partition=gpumedium
#SBATCH --output=outputs/s5_ddp/naive_ddp_%j.out
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --mem=32G
#SBATCH --time=00:10:00

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s5_ddp

# Section 5.2: verify naive DDP correctness (CPU gloo)
export MASTER_PORT=$((29500 + RANDOM % 1000))
uv run python cs336_systems/naive_ddp.py
