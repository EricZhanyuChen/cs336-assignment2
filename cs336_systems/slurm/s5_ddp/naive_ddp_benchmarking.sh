#!/bin/bash
#SBATCH --job-name=s5_naive_ddp_bench
#SBATCH --partition=gpumedium
#SBATCH --output=outputs/s5_ddp/naive_ddp_bench_%j.out
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s5_ddp

# Section 5.2: benchmark naive DDP overhead
# XL model, 2 GPUs (single node), measure step time and all-reduce communication time
export MASTER_PORT=$((29500 + RANDOM % 1000))
uv run python cs336_systems/naive_ddp_benchmarking.py
