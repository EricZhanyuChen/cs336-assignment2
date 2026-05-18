#!/bin/bash
#SBATCH --job-name=s5_flat_ddp_bench
#SBATCH --partition=gpumedium
#SBATCH --output=outputs/s5_ddp/flat_ddp_bench_%j.out
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --time=01:00:00

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s5_ddp

# Section 5.3.1: DDP with flat gradient all-reduce
# Compare vs naive per-parameter all-reduce on XL model, 2 GPUs (single node)
uv run python cs336_systems/minimal_ddp_flat_benchmarking.py
