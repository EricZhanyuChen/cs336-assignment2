#!/bin/bash
#SBATCH --job-name=flash_benchmark
#SBATCH --partition=gpumedium
#SBATCH --output=outputs/s4_attention/logs/flash_benchmark_%j.out
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=04:00:00

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s4_attention/logs

# Section 4.3(e): FlashAttention-2 Benchmarking
# Compare PyTorch vs Triton forward pass across seq_len, head_dim, dtype
# Triton backward skipped (NotImplementedError)
uv run python cs336_systems/flash_benchmark.py \
    --output_dir outputs/s4_attention \
    --skip_triton_bwd
