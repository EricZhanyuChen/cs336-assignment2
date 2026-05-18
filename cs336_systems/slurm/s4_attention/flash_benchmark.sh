#!/bin/bash
#SBATCH --job-name=flash_benchmark
#SBATCH --partition=gpumedium
#SBATCH --output=outputs/s4_attention/logs/flash_benchmark_%j.out
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s4_attention/logs

# Section 4.3(e): FlashAttention-2 Benchmarking
# Compare PyTorch vs Triton forward pass across seq_len, head_dim, dtype
# Triton backward skipped (NotImplementedError)

# Triton JIT needs Python.h for its launcher compilation
TRITON_PYTHON_INC="/cvmfs/hpc.rug.nl/versions/2023.01/rocky8/x86_64/amd/zen3/software/Python/3.12.3-GCCcore-13.3.0/include/python3.12"
export CPATH="$TRITON_PYTHON_INC:$CPATH"

uv run python cs336_systems/flash_benchmark.py \
    --output_dir outputs/s4_attention \
    --skip_triton_bwd
