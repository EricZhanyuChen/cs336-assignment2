#!/bin/bash
#SBATCH --job-name=s2_mixed_precision
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/s2_profiling/mixed_precision/logs/s2_mixed_precision_%j.out

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s2_profiling/mixed_precision/logs

# Section 2.1.5: Mixed Precision Benchmarking
# Compare FP32 vs BF16 for all model sizes
for SIZE in small medium large xl; do
    echo "=== [2.1.5] Size: $SIZE - FP32 ==="
    uv run python cs336_systems/benchmark.py \
        --size $SIZE \
        --warmup_steps 5 \
        --n_steps 10 \
        --context_length 512

    echo "=== [2.1.5] Size: $SIZE - BF16 ==="
    uv run python cs336_systems/benchmark.py \
        --size $SIZE \
        --warmup_steps 5 \
        --n_steps 10 \
        --context_length 512 \
        --mixed_precision
done