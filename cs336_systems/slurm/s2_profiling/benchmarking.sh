#!/bin/bash
#SBATCH --job-name=s2_benchmarking
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/s2_profiling/benchmarking/logs/s2_benchmarking_%j.out

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s2_profiling/benchmarking/logs

# 2.1.3b: 5 warmup, 10 steps, all sizes, three modes
for SIZE in small medium large xl 10B; do
    for MODE in "--forward_only" "" "--with_optimizer"; do
        echo "=== [2.1.3b] Size: $SIZE | Mode: $MODE ==="
        uv run python cs336_systems/benchmark.py \
            --size $SIZE \
            --warmup_steps 5 \
            --n_steps 10 \
            --context_length 512 \
            $MODE
    done
done

# 2.1.3c: varying warmup, small model
for WARMUP in 0 1 2 5; do
    echo "=== [2.1.3c] Warmup steps: $WARMUP ==="
    uv run python cs336_systems/benchmark.py \
        --size small \
        --warmup_steps $WARMUP \
        --n_steps 10 \
        --context_length 512
done
