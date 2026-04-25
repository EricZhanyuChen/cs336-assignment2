#!/bin/bash
#SBATCH --job-name=s4_torch_compile_full
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/s4_attention/logs/s4_torch_compile_full_%j.out

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s4_attention/logs

# Section 4.2(b): Compile entire Transformer model
# Compare vanilla vs compiled for forward, forward+backward, forward+backward+optimizer

echo "=========================================="
echo "Section 4.2(b): torch.compile on Transformer"
echo "=========================================="

for MODE in "--forward_only" "" "--with_optimizer"; do
    MODE_NAME=$(echo $MODE | sed 's/--//g' | sed 's/_/ /g')
    if [ -z "$MODE_NAME" ]; then MODE_NAME="forward+backward"; fi

    echo ""
    echo ">>> Mode: $MODE_NAME - Vanilla >>>"
    uv run python cs336_systems/benchmark.py \
        --size xl \
        --warmup_steps 5 \
        --n_steps 100 \
        --context_length 512 \
        $MODE

    echo ""
    echo ">>> Mode: $MODE_NAME - Compiled >>>"
    uv run python cs336_systems/benchmark.py \
        --size xl \
        --warmup_steps 5 \
        --n_steps 100 \
        --context_length 512 \
        --compile \
        $MODE
done