#!/bin/bash
#SBATCH --job-name=s4_pytorch_attention
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/s4_attention/logs/s4_pytorch_attention%j.out

cd /home5/s6398820/projects/cs336/assignment2
mkdir -p outputs/s4_attention/logs

export CPATH=/usr/include/python3.6m:$CPATH

uv run python cs336_systems/pytorch_attention.py
