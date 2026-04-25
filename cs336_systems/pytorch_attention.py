from cs336_systems.transformer import scaled_dot_product_attention
import torch
import timeit
import itertools
import numpy as np

"""
Section 4.1: PyTorch Attention Benchmarking

Problem (pytorch_attention): PyTorch Attention Benchmarking (2 points)

(a) Benchmark your attention implementation at different scales. Write a script that will:
    - Fix the batch size to 8 and don't use multihead attention (i.e. remove the head dimension).
    - Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding dimension d_model,
      and [256, 1024, 4096, 8192, 16384] for the sequenth length.
    - Create random inputs Q, K, V for the appropriate size.
    - Time 100 forward passes through attention using the inputs.
    - Measure how much memory is in use before the backward pass starts, and time 100 backward passes.
    - Make sure to warm up, and to call `torch.cuda.synchronize()` after each forward/backward pass.

Report the timings (or out-of-memory errors) you get for these configurations. At what size do you get out-of-memory errors?

Deliverable: A table with your timings, your calculations for the memory usage, and a 1-2 paragraph response.
"""


BATCH_SIZE = 8
d_model_list = [16, 32, 64, 128]
seq_len_list = [256, 1024, 4096, 8192, 16384]
WARMUP_STEPS = 5
MEASURE_STEPS = 100
DEVICE = "cuda"

def main():
    for d_model, seq_len in itertools.product(d_model_list, seq_len_list):
            try:
                Q = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True, device=DEVICE)
                K = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True, device=DEVICE)
                V = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True, device=DEVICE)

                for _ in range(WARMUP_STEPS):
                        attention = scaled_dot_product_attention(Q, K, V)
                        attention.sum().backward()
                        Q.grad = None
                        K.grad = None
                        V.grad = None

                times_forward = []
                times_backward = []

                for _ in range(MEASURE_STEPS):
                    start_time_forward = timeit.default_timer()
                    attention = scaled_dot_product_attention(Q, K, V)
                    torch.cuda.synchronize()
                    end_time_forward = timeit.default_timer()
                    times_forward.append(end_time_forward - start_time_forward)
                    mem = torch.cuda.memory_allocated() / 1024 / 1024

                    start_time_backward = timeit.default_timer()
                    attention.sum().backward()
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                    torch.cuda.synchronize()
                    end_time_backward = timeit.default_timer()
                    times_backward.append(end_time_backward - start_time_backward)

                mean_time_forward = np.mean(times_forward)
                std_time_forward = np.std(times_forward)
                mean_time_backward = np.mean(times_backward)
                std_time_backward = np.std(times_backward)

                print(f"------ Sequence Length: {seq_len}, Model Dimension: {d_model} (Forward) ------")
                print(f"Mean Time: {mean_time_forward}, Time Std: {std_time_forward}")
                print(f"Current Memory Used: {mem} MiB")
                print(f"------ Sequence Length: {seq_len}, Model Dimension: {d_model} (Backward) ------")
                print(f"Mean Time: {mean_time_backward}, Time Std: {std_time_backward}")
                print(f"Current Memory Used: {mem} MiB")

            except torch.cuda.OutOfMemoryError:
                print(f"OOM: d_model = {d_model}, seq_len = {seq_len}")
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()