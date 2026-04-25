from cs336_systems.transformer import scaled_dot_product_attention
import torch
import timeit
import itertools
import numpy as np

"""
Section 4.2: Torch Compile

Problem (torch_compile): Torch Compile (2 points)

This problem explores the effect of torch.compile on attention performance.

(a) Use the same benchmarking setup as in the previous problem (Section 4.1).
(b) Apply torch.compile to your attention function.
(c) Measure the speedup from compilation for different model sizes and sequence lengths.
(d) Report your findings in a table and 1-2 paragraph response.

Deliverable: Timing table comparing compiled vs non-compiled attention, and your analysis.
"""


BATCH_SIZE = 8
d_model_list = [16, 32, 64, 128]
seq_len_list = [256, 1024, 4096, 8192, 16384]
WARMUP_STEPS = 5
MEASURE_STEPS = 100
DEVICE = "cuda"

compiled_attention = torch.compile(scaled_dot_product_attention)
def main():
    for d_model, seq_len in itertools.product(d_model_list, seq_len_list):
            try:
                Q = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True, device=DEVICE)
                K = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True, device=DEVICE)
                V = torch.randn(BATCH_SIZE, seq_len, d_model, requires_grad=True, device=DEVICE)

                for _ in range(WARMUP_STEPS):
                        attention = compiled_attention(Q, K, V)
                        attention.sum().backward()
                        Q.grad = None
                        K.grad = None
                        V.grad = None

                times_forward = []
                times_backward = []

                for _ in range(MEASURE_STEPS):
                    start_time_forward = timeit.default_timer()
                    attention = compiled_attention(Q, K, V)
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