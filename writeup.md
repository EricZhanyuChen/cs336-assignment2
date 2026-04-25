# CS336 Assignment 2 Writeup

## Section 4.2: Torch Compile

### (a) Torch Compile Attention Benchmarking

We extended the PyTorch attention benchmarking script to include a compiled version using torch.compile(scaled_dot_product_attention). The same configurations were used: batch_size=8, d_model ∈ {16, 32, 64, 128}, seq_len ∈ {256, 1024, 4096, 8192, 16384}.

#### Forward Pass Timings Comparison (seconds)
| d_model | seq_len=256 | seq_len=1024 | seq_len=4096 | seq_len=8192 | seq_len=16384 |
|---------|-------------|--------------|--------------|--------------|---------------|
| 16 (Uncompiled) | 0.000068 | 0.000169 | 0.005245 | 0.020562 | 0.081806 |
| 16 (Compiled) | 0.000068 | 0.000128 | 0.001939 | 0.007307 | 0.027360 |
| 32 (Uncompiled) | 0.000064 | 0.000172 | 0.005246 | 0.020586 | 0.082060 |
| 32 (Compiled) | 0.000086 | 0.000158 | 0.002455 | 0.010010 | 0.028106 |
| 64 (Uncompiled) | 0.000066 | 0.000188 | 0.005413 | 0.020898 | 0.083894 |
| 64 (Compiled) | 0.000187 | 0.000627 | 0.003249 | 0.008550 | 0.030525 |
| 128 (Uncompiled) | 0.000066 | 0.000239 | 0.005710 | 0.022467 | 0.088593 |
| 128 (Compiled) | 0.000187 | 0.000677 | 0.003548 | 0.010120 | 0.035266 |

#### Backward Pass Timings Comparison (seconds)
| d_model | seq_len=256 | seq_len=1024 | seq_len=4096 | seq_len=8192 | seq_len=16384 |
|---------|-------------|--------------|--------------|--------------|---------------|
| 16 (Uncompiled) | 0.000141 | 0.000452 | 0.011087 | 0.043656 | 0.173958 |
| 16 (Compiled) | 0.000111 | 0.000258 | 0.004461 | 0.017302 | 0.068946 |
| 32 (Uncompiled) | 0.000142 | 0.000451 | 0.011091 | 0.043716 | 0.174104 |
| 32 (Compiled) | 0.000121 | 0.000369 | 0.005280 | 0.021421 | 0.080447 |
| 64 (Uncompiled) | 0.000145 | 0.000466 | 0.011257 | 0.043991 | 0.175859 |
| 64 (Compiled) | 0.000225 | 0.000782 | 0.005751 | 0.020535 | 0.082152 |
| 128 (Uncompiled) | 0.000143 | 0.000553 | 0.011815 | 0.046554 | 0.183226 |
| 128 (Compiled) | 0.000228 | 0.000867 | 0.006327 | 0.023119 | 0.089562 |

### Analysis

torch.compile provides significant speedup for attention, especially at larger sequence lengths. For d_model=128, seq_len=16384, forward speedup is ~2.5x, backward ~2x. Smaller configurations show modest or no speedup due to compilation overhead. Memory usage remains the same.