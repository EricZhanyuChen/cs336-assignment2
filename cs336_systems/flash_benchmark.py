import os
import argparse
import torch
import math
import triton
import itertools
import pandas as pd
from cs336_systems.flash_forward import FlashAttentionFuncTriton

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
HEAD_DIMS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]
BATCH = 1
IS_CAUSAL = True

def pytorch_attention(Q, K, V, is_causal=True):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = Q @ K.transpose(-2, -1) * scale
    if is_causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    return P @ V


def make_inputs(N, d, dtype, device="cuda"):
    Q = torch.randn(BATCH, N, d, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(BATCH, N, d, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(BATCH, N, d, dtype=dtype, device=device, requires_grad=True)
    return Q, K, V

def benchmark_config(N, d, dtype, device="cuda"):
    Q, K, V = make_inputs(N, d, dtype, device)
    dO = torch.randn(BATCH, N, d, dtype=dtype, device=device)

    O_pt = pytorch_attention(Q, K, V, is_causal=IS_CAUSAL)
    fwd_pt = triton.testing.do_bench(
        lambda: pytorch_attention(Q, K, V, is_causal=IS_CAUSAL)
    )

    bwd_pt = triton.testing.do_bench(
        lambda: O_pt.backward(dO, retain_graph=True)
    )
    def pt_e2e():
        O = pytorch_attention(Q, K, V, is_causal=IS_CAUSAL)
        O.backward(dO)
    e2e_pt = triton.testing.do_bench(pt_e2e)

    Q.grad = None
    K.grad = None
    V.grad = None

    fwd_triton_fwd_only = triton.testing.do_bench(
        lambda: FlashAttentionFuncTriton.apply(Q, K, V, IS_CAUSAL)
    )

    result = {
        "fwd_pt": fwd_pt,
        "bwd_pt": bwd_pt,
        "e2e_pt": e2e_pt,
        "fwd_triton": fwd_triton_fwd_only,
    }

    if not args.skip_triton_bwd:
        O_triton = FlashAttentionFuncTriton.apply(Q, K, V, IS_CAUSAL)
        bwd_triton = triton.testing.do_bench(
            lambda: O_triton.backward(dO, retain_graph=True)
        )
        def triton_e2e():
            O = FlashAttentionFuncTriton.apply(Q, K, V, is_causal=IS_CAUSAL)
            O.backward(dO)
        e2e_triton = triton.testing.do_bench(triton_e2e)
        result["bwd_triton"] = bwd_triton
        result["e2e_triton"] = e2e_triton

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/s4_attention")
    parser.add_argument("--skip_triton_bwd", action="store_true", help="skip triton backward (raises NotImplementedError)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for N, d, dtype in itertools.product(SEQ_LENS, HEAD_DIMS, DTYPES):
        print(f"Benchmarking N={N}, d={d}, dtype={dtype}...")
        try:
            result = benchmark_config(N, d, dtype)
            results.append({"N": N, "d": d, "dtype": str(dtype), **result})
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            results.append({"N": N, "d": d, "dtype": str(dtype), "fwd_pt": "OOM",
                            "bwd_pt": "OOM", "e2e_pt": "OOM", "fwd_triton": "OOM"})
        torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "flash_benchmark_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    print(df.to_string(index=False))