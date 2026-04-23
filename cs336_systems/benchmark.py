import torch, timeit
import numpy as np
import torch.cuda.nvtx as nvtx
from cs336_systems.transformer import Transformer, AdamW
from argparse import ArgumentParser
from contextlib import nullcontext


MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":  {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=100,
                        help="Total training steps, not warmup steps")
    parser.add_argument("--size", type=str, default="small",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument("--with_optimizer", action="store_true",
                    help="Include optimizer step in each measured step")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--memory_profile", action="store_true",
                        help="Enbale memory profile")
    return parser.parse_args()


def build_model(d_model, d_ff, num_layers, num_heads, vocab_size=10000, device="cuda"):
    transformer = Transformer(d_model, num_layers, num_heads, d_ff, vocab_size)
    return transformer.to(device)

def build_model_from_size(size: str, device: str):
    config = MODEL_CONFIGS[size]
    return build_model(
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        device=device
    )

def generate_batch(context_length: int, device: str, batch_size=4, vocab_size=10000):
    tokens = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long, device=device)
    return tokens

def run_step(model, tokens: torch.Tensor, context_length: int, forward_only: bool, ctx, optimizer=None, rope_theta=10000):
    with ctx:
        if forward_only:
            with torch.no_grad():
                logits = model(tokens, context_length, rope_theta)
        elif optimizer is not None:
            optimizer.zero_grad()
            logits = model(tokens, context_length, rope_theta)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
        else:
            model.zero_grad()
            logits = model(tokens, context_length, rope_theta)
            loss = logits.sum()
            loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    args = parse_args()
    print("Initializing Model...")
    model = build_model_from_size(args.size, args.device)
    print("Model Initialized!")
    optimizer = AdamW(model.parameters(), lr=1e-3) if args.with_optimizer else None
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.mixed_precision else nullcontext()

    print("Generating Data...")
    batch = generate_batch(args.context_length, args.device)
    print("Data Generated!")

    print("Warmup process begins...")
    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            run_step(model, batch, args.context_length, args.forward_only, ctx, optimizer)
    print("Warmup Finished!")

    times = []
    if args.memory_profile:
        mode = "fwd" if args.forward_only else ("train" if args.with_optimizer else "fwd_bwd")
        snapshot_name = f"outputs/s2_profiling/memory/{args.size}_ctx{args.context_length}_{mode}.pickle"
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    with nvtx.range("measurement"):
        for _ in range(args.n_steps):
            start_time = timeit.default_timer()
            run_step(model, batch, args.context_length, args.forward_only, ctx, optimizer)
            end_time = timeit.default_timer()
            times.append(end_time - start_time)
            if _ % 10 == 0:
                print(f"Step {_} finished.")
    if args.memory_profile:
        torch.cuda.memory._dump_snapshot(snapshot_name)
        torch.cuda.memory._record_memory_history(enabled=None)
    mean = np.mean(times)
    std = np.std(times)
    print(f"Mean: {mean:.4f}s, Std: {std:.4f}s")

if __name__ == "__main__":
    main()