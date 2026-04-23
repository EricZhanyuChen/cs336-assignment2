import torch, timeit
import numpy as np
from cs336_systems.transformer import Transformer, AdamW
from argparse import ArgumentParser
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint


MODEL_CONFIGS = {
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=100, help="Total training steps, not warmup steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument("--with_optimizer", action="store_true", help="Include optimizer step in each measured step")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--check_point", type=int, default=0, help="check point per n block, n must can be divided by 32")
    return parser.parse_args()


def build_model(d_model, d_ff, num_layers, num_heads, vocab_size=10000, device="cuda"):
    transformer = Transformer(d_model, num_layers, num_heads, d_ff, vocab_size)
    return transformer.to(device)


def build_model_from_size(size: str, device: str):
    config = MODEL_CONFIGS[size]
    return build_model(d_model=config["d_model"], d_ff=config["d_ff"], num_layers=config["num_layers"], num_heads=config["num_heads"], device=device)


def generate_batch(device: str, batch_size=4, vocab_size=10000, context_length=2048):
    tokens = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long, device=device)
    return tokens


def run_step(model, tokens: torch.Tensor, forward_only: bool, ctx, optimizer=None, rope_theta=10000):
    with ctx:
        if forward_only:
            with torch.no_grad():
                logits = model(tokens, 2048, rope_theta)
        elif optimizer is not None:
            optimizer.zero_grad()
            logits = model(tokens, 2048, rope_theta)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
        else:
            model.zero_grad()
            logits = model(tokens, 2048, rope_theta)
            loss = logits.sum()
            loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def apply_checkpoint(model, checkpoint_every: int):
    assert len(model.layers) % checkpoint_every == 0, f"checkpoint_every ({checkpoint_every}) must divide num_layers ({len(model.layers)})"
    original_layers = list(model.layers)

    def make_group_fn(layers):
        def group_fn(x, context_length, rope_theta):
            for layer in layers:
                x = layer(x, context_length, rope_theta)
            return x

        return group_fn

    groups = [original_layers[i : i + checkpoint_every] for i in range(0, len(original_layers), checkpoint_every)]

    def new_forward(in_indices, context_length, rope_theta):
        x = model.token_embedding(in_indices)
        for group in groups:
            fn = make_group_fn(group)
            x = checkpoint(fn, x, context_length, rope_theta, use_reentrant=False)
        x = model.final_rms(x)
        return model.output_embedding(x)

    model.forward = new_forward


def main():
    args = parse_args()
    print("Initializing Model...")
    model = build_model_from_size("xl", args.device)
    if args.check_point > 0:
        apply_checkpoint(model, args.check_point)
    print("Model Initialized!")
    optimizer = AdamW(model.parameters(), lr=1e-3) if args.with_optimizer else None
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.mixed_precision else nullcontext()

    print("Generating Data...")
    batch = generate_batch(args.device)
    print("Data Generated!")

    print("Warmup process begins...")
    for _ in range(args.warmup_steps):
        run_step(model, batch, args.forward_only, ctx, optimizer)
    print("Warmup Finished!")

    torch.cuda.reset_peak_memory_stats()
    run_step(model, batch, args.forward_only, ctx, optimizer)
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak memory: {peak_mib:.2f} MiB")

    times = []
    for _ in range(args.n_steps):
        start_time = timeit.default_timer()
        run_step(model, batch, args.forward_only, ctx, optimizer)
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
        if _ % 10 == 0:
            print(f"Step {_} finished.")
    mean = np.mean(times)
    std = np.std(times)
    print(f"Mean: {mean:.4f}s, Std: {std:.4f}s")


if __name__ == "__main__":
    main()
