import os
import torch
import torch.distributed as dist
import time
from cs336_systems.fsdp import FSDP
from cs336_systems.transformer import Transformer
import torch.multiprocessing as mp

OUTPUT_DIR = "outputs/s7_fsdp_accounting"

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, num_step=5):
    setup(rank, world_size)
    model = Transformer(d_model=2560,  d_ff=10240, num_layers=32, num_heads=32, vocab_size=1024).cuda(rank)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/initial_weights.pt", weights_only=True, map_location=f"cuda:{rank}"))
    print(f"Rank {rank} | after model init | {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    model = FSDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    for step in range(5):
        torch.manual_seed(step)
        torch.cuda.synchronize()
        x = torch.randint(0, 1024, (4, 512)).cuda(rank)
        shard_size = x.shape[0] // world_size
        x_shard = x[rank*shard_size: (rank+1)*shard_size]
        optimizer.zero_grad()
        output = model(x_shard, 512, 10000.0)
        loss = output.sum()
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    for step in range(num_step):
        torch.manual_seed(step)
        torch.cuda.synchronize()
        x = torch.randint(0, 1024, (4, 512)).cuda(rank)
        shard_size = x.shape[0] // world_size
        x_shard = x[rank*shard_size: (rank+1)*shard_size]
        optimizer.zero_grad()
        output = model(x_shard, 512, 10000.0)
        loss = output.sum()
        loss.backward()
        model.finish_gradient_synchronization()
        if step == 0:
            print(f"Memory before optimizer.step(): {torch.cuda.memory_allocated() / 1024**3}")
        optimizer.step()
        if step == 0:
            print(f"Memory after optimizer.step(): {torch.cuda.memory_allocated() / 1024**3}")

    cleanup()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    world_size = 2
    model = Transformer(d_model=2560, d_ff=10240, num_layers=32, num_heads=32, vocab_size=1024)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/initial_weights.pt")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

    