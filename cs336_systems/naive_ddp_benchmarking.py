import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from cs336_systems.transformer import Transformer

OUTPUT_DIR = "outputs/s5_ddp"

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

    
def train(rank, world_size, num_steps=5):
    setup(rank, world_size)

    model = Transformer(d_model=2560,  d_ff=10240, num_layers=32, num_heads=32, vocab_size=1024).cuda(rank)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/initial_weights.pt", weights_only=True, map_location=f"cuda:{rank}"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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

        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
    
    step_times = []
    reduce_times = []
    for step in range(num_steps):
        torch.manual_seed(step)
        torch.cuda.synchronize()
        t0 = time.time()
        x = torch.randint(0, 1024, (4, 512)).cuda(rank)
        shard_size = x.shape[0] // world_size
        x_shard = x[rank*shard_size: (rank+1)*shard_size]
        optimizer.zero_grad()
        output = model(x_shard, 512, 10000.0)
        loss = output.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        t1= time.time()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
        torch.cuda.synchronize()
        reduce_time = time.time() - t1
        reduce_times.append(reduce_time)
        optimizer.step()
        torch.cuda.synchronize()
        step_time = time.time() - t0
        step_times.append(step_time)
    
    print(f"Rank {rank}")
    print(f"Time spent: {sum(step_times) / len(step_times)}")
    print(f"Reduce time accounts for {(sum(reduce_times) / len(reduce_times)) / (sum(step_times) / len(step_times))}")
    cleanup()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    world_size = 2
    model = Transformer(d_model=2560, d_ff=10240, num_layers=32, num_heads=32, vocab_size=1024)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/initial_weights.pt")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)