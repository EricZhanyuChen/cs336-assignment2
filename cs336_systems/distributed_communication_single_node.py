import torch
import torch.distributed as dist
import os 
import torch.multiprocessing as mp
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size_mb",default="1", type=int)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--output_dir", default="outputs/s5_ddp")

    return parser.parse_args()

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def benchmark(rank, world_size, data_size_mb, backend, output_file=None):
    setup(rank=rank, world_size=world_size, backend=backend)

    num_elements = data_size_mb * 1024 * 1024 // 4
    tensor = torch.ones(num_elements, dtype=torch.float32, device=f"cuda:{rank}")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        dist.all_reduce(tensor)

    start.record()
    for _ in range(20):
        dist.all_reduce(tensor)
    end.record()
    torch.cuda.synchronize()
    time_spent = start.elapsed_time(end) / 20
    result = f"rank={rank}, data_size={data_size_mb}MB, world_size={world_size}, avg_time={time_spent:.2f}ms"
    print(result)
    if rank == 0 and output_file:
        with open(output_file, "w") as f:
            f.write(result + "\n")
    cleanup()
    

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"dist_comm_ws{args.world_size}_size{args.data_size_mb}mb.txt")
    mp.spawn(benchmark, args=(args.world_size, args.data_size_mb, args.backend, out_file), nprocs=args.world_size)