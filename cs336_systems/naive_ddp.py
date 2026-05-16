import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
    
def train(rank, world_size, num_steps=5):
    setup(rank, world_size)

    model = ToyModel()

    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    if rank == 0:
        torch.save(model.state_dict(), "initial_weights.pt")
    
    for step in range(num_steps):
        torch.manual_seed(step)
        x = torch.randn(4, 10)
        x_shard = x[rank*2: (rank+1)*2]
        optimizer.zero_grad()
        output = model(x_shard)
        loss = output.sum()
        loss.backward()

        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
        optimizer.step()
    if rank == 0:
        torch.save(model.state_dict(), "final_weights_ddp.pt")
    cleanup()

def single_process_train(num_steps=5):
    model = ToyModel()
    model.load_state_dict(torch.load("initial_weights.pt", weights_only=True))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for step in range(num_steps):
        torch.manual_seed(step)
        x = torch.randn(4, 10)  # 完整 batch，不切分
        
        optimizer.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
    
    return model

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    ddp_state = torch.load("final_weights_ddp.pt", weights_only=True)
    single_model = single_process_train()
    for (name, param), ddp_param in zip(single_model.named_parameters(), ddp_state.values()):
        match = torch.allclose(param.data, ddp_param, atol=1e-5)
        print(f"{name}: {'Yes' if match else 'No'}")