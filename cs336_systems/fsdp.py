import torch
import torch.nn as nn
import torch.distributed as dist
from cs336_basics.model import Embedding, Linear, RMSNorm
class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.module = module
        self.compute_dtype = compute_dtype
        self.original_shapes = {}
        self.sharded_modules = [m for m in self.module.modules() if isinstance (m, (Linear, Embedding))]
        self.un_sharded_modules = [m for m in self.module.modules() if m not in self.sharded_modules]
        
        def make_forward_pre_hook(m):
            def hook(module, args):
                full = torch.empty(m.weight.data.numel() * self.size, device=m.weight.data.device, dtype=compute_dtype)
                dist.all_gather_into_tensor(full, m.weight.data.to(compute_dtype))
                m.weight.data = full.view(self.original_shapes[m])
            return hook
        
        def make_forward_hook(m):
            def hook(module, input, output):
                m.weight.data = m.weight.data.flatten()
                start = self.rank * len(m.weight.data) // self.size
                end = (self.rank+1) * len(m.weight.data) // self.size
                local_param = m.weight.data[start: end]
                m.weight.data = local_param.clone().to(torch.float32)
            return hook
        
        def make_backward_pre_hook(m):
            def hook(module, grad_output):
                if not isinstance(m, Embedding):
                    full = torch.empty(m.weight.data.numel() * self.size, device=m.weight.data.device, dtype=compute_dtype)
                else:
                    full = torch.empty(m.weight.data.numel() * self.size, device=m.weight.data.device, dtype=m.weight.data.dtype)
                dist.all_gather_into_tensor(full, m.weight.data if isinstance(m, Embedding) else m.weight.data.to(compute_dtype))
                m.weight.data = full.view(self.original_shapes[m])
            return hook     
        
        def make_grad_hook(m):
            def hook(param):
                param.data = param.data.flatten()
                start = self.rank * len(param.data) // self.size
                end = (self.rank+1) * len(param.data) // self.size
                param.data = param.data[start: end].clone().to(torch.float32)
                shard_grad = torch.empty(
                    param.grad.numel() // self.size,
                    device = param.grad.device,
                    dtype = torch.float32
                )
                dist.reduce_scatter_tensor(shard_grad, param.grad.to(torch.float32).flatten())
                param.grad = shard_grad / self.size
                
            return hook
                



        for m in self.sharded_modules:
            self.original_shapes[m] = m.weight.shape
            m.weight.data = m.weight.data.flatten()
            start = self.rank * len(m.weight.data) // self.size
            end = (self.rank+1) * len(m.weight.data) // self.size
            local_param = m.weight.data[start: end]
            m.weight.data = local_param.clone()

            m.register_forward_pre_hook(make_forward_pre_hook(m))
            m.register_forward_hook(make_forward_hook(m))
            m.register_full_backward_pre_hook(make_backward_pre_hook(m))
            m.weight.register_post_accumulate_grad_hook(make_grad_hook(m))


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    

    def finish_gradient_synchronization(self):
        for m in self.un_sharded_modules:
            for param in m.parameters(recurse=False):
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad /= self.size

        




