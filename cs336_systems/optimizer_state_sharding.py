import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any): 
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.inner_optimizer = None
        self.param_to_rank = {}
        super().__init__(params, {})



    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        params = list(param_group["params"])
        start = self.rank * len(params) // self.world_size
        end = (self.rank + 1) * len(params) //self.world_size 
        local_params = params[start: end]

        for rank in range(self.world_size):
            start = rank * len(params) // self.world_size
            end = (rank + 1) * len(params) //self.world_size 
            for param in params[start: end]:
                self.param_to_rank[param] = rank

        if self.inner_optimizer is None:
            self.inner_optimizer = self.optimizer_cls(local_params, **{**self.kwargs, **{k: v for k, v in param_group.items() if k!="params"}})
        else:
            self.inner_optimizer.add_param_group({"params": local_params, **{k: v for k, v in param_group.items() if k!="params"}})

    def step(self, closure=None, **kwargs):
        self.inner_optimizer.step(closure, **kwargs)
        for param_group in self.param_groups:
            for param in param_group["params"]:
                dist.broadcast(param.data, src=self.param_to_rank[param])