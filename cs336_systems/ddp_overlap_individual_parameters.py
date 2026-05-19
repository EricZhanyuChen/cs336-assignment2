import torch
import torch.distributed as dist

class DDPOverIndividualParameter(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        self._handles = []
        def _grad_hook(param):
            param.grad /= dist.get_world_size()
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)

        
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(_grad_hook)        

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

