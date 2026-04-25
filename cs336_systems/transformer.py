import torch
import math
import numpy as np
import os

from torch._refs import view
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import BinaryIO, IO, Optional, Iterable

class Linear(torch.nn.Module):
    def __init__(self,in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = 1 / math.sqrt(self.in_features+self.out_features)
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("... i, o i -> ... o", x, self.W)
        return x
    
class Embedding(torch.nn.Module):
        # num_embeddings = vocab size
        # embedding shape: (vocab_size, d_model)
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding = torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype)
        self.embedding = torch.nn.Parameter(self.embedding)
        torch.nn.init.trunc_normal_(self.embedding, std=1, a=-3, b=3)

       # token_ids: (batch_size, sequence_length) 
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids = token_ids.to(self.embedding.device)
        token_ids.to(torch.long)
        embeddings = self.embedding[token_ids]

        return embeddings

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(torch.ones(self.d_model, dtype=self.dtype, device=self.device))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_square = torch.square(x)
        RMS_value = torch.sqrt(torch.mean(x_square, dim=-1, keepdim=True) + self.eps)
        x = (x/RMS_value)*self.W
        x = x.to(in_dtype)
        return x

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.w2 = Linear(d_ff, d_model, device=self.device, dtype=self.dtype)
        self.w3 = Linear(d_model, d_ff, device=self.device, dtype=self.dtype)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu_w1x = w1_x * torch.sigmoid(w1_x)

        w3_x = self.w3(x)
        gated_features = silu_w1x * w3_x
        output = self.w2(gated_features)
        return output
    
class RotaryPositionalEmbedding(torch.nn.Module):
    rope_buffer: torch.Tensor
    # Mathematical formula: theta_ik = i / [theta**(2k-2)/d]. 
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # num_groups is d_k / 2 because RoPE pairs adjacent dimensions
        self.num_groups = d_k // 2
        assert self.d_k % 2 == 0, "d_k must be even for RoPE"

        # k: shape (num_groups,) -> [1, 2, ..., d_k/2]
        k = torch.arange(1, self.num_groups + 1, device=self.device)
        
        # exponent: shape (num_groups,) -> [0/d, 2/d, 4/d, ..., (d-2)/d]
        exponent = (2*k - 2) / self.d_k
        
        # base_k: shape (num_groups,) -> [theta^0, theta^(2/d), ..., theta^((d-2)/d)]
        self.base_k = self.theta ** exponent

        # i: shape (max_seq_len,) -> [0, 1, ..., max_seq_len - 1], i is the position of a token in a sequence
        i = torch.arange(0, self.max_seq_len, device=self.device)
        
        # i_expanded: shape (max_seq_len, 1)
        i_expanded = i.unsqueeze(1)
        
        # base_k_expanded: shape (1, num_groups)
        base_k_expanded = self.base_k.unsqueeze(0)
        
        # theta_vals: shape (max_seq_len, num_groups)
        # Represents the rotation frequency for each position and dimension pair
        self.theta_vals = i_expanded / base_k_expanded

        # cos_vals / sin_vals: shape (max_seq_len, num_groups)
        cos_vals = torch.cos(self.theta_vals)
        sin_vals = torch.sin(self.theta_vals)

        # rope_buffer: shape (max_seq_len, num_groups, 2)
        # Stores [cos, sin] pairs for each position and dimension group
        rope_buffer = torch.stack([cos_vals, sin_vals], dim=2)
        
        # Register as a buffer so it moves with the model to GPU/CPU but is not a trainable parameter
        # persistent=False: do not store it into the model weights
        self.register_buffer("rope_buffer", rope_buffer, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies Rotary Position Embeddings (RoPE) to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): A tensor containing the token positions, of shape (..., seq_len).

        Returns:
            torch.Tensor: The tensor with rotary embeddings applied, having the same shape as the input x.
        """
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)
        
        # Validate the last dimension of the input tensor.
        if x.size(-1) != self.d_k:
            raise ValueError("Input tensor dimension is not correct")
        
        # Ensure the data type of token positions is long (int64), which is required for buffer indexing.
        token_positions = token_positions.long().to(self.rope_buffer.device)
        
        # rope_sliced: shape (..., seq_len, num_groups, 2)
        # Selects the precomputed cos/sin values from the buffer for the given token positions.
        rope_sliced = self.rope_buffer[token_positions] 
        
        # cos / sin: shape (..., seq_len, num_groups)
        # Separate the cosine and sine components.
        cos = rope_sliced[..., 0] 
        sin = rope_sliced[..., 1] 

        # x_reshaped: shape (..., seq_len, num_groups, 2)
        # Reshape the last dimension into pairs to apply the 2D rotation.
        x_reshaped = x.reshape(*x.shape[:-1], self.num_groups, 2)
        
        # x1 / x2: shape (..., seq_len, num_groups)
        # Separate the pairs into two components.
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]

        # Ensure cos/sin shapes are aligned for broadcasting against x1/x2.
        # x1 can be (Batch, Heads, Seq, Num_groups) or (Batch, Seq, Num_groups)
        if cos.ndim < x1.ndim:
            missing_dims = x1.ndim - cos.ndim
            seq_dim = cos.dim() - 2
            new_shape = cos.shape[:seq_dim] + (1,) * missing_dims + cos.shape[seq_dim:]
            cos = cos.reshape(*new_shape)
            sin = sin.reshape(*new_shape)

        
        # Apply the 2D rotation matrix formula:
        # [x1']   [cos  -sin] [x1]   [x1*cos - x2*sin]
        # [x2'] = [sin   cos] [x2] = [x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # rotated_x_reshaped: shape (..., seq_len, num_groups, 2)
        # Stack the rotated components back together.
        rotated_x_reshaped = torch.stack([rotated_x1, rotated_x2], dim=-1)
        
        # rotated_x: shape (..., seq_len, d_k)
        # Reshape the tensor back to its original input dimensionality.
        rotated_x = rotated_x_reshaped.reshape(*x.shape[:-1], self.d_k)

        return rotated_x
    
def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
     # Stabilize input by subtracting the maximum value along the specified dimension to prevent overflow
    row_max = in_features.max(dim=dim, keepdim=True)[0]
    in_features_stable = in_features - row_max
    
    # Compute exponentials of the stabilized input and normalize by the sum along the specified dimension
    in_features_exp = torch.exp(in_features_stable)
    row_exp_sum = in_features_exp.sum(dim=dim, keepdim=True)
    softmax_result = in_features_exp / row_exp_sum
   
    return softmax_result

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scaling_factor = K.shape[-1] ** 0.5
    scores = (Q @ K.transpose(-2,-1)) / scaling_factor
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    softmax_scores = softmax(scores, dim=-1)

    scaled_attention = softmax_scores @ V

    return scaled_attention


class MultiheadSelfAttention(torch.nn.Module):
    """
Given the key, query, and value projection weights of a naive unbatched
implementation of multi-head attention, return the output of an optimized batched
implementation. This implementation should handle the key, query, and value projections
for all heads in a single matrix multiply.
This function should not use RoPE.
See section 3.2.2 of Vaswani et al., 2017.

Args:
    d_model (int): Dimensionality of the feedforward input and output.
    num_heads (int): Number of heads to use in multi-headed attention.
    max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
    q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
    k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
    v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
    o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
    in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

Returns:
    Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
    implementation with the given QKV projection weights and input features.
"""
    def __init__(self, d_model: int, num_heads: int ):
        super().__init__()
        self.Q_weights = Linear(d_model, d_model)
        self.K_weights = Linear(d_model, d_model)
        self.V_weights = Linear(d_model, d_model)
        self.O_weights = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads
       


    def forward(self, in_features: Float[Tensor, " ... sequence_length d_in"]):
        """
        Forward pass for multi-head self-attention.
        
        Args:
            in_features: Input tensor with shape [..., sequence_length, d_in]
                        where ... represents arbitrary batch dimensions
            
        Returns:
            Output tensor with same shape as input: [..., sequence_length, d_model]
        """
        # Step 1: Project input to Q, K, V spaces
        # in_features: [..., seq_len, d_in] 
        # Q/K/V: [..., seq_len, d_model]
        Q, K, V = self.Q_weights(in_features), self.K_weights(in_features), self.V_weights(in_features)
        Q, K, V = Q.float(), K.float(), V.float()
        
        # Extract dimensions for reshaping
        # seq_len: sequence length
        # pre_dim: tuple of preceding dimensions [...]
        seq_len = Q.shape[-2]  # seq_len
        pre_dim = Q.shape[:-2]  # [...]
        
        # Step 2: Reshape and transpose for multi-head attention
        # Original: [..., seq_len, d_model]
        # Reshaped: [..., seq_len, num_heads, d_k]
        # Transposed: [..., num_heads, seq_len, d_k]
        Q_reshaped = Q.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K_reshaped = K.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V_reshaped = V.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        
        # Q_reshaped/K_reshaped/V_reshaped shape: [..., num_heads, seq_len, d_k]
        
        # Step 3: Create causal mask for self-attention
        # mask shape: [seq_len, seq_len] - lower triangular matrix
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        
        # Step 4: Apply scaled dot-product attention
        # Input shapes:
        #   Q: [..., num_heads, seq_len, d_k]
        #   K: [..., num_heads, seq_len, d_k]  
        #   V: [..., num_heads, seq_len, d_k]
        #   mask: [seq_len, seq_len]
        # Output shape: [..., num_heads, seq_len, d_k]
        attn_output = scaled_dot_product_attention(Q_reshaped, K_reshaped, V_reshaped, mask)
        
        # Step 5: Transpose back and reshape to original dimensions
        # Transpose: [..., seq_len, num_heads, d_k]
        # Reshape: [..., seq_len, d_model]
        attn_transposed = attn_output.transpose(-3, -2).reshape(*pre_dim, seq_len, self.d_model)
        
        # Step 6: Apply output projection
        # Input: [..., seq_len, d_model]
        # Output: [..., seq_len, d_model]
        return self.O_weights(attn_transposed)
    
class MultiheadSelfAttentionWithRope(torch.nn.Module):
    """
Given the key, query, and value projection weights of a naive unbatched
implementation of multi-head attention, return the output of an optimized batched
implementation. This implementation should handle the key, query, and value projections
for all heads in a single matrix multiply.
This function should not use RoPE.
See section 3.2.2 of Vaswani et al., 2017.

Args:
    d_model (int): Dimensionality of the feedforward input and output.
    num_heads (int): Number of heads to use in multi-headed attention.
    max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
    q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
    k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
    v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
    o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
    in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

Returns:
    Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
    implementation with the given QKV projection weights and input features.
"""
    def __init__(self, d_model: int, num_heads: int ):
        super().__init__()
        self.Q_weights = Linear(d_model, d_model)
        self.K_weights = Linear(d_model, d_model)
        self.V_weights = Linear(d_model, d_model)
        self.O_weights = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads
       


    def forward(self, in_features: Float[Tensor, " ... sequence_length d_in"], max_seq_len: int, theta: float, token_positions:Tensor| None = None):
        """
        Forward pass for multi-head self-attention.
        
        Args:
            in_features: Input tensor with shape [..., sequence_length, d_in]
                        where ... represents arbitrary batch dimensions
            
        Returns:
            Output tensor with same shape as input: [..., sequence_length, d_model]
        """
        # Step 1: Project input to Q, K, V spaces
        # in_features: [..., seq_len, d_in] 
        # Q/K/V: [..., seq_len, d_model]
        Q, K, V = self.Q_weights(in_features), self.K_weights(in_features), self.V_weights(in_features)
        Q, K, V = Q.float(), K.float(), V.float()
        
        # Extract dimensions for reshaping
        # seq_len: sequence length
        # pre_dim: tuple of preceding dimensions [...]
        seq_len = Q.shape[-2]  # seq_len
        pre_dim = Q.shape[:-2]  # [...]
        
        # Step 2: Reshape and transpose for multi-head attention
        # Original: [..., seq_len, d_model]
        # Reshaped: [..., seq_len, num_heads, d_k]
        # Transposed: [..., num_heads, seq_len, d_k]
        Q_reshaped = Q.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K_reshaped = K.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V_reshaped = V.reshape(*pre_dim, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        
        rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, in_features.device)

        if token_positions is None:
            seq_len = in_features.shape[-2]
            token_positions = torch.arange(seq_len, device=in_features.device).expand(in_features.shape[:-1])
        rope_Q = rope(Q_reshaped, token_positions)
        rope_K = rope(K_reshaped, token_positions)
        # Q_reshaped/K_reshaped/V_reshaped shape: [..., num_heads, seq_len, d_k]
        
        # Step 3: Create causal mask for self-attention
        # mask shape: [seq_len, seq_len] - lower triangular matrix
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        
        # Step 4: Apply scaled dot-product attention
        # Input shapes:
        #   Q: [..., num_heads, seq_len, d_k]
        #   K: [..., num_heads, seq_len, d_k]  
        #   V: [..., num_heads, seq_len, d_k]
        #   mask: [seq_len, seq_len]
        # Output shape: [..., num_heads, seq_len, d_k]
        attn_output = scaled_dot_product_attention(rope_Q, rope_K, V_reshaped, mask)
        
        # Step 5: Transpose back and reshape to original dimensions
        # Transpose: [..., seq_len, num_heads, d_k]
        # Reshape: [..., seq_len, d_model]
        attn_transposed = attn_output.transpose(-3, -2).reshape(*pre_dim, seq_len, self.d_model)
        
        # Step 6: Apply output projection
        # Input: [..., seq_len, d_model]
        # Output: [..., seq_len, d_model]
        return self.O_weights(attn_transposed)

class TransformerBlock(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            weights: Optional[dict[str, Tensor]]=None,
    ):
        super().__init__()
        self.rms_1 = RMSNorm(d_model)
        self.rms_2 = RMSNorm(d_model)

        self.mha = MultiheadSelfAttentionWithRope(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)

        with torch.no_grad():
            if weights is not None and len(weights) > 0:
                self.rms_1.load_state_dict({"W": weights["ln1.weight"]})
                self.rms_2.load_state_dict({"W": weights["ln2.weight"]})

                self.mha.load_state_dict({
                "Q_weights.W": weights["attn.q_proj.weight"],
                "K_weights.W": weights["attn.k_proj.weight"],
                "V_weights.W": weights["attn.v_proj.weight"],
                "O_weights.W": weights["attn.output_proj.weight"]
            })
                self.ffn.load_state_dict({
                "w1.W": weights["ffn.w1.weight"],
                "w2.W": weights["ffn.w2.weight"],
                "w3.W": weights["ffn.w3.weight"]
            })
                    

    def forward(self, in_features, max_seq_len, theta, token_positions: Tensor| None = None ):
        rms1_output = self.rms_1(in_features)
        mha_output = self.mha(rms1_output, max_seq_len, theta, token_positions)

        residual_1 = in_features + mha_output

        rms2_output = self.rms_2(residual_1)
        ffn_output = self.ffn(rms2_output)

        return ffn_output + residual_1
    
class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, vocab_size, weights: Optional[dict[str, Tensor]]=None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.token_embedding = Embedding(vocab_size, d_model)
        if weights is None:
            weights = {}
        for i in range(num_layers):
            prefix = f"layers.{i}."
            block_weights = {
                k.replace(prefix, ""): v for k, v in weights.items() if k.startswith(prefix)
            }

            transformer_block = TransformerBlock(d_model, num_heads, d_ff, block_weights)

            self.layers.append(transformer_block)

        self.final_rms = RMSNorm(d_model)

        self.output_embedding = Linear(d_model, vocab_size)

        with torch.no_grad():
            if len(weights) > 0:
                self.final_rms.W.copy_(weights["ln_final.weight"])
                self.token_embedding.embedding.copy_(weights["token_embeddings.weight"])
                self.output_embedding.W.copy_(weights["lm_head.weight"])

    def forward(self, in_indices, context_length, rope_theta):
        x = self.token_embedding(in_indices)
        for layer in self.layers:
            x = layer(x, context_length, rope_theta)

        x = self.final_rms(x)
        logits = self.output_embedding(x)

        return logits


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    inputs = inputs.view(-1, inputs.shape[-1])
    targets = targets.view(-1)

    row_max = inputs.max(dim=-1, keepdim = True)[0]
    adjusted_logits = inputs - row_max
    first_part = torch.log(torch.exp(adjusted_logits).sum(dim=-1, keepdim=False))
    second_part = adjusted_logits[torch.arange(len(targets)), targets]
    ce_res = first_part - second_part

    return torch.mean(ce_res)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <=betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <=betas[1] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
            }
        super().__init__(params, defaults)
    
    def step(self, closure = None):
        if closure is not None:
            loss =  closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['betas'][0]
            beta2 = group['betas'][1]
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                
                if p not in self.state:
                    self.state[p] = {}
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']

                state['exp_avg'].mul_(beta1).add_(grad, alpha=1-beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                denom = state['exp_avg_sq'].sqrt().add_(eps)

                step_size = (lr * bias_correction2 ** 0.5) / bias_correction1
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        lr = it/warmup_iters * max_learning_rate
    elif cosine_cycle_iters >= it:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_norm_sq = 0
    for p in parameters:
        if p.grad is not None:
            grad = p.grad
            total_norm_sq += grad.norm(2) ** 2
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    max_id = len(dataset) - context_length -1
    indices= np.random.randint(low = 0, high = max_id+1, size = batch_size)
    assert max_id >= 0, f"(Dataset is too short ! Now: {len(dataset)}), need at least {context_length + 1}!"

    x_list = []
    y_list = []

    for i in indices:
        x_seq = dataset[i: i+context_length]
        y_seq = dataset[i+1: i+context_length+1]
        x_list.append(x_seq)
        y_list.append(y_seq)

    x_np = np.stack(x_list)
    y_np = np.stack(y_list)

    x = torch.tensor(x_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.long)

    x = x.to(device)
    y = y.to(device)

    return x, y

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint_data = {
        "model_state": model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(checkpoint_data, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint_data = torch.load(src, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint_data["model_state"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state"])

    return checkpoint_data["iteration"]
















            







    
    

