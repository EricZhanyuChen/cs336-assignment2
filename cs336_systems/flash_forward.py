import torch
import math
import triton
import triton.language as tl
from torch.nn import functional as F

class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, d = Q.shape
        Bk = 32
        scale = 1 / math.sqrt(d)

        m = torch.full((B, N), float('-inf'), device=Q.device)
        l = torch.zeros((B, N), device=Q.device)
        O = torch.zeros((B, N, d), device=Q.device)

        for j in range(0, N, Bk):
            K_tilde = K[:, j: j+Bk]
            V_tilde = V[:, j: j+Bk]

            S = Q @ K_tilde.transpose(-2,-1) * scale
            m_tilde = torch.max(S, dim=-1).values
            m_new = torch.maximum(m_tilde, m)

            l_tilde = torch.exp(S-m_new.unsqueeze(-1)).sum(-1)
            l_new = l_tilde + l * torch.exp(m - m_new)
            
            O_new = O * torch.exp((m-m_new).unsqueeze(-1)) + torch.exp(S-m_new.unsqueeze(-1)) @ V_tilde

            m, l, O = m_new, l_new, O_new
        L = m + torch.log(l)
        O =  O / l.unsqueeze(-1)
        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)

        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = compiled_flash_bwd(Q, K, V, O, grad_output, L, ctx.is_causal)
        return dQ, dK, dV, None
            
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1,0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    K_tile_start = 0
    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = K_tile_start + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[None, :] < k_idx[:, None]
            S = S + tl.where(mask, -1e6, 0.0)

        m_new = tl.maximum(tl.max(S, axis=1), m_i)
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(tl.exp(S-m_new[:, None]), axis=1)
        O_new = tl.exp(m_i - m_new)[:, None] * O_acc + tl.dot(tl.exp(S-m_new[:, None]).to(V_tile.dtype), V_tile)

        m_i, l_i, O_acc = m_new, l_new, O_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        K_tile_start += K_TILE_SIZE

    L_i = m_i + tl.log(l_i)
    O_acc = O_acc/l_i[:, None]

    tl.store(O_block_ptr, O_acc.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))

def flash_attention_forward(Q, K, V, is_causal=False):
    return FlashAttentionFunc.apply(Q, K, V, is_causal)

class FlashAttentionFuncTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, d = Q.shape
        O = torch.zeros(B, N, d, device=Q.device, dtype=Q.dtype)
        L = torch.zeros(B, N, device=Q.device, dtype=torch.float32)

        scale = 1 / math.sqrt(d)
        Q_TILE_SIZE, K_TILE_SIZE = 32, 32
        grid = (triton.cdiv(N, Q_TILE_SIZE), B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N, N,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
def flash_attention_backward_pytorch(Q, K, V, O, dO, L, is_causal=False):
    B, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    D  = (O * dO).sum(dim=-1)
    S = (Q @ K.transpose(-2, -1)*scale)
    P = torch.exp(S - L.unsqueeze(-1))
    
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    dS = P * (dP - D.unsqueeze(-1))

    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale
    return dQ, dK, dV

compiled_flash_bwd = torch.compile(flash_attention_backward_pytorch)