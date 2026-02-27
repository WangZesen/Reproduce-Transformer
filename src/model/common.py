import torch.nn as nn
import torch
from typing import Optional, Tuple, cast
from flash_attn import flash_attn_varlen_func, flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_feedforward: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_feedforward, bias=bias)
        self.w_2 = nn.Linear(d_feedforward, d_model, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.dropout2(self.w_2(self.activation(self.dropout1(self.w_1(x)))))


class FlashAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, is_cross_attention: bool, dropout: float = 0.1):
        super().__init__()
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_head = d_model // num_heads
        self._is_cross_attention = is_cross_attention
        self._dropout_rate = dropout

        # For self-attention, fuse Q+K+V into one projection so the GEMM is
        # (total_tokens, d_model) @ (d_model, 3*d_model) instead of two smaller
        # matmuls, which gives better SM utilisation.
        # For cross-attention the queries still come from a different source, so
        # we keep separate projections there.
        if is_cross_attention:
            self._q_proj = nn.Linear(d_model, d_model, bias=True)
            self._kv_proj = nn.Linear(d_model, d_model * 2, bias=True)
            self._qkv_proj = None
        else:
            self._qkv_proj = nn.Linear(d_model, d_model * 3, bias=True)
            self._q_proj = None  # unused for self-attention
            self._kv_proj = None  # unused for self-attention
        self._out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        is_causal: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        dropout_p = self._dropout_rate if self.training else 0.0
        cu_seqlens_k_ = cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q
        max_seqlen_k_ = max_seqlen_k if max_seqlen_k is not None else max_seqlen_q

        if self._is_cross_attention:
            # Cross-attention: Q from decoder hidden states, K/V from encoder
            assert encoder_hidden_states is not None
            q = self._q_proj(hidden_states).view(-1, self._num_heads, self._d_head)  # type: ignore[misc]
            kv = self._kv_proj(encoder_hidden_states).view(-1, 2, self._num_heads, self._d_head)  # type: ignore[misc]

            if past_key_value is not None:
                # Reuse cached encoder KV; kv from projection is discarded
                k_cache, v_cache = past_key_value
                k = k_cache.view(-1, self._num_heads, self._d_head)
                v = v_cache.view(-1, self._num_heads, self._d_head)
                if use_cache:
                    past_key_value = (k_cache.detach(), v_cache.detach())
                attn_output = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k_,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k_,
                    causal=is_causal,
                    dropout_p=dropout_p,
                )
            else:
                if use_cache:
                    k, v = kv.unbind(dim=1)
                    past_key_value = (k.detach(), v.detach())
                # KV is already packed (total_k, 2, num_heads, d_head) — no unbind needed
                attn_output = flash_attn_varlen_kvpacked_func(
                    q, kv,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k_,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k_,
                    causal=is_causal,
                    dropout_p=dropout_p,
                )
        else:
            # Self-attention: fused QKV projection — one large GEMM
            qkv = self._qkv_proj(hidden_states).view(-1, 3, self._num_heads, self._d_head)  # type: ignore[misc]

            if past_key_value is not None:
                q, k, v = qkv.unbind(dim=1)
                k_cache, v_cache = past_key_value
                k_cache = k_cache.view(k.size(0), -1, self._num_heads, self._d_head)
                v_cache = v_cache.view(v.size(0), -1, self._num_heads, self._d_head)
                k = (
                    torch.cat([k_cache, k.view(k.size(0), 1, self._num_heads, self._d_head)], dim=1)
                    .contiguous()
                    .view(-1, self._num_heads, self._d_head)
                )
                v = (
                    torch.cat([v_cache, v.view(v.size(0), 1, self._num_heads, self._d_head)], dim=1)
                    .contiguous()
                    .view(-1, self._num_heads, self._d_head)
                )
                if use_cache:
                    past_key_value = (k.detach(), v.detach())
                attn_output = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k_,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k_,
                    causal=is_causal,
                    dropout_p=dropout_p,
                )
            else:
                if use_cache:
                    q, k, v = qkv.unbind(dim=1)
                    past_key_value = (k.detach(), v.detach())
                # QKV is already packed (total_q, 3, num_heads, d_head) — no unbind needed
                attn_output = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=max_seqlen_q,
                    causal=is_causal,
                    dropout_p=dropout_p,
                )

        attn_output = cast(torch.Tensor, attn_output)
        attn_output = attn_output.view(-1, self._d_model)
        attn_output = self._out_proj(attn_output)
        return attn_output, past_key_value
