import torch.nn as nn
import torch
from typing import Optional, Tuple, cast
from flash_attn import flash_attn_varlen_func


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

        self._q_proj = nn.Linear(d_model, d_model, bias=True)
        self._k_proj = nn.Linear(d_model, d_model, bias=True)
        self._v_proj = nn.Linear(d_model, d_model, bias=True)
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
        q = self._q_proj(hidden_states).view(-1, self._num_heads, self._d_head)

        if self._is_cross_attention and encoder_hidden_states is not None:
            k = self._k_proj(encoder_hidden_states).view(-1, self._num_heads, self._d_head)
            v = self._v_proj(encoder_hidden_states).view(-1, self._num_heads, self._d_head)
        else:
            k = self._k_proj(hidden_states).view(-1, self._num_heads, self._d_head)
            v = self._v_proj(hidden_states).view(-1, self._num_heads, self._d_head)

        if past_key_value is not None:
            k_cache, v_cache = past_key_value
            if self._is_cross_attention:
                k, v = k_cache, v_cache
            else:
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
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k if max_seqlen_k is not None else max_seqlen_q,
            causal=is_causal,
            dropout_p=self._dropout_rate if self.training else 0,
        )
        attn_output = cast(torch.Tensor, attn_output)
        attn_output = attn_output.view(-1, self._d_model)
        attn_output = self._out_proj(attn_output)
        return attn_output, past_key_value
