import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from src.conf import Config
from src.model.common import PositionwiseFeedForward, FlashAttention
from typing import Optional, Tuple

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_feedforward: int,
                 dropout: float = 0.3,
                 bias: bool = True):
        super().__init__()
        self._self_attn = FlashAttention(d_model, num_heads, is_cross_attention=False, dropout=dropout)
        self._cross_attn = FlashAttention(d_model, num_heads, is_cross_attention=True, dropout=dropout)
        self._ffn = PositionwiseFeedForward(d_model, d_feedforward, dropout, bias=bias)
        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)
        self._norm3 = nn.LayerNorm(d_model)
    
    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                cu_src_lens: torch.Tensor,
                cu_tgt_lens: torch.Tensor,
                max_src_len: int,
                max_tgt_len: int,
                past_key_value: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False):
        # 1. Causal self-attention
        norm_x = self._norm1(tgt)
        self_past_kv = past_key_value[0] if past_key_value is not None else None
        self_attn_output, self_kv_output = self._self_attn(
            hidden_states=norm_x,
            cu_seqlens_q=cu_tgt_lens,
            max_seqlen_q=max_tgt_len,
            is_causal=True,
            past_key_value=self_past_kv,
            use_cache=use_cache
        )
        tgt = tgt + self_attn_output

        # 2. Cross-attention
        norm_x = self._norm2(tgt)
        cross_past_kv = past_key_value[1] if past_key_value is not None else None
        cross_attn_output, cross_kv_output = self._cross_attn(
            hidden_states=norm_x,
            encoder_hidden_states=memory,
            cu_seqlens_q=cu_tgt_lens,
            cu_seqlens_k=cu_src_lens,
            max_seqlen_q=max_tgt_len,
            max_seqlen_k=max_src_len,
            is_causal=False,
            past_key_value=cross_past_kv,
            use_cache=use_cache
        )
        tgt = tgt + cross_attn_output

        # 3. FFN
        tgt = tgt + self._ffn(self._norm3(tgt))

        # 4. Update cache
        present_key_value = (self_kv_output, cross_kv_output) if use_cache else None
        return tgt, present_key_value
