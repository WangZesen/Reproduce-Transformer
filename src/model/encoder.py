import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from src.conf import Config
from src.model.common import PositionwiseFeedForward, FlashAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_feedforward: int,
                 dropout: float = 0.3,
                 bias: bool = True):
        super().__init__()
        self._self_attn = FlashAttention(d_model, num_heads, is_cross_attention=False, dropout=dropout)
        self._ffn = PositionwiseFeedForward(d_model, d_feedforward, dropout, bias=bias)
        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src: torch.Tensor, cu_src_lens: torch.Tensor, max_src_len: int) -> torch.Tensor:
        # Pre-LN
        norm_x = self._norm1(src)
        attn_output, _ = self._self_attn(
            hidden_states=norm_x,
            cu_seqlens_q=cu_src_lens,
            max_seqlen_q=max_src_len,
            is_causal=False
        )
        src = src + attn_output
        src = src + self._ffn(self._norm2(src))
        return src
