import math
import torch
import torch.nn as nn
from src.model.encoder import TransformerEncoderLayer
from src.model.decoder import TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]  # type: ignore


def generate_position_ids_from_cu_seqlens(cu_seqlens) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    start_idx = cu_seqlens[:-1]
    block_starts = torch.repeat_interleave(start_idx, seqlens)
    total_len = cu_seqlens[-1]
    global_position_ids = torch.arange(total_len, device=cu_seqlens.device)
    position_ids = global_position_ids - block_starts
    return position_ids


class TransformerModule(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_feedforward: int,
        dropout: float = 0.3,
        bias: bool = True,
    ):
        super().__init__()
        self._d_model = d_model
        self._token_embedding = nn.Embedding(vocab_size, d_model)
        self._positional_encoding = PositionalEncoding(d_model)
        self._enc_dropout = nn.Dropout(dropout)
        self._dec_dropout = nn.Dropout(dropout)

        self._encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout, bias) for _ in range(num_layers)]
        )
        self._decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, d_feedforward, dropout, bias) for _ in range(num_layers)]
        )

        self._linear = nn.Linear(d_model, vocab_size, bias=False)
        self._linear.weight = self._token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self._token_embedding.weight.data.uniform_(-initrange, initrange)

    def encode(self, src: torch.Tensor, cu_src_lens: torch.Tensor, max_src_len: int) -> torch.Tensor:
        for layer in self._encoder_layers:
            src = layer(src, cu_src_lens, max_src_len)
        return src

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        cu_src_lens: torch.Tensor,
        cu_tgt_lens: torch.Tensor,
        max_src_len: int,
        max_tgt_len: int,
        past_key_values=None,
        use_cache: bool = False,
    ):
        next_decoder_cache = () if use_cache else None
        for i, layer in enumerate(self._decoder_layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            tgt, next_kv = layer(
                tgt,
                memory,
                cu_src_lens,
                cu_tgt_lens,
                max_src_len,
                max_tgt_len,
                past_key_value=layer_past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                next_decoder_cache += (next_kv,)  # type: ignore

        logits = self._linear(tgt)

        return logits, next_decoder_cache

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        cu_src_lens: torch.Tensor,
        cu_tgt_lens: torch.Tensor,
        max_src_len: int,
        max_tgt_len: int,
    ) -> torch.Tensor:
        # generate position ids for source and target sequences
        src_position_ids = generate_position_ids_from_cu_seqlens(cu_src_lens)
        tgt_position_ids = generate_position_ids_from_cu_seqlens(cu_tgt_lens)

        # get the token embeddings and add positional encodings
        src_emb = self._token_embedding(src) * math.sqrt(self._d_model)
        src_emb = src_emb + self._positional_encoding(src_position_ids)
        src_emb = self._enc_dropout(src_emb)
        tgt_emb = self._token_embedding(tgt) * math.sqrt(self._d_model)
        tgt_emb = tgt_emb + self._positional_encoding(tgt_position_ids)
        tgt_emb = self._dec_dropout(tgt_emb)

        # run through the encoder and decoder
        memory = self.encode(src_emb, cu_src_lens, max_src_len)
        logits, _ = self.decode(memory, tgt_emb, cu_src_lens, cu_tgt_lens, max_src_len, max_tgt_len)

        return logits
