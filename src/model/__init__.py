import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.encoder import TransformerEncoderLayer
from src.model.decoder import TransformerDecoderLayer
from typing import Optional


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

    def encode(self, src: torch.Tensor, src_pos_ids: torch.Tensor, cu_src_lens: torch.Tensor, max_src_len: int) -> torch.Tensor:
        # get the token embeddings and add positional encodings
        src_emb = self._token_embedding(src) * math.sqrt(self._d_model)
        src_emb = src_emb + self._positional_encoding(src_pos_ids)
        src_emb = self._enc_dropout(src_emb)

        # run through the encoder layers
        for layer in self._encoder_layers:
            src_emb = layer(src_emb, cu_src_lens, max_src_len)
        return src_emb

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_pos_ids: torch.Tensor,
        cu_src_lens: torch.Tensor,
        cu_tgt_lens: torch.Tensor,
        max_src_len: int,
        max_tgt_len: int,
        past_key_values=None,
        cu_key_lens: Optional[torch.Tensor] = None,
        max_key_len: Optional[int] = None,
        use_cache: bool = False,
    ):
        # get the token embeddings and add positional encodings
        tgt_emb = self._token_embedding(tgt) * math.sqrt(self._d_model)
        tgt_emb = tgt_emb + self._positional_encoding(tgt_pos_ids)
        tgt_emb = self._dec_dropout(tgt_emb)

        next_decoder_cache = () if use_cache else None
        for i, layer in enumerate(self._decoder_layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            tgt_emb, next_kv = layer(
                tgt_emb,
                memory,
                cu_src_lens,
                cu_tgt_lens,
                max_src_len,
                max_tgt_len,
                past_key_value=layer_past_kv,
                cu_key_lens=cu_key_lens,
                max_key_len=max_key_len,
                use_cache=use_cache,
            )
            if use_cache:
                next_decoder_cache += (next_kv,)  # type: ignore

        logits = self._linear(tgt_emb)

        return logits, next_decoder_cache

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pos_ids: torch.Tensor,
        tgt_pos_ids: torch.Tensor,
        cu_src_lens: torch.Tensor,
        cu_tgt_lens: torch.Tensor,
        max_src_len: int,
        max_tgt_len: int,
    ) -> torch.Tensor:
        # run through the encoder and decoder
        memory = self.encode(src, src_pos_ids, cu_src_lens, max_src_len)
        logits, _ = self.decode(memory, tgt, tgt_pos_ids, cu_src_lens, cu_tgt_lens, max_src_len, max_tgt_len)
        return logits


@torch.no_grad()
def beam_search(
    model: TransformerModule,
    src: torch.Tensor,
    src_pos_ids: torch.Tensor,
    cu_src_lens: torch.Tensor,
    max_src_len: int,
    beam_size: int,
    tolerance: int,
    sos_token_id: int,
    eos_token_id: int,
    length_penalty: float = 0.6,
):
    model.eval()
    device = src.device
    vocab_size = model._token_embedding.num_embeddings
    cu_src_lens_cpu = cu_src_lens.numpy(force=True)
    src_lens = torch.tensor(cu_src_lens_cpu[1:] - cu_src_lens_cpu[:-1], device=device).view(-1, 1)

    # encode the source sequence
    memory = model.encode(src, src_pos_ids, cu_src_lens, max_src_len)
    batch_size = cu_src_lens.size(0) - 1

    # prepare for beam search decoding
    repeated_memory = torch.zeros((beam_size * round(cu_src_lens[-1].item()), memory.size(1)), device=device)
    repeated_cu_src_lens = []
    offset = 0
    for i in range(batch_size):
        for _ in range(beam_size):
            repeated_memory[offset : offset + cu_src_lens_cpu[i + 1] - cu_src_lens_cpu[i]] = memory[cu_src_lens_cpu[i] : cu_src_lens_cpu[i + 1]]
            repeated_cu_src_lens.append(offset)
            offset += cu_src_lens_cpu[i + 1] - cu_src_lens_cpu[i]
    repeated_cu_src_lens.append(offset)
    repeated_cu_src_lens = torch.tensor(repeated_cu_src_lens, device=device, dtype=torch.int32)

    # initialize the beam search variables
    sequences = torch.full((batch_size, beam_size, 1), sos_token_id, dtype=torch.int32, device=device).view(batch_size * beam_size, 1)
    sequence_scores = torch.zeros((batch_size, beam_size), device=device)
    sequence_scores[:, 1:] = -1e9  # set scores of non-initial beams to a very low value
    sequence_scores = sequence_scores.view(-1)
    effective_seqlens = torch.ones((batch_size, beam_size), device=device, dtype=torch.float32).view(-1)

    is_finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device).view(-1)

    # KV Cache for decoder layers
    past_key_values = None

    for step in range(max_src_len + tolerance):
        current_tokens = sequences[:, -1]
        pos_tgt = torch.full((batch_size * beam_size,), step, dtype=torch.int32, device=device)
        cu_tgt_lens = torch.arange(0, batch_size * beam_size + 1, dtype=torch.int32, device=device)
        cu_key_lens = torch.arange(0, (batch_size * beam_size + 1) * (step + 1), step=step + 1, dtype=torch.int32, device=device)
        max_seqlen_tgt = 1
        max_key_len = step + 1

        logits, next_key_values = model.decode(
            repeated_memory,
            current_tokens,
            pos_tgt,
            repeated_cu_src_lens,
            cu_tgt_lens,
            max_src_len,
            max_seqlen_tgt,
            past_key_values=past_key_values,
            cu_key_lens=cu_key_lens,
            max_key_len=max_key_len,
            use_cache=True,
        )

        next_token_logprobs = F.log_softmax(logits, dim=-1)
        next_token_logprobs[is_finished, :] = -1e9
        next_token_logprobs[is_finished, eos_token_id] = 0.0

        next_scores = sequence_scores.unsqueeze(1) + next_token_logprobs

        added_lengths = (~is_finished).float().unsqueeze(1)
        next_lengths = effective_seqlens.unsqueeze(1) + added_lengths
        # next_lengths = next_lengths.repeat(1, vocab_size)
        penalized_scores = next_scores / (((5 + next_lengths) / 6) ** length_penalty)

        next_penalized_scores_flat = penalized_scores.view(batch_size, -1)
        _, topk_indices = torch.topk(next_penalized_scores_flat, beam_size, dim=1, sorted=True)

        beam_indices = topk_indices // vocab_size
        next_token_indices = topk_indices % vocab_size

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        sequence_scores = next_scores.view(batch_size, -1)[batch_idx, topk_indices]
        sequence_scores = sequence_scores.view(-1)

        effective_seqlens = next_lengths.view(batch_size, beam_size)[batch_idx, beam_indices]
        is_finished = is_finished.view(batch_size, beam_size)[batch_idx, beam_indices] | (next_token_indices == eos_token_id) | (effective_seqlens > src_lens + tolerance)

        effective_seqlens = effective_seqlens.view(-1)
        is_finished = is_finished.view(-1)

        prev_seqs = sequences.view(batch_size, beam_size, -1)[batch_idx, beam_indices]
        sequences = torch.cat([prev_seqs, next_token_indices.unsqueeze(-1)], dim=-1)
        sequences = sequences.view(batch_size * beam_size, -1)

        if next_key_values is not None:
            past_key_values = reorder_kv_cache_batched(next_key_values, beam_indices, batch_size)

        if is_finished.all():
            break

    best_sequences = []
    for i in range(batch_size):
        batch_beams = sequences[i * beam_size : (i + 1) * beam_size]
        batch_scores = sequence_scores[i * beam_size : (i + 1) * beam_size]
        batch_effective_seqlens = effective_seqlens[i * beam_size : (i + 1) * beam_size]
        best_beam_idx = torch.argmax(batch_scores)
        best_sequence = batch_beams[best_beam_idx]
        best_sequences.append(best_sequence.numpy(force=True)[:round(batch_effective_seqlens[best_beam_idx].item())])
    return best_sequences


def reorder_kv_cache_batched(past_key_values, beam_indices, batch_size):
    reordered_cache = ()
    beam_size = beam_indices.size(1)
    
    batch_idx = torch.arange(batch_size, device=beam_indices.device).unsqueeze(1)

    for layer_cache in past_key_values:
        self_kv_cache, cross_kv_cache = layer_cache
        reordered_self_kv_cache = None

        if self_kv_cache is not None:
            k, v = self_kv_cache

            num_heads = k.size(1)
            head_dim = k.size(2)

            current_seqlen = k.size(0) // (batch_size * beam_size)
            
            k = k.view(batch_size, beam_size, current_seqlen, num_heads, head_dim)
            v = v.view(batch_size, beam_size, current_seqlen, num_heads, head_dim)

            k_reordered = k[batch_idx, beam_indices]
            v_reordered = v[batch_idx, beam_indices]

            k_reordered = k_reordered.view(-1, num_heads, head_dim)
            v_reordered = v_reordered.view(-1, num_heads, head_dim)

            reordered_self_kv_cache = (k_reordered, v_reordered)

        reordered_cache += ((reordered_self_kv_cache, cross_kv_cache),)
        
    return reordered_cache