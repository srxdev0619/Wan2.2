# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn)**-0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)


class GELU(nn.Module):

    def forward(self, x):
        # x: (B, *, D) - any shape ending in feature dimension
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # -> same shape as input


class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, L, D) or (B*T, D, H, W) - token embeddings or feature maps
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            self.eps)  # x: same shape as input
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x  # -> same shape as input


class T5Attention(nn.Module):

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # x: (B, L1, dim) - query sequence
        # context: (B, L2, dim) or None - key/value sequence (defaults to x for self-attention)
        # mask: (B, L2) or (B, L1, L2) or None - attention mask
        # pos_bias: (B, num_heads, L1, L2) or None - relative position bias
        # Returns: (B, L1, dim)

        # check inputs
        context = x if context is None else context  # context: (B, L2, dim)
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)  # q: (B, L1, num_heads, head_dim)
        k = self.k(context).view(b, -1, n, c)  # k: (B, L2, num_heads, head_dim)
        v = self.v(context).view(b, -1, n, c)  # v: (B, L2, num_heads, head_dim)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))  # attn_bias: (B, num_heads, L1, L2)
        if pos_bias is not None:
            attn_bias += pos_bias  # attn_bias: (B, num_heads, L1, L2)
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)  # mask: (B, 1, 1, L2) or (B, 1, L1, L2)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)  # attn_bias: (B, num_heads, L1, L2)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias  # attn: (B, num_heads, L1, L2)
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)  # attn: (B, num_heads, L1, L2)
        x = torch.einsum('bnij,bjnc->binc', attn, v)  # x: (B, L1, num_heads, head_dim)

        # output
        x = x.reshape(b, -1, n * c)  # x: (B, L1, dim_attn)
        x = self.o(x)  # x: (B, L1, dim)
        x = self.dropout(x)  # x: (B, L1, dim)
        return x  # -> (B, L1, dim)


class T5FeedForward(nn.Module):

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, dim) - input token embeddings
        x = self.fc1(x) * self.gate(x)  # x: (B, L, dim_ffn)
        x = self.dropout(x)  # x: (B, L, dim_ffn)
        x = self.fc2(x)  # x: (B, L, dim)
        x = self.dropout(x)  # x: (B, L, dim)
        return x  # -> (B, L, dim)


class T5SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        # x: (B, L, dim) - input token embeddings
        # mask: (B, L) or None - attention mask
        # pos_bias: (B, num_heads, L, L) or None - relative position bias
        # Returns: (B, L, dim)
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))  # e: (1, num_heads, L, L) or (B, num_heads, L, L)
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))  # x: (B, L, dim)
        x = fp16_clamp(x + self.ffn(self.norm2(x)))  # x: (B, L, dim)
        return x  # -> (B, L, dim)


class T5CrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def forward(self,
                x,
                mask=None,
                encoder_states=None,
                encoder_mask=None,
                pos_bias=None):
        # x: (B, L_dec, dim) - decoder input embeddings
        # mask: (B, L_dec, L_dec) or None - causal attention mask for decoder
        # encoder_states: (B, L_enc, dim) or None - encoder output
        # encoder_mask: (B, L_enc) or None - encoder attention mask
        # pos_bias: (B, num_heads, L_dec, L_dec) or None - relative position bias
        # Returns: (B, L_dec, dim)
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))  # e: (1, num_heads, L_dec, L_dec) or (B, num_heads, L_dec, L_dec)
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))  # x: (B, L_dec, dim)
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))  # x: (B, L_dec, dim)
        x = fp16_clamp(x + self.ffn(self.norm3(x)))  # x: (B, L_dec, dim)
        return x  # -> (B, L_dec, dim)


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        # lq: int - query sequence length
        # lk: int - key sequence length
        # Returns: (1, num_heads, lq, lk) - relative position bias
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)  # rel_pos: (lq, lk)
        rel_pos = self._relative_position_bucket(rel_pos)  # rel_pos: (lq, lk)
        rel_pos_embeds = self.embedding(rel_pos)  # rel_pos_embeds: (lq, lk, num_heads)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # rel_pos_embeds: (1, num_heads, lq, lk)
        return rel_pos_embeds.contiguous()  # -> (1, num_heads, lq, lk)

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        # ids: (B, L) - input token IDs
        # mask: (B, L) or None - attention mask
        # Returns: (B, L, dim) - encoded sequence
        x = self.token_embedding(ids)  # x: (B, L, dim)
        x = self.dropout(x)  # x: (B, L, dim)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None  # e: (1, num_heads, L, L) or None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)  # x: (B, L, dim)
        x = self.norm(x)  # x: (B, L, dim)
        x = self.dropout(x)  # x: (B, L, dim)
        return x  # -> (B, L, dim)


class T5Decoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        # ids: (B, L_dec) - decoder input token IDs
        # mask: (B, L_dec) or (B, L_dec, L_dec) or None - decoder attention mask
        # encoder_states: (B, L_enc, dim) or None - encoder output
        # encoder_mask: (B, L_enc) or None - encoder attention mask
        # Returns: (B, L_dec, dim) - decoded sequence
        b, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))  # mask: (1, L_dec, L_dec)
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))  # mask: (B, L_dec, L_dec)

        # layers
        x = self.token_embedding(ids)  # x: (B, L_dec, dim)
        x = self.dropout(x)  # x: (B, L_dec, dim)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None  # e: (1, num_heads, L_dec, L_dec) or None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)  # x: (B, L_dec, dim)
        x = self.norm(x)  # x: (B, L_dec, dim)
        x = self.dropout(x)  # x: (B, L_dec, dim)
        return x  # -> (B, L_dec, dim)


class T5Model(nn.Module):

    def __init__(self,
                 vocab_size,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        # encoder_ids: (B, L_enc) - encoder input token IDs
        # encoder_mask: (B, L_enc) or None - encoder attention mask
        # decoder_ids: (B, L_dec) - decoder input token IDs
        # decoder_mask: (B, L_dec) or (B, L_dec, L_dec) or None - decoder attention mask
        # Returns: (B, L_dec, vocab_size) - logits over vocabulary
        x = self.encoder(encoder_ids, encoder_mask)  # x: (B, L_enc, dim)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)  # x: (B, L_dec, dim)
        x = self.head(x)  # x: (B, L_dec, vocab_size)
        return x  # -> (B, L_dec, vocab_size)


def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=torch.float32,
        device='cpu',
        **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)

    # init tokenizer
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device).eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        # texts: list of str - input text prompts
        # device: torch.device - target device
        # Returns: list of (L_i, dim) tensors - encoded text embeddings per sample (variable length)
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)  # ids: (B, text_len), mask: (B, text_len)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()  # seq_lens: (B,) - actual sequence lengths
        context = self.model(ids, mask)  # context: (B, text_len, dim=4096 for umt5-xxl)
        return [u[:v] for u, v in zip(context, seq_lens)]  # -> list of (L_i, 4096) where L_i varies per sample
