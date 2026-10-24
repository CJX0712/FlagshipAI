"""L11 大模型内核 —— 世界级算法的手写可验证实现（forge 层）。

本层每一行都对应一篇公开发表的世界级成果，出处见 docs/REFERENCES.md:

  attention.flash_attention_forward : FlashAttention-2 分块在线 softmax
                                      (Dao 2023; Milakov & Gimelshein 2018 在线 softmax)
  attention.naive_attention_forward : Vaswani et al. 2017 原始缩放点积注意力（黄金参照）
  modules.RotaryEmbedding           : RoPE (Su et al. 2021)
  modules.RMSNorm                   : RMSNorm (Zhang & Sennrich 2019)
  modules.SwiGLUFeedForward         : SwiGLU FFN (Shazeer 2020)
  optim.AdamW                       : AdamW (Loshchilov & Hutter 2019)
  tokenizer.BPETokenizer            : BPE (Sennrich et al. 2016)
  narrator.GPT                      : Decoder-only Transformer (Radford et al. 2018;
                                      现代 pre-norm / SwiGLU 变体 Touvron et al. 2023 Llama)
"""
from __future__ import annotations

from . import attention, modules, optim, tokenizer, trainer
from .attention import (
    AttentionOutput,
    KVCache,
    flash_attention_forward,
    naive_attention_forward,
)
from .model import GPT, GPTConfig
from .modules import RotaryEmbedding, RMSNorm, SwiGLUFeedForward
from .optim import AdamW, clip_grad_norm, cosine_schedule
from .random import rng_for, set_seed
from .tokenizer import BPETokenizer
from .trainer import TrainConfig, TrainLog, evaluate, perplexity, train

__all__ = [
    "GPT",
    "GPTConfig",
    "AdamW",
    "AttentionOutput",
    "BPETokenizer",
    "KVCache",
    "RotaryEmbedding",
    "RMSNorm",
    "SwiGLUFeedForward",
    "TrainConfig",
    "TrainLog",
    "attention",
    "clip_grad_norm",
    "cosine_schedule",
    "evaluate",
    "flash_attention_forward",
    "modules",
    "naive_attention_forward",
    "optim",
    "perplexity",
    "rng_for",
    "set_seed",
    "tokenizer",
    "train",
    "trainer",
]
