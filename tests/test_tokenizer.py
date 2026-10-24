"""BPE 分词器测试 —— 往返无损、确定性、边界处理。"""
from __future__ import annotations

import numpy as np
import pytest

from morningstar.flash.tokenizer import BPETokenizer, compression_ratio, token_frequencies

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "晨星锻造世界顶级人工智能系统，零重写、集大成、可验证。",
    "Transformers scale: attention is all you need, but memory is the wall.",
    "RMSNorm, RoPE, SwiGLU — the Llama recipe.",
] * 8


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer().train(CORPUS, vocab_size=300)


def test_roundtrip_preserves_text(tok):
    for text in CORPUS[:4]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"往返失败: {text!r}"


def test_encode_is_deterministic(tok):
    a = tok.encode("attention is all you need")
    b = tok.encode("attention is all you need")
    assert a == b


def test_train_is_deterministic():
    t1 = BPETokenizer().train(CORPUS, vocab_size=300)
    t2 = BPETokenizer().train(CORPUS, vocab_size=300)
    assert t1.merges == t2.merges
    assert t1.vocab_size == t2.vocab_size


def test_vocab_size_respected(tok):
    assert tok.vocab_size <= 300


def test_min_vocab_size_guard():
    with pytest.raises(ValueError, match="vocab_size"):
        BPETokenizer().train(CORPUS, vocab_size=16)


def test_encode_empty_string(tok):
    assert tok.encode("") == []


def test_encode_batch(tok):
    batch = tok.encode_batch(["hello", "world"])
    assert len(batch) == 2
    assert all(isinstance(x, list) for x in batch)


def test_save_load_roundtrip(tmp_path, tok):
    path = str(tmp_path / "tok.json")
    tok.save(path)
    t2 = BPETokenizer.load(path)
    text = "晨星 FlagshipAI: 零重写"
    assert t2.encode(text) == tok.encode(text)
    assert t2.decode(t2.encode(text)) == text


def test_compression_ratio(tok):
    ratio = compression_ratio(tok, CORPUS)  # 签名: (tokenizer, texts)
    assert ratio > 0


def test_token_frequencies(tok):
    freq = token_frequencies(CORPUS, tok)
    assert len(freq) > 0
    assert sum(freq.values()) > 0


def test_unknown_chars_fallback(tok):
    """未见过的字符必须能安全编码 (回退到字节级)。"""
    ids = tok.encode("🚀🌟")
    assert len(ids) > 0
    assert tok.decode(ids) == "🚀🌟"
