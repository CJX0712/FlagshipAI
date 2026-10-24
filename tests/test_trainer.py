"""训练器测试 —— 批次构造、调度、评估、收敛性。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from morningstar.flash import GPT, GPTConfig
from morningstar.flash.trainer import (
    TrainConfig,
    cosine_schedule,
    evaluate,
    make_batches,
    perplexity,
    save_log,
    train,
)


def test_make_batches_shapes_and_alignment():
    rng = np.random.default_rng(0)
    tokens = np.arange(1000, dtype=np.int64)
    x, y = make_batches(tokens, block_size=16, batch_size=4, rng=rng)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    # y 必须是 x 左移一位
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_make_batches_rejects_short_corpus():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="语料太短"):
        make_batches(np.arange(5), block_size=16, batch_size=2, rng=rng)


def test_make_batches_deterministic():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    tokens = np.arange(500, dtype=np.int64)
    x1, y1 = make_batches(tokens, 16, 4, rng_a)
    x2, y2 = make_batches(tokens, 16, 4, rng_b)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_cosine_schedule_warmup_and_decay():
    cfg_lr, warmup, total = 3e-3, 10, 100
    assert cosine_schedule(0, total, warmup=warmup, base_lr=cfg_lr) < cfg_lr
    peak = cosine_schedule(warmup, total, warmup=warmup, base_lr=cfg_lr)
    assert abs(peak - cfg_lr) < 1e-12
    end = cosine_schedule(total, total, warmup=warmup, base_lr=cfg_lr, min_lr_ratio=0.1)
    assert abs(end - cfg_lr * 0.1) < 1e-9


def test_cosine_schedule_is_monotone_after_warmup():
    vals = [cosine_schedule(s, 100, warmup=10, base_lr=1.0) for s in range(10, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_perplexity():
    assert abs(perplexity(0.0) - 1.0) < 1e-12
    assert abs(perplexity(math.log(10)) - 10.0) < 1e-9


def test_evaluate_returns_finite_loss():
    cfg = GPTConfig(vocab_size=32, d_model=32, n_layer=1, n_head=4, n_kv_head=2, max_len=32)
    m = GPT(cfg, seed=3)
    tokens = np.random.default_rng(0).integers(0, 32, size=2000)
    tcfg = TrainConfig(n_steps=1, batch_size=4, block_size=16, eval_iters=2)
    v = evaluate(m, tokens, tcfg, np.random.default_rng(1))
    assert math.isfinite(v)
    assert v > 0


def test_train_reduces_loss_on_structured_corpus():
    """在一阶马尔可夫链上训练, 损失必须显著下降。"""
    rng = np.random.default_rng(7)
    S, G, V = 8, 8, 64
    trans = np.full((S, S), 0.05)
    for i in range(S):
        trans[i, (i + 1) % S] = 0.45
        trans[i, i] = 0.30
    trans /= trans.sum(1, keepdims=True)
    state = np.empty(6000, dtype=np.int64)
    state[0] = rng.integers(0, S)
    for t in range(1, 6000):
        state[t] = rng.choice(S, p=trans[state[t - 1]])
    tokens = (state * G + rng.integers(0, G, size=6000)).astype(np.int64)

    cfg = GPTConfig(vocab_size=V, d_model=48, n_layer=2, n_head=4, n_kv_head=2, max_len=32)
    m = GPT(cfg, seed=11)
    tcfg = TrainConfig(
        n_steps=150, batch_size=16, block_size=24, lr=3e-3, warmup=15,
        eval_every=50, eval_iters=3, log_every=50, seed=5,
    )
    log = train(m, tokens, tcfg, val_tokens=tokens[:1000])
    losses = [x for x in log.train_loss if not math.isnan(x)]
    assert len(losses) >= 2
    assert losses[-1] < losses[0] - 0.2, f"损失未显著下降: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_train_records_validation_curve():
    """验证损失必须被真实评估并记录 (不能整段 NaN)。"""
    cfg = GPTConfig(vocab_size=16, d_model=16, n_layer=1, n_head=2, n_kv_head=1, max_len=16)
    m = GPT(cfg, seed=1)
    tokens = np.random.default_rng(0).integers(0, 16, size=800)
    tcfg = TrainConfig(n_steps=20, batch_size=4, block_size=8, eval_every=5, eval_iters=2, log_every=5)
    log = train(m, tokens, tcfg, val_tokens=tokens[:200])
    assert all(math.isfinite(x) for x in log.val_loss)
    assert len(log.steps) == len(log.train_loss) == len(log.val_loss)


def test_train_is_reproducible():
    cfg = GPTConfig(vocab_size=16, d_model=16, n_layer=1, n_head=2, n_kv_head=1, max_len=16)
    tokens = np.random.default_rng(0).integers(0, 16, size=600)
    tcfg = TrainConfig(n_steps=15, batch_size=4, block_size=8, eval_every=5, eval_iters=2, log_every=5, seed=9)
    a = train(GPT(cfg, seed=1), tokens, tcfg)
    b = train(GPT(cfg, seed=1), tokens, tcfg)
    def same(u, v):
        return all((x != x and y != y) or x == y for x, y in zip(u, v))

    assert same(a.train_loss, b.train_loss)
    assert same(a.val_loss, b.val_loss)


def test_save_log_roundtrip(tmp_path):
    cfg = GPTConfig(vocab_size=16, d_model=16, n_layer=1, n_head=2, n_kv_head=1, max_len=16)
    m = GPT(cfg, seed=1)
    tokens = np.random.default_rng(0).integers(0, 16, size=600)
    log = train(m, tokens, TrainConfig(n_steps=5, batch_size=2, block_size=8, log_every=5))
    p = tmp_path / "log.json"
    save_log(log, str(p))
    assert p.exists()
    import json

    data = json.loads(p.read_text(encoding="utf-8"))
    assert "train_loss" in data and "steps" in data
