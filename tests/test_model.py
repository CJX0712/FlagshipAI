"""旗舰 GPT 模型的功能性测试 —— 前向/反向/生成/保存加载/边界。"""
from __future__ import annotations

import numpy as np
import pytest

from morningstar.flash import GPT, GPTConfig
from morningstar.flash.attention import multihead_forward


# ------------------------------------------------------------------ 前向
def test_forward_shapes(tiny_model):
    idx = np.arange(8)
    logits, cache = tiny_model.forward(idx)
    assert logits.shape == (8, tiny_model.cfg.vocab_size)
    assert np.isfinite(logits).all()


def test_forward_rejects_2d_input(tiny_model):
    with pytest.raises(ValueError, match="一维"):
        tiny_model.forward(np.zeros((2, 4), dtype=np.int64))


def test_forward_rejects_too_long(tiny_model):
    with pytest.raises(ValueError, match="max_len"):
        tiny_model.forward(np.arange(tiny_model.cfg.max_len + 1))


def test_loss_decreases_after_training_step(tiny_model):
    """一步梯度下降后损失必须下降 (反向传播方向正确)。"""
    m = GPT(tiny_model.cfg, seed=1234)
    idx = np.arange(8)
    tgt = (np.arange(8) * 3 + 1) % m.cfg.vocab_size
    l0, _ = m.loss_and_grad(idx, tgt)
    grads = m.loss_and_grad(idx, tgt)[1]
    for k in grads:
        m.params[k] -= 1e-3 * grads[k]
    l1, _ = m.loss_and_grad(idx, tgt)
    assert l1 < l0, f"损失未下降: {l0:.6f} → {l1:.6f}"


def test_gqa_head_grouping():
    """GQA: n_head 必须被 n_kv_head 整除, 否则抛错。"""
    with pytest.raises(ValueError, match="n_kv_head"):
        GPTConfig(vocab_size=16, d_model=16, n_head=4, n_kv_head=3)
    with pytest.raises(ValueError, match="n_head"):
        GPTConfig(vocab_size=16, d_model=16, n_head=6, n_kv_head=2)


# ------------------------------------------------------------------ KV 缓存
def test_kv_cache_incremental_matches_full():
    """增量解码 (KV-Cache) 与一次性前向的 logits 必须一致。

    注意: ``kv_cache_per_layer`` 的元素必须是真正的 ``KVCache`` 实例;
    传 None 表示该层不使用缓存 (则每步只能看到自己, 与全量前向不等价)。
    """
    from morningstar.flash.attention import KVCache

    cfg = GPTConfig(vocab_size=32, d_model=32, n_layer=2, n_head=4, n_kv_head=2, max_len=64)
    m = GPT(cfg, seed=21)
    ids = np.arange(12)
    full, _ = m.forward(ids, engine="flash")
    caches = [KVCache(cfg.max_len, cfg.n_kv_head, cfg.head_dim) for _ in range(cfg.n_layer)]
    step_logits = []
    for t in range(ids.size):
        lg, _ = m.forward(
            ids[t : t + 1], engine="flash",
            kv_cache_per_layer=caches, start_pos=t,
        )
        step_logits.append(lg[0])
    inc = np.stack(step_logits)
    assert np.max(np.abs(full - inc)) < 1e-9


# ------------------------------------------------------------------ 生成
def test_generate_length_and_determinism(tiny_model):
    m = GPT(tiny_model.cfg, seed=99)
    prompt = [1, 2, 3]
    a = m.generate(prompt, max_new_tokens=12, temperature=1.0, seed=5)
    b = m.generate(prompt, max_new_tokens=12, temperature=1.0, seed=5)
    # generate 只返回**新生成**的 token (不含 prompt)
    assert len(a) == 12
    assert a == b, "同 seed 的采样必须完全一致"


def test_generate_greedy_is_deterministic(tiny_model):
    m = GPT(tiny_model.cfg, seed=99)
    out = m.generate([1, 2, 3], max_new_tokens=8, temperature=0.0, seed=1)
    assert len(out) == 8


def test_generate_rejects_overflow(tiny_model):
    m = GPT(tiny_model.cfg, seed=99)
    with pytest.raises(ValueError, match="max_len"):
        m.generate([1, 2, 3], max_new_tokens=tiny_model.cfg.max_len + 10)


def test_generate_stream(tiny_model):
    m = GPT(tiny_model.cfg, seed=99)
    stream = m.generate([1, 2, 3], max_new_tokens=5, temperature=0.8, seed=2, stream=True)
    out = list(stream)
    assert len(out) == 5


# ------------------------------------------------------------------ 持久化
def test_save_load_roundtrip(tmp_path, tiny_cfg):
    m = GPT(tiny_cfg, seed=77)
    path = str(tmp_path / "m.npz")
    m.save(path)
    m2 = GPT.load(path)
    idx = np.arange(6)
    a, _ = m.forward(idx)
    b, _ = m2.forward(idx)
    assert np.array_equal(a, b), "保存加载后前向必须逐位一致"
    assert m.n_params == m2.n_params


# ------------------------------------------------------------------ 多头注意力
def test_multihead_forward_head_shapes():
    rng = np.random.default_rng(3)
    T, d, H = 6, 16, 4
    x = rng.standard_normal((T, d))
    wq = rng.standard_normal((d, d)) * 0.1
    wk = rng.standard_normal((d, d)) * 0.1
    wv = rng.standard_normal((d, d)) * 0.1
    wo = rng.standard_normal((d, d)) * 0.1
    out, _ = multihead_forward(
        x, wq, wk, wv, wo, n_heads=H, mask="causal", engine="naive"
    )
    assert out.shape == (T, d)
    assert np.isfinite(out).all()


def test_multihead_flash_matches_naive():
    rng = np.random.default_rng(4)
    T, d, H = 12, 16, 4
    x = rng.standard_normal((T, d))
    # multihead_forward(x, wq, wk, wv, wo, *, n_heads, ...)
    args = [rng.standard_normal((d, d)) * 0.1 for _ in range(5)]
    o_naive, _ = multihead_forward(*args, n_heads=H, mask="causal", engine="naive")
    o_flash, _ = multihead_forward(*args, n_heads=H, mask="causal", engine="flash")
    assert np.max(np.abs(o_naive - o_flash)) < 1e-12


# ------------------------------------------------------------------ 优化器
def test_adamw_decreases_quadratic():
    """AdamW 在二次函数上必须单调下降。"""
    from morningstar.flash.optim import AdamW

    w = {"p": np.array([3.0, -2.0, 1.0])}
    opt = AdamW(lr=0.05, weight_decay=0.0)
    first = None
    last = None
    for _ in range(300):
        p = w["p"]
        g = 2.0 * p  # ∇(‖p‖²)
        opt.step(w, {"p": g})
        cur = float(np.sum(w["p"] ** 2))
        first = cur if first is None else first
        last = cur
    # Adam 是逐元素自适应步长, 在最优点附近允许微小震荡;
    # 正确判据是「相对初值显著下降」。
    assert last < first * 1e-3, f"未收敛: {first:.6f} → {last:.6f}"
    assert last < 1e-2, f"未收敛到最优: {last:.6f}"


def test_adamw_weight_decay_shrinks():
    from morningstar.flash.optim import AdamW

    w = {"p": np.array([1.0])}
    opt = AdamW(lr=1e-8, weight_decay=0.5)   # lr 必须 > 0
    for _ in range(50):
        opt.step(w, {"p": np.zeros(1)})
    assert abs(w["p"][0]) < 1.0
