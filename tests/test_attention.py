"""注意力家族测试 —— 统一协议下的数值一致性、复杂度与鲁棒性。"""
from __future__ import annotations

import numpy as np
import pytest

from morningstar.attention import (
    BlockSparseAttention,
    EffAttnFuse,
    FlashAttention,
    FullAttention,
    LinearAttention,
    LocalWindowAttention,
    PerformerAttention,
    build_variant,
    list_variants,
)
from morningstar.flash.attention import flash_attention_forward, naive_attention_forward


@pytest.fixture(scope="module")
def qkv():
    rng = np.random.default_rng(2026)
    T, d = 64, 16
    return rng.standard_normal((T, d)), rng.standard_normal((T, d)), rng.standard_normal((T, d))


def test_registry_covers_all_variants():
    names = list_variants()
    for expected in (
        "full", "flash", "linear", "performer",
        "local_window", "block_sparse", "effattnfuse",
    ):
        assert expected in names
        assert build_variant(expected) is not None


def test_registry_rejects_unknown():
    with pytest.raises(KeyError):
        build_variant("no_such_attention")


def test_flash_matches_full(qkv):
    """Flash 与 Full 必须数值等价 (精确注意力)。"""
    q, k, v = qkv
    a = FullAttention().attend(q, k, v).output
    b = FlashAttention().attend(q, k, v).output
    assert np.max(np.abs(a - b)) < 1e-12


def test_attention_output_metadata(qkv):
    q, k, v = qkv
    for inst in (
        FullAttention(), FlashAttention(), LinearAttention(),
        PerformerAttention(m=64), LocalWindowAttention(window=8),
        BlockSparseAttention(block_size=16, n_global=2),
    ):
        out = inst.attend(q, k, v)
        assert out.output.shape == q.shape
        assert out.mem_peak_mb >= 0.0
        assert out.cost_s >= 0.0
        assert isinstance(inst.theoretical_complexity(q.shape[0], q.shape[1]), str)


def test_local_window_respects_window(qkv):
    """滑窗注意力: 输出必须与「只看窗口内」的精确注意力一致。"""
    q, k, v = qkv
    w = 4
    out = LocalWindowAttention(window=w).attend(q, k, v).output
    T, d = q.shape
    for i in range(T):
        lo, hi = max(0, i - w), min(T, i + w + 1)
        s = (q[i] @ k[lo:hi].T) / np.sqrt(d)
        s = s - s.max()
        p = np.exp(s) / np.exp(s).sum()
        assert np.allclose(out[i], p @ v[lo:hi], atol=1e-12)


def test_block_sparse_respects_pattern(qkv):
    """块稀疏注意力: 输出必须与「块内 + 全局 token」的精确注意力一致。"""
    q, k, v = qkv
    b, ng = 16, 2
    out = BlockSparseAttention(block_size=b, n_global=ng).attend(q, k, v).output
    T, d = q.shape
    for bi in range((T + b - 1) // b):
        qs, qe = bi * b, min(T, (bi + 1) * b)
        idx = [np.arange(ng)]
        for bj in (bi - 1, bi, bi + 1):
            if 0 <= bj < (T + b - 1) // b:
                idx.append(np.arange(bj * b, min(T, (bj + 1) * b)))
        allowed = np.unique(np.concatenate(idx))
        s = (q[qs:qe] @ k[allowed].T) / np.sqrt(d)
        s = s - s.max(axis=-1, keepdims=True)
        p = np.exp(s) / np.exp(s).sum(axis=-1, keepdims=True)
        assert np.allclose(out[qs:qe], p @ v[allowed], atol=1e-10)


def test_performer_shares_features_between_q_and_k():
    """FAVOR+ 回归锁: Q 与 K 必须使用**同一组**随机特征 (ω, b)。

    若二者各自独立采样随机特征, 内积的期望虽仍正确, 但协方差被破坏,
    估计方差急剧恶化 (实测相对误差从 ~0.04 恶化到 ~18)。
    """
    perf = PerformerAttention(m=64, seed=3)
    W1, b1 = perf._sample(8)
    W2, b2 = perf._sample(8)
    assert np.array_equal(W1, W2) and np.array_equal(b1, b2)


def test_performer_asymmetric_norm_is_unbiased_in_ratio():
    """FAVOR+ 非对称范数因子回归锁。

    Query 用 exp(−‖x‖²/2)、Key 用 exp(+‖x‖²/2) 时, 注意力比值无偏;
    若两侧同号, 估计量会引入随 ‖k_j‖² 变化的系统性偏倚。
    """
    from morningstar.attention import PerformerAttention as PA

    rng = np.random.default_rng(5)
    T, d = 48, 8
    q = rng.standard_normal((T, d)) * 0.3
    k = rng.standard_normal((T, d)) * 0.3
    v = rng.standard_normal((T, d))

    perf = PA(m=2048, seed=11)
    W, b = perf._sample(d)
    pq = perf._features(q, W, b, norm_sign=-1.0)
    pk = perf._features(k, W, b, norm_sign=+1.0)
    out = (pq @ (pk.T @ v)) / (pq @ pk.sum(axis=0))[:, None]

    s = (q @ k.T) / np.sqrt(d)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    ref = (e / e.sum(axis=-1, keepdims=True)) @ v
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert rel < 0.5, f"FAVOR+ 相对误差过大: {rel:.4f}"


def test_effattnfuse_router_selects_full_for_short():
    """路由: 短序列走精确注意力。"""
    fuse = EffAttnFuse(full_threshold=64)
    assert isinstance(fuse.choose(32, 16), FullAttention)


def test_effattnfuse_router_falls_back_when_fidelity_low():
    """质量守卫: 标定保真度不达标时回退到精确注意力。"""
    fuse = EffAttnFuse(full_threshold=8, calibration={64: 0.5}, retrieval_floor=0.98)
    assert isinstance(fuse.choose(64, 16), FullAttention)


def test_effattnfuse_router_uses_performer_when_calibrated():
    fuse = EffAttnFuse(full_threshold=8, calibration={64: 0.99}, retrieval_floor=0.98)
    assert isinstance(fuse.choose(64, 16), PerformerAttention)


def test_flash_memory_is_subquadratic():
    """Flash 的峰值内存必须是次二次的。"""
    T, d = 2048, 32
    rng = np.random.default_rng(1)
    q, k, v = (rng.standard_normal((T, d)) for _ in range(3))
    _, _, nbytes = flash_attention_forward(q, k, v, mask="none")
    naive_bytes = 2 * T * T * 8
    assert nbytes < naive_bytes / 20


def test_naive_attention_returns_stochastic_weights():
    rng = np.random.default_rng(2)
    q, k, v = (rng.standard_normal((12, 8)) for _ in range(3))
    _, w = naive_attention_forward(q, k, v, mask="none", return_weights=True)
    assert w.shape == (12, 12)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)
