"""硬不变量测试 —— 每个算子的「手写实现正确性」底线。

这些测试对应 docs/DESIGN.md 中的不变量编号, 任何一条失败都意味着
底层数学实现出现了回归。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from morningstar.flash.attention import (
    flash_attention_forward,
    naive_attention_forward,
)
from morningstar.flash.modules import RotaryEmbedding
from morningstar.flash.model import (
    GPT,
    GPTConfig,
    _cross_entropy_loss,
    _rmsnorm_bwd,
    _rmsnorm_fwd,
    _softmax,
    _swiglu_bwd,
    _swiglu_fwd,
)
from morningstar.verify import invariants


# ---------------------------------------------------------------- I1 / I2 / I3
def test_flash_equals_naive():
    """I1: FlashAttention-2 分块实现 == 精确 softmax 注意力。"""
    res = invariants.check_flash_naive_equivalence()
    assert res["passed"], res["detail"]


def test_row_stochastic():
    """I2: 注意力权重逐行和为 1。"""
    res = invariants.check_row_stochastic()
    assert res["passed"], res["detail"]


def test_causal_upper_triangle_zero():
    """I3: 因果掩码的上三角严格为 0。"""
    res = invariants.check_causal_zero()
    assert res["passed"], res["detail"]


def test_shift_invariance():
    """I4: logsumexp 数值稳定性 —— key 整体平移不改变输出。"""
    res = invariants.check_shift_invariance()
    assert res["passed"], res["detail"]


def test_kv_cache_consistency():
    """I6: KV-Cache 逐 token 解码 == 全序列一次性前向。"""
    res = invariants.check_kv_cache_consistency()
    assert res["passed"], res["detail"]


def test_memory_subquadratic():
    """I7: FlashAttention 的内存占用是次二次的。"""
    res = invariants.check_memory_subquadratic()
    assert res["passed"], res["detail"]


# ------------------------------------------------------------------------ R1
def test_rope_relative_position_property():
    """R1: RoPE 内积只依赖相对位移 (m−n)。"""
    res = invariants.check_rope_relative_property()
    assert res["passed"], res["detail"]


def test_rope_norm_preserving():
    """R2: RoPE 是正交变换, 保持范数。"""
    res = invariants.check_rope_norm_preserving()
    assert res["passed"], res["detail"]


def test_rope_inverse_is_true_inverse():
    """R2b: rotate_qk_inverse 必须严格是 rotate_qk 的逆。

    这条是历史 bug 的回归锁: 旧实现用 `rotate_qk(x, -positions)` 做逆旋转,
    而 numpy 会把负位置当作**负数组索引**回绕到缓存末尾, 导致逆旋转完全错误
    (自洽误差 1.12)。修复后应为机器精度。
    """
    rng = np.random.default_rng(3)
    dh, T, H = 8, 6, 4
    x = rng.standard_normal((T, H, dh))
    pos = np.arange(T)
    rot = RotaryEmbedding(dh)
    y = rot.rotate_qk(x, pos)
    xr = rot.rotate_qk_inverse(y, pos)
    assert np.max(np.abs(x - xr)) < 1e-12


def test_rope_inverse_backward_numerically():
    """R2c: RoPE 反向传播的解析梯度 == 中心差分。

    逆旋转若符号/缩放错误, 该梯度会立刻发散。
    """
    rng = np.random.default_rng(11)
    dh, T, H = 8, 5, 3
    rot = RotaryEmbedding(dh)
    pos = np.arange(T)
    x = rng.standard_normal((T, H, dh))

    def f(xx: np.ndarray) -> float:
        return float(np.sum(rot.rotate_qk(xx, pos) ** 2))

    y = rot.rotate_qk(x, pos)
    ana = rot.rotate_qk_inverse(2.0 * y, pos)  # dL/dy = 2y, 再逆旋转
    eps = 1e-6
    num = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        a = f(x)
        x[idx] = old - eps
        b = f(x)
        x[idx] = old
        num[idx] = (a - b) / (2 * eps)
    rel = np.max(np.abs(ana - num)) / np.max(np.abs(num))
    assert rel < 1e-6, f"RoPE 反向相对误差 {rel:.3e}"


# ------------------------------------------------------------------------ R3
def test_rmsnorm_unit_rms():
    """R3: RMSNorm 归一化后逐行 RMS = 1。"""
    res = invariants.check_rmsnorm_unit_rms()
    assert res["passed"], res["detail"]


def test_rmsnorm_backward_numerically():
    """R3b: RMSNorm 反向的解析梯度 == 中心差分。"""
    rng = np.random.default_rng(5)
    D = 8
    x = rng.standard_normal((5, D))
    g = rng.standard_normal(D) + 1.0
    w_out = rng.standard_normal(D)
    eps = 1e-8

    def f(xx: np.ndarray, gg: np.ndarray) -> float:
        y, _ = _rmsnorm_fwd(xx, gg, eps)
        return float(np.sum(y * w_out))

    y, inv = _rmsnorm_fwd(x, g, eps)
    dy = np.broadcast_to(w_out, y.shape).copy()
    dx, dg = _rmsnorm_bwd(dy, x, g, inv, eps, None)

    num_x = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + 1e-6
        a = f(x, g)
        x[i] = old - 1e-6
        b = f(x, g)
        x[i] = old
        num_x[i] = (a - b) / 2e-6

    num_g = np.zeros_like(g)
    for i in range(g.size):
        old = g[i]
        g[i] = old + 1e-6
        a = f(x, g)
        g[i] = old - 1e-6
        b = f(x, g)
        g[i] = old
        num_g[i] = (a - b) / 2e-6

    assert np.max(np.abs(dx - num_x)) / np.max(np.abs(num_x)) < 1e-6
    assert np.max(np.abs(dg - num_g)) / np.max(np.abs(num_g)) < 1e-6


def test_swiglu_backward_numerically():
    """R4: SwiGLU 反向的解析梯度 == 中心差分。"""
    rng = np.random.default_rng(9)
    d, F = 8, 16
    x = rng.standard_normal((5, d))
    wg = rng.standard_normal((d, F)) * 0.3
    wu = rng.standard_normal((d, F)) * 0.3
    wd = rng.standard_normal((F, d)) * 0.3
    w_out = rng.standard_normal(d)

    def f(xx, a, b, c):
        o, _ = _swiglu_fwd(xx, a, b, c)
        return float(np.sum(o * w_out))

    out, cache = _swiglu_fwd(x, wg, wu, wd)
    dy = np.broadcast_to(w_out, out.shape).copy()
    _, _, d_down, dx = _swiglu_bwd(cache, dy)

    num_d = np.zeros_like(wd)
    for i in np.ndindex(wd.shape):
        old = wd[i]
        wd[i] = old + 1e-6
        a = f(x, wg, wu, wd)
        wd[i] = old - 1e-6
        b = f(x, wg, wu, wd)
        wd[i] = old
        num_d[i] = (a - b) / 2e-6

    num_x = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + 1e-6
        a = f(x, wg, wu, wd)
        x[i] = old - 1e-6
        b = f(x, wg, wu, wd)
        x[i] = old
        num_x[i] = (a - b) / 2e-6

    assert np.max(np.abs(d_down - num_d)) / np.max(np.abs(num_d)) < 1e-6
    assert np.max(np.abs(dx - num_x)) / np.max(np.abs(num_x)) < 1e-6


# ------------------------------------------------------------------------ G1
def test_gpt_gradient_check():
    """G1: 全模型解析梯度 == 中心差分 (手写反向传播的硬标准)。"""
    res = invariants.gpt_gradient_check()
    assert res["passed"], res["detail"]


def test_gpt_gradient_check_deep_model():
    """G1b: 4 层模型上同样成立 (误差不随深度放大)。"""
    res = invariants.gpt_gradient_check(n_layer=4, T=6, seed=17)
    assert res["passed"], res["detail"]


# ------------------------------------------------------------------------ G4
def test_crossentropy_uniform_equals_ln_vocab():
    """G4: 均匀预测时交叉熵精确等于 ln(V)。"""
    res = invariants.check_crossentropy_uniform()
    assert res["passed"], res["detail"]


def test_softmax_sums_to_one():
    """G4b: softmax 逐行和为 1 且无 NaN。"""
    rng = np.random.default_rng(2)
    z = rng.standard_normal((10, 17)) * 50.0  # 极端量级
    p = _softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(p).all()


def test_crossentropy_matches_manual():
    """G4c: 交叉熵实现 == 手工 logsumexp 公式。"""
    rng = np.random.default_rng(4)
    z = rng.standard_normal((6, 13))
    t = rng.integers(0, 13, size=6)
    manual = float(
        np.mean([-z[i, t[i]] + math.log(np.sum(np.exp(z[i] - z[i].max()))) + z[i].max()
                for i in range(6)])
    )
    assert abs(_cross_entropy_loss(z, t) - manual) < 1e-12


# ------------------------------------------------------------------------ P1
def test_determinism():
    """P1: 同 seed ⇒ 逐位一致的初始化与前向。"""
    res = invariants.check_determinism()
    assert res["passed"], res["detail"]


# ------------------------------------------------------------------------ T1
def test_bpe_roundtrip():
    """T1: BPE 编解码无损往返。"""
    res = invariants.check_bpe_roundtrip()
    assert res["passed"], res["detail"]


def test_bpe_determinism():
    """T3: BPE 训练确定性。"""
    res = invariants.check_bpe_determinism()
    assert res["passed"], res["detail"]


# ------------------------------------------------------------------------ A4
def test_approx_attention_fidelity():
    """A4: 近似注意力 (Linear / Performer) 的输出保真度。"""
    res = invariants.check_approx_attention_fidelity()
    assert res["passed"], res["detail"]


# ------------------------------------------------------------ 套件整体自检
def test_all_invariants_registered():
    """注册表完整性: 全部不变量必须可运行且通过。"""
    res = invariants.run_all()
    failed = [r["name"] for r in res if not r["passed"]]
    assert not failed, f"失败的不变量: {failed}"
    assert len(res) >= 16
