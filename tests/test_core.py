"""核心数学层测试 —— 经典机器学习算子的不变量与跨库一致性。

core 层直接复用 NumPy / SciPy / scikit-learn 的世界级实现; 测试的目的是
确认封装层没有引入偏差 (归一化方向、符号、迭代终止条件), 并验证其
自带的 ``check_invariants()`` 与 ``compare_sklearn()`` 跨库交叉验证确实通过。
"""
from __future__ import annotations

import numpy as np
import pytest

from morningstar.core.mathx import linalg as lx
from morningstar.core.mathx import optimize as op
from morningstar.core.supervised import linear as sl
from morningstar.core.unsupervised import gmm as sg
from morningstar.core.unsupervised import kmeans as sk
from morningstar.core.unsupervised import pca as sp


# ------------------------------------------------------------------ 线性代数
def test_cholesky_reconstructs():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((30, 30))
    S = A @ A.T + 30 * np.eye(30)
    L = lx.cholesky(S)
    assert np.allclose(L @ L.T, S, atol=1e-8)


def test_jacobi_eigen_residual():
    """Jacobi 特征分解: ‖Av − λv‖ 应为机器精度。"""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((12, 12))
    A = A + A.T
    eigvals, eigvecs, info = lx.jacobi_eigen(A)
    assert eigvecs.shape == (12, 12)
    resid = lx.eigen_residual(A, eigvals, eigvecs)
    assert resid < 1e-9, f"特征残差 {resid:.3e}"
    assert lx.orthogonality_error(eigvecs) < 1e-9


def test_jacobi_on_non_symmetric_does_not_crash():
    """Jacobi 对非对称矩阵: 至少不得崩溃 (对称化或退化都可接受)。"""
    rng = np.random.default_rng(14)
    A = rng.standard_normal((6, 6))
    try:
        eigvals, eigvecs, _ = lx.jacobi_eigen(A)
    except ValueError:
        return  # 明确拒绝也是正确行为
    assert eigvals.shape == (6,)
    assert np.isfinite(eigvals).all()


def test_logsumexp_stability():
    """logsumexp 对极大/极小量级都必须稳定且无溢出。"""
    x = np.array([1000.0, 1000.0, 1000.0])
    assert np.isfinite(lx.logsumexp(x)).all()
    assert np.ndim(lx.logsumexp(x)) == 0, "axis=None 时应返回 0 维标量"
    assert abs(float(lx.logsumexp(x)) - (1000.0 + np.log(3))) < 1e-9
    y = np.array([-1000.0, -1000.0])
    assert np.isfinite(lx.logsumexp(y)).all()


def test_log_softmax_matches_softmax_log():
    rng = np.random.default_rng(15)
    z = rng.standard_normal((5, 7)) * 40
    assert np.allclose(np.exp(lx.log_softmax(z)), lx.softmax(z), atol=1e-12)


# -------------------------------------------------------------------- 优化
def test_gradient_check_passes():
    """中心差分 vs 解析梯度 —— 优化层的硬标准。"""
    def f(x):
        return float(np.sum(x ** 3) + 0.5 * np.sum(x ** 2))

    def g(x):
        return 3 * x ** 2 + x

    res = op.gradient_check(f, g, np.array([1.0, -2.0, 0.5]))
    assert res["passed"], res


def test_gradient_descent_converges_on_least_squares():
    """梯度下降必须收敛到最小二乘解析解。"""
    rng = np.random.default_rng(5)
    A = rng.standard_normal((20, 4))
    b = rng.standard_normal(20)
    x_star = np.linalg.lstsq(A, b, rcond=None)[0]

    def f(x):
        r = A @ x - b
        return float(r @ r)

    def g(x):
        return 2.0 * A.T @ (A @ x - b)

    res = op.gradient_descent(f, g, np.zeros(4), lr=0.02, n_iter=3000)
    assert np.max(np.abs(res["x"] - x_star)) < 1e-6
    assert op.check_monotone_decrease(res["f_hist"])


# ------------------------------------------------------------------ 监督学习
def test_linear_regression_invariants():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((50, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.1 * rng.standard_normal(50)
    m = sl.LinearRegression().fit(X, y)
    inv = m.check_invariants()
    for name, r in inv.items():
        assert r["passed"], f"{name}: {r}"


def test_linear_regression_matches_numpy():
    rng = np.random.default_rng(16)
    X = rng.standard_normal((40, 3))
    y = rng.standard_normal(40)
    m = sl.LinearRegression().fit(X, y)
    cmp = m.compare_numpy(X, y)
    assert cmp["passed"], cmp


def test_linear_regression_residual_orthogonal():
    """正规方程: 残差必须与设计矩阵列空间正交 (Xᵀ(y−Xŵ)=0)。"""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.1 * rng.standard_normal(50)
    m = sl.LinearRegression().fit(X, y)
    resid = y - m.predict(X)
    assert np.max(np.abs(X.T @ resid)) < 1e-8


# ------------------------------------------------------------------ 无监督学习
def test_pca_invariants():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((200, 5)) @ rng.standard_normal((5, 5))
    p = sp.PCA(n_components=3).fit(X)
    inv = p.check_invariants()
    for name, r in inv.items():
        assert r["passed"], f"{name}: {r}"


def test_pca_components_orthonormal():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((150, 6))
    p = sp.PCA(n_components=3).fit(X)
    V = p.components_[:3].T          # components_ 行 = 主成分
    assert V.shape == (6, 3)
    assert np.allclose(V.T @ V, np.eye(3), atol=1e-10)


def test_pca_matches_sklearn():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(10)
    X = rng.standard_normal((200, 5))
    p = sp.PCA(n_components=3).fit(X)
    cmp = p.compare_sklearn(X)
    assert cmp["passed"], cmp


def test_kmeans_invariants_and_stability():
    rng = np.random.default_rng(11)
    X = np.vstack([
        rng.standard_normal((40, 2)) + np.array([5.0, 5.0]),
        rng.standard_normal((40, 2)) + np.array([-5.0, -5.0]),
    ])
    km = sk.KMeans(n_clusters=2, seed=0).fit(X)
    inv = km.check_invariants()
    for name, r in inv.items():
        if isinstance(r, dict):
            assert r.get("passed", True), f"{name}: {r}"
    labels = km.predict(X)
    assert set(labels.tolist()) == {0, 1}
    for c in (0, 1):
        members = X[labels == c]
        if len(members):
            assert np.allclose(km.cluster_centers_[c], members.mean(axis=0), atol=1e-8)


def test_kmeans_matches_sklearn():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(12)
    X = rng.standard_normal((120, 3))
    km = sk.KMeans(n_clusters=3, seed=0).fit(X)
    cmp = km.compare_sklearn(X)
    assert cmp["passed"], cmp


def test_gmm_invariants_and_monotone_likelihood():
    """GMM: EM 定理 —— 对数似然单调不减; 责任度/权重归一; 协方差半正定。"""
    rng = np.random.default_rng(13)
    X = np.vstack([
        rng.standard_normal((50, 3)) + 4.0,
        rng.standard_normal((50, 3)) - 4.0,
    ])
    g = sg.GaussianMixture(n_components=2, seed=0).fit(X)
    inv = g.check_invariants()
    for name, r in inv.items():
        assert r["passed"], f"{name}: {r}"
    ll = g.loglik_hist_
    assert len(ll) >= 2
    for a, b in zip(ll, ll[1:]):
        assert b >= a - 1e-6, "EM 迭代的对数似然必须单调不减"


def test_gmm_matches_sklearn():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(17)
    X = np.vstack([
        rng.standard_normal((40, 2)) + 3.0,
        rng.standard_normal((40, 2)) - 3.0,
    ])
    g = sg.GaussianMixture(n_components=2, seed=0).fit(X)
    cmp = g.compare_sklearn(X)
    assert cmp["passed"], cmp


# ------------------------------------------------------------------ 深网层
def test_core_attention_runs_and_is_finite():
    from morningstar.core.deep.attention import scaled_dot_product_attention

    rng = np.random.default_rng(18)
    T, d = 8, 16
    Q, K, V = (rng.standard_normal((T, d)) for _ in range(3))
    out, P = scaled_dot_product_attention(Q, K, V, causal=True)
    assert out.shape == (T, d)
    assert np.isfinite(out).all()
    assert P.shape == (T, T)
    # 因果掩码: 上三角权重必须为 0
    assert np.allclose(np.triu(P, k=1), 0.0, atol=1e-12)


def test_core_attention_invariants():
    from morningstar.core.deep.attention import Attention

    rng = np.random.default_rng(19)
    T, d = 6, 12
    Q, K, V = (rng.standard_normal((T, d)) for _ in range(3))
    att = Attention(causal=True)
    inv = att.check_invariants(Q, K, V)
    for name, r in inv.items():
        assert r["passed"], f"{name}: {r}"


def test_mlp_learns_linear_function():
    """MLP 必须能拟合一个线性函数 (前向/反向/更新链路正确)。"""
    from morningstar.core.deep.neural_net import MLP

    rng = np.random.default_rng(20)
    X = rng.standard_normal((120, 3))
    y = (X @ np.array([1.5, -2.0, 0.8])).reshape(-1, 1)
    net = MLP(hidden=16, lr=0.05, n_iter=2000, seed=0)
    net.fit(X, y)
    pred = net.predict(X)          # predict 返回一维
    rmse = float(np.sqrt(np.mean((pred - y.ravel()) ** 2)))
    assert rmse < 0.5, f"MLP 拟合误差过大: rmse={rmse:.4f}"


def test_mlp_invariants():
    from morningstar.core.deep.neural_net import MLP

    rng = np.random.default_rng(21)
    X = rng.standard_normal((60, 2))
    y = rng.standard_normal((60, 1))
    net = MLP(hidden=8, n_iter=50, seed=0).fit(X, y)
    inv = net.check_invariants()
    for name, r in inv.items():
        if isinstance(r, dict):
            assert r.get("passed", True), f"{name}: {r}"
