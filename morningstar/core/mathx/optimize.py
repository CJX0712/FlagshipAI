"""L3 微积分优化 —— forge 纯实现 + 数学不变量。

核心不变量（世界级正确性硬标准）：
  * 梯度检验 maxRelErr = max_i |g_numᵢ − g_anaᵢ| / (|g_numᵢ| + |g_anaᵢ| + ε) ≈ 1e-7
    解析梯度 vs 中心差分数值梯度（链式法则之母，反向传播通用判官）。
"""
from __future__ import annotations

import numpy as np

EPS = np.finfo(np.float64).eps


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """中心差分数值梯度：∂f/∂xᵢ ≈ (f(x+εeᵢ) − f(x−εeᵢ)) / 2ε。"""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        g[idx] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return g


def gradient_check(f, grad_f, x: np.ndarray, eps: float = 1e-6) -> dict:
    """梯度检验：返回 {max_rel_err(显著分量), mean_rel_err, max_abs_err, passed}。

    判官标准：对显著梯度分量 max_rel_err < 1e-5 即可信；世界级 forge 通常 ≈ 1e-7。
    近零梯度分量相对误差无意义，故用绝对误差兜底（max_abs_err < 1e-7）。
    """
    x = np.asarray(x, dtype=np.float64)
    g_num = numerical_gradient(f, x, eps)
    g_ana = np.asarray(grad_f(x), dtype=np.float64)
    abs_err = np.abs(g_num - g_ana)
    denom = np.abs(g_num) + np.abs(g_ana) + EPS
    rel = abs_err / denom
    mask = (np.abs(g_num) + np.abs(g_ana)) > 1e-6
    max_rel = float(np.max(rel[mask])) if mask.any() else 0.0
    mean_rel = float(np.mean(rel[mask])) if mask.any() else 0.0
    max_abs = float(np.max(abs_err))
    return {
        "max_rel_err": max_rel,
        "mean_rel_err": mean_rel,
        "max_abs_err": max_abs,
        "passed": bool(max_rel < 1e-5 or max_abs < 1e-7),
    }


def gradient_descent(
    f,
    grad_f,
    x0: np.ndarray,
    lr: float = 0.1,
    n_iter: int = 1000,
    tol: float = 1e-8,
    lr_decay: float = 1.0,
) -> dict:
    """朴素梯度下降；返回 {x, f_hist, grad_norm_hist, n_iter}。

    凸函数一阶条件不变量：f(y) ≥ f(x) + ∇f(x)ᵀ(y−x)（下降单调）。
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    f_hist = []
    gnorm_hist = []
    lr_cur = lr
    for it in range(n_iter):
        g = np.asarray(grad_f(x), dtype=np.float64)
        f_val = float(f(x))
        f_hist.append(f_val)
        gnorm_hist.append(float(np.linalg.norm(g)))
        x = x - lr_cur * g
        lr_cur *= lr_decay
        if gnorm_hist[-1] < tol:
            break
    return {
        "x": x,
        "f_hist": np.array(f_hist),
        "grad_norm_hist": np.array(gnorm_hist),
        "n_iter": len(f_hist),
    }


def check_monotone_decrease(f_hist: np.ndarray, tol: float = 1e-9) -> bool:
    """验证目标函数值单调非增（凸优化不变量）。"""
    f_hist = np.asarray(f_hist, dtype=np.float64)
    return bool(np.all(np.diff(f_hist) <= tol))
