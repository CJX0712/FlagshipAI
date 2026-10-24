"""L1 线性代数 —— forge 纯实现 + 数学不变量。

零重写：稠密线性代数（BLAS/LAPACK）直接复用 numpy；此处只手写那些
「能用一条可验证不变量独立证明正确性」的关键原语，并用不变量自证。

不变量清单：
  * Cholesky  : A = L Lᵀ  ⇒  ‖A − L Lᵀ‖_F ≈ 0
  * Jacobi 特征: 对称阵特征向量彼此正交  ⇒  ‖VᵀV − I‖_F ≈ 0
  * Jacobi 特征: 残差  ‖A vᵢ − λᵢ vᵢ‖_2 ≈ ε (机器精度级)
  * logsumexp : 数值稳定，无误溢出
"""
from __future__ import annotations

import numpy as np

EPS = np.finfo(np.float64).eps  # ≈ 2.2e-16


def cholesky(A: np.ndarray) -> np.ndarray:
    """Cholesky 分解 A = L Lᵀ，L 为下三角（正定阵）。"""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        s = A[j, j] - np.sum(L[j, :j] ** 2)
        if s <= 0:
            # 非正定容错：加微小岭项后重试（数值健壮性）
            A = A + 1e-12 * np.eye(n)
            return cholesky(A)
        L[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L


def cholesky_inverse(L: np.ndarray) -> np.ndarray:
    """利用 Cholesky 因子求 (L Lᵀ)⁻¹ （前向/后向代入，O(n²)）。"""
    L = np.asarray(L, dtype=np.float64)
    n = L.shape[0]
    # 解 L Y = I  (下三角前向代入)
    Y = np.zeros_like(L)
    for j in range(n):
        Y[j, j] = 1.0 / L[j, j]
        for i in range(j + 1, n):
            Y[i, j] = -np.dot(L[i, j:i], Y[j:i, j]) / L[i, i]
    # A⁻¹ = (L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹ = Yᵀ Y
    return Y.T @ Y


def jacobi_eigen(A: np.ndarray, max_sweeps: int = 100, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, dict]:
    """经典 Jacobi 循环旋转法求对称阵特征分解。

    返回 (eigenvalues[升序], eigenvectors 列向量, info)。
    正交性不变量：VᵀV = I；残差不变量：‖A vᵢ − λᵢ vᵢ‖ ≈ ε。
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    # 强制对称
    A = 0.5 * (A + A.T)
    V = np.eye(n)
    off = lambda M: np.sqrt(np.sum(np.tril(M, -1) ** 2) * 2)
    sweeps = 0
    for sweep in range(max_sweeps):
        if off(A) < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) < 1e-300:
                    continue
                # 旋转角：tan(2φ) = 2 a_pq / (a_pp − a_qq)
                # θ = (a_qq − a_pp) / (2 a_pq)，t = tan φ
                denom = 2.0 * A[p, q]
                if abs(A[p, p] - A[q, q]) < 1e-300:
                    # a_pp == a_qq 时 φ = π/4（t=1），避免 θ=0 导致 t=0 卡死
                    t = 1.0
                else:
                    theta = (A[q, q] - A[p, p]) / denom
                    t = np.sign(theta) / (abs(theta) + np.sqrt(theta * theta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                # 全空间旋转矩阵 J = I，仅在 (p,q) 块为 [[c, s], [-s, c]]
                # 严格用矩阵乘法 A' = Jᵀ A J、V' = Jᵀ V，
                # 根除元素级更新破坏对称性（旧版漏设 A[q,p]=0 致对角失真）的隐患。
                J = np.eye(n)
                J[p, p] = c
                J[p, q] = s
                J[q, p] = -s
                J[q, q] = c
                A = J.T @ A @ J
                # 特征向量累积必须右乘：V' = V · J，使 A_0 = V_k A_k V_kᵀ 成立
                # （因 A_k = J_{k-1}ᵀ … J_0ᵀ A_0 J_0 … J_{k-1} ⇒
                #   A_0 = (J_0…J_{k-1}) A_k (J_0…J_{k-1})ᵀ = V_k A_k V_kᵀ ≈ V_k Λ V_kᵀ）
                V = V @ J
        sweeps = sweep + 1
    eigvals = np.diag(A)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    V = V[:, idx]
    info = {"sweeps": sweeps, "off_diag": float(off(A))}
    return eigvals, V, info


def logsumexp(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """数值稳定的 log(Σ exp(x))。防溢出核心原语。

    ``axis=None`` 时返回**0 维标量** (Python 标量语义), 与
    ``scipy.special.logsumexp`` 一致; 传 axis 时按该轴归约。
    """
    x = np.asarray(x, dtype=np.float64)
    if axis is None and not keepdims:
        xmax = np.max(x)
        if not np.isfinite(xmax):
            xmax = 0.0
        out = xmax + np.log(np.sum(np.exp(x - xmax)))
        return np.asarray(out, dtype=np.float64).reshape(())
    xmax = np.max(x, axis=axis, keepdims=True)
    xmax_safe = np.where(np.isfinite(xmax), xmax, 0.0)
    out = xmax_safe + np.log(np.sum(np.exp(x - xmax_safe), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis) if axis is not None else out
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax，基于 logsumexp。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """对数 softmax，用于稳定交叉熵。"""
    x = np.asarray(x, dtype=np.float64)
    return x - logsumexp(x, axis=axis, keepdims=True)


# ---------- 不变量验证器 ----------
def orthogonality_error(V: np.ndarray) -> float:
    """‖VᵀV − I‖_F —— 特征向量正交归一程度。"""
    V = np.asarray(V, dtype=np.float64)
    return float(np.linalg.norm(V.T @ V - np.eye(V.shape[1])))


def eigen_residual(A: np.ndarray, eigvals: np.ndarray, V: np.ndarray) -> float:
    """max_i ‖A vᵢ − λᵢ vᵢ‖_2 —— Jacobi 残差（应接近机器精度）。"""
    A = np.asarray(A, dtype=np.float64)
    resid = 0.0
    for i in range(V.shape[1]):
        v = V[:, i]
        r = A @ v - eigvals[i] * v
        resid = max(resid, np.linalg.norm(r))
    return float(resid)


def is_symmetric_psd(A: np.ndarray, tol: float = 1e-8) -> tuple[bool, float]:
    """对称半正定判定：对称且所有特征值 ≥ −tol。"""
    A = np.asarray(A, dtype=np.float64)
    sym_err = float(np.linalg.norm(A - A.T))
    e = np.linalg.eigvalsh(0.5 * (A + A.T))
    min_eig = float(e.min())
    return (sym_err < tol and min_eig >= -tol), min_eig
