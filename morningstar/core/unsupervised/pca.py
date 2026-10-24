"""PCA —— forge 协方差 + Jacobi 特征分解 + 不变量。

不变量（世界级硬标准）：
  * 协方差矩阵对称半正定（is_symmetric_psd）
  * 特征向量彼此正交归一：‖VᵀV − I‖_F ≈ 0
  * Eckart-Young 最优性：秩-k 截断重构误差 = Σ_{j>k} λⱼ（Frobenius 最优）
  * 跨库校验：与 sklearn.decomposition.PCA 重构误差/解释方差一致
"""
from __future__ import annotations

import numpy as np

from ..mathx.linalg import jacobi_eigen, orthogonality_error, is_symmetric_psd


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        N = X.shape[0]
        cov = (Xc.T @ Xc) / (N - 1)
        eigvals, eigvecs, info = jacobi_eigen(cov)
        # jacobi_eigen 已升序；取降序
        self.eigenvalues_ = eigvals[::-1]
        self.components_ = eigvecs[:, ::-1].T  # 行 = 主成分
        self.explained_variance_ = self.eigenvalues_
        self.explained_variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()
        self._Xc = Xc
        self._cov = cov
        self._info = info
        if self.n_components is None:
            self.n_components = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_[: self.n_components].T

    def reconstruct(self, X):
        Z = self.transform(X)
        return Z @ self.components_[: self.n_components] + self.mean_

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=np.float64)
        rec = self.reconstruct(X)
        return float(np.mean((X - rec) ** 2))

    def check_invariants(self) -> dict:
        psd, min_eig = is_symmetric_psd(self._cov)
        ortho = orthogonality_error(self.components_.T)
        # Eckart-Young：秩-k 截断重构误差（逐元素均值）最优。
        # Xc 的 Frobenius 残差² = Σ_{j>k} σ_j²，(N-1)·Σ_{j>k} λⱼ = Σ_{j>k} σ_j²。
        # reconstruction_error 用的是 np.mean((X-rec)²)（对全部 N·d 个元素求均值），
        # 故理论值须再除以 d：theo = (N-1)/(N·d) · Σ_{j>k} λⱼ。
        N, d = self._Xc.shape
        dropped = self.eigenvalues_[self.n_components:]
        theo = float(dropped.sum() * (N - 1) / (N * d))
        rec_err = self.reconstruction_error(self._Xc + self.mean_)
        return {
            "cov_symmetric_psd": {"value": min_eig, "threshold": -1e-8, "passed": bool(psd)},
            "eigenvectors_orthonormal": {"value": ortho, "threshold": 1e-8, "passed": bool(ortho < 1e-8)},
            "eckart_young_optimal": {
                "theoretical_dropped_energy": theo,
                "actual_recon_error": rec_err,
                "passed": bool(abs(theo - rec_err) < 1e-6 * max(1, theo)),
            },
        }

    def compare_sklearn(self, X) -> dict:
        from sklearn.decomposition import PCA as SKPCA
        X = np.asarray(X, dtype=np.float64)
        k = self.n_components
        sk = SKPCA(n_components=k).fit(X)
        our_err = self.reconstruction_error(X)
        sk_err = float(np.mean((X - sk.inverse_transform(sk.transform(X))) ** 2))
        evr_diff = float(np.max(np.abs(
            self.explained_variance_ratio_[:k] - sk.explained_variance_ratio_)))
        return {
            "recon_error_ours": our_err,
            "recon_error_sklearn": sk_err,
            "explained_variance_ratio_maxdiff": evr_diff,
            "passed": bool(abs(our_err - sk_err) < 1e-6 * max(1, sk_err) and evr_diff < 1e-6),
        }
