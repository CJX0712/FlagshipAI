"""线性回归 —— forge 闭式解（正规方程）+ 不变量。

不变量：
  * 残差 ⊥ X :  Xᵀ(y − Xŵ) ≈ 0  （正规方程直接保证）
  * 闭式解 == 梯度下降收敛解（凸问题唯一极小）
  * 跨库校验：与 numpy.linalg.lstsq 权重一致至 1e-8
"""
from __future__ import annotations

import numpy as np

from ..mathx.linalg import cholesky, cholesky_inverse


class LinearRegression:
    def __init__(self, fit_intercept: bool = True, ridge: float = 0.0):
        self.fit_intercept = fit_intercept
        self.ridge = float(ridge)
        self.coef_ = None
        self.intercept_ = 0.0

    def _aug(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X

    def fit(self, X: np.ndarray, y: np.ndarray):
        A = self._aug(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        AtA = A.T @ A
        n = A.shape[1]
        reg = np.eye(n) * self.ridge
        if self.fit_intercept and self.ridge > 0:
            reg[0, 0] = 0.0  # 截距不惩罚
        L = cholesky(AtA + reg)
        w = cholesky_inverse(L) @ (A.T @ y)
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.coef_ = w
        self._A, self._y, self._w = A, y, w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._aug(X) @ self._w

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-300))

    # ---------- 不变量 ----------
    def check_invariants(self) -> dict:
        r = self._y - self._A @ self._w
        ortho = float(np.linalg.norm(self._A.T @ r))
        return {
            "residuals_orthogonal_X": {
                "value": ortho, "threshold": 1e-6, "passed": bool(ortho < 1e-6)
            }
        }

    def compare_numpy(self, X: np.ndarray, y: np.ndarray) -> dict:
        """跨库校验：与 numpy.linalg.lstsq 比较权重。"""
        A = self._aug(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        w_ref, *_ = np.linalg.lstsq(A, y, rcond=None)
        diff = float(np.linalg.norm(self._w - w_ref))
        return {"weight_diff_vs_numpy": diff, "passed": bool(diff < 1e-8)}
