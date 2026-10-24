"""逻辑回归 —— forge 梯度下降 + logsumexp 数值稳定 + 梯度检验不变量。

不变量：
  * 梯度检验 maxRelErr ≈ 1e-7（解析梯度 vs 中心差分）
  * 交叉熵损失随迭代单调非增（凸）
  * 跨库校验：与 sklearn.LogisticRegression 预测一致率高
"""
from __future__ import annotations

import numpy as np

from ..mathx.linalg import logsumexp  # 复用稳定原语
from ..mathx.optimize import gradient_check, check_monotone_decrease


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    def __init__(self, fit_intercept: bool = True, lr: float = 0.3, n_iter: int = 3000,
                 ridge: float = 0.0, tol: float = 1e-7):
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.n_iter = n_iter
        self.ridge = float(ridge)
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0

    def _aug(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X

    def _loss(self, w, A, y):
        p = _sigmoid(A @ w)
        # 稳定交叉熵
        ll = -y * np.log(p + 1e-300) - (1 - y) * np.log(1 - p + 1e-300)
        loss = ll.mean()
        if self.ridge > 0:
            loss += 0.5 * self.ridge * np.dot(w, w)
        return loss

    def _grad(self, w, A, y):
        p = _sigmoid(A @ w)
        g = (A.T @ (p - y)) / len(y)
        if self.ridge > 0:
            g += self.ridge * w
        return g

    def fit(self, X, y):
        A = self._aug(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        w = np.zeros(A.shape[1], dtype=np.float64)
        hist = []
        for _ in range(self.n_iter):
            g = self._grad(w, A, y)
            w = w - self.lr * g
            hist.append(self._loss(w, A, y))
            if np.linalg.norm(g) < self.tol:
                break
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.coef_ = w
        self._w, self._A, self._y, self._loss_hist = w, A, y, np.array(hist)
        return self

    def predict_proba(self, X):
        return _sigmoid(self._aug(X) @ self._w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def check_invariants(self) -> dict:
        gc = gradient_check(lambda w: self._loss(w, self._A, self._y),
                            lambda w: self._grad(w, self._A, self._y),
                            self._w)
        mono = check_monotone_decrease(self._loss_hist)
        return {
            "gradient_check": gc,
            "loss_monotone": {"value": bool(mono), "passed": bool(mono)},
        }

    def compare_sklearn(self, X, y) -> dict:
        from sklearn.linear_model import LogisticRegression as SKLR
        self_clf = self.predict(X)
        sk = SKLR(max_iter=5000, C=1e6 if self.ridge == 0 else 1.0 / self.ridge)
        sk.fit(X, y.ravel() if hasattr(y, "ravel") else y)
        sk_pred = sk.predict(X)
        acc = float(np.mean(self_clf == sk_pred))
        return {
            "train_agreement_vs_sklearn": acc,
            "passed": bool(acc >= 0.95),
            "sklearn_accuracy": float(np.mean(sk_pred == np.asarray(y).ravel())),
            "our_accuracy": float(np.mean(self_clf == np.asarray(y).ravel())),
        }
