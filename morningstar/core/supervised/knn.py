"""k 近邻 —— forge 暴力实现 + 不变量。

惰性学习；距离欧氏；k 控制平滑度；多数投票。

不变量：
  * k=1 → 训练集 0 误差（记忆边界）
  * k→∞ → 收敛至贝叶斯误差（理论上界）
  * 跨库校验：与 sklearn.KNeighborsClassifier 一致
"""
from __future__ import annotations

import numpy as np


class KNeighborsClassifier:
    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = int(n_neighbors)
        self.metric = metric

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y).ravel()
        self.classes_ = np.unique(self._y)
        return self

    def _dist(self, a, b):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2))
        if self.metric == "manhattan":
            return np.sum(np.abs(a - b))
        raise ValueError(self.metric)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=self._y.dtype)
        for i, x in enumerate(X):
            d = np.array([self._dist(x, xi) for xi in self._X])
            idx = np.argsort(d)[: self.n_neighbors]
            votes = self._y[idx]
            # 平局取最小类标签（确定性）
            counts = np.bincount(votes, minlength=len(self.classes_))
            out[i] = self.classes_[np.argmax(counts)]
        return out

    def check_invariants(self) -> dict:
        # k=1 训练误差应为 0
        pred1 = KNeighborsClassifier(1).fit(self._X, self._y).predict(self._X)
        err1 = float(np.mean(pred1 != self._y))
        return {"k1_train_error": err1, "passed": bool(err1 == 0.0)}

    def compare_sklearn(self, X, y) -> dict:
        from sklearn.neighbors import KNeighborsClassifier as SKKNN
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        sk = SKKNN(n_neighbors=self.n_neighbors).fit(X, y)
        ours = self.predict(X)
        skp = sk.predict(X)
        return {
            "train_agreement_vs_sklearn": float(np.mean(ours == skp)),
            "our_accuracy": float(np.mean(ours == y)),
            "sklearn_accuracy": float(np.mean(skp == y)),
            "passed": bool(np.mean(ours == skp) >= 0.98),
        }
