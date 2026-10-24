"""K-Means —— forge Lloyd 算法 + 不变量。

不变量：
  * SSE（簇内平方和）随迭代单调非增
  * 收敛后簇分配稳定（不变）
  * 跨库校验：与 sklearn.KMeans 在已知 blob 上 ARI 高、inertia 接近
"""
from __future__ import annotations

import numpy as np


def _kmeans_plus_plus(X, k, rng):
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
    return np.array(centers)


class KMeans:
    def __init__(self, n_clusters=3, n_iter=300, seed=0, tol=1e-4):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.seed = seed
        self.tol = tol

    def _assign(self, X, C):
        d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        return np.argmin(d, axis=1)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        C = _kmeans_plus_plus(X, self.n_clusters, rng)
        self.inertia_hist_ = []
        labels = None
        for _ in range(self.n_iter):
            labels = self._assign(X, C)
            newC = np.array([X[labels == j].mean(axis=0) if np.any(labels == j)
                             else C[j] for j in range(self.n_clusters)])
            shift = np.linalg.norm(newC - C)
            C = newC
            d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
            inertia = float(np.sum(d[np.arange(len(X)), labels] ** 2))
            self.inertia_hist_.append(inertia)
            if shift < self.tol:
                break
        self.cluster_centers_ = C
        self.labels_ = labels
        self.inertia_ = self.inertia_hist_[-1]
        return self

    def predict(self, X):
        return self._assign(np.asarray(X, dtype=np.float64), self.cluster_centers_)

    def check_invariants(self) -> dict:
        hist = np.array(self.inertia_hist_)
        mono = bool(np.all(np.diff(hist) <= 1e-8))
        return {"sse_monotone_non_increasing": {"value": mono, "passed": mono},
                "final_inertia": self.inertia_}

    def compare_sklearn(self, X, y=None) -> dict:
        """与 scikit-learn KMeans 交叉验证。

        判据说明: K-Means 是**非凸优化**, 不同初始化会收敛到不同的局部最优,
        因此「惯性逐位相等」不是正确判据。正确的跨库判据是
        **簇结构等价性** —— 用调整兰德指数 (ARI) 度量两次划分的一致性,
        并辅以惯性相对差异上界。
        """
        from sklearn.cluster import KMeans as SKKM
        from sklearn.metrics import adjusted_rand_score
        X = np.asarray(X, dtype=np.float64)
        sk = SKKM(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(X)
        our = self.predict(X)
        ari = float(adjusted_rand_score(sk.labels_, our))
        inertia_rel = abs(self.inertia_ - sk.inertia_) / max(1.0, abs(sk.inertia_))
        res = {"our_inertia": self.inertia_,
               "sklearn_inertia": float(sk.inertia_),
               "inertia_rel_diff": inertia_rel,
               "ARI_vs_sklearn": ari}
        if y is not None:
            res["ARI_vs_true"] = float(adjusted_rand_score(y, our))
            res["ARI_sklearn_vs_true"] = float(adjusted_rand_score(y, sk.labels_))
        # 簇结构等价 (ARI ≥ 0.9) 或 收敛到等价盆地 (惯性相对差 < 1e-3) 即视为一致。
        # 判据不能用「惯性逐位相等」: K-Means 非凸, sklearn 用 n_init=10 多重启取
        # 最优而本实现单次 k-means++ 初始化, 二者可能落在不同但等价的局部最优。
        res["passed"] = bool(ari >= 0.9 or inertia_rel < 1e-3)
        return res
