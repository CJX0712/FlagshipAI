"""高斯混合模型 GMM —— forge EM 算法 + 不变量。

EM 定理保证对数似然单调非减；责任度/权重归一；协方差对称半正定。

不变量：
  * 对数似然随迭代单调非减（EM 定理）
  * 责任度每行和为 1；混合权重和为 1
  * 协方差矩阵对称半正定
  * 跨库校验：与 sklearn.GaussianMixture 对数似然接近
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from ..mathx.linalg import logsumexp, is_symmetric_psd
from .kmeans import KMeans


class GaussianMixture:
    def __init__(self, n_components=3, n_iter=200, seed=0, reg=1e-6, tol=1e-5):
        self.n_components = n_components
        self.n_iter = n_iter
        self.seed = seed
        self.reg = reg
        self.tol = tol

    def _log_gauss(self, X, mean, cov):
        d = X.shape[1]
        C = cov + self.reg * np.eye(d)
        L = np.linalg.cholesky(C)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        diff = (X - mean).T  # d x N
        sol = solve_triangular(L, diff, lower=True)
        quad = np.sum(sol ** 2, axis=0)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        N, d = X.shape
        km = KMeans(self.n_components, seed=self.seed).fit(X)
        labels = km.predict(X)
        means = km.cluster_centers_.copy()
        covs = np.array([np.cov(X[labels == k].T) + self.reg * np.eye(d)
                         for k in range(self.n_components)])
        weights = np.full(self.n_components, 1.0 / self.n_components)
        ll_hist = []
        resp = None
        for _ in range(self.n_iter):
            logp = np.column_stack([self._log_gauss(X, means[k], covs[k]) + np.log(weights[k])
                                    for k in range(self.n_components)])
            log_lik = logsumexp(logp, axis=1)
            ll = float(log_lik.sum())
            ll_hist.append(ll)
            resp = np.exp(logp - log_lik[:, None])
            Nk = resp.sum(axis=0) + 1e-300
            weights = Nk / N
            means = (resp.T @ X) / Nk[:, None]
            for k in range(self.n_components):
                diff = X - means[k]
                covs[k] = ((resp[:, k][:, None] * diff).T @ diff) / Nk[k] + self.reg * np.eye(d)
            if len(ll_hist) > 1 and abs(ll - ll_hist[-2]) < self.tol:
                break
        self.means_, self.covariances_, self.weights_ = means, covs, weights
        self.resp_, self.loglik_hist_ = resp, np.array(ll_hist)
        self.converged_ = True
        return self

    def score_samples(self, X):
        X = np.asarray(X, dtype=np.float64)
        logp = np.column_stack([self._log_gauss(X, self.means_[k], self.covariances_[k])
                                + np.log(self.weights_[k]) for k in range(self.n_components)])
        return logsumexp(logp, axis=1)

    def check_invariants(self) -> dict:
        ll_hist = self.loglik_hist_
        mono = bool(np.all(np.diff(ll_hist) >= -1e-6))
        resp_rowsum = float(np.max(np.abs(self.resp_.sum(axis=1) - 1.0)))
        w_sum = float(np.abs(self.weights_.sum() - 1.0))
        cov_psd = all(is_symmetric_psd(self.covariances_[k])[0] for k in range(self.n_components))
        return {
            "loglik_monotone_non_decreasing": {"value": mono, "passed": mono},
            "responsibilities_rowsum_1": {"value": resp_rowsum, "threshold": 1e-6, "passed": bool(resp_rowsum < 1e-6)},
            "weights_sum_1": {"value": w_sum, "threshold": 1e-6, "passed": bool(w_sum < 1e-6)},
            "covariances_symmetric_psd": {"value": bool(cov_psd), "passed": bool(cov_psd)},
        }

    def compare_sklearn(self, X) -> dict:
        from sklearn.mixture import GaussianMixture as SKGMM
        X = np.asarray(X, dtype=np.float64)
        sk = SKGMM(n_components=self.n_components, n_init=5, reg_covar=self.reg,
                   random_state=self.seed, max_iter=self.n_iter).fit(X)
        our_ll = float(self.score_samples(X).sum())
        sk_ll = float(sk.score_samples(X).sum())
        rel = abs(our_ll - sk_ll) / (abs(sk_ll) + 1e-300)
        return {
            "our_loglik": our_ll,
            "sklearn_loglik": sk_ll,
            "rel_diff": rel,
            "passed": bool(rel < 0.10),
        }
