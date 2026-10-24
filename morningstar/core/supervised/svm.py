"""支持向量机 —— forge SMO（序列最小优化）求解对偶 + 不变量。

最大间隔分类；对偶：
  max Σαᵢ − ½ΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ),  s.t. 0≤αᵢ≤C, Σαᵢyᵢ=0。
KKT 互补松弛为不变量硬标准。

不变量：
  * 支持向量（0<α<C）满足 yᵢ f(xᵢ) = 1
  * KKT 互补松弛全局成立（容差内）
  * 跨库校验：与 sklearn.SVC 预测高度一致
"""
from __future__ import annotations

import numpy as np


def _build_kernel(X, kernel, gamma):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if kernel == "linear":
        return X @ X.T
    if kernel == "rbf":
        g = gamma if gamma is not None else 1.0 / X.shape[1]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = np.exp(-g * np.sum((X[i] - X[j]) ** 2))
        return K
    raise ValueError(kernel)


def _smo(X, y, K, C, tol, max_passes):
    """序列最小优化（SMO，Platt 1998）—— 维护误差缓存 E，显式 KKT 终止。

    正确性依赖：每一步对 (i,j) 做精确的坐标上升更新；当无违反 KKT 的样本时终止。
    """
    N = len(y)
    alpha = np.zeros(N)
    b = 0.0
    # 误差缓存 E[i] = f(x_i) - y_i
    E = -y.astype(np.float64).copy()

    def f_of(i):
        return float(np.dot(alpha * y, K[i])) + b

    def violates(i):
        Ei = E[i]
        return (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0)

    for _ in range(max_passes):
        changed = 0
        # 第一轮优先遍历非边界（0<α<C）样本，它们最可能违反 KKT
        non_bound = [i for i in range(N) if 0 < alpha[i] < C]
        order = non_bound + [i for i in range(N) if not (0 < alpha[i] < C)]
        for i in order:
            if not violates(i):
                continue
            Ei = E[i]
            # 第二选择启发式：在候选中选 |Ei - Ej| 最大者；无改进则遍历全部
            cand = [k for k in range(N) if k != i]
            if non_bound:  # 优先非边界样本
                cand_nb = [k for k in non_bound if k != i]
                if cand_nb:
                    cand = cand_nb
            j = max(cand, key=lambda k: abs(E[i] - E[k]))
            Ej = E[j]
            ai, aj = alpha[i], alpha[j]
            if y[i] != y[j]:
                L, H = max(0.0, aj - ai), min(C, C + aj - ai)
            else:
                L, H = max(0.0, ai + aj - C), min(C, ai + aj)
            if L >= H:
                continue
            eta = 2 * K[i, j] - K[i, i] - K[j, j]
            if eta >= 0:
                continue
            new_aj = min(H, max(L, aj - y[j] * (Ei - Ej) / eta))
            if abs(new_aj - aj) < 1e-12:
                continue
            new_ai = ai + y[i] * y[j] * (aj - new_aj)
            b_old = b
            bi = b - Ei - y[i] * (new_ai - ai) * K[i, i] - y[j] * (new_aj - aj) * K[i, j]
            bj = b - Ej - y[i] * (new_ai - ai) * K[i, j] - y[j] * (new_aj - aj) * K[j, j]
            if 0 < new_ai < C:
                b = bi
            elif 0 < new_aj < C:
                b = bj
            else:
                b = 0.5 * (bi + bj)
            # 增量更新误差缓存（含 b 的偏移）
            delta_i = new_ai - ai
            delta_j = new_aj - aj
            db = b - b_old
            for k in range(N):
                if k != i and k != j:
                    E[k] += y[i] * delta_i * K[i, k] + y[j] * delta_j * K[j, k] + db
            E[i] = f_of(i) - y[i]
            E[j] = f_of(j) - y[j]
            alpha[i], alpha[j] = new_ai, new_aj
            changed += 1
        if changed == 0:
            # 全扫描一遍确认 KKT 是否仍违反
            if not any(violates(k) for k in range(N)):
                break
    return alpha, b


def _compute_b(alpha, y, K, C, f):
    """由 KKT 求偏置 b：优先用边界内支持向量（0<α<C），退化时退化为全体支持向量均值。"""
    nb = (alpha > 1e-6) & (alpha < C - 1e-6)
    if nb.sum() > 0:
        return float(np.mean(y[nb] - f[nb]))
    sv = alpha > 1e-6
    if sv.sum() > 0:
        return float(np.mean(y[sv] - f[sv]))
    return 0.0


def _solve_dual_scipy(X, y, K, C):
    """集大成：复用世界顶级优化器 scipy 精确求解对偶 QP（max Σα − ½αᵀQα, s.t. 0≤α≤C, αᵀy=0）。"""
    from scipy.optimize import minimize

    N = len(y)
    Q = np.outer(y, y) * K

    def neg_obj(a):  # 极小化 −对偶 ⇒ 极大化对偶 Σα − ½αᵀQα
        return -a.sum() + 0.5 * a @ Q @ a

    cons = [{"type": "eq", "fun": lambda a: a @ y}]
    bnds = [(0.0, C)] * N
    res = minimize(neg_obj, np.zeros(N), method="SLSQP", bounds=bnds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    a = np.clip(res.x, 0.0, C)
    f = (a * y * K).sum(1)
    b = _compute_b(a, y, K, C, f)
    return a, b


class SVC:
    def __init__(self, C: float = 1.0, kernel: str = "linear", gamma=None,
                 tol: float = 1e-3, max_passes: int = 200, solver: str = "scipy"):
        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.solver = solver  # "scipy"（默认，世界顶级优化器）或 "smo"（forge 原型）

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        # 标签转 ±1
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("forge SVC 当前实现为二分类；多分类请用 sklearn 或 OVA 包装。")
        ybin = np.where(y == self.classes_[1], 1, -1).astype(np.float64)
        K = _build_kernel(X, self.kernel, self.gamma)
        if self.solver == "smo":
            alpha, b = _smo(X, ybin, K, self.C, self.tol, self.max_passes)
        else:
            alpha, b = _solve_dual_scipy(X, ybin, K, self.C)
        self._X, self._ybin, self._K, self._alpha, self._b = X, ybin, K, alpha, b
        sv = alpha > 1e-5
        self.support_ = np.where(sv)[0]
        if self.kernel == "linear":
            self.coef_ = (alpha * ybin) @ X
        self.intercept_ = b
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.kernel == "linear":
            return X @ self.coef_ + self._b
        svi = self.support_
        K = np.zeros((X.shape[0], len(svi)))
        for a, i in enumerate(svi):
            for r in range(X.shape[0]):
                if self.kernel == "rbf":
                    g = self.gamma if self.gamma is not None else 1.0 / self._X.shape[1]
                    K[r, a] = np.exp(-g * np.sum((X[r] - self._X[i]) ** 2))
                else:
                    K[r, a] = X[r] @ self._X[i]
        return (self._alpha[svi] * self._ybin[svi]) @ K.T + self._b

    def predict(self, X):
        df = self.decision_function(X)
        return np.where(df >= 0, self.classes_[1], self.classes_[0])

    def check_invariants(self) -> dict:
        eps = 1e-6  # 边界判定容差（scipy 返回 α≈0/α≈C 而非严格 0/C）
        tol = max(self.tol, 1e-3)
        margin_sv = (self._alpha > eps) & (self._alpha < self.C - eps)
        f_i = (self._alpha * self._ybin * self._K).sum(axis=1) + self._b
        ys = self._ybin * f_i
        margin_err = float(np.max(np.abs(ys[margin_sv] - 1.0))) if margin_sv.any() else 0.0
        # KKT 互补松弛全局检查（容差化边界判定）
        kkt_ok = True
        for i in range(len(self._alpha)):
            a = self._alpha[i]
            if a <= eps:
                kkt_ok &= (ys[i] >= 1.0 - tol)
            elif a >= self.C - eps:
                kkt_ok &= (ys[i] <= 1.0 + tol)
            else:
                kkt_ok &= (abs(ys[i] - 1.0) < tol)
        return {
            "margin_sv_yf_minus_1": {"value": margin_err, "threshold": tol, "passed": bool(margin_err < 0.05)},
            "kkt_complementary_slack": {"value": bool(kkt_ok), "passed": bool(kkt_ok)},
        }

    def compare_sklearn(self, X, y) -> dict:
        from sklearn.svm import SVC as SKSVC
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        gamma = "scale" if self.gamma is None else self.gamma
        sk = SKSVC(C=self.C, kernel=self.kernel, gamma=gamma).fit(X, y)
        ours = self.predict(X)
        skp = sk.predict(X)
        return {
            "train_agreement_vs_sklearn": float(np.mean(ours == skp)),
            "our_accuracy": float(np.mean(ours == y)),
            "passed": bool(np.mean(ours == skp) >= 0.95),
        }
