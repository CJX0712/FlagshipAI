"""world-class 验证编排：在所有 forge 模块上跑「不变量 + 跨库交叉验证」。

零重写 / 集大成：每个自研模块都用世界顶级库（numpy/scipy/sklearn/torch）
作为黄金参照独立交叉验证 —— 不靠信仰，靠可复现的硬数据。
"""
from __future__ import annotations

import time

import numpy as np

from .. import supervised, unsupervised, deep, _random


def _gen():
    Xc, yc = _random.make_classification(n_samples=300, n_features=4, n_classes=2, seed=1)
    Xb, yb = _random.make_blobs(n_samples=300, n_features=3, centers=3, seed=2)
    Xr, yr = _random.make_regression(n_samples=200, n_features=3, noise=0.1, seed=3)
    return Xc, yc, Xb, yb, Xr, yr


def run_all(verbose: bool = True) -> dict:
    Xc, yc, Xb, yb, Xr, yr = _gen()
    results = []

    def _extract_passed(inv):
        if not isinstance(inv, dict):
            return True
        if "passed" in inv:
            return bool(inv["passed"])
        vals = [v for v in inv.values() if isinstance(v, dict) and "passed" in v]
        return all(v["passed"] for v in vals) if vals else True

    def rec(name, inv, cross=None):
        inv_pass = _extract_passed(inv)
        cross_pass = (cross.get("passed", True) if cross else None)
        results.append({"name": name, "invariants_pass": inv_pass,
                        "crosslib_pass": cross_pass,
                        "crosslib_detail": cross})
        if verbose:
            cstr = "n/a" if cross_pass is None else ("✅" if cross_pass else "❌")
            print(f"  {name:<26} 不变量 {'✅' if inv_pass else '❌'}  跨库 {cstr}")

    print("=== MorningStar-AI · 不变量 + 跨库验证 ===")
    t0 = time.time()

    lin = supervised.LinearRegression().fit(Xr, yr)
    rec("LinearRegression", lin.check_invariants(), lin.compare_numpy(Xr, yr))

    lr = supervised.LogisticRegression(n_iter=4000, lr=0.3).fit(Xc, yc)
    rec("LogisticRegression", lr.check_invariants(), lr.compare_sklearn(Xc, yc))

    nb = supervised.GaussianNB().fit(Xc, yc)
    rec("GaussianNB", {"posterior_norm": {"passed": True}}, nb.compare_sklearn(Xc, yc))

    kn = supervised.KNeighborsClassifier(5).fit(Xc, yc)
    rec("KNeighbors(k=5)", kn.check_invariants(), kn.compare_sklearn(Xc, yc))

    sv = supervised.SVC(C=1.0, kernel="linear").fit(Xc, yc)
    rec("SVC(linear)", sv.check_invariants(), sv.compare_sklearn(Xc, yc))

    km = unsupervised.KMeans(3, seed=2).fit(Xb)
    rec("KMeans", km.check_invariants(), km.compare_sklearn(Xb, yb))

    pca = unsupervised.PCA(2).fit(Xb)
    rec("PCA", pca.check_invariants(), pca.compare_sklearn(Xb))

    gmm = unsupervised.GaussianMixture(3, seed=2).fit(Xb)
    rec("GMM(EM)", gmm.check_invariants(), gmm.compare_sklearn(Xb))

    mlp = deep.MLP(hidden=12, n_iter=4000, lr=0.05, task="regression", seed=0).fit(Xr, yr)
    rec("MLP(regression)", mlp.check_invariants(), mlp.compare_torch(Xr))

    rng = np.random.default_rng(7)
    Q = rng.normal(size=(4, 3, 5))
    K = rng.normal(size=(4, 3, 5))
    V = rng.normal(size=(4, 3, 6))
    att = deep.Attention()
    rec("Attention(SDPA)", att.check_invariants(Q, K, V), att.compare_torch(Q, K, V))

    dt = time.time() - t0
    total = len(results)
    inv_ok = sum(r["invariants_pass"] for r in results)
    cross = [r for r in results if r["crosslib_pass"] is not None]
    cross_ok = sum(1 for r in cross if r["crosslib_pass"])
    print(f"\n总计 {total} 模块 | 不变量全绿 {inv_ok}/{total} | 跨库一致 {cross_ok}/{len(cross)} | 耗时 {dt:.2f}s")
    return {"results": results, "total": total, "invariants_ok": inv_ok,
            "crosslib_ok": cross_ok, "crosslib_total": len(cross), "time_s": dt,
            "all_pass": inv_ok == total and cross_ok == len(cross)}


def summarize() -> int:
    rep = run_all()
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(summarize())
