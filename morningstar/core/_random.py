"""可复现随机源。

零重写：直接基于 numpy 的 PCG64 (default_rng)。所有 forge 模块默认从这里取种子，
保证「可复现」是交付闭环的硬要求。
"""
from __future__ import annotations

import numpy as np


def rng(seed: int = 0) -> np.random.Generator:
    """返回确定性的 numpy Generator。"""
    return np.random.default_rng(seed)


def make_classification(
    n_samples: int = 300,
    n_features: int = 4,
    n_classes: int = 3,
    n_clusters_per_class: int = 1,
    seed: int = 0,
    noise: float = 0.1,
):
    """生成带标签的分类数据（封装 sklearn，零重写）。"""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 1),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        random_state=seed,
    )
    if noise > 0:
        X = X + rng(seed + 1).normal(0.0, noise, X.shape)
    return X, y


def make_blobs(
    n_samples: int = 300,
    n_features: int = 2,
    centers: int = 3,
    seed: int = 0,
    spread: float = 1.0,
):
    """生成聚类 blob（封装 sklearn，零重写）。"""
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=spread,
        random_state=seed,
    )
    return X, y


def make_regression(
    n_samples: int = 200,
    n_features: int = 3,
    noise: float = 0.1,
    seed: int = 0,
    coef: bool = False,
):
    """生成回归数据（封装 sklearn，零重写）。"""
    from sklearn.datasets import make_regression

    return make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=seed,
        coef=coef,
    )
