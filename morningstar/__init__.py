"""MorningStar FlagshipAI —— 世界顶级 AI 系统（旗舰版）。

作者: 晨星 (MorningStar)

设计信条（零重写 / 集大成 / 可验证 / 闭环交付）:

  * **零重写**: 数值重活与成熟算法直接复用世界顶级开源成果
    (numpy / scipy / scikit-learn / torch) 与本项目已验证资产,
    绝不从零造轮子。
  * **集大成**: 每一层都站在已发布的世界级算法之上
    (FlashAttention-2 / RoPE / RMSNorm / SwiGLU / AdamW / BPE / Jacobi / EM),
    并注明出处 (见 docs/REFERENCES.md)。
  * **可验证**: 每个算子至少有一条可被独立实现交叉验证的**硬不变量**
    (数值等价 / 正交性 / 梯度检验 / EM 单调 / 比特级确定性)。
  * **闭环交付**: 完整开源、可运行、可复现、可验证。

十二层架构映射 (详见 docs/ARCHITECTURE.md):

  L1  线性代数        → morningstar.core.mathx.linalg
  L2  概率统计        → morningstar.core.mathx / core.supervised.naive_bayes
  L3  微积分与优化     → morningstar.core.mathx.optimize  + flash.optim
  L4  线性/广义模型    → morningstar.core.supervised.linear / logistic
  L5  核方法与 SVM    → morningstar.core.supervised.svm
  L6  概率与非参模型   → morningstar.core.supervised.knn / naive_bayes
  L7  聚类            → morningstar.core.unsupervised.kmeans / gmm
  L8  降维            → morningstar.core.unsupervised.pca
  L9  神经网络        → morningstar.core.deep.neural_net / attention
  L10 高效注意力       → morningstar.attention  (full/flash/linear/performer/sparse/fuse)
  L11 大模型内核       → morningstar.flash      (GPT / RoPE / RMSNorm / KV-Cache / BPE)
  L12 集成验证与交付   → morningstar.verify     (不变量 + 跨库交叉验证 + 基准)
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "1.0.0"
__author__ = "晨星 (MorningStar)"
__license__ = "Apache-2.0"

# 惰性子模块导入: 既保留 `morningstar.flash` 这样的便捷访问,
# 又避免 `python -m morningstar.verify.report` 时的模块重复执行告警。
_LAZY = ("attention", "core", "flash", "verify")

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器
    from . import attention, core, flash, verify


def __getattr__(name: str):
    if name in _LAZY:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


def info() -> str:
    """一行自我介绍。"""
    return (
        f"MorningStar FlagshipAI v{__version__} | 作者 {__author__} | {__license__}\n"
        "零重写 · 集大成 · 不变量验证 · 跨库交叉验证 · 闭环交付"
    )


__all__ = ["attention", "core", "flash", "verify", "info", "__version__", "__author__"]
