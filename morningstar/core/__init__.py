"""MorningStar-AI —— 世界顶级 AI 系统底座。

设计哲学（零重写 / 集大成）：
  * 数值重活与成熟算法直接复用世界顶级库（numpy / scipy / scikit-learn / torch）。
  * 核心算法另以纯实现 forge 模块重写，并用「数学不变量 + 跨库交叉验证」
    独立证明其世界级正确性 —— 不靠信仰，靠可验证的硬标准。

十层架构映射（详见 docs/ARCHITECTURE.md）：
  L1 线性代数   → morningstar_ai.mathx.linalg
  L2 概率统计   → (嵌入各概率模块)
  L3 微积分优化 → morningstar_ai.mathx.optimize
  L4-6 监督学习 → morningstar_ai.supervised.*
  L7-8 无监督   → morningstar_ai.unsupervised.*
  L9   深度学习 → morningstar_ai.deep.*
  L10  集成交付 → benchmarks + CLI + (外部 1024 微指令工程)
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星 (MorningStar)"

from . import mathx, supervised, unsupervised, deep, benchmarks  # noqa: F401


def info() -> str:
    return (
        f"MorningStar-AI v{__version__} | 作者 晨星\n"
        "零重写 · 集大成 · 不变量验证 · 跨库交叉验证 · 闭环交付"
    )
