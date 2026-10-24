"""确定性可复现随机源（复盘闭环的硬要求）。

设计要点（零重写）:

  * 直接复用 numpy 的 **PCG64** bit generator (`np.random.default_rng`) ——
    这是 Melissa O'Neill PCG 家族的世界级实现 (O'Neill 2014)，
    我们不自写 PRNG，只做「种子派发」这一层。
  * 借鉴 EffAttnForge 的 **salt 派生** 模式: 全局 base seed + 固定 salt
    → 每个子流一个 Generator。salt 是代码里的常量（绝不随机），
    因此随机数消耗顺序跨运行完全一致 ⇒ 同 seed **比特级可复现**。

为什么这么做很重要:
    主流框架默认不保证 cross-run bit-exactness（多线程归约顺序不确定、
    隐式初始化顺序不确定）。科学复现要求「同 seed ⇒ 同权重 ⇒ 同指标」，
    本模块把这条约束显式化。
"""
from __future__ import annotations

import numpy as np

# 黄金比例常数（Knuth 散列），用于把 base seed 与 salt 混合，避免低位相关性。
_GOLDEN: int = 0x9E3779B1

_BASE_SEED: int = 1337


def set_seed(seed: int) -> int:
    """设定全局 base seed 并返回它。"""
    global _BASE_SEED
    _BASE_SEED = int(seed)
    return _BASE_SEED


def get_seed() -> int:
    """读取当前全局 base seed。"""
    return int(_BASE_SEED)


def rng_for(salt: int, seed: int | None = None) -> np.random.Generator:
    """派生确定性子流: ``base_seed XOR salt * golden_ratio`` (32-bit 截断)。

    Args:
        salt: 代码中的**常量**（每个消费点一个），用于隔离不同用途的随机流。
        seed: 覆盖全局 base seed（测试用）；None 表示用全局 seed。

    Returns:
        numpy Generator (PCG64)。
    """
    base = _BASE_SEED if seed is None else int(seed)
    mixed = (int(base) ^ (int(salt) * _GOLDEN)) & 0xFFFFFFFF
    return np.random.default_rng(mixed)


# 各用途的 salt 常量表 —— 改这里等于改实验协议，必须显式。
SALT_INIT: int = 0x0001      # 参数初始化
SALT_DATA: int = 0x0002      # 数据打乱 / 采样
SALT_DROPOUT: int = 0x0003   # dropout mask
SALT_BENCH: int = 0x0004     # 基准测试数据生成
SALT_SAMPLE: int = 0x0005    # 生成阶段采样
