"""测试夹具与全局配置。

所有测试必须**确定性可复现**: 统一在此设定全局随机种子, 并禁用可能在
测试中触发 RuntimeWarning 的路径 (pytest 配置里已把 RuntimeWarning 提升为错误)。
"""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import pytest

# 测试期间屏蔽库的 INFO 日志, 保持输出干净
logging.getLogger("morningstar").setLevel(logging.ERROR)

SEED = 20261024


def pytest_configure(config: pytest.Config) -> None:
    """pytest 全局配置钩子。"""
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """会话级确定性随机源。"""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def tiny_cfg():
    """一个极小的 GPT 配置, 供单元测试使用 (秒级完成)。"""
    from morningstar.flash import GPTConfig

    return GPTConfig(
        vocab_size=48, d_model=32, n_layer=2, n_head=4, n_kv_head=2, max_len=64
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg):
    """确定性初始化的极小模型。"""
    from morningstar.flash import GPT

    return GPT(tiny_cfg, seed=SEED)
