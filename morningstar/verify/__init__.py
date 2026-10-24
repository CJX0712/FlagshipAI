"""L12 集成验证层 —— 不变量 + 跨库交叉验证 + 基准 + 报告生成。

对外提供两个顶层能力:

  1. ``verify_suite()``   —— 数学不变量 + PyTorch 交叉验证的完整闭环
  2. ``benchmark_suite()`` —— 注意力家族的性能/内存/保真度评测

结果统一为可 JSON 序列化的 dict，供 CLI / CI / 文档自动消费。
"""
from __future__ import annotations

import importlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from . import crosscheck, invariants

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器
    from .benchmark import benchmark_attention, benchmark_memory_scaling
    from .report import format_markdown_table, write_reports

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "benchmark_attention",
    "benchmark_memory_scaling",
    "crosscheck",
    "format_markdown_table",
    "invariants",
    "run_full_verification",
    "verify_suite",
    "write_reports",
]

# 惰性导入的子模块成员: 避免父包提前 import 子模块导致
# ``python -m morningstar.verify.<子模块>`` 时 runpy 重复执行告警。
_LAZY = {
    "benchmark_attention": ".benchmark",
    "benchmark_memory_scaling": ".benchmark",
    "format_markdown_table": ".report",
    "write_reports": ".report",
}


def __getattr__(name: str):
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def verify_suite() -> dict:
    """跑全部不变量 + 跨库交叉验证。

    Returns:
        ``{"timestamp":..., "env":..., "invariants":[...], "crosschecks":[...],
           "summary": {...}}``
    """
    t0 = time.perf_counter()
    inv = invariants.run_all()
    cross = crosscheck.run_all()
    inv_pass = sum(1 for r in inv if r["passed"])
    cross_pass = sum(1 for r in cross if r["passed"])
    cross_avail = sum(1 for r in cross if r.get("available", True))
    summary = {
        "invariants_total": len(inv),
        "invariants_passed": inv_pass,
        "crosschecks_total": len(cross),
        "crosschecks_passed": cross_pass,
        "crosschecks_available": cross_avail,
        "all_green": inv_pass == len(inv) and cross_pass == len(cross),
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": __import__("numpy").__version__,
        },
        "invariants": inv,
        "crosschecks": cross,
        "summary": summary,
    }


def run_full_verification(
    include_benchmarks: bool = True,
    lengths: tuple[int, ...] = (128, 512, 1024),
    d: int = 32,
) -> dict:
    """一站式: 验证 + 基准。返回完整报告 dict。

    Args:
        include_benchmarks: 是否包含注意力基准 (较慢, 可关闭)。
        lengths:           基准的序列长度档位。
        d:                 基准的 head_dim。
    """
    report = verify_suite()
    if include_benchmarks:
        # 函数内导入: 模块级 __getattr__ 惰性暴露只作用于属性访问,
        # 不作用于同模块函数体内的裸名查找。
        from .benchmark import benchmark_attention, benchmark_memory_scaling

        report["benchmarks"] = {
            "attention": benchmark_attention(lengths=lengths, d=d),
            "memory": benchmark_memory_scaling(lengths=lengths, d=d),
        }
    return report
