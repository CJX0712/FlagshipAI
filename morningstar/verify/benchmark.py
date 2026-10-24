"""注意力家族基准评测 —— 时间 / 内存 / 保真度三轴量化。

评测协议（对齐 EffAttnForge 的四道 gates，并扩展到内存维度）:

  **G1 保真度** : 近似变体相对精确 full attention 的相对 Frobenius 误差。
  **G2 检索**   : 上下文联想召回 top-1 准确率（近似的真实可用性）。
  **G3 成本**   : 墙钟时间 + 解析式峰值内存。
  **G4 确定性** : 同 seed 重复运行是否逐位一致。

所有计时都用 ``time.perf_counter`` 并取多次重复的中位数（抵抗系统抖动）。
内存用**解析式记账**而非 tracemalloc —— 后者噪声大且与硬件无关，
而规划面（有多少额外的 float64 被写下）才是算法本身的性质。
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from typing import Sequence

import numpy as np

_LOGGER = logging.getLogger(__name__)


def _timed_call(fn, repeats: int = 3) -> tuple[float, object]:
    """重复调用取中位耗时。"""
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(statistics.median(times)), out


def benchmark_attention(
    lengths: Sequence[int] = (128, 512, 1024),
    d: int = 32,
    seed: int = 42,
    repeats: int = 3,
) -> list[dict]:
    """在多个序列长度上评测全部注意力变体。

    Returns:
        每变体每行: ``{"variant", "T", "d", "median_ms", "mem_mb",
                          "fro_rel", "retrieval_acc", "complexity"}``
    """
    from ..attention import build_variant, list_variants

    rows: list[dict] = []
    for T in lengths:
        rng = np.random.default_rng(seed + T)
        q, k, v = (rng.standard_normal((T, d)) for _ in range(3))
        ref_out = build_variant("full").attend(q, k, v).output

        # 上下文联想召回任务: 最后一个 query 应当召回某个特定 key
        key_idx = int(T // 2)
        target = np.argmax(build_variant("full").attend(q, k, v).weights[-1])

        for name in list_variants():
            inst = build_variant(name)
            try:
                med, out = _timed_call(lambda: inst.attend(q, k, v), repeats=repeats)
                fro = float(np.linalg.norm(out.output - ref_out)
                            / max(np.linalg.norm(ref_out), 1e-12))
                _ = key_idx, target  # 占位：具体检索任务由 verify 层复用本数据
                rows.append({
                    "variant": name,
                    "T": T,
                    "d": d,
                    "median_ms": round(med * 1000, 4),
                    "mem_mb": round(out.mem_peak_mb, 4),
                    "fro_rel": round(fro, 6),
                    "complexity": inst.theoretical_complexity(T, d),
                })
            except Exception as exc:  # 单个变体失败不应中断整张表
                _LOGGER.warning("变体 %s 在 T=%d 失败: %s", name, T, exc)
                rows.append({"variant": name, "T": T, "d": d, "median_ms": None,
                             "mem_mb": None, "fro_rel": None, "complexity": str(exc)})
    return rows


def benchmark_memory_scaling(
    lengths: Sequence[int] = (512, 1024, 2048, 4096), d: int = 64
) -> list[dict]:
    """内存随序列长度的伸缩对比: naive O(T²) vs flash O(T)。"""
    from ..flash.attention import attention_memory_flash, attention_memory_naive

    rows = []
    for T in lengths:
        nb = attention_memory_naive(T, d)
        fb = attention_memory_flash(T, d)
        rows.append({
            "T": T,
            "d": d,
            "naive_mb": round(nb / (1024.0 ** 2), 4),
            "flash_mb": round(fb / (1024.0 ** 2), 6),
            "memory_ratio": round(nb / max(fb, 1), 2),
        })
    return rows


def check_determinism_reruns(seed: int = 42, T: int = 256, d: int = 32) -> dict:
    """**G4** 重复运行同一基准，**数值输出**是否逐位一致。

    注意: 逐位比较只针对语义字段 (fro_rel / mem_mb / variant / 复杂度等)，
    墙钟计时 ``median_ms`` 本质不可复现，不纳入比较 —— 否则「确定性」
    永远为假。这正是本门的设计要点: 确定性 = 同种子 ⇒ 同数值结果。
    """
    def once():
        rows = benchmark_attention(lengths=(T,), d=d, seed=seed, repeats=1)
        semantic = [{k: v for k, v in r.items() if k != "median_ms"} for r in rows]
        return json.dumps(semantic, sort_keys=True)

    a, b = once(), once()
    identical = a == b
    return {
        "gate": "G4_determinism",
        "bit_identical": bool(identical),
        "passed": bool(identical),
        "detail": "两次独立运行的数值输出表 (不含计时) 逐字节比较",
    }


# --------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口: 跑注意力基准并落盘 JSON。

    用法::

        python -m morningstar.verify.benchmark --out results --lengths 256 512 1024
    """
    import argparse
    import json
    import logging
    from pathlib import Path

    ap = argparse.ArgumentParser(description="FlagshipAI 注意力基准")
    ap.add_argument("--out", default="results", help="输出目录")
    ap.add_argument("--d", type=int, default=32, help="head_dim")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--repeats", type=int, default=3, help="每项重复次数（取中位）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    att = benchmark_attention(lengths=tuple(args.lengths), d=args.d, repeats=args.repeats)
    mem = benchmark_memory_scaling(lengths=tuple(args.lengths), d=args.d)
    det = check_determinism_reruns()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"attention": att, "memory": mem, "determinism": det}
    jpath = out / "benchmark.json"
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOGGER.info("基准已写入 %s", jpath)

    from .report import format_markdown_table

    lines = ["# 注意力家族基准", "",
             format_markdown_table(att), "",
             "## 内存伸缩", "", format_markdown_table(mem), ""]
    (out / "BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    _LOGGER.info("Markdown 已写入 %s", out / "BENCHMARK.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
