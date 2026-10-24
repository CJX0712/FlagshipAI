"""报告生成 —— 把验证/基准结果渲染成 Markdown 表并落盘 JSON/MD。

为什么单独一层: 基准数据如果只躺在终端里，就无法进入文档、无法被 CI 消费、
无法被审稿者复现。**把「实验结果 → 文档表格」自动化**是可信交付的关键一环。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

_LOGGER = logging.getLogger(__name__)

_STATUS = {True: "✅ PASS", False: "❌ FAIL"}


def format_markdown_table(rows: Sequence[dict], columns: Sequence[str] | None = None) -> str:
    """把 dict 列表渲染成 Markdown 表格。"""
    if not rows:
        return "_（无数据）_"
    cols = list(columns) if columns else list(rows[0].keys())
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, bool):
                cells.append(_STATUS[v])
            elif isinstance(v, float):
                cells.append(f"{v:.3e}" if (abs(v) < 1e-3 and v != 0) else f"{v:.4f}")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_verification_md(report: dict) -> str:
    """把 verify_suite 结果渲染成完整 Markdown 报告。"""
    s = report["summary"]
    lines: list[str] = [
        "# MorningStar FlagshipAI · 验证报告",
        "",
        f"- 生成时间: {report['timestamp']} (UTC)",
        f"- 环境: Python {report['env']['python']} · numpy {report['env']['numpy']}",
        f"- 平台: {report['env']['platform']}",
        "",
        "## 总览",
        "",
        "| 项目 | 结果 |",
        "| --- | --- |",
        f"| 数学不变量 | {s['invariants_passed']}/{s['invariants_total']} PASS |",
        f"| 跨库交叉验证 (PyTorch) | {s['crosschecks_passed']}/{s['crosschecks_total']} PASS "
        f"（可用 {s['crosschecks_available']}） |",
        f"| 全绿 | {'✅ 是' if s['all_green'] else '❌ 否'} |",
        f"| 耗时 | {s['wall_s']} s |",
        "",
        "## 数学不变量明细",
        "",
        format_markdown_table(
            report["invariants"], ["name", "value", "threshold", "passed", "detail"]
        ),
        "",
        "## 跨库交叉验证明细 (vs PyTorch)",
        "",
        format_markdown_table(
            report["crosschecks"], ["name", "value", "threshold", "passed", "detail"]
        ),
        "",
    ]
    benches = report.get("benchmarks")
    if benches:
        if "attention" in benches:
            lines += ["## 注意力家族基准", "",
                      format_markdown_table(benches["attention"]), ""]
        if "memory" in benches:
            lines += ["## 内存伸缩对比 (naive O(T²) vs flash O(T))", "",
                      format_markdown_table(benches["memory"]), ""]
    return "\n".join(lines)


def write_reports(report: dict, outdir: str = "results") -> list[str]:
    """落盘 ``verification.json`` 与 ``VERIFICATION.md``。

    Args:
        report: ``verify_suite()`` / ``run_full_verification()`` 的返回值。
        outdir: 输出目录（不存在则创建）。

    Returns:
        写入的文件路径列表。
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / "verification.json"
    mpath = out / "VERIFICATION.md"
    jpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mpath.write_text(render_verification_md(report), encoding="utf-8")
    _LOGGER.info("报告已写入 %s 与 %s", jpath, mpath)
    return [str(jpath), str(mpath)]


# --------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口: 跑完整验证套件并落盘报告。

    用法::

        python -m morningstar.verify.report --out results
    """
    import argparse
    import logging

    from . import run_full_verification

    ap = argparse.ArgumentParser(description="FlagshipAI 验证套件")
    ap.add_argument("--out", default="results", help="输出目录")
    ap.add_argument("--no-bench", action="store_true", help="跳过基准评测")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_full_verification(include_benchmarks=not args.no_bench)
    paths = write_reports(report, args.out)
    s = report["summary"]
    _LOGGER.info("不变量 %d/%d · 跨库 %d/%d · 全绿=%s",
                 s["invariants_passed"], s["invariants_total"],
                 s["crosschecks_passed"], s["crosschecks_total"],
                 s["all_green"])
    for pth in paths:
        _LOGGER.info("  → %s", pth)
    return 0 if s["all_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
