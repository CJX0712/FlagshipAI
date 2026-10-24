#!/usr/bin/env python
"""旗舰 GPT 训练示例 —— 在可复现语料上完成一次真实收敛。

用法::

    python examples/train_tiny.py                  # 默认配置 (CPU 可跑)
    python examples/train_tiny.py --steps 2000     # 更长训练
    python examples/train_tiny.py --out results/  # 指定输出目录

设计说明
--------
语料采用**一阶马尔可夫链**而非随机序列：随机序列的信息论下界就是
``ln(V)``，模型无从学习，无法证明训练闭环有效。马尔可夫链的下一状态
可由当前状态预测，理论损失下界为 ``H(group)``，因此损失曲线具有
明确的「可证伪」目标。

复现性: 所有随机源（初始化 / 数据采样 / 采样生成）均由显式 seed 驱动。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from morningstar.flash import GPT, GPTConfig  # noqa: E402
from morningstar.flash.tokenizer import BPETokenizer  # noqa: E402
from morningstar.flash.trainer import TrainConfig, perplexity, save_log, train  # noqa: E402

_LOG = logging.getLogger("train_tiny")


# --------------------------------------------------------------------- 语料
def make_markov_corpus(
    n_tokens: int, n_states: int = 8, group: int = 8, seed: int = 7
) -> tuple[np.ndarray, int]:
    """构造一阶马尔可夫链语料。

    token = state * group + noise_index, 因此「下一状态」可由「当前 token」
    推出，而「噪声下标」不可预测 ⇒ 理论损失下界 = ln(group)。

    Returns:
        (token 流, 词表大小)
    """
    rng = np.random.default_rng(seed)
    trans = np.full((n_states, n_states), 0.05)
    for i in range(n_states):
        trans[i, (i + 1) % n_states] = 0.45
        trans[i, i] = 0.30
    trans /= trans.sum(axis=1, keepdims=True)

    states = np.empty(n_tokens, dtype=np.int64)
    states[0] = rng.integers(0, n_states)
    for t in range(1, n_tokens):
        states[t] = rng.choice(n_states, p=trans[states[t - 1]])
    tokens = (states * group + rng.integers(0, group, size=n_tokens)).astype(np.int64)
    return tokens, n_states * group


def make_text_corpus(texts: list[str], tokenizer: BPETokenizer) -> np.ndarray:
    """把文本语料编码为 token 流。"""
    ids: list[int] = []
    for t in texts:
        ids.extend(tokenizer.encode(t))
    return np.asarray(ids, dtype=np.int64)


# --------------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="旗舰 GPT 训练示例")
    ap.add_argument("--steps", type=int, default=600, help="训练步数")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-kv-head", type=int, default=2, help="GQA: KV 头数")
    ap.add_argument("--block", type=int, default=32, help="上下文长度")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="results", help="输出目录")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 语料: 80% 马尔可夫链 (可学习) + 20% 真实文本 (tokenizer 闭环)
    markov, markov_vocab = make_markov_corpus(n_tokens=120_000, seed=args.seed)
    tokenizer = BPETokenizer().train(
        [
            "晨星锻造世界顶级人工智能系统。",
            "零重写，集大成，可验证，闭环交付。",
            "attention is all you need; flash attention makes it fit.",
        ],
        vocab_size=300,
    )
    text_tokens = make_text_corpus(
        ["晨星 FlagshipAI：零重写 · 集大成 · 可验证。\n"] * 200, tokenizer
    )
    # 关键: **不可全局 shuffle** —— 那会破坏马尔可夫链的相邻转移结构，
    # 使「下一状态可预测」这一可学习信号消失，模型将永远学不到理论下界。
    train_tokens = np.concatenate([markov, text_tokens.astype(np.int64)])

    # 验证集: **独立生成**的同分布序列 (而非从训练流尾部切分 ——
    # 尾部切分会导致分布不一致, 验证困惑度虚高)。
    val_markov, _ = make_markov_corpus(n_tokens=6_000, seed=args.seed + 1)
    val_text = make_text_corpus(
        ["晨星 FlagshipAI：零重写 · 集大成 · 可验证。\n"] * 12, tokenizer
    )
    val_tokens = np.concatenate([val_markov, val_text.astype(np.int64)])

    # 模型词表必须覆盖两路语料的 token id 上界
    max_id = int(max(train_tokens.max(), val_tokens.max()))
    vocab = int(max(markov_vocab, tokenizer.vocab_size, max_id + 1))
    _LOG.info("语料: train=%d tok, val=%d tok, vocab=%d",
              train_tokens.size, val_tokens.size, vocab)
    _LOG.info("理论下界: ln(vocab)=%.4f", math.log(vocab))

    # ---- 模型
    cfg = GPTConfig(
        vocab_size=vocab,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        max_len=max(args.block, 256),  # 预留充足上下文供增量生成 (KV-Cache 路径)
    )
    model = GPT(cfg, seed=args.seed)
    _LOG.info("模型参数量: %s", f"{model.n_params:,}")

    # ---- 训练
    tcfg = TrainConfig(
        n_steps=args.steps,
        batch_size=args.batch,
        block_size=args.block,
        lr=args.lr,
        warmup=max(10, args.steps // 10),
        eval_every=max(25, args.steps // 6),
        eval_iters=5,
        log_every=max(10, args.steps // 12),
        seed=args.seed,
    )
    t0 = time.perf_counter()
    log = train(model, train_tokens, tcfg, val_tokens=val_tokens)
    wall = time.perf_counter() - t0

    train_losses = [x for x in log.train_loss if not math.isnan(x)]
    val_losses = [x for x in log.val_loss if not math.isnan(x)]
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    _LOG.info("训练完成: train %.4f → %.4f | val %.4f → %.4f | %.1fs",
              train_losses[0], final_train, val_losses[0], final_val, wall)
    _LOG.info("最终困惑度: train %.2f  val %.2f", perplexity(final_train), perplexity(final_val))

    # ---- 采样生成（验证推理闭环 + KV 缓存路径）
    prompt = tokenizer.encode("晨星 FlagshipAI")[:8]
    ids = model.generate(prompt, max_new_tokens=32, temperature=0.8, seed=args.seed)
    _LOG.info("采样输出: %r", tokenizer.decode(ids))

    # ---- 落盘
    save_log(log, str(out_dir / "train_log.json"))
    model.save(str(out_dir / "model.npz"))
    summary = {
        "config": {
            "n_steps": args.steps, "d_model": args.d_model, "n_layer": args.n_layer,
            "n_head": args.n_head, "n_kv_head": args.n_kv_head,
            "block_size": args.block, "batch_size": args.batch, "lr": args.lr,
            "seed": args.seed,
        },
        "n_params": model.n_params,
        "train_loss_first": train_losses[0],
        "train_loss_final": final_train,
        "val_loss_first": val_losses[0],
        "val_loss_final": final_val,
        "val_perplexity_final": perplexity(final_val),
        "entropy_floor_ln_vocab": math.log(vocab),
        "wall_seconds": wall,
        "tokens_per_second": train_tokens.size * args.steps / max(wall, 1e-9),
        "sample_text": tokenizer.decode(ids),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _LOG.info("结果已写入 %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
