"""L11 训练流水线 —— AdamW + 余弦调度 + 梯度裁剪 + 评估 + 早停。

设计（零重写）: 训练循环的组织方式对齐业界标准实践
  (Howard & Ruder 2018 ULMFiT 的渐进解冻思想之外，采用现代 LLM 预训练标配:
   AdamW 优化器 + 余弦 LR + warmup + grad-clip + grad-accum)，
  优化器算法本身出自 Loshchilov & Hutter 2019，见 flash/optim.py。

硬不变量:
  **P1 种子闭环**: 相同 seed ⇒ 逐位相同的初始化、数据顺序、最终损失
      （全链路只从 flash.random 取随机数）。
  **P2 训练损失下降**: 在合成可学习任务上，末期损失显著低于初期。
  **P3 梯度有界**: 每步裁剪后的全局梯度范数 ≤ clip_max。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .model import GPT
from .optim import AdamW, clip_grad_norm, cosine_schedule, l2_norm_of

_LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """训练超参。"""

    n_steps: int = 500
    batch_size: int = 8
    block_size: int = 64
    lr: float = 3e-3
    warmup: int = 20
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_iters: int = 5
    log_every: int = 10
    seed: int = 1337


@dataclass
class TrainLog:
    """训练过程记录（可直接 JSON 序列化）。"""

    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "lr": self.lr,
            "grad_norm": self.grad_norm,
            "wall_s": self.wall_s,
        }


def make_batches(
    tokens: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """随机采样一批 (x, y) 训练片段。

    Args:
        tokens:     一维 token id 流。
        block_size: 上下文长度。
        batch_size: 批大小。
        rng:        确定性随机源（数据顺序的可复现性由它保证）。

    Returns:
        (x (B, T), y (B, T))，其中 y 是 x 左移一位的语言模型目标。
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.size <= block_size + 1:
        raise ValueError(
            f"语料太短: 需要 > block_size+1={block_size + 1} 个 token，收到 {tokens.size}"
        )
    hi = tokens.size - block_size - 1
    starts = rng.integers(0, hi, size=batch_size)
    x = np.stack([tokens[s : s + block_size] for s in starts])
    y = np.stack([tokens[s + 1 : s + 1 + block_size] for s in starts])
    return x, y


def evaluate(
    model: GPT, tokens: np.ndarray, cfg: TrainConfig, rng: np.random.Generator
) -> float:
    """在若干随机片段上估计平均损失（无梯度）。"""
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = make_batches(tokens, cfg.block_size, cfg.batch_size, rng)
        total = 0.0
        for b in range(x.shape[0]):
            logits, cache = model.forward(x[b], y[b])
            total += float(cache["loss"])
        losses.append(total / x.shape[0])
    return float(np.mean(losses))


def train(
    model: GPT,
    train_tokens: np.ndarray,
    cfg: TrainConfig,
    val_tokens: np.ndarray | None = None,
    callback: Callable[[int, float, float], None] | None = None,
) -> TrainLog:
    """完整训练循环。

    Args:
        model:        待训练模型。
        train_tokens: 训练语料 token 流。
        cfg:          训练超参。
        val_tokens:   可选验证语料。
        callback:     每步回调 ``(step, loss, lr)``。

    Returns:
        TrainLog 记录。
    """
    from .random import SALT_DATA, set_seed

    set_seed(cfg.seed)
    data_rng = np.random.default_rng(cfg.seed ^ (SALT_DATA * 0x9E3779B1) & 0xFFFFFFFF)
    val_rng = np.random.default_rng(cfg.seed ^ ((SALT_DATA + 1) * 0x9E3779B1) & 0xFFFFFFFF)

    optimizer = AdamW(
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay
    )
    log = TrainLog()
    t_start = time.perf_counter()
    running = []

    val = evaluate(model, val_tokens, cfg, val_rng) if val_tokens is not None else float("nan")
    last_val = val
    if cfg.eval_every > 0:
        log.steps.append(0)
        log.train_loss.append(float("nan"))
        log.val_loss.append(val)
        log.lr.append(cfg.lr)
        log.grad_norm.append(0.0)

    for step in range(1, cfg.n_steps + 1):
        lr_now = cosine_schedule(
            step - 1, cfg.n_steps, warmup=cfg.warmup,
            base_lr=cfg.lr, min_lr_ratio=cfg.min_lr_ratio,
        )
        optimizer.set_lr(lr_now)

        x, y = make_batches(train_tokens, cfg.block_size, cfg.batch_size, data_rng)
        grads_acc: dict[str, np.ndarray] | None = None
        loss_acc = 0.0
        for b in range(x.shape[0]):
            loss, grads = model.loss_and_grad(x[b], y[b])
            loss_acc += loss
            if grads_acc is None:
                grads_acc = grads
            else:
                for k in grads_acc:
                    grads_acc[k] += grads[k]
        assert grads_acc is not None
        for k in grads_acc:
            grads_acc[k] /= x.shape[0]
        loss_avg = loss_acc / x.shape[0]

        grads_acc, gnorm = clip_grad_norm(grads_acc, cfg.grad_clip)
        optimizer.step(model.params, grads_acc)

        running.append(loss_avg)
        # 评估与日志是两个独立节拍: 无论是否落在 log 节拍上, 到评估点就评估,
        # 并把最近一次验证损失带入日志 (避免出现整段 NaN 掩盖真实曲线)。
        if val_tokens is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            last_val = evaluate(model, val_tokens, cfg, val_rng)
        if step % cfg.log_every == 0 or step == 1:
            recent = float(np.mean(running[-cfg.log_every :]))
            log.steps.append(step)
            log.train_loss.append(recent)
            log.val_loss.append(last_val)
            log.lr.append(lr_now)
            log.grad_norm.append(gnorm)
            _LOGGER.info(
                "step %4d/%d  loss %.4f  val %.4f  lr %.2e  |g| %.3f",
                step, cfg.n_steps, recent, last_val, lr_now, gnorm,
            )
        if callback is not None:
            callback(step, float(loss_avg), lr_now)

    log.wall_s = time.perf_counter() - t_start
    return log


def save_log(log: TrainLog, path: str) -> None:
    """把训练日志落盘 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log.to_dict(), f, ensure_ascii=False, indent=2)


def perplexity(loss: float) -> float:
    """交叉熵 → 困惑度: ``ppl = exp(loss)``。"""
    return float(np.exp(loss))
