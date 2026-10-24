"""优化器 —— AdamW (解耦权重衰减) + 余弦学习率调度 + 梯度裁剪。

世界级出处 (详见 docs/REFERENCES.md):

  * **AdamW** : Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay
    Regularization*. ICLR 2019. arXiv:1711.05101.
    关键差异: **权重衰减不走自适应动量**
    ``θ ← θ − η·λ·θ − η·m̂/(√v̂ + ε)``（而 Adam+L2 会把 λθ 混进一/二阶矩，
    导致自适应学习率被缩放错配 —— 这是 AdamW 在原 Adam 上的实质改进）。

  * **余弦退火** : Loshchilov & Hutter (2017). *SGDR: Stochastic Gradient Descent
    with Warm Restarts*. ICLR 2017.

  * **梯度裁剪** : Pascanu, R. et al. (2013). *On the difficulty of training
    recurrent neural networks*. ICML 2013. 全局范数裁剪。

硬不变量:
  **O1 梯度流自洽**: ``update()`` 返回的更新量方向与 Adam 动量统计一致；
      冻结 moment 时可退化为纯 SGD-with-decay。
  **O2 学习率轨道**: 余弦调度在 t=T 处精确到达 ``min_lr``，且全程单调非增。
  **O3 裁剪有界**: 裁剪后 ‖g‖₂ ≤ max_norm（数值边界），且方向与裁剪前一致。
"""
from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np

_LOGGER = logging.getLogger(__name__)


class AdamW:
    """AdamW 优化器 (Loshchilov & Hutter 2019)。

    Args:
        lr:           基础学习率 η。
        betas:        (β₁, β₂) 一/二阶矩衰减率 (默认 0.9, 0.95 —— 大模型训练惯例)。
        eps:          数值稳定项 ε (默认 1e-8)。
        weight_decay: 解耦权重衰减系数 λ。
        correct_bias: 是否做偏差校正 (原论文 Eq. 的 m̂/v̂)。

    更新式::

        m ← β₁ m + (1−β₁) g
        v ← β₂ v + (1−β₂) g²
        m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)
        θ ← θ·(1 − λ·lr) − lr·[ m̂/(√v̂+ε) ]      ← λθ **不**参与 m/v，
                                                       且作用于**更新前**的 θ
    """

    def __init__(
        self,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        correct_bias: bool = True,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr 必须为正，收到 {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas 必须在 [0, 1) 内，收到 {betas}")
        if eps <= 0:
            raise ValueError(f"eps 必须为正，收到 {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay 必须非负，收到 {weight_decay}")
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.correct_bias = bool(correct_bias)
        self.t: int = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """对一批参数执行一次 AdamW 更新（原地修改 grads 对应的 params）。

        Args:
            params: 参数名 → 数组。
            grads:  参数名 → 梯度数组（形状须逐个匹配）。
        """
        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t
        for name, p in params.items():
            g = grads.get(name)
            if g is None:
                continue
            g = np.asarray(g, dtype=np.float64)
            if g.shape != p.shape:
                raise ValueError(
                    f"梯度形状不匹配 [{name}]: grad {g.shape} vs param {p.shape}"
                )
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)
            m, v = self.m[name], self.v[name]
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            m_hat = m / bias1 if self.correct_bias else m
            v_hat = v / bias2 if self.correct_bias else v
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            # 解耦权重衰减 (decoupled weight decay): 衰减项**不参与 m/v**,
            # 且必须作用于**更新前**的参数 —— 与 torch.optim.AdamW 的
            # `param.mul_(1 - lr*wd)` 语义一致:
            #     p ← p·(1 − λ·lr) − lr·update
            # 若把衰减放在 Adam 步之后, 会得到 (p − lr·update)·(1 − λ·lr),
            # 引入 lr²·λ·update 的系统性偏差 (跨库交叉验证 X6 可捕捉到)。
            if self.weight_decay > 0.0:
                p -= self.lr * self.weight_decay * p
            p -= self.lr * update

    def set_lr(self, lr: float) -> None:
        """重设学习率（配合调度器使用）。"""
        if lr <= 0:
            raise ValueError(f"lr 必须为正，收到 {lr}")
        self.lr = float(lr)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"wd={self.weight_decay}, step={self.t})"
        )


def cosine_schedule(
    step: int,
    total_steps: int,
    *,
    warmup: int = 0,
    base_lr: float = 3e-4,
    min_lr_ratio: float = 0.1,
) -> float:
    """余弦学习率调度（含线性 warmup）。

    Args:
        step:        当前步（0-based）。
        total_steps: 总训练步数。
        warmup:      线性预热步数（前 warmup 步从 ~0 线性升到 base_lr）。
        base_lr:     峰值学习率。
        min_lr_ratio: 末端学习率 = base_lr × min_lr_ratio。

    Returns:
        当前步的学习率。

    参考: Loshchilov & Hutter 2017 SGDR；warmup 惯例来自 Goyal et al. 2017
    (*Accurate, Large Minibatch SGD*, arXiv:1706.02677)。
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps 必须为正，收到 {total_steps}")
    if step < 0:
        raise ValueError(f"step 必须非负，收到 {step}")
    min_lr = base_lr * min_lr_ratio
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def clip_grad_norm(
    grads: dict[str, np.ndarray], max_norm: float
) -> tuple[dict[str, np.ndarray], float]:
    """全局 L2 范数裁剪 (Pascanu et al. 2013)。

    Args:
        grads:    参数名 → 梯度。
        max_norm: 允许的最大全局范数。

    Returns:
        (裁剪后的梯度字典, 裁剪前的全局范数)。

    数学: 若 ‖g‖₂ > max_norm，则 g ← g · max_norm / ‖g‖₂
    （**方向不变、长度受限** —— 这是 RNN/Transformer 训练防爆的标配）。
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm 必须为正，收到 {max_norm}")
    arrays = [np.asarray(g, dtype=np.float64) for g in grads.values()]
    total = float(np.sqrt(sum(float(np.sum(a * a)) for a in arrays))) if arrays else 0.0
    if not np.isfinite(total):
        raise FloatingPointError("梯度范数非有限值（发散），请检查损失与初始化")
    if total <= max_norm or total == 0.0:
        return grads, total
    scale = max_norm / total
    return {k: np.asarray(v, dtype=np.float64) * scale for k, v in grads.items()}, total


def l2_norm_of(grads: Iterable[np.ndarray]) -> float:
    """计算一组梯度拼接后的全局 L2 范数（测试/日志用）。"""
    arrays = [np.asarray(g, dtype=np.float64) for g in grads]
    if not arrays:
        return 0.0
    return float(np.sqrt(sum(float(np.sum(a * a)) for a in arrays)))
