"""现代 Transformer 组件 —— RoPE / RMSNorm / SwiGLU。

每一组件都对应一篇世界级论文 (详见 docs/REFERENCES.md):

  * **RoPE 旋转位置编码** : Su, J. et al. (2021). *RoFormer: Enhanced Transformer
    with Rotary Position Embedding*. arXiv:2104.09864.
    用复数旋转把绝对位置注入 Q/K:  ⟨R_m q, R_n k⟩ 只依赖相对位移 m−n，
    因此**天然外推到更长上下文**——这是 Llama / Mistral / Qwen 的默认选择。

  * **RMSNorm** : Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer
    Normalization*. arXiv:1910.07467.
    只做重缩放、不做重居中: ``y = x / RMS(x) · g``，比 LayerNorm 少一次
    均值统计，训练更稳且更快。

  * **SwiGLU FFN** : Shazeer, N. (2020). *GLU Variants Improve Transformer*.
    arXiv:2002.05202.  ``FFN(x) = (Swish(xW_gate) ⊙ xW_up) W_down``
    —— Llama / PaLM 的标配，同参数量下优于 ReLU/GELU FFN。

硬不变量:
  **R1 RoPE 相对性**:  ⟨Rot(q, m), Rot(k, n)⟩ 只依赖 (m − n)
                       ⇒ 整体平移位置 (m+Δ, n+Δ) 后内积不变。
  **R2 RoPE 保范**:     ‖Rot(x, m)‖₂ = ‖x‖₂（旋转是正交变换）。
  **R3 RMSNorm 单位 RMS**: 归一化后若 g = 1，则 RMS(y) = 1 (逐行)。
  **R4 SwiGLU 前向有限**: 极端输入不产生 NaN/Inf。
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)


class RotaryEmbedding:
    """RoPE 旋转位置编码 (Su et al. 2021)。

    实现选择（零重写）: 直接采用 Llama/Torch 生态通行的 **半旋转配对** 形式:
    把 head_dim 维向量视作 d/2 个复数，第 j 个复数的旋转角为 ``m · θ_j``，
    其中 ``θ_j = 1 / base^{2j/d}``（base = 10000）。

    数学式: 对位置 m，
        ``Rot(x, m)_{2j}   = x_{2j} cos(mθ_j) − x_{2j+1} sin(mθ_j)``
        ``Rot(x, m)_{2j+1} = x_{2j} sin(mθ_j) + x_{2j+1} cos(mθ_j)``

    为什么它保留相对位置信息: 旋转矩阵满足
    ``⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩``，内积仅依赖相对位移。
    """

    def __init__(self, head_dim: int, base: float = 10000.0, max_len: int = 8192) -> None:
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim 必须为正偶数，收到 {head_dim}")
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.max_len = int(max_len)
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.head_dim, 2, dtype=np.float64)
                                        / self.head_dim))
        self.inv_freq = inv_freq  # (d/2,)
        self._cache_len = 0
        self._cos: np.ndarray = np.zeros((0, self.head_dim // 2))
        self._sin: np.ndarray = np.zeros((0, self.head_dim // 2))

    def _extend_cache(self, need: int) -> None:
        """按需扩展 cos/sin 表（避免一次性分配 max_len 的浪费）。"""
        if need <= self._cache_len:
            return
        positions = np.arange(0, need, dtype=np.float64)
        if need > self.max_len:
            raise ValueError(f"需要 {need} 个位置，超过 max_len={self.max_len}")
        angles = positions[:, None] * self.inv_freq[None, :]  # (need, d/2)
        self._cos = np.cos(angles)
        self._sin = np.sin(angles)
        self._cache_len = need

    def rotate_qk(
        self, x: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """对形状 (..., head_dim) 的张量施加旋转。

        Args:
            x:         任意前缀形状 + 最后一维为 head_dim。
            positions: 与 x 前导维度广播兼容的位置索引。

        Returns:
            旋转后的张量，形状同 x。
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"最后一维必须为 head_dim={self.head_dim}，收到 {x.shape[-1]}")
        positions = np.asarray(positions, dtype=np.int64)
        need = int(positions.max()) + 1 if positions.size else 0
        self._extend_cache(max(need, 1))

        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            # 单向量: 位置取标量，cos/sin 为 (d/2,)
            p = int(positions.reshape(-1)[0])
            cos, sin = self._cos[p], self._sin[p]
        else:
            # 若 positions 维度少于前导维，先在尾部补轴再广播，
            # 使 (T,) 能对齐 (T, H) 这类「序列 + 多头」布局。
            pos = np.asarray(positions, dtype=np.int64)
            ndim_lead = len(x.shape[:-1])
            if pos.ndim < ndim_lead:
                pos = pos.reshape(pos.shape + (1,) * (ndim_lead - pos.ndim))
            pos = np.broadcast_to(pos, x.shape[:-1])
            cos, sin = self._cos[pos], self._sin[pos]
        x1 = x[..., 0::2]  # 偶数位 (实部)
        x2 = x[..., 1::2]  # 奇数位 (虚部)
        out = np.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def rotate_qk_inverse(
        self, x: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """逆向旋转（反向传播用）。

        数学依据: RoPE 是**正交变换** ``R(m)``，其逆 ``R(m)^{-1} = R(-m) = R(m)^T``。
        旋转矩阵 ``R(m) = [[cos, -sin], [sin, cos]]``，故其转置为
        ``R(-m) = [[cos, +sin], [-sin, cos]]`` —— 即**复用正位置**的 cos/sin，
        但把 sin 的符号翻转即可，无需对位置取负（负索引在 numpy 会循环回绕，错误）。

        验证: ``rotate_qk_inverse(rotate_qk(x, m), m) == x`` (逐位)。
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"最后一维必须为 head_dim={self.head_dim}，收到 {x.shape[-1]}")
        positions = np.asarray(positions, dtype=np.int64)
        need = int(positions.max()) + 1 if positions.size else 0
        self._extend_cache(max(need, 1))

        if x.ndim == 1:
            p = int(positions.reshape(-1)[0])
            cos, sin = self._cos[p], self._sin[p]
        else:
            pos = np.asarray(positions, dtype=np.int64)
            ndim_lead = len(x.shape[:-1])
            if pos.ndim < ndim_lead:
                pos = pos.reshape(pos.shape + (1,) * (ndim_lead - pos.ndim))
            pos = np.broadcast_to(pos, x.shape[:-1])
            cos, sin = self._cos[pos], self._sin[pos]

        x1 = x[..., 0::2]  # 偶数位 (实部)
        x2 = x[..., 1::2]  # 奇数位 (虚部)
        out = np.empty_like(x)
        # R(-m) 应用: 翻转 sin 符号
        out[..., 0::2] = x1 * cos + x2 * sin
        out[..., 1::2] = -x1 * sin + x2 * cos
        return out

    def rotate(
        self, q: np.ndarray, k: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """同时旋转 q (T, H, dh) 与 k (T, H_kv, dh)。供 multihead_forward 调用。"""
        q_pos = positions
        k_pos = positions
        return self.rotate_qk(q, q_pos), self.rotate_qk(k, k_pos)


def rotate_half_numpy(x: np.ndarray, positions: np.ndarray, head_dim: int,
                      base: float = 10000.0) -> np.ndarray:
    """函数式 RoPE（便于测试与独立交叉验证 RotaryEmbedding 类）。

    这是 **不依赖对象状态** 的第二实现，用于交叉验证 RoPE 的正确性
    （两条独立路径应给出逐位一致的结果）。
    """
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    angles = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(np.asarray(x, dtype=np.float64))
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class RMSNorm:
    """RMSNorm (Zhang & Sennrich 2019): ``y = x / sqrt(mean(x²) + eps) · g``。

    与 LayerNorm 的区别: 不减去均值、不学习偏置，只做**重缩放**。
    在 Transformer 中更稳定且计算更省（少一次均值归约）。
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        if dim <= 0:
            raise ValueError(f"dim 必须为正，收到 {dim}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = np.ones(self.dim, dtype=np.float64)  # 可学习增益 g

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向: 逐行（最后一维）归一化后乘增益。"""
        x = np.asarray(x, dtype=np.float64)
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class SwiGLUFeedForward:
    """SwiGLU 前馈网络 (Shazeer 2020): ``(Swish(xW_gate) ⊙ xW_up) W_down``。

    其中 Swish(z) = z · σ(z)（亦称 SiLU）。Llama/PaLM/Qwen 均采用。
    注意显式的 ~2/3 hidden_dim 惯例（heuristic），保证与 ReLU FFN **参数量对齐**。
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, *, multiple: int = 256) -> None:
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim else _ffn_hidden(dim, multiple)

    def forward(
        self,
        x: np.ndarray,
        w_gate: np.ndarray,
        w_up: np.ndarray,
        w_down: np.ndarray,
    ) -> np.ndarray:
        """前向。权重形状: w_gate/w_up (dim, hidden), w_down (hidden, dim)。"""
        x = np.asarray(x, dtype=np.float64)
        gate = x @ np.asarray(w_gate, dtype=np.float64)
        up = x @ np.asarray(w_up, dtype=np.float64)
        act = gate * _sigmoid(gate)  # Swish / SiLU
        return (act * up) @ np.asarray(w_down, dtype=np.float64)

    def __call__(self, x, w_gate, w_up, w_down):
        return self.forward(x, w_gate, w_up, w_down)


def _ffn_hidden(dim: int, multiple: int = 256) -> int:
    """Llama 惯例: hidden = ⌈(2/3 · 4·dim) / multiple⌉ · multiple。"""
    raw = int((2.0 / 3.0) * (4 * dim))
    return ((raw + multiple - 1) // multiple) * multiple


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid（正负分支分离，避免大正数溢出）。"""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
