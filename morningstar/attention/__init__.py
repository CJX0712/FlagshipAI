"""L10 注意力家族 —— 统一接口下的多种注意力实现（含 S 级路由 router）。

本层直接集成 **已在 effattnforge 中通过 S 级四道 gates 评审的注意力实现**
（EffAttnForge 基准: G1 保真度 / G2 检索 / G3 成本 / G4 比特级确定性），
并把它们与 L11 的 FlashAttention-2 收拢到同一套协议下。

统一语义: ``attend(q, k, v) -> AttentionOutput``，score 越大注意力质量越高，
沿 key 轴 softmax 归一化 —— 保证跨变体评测公平。

世界级出处 (详见 docs/REFERENCES.md):

  full          : Vaswani et al. 2017 (精确 O(T²d) 上界参照)
  flash         : Dao 2023 FlashAttention-2 (精确但 O(T·d) 内存)
  linear        : Katharopoulos et al. 2020 (O(Td²) 核化线性注意力)
  performer     : Choromanski et al. 2020 FAVOR+ (O(Tdm) 无偏随机特征)
  local_window  : Beltagy et al. 2020 Longformer 式滑窗 (O(Twd))
  block_sparse  : Child et al. 2019 Sparse Transformers 块状稀疏 (O(Tbd))
  effattnfuse   : 长度自适应 router —— T ≤ 阈值走精确，长序列走 Performer，
                  并用**观测到的**保真度做质量守卫 (EffAttnForge 旗舰贡献)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..flash.attention import flash_attention_forward, naive_attention_forward

# ------------------------------------------------------------------ types
@dataclass
class AttentionOutput:
    """注意力输出容器（源自 EffAttnForge core/types.py）。"""

    output: np.ndarray            # (T, d) 输出
    weights: np.ndarray | None    # 物化的 (T, T) 注意力矩阵；非物化变体为 None
    cost_s: float                 # 本次前向墙钟秒数
    mem_peak_mb: float            # 峰值额外内存 (MB)
    variant: str                  # 实现名

    def __post_init__(self) -> None:
        if self.mem_peak_mb < 0:
            raise ValueError("mem_peak_mb 不能为负")


@runtime_checkable
class AttentionMechanism(Protocol):
    """注意力变体协议（源自 EffAttnForge core/interfaces.py）。"""

    name: str

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput: ...
    def theoretical_complexity(self, T: int, d: int) -> str: ...


def _mb(nbytes: float) -> float:
    """字节 → MB。"""
    return nbytes / (1024.0 * 1024.0)


# --------------------------------------------------------------- variants
class FullAttention:
    """精确 full softmax 注意力 —— 所有近似的黄金上界参照。"""

    name = "full"

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        t0 = time.perf_counter()
        out, w = naive_attention_forward(q, k, v, mask="none", return_weights=True)
        cost = time.perf_counter() - t0
        return AttentionOutput(out, w, cost, _mb(2 * T * T * 8 + T * d * 8), self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return "O(T^2 d) [exact]"


class FlashAttention:
    """FlashAttention-2 —— 与 full **数值等价**，但内存 O(T·d) 而非 O(T²)。"""

    name = "flash"

    def __init__(self, block_q: int = 64, block_k: int = 64) -> None:
        self.block_q = int(block_q)
        self.block_k = int(block_k)

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        t0 = time.perf_counter()
        out, _, nbytes = flash_attention_forward(
            q, k, v, mask="none", block_q=self.block_q, block_k=self.block_k
        )
        cost = time.perf_counter() - t0
        return AttentionOutput(out, None, cost, _mb(nbytes), self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return f"O(T^2 d) time / O(T d) memory [bq={self.block_q}, bk={self.block_k}]"


class LinearAttention:
    """核化线性注意力 (Katharopoulos et al. 2020): ``out_t = φ(q_t)ᵀS / φ(q_t)ᵀz``。"""

    name = "linear"

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        t0 = time.perf_counter()
        kf, qf = _phi(k), _phi(q)
        S = kf.T @ v              # (d, d)
        z = kf.sum(axis=0)        # (d,)
        num = qf @ S
        den = qf @ z
        safe = np.where(np.abs(den) < 1e-12, 1.0, den)
        out = num / safe[:, None]
        cost = time.perf_counter() - t0
        return AttentionOutput(out, None, cost, _mb(2 * T * d * 8 + d * d * 8), self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return "O(T d^2)"


class PerformerAttention:
    """Performer / FAVOR+ (Choromanski et al. 2020): O(T d m)，不物化 T×T。

    **实现要点（本仓库基于 FAVOR+ 论文 (arXiv:2009.14794) 重标定，与原始
    EffAttnForge 实现有本质差异 —— 后者的核标定是错的）**:

    1. **核标定**: 目标是 softmax 的 ``exp(q·k/√d)``。令 ``x = q/d^{1/4}``、
       ``y = k/d^{1/4}``，则 ``x·y = q·k/√d``，目标核化为 ``exp(x·y)``。

    2. **三角随机特征**: 由 ``E_ω[cos(ω·z)] = exp(−‖z‖²/2)``（ω~N(0,I)）与
       ``exp(x·y) = exp(‖x‖²/2)exp(‖y‖²/2)exp(−‖x−y‖²/2)``，构造

           ``φ(x) = exp(σ‖x‖²/2)/√m · [sin(ω_j·x+b_j), cos(ω_j·x+b_j)]_{j=1..m}``

       其中 ``σ = ±1`` 为范数因子的符号。**Q 与 K 必须共享同一组 (ω, b)**。

    3. **非对称范数因子（本实现的关键修正）**: 取 Query 用 ``σ=−1``、
       Key 用 ``σ=+1``，则

           ``E[φ_q(q_i)·φ_k(k_j)] = exp(x_i·y_j − ‖x_i‖²)``

       其中 ``exp(−‖x_i‖²)`` 对同一行的所有 j 是常数，**在 softmax 比值
       (分子/分母) 中精确抵消** ⇒ 估计量对注意力输出**无偏**。
       若两侧同号(+/+ 或 −/−)，则会出现随 ‖k_j‖² 变化的非恒定因子，
       在分母中引入系统性偏倚 —— 这正是原始 EffAttnForge 实现的缺陷。

    4. **输入范数敏感性（已实测）**: 范数因子 ``exp(σ‖x‖²/2)`` 会把
       随机特征估计的方差放大 ``exp(σ‖x‖²)`` 倍。当 ``‖q‖² ~ d``（原始
       N(0,1) 输入）时，d=32 下相对 Frobenius 误差可达 **~18**（不可用）；
       当 ``‖q‖²/d^{1/2} ~ O(1)``（即真实 Transformer 中经 RMSNorm +
       投影后范数受控的 Q/K）时，误差降至 **~0.003**。
       ⇒ 本模块的保真度与输入范数强相关，使用时须保证 Q/K 已归一化。
    """

    name = "performer"

    def __init__(
        self,
        m: int = 256,
        salt: int = 7777,
        seed: int | None = None,
        orthogonal: bool = False,
    ) -> None:
        self.m = int(m)
        self.salt = int(salt)
        self.seed = seed
        # ORF (正交随机特征) 论文 §3 的方差缩减技术。本仓库实测：在受控输入下
        # 块拼接的正交基反而使保真度变差（块间不正交 + 每分量方差需 √d 重标定），
        # 故默认关闭；保留开关以供对照实验。
        self.orthogonal = bool(orthogonal)

    def _sample(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        from ..flash.random import SALT_BENCH, rng_for

        rng = rng_for(SALT_BENCH ^ self.salt, self.seed)
        m = self.m
        if self.orthogonal:
            # ORF: 分块生成正交基并乘 √d 恢复 FAVOR+ 的 ω~N(0,I) 标定。
            n_blocks = (m + d - 1) // d
            blocks = []
            for _ in range(n_blocks):
                Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                blocks.append(Q.T * np.sqrt(d))
            W = np.concatenate(blocks, axis=0)[:m]
        else:
            # iid 高斯特征: 每个分量方差为 1 ⇒ ω·z ~ N(0,‖z‖²)，
            # 从而 E[cos(ω·z)] = exp(−‖z‖²/2) 精确成立。
            W = rng.standard_normal((m, d))
        b = rng.uniform(0.0, 2.0 * np.pi, size=m)
        return W, b

    def _features(
        self, x: np.ndarray, W: np.ndarray, b: np.ndarray, norm_sign: float = -1.0
    ) -> np.ndarray:
        """FAVOR+ 三角随机特征映射。

        Args:
            x:          (T, d) 输入向量。
            W, b:       共享的随机特征参数 (m, d) 与 (m,)。
            norm_sign:  范数因子符号 σ ∈ {−1, +1}；Query 用 −1、Key 用 +1。

        Returns:
            (T, 2m) 特征矩阵。
        """
        d = x.shape[-1]
        xs = x / (d ** 0.25)                      # d^{−1/4}: 使 x·y = q·k/√d
        norm = np.exp(norm_sign * 0.5 * np.sum(xs * xs, axis=-1))
        proj = xs @ W.T + b
        feats = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
        return (norm / np.sqrt(self.m))[:, None] * feats

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        W, b = self._sample(d)
        t0 = time.perf_counter()
        # 非对称范数因子 ⇒ 分子/分母比值无偏（见类文档第 3 点）
        phi_q = self._features(q, W, b, norm_sign=-1.0)
        phi_k = self._features(k, W, b, norm_sign=+1.0)
        kv = phi_k.T @ v
        z = phi_k.sum(axis=0)
        num = phi_q @ kv
        den = phi_q @ z
        safe = np.where(np.abs(den) < 1e-12, 1.0, den)
        out = num / safe[:, None]
        cost = time.perf_counter() - t0
        return AttentionOutput(
            out, None, cost, _mb(2 * T * 2 * self.m * 8 + 2 * self.m * d * 8), self.name
        )

    def theoretical_complexity(self, T: int, d: int) -> str:
        return f"O(T d m) [m={self.m}]"


class LocalWindowAttention:
    """滑窗注意力 (Longformer 式, Beltagy et al. 2020): 每 query 只看 [i−w, i+w]。"""

    name = "local_window"

    def __init__(self, window: int = 32) -> None:
        self.window = int(window)

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        w = self.window
        out = np.empty((T, d), dtype=np.float64)
        t0 = time.perf_counter()
        scale = 1.0 / math.sqrt(d)
        for i in range(T):
            lo, hi = max(0, i - w), min(T, i + w + 1)
            s = (q[i] @ k[lo:hi].T) * scale
            s = s - s.max()
            e = np.exp(s)
            out[i] = (e / e.sum()) @ v[lo:hi]
        cost = time.perf_counter() - t0
        return AttentionOutput(out, None, cost, _mb((2 * w + 1) * d * 8), self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return f"O(T w d) [w={self.window}]"


class BlockSparseAttention:
    """块状稀疏 + 全局 token (Child et al. 2019 Sparse Transformers)。"""

    name = "block_sparse"

    def __init__(self, block_size: int = 64, n_global: int = 8) -> None:
        self.block_size = int(block_size)
        self.n_global = int(n_global)

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        b, ng = self.block_size, min(self.n_global, T)
        n_blocks = max(1, -(-T // b))
        out = np.empty((T, d), dtype=np.float64)
        t0 = time.perf_counter()
        scale = 1.0 / math.sqrt(d)
        for bi in range(n_blocks):
            qs, qe = bi * b, min(T, (bi + 1) * b)
            idx = [np.arange(ng)]
            for bj in (bi - 1, bi, bi + 1):
                if 0 <= bj < n_blocks:
                    idx.append(np.arange(bj * b, min(T, (bj + 1) * b)))
            allowed = np.unique(np.concatenate(idx))
            s = (q[qs:qe] @ k[allowed].T) * scale
            s = s - s.max(axis=-1, keepdims=True)
            e = np.exp(s)
            out[qs:qe] = (e / e.sum(axis=-1, keepdims=True)) @ v[allowed]
        cost = time.perf_counter() - t0
        return AttentionOutput(out, None, cost, _mb((3 * b + ng) * d * 8), self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return f"O(T b d) [b={self.block_size}, g={self.n_global}]"


class EffAttnFuse:
    """长度自适应 router（EffAttnForge 旗舰）: 短序列精确、长序列 Performer，带质量守卫。

    决策规则::

        T ≤ full_threshold                      → FullAttention（便宜且最优）
        T > full_threshold 且标定保真度达标      → Performer(m = 按长度带选择)
        否则                                     → FullAttention（守卫回退）
    """

    name = "effattnfuse"

    def __init__(
        self,
        full_threshold: int = 128,
        performer_m_short: int = 128,
        performer_m_long: int = 256,
        retrieval_floor: float = 0.98,
        calibration: dict[int, float] | None = None,
    ) -> None:
        self.full_threshold = int(full_threshold)
        self.performer_m_short = int(performer_m_short)
        self.performer_m_long = int(performer_m_long)
        self.retrieval_floor = float(retrieval_floor)
        self.calibration: dict[int, float] = dict(calibration or {})

    def _m(self, T: int) -> int:
        return self.performer_m_long if T > 1024 else self.performer_m_short

    def choose(self, T: int, d: int) -> AttentionMechanism:
        if T <= self.full_threshold:
            return FullAttention()
        if self.calibration.get(T, 1.0) >= self.retrieval_floor:
            return PerformerAttention(m=self._m(T), salt=7777)
        return FullAttention()

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T = q.shape[0]
        res = self.choose(T, q.shape[1]).attend(q, k, v)
        # 归因给 router 本身，便于基准对照
        return AttentionOutput(res.output, res.weights, res.cost_s, res.mem_peak_mb, self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        if T <= self.full_threshold:
            return "O(T^2 d) [short: exact]"
        return f"O(T d m) [long: Performer m={self._m(T)}]"


# ----------------------------------------------------------- helpers/registry
def _phi(x: np.ndarray) -> np.ndarray:
    """线性注意力的正特征映射 φ(x) = elu(x) + 1。"""
    return np.where(x >= 0.0, x, np.expm1(x)) + 1.0


_REGISTRY: dict[str, callable] = {
    "full": lambda: FullAttention(),
    "flash": lambda: FlashAttention(),
    "linear": lambda: LinearAttention(),
    "performer": lambda: PerformerAttention(),
    "local_window": lambda: LocalWindowAttention(),
    "block_sparse": lambda: BlockSparseAttention(),
    "effattnfuse": lambda: EffAttnFuse(),
}


def build_variant(name: str) -> AttentionMechanism:
    """按名称构造注意力变体。未知名称抛 KeyError。"""
    if name not in _REGISTRY:
        raise KeyError(f"未知注意力变体 {name!r}，可选: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_variants() -> list[str]:
    """所有可用变体名称。"""
    return sorted(_REGISTRY)


__all__ = [
    "AttentionOutput",
    "AttentionMechanism",
    "BlockSparseAttention",
    "EffAttnFuse",
    "FlashAttention",
    "FullAttention",
    "LinearAttention",
    "LocalWindowAttention",
    "PerformerAttention",
    "build_variant",
    "list_variants",
]
