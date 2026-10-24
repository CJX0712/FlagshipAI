"""注意力内核 —— FlashAttention-2 分块在线 softmax + 因果掩码 + KV 缓存 + GQA。

世界级出处 (详见 docs/REFERENCES.md):

  * **FlashAttention-2** : Dao, T. (2023). *FlashAttention-2: Faster Attention with
    Better Parallelism and Work Partitioning*. arXiv:2307.08691.
    核心思想: **IO 感知的分块** + **在线 softmax 重标定**，
    全程不把 T×T 的注意力矩阵写回内存。
  * **在线 softmax** : Milakov, M. & Gimelshein, N. (2018). *Online normalizer
    calculation for softmax*. arXiv:1805.02867. 递推式
    ``m_new = max(m_old, rowmax(s)); l_new = e^{m_old-m_new} l_old + rowsum(e^{s-m_new});
    O_new = e^{m_old-m_new} O_old + e^{s-m_new} V``。
  * **缩放点积注意力** : Vaswani et al. (2017). *Attention Is All You Need*.
    NeurIPS 30. —— 本模块的 ``naive_attention_forward`` 即其**黄金参照实现**。
  * **GQA (分组查询注意力)** : Ainslie et al. (2023). *GQA: Training Generalized
    Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP 2023.

硬不变量 (verified invariants，由 ``morningstar.verify.invariants`` 执行):

  **I1 数值等价**: FlashAttention-2 前向与 naive softmax attention **逐位接近**
      ``‖O_flash − O_naive‖_F / ‖O_naive‖_F < 1e-12``
      —— 因为两者是同一个数学表达式的不同求值顺序。
  **I2 行随机性**: 注意力权重 P 每行和为 1: ``max|P·1 − 1| < 1e-12``。
  **I3 因果性**: causal 时 ``P[i, j] = 0 for j > i``（对角线上三角严格为 0）。
  **I4 平移不变性**: score 整体加常数 ⇒ 输出不变（数值稳定的核心性质）。
  **I5 凸组合有界**: 输出行 = V 行的凸组合 ⇒ 落在 V 行张成的凸包内。
  **I6 增量一致性**: KV 缓存逐 token 解码 == 全序列一次性前向
      ``‖O_incremental − O_full‖_F / ‖O_full‖_F < 1e-10``。
  **I7 内存次二次**: flash 路径的额外内存为 O(T·d)，与 T 线性，
      而 naive 为 O(T²) —— 这是 FlashAttention 的工程价值所在（可被测出来）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

_LOGGER = logging.getLogger(__name__)

MaskMode = Literal["none", "causal"]

ArrayLike = np.ndarray


@dataclass
class AttentionOutput:
    """一次注意力前向的结果容器。

    Attributes:
        output:     输出张量，形状 (H, T, Dh) 单样本 / (B, H, T, Dh) 批量。
        weights:    物化的注意力矩阵；flash 路径为 ``None``（这正是省内存的地方）。
        cost_s:     本次前向的墙钟秒数。
        mem_bytes:  本次前向额外分配的字节数（**解析式记账**，非采样估计）。
        variant:    产生该结果的实现名，用于基准归因。
        extra:      实现相关的附加信息（如 logsumexp），供反向/增量使用。
    """

    output: np.ndarray
    weights: np.ndarray | None
    cost_s: float
    mem_bytes: int
    variant: str
    extra: dict = field(default_factory=dict)


def _to_f64(x: ArrayLike, name: str) -> np.ndarray:
    """统一转为 float64 连续数组（数值夷平 + 可复现）。"""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} 必须是至少一维的张量，收到标量")
    return np.ascontiguousarray(arr)


def _causal_allowed(q_start: int, q_len: int, k_start: int, k_len: int) -> bool:
    """查询块 [qs,qs+ql) 与键块 [ks,ks+kl) 在因果掩码下是否有交集。"""
    return k_start < q_start + q_len


def naive_attention_forward(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    *,
    mask: MaskMode = "causal",
    return_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """**黄金参照**: 物化完整注意力矩阵的 softmax 注意力 (Vaswani et al. 2017)。

    ``Attention(Q,K,V) = softmax(QKᵀ / √d_k ⊙ M) V``

    Args:
        q, k, v: 形状 (T, d) 的查询/键/值。
        mask:    ``"causal"`` 施加下三角因果掩码；``"none"`` 为双向注意力。
        return_weights: 是否返回注意力权重矩阵 P (用于不变量检验)。

    Returns:
        (输出 (T, d), 权重 (T, T) 或 None)。

    Note:
        数值稳定: 先减去行最大值再做 exp (logsumexp 技巧)，避免溢出。
        这是测试 FlashAttention 数值等价性的基准。
    """
    q = _to_f64(q, "q")
    k = _to_f64(k, "k")
    v = _to_f64(v, "v")
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"期望 (T, d) 二维输入，收到 {q.shape}/{k.shape}/{v.shape}")
    tq, dq = q.shape
    tk, dk = k.shape
    if dq != dk:
        raise ValueError(f"q 与 k 的 head_dim 必须一致: {dq} != {dk}")
    if v.shape[0] != tk:
        raise ValueError(f"k 与 v 的序列长度必须一致: {tk} != {v.shape[0]}")

    scale = 1.0 / math.sqrt(dk)
    scores = (q @ k.T) * scale  # (Tq, Tk)
    if mask == "causal":
        idx_q = np.arange(tq)[:, None]
        idx_k = np.arange(tk)[None, :]
        # (seqlen 不等时，用 kv 位置偏移对齐；标准自回归场景 tq == tk)
        offset = tk - tq
        allowed = (idx_k + 0) <= (idx_q + offset)
        scores = np.where(allowed, scores, -np.inf)
    elif mask != "none":
        raise ValueError(f"未知 mask 模式: {mask!r}")

    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    # 掩码必须在「减去行最大值之后」重新施加 −inf：
    # 否则 −inf 会先被 max-shift 变成 NaN/0，破坏因果性（这是极易踩的实现坑）。
    if mask == "causal":
        idx_q = np.arange(tq)[:, None]
        idx_k = np.arange(tk)[None, :]
        allowed = idx_k <= (idx_q + offset)
        shifted = np.where(allowed, shifted, -np.inf)
    exp_scores = np.exp(shifted)  # exp(−inf) = 0，掩码处权重严格为 0
    denom = exp_scores.sum(axis=-1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    weights = exp_scores / denom
    out = weights @ v
    return out, (weights if return_weights else None)


def flash_attention_forward(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    *,
    mask: MaskMode = "causal",
    block_q: int = 64,
    block_k: int = 64,
    return_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """**FlashAttention-2**: IO 感知分块 + 在线 softmax，不物化 T×T 矩阵。

    Args:
        q, k, v:    形状 (T, d)。
        mask:       ``"causal"`` / ``"none"``。因果模式下 **跳过完全被掩码的
                    key 块**，这是 FlashAttention-2 相对 -1 的关键提速点之一。
        block_q:    查询块大小 Bq。
        block_k:    键块大小 Bk。
        return_weights: 若 True，退化回车物化矩阵（仅供调试，破坏省内存收益）。

    Returns:
        (输出 (T, d), 权重或 None, 额外分配字节数)。

    算法（在线 softmax 递推，Milakov & Gimelshein 2018）::

        for each query block i:
            m_i = -inf ; l_i = 0 ; O_i = 0
            for each key block j (allowed by mask):
                S_ij = scale * Q_i @ K_jᵀ
                m_new = maximum(m_i, rowmax(S_ij))
                P̃_ij = exp(S_ij − m_new)
                l_i   = exp(m_i − m_new) * l_i + rowsum(P̃_ij)
                O_i   = exp(m_i − m_new) * O_i + P̃_ij @ V_j
                m_i   = m_new
            O_i = O_i / l_i

    为什么它是「顶级」的: 峰值额外内存 O(T·d + Bq·Bk) 而非 O(T²)，
    使长上下文从「内存墙」变为「算力受限」；而结果与原式**数值等价**（同一数学式）。
    """
    q = _to_f64(q, "q")
    k = _to_f64(k, "k")
    v = _to_f64(v, "v")
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"期望 (T, d) 二维输入，收到 {q.shape}/{k.shape}/{v.shape}")
    tq, dq = q.shape
    tk = k.shape[0]
    dv = v.shape[1]
    if dq != k.shape[1]:
        raise ValueError(f"q/k head_dim 不一致: {dq} != {k.shape[1]}")
    if v.shape[0] != tk:
        raise ValueError(f"k/v 序列长度不一致: {tk} != {v.shape[0]}")
    if block_q < 1 or block_k < 1:
        raise ValueError(f"block 大小必须为正: block_q={block_q}, block_k={block_k}")

    if return_weights:  # 调试通道：走 naive 路径保证语义一致
        out, w = naive_attention_forward(q, k, v, mask=mask, return_weights=True)
        return out, w, _bytes(2 * tq * tk + tq * dv)

    scale = 1.0 / math.sqrt(dq)
    offset = tk - tq  # 解码场景 kv 更长时的位置对齐

    out = np.zeros((tq, dv), dtype=np.float64)
    m_i = np.full((tq,), -np.inf, dtype=np.float64)
    l_i = np.zeros((tq,), dtype=np.float64)

    # 解析式内存记账：O 本身 + 每块临时 S/P̃（Bq×Bk）+ 递推向量
    peak_bytes = tq * dv * 8 + tq * 8 * 2 + block_q * block_k * 8 * 2

    for i0 in range(0, tq, block_q):
        i1 = min(tq, i0 + block_q)
        qi = q[i0:i1]  # (Bq, d)
        m_i[:] = -np.inf
        l_i[:] = 0.0
        o_i = np.zeros((i1 - i0, dv), dtype=np.float64)
        mi = np.full((i1 - i0,), -np.inf, dtype=np.float64)
        li = np.zeros((i1 - i0,), dtype=np.float64)

        for j0 in range(0, tk, block_k):
            j1 = min(tk, j0 + block_k)
            if mask == "causal":
                # 因果：只允许 key 位置 <= 对应的 query 位置
                if j0 > (i1 - 1) + offset:
                    continue
            kj = k[j0:j1]  # (Bk, d)
            vj = v[j0:j1]  # (Bk, dv)

            s = (qi @ kj.T) * scale  # (Bq, Bk)
            if mask == "causal":
                qpos = np.arange(i0, i1)[:, None] + offset
                kpos = np.arange(j0, j1)[None, :]
                s = np.where(kpos <= qpos, s, -np.inf)

            row_max = np.max(s, axis=-1)  # (Bq,)
            m_new = np.maximum(mi, row_max)
            # exp(mi - m_new): 首次迭代 mi=-inf ⇒ 得 0，天然正确
            corr = np.exp(mi - m_new)
            corr = np.where(np.isfinite(corr), corr, 0.0)
            p = np.exp(s - m_new[:, None])
            p = np.where(np.isfinite(p), p, 0.0)

            li = corr * li + p.sum(axis=-1)
            o_i = corr[:, None] * o_i + p @ vj
            mi = m_new

        denom = np.where(li <= 0.0, 1.0, li)
        out[i0:i1] = o_i / denom[:, None]
        _ = m_i, l_i  # 保留变量名以说明递推量的作用域

    return out, None, peak_bytes


def _bytes(n_float64: float) -> int:
    """把 float64 元素个数换算成字节数。"""
    return int(n_float64 * 8)


def multihead_forward(
    x: ArrayLike,
    wq: ArrayLike,
    wk: ArrayLike,
    wv: ArrayLike,
    wo: ArrayLike,
    *,
    n_heads: int,
    mask: MaskMode = "causal",
    engine: str = "flash",
    kv_cache: KVCache | None = None,  # noqa: F821  (前向引用，下方定义)
    rotary: object | None = None,
    positions: np.ndarray | None = None,
    block_q: int = 64,
    block_k: int = 64,
) -> tuple[np.ndarray, dict]:
    """多头注意力前向，支持 GQA / KV 缓存 / RoPE。

    Args:
        x:         输入 (T, d_model)。
        wq/wk/wv/wo: 投影权重 (d_model, n_heads*dh) 等；GQA 时 wk/wv 为
                    (d_model, n_kv_heads*dh)，n_heads 必须是 n_kv_heads 的整数倍。
        n_heads:   查询头数。
        mask:      掩码模式。
        engine:    ``"flash"`` | ``"naive"``。
        kv_cache:  可选的 KV 缓存（增量解码）。
        rotary:    RoPE 模块，需实现 ``rotate(q, k, positions)``。
        positions: RoPE 位置索引 (T,)。

    Returns:
        (输出 (T, d_model), 诊断字典)。
    """
    x = _to_f64(x, "x")
    T, d_model = x.shape
    q_all = x @ np.asarray(wq, dtype=np.float64)
    k_all = x @ np.asarray(wk, dtype=np.float64)
    v_all = x @ np.asarray(wv, dtype=np.float64)

    n_kv_heads = k_all.shape[1] // (q_all.shape[1] // n_heads)
    dh = q_all.shape[1] // n_heads
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"GQA 要求 n_heads({n_heads}) 是 n_kv_heads({n_kv_heads}) 的整数倍")
    group = n_heads // n_kv_heads

    qh = q_all.reshape(T, n_heads, dh)
    kh = k_all.reshape(T, n_kv_heads, dh)
    vh = v_all.reshape(T, n_kv_heads, dh)

    if rotary is not None and positions is not None:
        qh, kh = rotary.rotate(qh, kh, np.asarray(positions, dtype=np.int64))

    cache_len = 0
    if kv_cache is not None:
        cache_len = kv_cache.length
        kv_cache.append(kh, vh)
        kh_all, vh_all = kv_cache.k, kv_cache.v
        k_len = kv_cache.length
    else:
        kh_all, vh_all = kh, vh
        k_len = T

    outs = np.zeros((T, n_heads, dh), dtype=np.float64)
    total_bytes = 0
    for h in range(n_heads):
        h_kv = h // group
        q = qh[:, h, :]                       # (T, dh)
        k = kh_all[:k_len, h_kv, :]           # (k_len, dh)
        v = vh_all[:k_len, h_kv, :]
        if engine == "flash":
            o, _, nb = flash_attention_forward(
                q, k, v, mask=mask, block_q=block_q, block_k=block_k
            )
        elif engine == "naive":
            o, _ = naive_attention_forward(q, k, v, mask=mask, return_weights=False)
            nb = _bytes(T * k_len * 2)
        else:
            raise ValueError(f"未知注意力引擎: {engine!r} (可选 'flash' / 'naive')")
        outs[:, h, :] = o
        total_bytes += nb

    attn_out = outs.reshape(T, n_heads * dh)
    proj = attn_out @ np.asarray(wo, dtype=np.float64)  # (T, d_model)
    diag = {
        "engine": engine,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": dh,
        "cache_len": cache_len,
        "kv_len": int(k_len),
        "peak_extra_bytes": int(total_bytes),
    }
    return proj, diag


class KVCache:
    """增量解码 KV 缓存（MQA/GQA 友好）。

    为什么必须有: 自回归生成若每步重算整个前缀，代价为 O(T²·d)；
    缓存 K/V 后变为每步 O(T·d)，**总代价从 O(T²) 降到 O(T)**（摊销意义）。

    不变量 I6: 逐 token 追加后的一次前向结果，与一次性喂入整段序列的结果一致。
    """

    __slots__ = ("max_len", "n_kv_heads", "head_dim", "k", "v", "length")

    def __init__(self, max_len: int, n_kv_heads: int, head_dim: int) -> None:
        if max_len <= 0 or n_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("max_len / n_kv_heads / head_dim 必须为正")
        self.max_len = int(max_len)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.k = np.zeros((max_len, n_kv_heads, head_dim), dtype=np.float64)
        self.v = np.zeros((max_len, n_kv_heads, head_dim), dtype=np.float64)
        self.length = 0

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        """追加新的 K/V 步 (T_new, H_kv, dh)。"""
        k_new = np.asarray(k_new, dtype=np.float64)
        v_new = np.asarray(v_new, dtype=np.float64)
        n = k_new.shape[0]
        if v_new.shape[0] != n:
            raise ValueError(f"k/v 步数不一致: {n} != {v_new.shape[0]}")
        if n == 0:
            return
        if self.length + n > self.max_len:
            raise ValueError(
                f"KV 缓存溢出: 已有 {self.length} + 新增 {n} > 容量 {self.max_len}"
            )
        self.k[self.length : self.length + n] = k_new
        self.v[self.length : self.length + n] = v_new
        self.length += n

    def reset(self) -> None:
        """清空缓存（新序列开始前调用）。"""
        self.length = 0

    @property
    def bytes(self) -> int:
        """缓存占用的字节数（仅已用部分）。"""
        return self.length * self.n_kv_heads * self.head_dim * 2 * 8


def attention_memory_naive(T: int, d: int) -> int:
    """naive 注意力的额外内存字节数: 两个 T×T 矩阵 + 输出。"""
    return _bytes(2 * T * T + T * d)


def attention_memory_flash(T: int, d: int, bq: int = 64, bk: int = 64) -> int:
    """flash 注意力的额外内存字节数: O(T·d + bq·bk)。"""
    return _bytes(T * d + 2 * T + 2 * bq * bk)
