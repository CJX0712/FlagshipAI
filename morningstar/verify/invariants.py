"""L12 数学不变量库 —— 「不靠信仰，靠可验证的硬标准」。

本仓库的核心差异化: **每个算子都绑定至少一条可被独立实现交叉验证的数学不变量**。
主流框架用「跑通 demo」证明能用；本仓库用「不变量 + 参照实现的数值一致性」证明正确。

每条不变量返回统一结构::

    {"name": str, "value": float|bool, "threshold": float, "passed": bool, "detail": str}

设计原则:
  * **解析 vs 数值**: 凡有闭式解处必交叉（如均匀分布交叉熵 = ln V）。
  * **双实现交叉**: 同一语义的两条独立路径必须一致（flash vs naive、
    RoPE 对象式 vs 函数式、手写反向 vs 中心差分）。
  * **往返可逆**: 编解码类算子（BPE）必须 ``decode(encode(s)) == s``。
  * **比特级确定性**: 同 seed 两次运行逐位一致。
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np

_LOGGER = logging.getLogger(__name__)


def _res(name: str, value, threshold, passed: bool, detail: str = "") -> dict:
    """统一结果结构。"""
    return {
        "name": name,
        "value": (float(value) if isinstance(value, (int, float, np.floating)) else value),
        "threshold": threshold,
        "passed": bool(passed),
        "detail": detail,
    }


def _rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    """相对 Frobenius 误差 ‖a−b‖_F / max(‖b‖_F, ε)。"""
    num = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    den = max(float(np.linalg.norm(np.asarray(b))), 1e-12)
    return num / den


# --------------------------------------------------------------- I1 flash ≡ naive
def check_flash_naive_equivalence(
    T: int = 128, d: int = 32, seed: int = 0, tol: float = 1e-12
) -> dict:
    """**I1** FlashAttention-2 与 naive softmax attention **数值等价**。

    为什么必然成立: 两者求值的是同一个数学表达式
    ``softmax(QKᵀ/√d)V``，仅归约顺序不同 ⇒ 浮点误差仅来自结合律，
    相对误差应为机器精度量级。
    """
    from ..flash.attention import flash_attention_forward, naive_attention_forward

    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((T, d)) for _ in range(3))
    on, _ = naive_attention_forward(q, k, v, mask="none")
    of, _, _ = flash_attention_forward(q, k, v, mask="none")
    err = _rel_fro(of, on)
    return _res("I1_flash_equals_naive", err, tol, err < tol,
                f"T={T}, d={d}: FlashAttention-2 vs 精确 softmax 注意力")


# ------------------------------------------------------- I2 权重矩阵行随机性
def check_row_stochastic(T: int = 64, d: int = 16, seed: int = 1, tol: float = 1e-12) -> dict:
    """**I2** 注意力权重 P 行和为 1（softmax 的定义性质）。"""
    from ..flash.attention import naive_attention_forward

    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((T, d)) for _ in range(3))
    _, w = naive_attention_forward(q, k, v, mask="none")
    err = float(np.max(np.abs(w.sum(axis=-1) - 1.0)))
    return _res("I2_row_stochastic", err, tol, err < tol, "max|P·1 − 1|")


# --------------------------------------------------------------- I3 因果性
def check_causal_zero(T: int = 32, d: int = 8, seed: int = 2, tol: float = 1e-14) -> dict:
    """**I3** 因果掩码严格: 上三角（严格）注意力权重恒为 0。"""
    from ..flash.attention import naive_attention_forward

    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((T, d)) for _ in range(3))
    _, w = naive_attention_forward(q, k, v, mask="causal")
    upper = np.triu(w, k=1)
    err = float(np.max(np.abs(upper))) if upper.size else 0.0
    return _res("I3_causal_upper_zero", err, tol, err < tol, "max|P[i>j]|")


# ----------------------------------------------------------- I4 平移不变性
def check_shift_invariance(T: int = 32, d: int = 8, seed: int = 3, tol: float = 1e-12) -> dict:
    """**I4** softmax 平移不变: score 整体加常数，输出不变。

    这是 logsumexp 数值稳定技巧成立的理论基础
    (``softmax(s + c) = softmax(s)``)，也检验我们的实现没有隐式依赖绝对值。
    """
    from ..flash.attention import naive_attention_forward

    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((T, d)) for _ in range(3))
    o1, _ = naive_attention_forward(q, k, v, mask="none")
    # 给所有 key 加一个巨大的常数偏置：等价于 score 整体平移 √d·c
    k2 = k + 1e3
    o2, _ = naive_attention_forward(q, k2, v, mask="none")
    err = _rel_fro(o2, o1)
    return _res("I4_shift_invariance", err, tol, err < tol,
                "key 加常数 1e3 后输出应不变（logsumexp 稳定性）")


# ------------------------------------------------------------ I5 凸组合有界
def check_convex_hull(T: int = 32, d: int = 8, seed: int = 4) -> dict:
    """**I5** 输出行是 V 行的凸组合 ⇒ 每行范数不超过 max‖v_j‖。

    推论: 注意力输出天然有界，不需要额外的输出裁剪。
    """
    from ..flash.attention import naive_attention_forward

    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((T, d)) for _ in range(3))
    o, _ = naive_attention_forward(q, k, v, mask="none")
    row_max = float(np.max(np.linalg.norm(o, axis=-1)))
    v_max = float(np.max(np.linalg.norm(v, axis=-1)))
    passed = row_max <= v_max + 1e-9
    return _res("I5_convex_hull_bounded", row_max, v_max, passed,
                f"输出行范数上界 = max||v_j|| = {v_max:.4f}")


# -------------------------------------------------- I6 KV 缓存增量一致性
def check_kv_cache_consistency(
    T: int = 24, d_model: int = 32, n_head: int = 4, n_kv_head: int = 2,
    seed: int = 5, tol: float = 1e-10,
) -> dict:
    """**I6** 逐 token 增量解码 == 整段一次性前向。

    这是 KV 缓存优化正确性的充要条件：若二者不一致，
    则「训练时用整段、推理时用缓存」会引入分布偏移（一个真实且常见的工程坑）。
    """
    from ..flash.attention import (
        multihead_forward,
    )
    from ..flash.model import GPT, GPTConfig

    cfg = GPTConfig(d_model=d_model, n_layer=1, n_head=n_head, n_kv_head=n_kv_head,
                    max_len=64, vocab_size=64)
    model = GPT(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, d_model))
    positions = np.arange(T, dtype=np.int64)
    p = model.params

    whole, _ = multihead_forward(
        x, p["l0.wq"], p["l0.wk"], p["l0.wv"], p["l0.wo"],
        n_heads=n_head, mask="causal", engine="naive",
        rotary=model.rotary, positions=positions,
    )
    from ..flash.attention import KVCache

    cache = KVCache(cfg.max_len, n_kv_head, cfg.head_dim)
    outs = []
    for t in range(T):
        xt = x[t : t + 1]
        o, _ = multihead_forward(
            xt, p["l0.wq"], p["l0.wk"], p["l0.wv"], p["l0.wo"],
            n_heads=n_head, mask="causal", engine="naive",
            kv_cache=cache, rotary=model.rotary,
            positions=np.arange(t, t + 1, dtype=np.int64),
        )
        outs.append(o[0])
    incremental = np.stack(outs)
    err = _rel_fro(incremental, whole)
    return _res("I6_kv_cache_consistency", err, tol, err < tol,
                f"逐 token 解码 {T} 步 vs 一次性前向")


# --------------------------------------------------------- I7 内存次二次
def check_memory_subquadratic(T: int = 4096, d: int = 64) -> dict:
    """**I7** FlashAttention 的额外内存随 T **线性**增长，naive 为二次。

    这是 FlashAttention (Dao 2023) 的工程价值主张本身，可直接被算术验证。
    """
    from ..flash.attention import attention_memory_flash, attention_memory_naive

    naive_bytes = attention_memory_naive(T, d)
    flash_bytes = attention_memory_flash(T, d)
    ratio = naive_bytes / max(flash_bytes, 1)
    passed = ratio > 10.0  # 该尺度下 naive 至少应为 flash 的 10 倍
    return _res("I7_memory_subquadratic", ratio, 10.0, passed,
                f"T={T}: naive {naive_bytes/1e6:.1f} MB vs flash {flash_bytes/1e3:.1f} KB")


# ------------------------------------------------------------ R1 RoPE 相对性
def check_rope_relative_property(
    d: int = 32, seed: int = 6, tol: float = 1e-12
) -> dict:
    """**R1** RoPE 的精髓: 内积只依赖相对位移。

    验证: ``⟨Rot(q,m), Rot(k,n)⟩ == ⟨Rot(q,m+Δ), Rot(k,n+Δ)⟩``
    这是 RoPE 能外推到训练长度之外的根本原因 (Su et al. 2021)。
    """
    from ..flash.modules import RotaryEmbedding

    rope = RotaryEmbedding(d)
    rng = np.random.default_rng(seed)
    q, k = rng.standard_normal((d,)), rng.standard_normal((d,))
    pairs = [((3, 7), (103, 107)), ((0, 5), (50, 55)), ((11, 11), (99, 99))]
    errs = []
    for (m, n), (m2, n2) in pairs:
        lhs = float(rope.rotate_qk(q, np.array(m)) @ rope.rotate_qk(k, np.array(n)))
        rhs = float(rope.rotate_qk(q, np.array(m2)) @ rope.rotate_qk(k, np.array(n2)))
        errs.append(abs(lhs - rhs) / max(abs(lhs), 1e-12))
    err = max(errs)
    return _res("R1_rope_relative_positions", err, tol, err < tol,
                "⟨R_m q, R_n k⟩ 仅在 n−m 上取值")


def check_rope_norm_preserving(d: int = 32, seed: int = 7, tol: float = 1e-12) -> dict:
    """**R2** RoPE 是正交变换 ⇒ 保持向量范数（不会放大/压缩表示）。"""
    from ..flash.modules import RotaryEmbedding, rotate_half_numpy

    rope = RotaryEmbedding(d)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((8, d))
    pos = np.arange(8, dtype=np.int64)
    xr = rope.rotate_qk(x, pos)
    err = float(np.max(np.abs(np.linalg.norm(xr, axis=-1) - np.linalg.norm(x, axis=-1))))
    # 双实现交叉验证: 对象式 vs 纯函数式
    cross = float(np.max(np.abs(xr - rotate_half_numpy(x, pos, d))))
    err = max(err, cross)
    return _res("R2_rope_norm_preserving", err, tol, err < tol,
                "‖Rot(x,m)‖ = ‖x‖ 且两条独立实现逐位一致")


# ------------------------------------------------------------- R3 RMSNorm
def check_rmsnorm_unit_rms(d: int = 64, seed: int = 8, tol: float = 1e-9) -> dict:
    """**R3** g=1 时归一化输出的 RMS 恒为 1（定义性质）。"""
    from ..flash.modules import RMSNorm

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((16, d)) * 3.0 + 5.0
    y = RMSNorm(d).forward(x)
    rms = np.sqrt(np.mean(y * y, axis=-1))
    err = float(np.max(np.abs(rms - 1.0)))
    return _res("R3_rmsnorm_unit_rms", err, tol, err < tol, "RMS(y) = 1 ∀ rows")


# --------------------------------------------------------- G1 GPT 梯度检验
def gpt_gradient_check(
    vocab_size: int = 32, d_model: int = 32, n_layer: int = 2, n_head: int = 4,
    n_kv_head: int = 2, T: int = 8, seed: int = 9, tol: float = 1e-6,
) -> dict:
    """**G1** 解析梯度 vs 中心差分 —— 手写反向传播的硬标准。

    世界级基线: PyTorch/JAX 官方实现的 gradient-check 容差约 1e-5~1e-7；
    本仓库在 float64 下通常达到 ~1e-9。
    """
    from ..flash.model import GPT, GPTConfig

    cfg = GPTConfig(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer,
                    n_head=n_head, n_kv_head=n_kv_head, max_len=64)
    model = GPT(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vocab_size, size=T)
    targets = rng.integers(0, vocab_size, size=T)

    loss0, grads = model.loss_and_grad(idx, targets)

    # 只检查若干代表参数 + 嵌入（穷举全部太慢）
    keys = ["wte", "l0.wq", "l0.wo", "l0.w_gate", "lm_head"] + (
        [f"l{n_layer-1}.wk", f"l{n_layer-1}.w_down"] if n_layer > 1 else []
    )
    eps = 1e-6
    worst = 0.0
    worst_key = ""
    for key in keys:
        p = model.params[key]
        flat_shape = p.shape
        for _ in range(8):
            i = tuple(rng.integers(0, s) for s in flat_shape)
            old = p[i]
            p[i] = old + eps
            lp, _ = model.loss_and_grad(idx, targets)
            p[i] = old - eps
            lm, _ = model.loss_and_grad(idx, targets)
            p[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[key][i]
            abs_err = abs(num - ana)
            # 组合容差（梯度检验标准做法，等价 np.allclose 的 rtol+atol）:
            # 极小梯度处有限差分的固有噪声用 atol 兜底，避免误判。
            ok = (abs_err / max(abs(num), abs(ana), 1e-12)) < tol or abs_err < 1e-7
            rel = abs_err / max(abs(num), abs(ana), 1e-12)
            if not ok and rel > worst:
                worst, worst_key = rel, key
    return _res("G1_gpt_gradient_check", worst, tol, worst < tol,
                f"最差点在 {worst_key}; loss={loss0:.4f}")


# ------------------------------------------------- G4 交叉熵解析值 (均匀分布)
def check_crossentropy_uniform(V: int = 64, tol: float = 1e-12) -> dict:
    """**G4** logits 全零 ⇒ 均匀分布 ⇒ 交叉熵精确等于 ln(V)（解析解）。"""
    from ..flash.model import _cross_entropy_loss

    logits = np.zeros((4, V))
    targets = np.arange(4)
    got = _cross_entropy_loss(logits, targets)
    want = math.log(V)
    err = abs(got - want)
    return _res("G4_crossentropy_analytic", err, tol, err < tol,
                f"ln({V}) = {want:.6f}, got {got:.6f}")


# ------------------------------------------------------------ P1 比特级确定性
def check_determinism(seed: int = 11) -> dict:
    """**P1** 同 seed ⇒ 逐位一致的初始化与前向输出。"""
    from ..flash.model import GPT, GPTConfig

    cfg = GPTConfig(vocab_size=32, d_model=32, n_layer=2, n_head=4, n_kv_head=2, max_len=64)
    m1, m2 = GPT(cfg, seed=seed), GPT(cfg, seed=seed)
    same_init = all(np.array_equal(m1.params[k], m2.params[k]) for k in m1.params)
    idx = np.arange(8)
    l1, _ = m1.forward(idx)
    l2, _ = m2.forward(idx)
    same_fwd = bool(np.array_equal(l1, l2))
    return _res("P1_bitwise_determinism", 0.0 if (same_init and same_fwd) else 1.0,
                0.5, bool(same_init and same_fwd),
                f"init_identical={same_init}, forward_identical={same_fwd}")


# ------------------------------------------------------------- T1 BPE 往返
def check_bpe_roundtrip(vocab_size: int = 300, seed: int = 12) -> dict:
    """**T1** BPE 编解码无损往返 —— 分词实现的正确性底线。"""
    from ..flash.tokenizer import BPETokenizer

    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "晨星锻造世界顶级人工智能系统，零重写、集大成、可验证。",
        "Transformers scale: attention is all you need, but memory is the wall.",
        "RMSNorm, RoPE, SwiGLU — the Llama recipe.",
    ] * 8
    tok = BPETokenizer().train(docs, vocab_size=vocab_size)
    ok = all(tok.decode(tok.encode(d)) == d for d in docs)
    return _res("T1_bpe_roundtrip", 0.0 if ok else 1.0, 0.5, ok,
                f"vocab_size={tok.vocab_size}, 语料 {len(docs)} 条")


def check_bpe_determinism(vocab_size: int = 300) -> dict:
    """**T3** 同语料 + 同 vocab_size ⇒ 完全相同的 merges 序列。"""
    from ..flash.tokenizer import BPETokenizer

    docs = ["determinism matters for reproducible science"] * 20
    a = BPETokenizer().train(docs, vocab_size=vocab_size)
    b = BPETokenizer().train(docs, vocab_size=vocab_size)
    same = a.merges == b.merges
    return _res("T3_bpe_determinism", 0.0 if same else 1.0, 0.5, bool(same),
                f"{len(a.merges)} merges")


# ------------------------------------------------------- A4 ≈ 注意力保真度
def _normalize_rows(x: np.ndarray, target_sq: float) -> np.ndarray:
    """把每个向量缩放到目标平方范数。

    用于模拟真实 Transformer 中进入注意力的 Q/K: Llama 风格模型在投影前
    经过 RMSNorm，输出范数受控。FAVOR+ 的范数因子 ``exp(±‖x‖²/2)`` 会把
    估计方差放大 ``exp(±‖x‖²)`` 倍，因此**输入范数直接决定近似保真度**
    （见 PerformerAttention 类文档第 4 点）。
    """
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * (np.sqrt(target_sq) / np.maximum(n, 1e-12))


def check_approx_attention_fidelity(
    T: int = 256, d: int = 32, seed: int = 13, m: int = 256
) -> dict:
    """近似注意力（Performer/Linear）相对精确注意力的输出保真度。

    输入采用**受控范数**的 Q/K（模拟 RMSNorm 后的真实注意力输入），
    这是 FAVOR+ 成立的前提条件。同时报告**原始尺度**下的误差作为对照，
    量化「输入范数 → 近似保真度」的敏感度。

    EffAttnForge G1 门的延续: 报告相对 Frobenius 误差，用于
    「多便宜换多少失真」的权衡表。
    """
    from ..attention import FullAttention, LinearAttention, PerformerAttention

    rng = np.random.default_rng(seed)
    q_raw, k_raw, v = (rng.standard_normal((T, d)) for _ in range(3))

    # 受控范数: ‖q‖² ≈ 0.5·√d ⇒ 特征空间 ‖x‖² = ‖q‖²/√d ≈ 0.5 ⇒ 范数因子 ≈ 1
    target_sq = 0.5 * (d ** 0.5)
    q = _normalize_rows(q_raw, target_sq)
    k = _normalize_rows(k_raw, target_sq)

    ref = FullAttention().attend(q, k, v).output
    out: dict[str, float] = {}
    for name, inst in [
        ("linear", LinearAttention()),
        ("performer", PerformerAttention(m=m)),
    ]:
        approx = inst.attend(q, k, v).output
        out[name] = _rel_fro(approx, ref)

    # 对照: 原始 N(0,1) 尺度下 FAVOR+ 的方差退化（记录不参与通过判定）
    raw_ref = FullAttention().attend(q_raw, k_raw, v).output
    raw_perf = PerformerAttention(m=m).attend(q_raw, k_raw, v).output
    raw_err = _rel_fro(raw_perf, raw_ref)

    worst = max(out.values())
    return _res(
        "A4_approx_attention_fidelity", worst, 0.5, worst < 0.5,
        "linear=%.4f, performer=%.4f (m=%d, ‖q‖²≈%.2f); 原始尺度 performer=%.3f"
        % (out["linear"], out["performer"], m, target_sq, raw_err),
    )


# ------------------------------------------------------------------ registry
ALL_INVARIANTS: dict[str, Callable[[], dict]] = {
    "I1_flash_naive_equivalence": lambda: check_flash_naive_equivalence(),
    "I2_row_stochastic": lambda: check_row_stochastic(),
    "I3_causal_upper_zero": lambda: check_causal_zero(),
    "I4_shift_invariance": lambda: check_shift_invariance(),
    "I5_convex_hull_bounded": lambda: check_convex_hull(),
    "I6_kv_cache_consistency": lambda: check_kv_cache_consistency(),
    "I7_memory_subquadratic": lambda: check_memory_subquadratic(),
    "R1_rope_relative_positions": lambda: check_rope_relative_property(),
    "R2_rope_norm_preserving": lambda: check_rope_norm_preserving(),
    "R3_rmsnorm_unit_rms": lambda: check_rmsnorm_unit_rms(),
    "G1_gpt_gradient_check": lambda: gpt_gradient_check(),
    "G4_crossentropy_analytic": lambda: check_crossentropy_uniform(),
    "P1_bitwise_determinism": lambda: check_determinism(),
    "T1_bpe_roundtrip": lambda: check_bpe_roundtrip(),
    "T3_bpe_determinism": lambda: check_bpe_determinism(),
    "A4_approx_attention_fidelity": lambda: check_approx_attention_fidelity(),
}


def run_all() -> list[dict]:
    """跑全部不变量，返回结果列表。"""
    results = []
    for name, fn in ALL_INVARIANTS.items():
        try:
            r = fn()
        except Exception as exc:  # 不变量失败必须显性呈现，不能静默
            _LOGGER.exception("不变量 %s 执行异常", name)
            r = _res(name, float("nan"), float("nan"), False, f"EXCEPTION: {exc}")
        results.append(r)
    return results
