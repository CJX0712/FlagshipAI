"""Flagship GPT —— Decoder-only Transformer，含手写反向传播与 KV 缓存增量解码。

世界级出处 (详见 docs/REFERENCES.md):

  * **Transformer / GPT** : Vaswani et al. (2017) NeurIPS; Radford et al. (2018)
    *Improving Language Understanding by Generative Pre-Training*.
  * **Pre-norm 结构** : 现代 LLM 标准做法（GPT-2 后），残差支路里做归一化
    ⇒ 深层模型训练稳定性显著优于 post-norm。
  * **Llama 风格组件组合** : Touvron et al. (2023) *LLaMA: Open and Efficient
    Foundation Language Models*. —— RMSNorm + SwiGLU + RoPE + GQA。
  * **权重初始化** : He et al. (2015) *Delving Deep into Rectifiers* (残差支路
    按 1/√n_layer 缩放，源自 GPT-2 的实现惯例)。

工程取舍（明确说明，避免误解）:

  * **训练路径**使用显式注意力矩阵（``engine="naive"``）—— 因为反向传播需要
    中间的 softmax 权重 P **或其雅可比**；直接写 flash 的 backward 属于
    纯工程优化，不改变数学结果（本仓库提供前向等价性不变量 I1 证明二者一致）。
  * **推理路径**默认 ``engine="flash"`` + KV 缓存 ⇒ 内存 O(T·d)、单步 O(T)。

硬不变量:
  **G1 梯度检验**: 解析梯度 vs 中心差分，maxRelErr < 1e-6（默认 float64，
      通常可达 ~1e-9）。
  **G2 增量一致性**: KV 缓存逐 token 解码 == 全序列一次性前向（见 attention.I6）。
  **G3 数值稳定**: log-softmax 用 logsumexp，任意 logits 量级都不产生 NaN。
  **G4 奇偶穷 lifeguard**: 交叉熵在均匀预测时等于 ln(V)（解析值校验）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .attention import multihead_forward
from .modules import RMSNorm, RotaryEmbedding, SwiGLUFeedForward, _ffn_hidden, _sigmoid
from .random import SALT_INIT, SALT_INIT, rng_for  # noqa: F401 (SALT 多用例占位)

_LOGGER = logging.getLogger(__name__)


@dataclass
class GPTConfig:
    """模型超参容器。默认值是一个「能在 CPU 上 60 秒内训出可见效果」的小模型。"""

    vocab_size: int = 512
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2          # GQA: 2 个 KV 头服务 4 个查询头
    max_len: int = 256
    d_ff: int | None = None     # None ⇒ 按 Llama 惯例自动推导
    rope_base: float = 10000.0
    dropout: float = 0.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model({self.d_model}) 必须能被 n_head({self.n_head}) 整除"
            )
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head({self.n_head}) 必须是 n_kv_head({self.n_kv_head}) 的整数倍"
            )
        if self.d_ff is None:
            object.__setattr__(self, "d_ff", _ffn_hidden(self.d_model))

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        d = dict(vars(self))
        d["head_dim"] = self.head_dim
        return d


class GPT:
    """Decoder-only Transformer (Llama 风格)。

    参数以 ``dict[str, np.ndarray]`` 存放，便于
      * 直接喂给 ``AdamW.step``；
      * 与 PyTorch state_dict 做逐张量交叉验证；
      * 序列化到本地 .npz。
    """

    def __init__(self, cfg: GPTConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.seed = seed
        self.params: dict[str, np.ndarray] = {}
        self._init_params()
        self.rotary = RotaryEmbedding(cfg.head_dim, base=cfg.rope_base, max_len=cfg.max_len)
        self._ffn = SwiGLUFeedForward(cfg.d_model, cfg.d_ff)

    # ------------------------------------------------------------- init
    def _init_params(self) -> None:
        """He/GPT-2 风格初始化; 残差支路权重按 1/√(2·L) 缩放 (GPT-2 惯例)。"""
        cfg = self.cfg
        rng = rng_for(SALT_INIT, self.seed)
        C, V = cfg.d_model, cfg.vocab_size
        dh = cfg.head_dim
        kv_dim = cfg.n_kv_head * dh
        F = int(cfg.d_ff or _ffn_hidden(C))

        def he(shape: tuple[int, ...], fan_in: int) -> np.ndarray:
            return rng.standard_normal(shape) * math.sqrt(2.0 / fan_in)

        p: dict[str, np.ndarray] = {}
        p["wte"] = rng.standard_normal((V, C)) * 0.02
        p["ln_f_w"] = np.ones(C, dtype=np.float64)
        # 残差支路缩放: 让残差方差随层数线性叠加而非爆炸
        resid_scale = 1.0 / math.sqrt(2.0 * cfg.n_layer)
        for li in range(cfg.n_layer):
            p[f"l{li}.ln1_w"] = np.ones(C, dtype=np.float64)
            p[f"l{li}.wq"] = he((C, cfg.n_head * dh), C) * resid_scale
            p[f"l{li}.wk"] = he((C, kv_dim), C) * resid_scale
            p[f"l{li}.wv"] = he((C, kv_dim), C) * resid_scale
            p[f"l{li}.wo"] = he((cfg.n_head * dh, C), cfg.n_head * dh) * resid_scale
            p[f"l{li}.ln2_w"] = np.ones(C, dtype=np.float64)
            p[f"l{li}.w_gate"] = he((C, F), C)
            p[f"l{li}.w_up"] = he((C, F), C)
            p[f"l{li}.w_down"] = he((F, C), F) * resid_scale
        p["lm_head"] = rng.standard_normal((C, V)) * 0.02
        self.params = p

    @property
    def n_params(self) -> int:
        """可训练参数总数。"""
        return int(sum(a.size for a in self.params.values()))

    # ----------------------------------------------------------- forward
    def forward(
        self,
        idx: np.ndarray,
        targets: np.ndarray | None = None,
        *,
        engine: str = "naive",
        kv_cache_per_layer: list | None = None,
        start_pos: int = 0,
    ) -> tuple[np.ndarray, dict]:
        """前向传播。

        Args:
            idx:       token id 序列，形状 (T,) 单样本。
            targets:   可选的目标 id，形状 (T,)；提供时同时返回交叉熵损失。
            engine:    ``"naive"``(训练/带梯度) 或 ``"flash"``(推理省内存)。
            kv_cache_per_layer: 增量解码用的 KV 缓存列表（每层一个）。
            start_pos: 当前 token 在全序列中的起始位置（配合 KV 缓存）。

        Returns:
            (logits (T, V), cache 字典) —— cache 保存反向所需的中间量。
        """
        cfg = self.cfg
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"idx 必须是一维序列，收到形状 {idx.shape}")
        T = idx.shape[0]
        if T > cfg.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={cfg.max_len}")
        p = self.params
        C = cfg.d_model

        x = p["wte"][idx]                       # (T, C) 嵌入查表
        cache: dict = {"T": T, "layers": [], "start_pos": start_pos}

        for li in range(cfg.n_layer):
            lc: dict = {}
            # ---- pre-norm 1 → attention
            h, inv1 = _rmsnorm_fwd(x, p[f"l{li}.ln1_w"], cfg.eps)
            lc["h1"], lc["inv1"] = h, inv1
            lc["x_in"] = x  # 本层输入：反向时 RMSNorm 求 dx 需要它

            q = h @ p[f"l{li}.wq"]              # (T, n_head*dh)
            k = h @ p[f"l{li}.wk"]
            v = h @ p[f"l{li}.wv"]

            if engine == "naive":
                # 显式注意力: 需要 P 参与反向
                attn_out, attn_cache = self._attention_explicit(
                    q, k, v, li, start_pos, need_grad=True
                )
                lc["attn"] = attn_cache
            else:
                attn_out, _diag = multihead_forward(
                    h,
                    p[f"l{li}.wq"], p[f"l{li}.wk"], p[f"l{li}.wv"], p[f"l{li}.wo"],
                    n_heads=cfg.n_head, mask="causal", engine="flash",
                    kv_cache=(kv_cache_per_layer[li] if kv_cache_per_layer else None),
                    rotary=self.rotary,
                    positions=np.arange(start_pos, start_pos + T, dtype=np.int64),
                )
                x2 = x + attn_out
                xr, inv2 = _rmsnorm_fwd(x2, p[f"l{li}.ln2_w"], cfg.eps)
                ffn_out, ffn_cache = _swiglu_fwd(
                    xr, p[f"l{li}.w_gate"], p[f"l{li}.w_up"], p[f"l{li}.w_down"]
                )
                lc.update({"h2": xr, "inv2": inv2, "ffn": ffn_cache, "x_pre": x2})
                x = x2 + ffn_out
                cache["layers"].append(lc)
                continue

            x2 = x + attn_out
            lc["x_pre"] = x2
            # ---- pre-norm 2 → SwiGLU FFN
            xr, inv2 = _rmsnorm_fwd(x2, p[f"l{li}.ln2_w"], cfg.eps)
            ffn_out, ffn_cache = _swiglu_fwd(
                xr, p[f"l{li}.w_gate"], p[f"l{li}.w_up"], p[f"l{li}.w_down"]
            )
            lc.update({"h2": xr, "inv2": inv2, "ffn": ffn_cache})
            x = x2 + ffn_out
            cache["layers"].append(lc)

        xf, inv_f = _rmsnorm_fwd(x, p["ln_f_w"], cfg.eps)
        cache["x_final_raw"], cache["inv_f"] = x, inv_f
        # 注意区分: x_final_raw 是 RMSNorm **之前**的输入（求 dx 需要），
        # x_final 是归一化并乘增益后的隐藏态（求 lm_head 梯度需要）。
        cache["x_final"] = xf
        logits = xf @ p["lm_head"]               # (T, V)

        loss = None
        if targets is not None:
            targets = np.asarray(targets, dtype=np.int64)
            if targets.shape != idx.shape:
                raise ValueError(f"targets 形状 {targets.shape} 与 idx {idx.shape} 不一致")
            loss = _cross_entropy_loss(logits, targets)
            cache["targets"] = targets
        cache["loss"] = loss
        return logits, cache

    def _attention_explicit(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
        layer_idx: int, start_pos: int, need_grad: bool
    ) -> tuple[np.ndarray, dict]:
        """显式多头因果注意力（保存 P 供反向）。

        返回投影后的输出 (T, C) 与反向所需的中间量。
        """
        cfg = self.cfg
        T = q.shape[0]
        dh = cfg.head_dim
        H, Hkv = cfg.n_head, cfg.n_kv_head
        p = self.params

        qh = q.reshape(T, H, dh)
        kh = k.reshape(T, Hkv, dh)
        vh = v.reshape(T, Hkv, dh)
        positions = np.arange(start_pos, start_pos + T, dtype=np.int64)
        qh = self.rotary.rotate_qk(qh, positions)
        kh = self.rotary.rotate_qk(kh, positions)

        group = H // Hkv
        scale = 1.0 / math.sqrt(dh)
        outs = np.zeros((T, H, dh), dtype=np.float64)
        # RoPE 旋转后必须缓存 dimensionless 的 positions，供反向的逆旋转复用
        ac: dict = {"qh": qh, "kh": kh, "vh": vh, "heads": [], "positions": positions}

        kpos = np.arange(start_pos, start_pos + T, dtype=np.int64)
        causal = (kpos[None, :] <= kpos[:, None])  # (T, T)

        for hh in range(H):
            hkv = hh // group
            qi, ki, vi = qh[:, hh, :], kh[:, hkv, :], vh[:, hkv, :]
            s = (qi @ ki.T) * scale
            s = np.where(causal, s, -np.inf)
            s_shift = s - np.max(s, axis=-1, keepdims=True)
            e = np.exp(s_shift)
            denom = e.sum(axis=-1, keepdims=True)
            denom = np.where(denom <= 0.0, 1.0, denom)
            P = e / denom
            outs[:, hh, :] = P @ vi
            if need_grad:
                ac["heads"].append({"P": P, "s": s_shift, "scale": scale})

        flat = outs.reshape(T, H * dh)
        out = flat @ p[f"l{layer_idx}.wo"]
        ac["flat"], ac["out"] = flat, out
        return out, ac

    # ---------------------------------------------------------- backward
    def backward(self, idx: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        """手写反向传播 —— 全模型的解析梯度。

        Args:
            idx:   前向时的输入序列。
            cache: ``forward`` 返回的中间量字典。

        Returns:
            参数名 → 梯度数组（形状与 params 一一对应）。
        """
        cfg = self.cfg
        p = self.params
        T = cache["T"]
        C = cfg.d_model
        V = cfg.vocab_size
        dh = cfg.head_dim
        H, Hkv = cfg.n_head, cfg.n_kv_head
        group = H // Hkv

        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in p.items()}
        targets = cache.get("targets")
        if targets is None:
            raise ValueError(
                "backward 需要训练目标。请使用 GPT.loss_and_grad(idx, targets) 一站式接口，"
                "或在 forward 时提供 targets。"
            )
        if "logits" not in cache:
            raise KeyError(
                "cache 缺少 'logits'：请使用 GPT.loss_and_grad() 一站式接口，"
                "它会自动把 logits 注入 cache 以避免重复前向。"
            )

        # ---- 损失层 (交叉熵 + softmax 的合并雅可比): dL/dlogits = (p − onehot)/T
        logits = cache["logits"]
        probs = _softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= float(T)

        # lm_head 梯度用的是 **归一化后** 的隐藏态
        grads["lm_head"] += cache["x_final"].T @ dlogits
        dhidden = dlogits @ p["lm_head"].T
        dx, dg_final = _rmsnorm_bwd(
            dhidden, cache["x_final_raw"], p["ln_f_w"], cache["inv_f"], cfg.eps, None
        )
        grads["ln_f_w"] += dg_final

        for li in reversed(range(cfg.n_layer)):
            lc = cache["layers"][li]
            # ---- FFN 残差支路
            dffn_out = dx                                       # 残差直连
            dx2 = dx.copy()                                     # 另一路往回传
            g_gate, g_up, g_down, dxr = _swiglu_bwd(lc["ffn"], dffn_out)
            grads[f"l{li}.w_gate"] += g_gate
            grads[f"l{li}.w_up"] += g_up
            grads[f"l{li}.w_down"] += g_down
            dpre = _rmsnorm_bwd(
                dxr, lc["x_pre"], p[f"l{li}.ln2_w"], lc["inv2"], cfg.eps, None
            )
            grads[f"l{li}.ln2_w"] += dpre[1]
            dx2 += dpre[0]

            # ---- Attention 残差支路
            ac = lc["attn"]
            dflat = dx2 @ p[f"l{li}.wo"].T                       # (T, H*dh)
            grads[f"l{li}.wo"] += ac["flat"].T @ dx2

            dqh = np.zeros_like(ac["qh"])
            dkh = np.zeros_like(ac["kh"])
            dvh = np.zeros_like(ac["vh"])
            for hh in range(H):
                hkv = hh // group
                hd = ac["heads"][hh]
                P = hd["P"]
                scale = hd["scale"]
                dout_h = dflat[:, hh * dh : (hh + 1) * dh]       # (T, dh)

                dvh[:, hkv, :] += P.T @ dout_h
                dP = dout_h @ ac["vh"][:, hkv, :].T              # (T, T)
                rowsum = np.sum(dP * P, axis=-1, keepdims=True)
                dS = P * (dP - rowsum)                           # softmax 雅可比
                causal = np.tril(np.ones((T, T), dtype=bool))
                dS = np.where(causal, dS, 0.0)
                dqh[:, hh, :] += (dS @ ac["kh"][:, hkv, :]) * scale
                dkh[:, hkv, :] += (dS.T @ ac["qh"][:, hh, :]) * scale

            pos = ac["positions"]
            dqh = self.rotary.rotate_qk_inverse(dqh, pos)
            dkh = self.rotary.rotate_qk_inverse(dkh, pos)
            dq = dqh.reshape(T, H * dh)
            dk = dkh.reshape(T, Hkv * dh)
            dv = dvh.reshape(T, Hkv * dh)
            grads[f"l{li}.wq"] += lc["h1"].T @ dq
            grads[f"l{li}.wk"] += lc["h1"].T @ dk
            grads[f"l{li}.wv"] += lc["h1"].T @ dv
            dh1 = dq @ p[f"l{li}.wq"].T + dk @ p[f"l{li}.wk"].T + dv @ p[f"l{li}.wv"].T
            dnorm1 = _rmsnorm_bwd(dh1, lc["x_in"], p[f"l{li}.ln1_w"],
                                  lc["inv1"], cfg.eps, lc["h1"])
            grads[f"l{li}.ln1_w"] += dnorm1[1]
            # 残差拓扑: x_pre = x + attn_out ⇒ 流向本层输入 x 的梯度是两条支路之和
            #   (1) 残差直连 dx2      (2) 经 attention 支路回传的 dnorm1[0]
            dx = dx2 + dnorm1[0]

        # ---- Embedding
        np.add.at(grads["wte"], idx, dx)
        return grads

    def loss_and_grad(
        self, idx: np.ndarray, targets: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """一站式: 前向 + 反向，返回 (损失, 梯度字典)。推荐用于训练。"""
        logits, cache = self.forward(idx, targets)
        cache["logits"] = logits
        grads = self.backward(idx, cache)
        loss = float(cache["loss"])
        return loss, grads

    # ------------------------------------------------------------ generate
    def generate(
        self,
        prompt_ids: list[int] | np.ndarray,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        engine: str = "flash",
        stream: bool = False,
    ) -> Iterator[int] | list[int]:
        """KV 缓存自回归生成。

        Args:
            prompt_ids:      提示序列。
            max_new_tokens:  最多生成多少个 token。
            temperature:     >1 更随机，<1 更确定；→0 退化为贪心。
            top_k:           top-k 截断采样 (Fan et al. 2018 *Hierarchical Neural
                             Story Generation* 提出的截断采样族)。
            seed:            采样随机种子（可复现）。
            engine:          推理引擎，默认 flash（省内存）。
            stream:          True 时返回生成器，逐 token 产出。

        复杂度: 无缓存 O(T²·d)；有缓存 O(T·d) —— 长文本生成的关键优化。
        """
        cfg = self.cfg
        ids = list(prompt_ids)
        if len(ids) == 0:
            raise ValueError("提示序列不能为空")
        if len(ids) + max_new_tokens > cfg.max_len:
            raise ValueError(
                f"总长 {len(ids) + max_new_tokens} 超过 max_len={cfg.max_len}"
            )
        from .attention import KVCache

        caches = [
            KVCache(cfg.max_len, cfg.n_kv_head, cfg.head_dim)
            for _ in range(cfg.n_layer)
        ]
        rng = None if (temperature == 0.0 or seed is None) else (
            __import__("numpy").random.default_rng(seed)
        )

        def _produce() -> Iterator[int]:
            pos = 0
            # --- prefill: 一次吃掉整个提示（含内部因果注意力）
            logits, _ = self.forward(
                np.asarray(ids, dtype=np.int64), engine=engine,
                kv_cache_per_layer=caches, start_pos=pos,
            )
            pos = len(ids)
            cur = int(np.argmax(logits[-1])) if temperature == 0.0 else _sample(
                logits[-1], temperature, top_k, rng
            )
            for nxt in [cur]:
                pass
            produced = [cur]
            yield cur

            for _ in range(max_new_tokens - 1):
                logits, _ = self.forward(
                    np.asarray([produced[-1]], dtype=np.int64), engine=engine,
                    kv_cache_per_layer=caches, start_pos=pos,
                )
                pos += 1
                last = logits[-1]
                if temperature == 0.0:
                    nxt = int(np.argmax(last))
                else:
                    nxt = _sample(last, temperature, top_k, rng)
                produced.append(nxt)
                yield nxt

        if stream:
            return _produce()
        return list(_produce())

    # ------------------------------------------------------------- io
    def save(self, path: str) -> None:
        """保存权重为 .npz（含配置）。"""
        meta = {"cfg": np.array([0])}  # 占位以保持 npz 纯数值
        import json

        np.savez(path, **self.params, __cfg__=np.frombuffer(
            json.dumps(self.cfg.to_dict()).encode("utf-8"), dtype=np.uint8
        ))

    @classmethod
    def load(cls, path: str) -> "GPT":
        """从 .npz 恢复模型。"""
        import json

        with np.load(path, allow_pickle=False) as z:
            raw = z["__cfg__"].tobytes().decode("utf-8")
            cfgd = json.loads(raw)
            cfgd.pop("head_dim", None)
            cfg = GPTConfig(**cfgd)
            model = cls.__new__(cls)
            model.cfg = cfg
            model.seed = None
            model.params = {k: z[k] for k in z.files if k != "__cfg__"}
            model.rotary = RotaryEmbedding(cfg.head_dim, base=cfg.rope_base, max_len=cfg.max_len)
            model._ffn = SwiGLUFeedForward(cfg.d_model, cfg.d_ff)
            return model


# ============================================================ helpers
def _rmsnorm_fwd(x: np.ndarray, g: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """RMSNorm 前向，返回 (输出, 1/rms) 以便反向复用。"""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    inv = 1.0 / rms
    return x * inv * g, inv


def _rmsnorm_bwd(
    dy: np.ndarray,
    x_src: np.ndarray | None,
    g: np.ndarray,
    inv: np.ndarray,
    eps: float,
    x_normed: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """RMSNorm 反向。

    ``x_hat = x · inv``；``inv = 1/sqrt(mean(x²)+eps)``。
    推导（忽略 eps 的导数，业界惯例）::

        dy_hat = dy · g
        Σ     = Σ_i dy_hat_i x_i
        dx    = inv · dy_hat − inv³ · x · Σ / D
        dg    = Σ_rows (dy · x_hat)
    """
    x = x_src if x_src is not None else (dy * 0.0)
    if x_src is None:
        # 无源张量时无法求 dx（仅需 dg 的场景），返回零
        xhat = x_normed if x_normed is not None else np.zeros_like(dy)
        return np.zeros_like(dy), np.sum(dy * xhat, axis=0)
    D = x.shape[-1]
    dy_hat = dy * g
    sigma = np.sum(dy_hat * x, axis=-1, keepdims=True)
    dx = inv * dy_hat - (inv ** 3) * x * sigma / D
    xhat = x * inv
    dg = np.sum(dy * xhat, axis=0)
    return dx, dg


def _swiglu_fwd(
    x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray
) -> tuple[np.ndarray, dict]:
    """SwiGLU 前向，缓存反向所需的中间量。"""
    gate = x @ w_gate
    up = x @ w_up
    sig = _sigmoid(gate)
    act = gate * sig
    hidden = act * up
    out = hidden @ w_down
    return out, {"x": x, "gate": gate, "sig": sig, "act": act, "up": up,
                 "hidden": hidden, "w_down": w_down, "w_gate": w_gate, "w_up": w_up}


def _swiglu_bwd(cache: dict, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SwiGLU 反向。返回 (dW_gate, dW_up, dW_down, dx)。"""
    x, gate, sig, act, up = (cache[k] for k in ("x", "gate", "sig", "act", "up"))
    hidden, w_down, w_gate, w_up = (
        cache[k] for k in ("hidden", "w_down", "w_gate", "w_up")
    )
    dhid = dout @ w_down.T                      # (T, F)
    d_down = hidden.T @ dout
    d_up = dhid * act
    d_act = dhid * up
    # swish'(g) = σ(g)·(1 + g·(1−σ(g)))
    d_gate_pre = d_act * (sig * (1.0 + gate * (1.0 - sig)))
    d_gate = x.T @ d_gate_pre
    d_wup = x.T @ d_up
    dx = d_gate_pre @ w_gate.T + d_up @ w_up.T
    return d_gate, d_wup, d_down, dx


def _softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定 softmax。"""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    s = e.sum(axis=-1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return e / s


def _cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """平均交叉熵: ``-1/T Σ log softmax(logits)[target]`` (logsumexp 稳定版)。

    不变量 G4: 当 logits 全为 0 时，loss 应等于 ln(V)（均匀分布）。
    """
    T = logits.shape[0]
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - logsumexp
    picked = log_probs[np.arange(T), targets]
    return float(-np.mean(picked))


def _sample(
    logits: np.ndarray, temperature: float, top_k: int | None, rng
) -> int:
    """温度缩放 + top-k 截断采样 (Fan et al. 2018)。"""
    if rng is None:
        return int(np.argmax(logits))
    z = np.asarray(logits, dtype=np.float64) / max(1e-12, temperature)
    if top_k is not None and 0 < top_k < z.shape[0]:
        thresh = np.sort(z)[-top_k]
        z = np.where(z >= thresh, z, -np.inf)
    z = z - np.max(z)
    probs = np.exp(z)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))
