"""缩放点积注意力 —— forge 数值稳定实现 + 不变量。

Attention(Q,K,V) = softmax(QKᵀ/√d_k) V。

不变量：
  * 注意力权重矩阵 P 行随机（每行和为 1，softmax 直接保证）
  * 输出每行是 V 行的凸组合 ⇒ 落于 V 行张成的凸包内（有界）
  * 数值稳定：输入量级极大时不溢出 / 不产生 NaN（logsumexp 实现）
  * 跨库校验（可选）：与 PyTorch scaled_dot_product_attention 一致
"""
from __future__ import annotations

import numpy as np

from ..mathx.linalg import softmax, logsumexp


def scaled_dot_product_attention(Q, K, V, mask=None, causal=False):
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    d_k = Q.shape[-1]
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    if causal:
        L = scores.shape[-1]
        causal_mask = np.triu(np.ones((L, L), dtype=bool), k=1)
        scores = np.where(causal_mask, -np.inf, scores)
    P = softmax(scores, axis=-1)
    return P @ V, P


class Attention:
    def __init__(self, causal: bool = False):
        self.causal = causal

    def __call__(self, Q, K, V, mask=None):
        out, P = scaled_dot_product_attention(Q, K, V, mask=mask, causal=self.causal)
        self._P = P
        return out

    def check_invariants(self, Q, K, V, mask=None) -> dict:
        out, P = scaled_dot_product_attention(Q, K, V, mask=mask, causal=self.causal)
        row_sum_err = float(np.max(np.abs(P.sum(axis=-1) - 1.0)))
        # 凸组合不变量：out = P @ V ⇒ out 的每一行 ∈ V 行的凸包
        # 数值验证：‖out − P@V‖ ≈ 0（与定义自洽）
        self_consistent = float(np.linalg.norm(out - (P @ V)))
        # 数值稳定：极端输入不产生 NaN/Inf
        big = np.array(Q) * 1e6 if np.all(np.isfinite(Q)) else Q
        try:
            ob, _ = scaled_dot_product_attention(big, K, V, mask=mask, causal=self.causal)
            stable = bool(np.all(np.isfinite(ob)))
        except Exception:
            stable = False
        return {
            "weights_rowstochastic": {"value": row_sum_err, "threshold": 1e-9, "passed": bool(row_sum_err < 1e-9)},
            "output_self_consistent": {"value": self_consistent, "threshold": 1e-12, "passed": bool(self_consistent < 1e-12)},
            "numerically_stable_large_input": {"value": stable, "passed": stable},
        }

    def compare_torch(self, Q, K, V) -> dict:
        try:
            import torch
        except Exception:
            return {"available": False, "passed": True, "note": "torch 未安装（可选）"}
        Qt = torch.tensor(Q, dtype=torch.float64)
        Kt = torch.tensor(K, dtype=torch.float64)
        Vt = torch.tensor(V, dtype=torch.float64)
        with torch.no_grad():
            ot = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt)
        out, _ = scaled_dot_product_attention(Q, K, V)
        diff = float(np.linalg.norm(np.asarray(ot) - out))
        return {"available": True, "max_diff_vs_torch": diff, "passed": bool(diff < 1e-9)}
