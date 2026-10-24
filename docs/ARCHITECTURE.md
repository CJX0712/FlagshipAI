# 架构说明 · MorningStar FlagshipAI

> 作者：晨星 (MorningStar) · v1.0.0

---

## 一、十二层映射

| 层 | 模块 | 职责 | 世界级出处 |
| --- | --- | --- | --- |
| **L12** | `morningstar.verify` | 不变量 · 跨库交叉验证 · 基准 · 报告生成 | — （本项目原创） |
| **L11** | `morningstar.flash` | GPT 内核：RoPE / RMSNorm / SwiGLU / GQA / KV-Cache / BPE / AdamW | Vaswani 2017; Dao 2023; Su 2021; Zhang 2019; Shazeer 2020; Touvron 2023; Sennrich 2016; Loshchilov 2019 |
| **L10** | `morningstar.attention` | 注意力家族：full / flash / linear / performer / local / sparse / fuse | Vaswani 2017; Dao 2023; Katharopoulos 2020; Choromanski 2021; Beltagy 2020; Child 2019 |
| **L9** | `morningstar.core.deep` | MLP · 多头注意力 | Rumelhart 1986; Vaswani 2017 |
| **L8** | `morningstar.core.unsupervised.pca` | 主成分分析 | Tipping & Bishop 1999; Pearson 1901 |
| **L7** | `morningstar.core.unsupervised.kmeans` `gmm` | K-Means · 高斯混合 | Lloyd 1982; Dempster 1977 |
| **L6** | `morningstar.core.supervised.knn` `naive_bayes` | k-NN · 朴素贝叶斯 | Cover & Hart 1967; McCallum & Nigam 1998 |
| **L5** | `morningstar.core.supervised.svm` | 支持向量机 | Cortes & Vapnik 1995; Boser 1992 |
| **L4** | `morningstar.core.supervised.linear` `logistic` | 线性回归 · 逻辑回归 | Nelder & Wedderburn 1972; Verhulst 1838 |
| **L3** | `morningstar.core.mathx.optimize` + `flash.optim` | 梯度下降 · 梯度检验 · AdamW | Cauchy 1847; Loshchilov & Hutter 2019 |
| **L2** | `morningstar.core.mathx` | 概率统计 · 信息论 · logsumexp | Shannon 1948; Pearl 1988 |
| **L1** | `morningstar.core.mathx.linalg` | 线性代数：Cholesky / Jacobi / QR | Cholesky 1910; Jacobi 1846 |

---

## 二、模块依赖图

```
                        morningstar/
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   core/                attention/            verify/
   (L1–L9)               (L10)                (L12)
        │                    │                    │
        │                    │  依赖 flash.attention │
        │                    ▼                    │
        │              flash/attention.py         │
        │                    │                    │
        │                    ▼                    ▼
        └──────────────▶ flash/  ◀───────────────┘
                        (L11)
                     attention  optim  modules
                     model  trainer  tokenizer  random
```

**分层规则**（严格单向，禁止反向依赖）：

1. `core` **不依赖** `flash` / `attention` / `verify`——它是纯数学底座。
2. `flash` 依赖 `core`（可选）与 numpy；**不依赖** `attention` 家族层
   （它有自己的 `flash/attention.py` 内核实现）。
3. `attention` 依赖 `flash.attention` 的底层原语（`naive_attention_forward` /
   `flash_attention_forward` / `KVCache`）。
4. `verify` 依赖全部下层——它是**唯一**允许横向依赖的层，因为它的职责
   恰恰是交叉验证。

> **为何 `flash/attention.py` 与 `attention/__init__.py` 并存**：
> 前者是**内核原语**（给定 q/k/v 矩阵做一次注意力，被 GPT 模型内部调用）；
> 后者是**注意力家族策略层**（统一的 `attend()` 协议 + 基准 + 路由）。
> 两者职责不同，故分层而非合并。

---

## 三、核心数据流

### 3.1 训练（`engine="naive"`）

```
tokens ──make_batches──▶ (x, y)                     每样本独立前向
                             │
   idx (T,) ──▶ embed lookup ──▶ x (T, C)
                             │
        ┌────────────────────┴────────────────────┐
        │  每层:                                  │
        │   h1 = RMSNorm(x)                      │
        │   q,k,v = h1 @ W{q,k,v}                │
        │   qh,kh = RoPE(q,k, pos)               │
        │   P = softmax(qh·khᵀ/√dh + causal)     │  ← 物化 (T,T)，反向需要
        │   attn_out = (P·vh) @ Wo               │
        │   x2 = x + attn_out                    │
        │   h2 = RMSNorm(x2)                     │
        │   ffn = (Swish(h2@Wg) ⊙ h2@Wu) @ Wd    │
        │   x  = x2 + ffn                        │
        └────────────────────┬────────────────────┘
                             │
                    xf = RMSNorm(x)
                    logits = xf @ lm_head
                    loss = CrossEntropy(logits, targets)
                             │
                        backward()  ← 手写解析梯度
                             │
                        AdamW.step(params, grads)
```

**残差拓扑（反向的核心）**：

```
x_out = x2 + ffn,   x2 = x_in + attn_out

∂x_in = ∂x2 (残差直连)  +  ∂x2/∂attn_out · ∂attn_out/∂x_in
      = dx2             +  RMSNorm₁_bwd( dattn_out )
```

任何残差直连项的遗漏都会让梯度误差**随层数放大并反号**——这正是
[DESIGN.md §3.1](DESIGN.md#31-rope-逆旋转负索引回绕梯度检验-g1) 排查时的现象。

### 3.2 推理（`engine="flash"` + KV-Cache）

```
prefill:  forward(prompt, engine="flash", kv_cache_per_layer=caches, start_pos=0)
          → KV 缓存写入 prompt 全部位置的 K/V
          → 取最后一行 logits 采样第一个新 token

decode:   for t in range(max_new_tokens - 1):
              forward([last], engine="flash",
                      kv_cache_per_layer=caches, start_pos=pos)
              → 缓存追加 1 个位置的 K/V
              → 注意力对「缓存中全部位置 + 新位置」计算
              → 采样下一个 token
```

复杂度：prefill O(T²d)，每步 decode O(Td) —— **总代价 O(T²d) 但显存 O(T·d)**。

**位置对齐**（`start_pos`）：query 的绝对位置 = `start_pos + i`；
RoPE 用同一组位置旋转 Q 与 K，缓存中已存的 K 保持其原始旋转角度。

### 3.3 验证与基准

```
morningstar.verify
├── invariants.run_all()      16 条数学不变量（纯 numpy，无外部依赖）
├── crosscheck.run_all()      7 项 PyTorch 交叉验证（需 torch，缺失则标记跳过）
├── benchmark
│   ├── benchmark_attention() 7 变体 × 耗时/内存/保真度/检索率
│   ├── benchmark_memory_scaling()  naive O(T²) vs flash O(T)
│   └── check_determinism_reruns() 同 seed 重复运行
└── report
    ├── format_markdown_table()
    ├── render_verification_md()
    └── write_reports()  →  results/verification.json + results/VERIFICATION.md
```

---

## 四、扩展指南

### 加一个新的注意力变体

```python
from morningstar.attention import AttentionOutput

class MyAttention:
    name = "my_attention"
    def attend(self, q, k, v) -> AttentionOutput:
        out = ...          # (T, d)
        return AttentionOutput(out, None, cost_s, mem_peak_mb, self.name)
    def theoretical_complexity(self, T, d) -> str:
        return "O(...)"

# 注册
_REGISTRY["my_attention"] = MyAttention
```

协议（`AttentionMechanism` Protocol）要求：`name` 属性 + `attend()` +
`theoretical_complexity()`。注册后自动进入基准与 `list_variants()`。

### 加一条硬不变量

```python
from morningstar.verify.invariants import _res

def check_my_invariant(tol: float = 1e-9) -> dict:
    ...
    return _res("X9_my_invariant", value, tol, value < tol, "说明")

# 注册到 ALL_INVARIANTS
ALL_INVARIANTS["X9_my_invariant"] = check_my_invariant
```

`_res` 统一产出 `{name, value, threshold, passed, detail}`，自动被报告生成器
渲染成 Markdown 表格。

### 加一项跨库交叉验证

```python
from morningstar.verify.crosscheck import _res, _torch

def cross_my_op(seed: int = 0, tol: float = 1e-12) -> dict:
    torch = _torch()
    if torch is None:
        return _skip("X10_my_op")
    ...
    return _res("X10_my_op", diff, tol, diff < tol, "说明")
```

---

## 五、依赖策略

| 依赖 | 必需性 | 用途 |
| --- | --- | --- |
| `numpy` | **必需** | 全部数值计算 |
| `torch` | 可选 | 跨库交叉验证（X1–X7）；缺失时相关项标记 `available=False` 而非失败 |
| `scipy` | 可选 | 经典 ML 层的部分对照实验 |
| `scikit-learn` | 可选 | PCA / KMeans / GMM 的跨库对照 |
| `pytest` | 开发 | 单元测试 |

**设计原则**：核心功能零重依赖（只需 numpy），可选依赖只用于**验证**而非
**实现**——这样任何人都能在裸 numpy 环境下跑通完整的不变量套件。
