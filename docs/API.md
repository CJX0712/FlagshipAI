# API 使用说明 · MorningStar FlagshipAI

> 作者：晨星 (MorningStar) · v1.0.0

全部公开接口速查。约定：`np` = numpy，`torch` = PyTorch（仅跨库验证需装）。

```python
import numpy as np
from morningstar.flash import GPT, GPTConfig
from morningstar.attention import build_variant, list_variants
from morningstar.verify import verify_suite, run_full_verification
```

---

## 一、`morningstar.flash` — 旗舰大模型内核

### 1.1 `GPTConfig`

```python
@dataclass
class GPTConfig:
    vocab_size: int = 512
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2        # GQA：KV 头数，必须整除 n_head
    max_len: int = 256
    d_ff: int | None = None   # None ⇒ 按 Llama 惯例自动推导 (≈ 8/3·d_model)
    rope_base: float = 10000.0
    dropout: float = 0.0
    eps: float = 1e-8
```

**约束**（违反时抛 `ValueError`）：

- `d_model % n_head == 0`
- `n_head % n_kv_head == 0`
- `d_ff` 为 `None` 时自动取 `⌈(2/3)·4·d_model⌉`（Llama 惯例）

**派生属性**：

| 属性 | 含义 |
| --- | --- |
| `head_dim` | `d_model // n_head` |
| `to_dict()` | 序列化为 dict（含 `head_dim`） |

### 1.2 `GPT`

```python
model = GPT(cfg: GPTConfig, seed: int | None = None)
```

| 成员 | 说明 |
| --- | --- |
| `params: dict[str, np.ndarray]` | 参数名 → 数组。直接喂 `AdamW.step`、与 PyTorch `state_dict` 对齐 |
| `n_params: int` | 参数量 |
| `rotary: RotaryEmbedding` | RoPE 模块 |
| `save(path)` / `GPT.load(path)` | 存为 `.npz`（含配置），往返逐位一致 |
| `forward(idx, targets=None, *, engine="naive", kv_cache_per_layer=None, start_pos=0)` | 返回 `(logits, cache)` |
| `backward(idx, cache)` | 返回 `{参数名: 梯度}` |
| `loss_and_grad(idx, targets)` | **推荐入口**：返回 `(loss, grads)` |
| `generate(prompt_ids, max_new_tokens, *, temperature=1.0, top_k=None, seed=None, engine="flash", stream=False)` | 自回归生成 |

#### `forward` 细节

```python
logits, cache = model.forward(idx)                # idx: (T,) int64
logits, cache = model.forward(idx, targets)       # 同时返回交叉熵 loss
logits, cache = model.forward(idx, engine="flash")# 省内存路径
```

- `engine="naive"`（默认，训练用）：物化注意力矩阵，反向需要 P 的雅可比。
- `engine="flash"`（推理用）：分块在线 softmax，内存 O(T·d)。**与 naive 前向数值等价**（不变量 I1）。
- `cache` 内容：`{"T", "layers", "start_pos", "logits", "targets", "loss", "x_final", "x_final_raw", "inv_f"}`；
  `cache["layers"][i]` 含 `{"h1","inv1","x_in","attn","h2","inv2","ffn","x_pre"}`。

**异常**：

| 情形 | 异常 |
| --- | --- |
| `idx` 非一维 | `ValueError("idx 必须是一维序列…")` |
| `T > max_len` | `ValueError("序列长度 T 超过 max_len=…")` |
| `targets` 形状不符 | `ValueError("targets 形状 … 与 idx … 不一致")` |

#### `generate` 细节

```python
ids = model.generate([1, 2, 3], max_new_tokens=64, temperature=0.8, seed=42)
stream = model.generate([1, 2, 3], max_new_tokens=64, stream=True)  # 逐 token 生成器
```

- **返回的是新生成的 token**（不含 prompt）。
- `temperature=0.0` ⇒ 贪心（argmax）。
- `top_k` ⇒ top-k 截断采样（Fan et al. 2018）。
- `seed` ⇒ 可复现采样（同 seed 输出逐位一致）。
- 内部使用 **KV-Cache**，总代价从 O(T²) 降到 O(T)。
- `len(prompt) + max_new_tokens > max_len` ⇒ `ValueError`。

### 1.3 `RotaryEmbedding`

```python
rot = RotaryEmbedding(head_dim, base=10000.0, max_len=8192)
rot.rotate_qk(x, positions)          # (..., head_dim)，positions 需与前导维广播兼容
rot.rotate_qk_inverse(x, positions)  # 严格逆变换：rotate_qk_inverse(rotate_qk(x)) == x
rot.rotate(q, k, positions)          # 同时处理 Q 与 K
```

> ⚠️ `rotate_qk_inverse` **不可**用「位置取负」实现——numpy 负索引会回绕。
> 正确做法见 [DESIGN.md §3.1](DESIGN.md#31-rope-逆旋转负索引回绕梯度检验-g1)。

### 1.4 `AdamW`

```python
opt = AdamW(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
opt.set_lr(lr)                       # 配合调度器
opt.step(model.params, grads)        # 原地更新
```

更新式（与 `torch.optim.AdamW` 逐轨迹一致，X6 实测 2.22e-16）：

```
m ← β₁m + (1−β₁)g ;  v ← β₂v + (1−β₂)g²
m̂ = m/(1−β₁ᵗ) ;  v̂ = v/(1−β₂ᵗ)
θ ← θ·(1 − λ·lr) − lr·[ m̂/(√v̂+ε) ]
```

**约束**：`lr > 0`、`eps > 0`、`0 ≤ β < 1`、`weight_decay ≥ 0`（违反抛 `ValueError`）。

### 1.5 `BPETokenizer`

```python
tok = BPETokenizer().train(texts, vocab_size=300)
ids = tok.encode("晨星 FlagshipAI")
text = tok.decode(ids)               # 无损往返
tok.save("tok.json"); tok = BPETokenizer.load("tok.json")
tok.encode_batch(["a", "b"])
tok.vocab_size
```

- `vocab_size` 必须 ≥ 258（256 字节 + 至少 2 次合并）。
- 未见字符回退到**字节级**，保证中文 / emoji 无损。
- 同语料训练结果逐位一致（不变量 T3）。

辅助函数：`compression_ratio(tokenizer, texts)`（UTF-8 字节/token）、
`token_frequencies(texts, tokenizer)`。

### 1.6 训练器

```python
from morningstar.flash.trainer import TrainConfig, train, evaluate, make_batches

cfg = TrainConfig(n_steps=800, batch_size=16, block_size=32, lr=3e-3,
                  warmup=80, eval_every=133, eval_iters=5, log_every=66, seed=42)
log = train(model, train_tokens, cfg, val_tokens=val_tokens)
```

`TrainLog` 字段：`steps`、`train_loss`、`val_loss`、`lr`、`grad_norm`、`wall_s`。
`save_log(log, path)` 落盘 JSON；`perplexity(loss)` 换算困惑度。

> `val_loss` 记录的是**最近一次评估**的值（阶梯曲线），评估节拍由
> `eval_every` 独立控制，与 `log_every` 无关。

---

## 二、`morningstar.attention` — 注意力家族

统一协议：

```python
@dataclass
class AttentionOutput:
    output: np.ndarray         # (T, d)
    weights: np.ndarray|None  # 物化的 (T,T) 权重；非物化变体为 None
    cost_s: float             # 墙钟秒
    mem_peak_mb: float        # 解析式峰值内存
    variant: str
```

```python
from morningstar.attention import build_variant, list_variants

list_variants()
# ['block_sparse', 'effattnfuse', 'flash', 'full', 'linear',
#  'local_window', 'performer']

att = build_variant("performer")
res = att.attend(q, k, v)
res.output, res.cost_s, res.mem_peak_mb, att.theoretical_complexity(T, d)
```

| 变体 | 构造参数 | 复杂度 |
| --- | --- | --- |
| `FullAttention()` | — | O(T²d) |
| `FlashAttention()` | `block_q=64, block_k=64` | O(T²d) 时间 / O(Td) 内存 |
| `LinearAttention()` | — | O(Td²) |
| `PerformerAttention()` | `m=256, salt=7777, seed=None, orthogonal=False` | O(Tdm) |
| `LocalWindowAttention()` | `window=32` | O(Twd) |
| `BlockSparseAttention()` | `block_size=64, n_global=8` | O(Tbd) |
| `EffAttnFuse()` | `full_threshold=128, performer_m_short=128, performer_m_long=256, retrieval_floor=0.98, calibration=None` | 自适应 |

**`EffAttnFuse` 路由规则**：

```
T ≤ full_threshold                        → FullAttention
T > full_threshold 且标定保真度 ≥ floor   → Performer（按长度选 m）
否则                                       → FullAttention（质量守卫回退）
```

`build_variant("unknown")` ⇒ `KeyError`（消息含可选列表）。

---

## 三、`morningstar.verify` — 验证与基准

```python
from morningstar.verify import verify_suite, run_full_verification
from morningstar.verify.invariants import run_all as run_invariants
from morningstar.verify.crosscheck import run_all as run_crosschecks
from morningstar.verify.benchmark import benchmark_attention, benchmark_memory_scaling
from morningstar.verify.report import write_reports, format_markdown_table
```

| 接口 | 返回 |
| --- | --- |
| `invariants.run_all()` | `list[dict]`，每项 `{name, value, threshold, passed, detail}` |
| `crosscheck.run_all()` | `list[dict]`（PyTorch 不可用时标记 `available=False`） |
| `verify_suite()` | `{"timestamp","env","invariants","crosschecks","summary"}` |
| `run_full_verification(include_benchmarks=True, lengths=..., d=32)` | 追加 `benchmarks` 键 |
| `benchmark_attention(lengths, d, seed, repeats)` | 每变体每行的耗时/内存/保真度/检索率 |
| `benchmark_memory_scaling(lengths, d)` | naive O(T²) vs flash O(T) 的内存对比 |
| `write_reports(report, outdir="results")` | 落盘 `verification.json` + `VERIFICATION.md` |

### 命令行

```bash
python -m morningstar.verify.report   --out results [--no-bench]
python -m morningstar.verify.benchmark --out results --lengths 256 512 1024 2048
```

退出码：`report` 全绿返回 0，否则返回 1（可直接用于 CI 闸门）。

---

## 四、`morningstar.core` — 数学与经典 ML 底座

每个类都提供 `check_invariants()` 与跨库对比方法。

```python
from morningstar.core.unsupervised.pca import PCA
from morningstar.core.unsupervised.kmeans import KMeans
from morningstar.core.unsupervised.gmm import GaussianMixture
from morningstar.core.supervised.linear import LinearRegression
from morningstar.core.deep.neural_net import MLP
from morningstar.core.mathx import linalg, optimize

p = PCA(n_components=3).fit(X)
p.check_invariants()          # → {name: {value, threshold, passed}}
p.compare_sklearn(X)          # → {recon_error_ours, recon_error_sklearn, passed, …}

km = KMeans(n_clusters=3, seed=0).fit(X)
km.compare_sklearn(X)         # → 含 ARI_vs_sklearn 与 inertia_rel_diff
```

| 模块 | 主要 API |
| --- | --- |
| `core.mathx.linalg` | `cholesky` `cholesky_inverse` `jacobi_eigen` `logsumexp` `softmax` `log_softmax` `orthogonality_error` `eigen_residual` `is_symmetric_psd` |
| `core.mathx.optimize` | `numerical_gradient` `gradient_check` `gradient_descent` `check_monotone_decrease` |
| `core.supervised.linear` | `LinearRegression` |
| `core.supervised.logistic` | `LogisticRegression` |
| `core.supervised.svm` | `SVM` |
| `core.supervised.knn` | `KNN` |
| `core.supervised.naive_bayes` | `NaiveBayes` |
| `core.unsupervised.pca` | `PCA` |
| `core.unsupervised.kmeans` | `KMeans` |
| `core.unsupervised.gmm` | `GaussianMixture` |
| `core.deep.neural_net` | `MLP` |
| `core.deep.attention` | `scaled_dot_product_attention` `Attention` |
| `core.benchmarks.run` | 基准编排 |

> `MLP.predict` 返回**一维**数组；`fit` 接受 `(n,)` 或 `(n,1)`。

---

## 五、常见问题

**Q：`forward(kv_cache_per_layer=[None, None])` 为什么和全量前向不一致？**
A：元素为 `None` 表示该层**不使用缓存**，每步只能看到自己。增量解码必须传
真正的 `KVCache` 实例：

```python
from morningstar.flash.attention import KVCache
caches = [KVCache(cfg.max_len, cfg.n_kv_head, cfg.head_dim) for _ in range(cfg.n_layer)]
for t, tok in enumerate(ids):
    logits, _ = model.forward([tok], engine="flash", kv_cache_per_layer=caches, start_pos=t)
```

**Q：为什么训练用 `engine="naive"` 而不是 flash？**
A：反向传播需要注意力权重 P 的雅可比。前向两者数值等价（不变量 I1，
误差 4.94e-16），但 flash 的反向需要重算 P 或自定义 kernel，属纯工程优化。

**Q：FAVOR+ / Performer 为什么在原始 N(0,1) 输入上误差很大？**
A：范数因子 `exp(±‖x‖²/2)` 把估计方差放大 `exp(‖x‖²)` 倍。这是 Performer 的
固有效用前提——真实 Transformer 的 Q/K 经 RMSNorm 后范数受控。
详见 [DESIGN.md §3.2](DESIGN.md#32-favor-核标定范数因子符号与输入尺度保真度-a4)。

**Q：如何复现某次实验？**
A：所有随机源由显式 `seed` 驱动，同 seed 下初始化与前向输出逐位一致
（不变量 P1）。
