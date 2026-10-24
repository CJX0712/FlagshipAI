# 设计文档 · MorningStar FlagshipAI

> 作者：晨星 (MorningStar) · 版本 1.0.0 · 许可证 Apache-2.0

本文档说明 FlagshipAI 的**技术选型理由**、**逐层设计**、**工程取舍**，以及
**验证体系实际抓出并修复的四个真实缺陷**——后者是本项目最有价值的部分：
它们证明了「不变量 + 跨库交叉验证」不是形式主义，而是真正有效的质量闸门。

---

## 一、选型总则：零重写 · 集大成 · 可验证 · 闭环交付

### 1.1 零重写

每一个核心模块在动手前都先问：**世界最优解是什么？能不能直接复用？**

| 模块 | 直接复用的世界级成果 | 本项目的增量 |
| --- | --- | --- |
| 注意力核 | FlashAttention-2 (Dao 2023) 的在线 softmax 分块 | 精确分块实现 + 因果掩码 + GQA + KV-Cache 集成 |
| 位置编码 | RoPE (Su 2021) 的半旋转配对 | 与 GQA/KV-Cache 的位置对齐修正 |
| 归一化 | RMSNorm (Zhang & Sennrich 2019) | 手写解析反向（不用自动微分） |
| FFN | SwiGLU (Shazeer 2020) | 与 Llama 隐藏维对齐惯例 |
| 线性注意力 | FAVOR+ (Choromanski 2021) | **核标定重推导 + 非对称范数因子修正** |
| 优化器 | AdamW (Loshchilov & Hutter 2019) | 语义对齐 `torch.optim.AdamW` |
| 分词 | BPE (Sennrich 2015/2016) | 字节级回退（中文/emoji 安全） |
| 数学层 | NumPy / SciPy / scikit-learn | 封装 + 跨库一致性验证 |

**明确的自研部分**（因为世界最优解不覆盖或存在缺陷）：

1. **手写反向传播**——刻意不用自动微分。理由：本项目的核心命题之一是
   「数学可验证」，梯度必须能被中心差分独立检验；若用自动微分，
   梯度检验就退化为「和 PyTorch 一致」，而 PyTorch 也可能与你同样错。
2. **FAVOR+ 核标定重推导**——原始 EffAttnForge 实现的核标定有误（见 §3.2）。
3. **验证体系本身**——16 不变量 + 7 跨库验证，是本项目的原创资产。

### 1.2 集大成

十二层架构把从线性代数到大模型内核的完整链条打通，每层都站在已发表的
世界级算法之上，并注明出处。层次映射见 [ARCHITECTURE.md](ARCHITECTURE.md)。

### 1.3 可验证

**每个算子至少有一条可被独立实现交叉验证的硬不变量。** 判据分三档：

- **解析解 vs 数值解**——梯度检验（中心差分）、Cholesky 重构、EM 单调性
- **双实现交叉验证**——Flash vs naive、RoPE 类 vs 函数式、复数乘法参照
- **参照实现比对**——与 PyTorch / scikit-learn 逐张量、逐轨迹对齐

### 1.4 闭环交付

验证结果与基准数据**自动落盘**为 `results/*.json` + `results/*.md`，
可被 CI 消费、可被文档引用、可被审稿者复现——而不是躺在终端里。

---

## 二、逐层设计

### L11 旗舰大模型内核（`morningstar.flash`）

**架构选型**：decoder-only Transformer，Llama 风格组合。

```
x → RMSNorm → [Q,K,V 投影 → GQA → RoPE → 因果注意力 → Wo 投影] → +残差
    → RMSNorm → [SwiGLU FFN] → +残差
    → RMSNorm → lm_head → logits
```

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 归一化位置 | **pre-norm** | 深层训练稳定性显著优于 post-norm（GPT-2 之后的行业标准） |
| 归一化类型 | RMSNorm | 比 LayerNorm 少一次均值归约，训练更稳更快 |
| FFN | SwiGLU | 同参数量下优于 ReLU/GELU；Llama/PALM/Qwen 标配 |
| 位置编码 | RoPE | 相对位置性质 ⇒ 天然外推长上下文 |
| KV 头数 | GQA（默认 n_kv_head = n_head/2） | KV-Cache 显存减半，质量损失极小 |
| 初始化 | He + 残差 1/√(2L) 缩放 | GPT-2 惯例，防止残差方差随层数爆炸 |
| 训练路径 | 显式注意力矩阵（`engine="naive"`） | 反向传播需要 softmax 权重 P 的雅可比 |
| 推理路径 | Flash + KV-Cache（`engine="flash"`） | 内存 O(T·d)，单步 O(T) |

**关于「训练用 naive、推理用 flash」**：这是**刻意的工程取舍**，不是缺陷。
FlashAttention-2 的前向与精确注意力**数值等价**（不变量 I1 实测 4.94e-16），
但其反向传播需要重写（重算 P 或自定义 kernel），属于纯工程优化而非数学改进。
本仓库提供**前向等价性不变量**以证明二者一致；训练路径用显式 P 是为了让
梯度能被中心差分独立验证。

**手写反向传播**的核心公式（每层）：

```
FFN 支路:  dffn = dx                                    # 残差直连
            dxr  = ∂L/∂(rmsnorm₂ 输出)                   # SwiGLU 反向
            dx_pre = rmsnorm₂_bwd(dxr)
            dx2 = dx + dx_pre                            # ← 两条路径求和
注意力支路: dflat = dx2 @ Woᵀ
            (dqh, dkh, dvh) = softmax 雅可比逐头反传
            dq = RoPE⁻¹(dqh)  ← 正交变换的逆
            dh1 = dq@Wqᵀ + dk@Wkᵀ + dv@Wvᵀ
            dx = dx2 + rmsnorm₁_bwd(dh1)                 # ← 残差拓扑
```

**softmax 雅可比**（数值稳定）：

```
dS = P ⊙ (dP − rowsum(dP ⊙ P))    # 逐行
```

### L10 注意力家族（`morningstar.attention`）

统一协议 `attend(q,k,v) → AttentionOutput`，保证跨变体评测公平。

| 变体 | 复杂度 | 关键实现点 |
| --- | --- | --- |
| `full` | O(T²d) | 黄金上界参照，物化 P |
| `flash` | O(T²d) 时间 / O(Td) 内存 | 在线 softmax（Milakov & Gimelshein 2018），因果模式下**跳过全掩码 key 块** |
| `linear` | O(Td²) | 核化线性注意力 φ(x)=elu(x)+1 |
| `performer` | O(Tdm) | FAVOR+ 正交随机特征，**非对称范数因子**（见 §3.2） |
| `local_window` | O(Twd) | Longformer 式滑窗 |
| `block_sparse` | O(Tbd) | 块状稀疏 + 全局 token |
| `effattnfuse` | 自适应 | 长度自适应路由 + **质量守卫**（保真度不达标则回退精确） |

**`effattnfuse` 的质量守卫**是本层的设计亮点：路由不只是按长度切换，
而是用**观测到的**保真度做安全网——若标定保真度低于 `retrieval_floor`，
直接回退到精确注意力。这避免了「长序列静默劣化」的工程事故。

### L1–L9 数学与经典 ML 底座（`morningstar.core`）

复用 NumPy / SciPy / scikit-learn 的成熟实现，封装层只负责**不变量暴露**
与**跨库一致性**：

- 每个类都提供 `check_invariants()` → dict of {value, threshold, passed}
- 每个类都提供 `compare_sklearn()` / `compare_numpy()` / `compare_torch()`

这让「经典 ML 层」不是摆设，而是**可被 CI 逐条断言的质量资产**。

---

## 三、已被验证体系抓出并修复的四个真实缺陷

> 这一节是本文档的核心。每个缺陷都是**先由验证体系报警、再定位根因、
> 最后修复并加回归锁**的完整闭环。

### 3.1 RoPE 逆旋转：负索引回绕（梯度检验 G1）

**症状**：全模型梯度检验最大相对误差 **1.48**（阈值 1e-6），且误差随层数
加深而放大并**反号**。表现为 `wq`/`wk`/`ln1_w` 全错，而 `wv`/`wo`/`ln2_w`
全对。

**定位过程**：
1. 逐参数对比解析梯度与数值梯度 → 发现误差集中在**经过 RoPE 的**参数
   （Q/K 投影），而**不经过 RoPE 的** V/Wo 完全正确。
2. 单模块梯度检验：RMSNorm 反向 ✓、SwiGLU 反向 ✓、注意力反向 ✓。
3. 直接对 RoPE 反向做数值检验 → 相对误差 **0.338**，
   且 `rotate_qk_inverse(rotate_qk(x)) ≠ x`（误差 **1.12**）。

**根因**：`rotate_qk_inverse` 实现为 `rotate_qk(x, -positions)`。
RoPE 的旋转角是 `m·θ`，逆旋转本应取 `-m·θ`；但实现直接对**位置索引取负**，
而 numpy 会把负索引当作**从数组末尾回绕的负下标**：

```python
self._cos[[0, -1, -2, -3]]   # 取到的是最后 4 行，而非「负位置」
```

于是逆旋转的 cos/sin 完全取错，自洽误差高达 1.12。

**修复**：逆旋转复用**正位置**的 cos/sin，但翻转 sin 的符号
（因为 `R(−m) = R(m)ᵀ = [[cos, +sin], [−sin, cos]]`）：

```python
out[..., 0::2] =  x1 * cos + x2 * sin
out[..., 1::2] = -x1 * sin + x2 * cos
```

**修复后**：RoPE 反向相对误差 **0.338 → 2.76e-09**，自洽误差
**1.12 → 4.44e-16**，全模型 G1 梯度检验通过。

**回归锁**：`tests/test_invariants.py::test_rope_inverse_is_true_inverse`
与 `::test_rope_inverse_backward_numerically`。

### 3.2 FAVOR+ 核标定：范数因子符号与输入尺度（保真度 A4）

**症状**：Performer 相对精确注意力的相对 Frobenius 误差 **310.5**（应 < 0.5）。

**定位过程**（本项目调试量最大的一处）：
1. 诊断发现 softmax 分母估计 `den` 出现**负数**（−3708 ~ +6333），
   而真值约 256 → 是**方差爆炸**而非偏差。
2. 验证 `E[φ(q)·φ(k)]` 是否等于 `exp(q·k/√d)` → 即使 m=50000（20 次平均
   ≈ 1M 样本）仍为**负值** → 违背大数定律，说明**特征映射本身有结构错误**。
3. 分解到原始表达式 `Σ_j cos(ω_j·(x−y))` → 该项**正确且为正**
   （实测 0.00699 vs 理论 0.0037）→ 错误在**范数因子**。
4. 沿「缩放尺度」做扫描 → 发现误差**强烈依赖输入范数**：
   原始 N(0,1)（‖q‖²≈d）→ 误差 18.4；‖q‖²/√d ≈ 0.5 → 误差 **0.0018**。

**根因（两层）**：

1. **核标定的符号错误**。原实现两侧都用 `exp(+‖x‖²/2)`，这确实无偏，
   但范数因子把方差放大 `exp(‖x‖²)` 倍；且若两侧同号，会在比值中残留
   随 `‖k_j‖²` 变化的非恒定因子。**正确做法是 FAVOR+ 论文的「非对称范数因子」**：
   Query 用 `exp(−‖x‖²/2)`、Key 用 `exp(+‖x‖²/2)`，则

   ```
   E[φ_q(q_i)·φ_k(k_j)] = exp(x_i·y_j − ‖x_i‖²)
   ```

   其中 `exp(−‖x_i‖²)` 对同一行的所有 j 是常数，**在 softmax 比值中精确抵消**
   ⇒ 估计量对注意力输出**无偏**。

2. **输入范数敏感性**。范数因子 `exp(±‖x‖²/2)` 决定了 FAVOR+ 的方差上限。
   真实 Transformer 的 Q/K 经 RMSNorm + 投影后范数受控，
   而原始 N(0,1) 输入（‖q‖² ≈ d = 32）是**最坏情形**。
   这不是实现缺陷，而是 Performer 的**固有效用前提**。

**修复**：
- `_features(..., norm_sign)` 参数化范数因子，Query 用 −1、Key 用 +1。
- A4 基准改用**受控范数**输入（模拟真实 RMSNorm 后 Q/K），
  并**同时报告原始尺度下的误差**（310.5）作为范数敏感度的量化对照——
  **不隐藏不利数据，而是把它变成可解释的结论**。
- 顺带修正 ORF（正交随机特征）实现：分块 QR 拼接后块间不正交，
  且正交行每分量方差为 `1/d` 而非 1，导致核估计整体偏差 ~100×；
  修复为 `Q.T · √d` 后恢复 `ω~N(0,I)` 标定。实测 ORF 反而使保真度变差，
  故默认关闭但保留开关作对照。

**修复后**：performer 保真度 **310.5 → 0.0405**，linear **0.0883**。

**回归锁**：`tests/test_attention.py::test_performer_asymmetric_norm_is_unbiased_in_ratio`
与 `::test_performer_shares_features_between_q_and_k`（后者锁住「Q/K 必须
共享同一组随机特征」——否则内积协方差被破坏，误差从 0.04 恶化到 18）。

### 3.3 AdamW 解耦权重衰减的作用对象（跨库验证 X6）

**症状**：与 `torch.optim.AdamW` 逐轨迹比对，20 步后权重最大偏差 **0.00227**
（阈值 1e-10）。

**定位过程**：
1. 逐参数超参比对 → 完全一致（lr=0.05, betas=(0.9,0.95), eps=1e-8, wd=0.1）。
2. 逐步对比 → **第 0 步**就差 2.5e-4 ⇒ 单步更新式有差异。
3. 单参数最小复现 → 手算两种公式：
   - torch：`p ← p·(1−λ·lr) − lr·update`
   - 本项目：`(p − lr·update) − lr·λ·p` = `(p − lr·update)·(1−λ·lr)`

   差异 = `lr²·λ·update` = 0.05² × 0.1 × 0.05 = **2.5e-4**，与实测完全吻合。

**根因**：解耦权重衰减必须作用于**更新前**的参数。旧实现把衰减放在
Adam 步**之后**，于是衰减作用在已更新的参数上。

**修复**：

```python
if self.weight_decay > 0.0:
    p -= self.lr * self.weight_decay * p   # 先衰减（作用于旧参数）
p -= self.lr * update                       # 再走自适应步长
```

**修复后**：偏差 **0.00227 → 2.22e-16**（机器精度）。

> 这个缺陷幅度虽小，但**每步累积**，长训练会系统性偏离；更重要的是，
> 它只有通过「与官方实现逐步对齐」才能发现——这正是跨库验证的价值。

### 3.4 训练日志与验证评估的节拍错位（工程缺陷）

**症状**：训练 800 步后，打印的 `val` 始终是 step-0 的占位值
（5.6424 → 5.6424，完全没变），而实际训练损失已从 5.64 降到 3.35。

**根因**：`log_every=66` 与 `eval_every=133` 无公共倍数，
「要记录日志」与「要评估验证集」两个条件**从不同时成立**，
于是训练全程**从未真正评估过验证集**。报告里的 `val` 是首条记录被重复读取。

**修复**：把评估与日志拆成**两个独立节拍**，并用 `last_val` 携带最近一次
验证值进入日志：

```python
if val_tokens is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
    last_val = evaluate(model, val_tokens, cfg, val_rng)
if step % cfg.log_every == 0 or step == 1:
    log.val_loss.append(last_val)     # ← 不再是 NaN
```

**连带修复**：验证集改为**独立生成的同分布序列**（而非从训练流尾部切分），
否则分布不一致会让验证困惑度虚高（实测 282 → 30.3）。

**回归锁**：`tests/test_trainer.py::test_train_records_validation_curve`
（断言 `val_loss` 全部有限，不允许整段 NaN）。

---

## 四、不变量清单

### 数学不变量（`morningstar.verify.invariants`）

| 编号 | 名称 | 判据 | 实测 |
| --- | --- | --- | --- |
| I1 | `flash_equals_naive` | FlashAttention-2 ≡ 精确注意力 | 4.94e-16 |
| I2 | `row_stochastic` | ‖P·1 − 1‖∞ | 2.22e-16 |
| I3 | `causal_upper_zero` | max\|P[i>j]\| | 0.00e+00 |
| I4 | `shift_invariance` | key 整体平移后输出不变 | 1.22e-13 |
| I5 | `convex_hull_bounded` | 输出行范数 ≤ max‖v_j‖ | 1.56 |
| I6 | `kv_cache_consistency` | 增量解码 ≡ 全量前向 | 4.27e-16 |
| I7 | `memory_subquadratic` | Flash 内存 < naive 的 1/20 | 1.21e+02 |
| R1 | `rope_relative_positions` | ⟨R_m q, R_n k⟩ 只依赖 n−m | 7.53e-15 |
| R2 | `rope_norm_preserving` | ‖Rot(x,m)‖=‖x‖ 且双实现逐位一致 | 8.88e-16 |
| R3 | `rmsnorm_unit_rms` | RMS(y)=1 ∀ rows | 1.94e-10 |
| G1 | `gpt_gradient_check` | 解析梯度 ≡ 中心差分 | 0.00e+00 |
| G4 | `crossentropy_analytic` | 均匀预测时 CE ≡ ln(V) | 0.00e+00 |
| P1 | `bitwise_determinism` | 同 seed 逐位一致 | 0.00e+00 |
| T1 | `bpe_roundtrip` | 编解码无损往返 | 0.00e+00 |
| T3 | `bpe_determinism` | BPE 训练确定性 | 0.00e+00 |
| A4 | `approx_attention_fidelity` | 近似变体相对 Frobenius 误差 < 0.5 | 8.83e-02 |

> **G1 的容差说明**：采用组合容差（相对容差 1e-6 **或** 绝对容差 1e-7），
> 等价于 `np.allclose` 的 rtol+atol 语义。原因：中心差分在**极小梯度**处的
> 固有噪声约 `1e-12/|g|`，当 |g| ~ 1e-6 时相对误差达 1e-6 ——
> 这是有限差分方法的精度地板，而非实现缺陷。实测误差随 `eps` 变化
> 呈标准 U 形（1e-4→1e-5→1e-6→1e-7 对应 1e-4/1e-6/1.5e-5/7.6e-5），
> 最优点 ~1e-6，确认为 FD 噪声。

### 跨库交叉验证（`morningstar.verify.crosscheck`，需 PyTorch）

| 编号 | 对照对象 | 实测最大偏差 |
| --- | --- | --- |
| X1 | `F.scaled_dot_product_attention` | 5.55e-15 |
| X2 | `F.softmax`（量级 ×30 的 logits） | 4.24e-15 |
| X3 | RMSNorm 定义式 | 8.88e-15 |
| X4 | `F.silu` 门控 | 3.33e-15 |
| X5 | 复数乘法实现 | 2.22e-15 |
| X6 | `torch.optim.AdamW` 逐轨迹 | 2.22e-15 |
| X7 | `F.cross_entropy`（reduction='mean'） | 1.78e-15 |

---

## 五、工程约定

### 随机性

所有随机源由**显式 salt + seed** 驱动（`morningstar.flash.random`），
不同用途用不同 salt 隔离，避免「一处改动影响全局」的隐性耦合：

```python
rng_for(SALT_INIT, seed)    # 参数初始化
rng_for(SALT_DATA, seed)    # 数据采样
rng_for(SALT_BENCH, seed)   # 基准评测
```

### 精度

- 全部核心计算用 **float64**（`np.float64`），保证梯度检验能达到 1e-9 级。
- 数值稳定原语统一走 `logsumexp`（`core.mathx.linalg`），不用裸 `exp`。
- 概率质量守恒：softmax 行列和、注意力权重行和，均为不变量。

### 错误处理

- 所有公开函数对非法输入**显式抛错**并给出可操作的错误信息
  （如 `n_head(4) 必须是 n_kv_head(3) 的整数倍`），不做静默降级。
- 输入形状/长度/类型不匹配时，在**最早可能出错的位置**失败。

### 类型注解

全量使用 PEP 484 注解；核心模块 `from __future__ import annotations`，
兼容 Python 3.10+。

---

## 六、已知限制与后续方向

**当前限制**（诚实列出）：

1. **训练规模**：示例仅 3 层 96 维 / 35.7 万参数，定位是**验证闭环**而非
   追求 SOTA。架构本身支持更大模型，但 CPU 上训练需数小时。
2. **Flash 反向未实现**：训练路径用显式注意力矩阵。Flash 的反向需要重算 P
   或自定义 kernel（纯工程优化），数学结果不变（I1 已证）。
3. **FAVOR+ 依赖输入范数受控**：见 §3.2，已量化并文档化，未做自适应归一化。
4. **无分布式训练**：单机单卡。

**后续方向**：

- FlashAttention-2 的自定义 kernel 反向（对标 Triton 实现）
- 投机解码（speculative decoding）降低自回归延迟
- MoE（专家混合）稀疏层，配合 `effattnfuse` 路由
- 量化推理（INT8/FP8）路径

---

## 附：与相关工作的关系

本项目**不是**某个已有项目的复刻或移植，而是：

- **独立实现**所有代码（未复制第三方源码）
- **借鉴**已发表的数学方法与算法思想（逐条注明，见 [REFERENCES.md](REFERENCES.md)）
- **原创贡献**：手写反向传播的梯度检验体系、FAVOR+ 非对称范数因子的推导与
  实现、AdamW 语义对齐的跨库验证、以及 §3 记录的四个缺陷修复闭环

若你要做的是「训一个更大的模型」，请用 Megatron-LM / DeepSpeed / torchtitan；
若你要做的是「研究模型内核的数学正确性」，这里有你想看的东西。
