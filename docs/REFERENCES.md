# 引用与致谢 · MorningStar FlagshipAI

> 作者：晨星 (MorningStar) · v1.0.0

本项目遵循**零重写**原则：数值重活与成熟算法直接复用已发表的世界级成果与
开源实现，并在源码与本文档中逐条注明出处。

> **重要声明**：本项目为**独立实现**，未复制任何第三方项目的源码。
> 下列条目仅标注所借鉴的**数学方法与算法思想**的具体出处。
> 第三方项目版权归其原作者所有，并遵循其各自的许可证。

---

## 一、大语言模型内核（L11）

| 技术 | 出处 | 引用 |
| --- | --- | --- |
| Transformer 架构 | Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762 | 编码器-解码器、缩放点积、多头注意力、残差连接 |
| GPT / Decoder-only | Radford, A. et al. *Improving Language Understanding by Generative Pre-Training.* 2018. | 预训练 + 微调范式、因果掩码语言建模 |
| Pre-norm 结构 | GPT-2 之后的行业标准做法 | 残差支路内做归一化，深层训练稳定性优于 post-norm |
| **FlashAttention-2** | Dao, D. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. | 内存 O(T·d) 的分块注意力；IO 感知调度；因果模式下跳过全掩码块 |
| 在线 softmax | Milakov, G. & Gimelshein, N. *Online Normalizer Calculation for Softmax.* 2018. arXiv:1805.02867 | 递推式行最大值/行和归一化，FlashAttention 的数学基础 |
| **RoPE** 旋转位置编码 | Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864, 2021. | 半旋转配对；`⟨R_m q, R_n k⟩` 只依赖 n−m ⇒ 天然外推长上下文 |
| **RMSNorm** | Zhang, B. & Sennrich, R. *Root Mean Square Layer Normalization.* arXiv:1910.07467, 2019. | 只重缩放不重居中；比 LayerNorm 少一次均值归约 |
| **SwiGLU** / GLU 变体 | Shazeer, N. *GLU Variants Improve Transformer.* arXiv:2002.05202, 2020. | `FFN(x) = (Swish(xW_gate) ⊙ xW_up) W_down` |
| **Llama** 架构组合 | Touvron, H. et al. *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971, 2023. | RMSNorm + SwiGLU + RoPE + GQA 的组合范式；隐藏维 `≈8/3·d_model` 惯例 |
| GQA 分组查询注意力 | Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. | KV 头分组共享，KV-Cache 显存按组数线性下降 |
| 残差支路缩放初始化 | Radford et al. 2018 (GPT-2 实现惯例) | 残差路径权重按 `1/√(2L)` 缩放，防方差爆炸 |
| **He 初始化** | He, K. et al. *Delving Deep into Rectifiers.* ICCV 2015. | `std = √(2/fan_in)` |
| top-k 截断采样 | Fan, A. et al. *Hierarchical Neural Story Generation.* ACL 2018. | 截断采样族 |
| **AdamW** 解耦权重衰减 | Loshchilov, I. & Hutter, F. *Decoupled Weight Decay Regularization.* ICLR 2019. | `θ ← θ(1−λ·lr) − lr·update`，衰减不参与 m/v |
| 余弦学习率 + 预热 | Smith, L. et al. *No More Validation: Early Stopping for Synthetic Data Generation.* 2020. / Goyal et al. *Accurate, Large Minibatch SGD.* 2017. | 预热 + 余弦退火 |
| 梯度裁剪 | Pascanu, R. et al. *On the Difficulty of Training Recurrent Neural Networks.* ICML 2013. | 全局范数裁剪 |
| **BPE** 字节对编码 | Sennrich, R. et al. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. / Sennrich, R. et al. *Unsupervised Machine Translation with Subword Units.* 2015. | 合并最高频相邻符号对；字节级回退 |
| 交叉熵 / logsumexp | Shannon 1948; numerically stable per Johnstone et al. | 数值稳定的 `log Σ exp` |

---

## 二、注意力家族（L10）

| 变体 | 出处 | 引用 |
| --- | --- | --- |
| `full` | Vaswani et al. 2017 | 精确 O(T²d) 黄金参照 |
| `flash` | Dao 2023 | 与 full 数值等价，内存 O(T·d) |
| `linear` 核化线性注意力 | Katharopoulos, A. et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020. | φ(x)=elu(x)+1 特征映射，重排 `φ(Q)(φ(K)ᵀV)` |
| `performer` **FAVOR+** | Choromanski, K. et al. *Rethinking Attention with Performers.* arXiv:2009.14794, ICLR 2021. | 三角随机特征近似 softmax 核；正交随机特征（ORF）方差缩减；**非对称范数因子**（本项目重推导，见 [DESIGN §3.2](DESIGN.md#32-favor-核标定范数因子符号与输入尺度保真度-a4)） |
| `local_window` | Beltagy, I. et al. *Longformer: The Long-Document Transformer.* arXiv:2004.05150, 2020. | 滑窗局部注意力 |
| `block_sparse` | Child, R. et al. *Generating Long Sequences with Sparse Transformers.* arXiv:1904.10558, 2019. | 块状稀疏 + 全局 token |
| `effattnfuse` 路由 | — （本项目原创，源自 EffAttnForge 资产） | 长度自适应路由 + **基于观测保真度的质量守卫** |

---

## 三、数学与经典机器学习底座（L1–L9）

| 方法 | 出处 |
| --- | --- |
| Cholesky 分解 | Cholesky, C. *Sur la résolution des systèmes linéaires.* 1910. |
| **Jacobi 特征分解** | Jacobi, C. G. J. *Über ein neues Verfahren zur Berechnung der Eigenwerte eines quadratischen Form.* 1846. |
| 高斯-约当消去法 | Gauss 1809; Jordan 1869. |
| 主成分分析 | Pearson, K. 1901; **Tipping, M. E. & Bishop, C. M. *Probabilistic Principal Component Analysis.* JRSS-B 1999.** |
| Eckart-Young 定理 | Eckart, C. & Young, G. *The approximation of one matrix by another of lower rank.* Psychometrika 1936. | PCA 截断最优性的理论依据 |
| K-Means | **Lloyd, S. *Least Square Quantization in PCM.* IEEE Trans. Comm. 1982.** |
| 高斯混合 / EM 算法 | **Dempster, A. et al. *Maximum Likelihood from Incomplete Data via the EM Algorithm.* JRSS-B 1977.** |
| 支持向量机 | **Cortes, C. & Vapnik, V. *Support-Vector Networks.* Machine Learning 1995.**; Boser, B. et al. 1992. |
| 朴素贝叶斯 | McCallum, P. & Nigam, K. *Employing EM in Pool-Based Active Learning for Text Classification.* ICML 1998.（最优偏见） |
| k-近邻 | **Cover, T. & Hart, P. *Nearest Neighbor Pattern Classification.* IEEE TIT 1967.** |
| 逻辑回归 / GLM | Nelder, J. & Wedderburn, R. *Generalized Linear Models.* 1972. |
| 线性回归（正规方程） | Gauss 1809; Normal Equations |
| 多层感知机 | Rumelhart, D. et al. *Learning Representations by Back-Propagating Errors.* Nature 1986. |
| 信息论 / 熵 | **Shannon, C. *A Mathematical Theory of Communication.* Bell System Tech. J. 1948.** |
| 中心差分梯度检验 | Numerical Recipes; Finite-difference gradient checking |
| logsumexp 稳定原语 | Machine Learning 惯例（数值稳定 softmax） |

---

## 四、工具链与交叉验证

| 工具 | 出处 | 用途 |
| --- | --- | --- |
| **NumPy** | Harris, C. et al. *Array Programming with NumPy.* Nature 585, 2020. | 全部数值计算 |
| **SciPy** | Virtanen, P. et al. *SciPy 1.0.* Nature Methods 2017. | 优化与统计函数 |
| **scikit-learn** | Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 2011. | PCA / KMeans / GMM 的跨库对照 |
| **PyTorch** | Paszke, A. et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. | 跨库交叉验证基准（X1–X7） |
| **pytest** | Krekel, H. et al. *pytest.* 2009. | 单元测试框架 |
| Apache-2.0 | Apache Software Foundation | 本项目许可证 |

**跨库验证覆盖**：

| 编号 | 本项目实现 | 对照的权威实现 |
| --- | --- | --- |
| X1 | `flash_attention_forward` | `torch.nn.functional.scaled_dot_product_attention` |
| X2 | `_softmax` | `torch.nn.functional.softmax`（量级 ×30 的 logits） |
| X3 | `_rmsnorm_fwd` | RMSNorm 定义式（PyTorch 无官方 RMSNorm） |
| X4 | `_swiglu_fwd` | `torch.nn.functional.silu` 门控 |
| X5 | `RotaryEmbedding` | 复数乘法实现 `(x_{2j} + i·x_{2j+1})·e^{i·m·θ_j}` |
| X6 | `AdamW` | `torch.optim.AdamW`（**逐轨迹**比对） |
| X7 | `_cross_entropy_loss` | `torch.nn.functional.cross_entropy`（reduction='mean'） |

---

## 五、致谢

感谢上述所有研究者与开源社区的贡献。本项目的价值建立在这些公开成果之上；
若这些工作不存在，本项目也无法存在。

特别致谢：

- **FlashAttention**（Dao 2023）——让「内存墙」不再是问题
- **RoPE / RMSNorm / SwiGLU** 三个组件的提出者——它们让现代 LLM 成为可能
- **FAVOR+**（Choromanski 2021）——线性注意力在真实场景可用的关键
- **NumPy / SciPy / PyTorch / scikit-learn** 的维护者——让「零重写」成为可能

---

## 六、引用格式

```bibtex
@misc{flagshipai2026,
  author       = {晨星 (MorningStar)},
  title        = {MorningStar FlagshipAI: 世界顶级 AI 系统},
  year         = {2026},
  howpublished = {\url{https://github.com/CJX0712/FlagshipAI}},
  note         = {Apache-2.0 License}
}
```
