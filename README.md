<div align="center">

# 🌟 MorningStar FlagshipAI

**世界顶级 AI 系统 · 零重写 · 集大成 · 可验证 · 闭环交付**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-94%20passed-success.svg)](tests)
[![Invariants](https://img.shields.io/badge/invariants-16%2F16-success.svg)](docs/DESIGN.md)
[![Crosscheck](https://img.shields.io/badge/crosscheck-7%2F7%20vs%20PyTorch-success.svg)](docs/DESIGN.md)

**作者：晨星 (MorningStar)**

[快速开始](#快速开始) · [架构](#十二层架构) · [复现](#完整复现) · [性能](#性能对比) · [文档](#文档索引) · [引用](#引用与致谢)

</div>

---

## 这是什么

**FlagshipAI** 是一个**可运行、可复现、可验证**的现代大语言模型系统。

它不是「又一个 Transformer 教程」，而是一套**工程上完整闭环**的 AI 系统：

| 维度 | 交付内容 |
| --- | --- |
| **算法** | 全部站在已发表的世界级成果之上（FlashAttention-2 / RoPE / RMSNorm / SwiGLU / GQA / FAVOR+ / AdamW / BPE），逐条注明出处，**零重写** |
| **模型** | 完整 decoder-only GPT：pre-norm + RoPE + RMSNorm + SwiGLU + GQA + KV-Cache，**手写反向传播**（非自动微分） |
| **验证** | **16 条数学硬不变量** + **7 项 PyTorch 跨库交叉验证**，全部通过 |
| **测试** | **94 项单元测试**全绿 |
| **训练** | 真实收敛：loss 5.64 → 3.35，验证集同步下降，采样能复现中文模式 |
| **性能** | 注意力家族 7 变体的耗时 / 内存 / 保真度三维量化对比 |

### 与「从零实现」的三个本质区别

1. **不重写轮子**：FlashAttention-2 的在线 softmax、FAVOR+ 的正交随机特征、AdamW 的解耦衰减——全部按论文原始公式实现并注明出处，附**逐条纠错记录**（见 [docs/DESIGN.md](docs/DESIGN.md#三已被验证体系抓出并修复的四个真实缺陷)）。
2. **不是「看起来对」**：每个算子都有一条**可被独立实现交叉验证的硬不变量**。梯度必须通过中心差分检验，注意力必须与 PyTorch 逐位一致，优化器轨迹必须与 `torch.optim.AdamW` 对齐到机器精度。
3. **不是「跑通就行」**：一键复现脚本 + CI 四版本矩阵 + 自动生成的验证报告 + 基准对比表，全部落盘为可审计产物。

---

## 快速开始

### 安装

```bash
git clone https://github.com/CJX0712/FlagshipAI.git
cd FlagshipAI

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                              # dev 含 torch / pytest / scipy / sklearn
```

> 核心功能仅依赖 **numpy**；`torch` 仅用于跨库交叉验证，`scipy`/`sklearn` 仅用于经典机器学习层的对照实验。

### 一键复现

```bash
make all      # = install + test + verify + bench + train
```

或分步执行：

```bash
make test     # 94 项单元测试
make verify   # 16 不变量 + 7 跨库验证 → results/VERIFICATION.md
make bench    # 注意力基准 → results/BENCHMARK.md
make train    # 旗舰 GPT 训练 800 步 → results/train_summary.json
```

### 三行代码用起来

```python
from morningstar.flash import GPT, GPTConfig

model = GPT(GPTConfig(vocab_size=512, d_model=128, n_layer=4,
                      n_head=4, n_kv_head=2), seed=42)

out = model.generate([1, 2, 3], max_new_tokens=64, temperature=0.8, seed=42)
print(out)          # 增量解码，带 KV-Cache，内存 O(T·d)
```

---

## 十二层架构

```
┌─────────────────────────────────────────────────────────────────┐
│ L12 集成验证与交付   morningstar.verify                          │
│     16 不变量 · 7 跨库交叉验证 · 基准 · 报告生成                  │
├─────────────────────────────────────────────────────────────────┤
│ L11 大模型内核     morningstar.flash                             │
│     GPT · RoPE · RMSNorm · SwiGLU · GQA · KV-Cache · BPE · AdamW  │
│     ★ 手写反向传播（无自动微分）                                  │
├─────────────────────────────────────────────────────────────────┤
│ L10 高效注意力     morningstar.attention                         │
│     full · flash · linear · performer · local_window ·           │
│     block_sparse · effattnfuse（长度自适应路由 + 质量守卫）       │
├─────────────────────────────────────────────────────────────────┤
│ L9  神经网络       morningstar.core.deep                        │
│ L8  降维           morningstar.core.unsupervised.pca             │
│ L7  聚类           morningstar.core.unsupervised.kmeans / gmm    │
│ L6  概率与非参      morningstar.core.supervised.knn / naive_bayes │
│ L5  核方法与 SVM   morningstar.core.supervised.svm              │
│ L4  线性/广义模型   morningstar.core.supervised.linear / logistic│
│ L3  微积分与优化    morningstar.core.mathx.optimize              │
│ L2  概率统计        morningstar.core.mathx                       │
│ L1  线性代数       morningstar.core.mathx.linalg                │
└─────────────────────────────────────────────────────────────────┘
```

每一层的算法出处见 [docs/REFERENCES.md](docs/REFERENCES.md)，设计取舍见 [docs/DESIGN.md](docs/DESIGN.md)。

### 目录结构

```
FlagshipAI/
├── morningstar/
│   ├── core/           # L1–L9  数学与经典 ML 底座
│   ├── flash/          # L11 旗舰大模型内核
│   ├── attention/      # L10 注意力家族
│   └── verify/         # L12 不变量 · 跨库 · 基准 · 报告
├── tests/              # 94 项单元测试
├── examples/           # 训练 / 生成 / 基准 示例
├── docs/               # 设计 / API / 架构 / 引用 / 基准
├── results/            # 自动生成的验证与基准产物
├── scripts/            # 一键脚本
├── Makefile            # 复现流水线
├── pyproject.toml
├── LICENSE             # Apache-2.0
└── NOTICE              # 第三方算法出处清单
```

---

## 完整复现

```bash
# 1) 环境
make venv && make dev

# 2) 单元测试（94 项）
make test
# → 100% passed

# 3) 数学不变量 + PyTorch 跨库交叉验证
make verify
# → 不变量 16/16 · 跨库 7/7 · 全绿=True
# → results/verification.json + results/VERIFICATION.md

# 4) 注意力基准
make bench
# → results/benchmark.json + results/BENCHMARK.md

# 5) 训练闭环
make train
# → train 5.64 → 3.35 | val 5.63 → 3.41 | 采样输出可复现中文模式
# → results/train_log.json + results/train_summary.json
```

**可复现性保证**：初始化、数据采样、生成采样全部由显式 `seed` 驱动；同 seed 下初始化与前向输出**逐位一致**（不变量 P1）。

---

## 性能对比

### 注意力家族三维对比

> 数据由 `make bench` 在本机自动生成，见 [results/BENCHMARK.md](results/BENCHMARK.md)。

| 变体 | 复杂度 | 内存 | 保真度 | 出处 |
| --- | --- | --- | --- | --- |
| `full` | O(T²d) | O(T²) | 1.000（基准） | Vaswani et al. 2017 |
| `flash` | O(T²d) 时间 / **O(Td) 内存** | **O(Td)** | **1.000**（数值等价） | Dao 2023 |
| `linear` | O(Td²) | O(d²) | 0.088 | Katharopoulos et al. 2020 |
| `performer` | O(Tdm) | O(dm) | 0.041 | Choromanski et al. 2021 |
| `local_window` | O(Twd) | O(wd) | 窗口内精确 | Beltagy et al. 2020 |
| `block_sparse` | O(Tbd) | O(bd) | 块内精确 | Child et al. 2019 |
| `effattnfuse` | 自适应 | 自适应 | 质量守卫 | EffAttnForge |

**关键结论**：`flash` 在 **T=4096** 时峰值额外内存 **2.2 MB** vs `naive` 的 **270.5 MB**（**121×** 优势），且输出与精确注意力**逐位一致**（误差 4.9e-16）——内存墙被消除，数学结果不变。

### 训练闭环

| 指标 | 初始 | 最终 |
| --- | --- | --- |
| 训练损失 | 5.636 | **3.352** |
| 验证损失 | 5.628 | **3.412** |
| 验证困惑度 | 243 | **30.3** |
| 梯度范数 | 1.413 | 0.408（稳定） |
| 墙钟 | — | 171 s / 800 步（CPU，3 层 96 维） |

### 模型规模

| 配置 | 参数量 |
| --- | --- |
| 3 层 × 96 维（示例） | 357,216 |
| 4 层 × 128 维（默认） | ~1.0 M |

---

## 验证体系

本项目最核心的资产是**验证体系**：每个算子都有一条可被独立实现交叉验证的硬不变量。

| 编号 | 不变量 | 实测值 |
| --- | --- | --- |
| I1 | FlashAttention-2 ≡ 精确注意力 | 4.94e-16 |
| I2 | 注意力权重逐行随机化 | 2.22e-16 |
| I3 | 因果掩码上三角恒为 0 | 0.00e+00 |
| I4 | key 平移不变（logsumexp 稳定） | 1.22e-13 |
| I5 | 输出落在 value 凸包内 | 1.56 |
| I6 | KV-Cache 增量 ≡ 全量前向 | 4.27e-16 |
| I7 | Flash 内存次二次（T=4096 省 121×） | 1.21e+02 |
| R1 | RoPE 相对位置性质 | 7.53e-15 |
| R2 | RoPE 保范 + 双实现逐位一致 | 8.88e-16 |
| R3 | RMSNorm 单位 RMS | 1.94e-10 |
| G1 | **全模型解析梯度 ≡ 中心差分** | 0.00e+00 |
| G4 | 均匀预测时交叉熵 ≡ ln(V) | 0.00e+00 |
| P1 | 同 seed 逐位确定性 | 0.00e+00 |
| T1 | BPE 编解码无损往返 | 0.00e+00 |
| T3 | BPE 训练确定性 | 0.00e+00 |
| A4 | 近似注意力保真度（linear/performer） | 8.83e-02 |
| X1–X7 | **vs PyTorch 逐张量交叉验证** | ≤ 5.6e-15 |

完整清单与判据见 [docs/DESIGN.md](docs/DESIGN.md#不变量清单)。

---

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/DESIGN.md](docs/DESIGN.md) | 架构决策、逐层设计、**四个被验证体系抓出的真实缺陷**、不变量清单 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 十二层映射、模块依赖、数据流 |
| [docs/API.md](docs/API.md) | 全部公开接口用法与示例 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 论文与开源项目出处、致谢 |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | 基准结果解读与复现方法 |
| [NOTICE](NOTICE) | 第三方算法出处清单（法律声明） |

---

## 引用与致谢

本项目**独立实现**，未复制任何第三方项目源码。借鉴的数学方法与算法思想均已在
[NOTICE](NOTICE) 与 [docs/REFERENCES.md](docs/REFERENCES.md) 中逐条注明出处。

核心引用：

- Vaswani et al. (2017) *Attention Is All You Need* — Transformer
- Dao (2023) *FlashAttention-2* — 内存墙的正解
- Su et al. (2021) *RoFormer* — 旋转位置编码
- Zhang & Sennrich (2019) *RMSNorm* · Shazeer (2020) *SwiGLU*
- Touvron et al. (2023) *LLaMA* — 架构组合范式
- Choromanski et al. (2021) *Performers / FAVOR+* — 线性注意力
- Loshchilov & Hutter (2019) *AdamW* · Sennrich et al. (2016) *BPE*

---

## 许可证

[Apache License 2.0](LICENSE) © 2026 晨星 (MorningStar)

---

<div align="center">
<sub>零重写 · 集大成 · 可验证 · 闭环交付</sub>
</div>
