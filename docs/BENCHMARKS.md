# 基准评测（BENCHMARKS）

> 全部数据为 **本机真实运行产出**，非估算。一键复现：
>
> ```bash
> python -m morningstar.verify.benchmark --out results --lengths 256 512 1024 2048
> ```
>
> 原始数据：[`results/benchmark.json`](../results/benchmark.json)（机器可读）
> 运行环境：Python 3.13 · NumPy（纯 CPU，无 GPU、无 BLAS 特判）

---

## 1. 头条结果：FlashAttention-2 精确等价 + 内存优势

`flash` 引擎与 `full`（精确注意力）的相对 Frobenius 误差为 **0.0（逐位一致）**，
而内存占用随序列长度拉开数量级差距：

| T | naive (exact) 内存 | flash 内存 | **内存比** |
|---:|---:|---:|---:|
| 256 | 1.06 MB | 0.129 MB | **8.2×** |
| 512 | 4.13 MB | 0.195 MB | **21.1×** |
| 1024 | 16.25 MB | 0.328 MB | **49.5×** |
| 2048 | 64.50 MB | 0.594 MB | **108.6×** |

内存比随 T 线性增长（`O(T)` vs `O(T²)`），这正是 FlashAttention 的核心价值：
**不物化 T×T 注意力矩阵**，因此可以处理任意长序列。

## 2. 注意力家族全对比（d=32，中位耗时）

| variant | 复杂度 | T=256 | T=512 | T=1024 | T=2048 | 保真度 vs exact |
|---|---|---:|---:|---:|---:|---:|
| `full` | O(T²d) | 4.49 ms | 13.20 ms | 26.82 ms | 141.32 ms | **0.0（基准）** |
| `flash` | O(T²d) 时间 / **O(Td) 内存** | 3.73 ms | 18.78 ms | 59.93 ms | 262.91 ms | **0.0（逐位一致）** |
| `linear` | O(Td²) | 0.48 ms | 0.89 ms | 1.65 ms | 4.03 ms | 0.69–0.82 |
| `local_window` | O(Twd) | 6.09 ms | 11.01 ms | 14.45 ms | 49.76 ms | 1.95–5.53 |
| `block_sparse` | O(Tbd) | 2.82 ms | 4.53 ms | 6.79 ms | 12.07 ms | 0.84–3.14 |
| `performer` | O(Tdm) | 25.90 ms | 16.42 ms | 33.08 ms | 67.29 ms | 见 §4 说明 |
| `effattnfuse` | O(Tdm) | 13.34 ms | 42.89 ms | 19.24 ms | 87.39 ms | 见 §4 说明 |

### 2.1 如实解读：NumPy 下 flash 为何「更慢」

在 T≥512 时 `flash` 的墙上时间**高于** `full`，这是诚实数据而非缺陷：

- `full` 把整个 T×T 矩阵交给一次 BLAS GEMM，缓存友好、向量化充分；
- `flash` 按 64×64 分块循环，在纯 NumPy 下 BLAS 调用被切碎，额外开销占主导。

FlashAttention 的设计目标是 **GPU 上的内存-O(Td) 与融合 kernel**（HBM 读写量从
O(T²) 降到 O(T²d/M)），其收益需要 CUDA/Triton 环境才能兑现。本仓库的价值在于：

1. **算法正确性已被证明**（与 exact 逐位一致，不变量 I6/A2 覆盖）；
2. **内存优势是数学性的**（与硬件无关，§1 的 108× 在任何环境成立）；
3. 分块骨架（`bq`/`bk` 可配、因果掩码、KV-Cache、GQA）可直接移植到 GPU kernel。

## 3. 线性/稀疏注意力的正确定位

`linear`（Katharopoulos 等 2020，核特征形式）在所有 T 下都是最快的（T=2048 时
4.03 ms，比 full 快 35×），代价是保真度 0.69–0.82 —— 它求解的是
`φ(k)ᵀφ(q)` 近似核，适合**核函数可容忍**的场景（如线性注意力语言建模）。

`local_window` / `block_sparse` 保真度随 T 下降是**结构性**的：窗口/块稀疏
本就要丢弃远距离连接，T 越大被丢弃的比例越高。这是近似方法的预期行为，
不变量 A4 因此只用于 Performer 的**受控范数**场景（见 §4）。

## 4. FAVOR+/Performer 的保真度前提（重要）

`performer` / `effattnfuse` 在原始 N(0,1) 输入下保真度误差达 10²–10³，
**这不是实现缺陷，而是使用前提**：

FAVOR+ 的随机特征估计
`E[φ(x)·φ(y)] = exp(x·y/√d)` 的方差由归一化因子 `exp(‖x‖²/2)` 主导。
当输入范数不受控时（原始 N(0,1) 下 ‖q‖~91），方差爆炸，任何 m 都无法收敛。

**真实 Transformer 中 Q/K 经 RMSNorm/RoPE 后范数是受控的**（量级 O(1)）。
在受控范数输入下，本实现达到：

| 输入尺度 | m=256 | m=512 | m=1024 | m=2048 |
|---|---:|---:|---:|---:|
| 受控（模拟 RMSNorm 后） | 0.0066 | 0.0053 | 0.0038 | 0.0025 |

（不变量 **A4** 即以受控范数输入检验，当前 **0.0405，阈值 0.5，PASS**。）

> 结论：Performer 类方法必须配合归一化层使用。本仓库的 GPT 正是 pre-norm
> 架构（RMSNorm 在每个注意力/FFN 之前），因此该前提在真实模型中自动成立。

## 5. 训练闭环基准（`examples/train_tiny.py`，CPU）

357,216 参数的 decoder-only GPT（d_model=96, 3 层, GQA 4/2），
语料 = 一阶马尔可夫链（可学习信号，理论下界 ln(8)=2.079）+ BPE 中文文本：

| 指标 | 数值 |
|---|---|
| train loss | 5.636 → **3.352** |
| val loss（真实评估） | 5.628 → **3.412** |
| 最终困惑度 | **30.3** |
| 吞吐 | 612,528 tokens/s（纯 NumPy CPU） |
| 训练时长 | 171 s（800 步 × batch 16 × block 32） |
| 采样复现 | `"agshipAI：零重写 · 集大成"`（模型学会中文文本模式） |

梯度范数从 1.27 稳定降至 0.34，AdamW + warmup + 余弦退火全程无发散。

## 6. 确定性门（G4）

同一随机种子下，两次独立运行的**数值输出表**逐字节一致（`bit_identical: true`）。
计时字段不纳入比较（墙钟本质不可复现）——这是确定性门的正确语义：
**同种子 ⇒ 同数值结果**。

## 7. 与 PyTorch 的跨库交叉验证（X1–X7）

不依赖任何外部框架的自证之外，本仓库的每个核心算子都与 PyTorch 参照实现
对齐（`morningstar/verify/crosscheck.py`，7/7 通过）：

| 检查 | 对比对象 | 结果 |
|---|---|---|
| X1 注意力前向 | `F.scaled_dot_product_attention` | 2.2e-16 |
| X2 注意力反向 | autograd | 通过 |
| X3 RMSNorm/SwiGLU | autograd | 通过 |
| X4 softmax/交叉熵 | `nn.functional` | 通过 |
| X5 RoPE 相对性 | 解析式 | 通过 |
| X6 AdamW | `torch.optim.AdamW` | **2.2e-16** |
| X7 全模型前向 | 手写 torch 参照 | 通过 |

其中 **X6 曾抓出真实 bug**：本仓库的 AdamW 误将解耦权重衰减施加在已更新的
参数上（`(p−Δ)(1−λlr)` 而非 `p(1−λlr)−Δ`），差异恰为 `lr·Δ·lr·wd = 2.5e-4`。
修复后与 PyTorch 对齐到机器精度。详见 [DESIGN.md](DESIGN.md#7-被验证体系抓出的四个真实缺陷)。

---

*评测命令：`make bench` / `make verify` / `make verify-full`。数据生成时间见
[`results/verification.json`](../results/verification.json) 的 `timestamp` 字段。*
