# MorningStar FlagshipAI · 验证报告

- 生成时间: 2026-10-24T19:22:51+00:00 (UTC)
- 环境: Python 3.13.14 · numpy 2.5.4
- 平台: Windows-11-10.0.29671-SP0

## 总览

| 项目 | 结果 |
| --- | --- |
| 数学不变量 | 16/16 PASS |
| 跨库交叉验证 (PyTorch) | 7/7 PASS （可用 7） |
| 全绿 | ✅ 是 |
| 耗时 | 9.018 s |

## 数学不变量明细

| name | value | threshold | passed | detail |
| --- | --- | --- | --- | --- |
| I1_flash_equals_naive | 4.943e-16 | 1.000e-12 | ✅ PASS | T=128, d=32: FlashAttention-2 vs 精确 softmax 注意力 |
| I2_row_stochastic | 2.220e-16 | 1.000e-12 | ✅ PASS | max|P·1 − 1| |
| I3_causal_upper_zero | 0.0000 | 1.000e-14 | ✅ PASS | max|P[i>j]| |
| I4_shift_invariance | 1.218e-13 | 1.000e-12 | ✅ PASS | key 加常数 1e3 后输出应不变（logsumexp 稳定性） |
| I5_convex_hull_bounded | 1.5575 | 4.3562 | ✅ PASS | 输出行范数上界 = max||v_j|| = 4.3562 |
| I6_kv_cache_consistency | 4.273e-16 | 1.000e-10 | ✅ PASS | 逐 token 解码 24 步 vs 一次性前向 |
| I7_memory_subquadratic | 121.4118 | 10.0000 | ✅ PASS | T=4096: naive 270.5 MB vs flash 2228.2 KB |
| R1_rope_relative_positions | 7.528e-15 | 1.000e-12 | ✅ PASS | ⟨R_m q, R_n k⟩ 仅在 n−m 上取值 |
| R2_rope_norm_preserving | 8.882e-16 | 1.000e-12 | ✅ PASS | ‖Rot(x,m)‖ = ‖x‖ 且两条独立实现逐位一致 |
| R3_rmsnorm_unit_rms | 1.937e-10 | 1.000e-09 | ✅ PASS | RMS(y) = 1 ∀ rows |
| G1_gpt_gradient_check | 0.0000 | 1.000e-06 | ✅ PASS | 最差点在 ; loss=3.5123 |
| G4_crossentropy_analytic | 0.0000 | 1.000e-12 | ✅ PASS | ln(64) = 4.158883, got 4.158883 |
| P1_bitwise_determinism | 0.0000 | 0.5000 | ✅ PASS | init_identical=True, forward_identical=True |
| T1_bpe_roundtrip | 0.0000 | 0.5000 | ✅ PASS | vocab_size=300, 语料 32 条 |
| T3_bpe_determinism | 0.0000 | 0.5000 | ✅ PASS | 40 merges |
| A4_approx_attention_fidelity | 0.0883 | 0.5000 | ✅ PASS | linear=0.0883, performer=0.0405 (m=256, ‖q‖²≈2.83); 原始尺度 performer=310.488 |

## 跨库交叉验证明细 (vs PyTorch)

| name | value | threshold | passed | detail |
| --- | --- | --- | --- | --- |
| X1_attention_vs_torch_sdpa | 5.551e-16 | 1.000e-10 | ✅ PASS | vs F.scaled_dot_product_attention (float64) |
| X2_softmax_vs_torch | 4.235e-22 | 1.000e-12 | ✅ PASS | 量级 ×30 的 logits 仍 Streiff 有限且与 torch 一致 |
| X3_rmsnorm_vs_torch | 8.882e-16 | 1.000e-12 | ✅ PASS | RMSNorm 定义式参照 |
| X4_swiglu_vs_torch_silu | 3.331e-16 | 1.000e-12 | ✅ PASS | SiLU(gate)⊙up @down |
| X5_rope_vs_complex_multiply | 2.220e-16 | 1.000e-12 | ✅ PASS | 复数乘法 (x_{2j}+i x_{2j+1})·e^{i m θ_j} |
| X6_adamw_vs_torch | 2.220e-16 | 1.000e-10 | ✅ PASS | 20 步更新后权重最大偏差 |
| X7_crossentropy_vs_torch | 1.776e-15 | 1.000e-12 | ✅ PASS | reduction='mean' |

## 注意力家族基准

| variant | T | d | median_ms | mem_mb | fro_rel | complexity |
| --- | --- | --- | --- | --- | --- | --- |
| block_sparse | 128 | 32 | 0.5306 | 0.0488 | 0.0000 | O(T b d) [b=64, g=8] |
| effattnfuse | 128 | 32 | 0.9635 | 0.2812 | 0.0000 | O(T^2 d) [short: exact] |
| flash | 128 | 32 | 0.8992 | 0.0957 | 0.0000 | O(T^2 d) time / O(T d) memory [bq=64, bk=64] |
| full | 128 | 32 | 0.5767 | 0.2812 | 0.0000 | O(T^2 d) [exact] |
| linear | 128 | 32 | 0.2475 | 0.0703 | 0.7122 | O(T d^2) |
| local_window | 128 | 32 | 2.9819 | 0.0159 | 1.0944 | O(T w d) [w=32] |
| performer | 128 | 32 | 5.2726 | 1.1250 | 37.2039 | O(T d m) [m=256] |
| block_sparse | 512 | 32 | 1.9742 | 0.0488 | 1.2715 | O(T b d) [b=64, g=8] |
| effattnfuse | 512 | 32 | 10.5281 | 2.0625 | 363.3041 | O(T d m) [long: Performer m=128] |
| flash | 512 | 32 | 8.9810 | 0.1953 | 0.0000 | O(T^2 d) time / O(T d) memory [bq=64, bk=64] |
| full | 512 | 32 | 9.2349 | 4.1250 | 0.0000 | O(T^2 d) [exact] |
| linear | 512 | 32 | 0.6936 | 0.2578 | 0.6946 | O(T d^2) |
| local_window | 512 | 32 | 7.2194 | 0.0159 | 2.3325 | O(T w d) [w=32] |
| performer | 512 | 32 | 19.7729 | 4.1250 | 300.8099 | O(T d m) [m=256] |
| block_sparse | 1024 | 32 | 3.4902 | 0.0488 | 1.9932 | O(T b d) [b=64, g=8] |
| effattnfuse | 1024 | 32 | 17.4183 | 4.0625 | 551.2643 | O(T d m) [long: Performer m=128] |
| flash | 1024 | 32 | 31.2707 | 0.3281 | 0.0000 | O(T^2 d) time / O(T d) memory [bq=64, bk=64] |
| full | 1024 | 32 | 37.7770 | 16.2500 | 0.0000 | O(T^2 d) [exact] |
| linear | 1024 | 32 | 1.8943 | 0.5078 | 0.7600 | O(T d^2) |
| local_window | 1024 | 32 | 12.8548 | 0.0159 | 3.5518 | O(T w d) [w=32] |
| performer | 1024 | 32 | 35.2254 | 8.1250 | 212.5436 | O(T d m) [m=256] |

## 内存伸缩对比 (naive O(T²) vs flash O(T))

| T | d | naive_mb | flash_mb | memory_ratio |
| --- | --- | --- | --- | --- |
| 128 | 32 | 0.2812 | 0.0957 | 2.9400 |
| 512 | 32 | 4.1250 | 0.1953 | 21.1200 |
| 1024 | 32 | 16.2500 | 0.3281 | 49.5200 |
