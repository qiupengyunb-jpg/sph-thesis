# Stage 21 —— 两尺度核最小实现报告

**日期**：2026-09-24｜**仓库**：`E:\SPHINXsys_Dev\sphinxsys`（分支 `feature/cg-axisym-singlemode-scheme-e`）
**状态**：**Phase 1/2 完成；Phase 3 已实现并编译通过；Phase 4-A 通过；Phase 4-B 单元测试 FAIL（根因已定位）；Phase 4-C 短算例稳定**
**结论一句话**：**机制已可开关实现、legacy 逐位不变，但两尺度核与现有配对循环的"W ≥ 0"假设冲突，导致 energy–force 一致性测试失败；在修复该门控前不得用于任何物理解读。**

---

## Phase 1：代码地图 → `Stage21_code_map.md`（已完成）

要点：kernel 在 L1590–1615；`SGDifferencePairPref()` L1881（`c=144/(5h²)`、`pref=0.5λV0²c`）；**λ 只经该 pref 进入**（共轭 sweep、力 sweep、from-scratch 能量/力、dump 四处）；邻居截断 `2.6·dx`；
⚠ `selfcheck.csv` 的 `sg_energy` 是 **Scheme A 口径**，不是 Scheme E 能量。

## Phase 2：数学设计 → `Stage21_kernel_design.md`（已完成）

`K_new = [c_h W_h − β c_ξ W_ξ]/(1−β)`，`c=144/(5h²)`；归一化保证 `∫|u|²K=4` ⇒ λ 语义不变。
小 k 展开 `Ê(k) ≈ (λ/2)|a|²[A k² + B k⁴]`，A>0、B 可负 ⇒ 预测 `k* ~ 1/ξ`、`λ* ≈ 2πξ`（ξ=5.6σ 时 λ*≈30–40σ，与观测 40–80σ 同量级）。
**首要工程约束**：`ξ ≤ 0.99·2.6·ParticleSpacing()`；实测 `ParticleSpacing≈1.154` ⇒ **ξ ≤ 2.97σ**（详见 §4-C）。

## Phase 3：实现（已完成并编译）

| 改动 | 内容 |
|---|---|
| CLI | `--interface-kernel=legacy\|two_scale`（默认 legacy）、`--interface-xi`、`--interface-beta` |
| 校验 | two_scale ⇒ 必须 axisymmetric + `difference_energy` + `0<β<1` + `ξ>h_ρ` + `ξ ≤ 0.99·2.6·ParticleSpacing()` |
| 工厂 | `SchemeKernelW/SchemeKernelDW`（legacy 分支**逐字调用原函数**）；`SchemeKernelSupport()` 给出配对循环上界 |
| 语义修正（重要） | **密度估计量 ρ=ΣW_h 仍用 legacy 核**（及其导数用于密度链式支）；**只有"能量核"换成两尺度**——共轭场、显式 ∂K̃/∂x 支、from-scratch 能量与显式核导数支用新核；链式项用 legacy 核导数 |
| 未改 | λ、V0/ρ_ref、h_ρ、Morse/WCA/wall、积分器、随机数、默认值 |

## Phase 4-A：zero-path 逐位回归 —— **通过**

| 算例 | 期望（历史基线） | 实测（新 exe） | 判定 |
|---|---|---|---|
| strip λ=0 | `6F0181C3 / DDE80488 / 0AEE50C0` | **`6F0181C3 / DDE80488 / 0AEE50C0`** | ✅ 逐位一致 |
| A-1 λ=12 | `DC72C786 / 16015253 / ED462B63` | **`DC72C786 / 16015253 / ED462B63`** | ✅ 逐位一致 |
| axisym Scheme E + legacy kernel | 运行正常 | T=20 raw 8667 步完成、无异常 | ✅（结构上 SchemeKernelW 在 legacy 下逐字返回 RhoKernelW） |

（回归在**加了 `SchemeKernelSupport()` 修正之后**又跑了一次，仍然逐位一致。）

## Phase 4-B：新机制单元测试（`--sg-fd-check`）—— **FAIL（已定位根因）**

`--interface-kernel=two_scale --interface-xi=2.9 --interface-beta=0.3`：

```
delta=1e-4 : Fr max rel err = 2.30e+08   abs err = 2.54e+06
delta=3e-5 : Fr max rel err = 2.07e+07   abs err = 2.29e+05
delta=1e-5 : Fr max rel err = 2.30e+06   abs err = 2.54e+04
delta=3e-6 : Fr max rel err = 2.07e+05   abs err = 2.29e+03
```

**误差 ∝ 1/δ**（δ 每缩小 3.3×，误差放大 ~10×）⇒ **FD 能量对径向位移几乎不敏感**，即两侧的"配对集合"不一致。

**根因（已定位，未在本轮修复）**：
legacy 的配对循环用 **`if (w <= 0.0) continue;`** 门控（共轭 sweep 与 force sweep 都有此写法），
这一写法**默认核非负**（legacy Wendland 恒为正）。而两尺度核 `K_new = [c_h W_h − β c_ξ W_ξ]/(1−β)` 的**长程瓣为负**（这正是它产生有限 k 极值的原因），
于是共轭场/力路径**跳过负瓣配对**，而 from-scratch 能量（按 `r < support` 遍历）**包含**这些配对 ⇒ 两边不是同一个泛函 ⇒ FD 必然失败。
**修复方向（一步）**：把两处 `if (w <= 0.0) continue;` 改为**按支集门控** `if (r >= SchemeKernelSupport()) continue;`（保留负值配对），并把 force sweep 的 `w<=0 && dw==0` 同样改为支集门控；随后重跑 FD 门槛。
（legacy 路径不受影响：legacy 下 `SchemeKernelSupport()==h_ρ`，且 W_h 在支集内恒正 ⇒ 行为逐位不变，已由 4-A 证明。）

## Phase 4-C：小算例 R0=3 / h=3 / Lz=80（two_scale ξ=2.9, β=0.3, t=50）

```
T=50.0000 N=201 T_kin=1.0578 max_speed=3.8152 min_sep=0.8878 guard=0 hard=0 wrapped=0
finished: steps=6667 wall_seconds=3.0543
```
⇒ **稳定、无 NaN、无 guard/hard、无异常聚集**（T_kin≈1.06 正常）。
**但**：在 4-B 通过之前，该算例**不得用于物理解读**（其力与能量不一致）。

**重要发现（ξ 窗被截断封死）**：`ParticleSpacing()` 由 `LargestPairCutoff()/2.6` 决定（Morse 截断 3σ ⇒ 1.154σ），
故 `ξ ≤ 0.99·2.6·1.154 = 2.97σ`，而物理有意义的窗口是 **ξ ≳ 2h_ρ = 5.6σ**（λ*≈35σ）。
⇒ **当前框架下两尺度核只能取到 ξ∈(2.8, 2.97]σ（约 6% 宽），无法进入 40–80σ 的观测区间**；
要真正测试该机制必须**放大邻居搜索半径**（框架级改动，超出"最小 opt-in 核"的范围）。

## 最终回答

**1. 代码地图**：见 `Stage21_code_map.md`（λ 入口唯一、scheme A/E 口径区分、邻居约束）。
**2. 数学设计**：见 `Stage21_kernel_design.md`（λ 不需重标定；预测 λ*≈2πξ）。
**3. 实现**：已完成（opt-in、默认关、legacy 逐位不变）。
**4. 验证**：zero-path **PASS**；单元测试 **FAIL（根因=负瓣核与 `w<=0` 门控冲突，修复方向明确）**；小算例稳定但暂不可解读。
**5. 下一步（建议，待批准）**：① 修 3 处支集门控 → 重跑 FD；② 若 FD 通过，再评估是否放大邻居截断以解锁 ξ≥5σ；③ **在此之前不要进入任何生产矩阵**。

## 交付

| 内容 | 路径 |
|---|---|
| 代码地图 / 设计 / 本报告 | `Stage21_two_scale_kernel\Stage21_{code_map,kernel_design,two_scale_kernel_implementation_report}.md`（仓库副本 `tools/cg_analysis/m27/`）|
| solver 改动 | `cg_self_assembly_pri.cpp`（新增 CLI + 工厂 + 3 处门控/语义修正；commit 见下）|
| 回归日志 | `E:\哈哈\_stage21\reg*\{strip,a1,axi}` |
| 单元测试 / 小算例 | `E:\哈哈\_stage21\{fd,small,small_legacy}` |
| 本轮未做 | 未进入生产矩阵；未改 λ/V0/h_ρ/Morse/WCA/wall/积分器/随机数 |
