# Stage 21R —— Two-scale kernel support gate repair 报告

**日期**：2026-09-24｜**仓库**：`E:\SPHINXsys_Dev\sphinxsys`（分支 `feature/cg-axisym-singlemode-scheme-e`）
**结论一句话**：**支集门控修复生效（1/δ 特征消失、legacy 逐位一致），但暴露出第二个"径向专属、δ 无关"的不一致，two_scale 的 FD 门槛仍未通过；按任务要求未调参数、未跑生产。**

---

## Phase 1：kernel sign gate 审计（`Stage21R_gate_audit.md`）

全文件共 7 处 kernel 相关门控：

| 行 | 代码 | 所在函数 | 用途 | 处置 |
|---|---|---|---|---|
| L1760 | `if (w <= 0.0)  // outside the kernel support` | `CGSquareGradientDensity` | 密度估计量 ρ=ΣW_h 的配对过滤 | **保留**（密度恒用 legacy 核，W_h≥0 成立） |
| L1816 | `if (w1 == 0.0)`（w1=RhoKernelDW）| `CGSquareGradientForce`（Scheme A） | Scheme A 链式项 | **保留**（Scheme A 不使用两尺度核） |
| L1877 | `if (w1 == 0.0)` | Scheme A 的 REF-CACHED 路径 | 同上 | **保留** |
| L1928 | `if (w1 == 0.0)` | Scheme A 的 `DumpForceSplit` | 同上 | **保留** |
| **L2048** | `if (w <= 0.0)` | **`CGSquareGradientConjugate`** | Scheme E 共轭场配对过滤 | **改** |
| **L2106** | `if (w <= 0.0 && dw == 0.0)` | **`ForceFromCachedGeometry`** | Scheme E 力配对过滤 | **改** |
| **L2173** | `if (w <= 0.0 && dw == 0.0)` | **`DumpForceSplit`（Scheme E 分支）** | 诊断 dump 配对过滤 | **改** |

**问题本质**：legacy 写法 `w <= 0` 隐含"核非负"假设；两尺度核 `K_new=[c_hW_h−βc_ξW_ξ]/(1−β)` 的**长程瓣为负**（这正是产生有限 k 极值的来源），于是共轭场/力路径**跳过负瓣配对**，而 from-scratch 能量按 `r < support` **包含**它们 ⇒ 两侧不是同一个泛函。

## Phase 2：pair selection 修复（已实施）

三处统一改为**按支集门控**：

```cpp
if (r >= SchemeKernelSupport())   // support gate (keeps negative lobes)
    continue;
```

`SchemeKernelSupport()` = `two_scale ? max(h_ρ, ξ·σ) : h_ρ`。
**legacy 等价性**：legacy 下 W_h 在 `r<h_ρ` 内恒正、外边恒 0，故 `r >= h_ρ` 与 `w <= 0` 判定**完全同一配对集** ⇒ 逐位不变（Phase 3 已验证）。
**未使用 `w > 0`**（任务要求）✓。

## Phase 3：legacy zero-path 回归 —— **通过** ✅

| 算例 | 历史基线 | 修复后实测 | 判定 |
|---|---|---|---|
| strip λ=0 | `6F0181C3 / DDE80488 / 0AEE50C0` | **完全一致** | ✅ |
| A-1 λ=12 | `DC72C786 / 16015253 / ED462B63` | **完全一致** | ✅ |

## Phase 4：FD 复测（δ = 1e-4 / 3e-5 / 1e-5 / 3e-6）

### 4.1 关键判据：**1/δ 特征已消失**（修复生效的直接证据）

| 配置 | δ=1e-4 | δ=3e-5 | δ=1e-5 | δ=3e-6 | 特征 |
|---|---|---|---|---|---|
| two_scale（修复前） | 2.30e+08 | 2.07e+07 | 2.30e+06 | 2.07e+05 | **∝1/δ**（配对集不一致） |
| **two_scale（修复后）** | **1.07e+01** | **1.07e+01** | **1.07e+01** | **1.07e+01** | **δ 无关**（新的、有限的不一致） |

⇒ 支集门控确实解决了"两侧配对集不同"的问题。

### 4.2 但它暴露出**第二个不一致**，且**只在径向**：

| 配置（同一算例：R0=5、h=5、ε_pf=13、packing 0.63、Lz=160） | Fr max rel err | **Fz max rel err** |
|---|---|---|
| legacy | **2.87e-08**（δ=1e-4）→ 9.59e-09（δ=3e-6） | 0.000e+00 |
| **two_scale（修复后）** | **1.07e+01**（δ 无关） | **0.000e+00** |

**诊断要点**：
* legacy 在同一算例**通过**（2.9e-8）⇒ 不是 FD 测试框架的问题；
* two_scale 的 **Fz 误差精确为 0**，只有 **Fr** 错 ⇒ 不一致**局限在径向分量**；
* 与 δ 无关 ⇒ 不是离散/截断误差，而是**解析力与能量梯度之间的常数级差**（比值 ~10.7）。

**已定位的两个候选根因（下一步修，本轮未改）**：
1. **归一化重复计入**：`SGDifferencePairPref()` 内已含 `c_h = 144/(5h²)`（pair 权重 = pref × 裸 Wendland），而本次实现的 `SchemeKernelW/DW` **又乘了一次 c**（`[c_hW_h − βc_ξW_ξ]/(1−β)`）。
   正确形式应为 `[W_h − β(h²/ξ²)W_ξ]/(1−β)`（即把 c_h 提出去）。
   由于该因子在力与能量的解析路径中同时出现，它**不解释** FD 的 Fr 失配，但**会同时把 λ 的等效尺度放大 c_h≈3.67 倍**——是一处必须修的标定错误。
2. **径向显式支的配对一致性**：Fr 专属 ⇒ 需逐项核对 `SGDifferenceDJdr()·w·ĵ`（测度导数支）在力路径与 from-scratch 力之间的对应，以及 from-scratch 能量在 `r∈[h_ρ, ξ]` 区间的 ∂(J)/∂r 贡献是否与力路径同口径。

## Phase 5：two_scale 小算例（R0=3、h=3、Lz=80、ξ=2.9、β=0.3、t=50）

```
T=50.0000 N=201 T_kin=1.0177 max_speed=3.9581 min_sep=0.8948 guard=0 hard=0 wrapped=0 periodic_pairs=2948
finished: steps=6667 wall_seconds=3.0360
```
* **稳定、无 NaN、guard=hard=0**（T_kin≈1.02 正常）；
* **energy–force consistency 未通过**（Phase 4）⇒ 按任务要求**不做任何周期结构/物理解读**。

## 最终回答（对应停止条件）

| 停止条件 | 状态 |
|---|---|
| 1. gate audit | ✅ 完成（7 处审计、3 处修改） |
| 2. force–energy consistency | ❌ **未通过**（1/δ 已修掉，但残余径向常数级失配 ~10.7；根因缩到 2 个候选） |
| 3. legacy zero-path | ✅ 通过（strip / A-1 逐位一致） |

## 交付

| 内容 | 路径 |
|---|---|
| 本报告 | `Stage21_two_scale_kernel\Stage21R_two_scale_consistency_report.md`（仓库副本 `tools/cg_analysis/m27/`）|
| 门控审计 | `Stage21_two_scale_kernel\Stage21R_gate_audit.md` |
| solver 改动 | `cg_self_assembly_pri.cpp`（3 处支集门控）|
| 回归 / FD / 小算例数据 | `E:\哈哈\_stage21r\{reg,fd_ts,fd_leg,small}` |
| 本轮未做 | 未改物理公式 / λ / V0 / β 默认值 / legacy 核；未跑生产算例 |

---

# 追加：Phase 4 深入排查（第二轮，2026-09-24）

## 4.3 已修的两处（均保留，zero-path 复验仍逐位一致）

1. **归一化重复计入**：`SGDifferencePairPref()` 已含 `c_h=144/(5h²)`，原两尺度核又乘一次 c ⇒ 改为
   `SchemeKernelW/DW = [W_h − β(h²/ξ²)W_ξ]/(1−β)`（等效 λ 不再被放大 3.67 倍）。
   **FD 结果完全不变**（1.074e+01 → 1.074e+01）⇒ 该因子两侧共有，**不是 FD 失配的原因**，但是必须保留的标定修正。
2. **from-scratch 密度回路**：`SGDifferenceConfigEnergy/Force` 的 ρ 重建原本调用 `SGDifferenceW`（两尺度核）⇒ 改为
   `rho[i] += RhoKernelW(r)`（legacy、支集 h_ρ），**能量核**只在配对能量/显式支使用（支集 `SchemeKernelSupport()`）。

## 4.4 仍然失败，但误差特征完全一致（δ 无关，仅径向）

| δ | Fr max rel err | Fz max rel err |
|---|---|---|
| 1e-4 / 3e-5 / 1e-5 / 3e-6 | **1.074e+01（四档相同）** | **0.000e+00** |

legacy 同配置：2.87e-08（δ=1e-4）→ 9.59e-09（δ=3e-6）✅

## 4.5 决定性新证据：**参考实现与 solver 的 ρ 本身就不一致**

`--sg-fd-check` 顶部诊断（two_scale, h=5）：

```
solver vs independent reference (max rel): rho=0.31246   grad rho=0.490075   force=5.7874
max |rho|: reference=0.526621   solver=0.526621
worst grad mismatch at i=149  pos=(58.8, 7.57812)   reference neighbours within h_rho=18
   rho:  reference=0.526621   solver=0.44332
   grad: reference=(-1.2e-15,-2.4e-17)   solver=(-0.16742, 2.8e-17)
```

* **同一个粒子**：参考 ρ=0.5266（18 个 h_ρ 内邻居）对 solver ρ=0.4433 ⇒ **相差 31%**；
* 参考梯度 ≈0（该粒子在参考中处于近均匀位置）而 solver 梯度 = −0.167；
* **该不一致在 legacy 下不存在**（legacy FD 通过）。

⇒ **结论：FD 失配的当前根因不在"配对过滤"（那已修好），而在"参考实现与 solver 的密度场 ρ 不是同一个对象"。**
solver 的 ρ 偏小 ⇒ solver 侧**少算了邻居**（或邻居表在该配置下未包含全部 h_ρ 内配对）。这是下一轮要查的第一项：
核对 FD 检查时 solver 的邻居表是否在**同一时刻、同一位置**刷新（以及 `--sg-fd-check` 路径是否在 rho/grad sweep 之后才构建参考）。

## 4.6 Phase 5 小算例（本轮重跑，gate 仍未过 ⇒ 不做物理解读）

```
R0=3 h=3 Lz=80 xi=2.9 beta=0.3 t=50 : N=201 T_kin=1.0177 guard=0 hard=0 无 NaN, finished steps=6667
```

## 4.7 本轮停止状态

| 条件 | 状态 |
|---|---|
| gate audit | ✅ |
| pair selection 修复（支集门控） | ✅（1/δ 特征消失） |
| legacy zero-path | ✅ 逐位一致 |
| force–energy consistency | ❌（残余径向常数失配 10.74；新证据指向 ρ 场不一致） |
| small smoke | ✅ 稳定 |
