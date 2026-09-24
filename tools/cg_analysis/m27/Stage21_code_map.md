# Stage 21 Phase 1 —— Scheme E 代码地图（code map）

**仓库**：`E:\SPHINXsys_Dev\sphinxsys`｜**文件**：`tests/user_examples/test_2d_cg_self_assembly_pri/cg_self_assembly_pri.cpp`
（行号为本轮实测；**只读调查，未改代码**）

| # | 问题 | 位置 | 事实 |
|---|---|---|---|
| 1 | **kernel 定义位置** | L1590 / L1593 / L1604 / L1615 | `RhoSmoothingH()` = `interface_h * Sigma()`（默认 2.8σ）；`RhoKernelW(r)` = Wendland C2，`7/(π h²)·(1−q)⁴(4q+1)`，q=r/h；`RhoKernelDW(r)` = `−(140/(π h³))·q(1−q)³`；`RhoKernelDDW(r)`（仅 Scheme A 的链式项用）|
| 2 | **邻居搜索范围** | 邻居表 cutoff = `2.6·dx`（校验见 `--interface-gradient-h` 检查）| Scheme E 的配对求和**只遍历邻居表**；`W=0` 超出 `h_ρ` ⇒ **h_ρ 必须 < 2.6·dx**，当前校验强制 `interface_h ≤ 0.99·2.6·dx` |
| 3 | **c 系数来源** | L1881–1886 `SGDifferencePairPref()` | `ck = 144.0/(5.0*h*h)`（由 `∫|u|²K dA = 4` 闭式推得，h=h_ρ）；返回 **`0.5·λ·V0²·ck`**，`V0 = 1/interface_rho_ref` |
| 4 | **λ 进入 force 的位置** | 仅经 L1881 的 pref | ① 共轭场 sweep `CGSquareGradientConjugate`（L1907）；② 力 sweep `CGSquareGradientDifferenceForce`（L1953，`ForceFromCachedGeometry` 内含显式 ∂(measure)/∂x 与 ∂W/∂x 支）；③ from-scratch 能量/力 `SGDifferenceConfigEnergy`（L2557）/`SGDifferenceConfigForce`（L2595，供 `--sg-fd-check`）；④ dump（`DumpForceSplit`）。**没有第二处 λ 入口** ⇒ 换核只需替换 kernel 与 pref 的一致组合 |
| 5 | **energy selfcheck 如何记录** | L2442–2446 | ⚠ **注意**：`selfcheck.csv` 的 `sg_energy` 列的 `energy_scale = 0.5·λ/ρ_ref`，`e_weighted = Σ J|G_i|²` ⇒ 这是 **Scheme A（point-gradient）口径**，**不是** Scheme E 的能量；控制台 `sgE=` 同源。Scheme E 的能量**只在 `--sg-fd-check` 路径**里被求值。⇒ 新增核时必须**单独**给出 Scheme E 能量诊断，不能复用该列 |

## 对两尺度核的直接后果（Phase 2 用）

* 换核只需改 **kernel 两个函数**（W、dW）+ 与 `SGDifferencePairPref()` 保持一致的归一化；
* **λ 的入口唯一** ⇒ 不会与 Scheme A 的 `|G|²` 口径混淆；
* **最大工程约束**：新核的长程支集 `ξ` 必须 ≤ `0.99·2.6·dx`。当前 `dx ≈ 1.14–1.20σ` ⇒ **ξ ≤ 约 3.0σ**；
  若要试验 ξ ≥ 5σ，必须**同时放大邻居截断**（改框架参数，属另一层改动）或**粗化晶格**（增大 dx）。
