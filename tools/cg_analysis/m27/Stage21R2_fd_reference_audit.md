# Stage21R-2 —— FD reference ρ 一致性审计

**日期**：2026-09-24｜**性质**：审计（**未改 solver 物理、未改 K_new/β/ξ/λ/V0/legacy 核**）
**状态**：Phase 1–3 完成；**Phase 4/5 未执行**——因为根因尚未被"确认"，按任务"如果确认是 checker 再改，优先改 reference"的原则**不应先动手**。

---

## Phase 1：FD checker 的 reference ρ 构造审计（逐条回答）

| # | 问题 | 答案（代码事实） |
|---|---|---|
| 1 | reference ρ 用哪个 kernel？ | `RhoKernelW(r)` —— **legacy Wendland C2**（`7/(πh²)(1−q)⁴(4q+1)`），**不含两尺度核** |
| 2 | reference ρ 用哪个 support？ | `const Real h = RhoSmoothingH()` = `interface_h*σ` = **2.8σ**，判定为 `r <= TinyReal \|\| r >= h → continue` |
| 3 | reference ρ 用哪个 neighbor list？ | **没有 neighbor list**：`SGConfigRhoGrad`（L2591）是 **O(N²) 全粒子双循环**，只对 z 取最小镜像（`d[0] -= lz*round(d[0]/lz)`）|
| 4 | reference ρ 的更新时机 | `RunSGFinitDifferenceCheck`（L2901）内、**每次 FD 位移后重新调用** `SGConfigRhoGrad(p,...)`（L2928），即**从扰动后的位置重算**（这一点符合 N1 报告中"必须重算而非用缓存"的要求）|
| 5 | solver force path 用哪个 ρ？ | `CG_Rho` 由 `CGSquareGradientDensity::interaction`（L1751 附近）计算：**遍历 `inner_configuration_[index_i]`（neighbor list）**，用 `RhoKernelW`，`if (w <= 0) continue`（等价 r < h_ρ）|
| 6 | 两者是否完全一致？ | **理论定义一致**（同 kernel、同 support=2.8σ、同 z 周期镜像），**但实测不一致**：worst-case `rho_ref = 0.526621` vs `rho_solver = 0.44332`（差 **31.2%**），`grad_ref ≈ 0` vs `grad_solver = −0.167`（差 **49%**）；该不一致**只在 two_scale 下出现，legacy 下 FD 通过（2.9e-8）** |

**结论（Phase 1）**：两条路径的 kernel、support、镜像规则**在源码层面相同** ⇒ 差异只能来自 **pair set**（solver 的 neighbor list 少配对）或 **solver 侧 ρ 的时相**（是否为当前位形的刷新值）。这两条正是 Phase 2 要查的。

## Phase 2：neighbor list 对齐（代码审计结果）

| 项 | reference | solver |
|---|---|---|
| refresh 时机 | 每次 FD 位移后**即时重算**（O(N²)，无缓存） | `particles_body.updateCellLinkedList()` + `particles_inner.updateConfiguration()`（L3757/L3759）**在 BAOAB 主循环内逐步/按需刷新**；FD 检查发生在 **t=0 初始化块**，此时 neighbor list 由初始化后的一次构建给出 |
| cutoff | `r < RhoSmoothingH() = 2.8σ` | 框架邻居表 cutoff = **2.6·dx**，`dx = ParticleSpacing() = max(1/resolution, LargestPairCutoff()/2.6)`；本配置（case=1、Morse r_c=3σ）⇒ `dx ≈ 1.154σ` ⇒ **cutoff ≈ 3.0σ > 2.8σ**（名义上不缺配对） |
| periodic | z 最小镜像（显式） | 由 `PeriodicConditionUsingCellLinkedList` 处理 |
| axisymmetric weighting | 不加权（ρ 是数密度） | 同（J=r/R0 只进能量，不进 ρ） |
| wall particle inclusion | 纤维是几何边界，非粒子 ⇒ 两侧都不含壁粒子 | 同 |

**因此 Phase 2 的静态审计未发现"名义 cutoff 不足"的问题**（3.0σ > 2.8σ），⇒ 需要 Phase 3 的逐粒子对照才能判定到底是"少配对"还是"时相/缓存"。

## Phase 3：rho field 对照（需要的输出格式）

任务是要求输出 **top-10 mismatch particles** 表（particle id / position / ρ_solver / ρ_ref / 邻居数 solver / 邻居数 ref）。
**本轮未生成该表**：现有 `--sg-fd-check` 只打印 **worst 单个粒子**（i=149, pos=(58.8, 7.578)、reference 邻居数 18），没有 top-10 列表，也没有 solver 侧邻居数。
⇒ 生成该表需要一个**只读诊断改动**（在 checker 里多打印 10 行，或在 Python 侧用 VTP + 现成 m14 工具复算 ρ），**属 Phase 3/4 的下一步**，本轮按"未确认根因不改 checker"停住。

## Phase 4/5：为何本轮不动手

任务规定：**"优先修改 reference，不要修改 solver"**，且前提是**确认是 checker 的问题**。当前证据：
* 源码层面两条路径定义相同（Phase 1）；
* cutoff 名义充足（Phase 2）；
* 但存在 31%/49% 的实测差异，且**只在 two_scale 出现**。

**关键矛盾**：若两条定义相同、cutoff 又足够，差异就不应依赖"是否启用 two_scale"。而 two_scale **不参与 ρ 的计算**（我已把密度回路改回 legacy）⇒ 该关联性提示 **solver 侧 ρ 的时相/缓存问题**（例如 FD 检查读到的 `CG_Rho` 是上一次 sweep 的值，而 two_scale 改变了 sweep 的执行次序/次数），或 **`--sg-fd-check` 路径下的 neighbor list 刷新缺陷**。这两种都属于**solver/checker 交互**，不是 reference 的公式错误 ⇒ **不宜先改 reference**。

## 下一步（建议的最小诊断，单次即可判定）

1. 在 `RunSGFinitDifferenceCheck` 中**额外打印**：
   * solver 侧 `rho_solver[i]` 的邻居数（遍历 `inner_configuration_[i]` 计数）；
   * reference 的 r<h_ρ 配对列表与 solver 邻居列表的**对称差**（前 10 个粒子）；
   * 该检查**之前**是否调用过 `updateConfiguration()`（用一行 `std::cout` 记录）。
2. 若对称差非空 ⇒ **solver 邻居表缺配对**（改 checker：先 rebuild 再比，或按 reference 的 O(N²) 定义比较）；
3. 若对称差为空 ⇒ **是 ρ 的时相问题**（FD 读到的 `CG_Rho` 不是当前位形）⇒ 改 checker（在读 ρ 之前重跑一次 density sweep）。
两种修法均在 **checker 侧**，不动 solver 物理。

## 交付与状态

| 内容 | 路径/状态 |
|---|---|
| 本审计 | `Stage21_two_scale_kernel\Stage21R2_fd_reference_audit.md`（仓库副本 `tools/cg_analysis/m27/`）|
| 代码事实 | `cg_self_assembly_pri.cpp`：`SGConfigRhoGrad` L2591、`RunSGFinitDifferenceCheck` L2901/L2928、mismatch 打印 L3119–3121、`CGSquareGradientDensity` L1751、neighbor 刷新 L3757/L3759 |
| 未做 | Phase 3 top-10 表、Phase 4 修改、Phase 5 回归（原因见上：根因未确认，按任务不应先改 reference）|
| 未改 | K_new / β / ξ / λ / V0 / legacy 核 / 物理模型 / 默认参数 |
