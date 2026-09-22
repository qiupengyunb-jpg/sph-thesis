# M1.3R WORKLOG

自主长时任务：轴对称测度一致性模型重设计 + 原型实现 + 验证闭环。

| 项 | 值 |
|---|---|
| 起始 HEAD | `f4fa2140d`（分支 `feature/cg-axisymmetric-cylinder`） |
| 新分支 | **`feature/cg-axisymmetric-measure-redesign`** |
| Checkpoint tag | **`m13r-checkpoint-before-redesign`** → `f4fa2140d`（随时可回） |
| 保护 | 不 force push / 不 rebase / 不 amend / 不 reset --hard / 不删旧报告 / 不覆盖原始数据 |

---

## 2026-09-22 | 阶段 0：问题重述与候选初筛

**change**：读 N1–N3 报告与 v2 设计；把 Scheme A 的失败重新定位到"能量本身"这一层。

**关键重述（这是本轮设计的出发点）**

Scheme A：`F_A = (λV0/2) Σ_i J_i |G_i|²`，`J_i = r_i/R0`。

* 在**均匀物理态**下，连续理论给 `|∇ρ| = 0` ⇒ `F_A = 0`、净径向力 = 0；
* 但**离散** ⟨|G_i|²⟩ = 1.776e-3 ≠ 0（纯核和残差）；
* 而 `J_i` 乘在粒子上、随粒子移动 ⇒ 变分产生 `② = −(λV0/2R0)|G_i|²r̂`；
* ⇒ **② 把离散的 |G_i|² 残差直接转成一个 O(1/R0) 的系统向内力。**

**⇒ 本质结论（本轮的总纲）**：**任何"权重 × |G|²"形式的离散界面能，只要 `|G_i|²` 在均匀态不为零，就必然把格点残差转成假力。**
修法不能是"删 ②"或"加一个 −②"（都属禁止的伪修复），而必须是**换一个在均匀态能量恒为零的离散泛函**。

**候选初筛（T0 层，只看数学定义）**

| 方案 | 均匀态能量 | 判定 |
|---|---|---|
| A moving-J | `(λV0/2)ΣJ|G|²`，`|G|²≠0` ⇒ **能量本身不为零** | **FAIL（已定，第 N3 轮）** |
| B material/reference J | 同上（`|G|²≠0`），且引入参考格点标签 | **先验差**：均匀态能量仍不为零；且 T7 大形变不成立 ⇒ **REJECTED** |
| **C measure-consistent ring particle** | 需改 mass/Θ/FDT | 可能是最终形态，但属 Level C，先只在 toy 层评估 |
| D reproducing gradient | 需保证常数场 `G=0` | `G^D=ΣV_j(ρ_j−ρ_i)∇W` 在常数场**精确为零** ✓ 但力的完整变分复杂 |
| **E pairwise-difference energy** | `(ρ_i−ρ_j)²` 在均匀态**恒等于零** | **最有希望**（见下） |
| F finite-volume/fixed cell | measure 固定在 cell 上 | 偏离纯 particle 模型，复杂度高 |
| G background-subtracted | 需 `B_i` 定义 | 若 `B` 依赖初始格点/ID/历史 ⇒ 淘汰 |

**Scheme E 的核心优势（先写在纸上）**

```
F_E = (λ/4) Σ_i Σ_j V_i V_j J_ij (rho_i - rho_j)^2 K_ij
      J_ij = J((r_i + r_j)/2)      ← 配对中点，对称
      K_ij = K(r_ij)               ← 归一化使 Σ_j V_j K_ij r_ij^2 = 2
```

* **均匀态 `ρ_i = ρ_j`（完美格点上精确成立）⇒ 每一项 `(ρ_i−ρ_j)² = 0` ⇒ 能量恒为零，力也恒为零（不是抵消）**；
* `J` 只以 `J_ij`（配对中点）出现 ⇒ **没有 `dJ/dr` 单粒子支**，不存在 term ② 那种项；
* 配对对称 ⇒ 力逐对反对称 ⇒ 动量守恒；
* 大形变下 `J_ij` 用**当前位置**的中点 ⇒ **不依赖参考格点标签**（T7）；
* 连续极限：`(1/2)∬(ρ−ρ')²K dA dA' = (1/2)∫|∇ρ|²dA`（当 `∫u²K dA = 2`）；
* 计算成本与现有 SG 力同阶（同一个邻居循环），甚至更简单（不需要累加 G）。

**decision**：把资源集中在 **Scheme E**（主）与 **Scheme D**（备），先做 Python toy。
