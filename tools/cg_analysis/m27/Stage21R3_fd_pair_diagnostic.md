# Stage21R-3 —— FD ρ mismatch 只读诊断（设计与判定树）

**日期**：2026-09-24｜**性质**：只读诊断（**未改 solver 物理 / K_new / kernel / λ / V0 / β / ξ / 默认行为 / reference 算法**）
**执行状态（如实）**：**本轮的 C++ 诊断已设计完成，但"加打印 → 重编 → 跑失败配置"三步未在本轮执行完**（见 §4「未执行原因与可直接套用的补丁」）。因此 §1–§3 是**诊断设计 + 已有数据可推出的部分结论**，§4 给出下一步一条命令即可完成的补丁。

---

## 1. worst particle 对比（现有数据）

`--sg-fd-check` 已打印（two_scale, h=5, ξ=2.9, β=0.3）：

```
worst grad mismatch at i=149  pos=(58.8, 7.57812)   reference neighbours within h_rho=18
   rho:  reference=0.526621   solver=0.44332          ← 差 0.08330（-15.8%）
   grad: reference=(-1.2e-15,-2.4e-17)   solver=(-0.16742, 2.78e-17)
```

**已有信息**：reference 邻居数 = **18**；ρ 差 = **0.08330**（参考值的 15.8%）；grad 参考 ≈ 0 而 solver ≈ −0.167（纯径向）。
**缺失信息**（Phase 1 要补）：solver 侧邻居数、particle OriginalID、pair 列表对称差、以及配置刷新状态。

## 2. pair set 对称差（Phase 1–2 要输出的内容）

需要的输出（对 worst particle）：`reference neighbor ids` / `solver neighbor ids` / `intersection` / `reference-only` / `solver-only` → `pair_difference_report.csv`。
**判定**：
* `reference-only ≠ ∅` ⇒ **情况 A**：solver 邻居表与 reference 不一致（缺配对）；
* 两者相同而 ρ 仍不同 ⇒ **情况 B**：同一 pair 集下**密度累加**不一致（缓存/权重/调用次数）。

**来自现有数据的算术提示（供 A/B 判定参考）**：该粒子 18 个 h_ρ 内邻居、最近邻距 ≈1.2σ、次近 ≈2.08σ、第三壳 ≈2.4σ、第四壳 ≈2.77σ（W 在此趋于 0）。
若 ρ 差的 0.0833 恰好等于**剔除某一完整近邻壳**（6 个粒子）的权重和，则强烈指向**邻居表缺壳**（A）；
若差值落在"任意壳都不匹配"的零散量级，则更可能是**累加口径/权重**问题（B）。

## 3. configuration 刷新状态（Phase 3 要确认）

`--sg-fd-check` 路径的执行顺序（代码事实）：

```
pair_interaction.exec();            // 力
wall_force.exec();
exec_interface_energy(0.0);         // ← 内部：sg_density.exec() → 写 CG_Rho / CG_RhoGrad
if (sg_force_dump) ...
if (sg_fd_check) { RunSGFinitDifferenceCheck(..., CG_Rho, CG_RhoGrad, ...); return 0; }
```

⇒ 读 `CG_Rho` **紧跟**在密度 sweep 之后，**同一次 t=0 位形**；`updateCellLinkedList()`/`updateConfiguration()` 只在 BAOAB 主循环内（L3757/L3759）或初始化后调用一次。
**要确认的一点**：FD 检查发生的这一次密度 sweep，其邻居表是**初始化后构建的那一份**（未被 FD 的虚拟位移污染）——这一点需要 Phase 1 打印的"配置刷新标记"来确证。

## 4. 未执行原因与**可直接套用的补丁**

**未执行原因**：本轮上下文体量已不足以完成「改 C++ → 重编（~45 s）→ 跑失败配置 → 解读」这一整链，且该改动必须重编才能生效；我选择**不留下半成品代码**（改而不编译会让仓库处于不可验证状态）。

**下一步只需一次改动**（纯诊断、默认行为不变）：在 `RunSGFinitDifferenceCheck`（`cg_self_assembly_pri.cpp` L2901）中，紧邻现有 `worst grad mismatch` 打印处（L3119–3121）加入：

```cpp
    // ---- Stage21R-3 read-only diagnostic --------------------------------
    // worst RHO mismatch (separate from the grad mismatch above)
    size_t rho_worst = 0; Real rho_worst_d = -1.0;
    for (size_t i = 0; i < n; ++i)
    {
        const Real dd = std::abs(rho_ref[i] - rho_solver[i]);
        if (dd > rho_worst_d) { rho_worst_d = dd; rho_worst = i; }
    }
    {
        const Real hh = RhoSmoothingH();
        const Real lzz = DomainLength();
        std::vector<size_t> ref_ids;
        for (size_t j = 0; j < n; ++j)
        {
            if (j == rho_worst) continue;
            Vecd d = p[rho_worst] - p[j];
            d[0] -= lzz * std::round(d[0] / lzz);
            if (d.norm() < hh) ref_ids.push_back(j);
        }
        std::cout << "  [stage21R3] worst rho mismatch i=" << rho_worst
                  << " pos=(" << p[rho_worst][0] << "," << p[rho_worst][1] << ")"
                  << " rho_ref=" << rho_ref[rho_worst]
                  << " rho_solver=" << rho_solver[rho_worst]
                  << " d_rho=" << rho_worst_d
                  << " ref_neighbours=" << ref_ids.size() << "\n";
        std::cout << "  [stage21R3] reference neighbour ids:";
        for (size_t k = 0; k < ref_ids.size() && k < 24; ++k) std::cout << " " << ref_ids[k];
        std::cout << "\n  [stage21R3] solver CG_Rho (same ids) :";
        for (size_t k = 0; k < ref_ids.size() && k < 24; ++k) std::cout << " " << rho_solver[ref_ids[k]];
        std::cout << "\n";
        // if the sweep object exposes its neighbourhood, also print
        //   dens.particles()->... / inner_configuration_[rho_worst].current_size_
        // and the symmetric difference of the pair lists here.
    }
```

（若 `InteractionWithUpdate<CGSquareGradientDensity>` 能取到 `InnerRelation`，则再补打印 `inner_configuration_[rho_worst].current_size_` 与 `j_[]` 列表，即可一次性给出 **solver 侧邻居数 + 对称差**；否则以"reference ids 上的 ρ 求和是否等于 rho_solver"作为等价判据：**求和 ≠ rho_solver ⇒ 情况 A；求和 = rho_solver 而 ≠ rho_ref ⇒ 情况 B**。）

**判定树（Phase 4）**
* **A：pair set 不一致** ⇒ 只查/改 **neighbor rebuild 与 checker 调用顺序**（例如在 `RunSGFinitDifferenceCheck` 开头显式 `particles_inner.updateConfiguration()` 后再读 `CG_Rho`，或让 checker 自己按 O(N²) 定义求 ρ 作为唯一口径）；
* **B：pair set 一致、累加不一致** ⇒ 依次查 **rho cache（是否读到上一次 sweep 的值）/ particle volume（V0 是否参与 ρ）/ weight 调用（RhoKernelW vs 其他）/ axisymmetric factor（ρ 是否被 J 加权——不应加权）**。

## 5. 回归要求（Phase 5）

本轮**未跑**：生产算例（禁止）、周期结构分析（禁止）、参数调整（禁止）。
下一轮执行上述补丁后，回归要求仍为：`strip` 与 `A-1` 的 VTP SHA256 **逐位一致**（历史基线 `6F0181C3/DDE80488/0AEE50C0`、`DC72C786/16015253/ED462B63`），并且 two_scale 的 FD 在 δ=1e-4/3e-5/1e-5/3e-6 上**随 δ 收敛**。

## 6. 交付

| 内容 | 路径/状态 |
|---|---|
| 本诊断设计与判定树 | `Stage21_two_scale_kernel\Stage21R3_fd_pair_diagnostic.md`（仓库副本 `tools/cg_analysis/m27/`）|
| worst particle 现有数据 | §1（rho_ref 0.526621 / rho_solver 0.44332 / ref 邻居 18，i=149, pos=(58.8,7.578)）|
| 待补 | solver 邻居数、pair 对称差、配置刷新标记（§4 补丁，需重编）|
| 未改 | solver 物理 / K_new / kernel / λ / V0 / β / ξ / 默认行为 / reference 算法 |
