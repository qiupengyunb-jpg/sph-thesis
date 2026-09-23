# Stage 1 周期结构搜索 —— PC1 → PC2 交接

**日期**：2026-09-23｜**状态**：PC1 研发准备完成，**可交付批量搜索**（PC1 不再跑长算例）
**Stage 1 目标**：不是先证明 RP，而是**找到一个能从圆柱连续液膜出发、产生"多个、有间距、能维持一段时间"珠状结构的**最简模型。

---

## 1. 分支 / commit / 构建

| 项 | 值 |
|---|---|
| 分支 | **`feature/cg-axisym-singlemode-scheme-e`** |
| 本轮 commit | `2d5d375d8`（M1.4）＋ **`a43ed1466`**（Stage 1：broadband 初态 + m15 珠结构分析工具） |
| 基线 | `20eec0757`（M1.3R Scheme E 进 solver，默认 `moving_j`）|
| solver 源 | `tests/user_examples/test_2d_cg_self_assembly_pri/cg_self_assembly_pri.cpp` |
| exe | `build_msvc145_current/.../bin/Release/test_2d_cg_assembly_pri.exe`，SHA256 `A9803AA0A479781D6116BDD2FADFF89589743C305960C6853C4946625028A546` |
| 构建 | VS 2026 + 工具集 14.51.36231；`cmake --build . --config Release --target test_2d_cg_assembly_pri` |
| **默认 scheme 仍是 `moving_j`** | 必须**显式**加 `--axisym-interface-scheme=difference_energy` |

## 2. 冻结的模型

`--case=1 --morse-depth-d=3 --interface-gradient-lambda=12 --interface-gradient-h=2.8`
等半径 axisymmetric cylinder + Scheme E；WCA / Morse pp / Morse wall / BAOAB+OU / m=1 全不动。
**T = 1**（Stage 1 允许直接有限温；T=0 会冻结，只作机制背景）。

## 3. 两类初态

| 初态 | 命令 | 说明 |
|---|---|---|
| **A 均匀膜（首选）** | `--film-perturb-amp=0` | 靠热涨落自发选择结构 |
| **B 小幅 broadband** | `--film-perturb-amp=0.02 --film-broadband=1` | 8 个 mode（1,2,3,4,5,6,8,10）、golden-ratio 相位、等权重；`--film-perturb-amp` 是**总幅度预算**，`|f−1| ≤ A` 逐点成立 ⇒ 与单模同 A 有相同的 WCA 安全裕度；不偏置任何理论波长 |

单模（M1.4 遗留）仍可用：`--film-perturb-amp=0.01 --film-perturb-mode=m`（A0 ≤ 0.01 才能保证全 mode 最小对距 > WCA 核心 1.12246）。

## 4. 首轮 screening set（9 例，分层）

| # | R0 | requested h | ε_pf | 科学目的 |
|---|---|---|---|---|
| S1 | 5 | 3 | 11 | 强曲率 + 薄 + 弱 wall：曲率是否能在低液量下把膜切成多珠 |
| S2 | 10 | 5 | 20 | **主基准**：与 M1.4 工作点一致，作为所有比较的参照 |
| S3 | 20 | 7 | 30 | 大半径 + 厚 + 强 wall：接近平板极限，检验是否退化为普通 dewetting |
| S4 | 10 | 5 | 20 | 同 S2 但用 **broadband 初态**：检验启动速度是否受初态影响（对照 S2） |
| S5 | 5 | 5 | 20 | R0 对照（固定 h、ε_pf）：小 R 是否改变珠间距/稳定性 |
| S6 | 20 | 5 | 20 | R0 对照（固定 h、ε_pf）：大 R 是否抑制多珠 |
| S7 | 10 | 3 | 20 | 薄膜对照（固定 R0、ε_pf）：薄膜是否更易成多珠 |
| S8 | 10 | 7 | 20 | 厚膜对照：厚膜是否 coarsening 成少量大液滴 |
| S9 | 10 | 5 | 11 | 弱 wall：残余液膜(neck)是否消失、是否变成孤立液滴 |

分层执行：**Stage A** S1–S4（代表点短跑 t=200）→ 有 F2/F3 倾向的点扩到 **Stage B**（t=600–1000）→ 候选周期结构进 **Stage C**（≥3 seeds）。**不要**一次把 R0×h×ε_pf 全正交展开。

## 5. 每个 case 的运行建议

| 项 | 建议 |
|---|---|
| dt | `0.0075` |
| Stage A 时长 | `--end-time=200 --frames=200`（N ≈ 250–700，约 20–60 s/例/线程）|
| Stage B 时长 | `--end-time=1000 --frames=200`（仅在 Stage A 出现 F2/F3 倾向时）|
| 输出频率 | 每 1 个时间单位 1 帧（`frames = end_time`）——珠结构时间尺度 ~10¹ τ |
| seed | Stage A/B 用 `20260916`；Stage C 用 `20260916 + {0,1,2}`（≥3）|
| 并行 | 每进程 1 线程 × 多进程（不要每进程多线程）|
| 必须打开 | `--sg-budget-every=500`（残差预算 B_r 时间序列）|

## 6. 自动分析

```powershell
# 逐帧珠结构 + F0–F5 分类 + 周期判据
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m15\m15_beads.py <run_dir> `
    --R0 <R0> --Lz 80 --t-lo 20 --t-hi <end_time>
# 输出: m15_beads.csv（逐帧）、m15_summary.csv（模态类别/占时比例/是否周期结构）

# 单模线性响应（M1.4，可选用）
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m14\m14_axisym_modes.py <run_dir> --R0 <R0> --Lz 80 --target-m <m>
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m14\m14_ensemble.py --mode <m> --pert <dirs...> --base <dirs...>
```

## 7. F0–F5 形貌分类（自动，m15 内实现）

| 类 | 定义（同一帧） |
|---|---|
| **F0** | 连续膜：调制 `A_dom/⟨h⟩ < 0.05` 且 bead ≤ 1 |
| **F1** | 弱调制/丘陵：bead ≤ 1，调制 < 0.35（或 2 峰但调制 < 0.30）|
| **F2** | 多珠但不规则/未持续：bead ≥ 3，未同时满足 F3 门槛 |
| **F3** | **准周期多珠（目标）**：bead ≥ 4 **且** CV_spacing ≤ 0.35 **且** largest_fraction ≤ 0.45 **且** 调制 ≥ 0.15，**且** 持续 ≥ 20 个时间单位 |
| **F4** | 少量大液滴：bead = 2 且调制 ≥ 0.30 |
| **F5** | 单一主团：bead ≤ 1 且调制 ≥ 0.35 |

## 8. 周期结构工作判据（草案，待主线确认后 PC2 批量使用）

在**声明窗口**（默认 t ∈ [20, end_time]）内同时满足：

1. `bead_count ≥ 4`；
2. `CV_spacing ≤ 0.35`；
3. `largest_fraction ≤ 0.45`（没有单个珠子吞掉大部分液体）；
4. 调制 `A_dom/⟨h⟩ ≥ 0.15`；
5. `dominant_lambda` 在窗口内漂移 ≤ 25%；
6. 满足 1–4 的状态**连续保持 ≥ 20 个时间单位**。

未同时满足 1–6 ⇒ 不得称周期结构（“某一帧峰多”不算）。
**这些阈值是先声明后使用的，不得为某个 case 单独调。**

## 9. 自动延长 / 淘汰 / 多 seed

| 触发 | 动作 |
|---|---|
| Stage A 结束时 modal class ∈ {F2,F3} 或 `median_bead_count ≥ 2` | → Stage B 延长到 t = 1000 |
| Stage A 结束时 modal class ∈ {F0,F1} 且 modulation < 0.1 | 判定"启动不足"；**先**改 broadband 初态重跑一次，仍无则淘汰 |
| guard > 0 或 hard > 0 或爆炸 | **INVALID**，淘汰并记录（不计入 stability map）|
| `largest_fraction > 0.8` 且持续 | 已粗化（F4/F5 方向），淘汰 |
| Stage B 出现 F3 且 `max_F3_hold_time ≥ 50` | → Stage C：≥3 seeds，比较 ⟨bead_count⟩/⟨CV_spacing⟩ |

## 10. PC1 已知限制（**必须带进分析**）

1. **h(z) 重建噪声地板 ≈ 0.1–0.3σ**：均匀膜在 t≈10 之后（外层弛豫 + 层屈曲）重建谱在 m=1–12 上普遍有 0.1–0.45σ 的伪幅值；frame-0 干净。
   ⇒ **只有调制 ≳ 0.3σ 的珠结构才可信**；`modulation < 0.15` 的帧不得当结构证据。
   ⇒ 判定"是否成珠"要看 `bead_count`/`CV_spacing`/`largest_fraction` 的组合，而不是单看 A_dom。
2. **broadband 种子幅度（0.01σ）远低于该噪声地板** ⇒ S4 与 S2 的差异只能在**后期**（结构长到 ≳0.3σ）才可判读。
3. 高模（m ≥ 8，λ ≤ 10σ）在 Lz=80、N≈350 下重建信噪比差（R²<0.1）。
4. 残差预算 B_r = |⟨Fr_measure⟩|/RMS(F_int) ≈ 0.011–0.014（M1.4 实测），不随模式/时间增长。

## 11. 需要停下问用户的情况（沿用任务书 §12）

改 λ / h_ρ / ε_pp；改 Scheme E 数学形式；新增物理作用；变半径纤维；回到 RP 理论验证主线；
或**参数搜索暗示现有模型根本无法产生多珠结构**。

---

## 附：PC1 本轮已做的短测（smoke test，§10）

| case | 参数 | 结果 |
|---|---|---|
| S1 | R0=5, h=3, ε_pf=11 | t=50：6 667 步 / 3.0 s，**guard=hard=0**，F1(0.71)+F5(0.29)，尚未成珠 |
| S2 | R0=10, h=5, ε_pf=20 | t=50：5.7 s，guard=hard=0，F1=1.00 |
| S3 | R0=20, h=7, ε_pf=30 | t=50：8.2 s，guard=hard=0，F1=1.00 |
| S4 | S2 + broadband（A=0.02） | t=50：5.6 s，guard=hard=0，F0(0.10)+F1(0.90) |

**结论**：init 正常、不爆炸、guard/hard=0、分析链（m15）能读并能分类、输出与参数记录正确 ⇒ **smoke test 达标，PC1 到此停止**（短测时间仅 t=50，不足以判形貌，仅验证链路）。
