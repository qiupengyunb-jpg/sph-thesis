# Stage 13：非平衡成珠与粗化动力学框架

> **LEGACY NORMALIZATION**：packing = 0.63、nominal λ = 12；**实际 Scheme-E normalization 与 nominal λ 并不严格一致**（固定 `V0 = 1/ρ_ref = 1.1364` 被保留以维持 Stage 1–12 历史可比；φ=0.63 下 `V0·n0 = 0.911`，对应 `λ_eff ≈ 9.96`）。**本轮不修 V0**；如将来要修，必须开新 branch / 新模型版本并重建 baseline。

**日期**：2026-09-23｜**性质**：框架建立 + 已有数据统一再分析（**未改 solver、未跑新矩阵、未修 V0、未回到 RP 单模**）

## 1. 新科学问题（替代"哪个波长最快"）

1. 连续膜何时开始形成可辨认珠结构？ 2. `N_bead(t)` 如何变化？ 3. 特征尺度 `L(t)` 是否增长？ 4. 多珠状态能维持多久？ 5. 最终走向（连续膜/多珠/少珠大团/单一主团）？ 6. R0、h、ε_pf 分别怎样改变 onset / 最大珠数 / 粗化速度 / 多珠寿命 / 最终形貌？ 7. 哪些趋势在多 seed 下可重复？

**F0–F5 仅作 morphology label**；F2（multiple beads）本身即有意义状态，F3 不再是唯一目标，不得为 F3 调 detector 或参数。

## 2. 工具（本轮交付）

* `tools/cg_analysis/m23/m23_coarsening.py`：单 case → `m23_timeseries.csv` + `m23_events.csv`
  逐帧输出 `N_bead, n_b=N/Lz, mean/median spacing, CV_spacing, largest_fraction, modulation, dominant_lambda, |H_1..4|², P_low = Σ_{m=1..4}|H_m|², L_bead = Lz/N (N≥2), L_spec = 2π/⟨k⟩, ⟨k⟩ = Σ k P(k)/Σ P(k)`
  事件时间：`t_mod, t_bead2, t_bead4, t_Nmax, t_coarsen, T_F2, t_dominant`
  动力学分类：`D0` 近连续 / `D1` 长寿命多珠 / `D2` 活跃粗化 / `D3` 快速主导-坍缩
  （冻结阈值：`modulation ≥ 0.15`、`largest_fraction > 0.50`、判"稳定"需连续 ≥3 帧；分类依据 = 从 `t_Nmax` 起的 `dN/dt` 与 `d(largest_fraction)/dt`）
* `tools/cg_analysis/m23/m23_matrix.py`：matrix root → 汇总表
* 全部复用冻结工具：m14 重建、m17 **Method D**、m15 frozen peak rule；tracking 的 `TRUE_MERGE / TRUE_DEATH / DETECTION_GAP / UNRESOLVED` 语义沿用 Stage 8，但本轮结论以**宏观量**为准（不让单个事件识别误差主导）。

## 3. 已有数据统一再分析（14 例 OFAT，t=200，T=1，packing 0.63，Lz=80）

| case（变量） | t_bead2 | t_bead4 | N_max | n_b,max | t_coarsen | T_F2 | 粗化率 dN/dt | 类别 |
|---|---|---|---|---|---|---|---|---|
| ep11（基准） | 5.0 | 6.0 | 7 | 0.0875 | 5.0 | 36.8 | 0.0205 | D2 |
| ep13 | 3.0 | 9.0 | 5 | 0.0625 | 7.0 | 49.8 | 0.0207 | D2 |
| ep15 | 4.0 | 8.0 | 6 | 0.0750 | 16.0 | **80.6** | 0.0217 | D2 |
| ep17 | 3.0 | 10.0 | 6 | 0.0750 | 9.0 | 77.6 | 0.0209 | D3 |
| ep20 | 3.0 | 6.0 | 6 | 0.0750 | 5.0 | 29.9 | **0.0256** | D3 |
| h2 / h2.5 | 6.0 | 13.0 | 6 | 0.0750 | 13.0 | **99.5** | 0.0214 | D3 |
| h3 / h3.5 | 5.0 | 6.0 | 7 | 0.0875 | 5.0 | 36.8 | 0.0205 | D2 |
| h4 | 4.0 | 6.0 | 5 | 0.0625 | 4.0 | 31.8 | 0.0204 | D2 |
| R4 | 5.0 | 8.0 | 6 | 0.0750 | 8.0 | 45.8 | 0.0208 | D2 |
| R5 | 5.0 | 6.0 | 7 | 0.0875 | 5.0 | 36.8 | 0.0205 | D2 |
| R6 | 5.0 | 15.0 | 6 | 0.0750 | 13.0 | **75.6** | 0.0160 | D2 |
| R8 | 5.0 | 7.0 | 6 | 0.0750 | 11.0 | 73.6 | **0.0159** | D2 |

**长时基准**（`_stage10/probe300`，R0=5、h=2、ε_pf=13、Lz=160，t=300）：
`t_bead2 = 1.2`、`t_bead4 = 5.8`、`N_max = 11`（n_b=0.069）、`t_coarsen = 41`、`T_F2 = 72.3`、粗化率 0.027、类别 **D2**。

## 4. 参数作用地图（只写数据支持的方向）

| 参数 ↑ | 成珠 onset `t_bead2` | 最大珠密度 `n_b,max` | 粗化率 | F2 寿命 `T_F2` | 最终形貌 |
|---|---|---|---|---|---|
| **ε_pf**（11→20） | **↓ 提前**（5→3） | ≈ 平（0.0625–0.0875） | **↑**（0.0205→0.0256） | ↑ 到 15 峰值（80.6）→ 20 崩塌（29.9） | D2 → **D3**（≥17） |
| **h**（2→4） | **↓ 提前**（6→4） | ↓（0.075→0.0625） | ≈ 平（0.0214→0.0204） | **薄膜最长**（99.5 @h≤2.5）→ 厚膜 31.8 | D3（h≤2.5）→ D2 |
| **R0**（4→8） | ≈ 无趋势（均 5.0） | ≈ 平 | **↓ 变慢**（0.0208→0.0159） | ↑（45.8→73.6） | 均 D2 |

**UNRESOLVED**：onset 与 n_b,max 的 R0 依赖（数据无趋势）；`L_spec` 的增长方向（各 case 斜率符号不一致，−0.0027…+0.0076）。

## 5. 最终回答（§二十二）

**A. 现有数据中是否存在清晰 slow / medium / fast 三类动力学？→ 只有 medium（D2）与 fast（D3），没有 slow（D1）。**
14 例中 D2 = 11、D3 = 3，**没有任何 case 满足 D1（长寿命多珠）**；T_F2 最长 ≈ 99.5 t（h=2/2.5），远小于观测窗 200 t ⇒ 多珠状态都不"长寿"。

**B. 哪个参数对成珠 onset 影响最大？→ ε_pf（与 h 次之）。**
ε_pf 11→20 把 `t_bead2` 从 5.0 提前到 3.0（−40%）；h 4→2 从 4.0 提前到 6.0…（h 更大时更早，4.0 @h=4 vs 6.0 @h=2，方向与直觉相反，记录为数据事实）；R0 **无趋势**。

**C. 哪个参数对粗化速率影响最大？→ ε_pf（快）与 R0（慢），但幅度都不大（±25%）。**
ε_pf 11→20：0.0205→0.0256（+25%）；R0 4→8：0.0208→0.0159（−24%）；h 2→4：0.0214→0.0204（−5%，UNRESOLVED）。

**D. 哪个区域最适合研究 long-lived multi-bead？→ 薄膜 + 中强壁 + 较大 R0：h ≈ 2–2.5σ、ε_pf ≈ 15、R0 ≈ 6–8。**
依据：该组合给出最长 `T_F2`（h≤2.5σ 的 99.5 t、ε_pf=15 的 80.6 t、R6/R8 的 75.6/73.6 t）与最低粗化率（R8 的 0.0159）。**但注意：即便此处也只是 D3/D2，不是 D1** ⇒ 该区域是"最接近长寿命多珠"的候选，不是已确认的长寿命区。

**E. PC2 下一轮最值得跑的 6–9 个 case？**（围绕上表，覆盖三类动力学；**统一 Lz=160、t_end=1000、1 seed 起**）

| # | R0 | h | ε_pf | 为什么选它 / 区分什么动力学 |
|---|---|---|---|---|
| P1 | 5 | 2 | 13 | **基准**（与 probe300 同参数、Lz=160）→ 校准长时行为与 t=300 的衔接 |
| P2 | 5 | 2 | 20 | ε_pf 上端：区分"更快粗化/更早 D3"是否在 Lz=160 下仍成立 |
| P3 | 5 | 2 | 11 | ε_pf 下端：与 P2 配对，量化 ε_pf 对粗化率的符号与幅度 |
| P4 | 5 | 4 | 15 | 厚膜对照：区分"薄膜才有长 T_F2"是否成立 |
| P5 | 5 | 2.5 | 15 | **长寿命候选**（薄膜+中壁）：检验 T_F2 能否突破 100 t |
| P6 | 8 | 2.5 | 15 | 大 R0 长寿命候选：区分"R0↑ 减慢粗化"能否给出 D1 |
| P7 | 8 | 2 | 13 | 大 R0 + 薄膜：另一条 D1 候选路径 |
| P8 | 4 | 2.5 | 15 | 小 R0 对照：区分 curvatur 更强时是否反而更快坍缩 |
| P9 | 5 | 2 | 15 | 中壁基准（与 P5 只差 h）→ 分离 h 与 ε_pf 的贡献 |

> 9 例中 **P5/P6/P7 是 D1（长寿命多珠）的主要候选**；P2 是 D3（快速坍缩）候选；P4/P9 是"近连续膜/弱演化"（D0）候选。

**F. 是否已经具备"参数 → 动力学 → 最终形貌"完整研究链？→ 框架已具备，证据链尚未闭合。**
已具备：统一指标（§2）、事件定义（§4 表头）、动力学分类 D0–D3、参数方向图（§4）、可复用工具 m23。
未具备：① **D1 区间的实测存在性**（当前 14 例 + probe300 全为 D2/D3）；② 多 seed 可重复性（尚未做）；③ `L_spec` 趋势不稳定（UNRESOLVED）；④ legacy normalization 与 nominal λ 的偏差（已在文首声明，未修）。

## 6. Production 协议（给 PC2）

| 项 | 值 |
|---|---|
| Lz | **160**（Stage 4 已证明 80 太短限制可观察珠数；300 成本高且未带来波长选择 ⇒ 冻结 160）|
| t_end | **1000**（第一阶段），1 seed |
| 输出 | 固定 Δ=1.0（**足够辨认 `t_bead2/t_Nmax/粗化区间`**）；若需更密 onset，可只在早期用 Δ=0.2（solver 不支持变频则统一 Δ=1.0）|
| seed | 第一阶段 1 seed；随后从 D1/D2/D3 三类各选 1–2 例补到 3 seeds（`seed = 20260916 + {0,1,2}`）|
| 命令 | `<exe> ... --axisymmetric=filmonly --init=film --fibre-radius-const=<R0> --film-thickness=<h> --wall-depth-d0=<ε_pf> --cluster-packing=0.63 --axisymmetric-interface-scheme=difference_energy --temperature=1 --domain-length=160 --domain-height=40 --end-time=1000 --frames=1000 --threads=1 --seed=...` |
| 分析 | `python m23_coarsening.py <case_dir> --R0 <R0> --Lz 160 --label <name>`→ `m23_events.csv`；全部 case 汇总 `python m23_matrix.py <matrix_root> --out table.csv` |
| 判读 | 主看 `N_bead(t)/n_b`、`L_bead`、`L_spec`、`largest_fraction(t)`、`T_F2`、`dynamics_class`；**不再以 F3 占比为目标** |

## 7. 术语边界（§十七）

可以说：capillary / interface-energy-driven morphology evolution、bead formation、coarsening、non-equilibrium restructuring。
谨慎：RP-like morphology。**禁止**："classical RP wavelength selection confirmed"、"fastest-growing RP mode reproduced"。

## 8. 交付与 commit

| 内容 | 路径 |
|---|---|
| 框架文档 | 本文件（仓库副本 `tools/cg_analysis/m23/`）|
| 工具 | `tools/cg_analysis/m23/m23_coarsening.py`、`m23_matrix.py` |
| OFAT 统一再分析表 | `stage13_ofat_map.csv`（14 例）|
| 逐 case 时序/事件 | `E:\哈哈\_stage2\ofat\<case>\m23_{timeseries,events}.csv`、`E:\哈哈\_stage10\probe300\m23_*.csv` |
| commit | 见本轮提交（仅新增分析工具与文档）|
