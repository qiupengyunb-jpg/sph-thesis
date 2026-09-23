# PC2 Stage 2 定向矩阵（基于 PC1 单因素短测）

**日期**：2026-09-23｜**依据**：`Stage2_mechanism_report.md`（单因素表 `stage2_ofat_table.csv`）
**原则**：只组合**已被短测支持**的方向，不随机扫描。

---

## 1. 短测给出的方向（只用数据支持的）

| 方向 | 证据 | 类型 |
|---|---|---|
| ε_pf **13–17**（而非 11 或 20） | birth 17→**37/20/31**、max bead 4→**6/6/8**；ε_pf=20 反而寿命更短、F4/F5 上升 | **N+** |
| **薄膜 h ≈ 2–2.5σ** | 2 层膜 birth **42** vs 3 层膜 17 | **N+** |
| **R0 = 4–5**（不要 6–8） | 迁移速度 0.571/0.641 vs 0.708/0.751；寿命 6.0 最佳；R0=8 有 34 次检测闪烁 | **避免 M−** |
| **保持 T=1、packing=0.63、λ=12（冻结）** | — | — |

**没有**被短测支持的方向：任何"降低合并"的参数（本轮 UNRESOLVED）。

---

## 2. 定向矩阵（9 例 + 2 例延长 = 11 例）

固定：`--case=1 --morse-depth-d=3 --interface-gradient-lambda=12 --interface-gradient-h=2.8 --axisym-interface-scheme=difference_energy --cluster-packing=0.63 --dt=0.0075 --frames=<end_time> --sg-budget-every=1000 --threads=1`

| # | R0 | requested h | ε_pf | Lz | domain-height | end_time | 为什么选它 |
|---|---|---|---|---|---|---|---|
| M1 | 5 | 2.5 | **13** | 120 | 40 | 400 | **N+ 组合主力**：薄膜(2 层)+中壁；短测 birth 42、寿命 3 —— 检验"多珠能否维持"|
| M2 | 5 | 2.5 | **15** | 120 | 40 | 400 | 同上但壁更强一点：验证 F3 占时是否随 ε_pf 出现峰值（短测峰值在 15）|
| M3 | 5 | 2.5 | 11 | 120 | 40 | 400 | N+ 的**对照**（无中壁增强），分离"薄膜"与"中壁"各自的贡献 |
| M4 | 4 | 2.5 | 13 | 120 | 40 | 400 | 迁移最慢的 R0 + 薄膜 + 中壁：唯一曾测到真实 merge 的谱系 |
| M5 | 5 | 3 | 13 | 120 | 40 | 400 | 基准膜厚下的 N+ 组合（与 Stage 1 S1 同 h，便于对照历史）|
| M6 | 5 | 3 | 20 | 120 | 40 | 400 | **负对照**：已知更快走向 F4/F5，检验 PC1 的"寿命缩短"判读 |
| M7 | 6 | 2.5 | 13 | 120 | 40 | 400 | R0 稍大：检验"R0 ↑ 迁移 ↑"是否在薄膜下依然成立 |
| M8 | 5 | 2.5 | 13 | **80** | 40 | 400 | **Lz 对照**：短 Lz 是否因周期限制而人为减少珠数 |
| M9 | 5 | 2.5 | 13 | 120 | 40 | **800** | **延长档**：短测只见 birth，未见"维持"；800 用来判 F3 是否自发出现 |
| M10 | 4 | 2.5 | 15 | 120 | 40 | **800** | 最保守组合（慢迁移 + 薄 + 中偏强壁）的延长档 |
| M11 | 5 | 2.5 | 17 | 120 | 40 | 400 | ε_pf 上限侧：确认 17 是否比 13/15 更好或已开始变差 |

> M8 与 M1 只差 Lz，用于分离"有限周期长度"对珠数与 spacing 统计的影响。

---

## 3. 每个 case 的分析命令与读表

```powershell
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m15\m15_beads.py <run_dir> --R0 <R0> --Lz <Lz> --t-lo 40 --t-hi <end_time>
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m15\m15_beadtrack.py <run_dir> --R0 <R0> --Lz <Lz> --t-lo 40 --t-hi <end_time> --label M1
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m15\m15_collect.py --out table.csv M1=<dir>=<R0> M2=...
```

看三件事（按优先级）：

1. `m15_track_summary.csv` 的 **birth_count / median_bead_lifetime / median_migration_speed**（机制量）；
2. `m15_track_merge_verify.csv` 的 **monotone_decreasing**（合并是否真实）；
3. `m15_summary.csv` 的 **periodic_structure / bead4_hold_time / median_CV_spacing**（是否 F3）。

---

## 4. 判定与流程

| 观察 | 结论 | 下一步 |
|---|---|---|
| M1/M2/M4 出现 `F3` 占时 ≥ 0.5 且 `bead4_hold_time ≥ 50` | 找到可行周期结构区 | → 3 seeds 复现（`seed=20260916+{0,1,2}`）|
| M1–M11 都只有 F2（bead ≥ 4 但 CV > 0.35） | 成核够了、规则度不够 | 回报用户（§十五 第 2 条）：需要新机制或接受 F2 |
| `merge_verify` 仍几乎为空 | 合并仍不可测 | 先把 Lz/分辨率提高再判（分析侧改动，不改物理）|
| guard/hard > 0 或爆炸 | INVALID | 淘汰重跑 |

**禁止**：不要在没有新证据前扩大 R0×h×ε_pf 全矩阵；不要用加大 A0 预置多珠；不要改 λ/h_ρ/ε_pp/Morse/WCA/阻尼。
