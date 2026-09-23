# Stage 2 参数搜索：给 PC2 的注意事项

**日期**：2026-09-23｜**基线**：`feature/cg-axisym-singlemode-scheme-e` @ `763020872`（+ 本轮工具 commit）
**配套**：`PC2_STAGE1_PERIODIC_SEARCH_HANDOFF.md`（命令/矩阵/判据）、`Stage2_precheck_report.md`（检查与复算）

---

## 0. 一句话

Stage 1 的 S1 已经**能成核多珠但不规则**；机制归因是 **成核不足 + 珠合并**（不是驱动力不足）。
所以 Stage 2 的搜索目标是**提高成核数密度、减慢合并**，而不是继续加界面驱动力。

---

## 1. 三条必须先记住的操作性要求（本轮检查发现）

1. **`--domain-height` 必须显式给**（旧判据对轴对称过严）：

| R0 | 推荐 `--domain-height` |
|---|---|
| 5 | 40 |
| 10 | 40 |
| **20** | **60**（默认 40 会误报 `fibre does not fit inside the domain`）|
| 20 + h=7 | 60 |

   规则：`H ≥ R0 + 0.5 + h_film + 2.1 + 2.0`（整数向上取）。
2. **`--cluster-packing` 必须显式写**：默认 **0.63**（上一版 handoff 误写的 0.70 已更正）。
   比较不同算例时以 `axisym_film_selfcheck.csv` 的 `realised_thickness` 为准（层量化会差一层）。
3. **`--domain-length` 必须显式给**（= Lz，决定可容纳的珠数与波数分辨率）：建议 **Lz = 120**（Stage 2 起）。

---

## 2. 搜索方向（按机制归因给出，不是无脑加驱动力）

| 目标 | 手段 | 依据 |
|---|---|---|
| **提高成核数密度** | ↑ R0（5→10→20）：同样 h 下可用波长更多、单个珠占液量更小 | S1 只有 4 个珠且 t=190 才凑齐 |
| | ↓ h（3 → 2.5）：液量少 ⇒ 珠数上限高 | P2 强壁下 h 不变但珠数降到 2，说明液量/结合能竞争敏感 |
| | ↑ Lz（80→120）：给更多模式的成核机会，同时改善 spacing 统计 | S1 在 Lz=80 下 spacing≈20σ，只有 ~4 个样本 |
| **减慢合并** | ε_pf 取 **11–20 的中段**：弱壁保留更多珠（P1 4 珠 vs P2 2 珠），强壁则抑制 bead 迁移但促进"少而大" | P1/P2 对照 |
| | 若目标是"持久残留膜（neck）"：ε_pf → 20–30，但**接受珠数变少** | P2：F5 占 71% |
| **不要做** | 不要再加界面驱动力（λ 冻结）；不要靠加大 A0 强行预置多珠 | 调制已达 0.57，D 不成立 |

---

## 3. 建议的 Stage 2 首轮矩阵（10 例，仍为分层筛选）

固定：`ε_pp=3, λ=12, h_ρ=2.8, T=1, dt=0.0075, scheme=difference_energy, packing=0.63, seed=20260916`。
每例：`--end-time=400 --frames=400`（成核要到 t≈94，**t=200 不够**）。

| # | R0 | requested h | ε_pf | Lz | domain-height | 科学目的 |
|---|---|---|---|---|---|---|
| T1 | 5 | 3 | 11 | 80 | 40 | S1 复现基准（已知 F2，作为对照）|
| T2 | 5 | 3 | 11 | **120** | 40 | 检验"加长轴向域"是否直接给出更多珠（成核位点↑）|
| T3 | 10 | 3 | 11 | 120 | 40 | R0↑ + 薄膜：小 h 大 R 是否同时提高珠数与规则度 |
| T4 | 10 | 5 | 11 | 120 | 40 | 中等 h 的中性对照 |
| T5 | **20** | 3 | 11 | 120 | **60** | 大 R + 薄膜：最强"多模式"候选 |
| T6 | 20 | 5 | 20 | 120 | **60** | 大 R + 中壁：spacing 统计最干净的候选 |
| T7 | 10 | 3 | 20 | 120 | 40 | 强壁 + 薄膜：检验"残留膜是否抑制合并"|
| T8 | 20 | 3 | 30 | 120 | **60** | 强壁极端点：是否退化为少而大（F4/F5）|
| T9 | 10 | 5 | 30 | 120 | 40 | 强壁基准（S3 的更小 R 版本）|
| T10 | 10 | 2.5 | 20 | 120 | 40 | 最薄膜：珠数上限最高的点 |

**不要**一次把 R0×h×ε_pf 全正交展开。

---

## 4. 分析流程（每个 case）

```powershell
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m15\m15_beads.py <run_dir> `
    --R0 <R0> --Lz <Lz> --t-lo 40 --t-hi <end_time>
```
读 `m15_summary.csv` 的这些字段判读：

| 字段 | 判读 |
|---|---|
| `t_first_bead_ge2 / ge4` | 成核时刻；> 0.3·t_end 说明成核太慢（先加长 Lz 或 R0）|
| `merge_events` / `mean_spacing_slope` | 合并强度；斜率显著为负 ⇒ 仍在粗化，不能算"维持" |
| `largest_fraction_slope` | >0 且上升 ⇒ 走向单主团（F4/F5）|
| `median_CV_spacing` / `median_modulation` | F3 判据的两个关键量 |
| `bead4_hold_time` / `periodic_structure` | 是否**持续**多珠（≥20 t）|

---

## 5. 自动延长 / 淘汰 / 多 seed（Stage 2 版）

| 触发 | 动作 |
|---|---|
| `t_first_bead_ge4` 为 NaN 且 modulation ≥ 0.3 | 成核不足 ⇒ 换更大 R0 或更大 Lz 重跑（不要加 A0）|
| `t_first_bead_ge4` 存在但 `bead4_hold_time < 20` | 合并过快 ⇒ 该点进入 Stage B（t=800）或调 ε_pf |
| `periodic_structure == True`（A、B 两法都是） | → Stage C：3 seeds，比较 ⟨bead_count⟩/⟨CV_spacing⟩/⟨λ_dom⟩ |
| `largest_fraction > 0.8` 持续 ≥ 50 t | 已粗化 ⇒ 淘汰并记录（不计入 stability map）|
| guard/hard > 0 或爆炸 | INVALID ⇒ 淘汰重跑 |
| 全部 10 例都无 `bead_count ≥ 4` | **停下来问用户**（任务书 §12 第 6 条：参数搜索暗示模型无法产生多珠）|

---

## 6. 判据（沿用 Stage 1，未改）

`bead_count ≥ 4` ∧ `CV_spacing ≤ 0.35` ∧ `largest_fraction ≤ 0.45` ∧ `调制 ≥ 0.15` ∧ `dominant λ 漂移 ≤ 25%` ∧ **连续保持 ≥ 20 t**。

**注意本轮新增的可信度限制**：弛豫后 h(z) 重建的噪声地板 ≈ 0.1–0.3σ ⇒ **调制 < 0.3σ 的"多珠"不得当结构证据**；
同时要求**方法 A 与方法 B 的 `bead_count` 在 ±1 内一致**（两法差异 > 1 时判 UNRESOLVED）。

---

## 7. 明确不要做

* 不改 λ / h_ρ / ε_pp / Morse / WCA / 粒子质量 / Scheme E 公式；
* 不加新物理（disjoining、滑移、SALR…）；
* 不回到 RP 理论验证主线；
* 不用加大 A0 去"预置"多珠（A0 ≤ 0.01）。
