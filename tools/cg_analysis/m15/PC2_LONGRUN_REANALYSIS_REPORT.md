# PC2 长算例再分析：detector / tracking 稳健化（Stage 2）

**日期**：2026-09-23｜**数据**：PC2 数据包（ZIP SHA256 `E6CF912D…3E70C`，完整性见 `PC2_DATA_INTEGRITY_REPORT.md`）
**脚本**：`tools/cg_analysis/m15/{m15_beads.py, m15_beadtrack.py, m15_batch.py}`（本轮升级）
**约束**：未改 solver 物理、未改 Scheme E/λ/h_ρ/ε_pp/Morse/WCA/阻尼，未跑新算例。

---

## 1. 三例对照：帧级 detector（§五、§六、§七）

帧级分析窗口 t ≥ 40；`detector_sensitivity.csv` 为完整表。四套配置（**全局同一套，未按 case 调**）：

| config | 判定规则 |
|---|---|
| `legacy` | 旧规则（相邻 bin 差当 prominence）+ 无平滑 |
| `new(rel-only)` | 标准 prominence，但阈值 = `max(0.02, 0.25·amp)`（**只有相对阈值**）|
| `strict` | smooth 3 bins + `0.35·amp` |
| **`frozen`** | **smooth 2 bins + `max(0.30σ, 0.25·amp)`** ←本轮冻结 |

| config | case | mean\|Δbead\| | bead≥4 占比 | F2 占比 | **F3 占比** | F4+F5 | max bead | 调制中位 |
|---|---|---|---|---|---|---|---|---|
| legacy | S1_F2 | 0.220 | 0.228 | 0.530 | 0.026 | 0.440 | 5 | 0.79 |
| legacy | S3_F1 | 0.094 | 0.000 | 0.000 | 0.000 | 0.000 | 2 | 0.07 |
| legacy | S5_F5 | 0.019 | 0.000 | 0.000 | 0.000 | 0.789 | 2 | 0.40 |
| new(rel-only) | S1_F2 | 0.384 | 0.914 | 0.920 | 0.068 | 0.012 | 10 | 0.79 |
| new(rel-only) | **S3_F1** | 0.656 | **0.727** ⚠ | **0.950** ⚠ | 0.000 | 0.000 | 7 | 0.07 |
| new(rel-only) | S5_F5 | 0.188 | 0.031 | 0.081 | 0.000 | 0.826 | 5 | 0.40 |
| strict | S1_F2 | 0.177 | 0.351 | 0.638 | 0.109 | 0.241 | 5 | 0.67 |
| strict | S3_F1 | 0.150 | 0.000 | 0.106 | 0.000 | 0.000 | 3 | 0.06 |
| strict | S5_F5 | 0.044 | 0.000 | 0.000 | 0.000 | 0.634 | 2 | 0.37 |
| **frozen** | **S1_F2** | **0.208** | **0.590** | **0.798** | **0.080** | 0.121 | **7** | 0.72 |
| **frozen** | **S3_F1** | 0.338 | **0.056** | 0.366 | **0.000** | 0.000 | 4 | 0.07 |
| **frozen** | **S5_F5** | **0.062** | **0.000** | **0.025** | **0.000** | 0.739 | 3 | 0.38 |

### 读法（这是本轮最重要的定量结论）

1. **纯相对阈值会灾难性过检**：`new(rel-only)` 把**近连续膜 S3_F1** 判成 95% F2、73% 的帧 ≥4 珠
   —— 因为弱调制下 `0.25·amp` 退化到下限 0.02σ，**重建噪声被当成珠**。
2. **绝对 prominence 下限是关键修复**：加 `0.30σ` 地板后，S3_F1 的 ≥4 珠从 **0.727 → 0.056**、
   F3 从 0 → 0，而 S1_F2 的真实多珠**没有被抹掉**（≥4 珠 0.590、F2 0.798、max bead 7）。
3. **S5_F5 无假多珠**：frozen 下 ≥4 珠 = **0.000**、F2 = 0.025，F4+F5 = 0.739 ✅（§五 B 项通过）。
4. **旧 detector 反而漏检 S1 的真实多珠**（≥4 珠仅 0.228）⇒ `legacy` 不是"更保守所以更好"。
5. **残留问题**：frozen 下 S3_F1 仍有 **36.6% 的帧被判 F2**（调制 0.067 ≈ 0.3σ，正好在地板附近）
   ⇒ 在地板附近的 case 仍不可靠，已在 §5 明确标注。

---

## 2. 升级后的 tracking（§八、§九、§十）

`m15_beadtrack.py` v2 新增：**persistence**（新峰需连续 2 帧才登记 birth）、**gap bridging**
（允许最多 2 帧丢失后按预测位置重捕 ⇒ 记 `detection_gap` 而不是 death+birth）、
**速度预测匹配**（周期 z 正确，跨 z=0/Lz 不产生假生死）、**严格 merge 判定**
（要求两个稳定 track + 配对距离单调收缩 + survivor 尺寸相容）。

冻结配置下的结果（t ∈ [40, 1000]）：

| case | stable beads | flicker | max bead | **TRUE_MERGE** | UNRESOLVED | TRUE_DEATH | detection_gap 事件 | 寿命中位 | 迁移速度中位 |
|---|---|---|---|---|---|---|---|---|---|
| **S1_F2** | 57 | 26 | **7** | **6** | 26 | 20 | 100 | **18.0** | 0.738 |
| S3_F1 | 11 | 12 | 4 | 0 | 0 | 9 | 17 | 11.0 | 0.687 |
| S5_F5 | 4 | 1 | 3 | 0 | 0 | 3 | 12 | 19.0 | 0.796 |

**与旧 detector 对比（同一条 S1 轨迹）**：

| 量 | legacy | **frozen（新）** |
|---|---|---|
| stable beads | 46 | **57** |
| flicker | 16 | 26 |
| max bead | 5 | **7** |
| TRUE_MERGE | 3 | **6** |
| median lifetime | 17.5 | **18.0** |

⇒ **新 detector 确实减少了非物理漏检/抖动**（多找出真实珠、寿命更稳定），
同时**没有**把明显真实珠子删掉；代价是把"闪烁"计数得更诚实（26 vs 16）。

**"18 个 merge"问题定论**：在全长 960 t 的 S1 里，**严格判定的 TRUE_MERGE 只有 6 次**
（R_merge ≈ 0.006/t），另有 26 次 UNRESOLVED、**100 次 detection_gap**。
⇒ 之前报的"频繁合并"确实是**检测层面事件为主**，真实合并率低约 3 倍以上。

---

## 3. S1 新版时序统计（§十一 交付 6）

| 问题 | 数据支持的答案 |
|---|---|
| 1. S1 的 F2 是否真实存在？ | **是**。frozen 配置下 79.8% 的帧为 F2、59% 的帧 ≥4 珠、max bead = 7，且**两套独立重建（方法 A/B）一致**（工具同时输出）|
| 2. 稳定 bead 能活多久？ | **中位 18.0 t**（最大寿命见 `m15_track_beads.csv`）|
| 3. 真实 bead 数大致多少？ | 可跟踪稳定珠 **57 个/960 t**；**瞬时同时存在**通常 3–7 个 |
| 4. CV≈0.5 是否仍成立？ | **不再成立**：frozen 下 median CV_spacing 见 `m15_summary.csv`（S1 约 0.3–0.5 随配置变化）⇒ **CV 本身对检测器敏感**，不能单独作为判据 |
| 5. CV 大来自什么？ | **两者都有**：位置确实不规则 + detection jitter。Jitter 的量化上限：mean\|Δbead\| = 0.208 帧⁻¹（frozen）vs 0.384（rel-only）|
| 6. dominant wavelength 漂移？ | 仍存在（见 `m15_beads.csv` 的 `dominant_lambda` 列），且随 detector 配置变化 ⇒ **暂不可作为判据** |
| 7. 有没有可信 true merge？ | **有 6 次**（严格判据），其余 26 次 UNRESOLVED |
| 8. 若没有频繁 merge，bead 消失更可能是什么？ | **detection_gap（100 次）> TRUE_DEATH（20 次）> TRUE_MERGE（6 次）** ⇒ 主要是**检测丢失**（谷变浅/重建噪声），其次是真实的珠消失 |

---

## 4. h(z) 噪声地板再评估（§十三 交付 8）

| case | 调制中位 `A_dom/⟨h⟩` | 结论 |
|---|---|---|
| S3_F1（近连续膜，应为 F1）| **0.067** | 该值对应的 h 起伏 ≈ **0.3σ**，且**不应**产生珠 ⇒ 这是噪声/粒子尺度起伏的上限 |
| S5_F5（单主团）| 0.383 | 结构真实 |
| S1_F2（多珠）| 0.724 | 结构真实 |

**结论：旧的 0.1–0.3σ 估计仍然合理，且应取上端 0.3σ 作为判据地板。**
本轮把它固化成 **`PROM_ABS_SIGMA = 0.30`**（绝对 prominence 下限），
并**不是**为了让任何 case 通过 F3（S3_F1 反而因此被正确判回 F0/F1）。

---

## 5. 最终冻结的 global detector config（交付 7）

```
h(z) 重建: 方法A 滑窗极大值(半宽 1.0σ) / 方法B 外层粒子选择, nbins = 2 per sigma
平滑      : Gaussian σ = 2 bins
峰判定    : scipy.signal.find_peaks, distance = 2σ, prominence ≥ max(0.30σ, 0.25·amp)
跟踪      : persist = 2 帧, gap bridging = 2 帧, 速度预测匹配（周期 z）
merge     : 需 2 个 stable track + 配对距离单调收缩 + survivor 尺寸相容 ⇒ TRUE_MERGE
分类      : F0–F5 定义不变；F3 判据不变（bead≥4, CV≤0.35, largest≤0.45, 调制≥0.15,
            λ 漂移≤25%, 连续≥20 t.u.）
ROBUST-F3 : 只在 {frozen, strict(smooth3/prom0.35), loose(smooth1/prom0.20)} 三套配置
            下都满足 F3 才允许标记；只有一部分满足 ⇒ ANALYSIS-SENSITIVE
```

**判据可信区间（明确写出）**：调制 **≥ 0.3σ** 时 detector 可信；**0.1–0.3σ** 为灰区（UNRESOLVED）；
**< 0.1σ** 不得称结构。S3_F1 的 36.6% F2 帧正落在灰区，**不得作为多珠证据**。

---

## 6. PC2 后续运行的分析命令（交付 9）

```powershell
# 单个 case（frozen 配置 + 跟踪 + 统一 summary）
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m15\m15_beads.py <case_dir> `
    --R0 <R0> --Lz <Lz> --t-lo 40 --smooth 2 --prom-frac 0.25
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m15\m15_beadtrack.py <case_dir> `
    --R0 <R0> --Lz <Lz> --t-lo 40 --smooth 2 --prom-frac 0.25 --label <name>

# 整个 matrix root：自动 F0–F5 + F3 + ROBUST-F3
E:\.venv\Scripts\python.exe <repo>\tools\cg_analysis\m15\m15_batch.py <matrix_root> `
    --R0 <R0> --Lz <Lz> --t-lo 40 --matrix --out robust_f3_table.csv
```

---

## 7. 四个问题的最终回答（§十七）

**A. 当前 detector 是否足够可靠，可以判断"有没有真实多珠"？**
**在调制 ≥ 0.3σ 时：是。** S1_F2（0.59 的帧 ≥4 珠、F2 0.80、max 7）与 S5_F5（≥4 珠 **0**、F2 0.025）
形成干净对照，且新 detector 不漏掉真实多珠。
**在 0.1–0.3σ 灰区：否** —— S3_F1 仍有 36.6% 的帧被误判 F2。

**B. 是否足够可靠，可以判断 F2 vs F3？**
**否（ANALYSIS-SENSITIVE）。** 同一条 S1 轨迹的 F3 占比在 0.026（legacy）↔ 0.080（frozen）↔ 0.109（strict）
之间翻转，最长 F3 持续 ~11 t < 20 t 门槛 ⇒ **任何 F3 结论都必须标 ANALYSIS-SENSITIVE，不能作为正式周期结构证据**。

**C. bead identity / lifetime 是否已经可信？**
**寿命统计：可信**（中位 17.5–18.0 t 在两种 detector 下一致）。
**事件级身份：仅在有 gap bridging 时可信**（S1 有 100 次 detection_gap）⇒ 可以用于寿命/迁移统计，**不可**用于精确的 birth/death 计数。

**D. true merge 是否已经可以可靠识别？**
**可以识别，但必须严格。** 严格判据下 S1（960 t）只有 **6 次 TRUE_MERGE**；26 次 UNRESOLVED、
100 次 DETECTION_GAP。旧口径的 "18 个 merge" **被证伪**（多数是检测事件）。

**E. PC2 Stage 1 的核心结论"存在真实 F2，但没有可信 F3"是否仍成立？**
**两半都成立，但需要加限定**：
* "**存在真实 F2**" —— **加强成立**（frozen 配置：59% 的帧 ≥4 珠、F2 占 80%、max 7 珠、与 S5 的 0 假阳性形成对照）；
* "**没有可信 F3**" —— **成立**（F3 占比 2.6–10.9% 且随合理 detector 配置翻转，最长持续 ~11 t < 20 t）⇒ 归入 ANALYSIS-SENSITIVE。
