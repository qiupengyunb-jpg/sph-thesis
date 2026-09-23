# Stage 2 参数搜索前：PC1 研发检查报告

**日期**：2026-09-23｜**基线**：`feature/cg-axisym-singlemode-scheme-e` @ `763020872`
**性质**：只读检查 + 只改分析工具。**未改 solver 物理、未改 Scheme E、未改 λ/h_ρ/Morse/WCA，未跑任何长算例。**

---

## 一、代码检查（§二）

| 检查项 | 结论 | 证据 |
|---|---|---|
| Scheme E **默认关闭** | ✅ | `cg_self_assembly_pri.cpp`：`std::string axisym_interface_scheme = "moving_j";`；`PairDifferenceScheme()` 要求 `axisymmetric && scheme=="difference_energy"` |
| `difference_energy` 开关正常 | ✅ | 白名单校验（非法值 exit 2）+ 必须配 `--axisymmetric=filmonly`；M1.3R S0–S8 已过 |
| M1.4 broadband 初态正常 | ✅ | `--film-broadband=1` + `--film-perturb-amp`（总预算，`|f−1| ≤ A`）；smoke S4 正常，guard/hard=0 |
| `m15_beads.py` 能吃 PC2 输出格式 | ✅ | 工具用 `**/CGParticles_ite_*.vtp` 递归 glob + `selfcheck.csv` 取时间轴（VTP 的 `TimeValue` 恒为 0，已绕开）|
| 版本一致性 | ✅ | 本地 HEAD = 远端 `thesis/...` = `763020872`；工作区 clean |

**PC2 拉取即可用的工具链**：`tools/cg_analysis/m15/{m15_beads.py, m15_ensemble.py?}`（后者在 m14）、`tools/cg_analysis/m14/{m14_axisym_modes.py, m14_ensemble.py}`。

---

## 二、问题 1：轴对称 `domain_height` 判断（**只分析，未改代码**）

### 现象
PC2 在 **R0 = 20 + 默认 `--domain-height=40`** 时收到
`error: fibre does not fit inside the domain.`

### 代码事实

```cpp
Real domain_height = 40.0;                     // 默认
Real ExclusionRadius() { return FibreRadius() + run_options.exclusion_margin; }
Real FibreRadius() { return Axisymmetric() ? AxisRadius() : (cylinder ? fibre_radius : FibreHalfWidth()); }

// ParseCommandLine:
if (2.0 * ExclusionRadius() >= run_options.domain_height)
    throw std::runtime_error("fibre does not fit inside the domain.");
```

### 分析：这条检查与轴对称几何**不匹配**

* 该判据的语义是**二维条带/胶囊**：纤维在 y 方向**居中**、半宽 `FibreRadius()`，因此要求 `2(R+margin) < H`。
* 轴对称分支里纤维是**半径 R0 的圆柱、贴在内边界 r=0**，占据 `r < R0`；它**不居中**，正确的几何要求是**一次量级**：

```
R0 + margin + h_in + h_film + dr + σ < domain_height
```

* 于是旧判据**过严**：它等价于 `R0 < H/2 − 0.5`。

| R0 | H=40 旧判据 | 物理正确判据（h=7, φ=0.63: h_in=0.5, dr≈1.04, σ=1） | 
|---|---|---|
| 5 | 2×5.5 = 11 < 40 ✅ 通过 | 5+0.5+0.5+7+1.04+1 = 15.0 < 40 ✅ |
| 10 | 2×10.5 = 21 < 40 ✅ 通过 | 20.0 < 40 ✅ |
| **20** | 2×20.5 = **41 ≥ 40 ✗ 抛错** | **29.5 < 40 ✅ 应当通过** |

⇒ **PC2 遇到的报错是旧判据的假阳性**，不是几何真的装不下。

* 附带事实：`--init=film` 路径里**已经有**一条正确的轴对称判据
  （`reach_r + sigma >= domain_height` → `"--init=film: fibre plus film does not fit inside the domain"`），
  所以薄膜初态本身是受保护的，只是旧的通用判据先一步抛出。

### 建议（**本轮未实施**，需主线批准）

最小改动（**只影响 axisymmetric 分支的抛错条件，二维路径不动**）：

```cpp
const Real needed = Axisymmetric()
    ? (ExclusionRadius() + run_options.film_thickness + 3.0 * Sigma())
    : (2.0 * ExclusionRadius());
if (needed >= run_options.domain_height) throw ...;
```

二维路径 `Axisymmetric()==false` ⇒ `needed == 2*ExclusionRadius()`，与现状逐位一致（只是判据值相同、抛错时机相同），
再做一次 strip/A-1 的 VTP SHA256 逐位回归即可证明无副作用。

### PC2 现在就能用的**无代码修改**做法

**所有轴对称算例显式给 `--domain-height`**，取
`H ≥ R0 + 0.5 + h_film + 1.1 + 1.0 + 2.0`（即留 ≥2σ 余量），
推荐：R0=5→40、R0=10→40、**R0=20→60**、R0=20+h=7→60。
这条已写进 handoff 与指导文档。

---

## 三、问题 2：`cluster-packing` 默认值（文档修正，未改代码）

* 代码默认 **`Real cluster_packing = 0.63;`**；PC2 用的 0.63 = **与默认一致** ✅。
* **film 初态复用同一个字段**：`prepareAxisymmetricFilm()` 里 `const Real phi = run_options.cluster_packing;` ——
  即"膜的面内堆积率"由 `--cluster-packing` 控制（名字沿用了 cluster 初态，属命名历史包袱，功能正常）。
* **上一版 handoff 写了 packing = 0.70，这是错的**（那是我 smoke test 里显式传的值，不是默认）。
  ⇒ **已在本轮修正 handoff**：默认 **0.63**，要求 PC2 **显式写出** `--cluster-packing=0.63` 以免歧义。
* 影响：φ 决定晶格间距 `d = σ√(π/4)/(0.5√3 φ)^{1/2}` 与径向层距 `dr = 0.5√3 d`。
  φ=0.63 → d = 1.1998、dr = 1.0392；φ=0.70 → d = 1.1382、dr = 0.9857。
  两者**realised thickness 的层数可能不同**（量化），比较时必须以 `realised_thickness` 为准（handoff 已写明）。

---

## 四、问题 3：分析工具是否够用（**结论：不够，已补**）

原 `m15_beads.py` 已输出：`bead_count / mean_spacing / CV_spacing / largest_fraction / dominant_lambda / modulation / mean_h / min_h / max_h / spectrum`（逐帧）——**F3 判据的 6 条全部可判**。

但 §四 要求的**机制归因**（成核时间、合并事件、间距趋势、最大珠分数增速）**缺失**。
⇒ 本轮**只改分析工具**，新增 8 个 run-level 指标（commit 见 §六）：

| 新指标 | 含义 |
|---|---|
| `t_first_bead_ge2` / `t_first_bead_ge4` | 首次出现 ≥2 / ≥4 个珠的时间（成核时刻）|
| `merge_events` | 逐帧 bead_count **下降**的次数（合并事件计数）|
| `max_bead_count` | 全窗口最大珠数 |
| `bead4_hold_time` | `bead_count ≥ 4` 的最长连续时长 |
| `largest_fraction_slope` | 最大珠占比的线性趋势（粗化方向）|
| `CV_spacing_slope` / `mean_spacing_slope` | 间距规则度 / 平均间距的时间趋势 |

（未改判据阈值，未改分类定义。）

---

## 五、Stage 1 数据复算：S1 为什么"多珠但不规则"（§四）

PC2 的 Stage 1 原始数据**不在本机**（全盘无 `m15_*.csv`，也无 stage1 结果目录），
因此 PC1 用 **PC2 的同一参数口径**（`--cluster-packing=0.63`，T=1，difference_energy，dt=0.0075，Lz=80）
在 **S1 及其变体**上各跑 **1 个 t=200 的短算例**（各 26 667 步，≈11.7 s，非矩阵）：

| case | R0 | h | ε_pf | 结果类别（t∈[20,200]）| 首次≥2珠 | 首次≥4珠 | 合并事件 | max珠数 | ≥4珠持续 | mean_spacing 斜率 | largest_fraction 斜率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1 = S1** | 5 | 3 | 11 | F1 .08 / F2 .09 / F4 .34 / F5 .49 | **t=94** | t=190（仅 1 帧）| **18** | 4 | 1.0 | **−0.058 σ/t** | −0.0034 |
| P2 = S1 强壁 | 5 | 3 | 20 | F1 .29 / F5 .71 | t=39 | 未出现 | 3 | 2 | 0 | ≈0 | +0.0003 |

### 判读（对应 §四 的 A–D）

1. **成核晚且种子少**：P1 到 t≈94 才出现第二个珠，t≈190 才短暂凑到 4 个 ⇒ **A（初始成核不足）成立**。
2. **合并频繁且间距持续收缩**：`merge_events = 18`、`mean_spacing` 斜率 **−0.058 σ/t**（负） ⇒
   **B（珠迁移/合并导致合并）成立**，且是当前**主导**限制。
3. **不是"一个珠吞掉全部"**：`largest_fraction` 斜率 ≈ 0（甚至略负），modal class 走到 F4/F5 是靠**同量级珠互相合并** ⇒
   **C（液量过大导致粗化）只部分成立**：液量控制的是"总珠数上限"，不是"单个主团"。
4. **不是曲率驱动力不足**：调制幅度中位数 **0.57**（远高于 0.15 判据）⇒ **D 不成立**（当前模型确实在驱动界面失稳）。

**结论**：S1 的"多珠但不规则"= **成核不足（A）+ 珠迁移合并（B）**，液量给定了珠数上限（C 次要），曲率驱动不缺（D 否）。
⇒ 参数搜索的方向应是**增加可用成核位点并减慢合并**，而不是再加界面驱动力。

---

## 六、本轮 commit

| 内容 | SHA |
|---|---|
| `feat(stage2-prep): bead tracking metrics (nucleation time, merge events, spacing / largest-fraction trends)` | 见下（本报告同批） |

（既有：`763020872` handoff 入库、`a43ed1466` Stage1 工具、`2d5d375d8` M1.4。）

**exe SHA256（未变，未重编）**：`A9803AA0A479781D6116BDD2FADFF89589743C305960C6853C4946625028A546`
