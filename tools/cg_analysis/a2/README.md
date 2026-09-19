# A-2 接触角分析工具（cg_analysis/a2）

**本目录只是分析工具，不是模型代码。** A-2 求解器 commit 是 `da2a0ec51`（在 A-1 的 `50f6cacdb` 之上只新增 `--cluster-offset-y`），本目录的加入不改变任何求解器物理。

用途：对 A-2 平壁润湿算例（A-1 square-gradient 模型 + `--fibre-shape=strip` 平壁）做统一口径的接触角测量。

## 脚本清单

| 脚本 | 作用 | 输出 |
|---|---|---|
| `cg_contact_angle.py` | 晚时平均密度场 → `0.5ρ_bulk` 等密度轮廓 → 排除 h<2σ 近壁带 → Kasa 圆拟合 → 切线接触角 | `<root>/contact_angle_A1.csv`、`<root>/analysis_log_contact_angle.txt` |

## 口径定义（冻结，勿改）

密度场、`ρ_bulk`、界面宽度**直接从 a1 导入**（`cg_iface.py` 的 `density_field` / `bulk_density` / `contour_metrics` / `radial_profile` / `interface_width`），因此 A-1 与 A-2 的数字不可能各说各话。

```
密度场      : 0.5σ 网格计数 / 格面积，再作 1σ 高斯平滑（x 周期、y 镜像）——与 A-1 完全相同
ρ_bulk      : 密度场 99.9 分位为参照，> 0.7·参照 的格子的中位数——与 A-1 完全相同
界面        : 0.5·ρ_bulk 等密度轮廓（marching squares，取最长闭合段）
近壁排除    : 保留 h ≥ 2σ 的轮廓点；h = 粒子中心到壁面距离 = |y - Ly/2| - w
圆拟合      : Kasa 最小二乘，先对轮廓点做 x 周期展开（避免跨周期）
接触角      : θ = 90° + asin(yc / R)，yc = 圆心相对壁面的高度
              （yc = 0 → 90°；yc = R → 180°；yc = -R → 0°）
界面宽度    : 沿用 A-1 的径向剖面 tanh 拟合宽度（对坐滴是"等效"宽度，已在列名标出）
equilibrium_time : 以液滴回转半径 Rg(t) 为准，取"此后 Rg 不再偏离末段均值 ±2%"的第一个时刻
```

几何注意：`strip` 壁有上表面（`y = Ly/2 + w`，法线 +y）与下表面（`y = Ly/2 - w`，法线 -y），h 取较近的一面。用 `--cluster-offset-y` 的算例液滴整体只在一侧，另一侧为 0 粒子。

## 输出列

必需列（A-2 简报规定的格式）：

```
case, epsilon_pf, lambda, h_rho, theta, rho_bulk, interface_width, C, AR, equilibrium_time
```

其后追加的诊断列（不影响必需列）：

```
side, n_frames, n_avg, N_side, N_other, R_fit, yc, fit_rms, footprint, height,
cap_ratio, h_min_face, n_fit, n_pts, w_1090, rho_bulk_core, rho_dilute,
Rg_ref, Rg_last, theta_scatter_std, theta_scatter_n, theta_other
```

## 调用

```bash
cd <repo>/tools/cg_analysis/a2
python cg_contact_angle.py <results-root> [--avg N] [--theta-scatter K]
```

`<results-root>` 下每个算例一个子目录，每个子目录需要 `parameters.csv`、`selfcheck.csv` 与 `output_<case>/CGParticles_ite_*.vtp`（与 A-1 的目录约定相同）。`--avg` 是晚时平均的帧数（默认 12，与 A-1 的晚时窗口一致）。

## 自检判据（A-2.0 验收用）

* `N_other == 0`：液滴只在一侧，几何正确；
* `cap_ratio = footprint / (2·sqrt(R² - yc²))` 应 ≈ 1：轮廓确实是球冠（这是"密度轮廓形成球冠"的量化判据）；若为 `nan`，说明拟合圆根本没碰到壁面（液滴还悬着）；
* `fit_rms << R_fit`：圆拟合有效；
* `theta_scatter_std` 小：晚时形状稳定；
* `equilibrium_time` 应落在 T_end 之前，否则说明还没平衡，`theta` 不可用。