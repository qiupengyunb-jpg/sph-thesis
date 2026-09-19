# A-2 接触角分析工具（cg_analysis/contact_angle）

**只是分析工具，不改求解器。** 求解器仍为 `50f6cacdb`（A-1 square-gradient），A-2 不引入任何新物理。

本目录只有一个脚本 `cg_contact_angle.py`，它复用 `../a1/cg_iface.py` 的密度场与等密度轮廓定义
（同一套 `0.5σ` 网格 + `1σ` 高斯平滑 + `0.5·ρ_bulk` 轮廓），**不要另外实现一套**。

## 用途

对**平壁（strip 纤维）上的液滴**逐帧取等密度轮廓，量出

```
H      : 轮廓最高点相对接触面（h = wall_re = 0.5σ）的高度
l      : 轮廓在 h = 1.0σ 处的 x 跨度（等密度接触宽度）
θ_iso  : 2·atan(2H / l)                        ← 主口径（圆帽几何）
θ_fit  : 自由界面段（h > --exclude-h，默认 2σ）最小二乘圆拟合反解的角度  ← 独立校核
```

近壁粒子层因此**不参与**几何拟合；ε_pf=0（液滴不贴壁）或轮廓不穿 h≈1σ 时该算例会被跳过并打印 `(no usable frame)`，对应物理含义是 θ ≥ 180°（完全非润湿）。

## 调用

```bash
cd <repo>/tools/cg_analysis/contact_angle
python cg_contact_angle.py <results-root> \
       [--exclude-h 2.0] [--width-h 1.0] [--window 0.85] \
       [--out contact_angle.csv]
```

`--exclude-h`、`--width-h`、`--window` 都支持 `--k=v` 与 `--k v` 两种写法。

| 选项 | 默认 | 含义 |
|---|---|---|
| `--exclude-h` | 2.0 | 圆拟合只取 h 大于该值的轮廓点（σ） |
| `--width-h` | 1.0 | 量接触宽度 l 所用的 h（±0.6σ 带，固定） |
| `--window` | 0.85 | 取 t > window·t_end 的帧做平均 |
| `--out` | `contact_angle.csv` | 输出文件名（写在 `<results-root>/` 下） |

## 输入目录格式

与 `cg_analysis/a1` 完全一致：

```
<results-root>/<case>/parameters.csv          # 必须有 domain_length/domain_height/fibre_half_width/wall_re/wall_depth_D0
<results-root>/<case>/selfcheck.csv           # 第 1 列 time、末列 steps
<results-root>/<case>/output_<case>/CGParticles_ite_%010d.vtp
```

墙几何**从每个算例自己的 `parameters.csv` 读取**（`fibre_half_width` 决定壁面位置，`wall_re` 决定接触面），不再写死。

## 输出

`<results-root>/<out>`，CSV 列：

```
case, eps_pf, H, l, theta_iso, theta_iso_std, R_fit, H_c, theta_fit,
n_frames, window_lo, t_end, exclude_h, width_h
```

* `theta_iso` = 主口径角度（度）；`theta_iso_std` = **同一轨迹内**逐帧散布
* `theta_fit` = 圆拟合校核；`R_fit`、`H_c` = 拟合圆半径与圆心高度
* 所有参数都写进 CSV，便于复核当时用的窗口与阈值

给主线回报时请把文件命名为 **`contact_angle_A1.csv`**（`--out contact_angle_A1.csv`），便于与后续批次区分。

## 与旧口径的一致性

本脚本是从已完成 A-0/界面轮使用的 `cg_contact_contour.py` **原样移植**（只把路径/CLI 参数化），
在 `cg_contact_20260919` 四例上复算结果与旧输出**完全一致**：

```
ct_pf1 θ_iso=147.482   ct_pf2 θ_iso=108.351   ct_pf3 θ_iso=97.194   ct_pf5 θ_iso=47.802
```

测量定义见 `A2_contact_angle_protocol.md`（**冻结，不得改动**）。
