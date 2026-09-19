# A-1 Phase 1 分析工具（cg_analysis/a1）

**这批文件只是分析工具，不是模型代码。求解器 commit 仍然是 `50f6cacdb`，本目录的加入不改变任何求解器物理。**

用途：对 A-1 square-gradient 生产算例（自由凝聚体，无纤维）做统一口径的界面/动力学分析。
所有指标定义在下面的「口径定义」一节里被**冻结**——只做工程化（路径/CLI）适配，不得改动定义。

## 脚本清单

| 脚本 | 作用 | 输出 |
|---|---|---|
| `cg_iface.py` | 逐帧建立粗粒化密度场 ρ(x,y)（0.5σ 网格 + 1σ 高斯）、取 `0.5ρ_bulk` 等密度轮廓，计算 A、P、C=4πA/P²、AR、轮廓粗糙度、ρ(r) 剖面与界面宽度、真实数密度 | `<root>/iface_series.csv` |
| `cg_iface_spectrum.py` | 由等密度轮廓的角向谱反解线张力：log-log 拟合斜率（毛细波应为 −2）与 `γ_est = k_BT/(2π·S(q)q²)` | `<root>/iface_spectrum.csv` |
| `cg_iface_dynamics.py` | 内部动力学：按 OriginalID 追踪，最小镜像展开的 MSD(Δt)、有效 D=MSD/(4Δt)、扩散指数 α、邻居替换率（连通判据 1.5σ） | `<root>/iface_dynamics.csv` |
| `cg_iface_summary_figs.py` | 把上面三份 CSV 画成汇总图（形状松弛 4 联曲线、归一化径向剖面、毛细波谱） | `<root>/fig_iface_relax.png`、`fig_iface_profile.png`、`fig_iface_spectrum.png` |
| `cg_sg_g1_bulk.py` | **Gate-1 均匀体相专用**：只统计离反射壁 3σ 以内的内部区域，给出真实数密度 ρ_true（逐帧瞬时）、核密度、T_kin、MSD、D、邻居交换率，并输出相对 λ=0 基线的 ρ/ρ₀ 与 D/D₀ | `<root>/g1_bulk.csv` |

## 口径定义（冻结，勿改）

```
密度场      : 0.5σ 网格计数 / 格面积，再作 1σ 高斯平滑（x 周期、y 镜像）
ρ_bulk      : 密度场 99.9 分位为参照，> 0.7·参照 的格子的中位数
界面        : 0.5·ρ_bulk 等密度轮廓（marching squares，取最长闭合段）
A, P        : 轮廓面积 / 周长；C = 4πA/P²；AR = 轮廓二阶矩主轴比
界面宽度 w  : ρ(r)/ρ_bulk 的 tanh 拟合宽度（10–90% 宽度写在同一 CSV 的 w_1090 列）
γ_est       : ⟨|r_q|²⟩ = k_BT/(2πγq²) → 对 q∈[4,20] 作 log-log 拟合
MSD         : 最小镜像展开的 ⟨|Δr|²⟩；D = MSD(Δt=1500)/(4·1500)
邻居替换率  : 1 − |N(0) ∩ N(Δt)| / |N(0)|，连通判据 r<1.5σ
ρ_true      : 内部区域内粒子数 / 内部面积（逐帧瞬时平均，不用"全程留在内部"的子集）
```

## 输入目录格式

`<results-root>/` 下每个算例一个子目录，目录名 = 算例名：

```
<results-root>/<case>/parameters.csv                              # 求解器写出的全部参数 + git commit
<results-root>/<case>/selfcheck.csv                               # 第 1 列 time、第 2 列 particles、末列 steps
<results-root>/<case>/output_<case>/CGParticles_ite_%010d.vtp     # 帧序列（文件名数字 = 积分步数）
```

注意：

* 文件名里的数字是**积分步数**，`time ↔ steps` 的对应关系在 `selfcheck.csv`。
* VTP 里粒子坐标在数组 `Position`（另有 `Velocity`、`CG_Force`、`OriginalID`）。
* 只有纤维类算例才另有 `Fibre_ite_*.vtp` 与 `wall_*` 参数；A-1 Phase 1 是无纤维算例。
* `--init=lattice` 的算例，`parameters.csv` 的 `particle_number` 写的是**按 area_fraction 推算的目标数**，真实粒子数请读 `selfcheck.csv` 的 `particles` 列。

## PC2 调用命令

依赖：Python 3 + numpy + scipy + matplotlib。

```bash
cd <repo>/tools/cg_analysis/a1

# 1) 逐帧指标（必需，后面两个依赖它的输出）
python cg_iface.py           <results-root>
# 2) 线张力（毛细波谱）
python cg_iface_spectrum.py  <results-root>
# 3) 内部动力学（MSD / D / 邻居交换率）
python cg_iface_dynamics.py  <results-root>
# 4) 汇总图（可指定要画的算例，逗号分隔）
python cg_iface_summary_figs.py <results-root> \
       P1_h2p0_lam2,P1_h2p0_lam5,P1_h2p0_lam8,P1_h2p0_lam12 \
       P1_h2p0_lam2,P1_h2p8_lam12
# 5) 仅 Gate-1 均匀体相用（指定算例名）
python cg_sg_g1_bulk.py <results-root> G1b_lat_lam0 G1b_lat_lam25
```

A-1 Phase 1 的 `<results-root>` 就是 9 个生产算例所在的根目录（每个算例一个子目录），例如
`P1_lam0, P1_h2p0_lam2, P1_h2p0_lam5, P1_h2p0_lam8, P1_h2p0_lam12, P1_h2p8_lam2, P1_h2p8_lam5, P1_h2p8_lam8, P1_h2p8_lam12`。

## 输出文件列表（都写在 `<results-root>/`）

```
iface_series.csv      # 每帧：time, A, P, C, AR, R_eq, cx, cy, rho_bulk, rho_dilute, w_1090, w_fit, R_fit, rough
iface_spectrum.csv    # 每算例每模数：case, q, S(q)
iface_dynamics.csv    # case, kind(msd|neighbour_replaced), lag, value
fig_iface_relax.png / fig_iface_profile.png / fig_iface_spectrum.png
g1_bulk.csv           # 仅 cg_sg_g1_bulk.py：rho_true, rho_kernel, T_kin, MSD, D_eff, 邻居交换率, N_int
```

## Phase 1 汇报哪些量

γ_est（谱反解）、D 与邻居交换率、ρ_bulk（ρ_true 与核密度都给）、C、AR、界面宽度 w、毛细波斜率（应≈−2）、T_kin、稳定性（min_sep / max_speed / `sg_net_force`）。
筛选门槛：`D/D₀ ≥ 0.8`、`|Δρ_bulk|/ρ_bulk ≤ 5%`、数值稳定、毛细波谱斜率仍接近 −2；在此约束下选 γ 提升最大且 λ→γ 单调的 1–2 个 (λ, h)。
