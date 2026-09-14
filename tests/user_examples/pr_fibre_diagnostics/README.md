# PR 湿纤维项目 —— 离线诊断脚本

这些脚本**只读 VTP/CSV 输出**，不接触求解器，不写入任何算例目录。
它们支撑 2026-09-12 ~ 09-14 五轮诊断与标定，结论见 `E:\ai项目推进信息\` 下的两份状态文件
（`湿纤维PR不稳定性项目摘要.txt` 与 `PROJECT_STATE.md`）。

## 运行环境

```
E:\.venv\Scripts\python.exe        # numpy / scipy / matplotlib
```

脚本里的路径是绝对路径（指向 `E:\sphmethod\SPH_results_center\...`），换机器需要改。

## 关键约定（阅读脚本前必看）

1. **时间一律直接读 VTP 的 `TimeValue`，再除以该算例的时间换算系数**，
   禁止用 `sorted(glob('LiquidFilmHalf_*.vtp'))` 的下标推算 —— T=0 的初始帧名是
   `..._ite_0000000000.vtp`，字符串排序会把它排到最后，导致整个时间轴错位。
   `check_frame_index.py` 就是当年那个错误的复现与验证。
2. **三个"核完整性"类量不能混用**：
   - `PositionDivergence`（`−Σ∇W·V·r`）：含梯度含体积权重，**并且它就是 `Indicator` 的判据本身**（阈值 1.5）；
   - summation density `ρ_s = S/σ0`：绝对值，带着自由表面截断基线（健康表面粒子 T=0 时只有 0.98）；
   - 自参照亏损 `δ = 1 − S(T)/S(0)`：只用比值，基线按构造消掉 —— **门控该用的是这个**。
3. 核与归一化：`KernelWendlandC2`、`h = 1.3·dx`、截断 `2h`、自含 `W(0)`，
   `W2D(q) = 7/(4πh²)(1−q/2)⁴(1+2q)`。

## 脚本清单

| 脚本 | 用途 |
|---|---|
| `v2io.py` / `v1io.py` | 第二版 / 第一版算例的 VTP 读取与帧-时间映射 |
| `diag_core.py` | 核函数、σ0、邻域求和的公共实现 |
| `check_impl.py` | 实现自检：T=0 完整膜上离线求和应给出 ρ≈1 |
| `run_diag.py` | 逐帧指标（间隙/支撑/两种密度/连通性） |
| `run_metrics.py` | **通用逐帧指标表**（gap / h_max / 连通性 / max_speed），两个算例对比都用它 |
| `support_stats.py` | 支撑比分布与最差粒子队列 |
| `verify_cohort.py` | 队列统计的独立复核（曾发现一次区域/队列混用的 bug） |
| `lesion.py`、`who.py`、`crossings.py` | 病灶定位、受损粒子身份、阈值穿越时刻 |
| `extras.py` | 压力、状态方程残差、两种求和写法的比较 |
| `identical_check.py` | 逐粒子比对两次运行是否完全相同（surface-only 空实验的判定依据） |
| `why_zero_shift.py` | 正则化算子为何空转（`TopologyNormal` 恒为 0） |
| `residual_direction.py` | 核梯度残差的量与方向（粒子重整失控的量化依据） |
| `calib_delta.py` / `calib_extra.py` | **2C 标定**：δ 门控可分性、翻转分布、出图 |
| `figs.py`、`figs_compare.py`、`morph.py`、`morph_srf.py`、`explore.py`、`check_frame_index.py` | 出图与一次性检查 |
| `watch_T_progress.ps1` | 只显示"当前 T / 目标 T"的进度窗口（**必须用 `pwsh` 运行**） |

## 输出位置

脚本默认把结果写到 `E:\哈哈\_pr_diag\out\`；已归档的正式交付物在
`E:\sphmethod\SPH_results_center\recon_tapered_20260914\` 下的
`diag_geometry_support_20260914\`、`diag_surface_reg_20260914\`、
`design_kernel_support_20260914\`、`calib_delta_20260914\`。
