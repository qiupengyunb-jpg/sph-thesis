# PC2 → PC1 Stage 2 数据包完整性报告

**核对时间**：2026-09-23｜**核对人**：PC1

## 0. 数据包标识

| 项 | 值 |
|---|---|
| 原始 ZIP 路径 | `C:\Users\qiupe\xwechat_files\wxid_0xyl4zbg6lgw22_181b\temp\RWTemp\2026-09\9e20f478899dc29eb19741386f9343c8\PC2_TO_PC1_STAGE2_DATA.zip` |
| 大小 | 17 211 319 B（16.41 MB），1440 条目 |
| **ZIP SHA256** | **`E6CF912DA2FB770A37D8741F9731FB80A6EBACD5C61CF1EB6424FF6BA913E70C`** |
| 接收时间 | 2026-09-23 12:51:03 |
| 解压到 | `E:\ai项目推进信息\PC2_DATA\PC2_TO_PC1_STAGE2_DATA\`（**未放进 git 仓库**）|
| 原始 ZIP | **未修改**；PC2 原始 CSV/VTP **未修改**（分析在副本 `…\PC2_DATA\analysis\cases\` 中进行）|

## 1. 与 TRANSFER_MANIFEST.txt 的核对

| case | 期望 | 实测 | 一致 |
|---|---|---|---|
| S1_F2 | 1001 VTP, t=0..1000 | **1001 VTP**, `ite_0000000000` … `ite_0000133334`, 1013 文件 | ✅ |
| S5_F5 | 201 VTP, t=0..200 | **201 VTP**, `ite_0000000000` … `ite_0000026667`, 213 文件 | ✅ |
| S3_F1 | 201 VTP, t=0..200 | **201 VTP**, 213 文件 | ✅ |

* **selfcheck 行数 = VTP 数量**：S1_F2 1001/1001 ✅、S5_F5 201/201 ✅、S3_F1 201/201 ✅
  ⇒ 时间轴可与 VTP 逐帧对应（本工具正是用 `selfcheck.csv` 的 time 列当帧时间，因为 VTP 的 `TimeValue` 恒为 0）。
* **0 字节文件**：每例各 1 个 = `stderr.log`（空日志，**符合预期**，说明求解器无报错输出）。
* 必需文件齐全：`parameters.csv`、`selfcheck.csv`、`CGParticles_*.vtp`、`axisym_film_selfcheck.csv`、
  `sg_budget.csv`、`m15_beads.csv`、`m15_summary.csv`、`run_meta.txt` **三例全部存在** ✅
  （另附 `axisym_geometry_selfcheck.csv` / `stdout.log`）。
* 附加核对：`run_meta.txt` 记录 solver commit `763020872`、seed `20260916`、exe SHA256
  `3C671952…4865B0`，exit_code 0、stderr 0 字节 ✅

## 2. 结论

**数据包完整、可用**：核心的 `CGParticles_*.vtp` 与 `selfcheck.csv` 三例齐全且数量一致，
不存在 0 字节的核心文件。**本轮核心任务不被阻塞。**
