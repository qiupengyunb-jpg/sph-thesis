# Stage 21R —— kernel sign gate 审计

| 行 | 代码 | 函数 | 用途 | 处置 |
|---|---|---|---|---|
| L1760 | `if (w <= 0.0)  // outside the kernel support` | `CGSquareGradientDensity` | 密度估计量（ρ=ΣW_h）配对过滤 | 保留（密度恒 legacy）|
| L1816 | `if (w1 == 0.0)` | `CGSquareGradientForce`（Scheme A）| 链式项过滤 | 保留（Scheme A 不用两尺度核）|
| L1877 | `if (w1 == 0.0)` | Scheme A REF-CACHED | 同上 | 保留 |
| L1928 | `if (w1 == 0.0)` | Scheme A DumpForceSplit | 同上 | 保留 |
| L2048 | `if (w <= 0.0)` | `CGSquareGradientConjugate` | Scheme E 共轭场过滤 | **改为 `r >= SchemeKernelSupport()`** |
| L2106 | `if (w <= 0.0 && dw == 0.0)` | `ForceFromCachedGeometry` | Scheme E 力过滤 | **同上** |
| L2173 | `if (w <= 0.0 && dw == 0.0)` | `DumpForceSplit`（Scheme E 分支）| 诊断过滤 | **同上** |

**原则**：legacy 保留原逻辑（等价：`r >= h_ρ ⇔ w == 0`）；two_scale 必须按支集 `r >= max(h_ρ, ξσ)` 判定，**不得**用 `w > 0`（两尺度核有负瓣）。同一条规则必须同时用于 energy / force / derivative / FD 四条路径。
