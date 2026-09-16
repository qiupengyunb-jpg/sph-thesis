<#
Progress window: shows only "current T / target T".

Unlike a log tail, this reads the NEWEST output frame's iteration number, so it
never lags behind stdout buffering.  Frame names carry the solver time as an
integer tag:
    second-version (breakup) program : tag = solver_time * 1e6   <-- default
    first-version program            : tag = solver_time * 1e7
  e.g. LiquidFilmHalf_0001341640.vtp -> 1.341640786 -> T = 1.3416408/0.67082 = 2.0
    T = solver time / time scale     (time scale = solver_end_time / end_T)

Run with pwsh (PowerShell 7), ASCII only on purpose.
#>
param(
    [string]$OutputDir   = 'E:\sphmethod\SPH_results_center\recon_tapered_20260914\r32_broadband_T80\output_recon_r32_T80',
    [double]$TargetT     = 80,
    [double]$TimeScale   = 0.67082,
    [double]$TagScale    = 1e6,
    [int]   $ProcId      = 56988,
    [int]   $IntervalSec = 15
)

$Host.UI.RawUI.WindowTitle = "PR progress - current T / target T"

function Get-CurrentT {
    $files = @(Get-ChildItem -LiteralPath $OutputDir -Filter 'LiquidFilmHalf_*.vtp' -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { return @{ T = $null; N = 0 } }
    $best = -1.0
    foreach ($f in $files) {
        $tag = $f.BaseName -replace '^LiquidFilmHalf_', ''
        if ($tag -eq 'ite_0000000000') { $solverTime = 0.0 }
        else {
            $digits = ($tag -replace '\D', '')
            if ($digits.Length -eq 0) { continue }
            $solverTime = [double]$digits / $TagScale
        }
        if ($solverTime -gt $best) { $best = $solverTime }
    }
    if ($best -lt 0) { return @{ T = $null; N = $files.Count } }
    return @{ T = $best / $TimeScale; N = $files.Count }
}

function Set-Cursor([int]$X, [int]$Y) {
    try { [Console]::SetCursorPosition($X, $Y) } catch { }
}

try { Clear-Host } catch { }
Write-Host ''

$finalState = 'unknown'
while ($true) {
    $cur = Get-CurrentT
    $t = $cur.T
    $alive = $null -ne (Get-Process -Id $ProcId -ErrorAction SilentlyContinue)
    if ($alive) { $state = 'running' }
    elseif ($null -ne $t -and $t -ge ($TargetT - 1e-6)) { $state = 'done' }
    else { $state = 'stopped' }

    $nowTxt = if ($null -eq $t) { '   n/a' } else { '{0,7:F2}' -f $t }
    $pct = if ($null -eq $t) { 0.0 } else { 100.0 * $t / $TargetT }
    $l1 = ('  Current T = {0}   [{1}]' -f $nowTxt, $state).PadRight(78)
    $l2 = ('  Target  T = {0,7:F2}' -f $TargetT).PadRight(78)
    $l3 = ('  frames    = {0,4}   solver pct = {1,5:F1}%' -f $cur.N, $pct).PadRight(78)
    Set-Cursor -X 0 -Y 0
    Write-Host $l1
    Write-Host $l2
    Write-Host $l3

    if (-not $alive) { $finalState = $state; break }
    Start-Sleep -Seconds $IntervalSec
}

Set-Cursor -X 0 -Y 4
if ($finalState -eq 'done') {
    Write-Host ('  Finished: reached T = {0:F2} of {1:F2}.' -f $t, $TargetT)
} else {
    $nowTxt = if ($null -eq $t) { 'n/a' } else { '{0:F2}' -f $t }
    Write-Host ('  Solver no longer running. Last T = {0} (target {1:F2}).' -f $nowTxt, $TargetT)
}
Write-Host ''
