<#
Minimal progress window: shows only the current T and the target T.
Reads the newest "T=<number>" line at the start of a line in the solver log.
Must be run with pwsh (PowerShell 7).

Usage:
  pwsh -ExecutionPolicy Bypass -File E:\sphmethod\watch_T_progress.ps1
  pwsh -ExecutionPolicy Bypass -File E:\sphmethod\watch_T_progress.ps1 -Log <stdout.log> -TargetT 50 -ProcId 12345
#>
param(
    [string]$Log         = 'E:\sphmethod\SPH_results_center\recon_tapered_20260914\r24_surfreg_T50\stdout.log',
    [double]$TargetT     = 50,
    [int]   $ProcId      = 54112,
    [int]   $IntervalSec = 10
)

$Host.UI.RawUI.WindowTitle = "PR progress - now T / target T"

$rx = [regex]'(?m)^T=([0-9]+\.?[0-9]*)'

function Get-CurrentT {
    if (-not (Test-Path -LiteralPath $Log)) { return $null }
    try { $raw = Get-Content -LiteralPath $Log -Raw -ErrorAction Stop } catch { return $null }
    if ([string]::IsNullOrEmpty($raw)) { return $null }
    $hits = $rx.Matches($raw)
    if ($hits.Count -eq 0) { return $null }
    return [double]$hits[$hits.Count - 1].Groups[1].Value
}

function Write-Panel($t, $state) {
    $nowTxt = if ($null -eq $t) { '   n/a' } else { '{0,7:F2}' -f $t }
    $line1 = '  Current T = {0}   [{1}]' -f $nowTxt, $state
    $line2 = '  Target  T = {0,7:F2}' -f $TargetT
    $line1 = $line1.PadRight(78)
    $line2 = $line2.PadRight(78)
    Set-Cursor -X 0 -Y 0
    Write-Host $line1
    Write-Host $line2
}

function Set-Cursor([int]$X, [int]$Y) {
    # Only valid with a real console handle; silently ignore when redirected.
    try { [Console]::SetCursorPosition($X, $Y) } catch { }
}

try { Clear-Host } catch { }
Write-Host ''

$finalState = 'unknown'
while ($true) {
    $t = Get-CurrentT
    $alive = $null -ne (Get-Process -Id $ProcId -ErrorAction SilentlyContinue)

    if ($alive) {
        $state = 'running'
    }
    elseif ($null -ne $t -and $t -ge ($TargetT - 1e-6)) {
        $state = 'done'
    }
    else {
        $state = 'stopped'
    }

    Write-Panel $t $state

    if (-not $alive) { $finalState = $state; break }
    Start-Sleep -Seconds $IntervalSec
}

Set-Cursor -X 0 -Y 3
if ($finalState -eq 'done') {
    Write-Host ('  Finished: reached T = {0:F2} of {1:F2}.' -f $t, $TargetT)
} else {
    $nowTxt = if ($null -eq $t) { 'n/a' } else { '{0:F2}' -f $t }
    Write-Host ('  Solver is no longer running. Last T = {0} (target {1:F2}).' -f $nowTxt, $TargetT)
}
Write-Host ''
