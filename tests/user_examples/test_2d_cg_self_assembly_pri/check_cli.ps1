<#
  Minimal CLI-validation check for test_2d_cg_assembly_pri.

  Guards the fix in "fix(cg-assembly): report invalid --frames instead of
  aborting silently":

    1. an invalid command line must print a readable message on stderr and
       exit with a defined non-zero status -- NOT the silent MSVC fastfail
       0xC0000409 that an escaped exception used to produce;
    2. the legal `--frames=2` path must be unaffected and still exit 0.

  Usage:
      .\check_cli.ps1                       # auto-locate the Release exe
      .\check_cli.ps1 -Exe <path\to.exe>    # explicit

  Exit status: 0 = all checks passed, 1 = at least one check failed.
#>
[CmdletBinding()]
param(
    [string]$Exe = '',
    [string]$WorkDir = (Join-Path $env:TEMP 'cg_cli_check')
)

$ErrorActionPreference = 'Stop'
$FailFast = -1073740791   # 0xC0000409, what an escaped exception used to give

if (-not $Exe) {
    $repo = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
    $cand = Get-ChildItem -LiteralPath $repo -Directory -Filter 'build*' -ErrorAction SilentlyContinue |
        ForEach-Object {
            Join-Path $_.FullName 'tests\user_examples\test_2d_cg_self_assembly_pri\bin\Release\test_2d_cg_assembly_pri.exe'
        } | Where-Object { Test-Path -LiteralPath $_ } |
        ForEach-Object { Get-Item -LiteralPath $_ } |
        Sort-Object LastWriteTime -Descending
    if (-not $cand) { throw 'cannot locate test_2d_cg_assembly_pri.exe; pass -Exe <path>' }
    $Exe = $cand[0].FullName
}
if (-not (Test-Path -LiteralPath $Exe)) { throw "exe not found: $Exe" }

Write-Host "exe : $Exe"
Write-Host "work: $WorkDir"

function Invoke-Case {
    param([string[]]$CaseArgs, [string]$Tag)
    $dir = Join-Path $WorkDir $Tag
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $outFile = Join-Path $dir 'stdout.log'
    $errFile = Join-Path $dir 'stderr.log'
    Push-Location $dir
    try {
        & $Exe @CaseArgs 1> $outFile 2> $errFile
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    [pscustomobject]@{
        Tag      = $Tag
        ExitCode = $code
        StdOut   = (Get-Content -LiteralPath $outFile -Raw -ErrorAction SilentlyContinue)
        StdErr   = (Get-Content -LiteralPath $errFile -Raw -ErrorAction SilentlyContinue)
    }
}

$failures = @()

# ---- 1. invalid: --frames=1 must be rejected READABLY ------------------------
$bad = Invoke-Case -Tag 'frames1' -CaseArgs @(
    '--case=1', '--end-time=2', '--frames=1', '--threads=1', '--output-tag=cli_bad')
Write-Host ("`n[1] --frames=1   exit={0}" -f $bad.ExitCode)
Write-Host ("    stderr: {0}" -f ($bad.StdErr -replace "`r?`n", ' ').Trim())

if ($bad.ExitCode -eq 0) {
    $failures += '--frames=1 was accepted (expected a non-zero exit status)'
}
if ($bad.ExitCode -eq $FailFast) {
    $failures += "--frames=1 still dies by fastfail ($FailFast) instead of reporting"
}
if ([string]::IsNullOrWhiteSpace($bad.StdErr)) {
    $failures += '--frames=1 produced no message on stderr (silent abort)'
} elseif ($bad.StdErr -notmatch 'frames') {
    $failures += "--frames=1 message does not mention --frames: $($bad.StdErr.Trim())"
}

# ---- 2. legal: --frames=2 must still work ------------------------------------
$ok = Invoke-Case -Tag 'frames2' -CaseArgs @(
    '--case=1', '--morse-depth-d=2', '--wall-depth-d0=3', '--area-fraction=0.2',
    '--end-time=2', '--frames=2', '--threads=1', '--output-tag=cli_ok')
Write-Host ("`n[2] --frames=2   exit={0}" -f $ok.ExitCode)
Write-Host ("    stderr bytes: {0}" -f $ok.StdErr.Length)

if ($ok.ExitCode -ne 0) {
    $failures += "--frames=2 exited with $($ok.ExitCode) (expected 0)"
}
if ($ok.StdOut -notmatch 'finished:') {
    $failures += '--frames=2 did not report a finished run'
}
if (-not [string]::IsNullOrWhiteSpace($ok.StdErr)) {
    $failures += "--frames=2 wrote to stderr: $($ok.StdErr.Trim())"
}

# ---- verdict -----------------------------------------------------------------
if ($failures.Count -gt 0) {
    Write-Host "`nFAIL" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host "`nPASS" -ForegroundColor Green
exit 0
