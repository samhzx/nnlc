param(
    [Parameter(Mandatory = $true)]
    [string]$JuliaDir,
    [string]$Python = "py -3.11",
    [string]$JuliaDepot = "",
    [switch]$SkipJuliaPackages
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string]$CommandLine)
    # Accept one command string so pip flags such as ``-e`` are passed through
    # to Python instead of being interpreted as PowerShell function parameters.
    & cmd.exe /c "$Python -m $CommandLine"
    if ($LASTEXITCODE -ne 0) { throw "Python command failed ($LASTEXITCODE)" }
}

$JuliaDir = (Resolve-Path $JuliaDir).Path
$JuliaDepotSource = ""
if ($JuliaDepot) {
    $JuliaDepotSource = (Resolve-Path $JuliaDepot).Path
}
if (-not (Test-Path (Join-Path $JuliaDir "bin\julia.exe"))) {
    throw "JuliaDir must point to a Julia installation containing bin\julia.exe"
}

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "julia-runtime"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "julia-depot"
Copy-Item -Recurse -Force $JuliaDir "julia-runtime"
if ($JuliaDepot) {
    Copy-Item -Recurse -Force $JuliaDepotSource "julia-depot"
} else {
    New-Item -ItemType Directory -Force "julia-depot" | Out-Null
}
$env:JULIA_DEPOT_PATH = (Resolve-Path "julia-depot").Path
$env:NNLC_WINDOWS_CPU_BUILD = "1"

if (-not $JuliaDepot -and -not $SkipJuliaPackages) {
    Write-Host "Installing and precompiling Julia packages into $env:JULIA_DEPOT_PATH ..."
    & "julia-runtime\bin\julia.exe" --startup-file=no "training\install_packages.jl"
    if ($LASTEXITCODE -ne 0) { throw "Julia package installation failed ($LASTEXITCODE)" }
}

Write-Host "Removing Julia package-manager caches ..."
foreach ($CacheDir in @("julia-depot\scratchspaces", "julia-depot\logs", "julia-depot\clones")) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $CacheDir
}

Write-Host "Installing Python dependencies and PyInstaller ..."
Invoke-Python "pip install --upgrade pip"
Invoke-Python "pip install -e . pyinstaller"

Write-Host "Cleaning previous PyInstaller output ..."
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "build"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "dist"

Write-Host "Building NNLC_Trainer.exe ..."
Invoke-Python "PyInstaller --clean --noconfirm nnlc_windows.spec"
$BundleDir = Join-Path $ProjectDir "dist\NNLC_Trainer"
$BundleExe = Join-Path $BundleDir "NNLC_Trainer.exe"
if (-not (Test-Path $BundleExe)) {
    throw "one-dir build validation failed: missing $BundleExe"
}
if (Test-Path (Join-Path $ProjectDir "dist\NNLC_Trainer.exe")) {
    throw "one-dir build validation failed: unexpected one-file executable was created"
}
$BundleFileCount = @(Get-ChildItem -Recurse -File $BundleDir).Count
if ($BundleFileCount -lt 10) {
    throw "one-dir build validation failed: bundle contains only $BundleFileCount files"
}
Write-Host "Validated one-dir bundle: $BundleFileCount files"
Write-Host "Done: $BundleExe"
