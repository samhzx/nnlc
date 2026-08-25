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

if (-not $JuliaDepot -and -not $SkipJuliaPackages) {
    Write-Host "Installing and precompiling Julia packages into $env:JULIA_DEPOT_PATH ..."
    & "julia-runtime\bin\julia.exe" --startup-file=no "training\install_packages.jl"
    if ($LASTEXITCODE -ne 0) { throw "Julia package installation failed ($LASTEXITCODE)" }
}

Write-Host "Installing Python dependencies and PyInstaller ..."
Invoke-Python "pip install --upgrade pip"
Invoke-Python "pip install -e . pyinstaller"

Write-Host "Building NNLC_Trainer.exe ..."
Invoke-Python "PyInstaller --clean --noconfirm nnlc_windows.spec"
Write-Host "Done: $ProjectDir\dist\NNLC_Trainer.exe"
