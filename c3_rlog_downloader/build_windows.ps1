param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $SourceDir ".venv"
$VenvScriptsDir = Join-Path $VenvDir "Scripts"
$VenvPython = Join-Path $VenvScriptsDir "python.exe"
$DistDir = Join-Path $SourceDir "dist"
$WorkDir = Join-Path $SourceDir "build"

Write-Host "Creating or reusing the standalone downloader build environment ..."
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    & $PythonLauncher ("-" + $PythonVersion) -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Could not create the downloader virtual environment." }
}

Write-Host "Installing only downloader build dependencies ..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip in the downloader build environment." }
& $VenvPython -m pip install -r (Join-Path $SourceDir "requirements-build.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install the downloader build dependencies." }

Write-Host "Generating the multi-resolution application icon ..."
& $VenvPython (Join-Path $SourceDir "make_icon.py")
if ($LASTEXITCODE -ne 0) { throw "Could not generate the downloader icon." }

Write-Host "Building C3_Rlog_Downloader.exe ..."
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath $DistDir `
    --workpath $WorkDir `
    (Join-Path $SourceDir "C3_Rlog_Downloader.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

$BundleExe = Join-Path $DistDir "C3_Rlog_Downloader.exe"
if (-not (Test-Path -LiteralPath $BundleExe -PathType Leaf)) {
    throw "Build output is missing: $BundleExe"
}

$VersionOutput = (& $BundleExe --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $VersionOutput -ne "1.1.0") {
    throw "Standalone executable smoke test failed: $VersionOutput"
}

$SizeMb = [math]::Round((Get-Item -LiteralPath $BundleExe).Length / 1MB, 1)
Write-Host "Validated standalone executable: $SizeMb MB"
Write-Host "Done: $BundleExe"
