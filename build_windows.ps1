param(
    [string]$Python = "py -3.11"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string]$CommandLine)
    & cmd.exe /c "$Python -m $CommandLine"
    if ($LASTEXITCODE -ne 0) { throw "Python command failed ($LASTEXITCODE)" }
}

Write-Host "Installing Python dependencies and PyInstaller ..."
Invoke-Python "pip install --upgrade pip"
Invoke-Python "pip install -e . pyinstaller"

Write-Host "Cleaning previous PyInstaller output ..."
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "build"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "dist"

Write-Host "Building one-file NNLC_Trainer.exe ..."
Invoke-Python "PyInstaller --clean --noconfirm nnlc_windows.spec"
$BundleExe = Join-Path $ProjectDir "dist\NNLC_Trainer.exe"
if (-not (Test-Path -LiteralPath $BundleExe -PathType Leaf)) {
    throw "one-file build validation failed: missing $BundleExe"
}
if (Test-Path -LiteralPath (Join-Path $ProjectDir "dist\NNLC_Trainer") -PathType Container) {
    throw "one-file build validation failed: unexpected one-dir output was created"
}

Write-Host "Testing --help without a Julia runtime ..."
$HelpOutput = (& $BundleExe --help 2>&1 | Out-String)
$HelpExitCode = $LASTEXITCODE
Write-Host $HelpOutput
if ($HelpExitCode -ne 0 -or $HelpOutput -match "Traceback|UnicodeEncodeError") {
    throw "one-file NNLC_Trainer --help failed or printed a Python traceback ($HelpExitCode)"
}

Write-Host "Testing isolated rlog worker startup and pipe communication ..."
$WorkerStartInfo = New-Object System.Diagnostics.ProcessStartInfo
$WorkerStartInfo.FileName = $BundleExe
$WorkerStartInfo.Arguments = "--run-module nnlc_tools.extract_rlog_worker"
$WorkerStartInfo.UseShellExecute = $false
$WorkerStartInfo.CreateNoWindow = $true
$WorkerStartInfo.RedirectStandardInput = $true
$WorkerStartInfo.RedirectStandardOutput = $true
$WorkerStartInfo.RedirectStandardError = $true
$WorkerProcess = [System.Diagnostics.Process]::Start($WorkerStartInfo)
$WorkerProcess.StandardInput.WriteLine("{}")
$WorkerProcess.StandardInput.Close()
if (-not $WorkerProcess.WaitForExit(30000)) {
    $WorkerProcess.Kill()
    throw "one-file build validation failed: isolated rlog worker did not exit after stdin closed"
}
$WorkerStdout = $WorkerProcess.StandardOutput.ReadToEnd()
$WorkerStderr = $WorkerProcess.StandardError.ReadToEnd()
if ($WorkerProcess.ExitCode -ne 0) {
    $Message = "one-file build validation failed: isolated rlog worker exited " +
        "with $($WorkerProcess.ExitCode): $WorkerStderr"
    throw $Message
}
try {
    $WorkerResponse = $WorkerStdout | ConvertFrom-Json
} catch {
    throw "one-file build validation failed: worker returned invalid JSON: $WorkerStdout"
}
if ($WorkerResponse.status -ne "error") {
    throw "one-file build validation failed: worker returned an unexpected response: $WorkerStdout"
}

$SizeMb = [math]::Round((Get-Item -LiteralPath $BundleExe).Length / 1MB, 1)
Write-Host "Validated one-file executable: $SizeMb MB"
Write-Host "Done: $BundleExe"
