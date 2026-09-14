param(
    [Parameter(Mandatory = $true)]
    [string]$JuliaDir,
    [string]$JuliaDepot = ""
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Remove-JuliaPackageDevelopmentFiles {
    param([Parameter(Mandatory = $true)][string]$DepotPath)

    $PackagesRoot = Join-Path $DepotPath "packages"
    if (-not (Test-Path -LiteralPath $PackagesRoot -PathType Container)) {
        return
    }

    $OptionalDirectoryNames = @(
        ".github",
        "benchmark",
        "benchmarks",
        "docs",
        "example",
        "examples",
        "test",
        "tests"
    )
    $RemovedCount = 0
    foreach ($PackageDir in (Get-ChildItem -LiteralPath $PackagesRoot -Directory -Force)) {
        foreach ($PackageVersionDir in (Get-ChildItem -LiteralPath $PackageDir.FullName -Directory -Force)) {
            foreach ($DirectoryName in $OptionalDirectoryNames) {
                $Candidate = Join-Path $PackageVersionDir.FullName $DirectoryName
                if (Test-Path -LiteralPath $Candidate) {
                    Remove-Item -LiteralPath $Candidate -Recurse -Force
                    $RemovedCount += 1
                }
            }
        }
    }
    Write-Host "Removed $RemovedCount Julia package documentation/test directories."
}

$Config = Get-Content (Join-Path $ProjectDir "windows_runtime.json") -Raw | ConvertFrom-Json
$JuliaDir = (Resolve-Path $JuliaDir).Path
if (-not (Test-Path -LiteralPath (Join-Path $JuliaDir "bin\julia.exe") -PathType Leaf)) {
    throw "JuliaDir must contain bin\julia.exe: $JuliaDir"
}
if ($JuliaDepot) {
    $JuliaDepot = (Resolve-Path $JuliaDepot).Path
}

$DistDir = Join-Path $ProjectDir "dist"
$StagingRoot = Join-Path $DistDir "julia-runtime-staging"
$RuntimeDir = Join-Path $StagingRoot $Config.runtime_directory
$RuntimeTarget = Join-Path $RuntimeDir "julia-runtime"
$DepotTarget = Join-Path $RuntimeDir "julia-depot"
$ArchivePath = Join-Path $DistDir $Config.archive_name

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $StagingRoot
Remove-Item -Force -ErrorAction SilentlyContinue $ArchivePath
New-Item -ItemType Directory -Force $RuntimeDir | Out-Null

Write-Host "Copying Julia $($Config.julia_version) runtime ..."
Copy-Item -Recurse -Force $JuliaDir $RuntimeTarget
if ($JuliaDepot) {
    Write-Host "Copying preinstalled Julia depot ..."
    Copy-Item -Recurse -Force $JuliaDepot $DepotTarget
} else {
    New-Item -ItemType Directory -Force $DepotTarget | Out-Null
}

$JuliaExe = Join-Path $RuntimeTarget "bin\julia.exe"
$env:JULIA_DEPOT_PATH = $DepotTarget
$env:JULIA_PKG_PRECOMPILE_AUTO = "0"
$env:NNLC_WINDOWS_CPU_BUILD = "1"

if (-not $JuliaDepot) {
    Write-Host "Installing CPU-only Julia packages ..."
    $env:NNLC_SKIP_PRECOMPILE = "1"
    try {
        & $JuliaExe --startup-file=no (Join-Path $ProjectDir "training\install_packages.jl")
        if ($LASTEXITCODE -ne 0) { throw "Julia package installation failed ($LASTEXITCODE)" }
    } finally {
        Remove-Item Env:\NNLC_SKIP_PRECOMPILE -ErrorAction SilentlyContinue
    }
}

Write-Host "Precompiling packages in their final runtime path ..."
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $DepotTarget "compiled")
$env:NNLC_PRECOMPILE_ONLY = "1"
try {
    & $JuliaExe --startup-file=no (Join-Path $ProjectDir "training\install_packages.jl")
    if ($LASTEXITCODE -ne 0) { throw "Julia package precompile failed ($LASTEXITCODE)" }
} finally {
    Remove-Item Env:\NNLC_PRECOMPILE_ONLY -ErrorAction SilentlyContinue
}

foreach ($CacheDirName in @("scratchspaces", "logs", "clones")) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $DepotTarget $CacheDirName)
}
Remove-JuliaPackageDevelopmentFiles -DepotPath $DepotTarget

$VersionOutput = (& $JuliaExe --startup-file=no --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $VersionOutput -notmatch [regex]::Escape($Config.julia_version)) {
    throw "Staged Julia version mismatch: $VersionOutput"
}

$Manifest = [ordered]@{
    schema_version = [int]$Config.schema_version
    runtime_version = [string]$Config.runtime_version
    platform = [string]$Config.platform
    julia_version = [string]$Config.julia_version
    archive_name = [string]$Config.archive_name
    release_tag = [string]$Config.release_tag
    created_utc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
}
$Manifest | ConvertTo-Json | Set-Content (Join-Path $RuntimeDir "runtime-manifest.json") -Encoding UTF8

Write-Host "Creating ZIP archive ..."
$SevenZip = (Get-Command 7z.exe -ErrorAction SilentlyContinue).Source
if ($SevenZip) {
    $ArchiveExcludes = @(
        "-xr!.github",
        "-xr!benchmark",
        "-xr!benchmarks",
        "-xr!docs",
        "-xr!example",
        "-xr!examples",
        "-xr!test",
        "-xr!tests"
    )
    Push-Location $StagingRoot
    try {
        & $SevenZip a -tzip -mx=9 $ArchivePath $Config.runtime_directory @ArchiveExcludes | Write-Host
        if ($LASTEXITCODE -ne 0) { throw "7-Zip archive creation failed ($LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
    & $SevenZip t $ArchivePath | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "Runtime ZIP integrity test failed ($LASTEXITCODE)" }
} else {
    Compress-Archive -Path $RuntimeDir -DestinationPath $ArchivePath -CompressionLevel Optimal
}

if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
    throw "Runtime ZIP was not created: $ArchivePath"
}
$ArchiveMb = [math]::Round((Get-Item -LiteralPath $ArchivePath).Length / 1MB, 1)
Remove-Item -Recurse -Force $StagingRoot
Write-Host "Done: $ArchivePath ($ArchiveMb MB)"
