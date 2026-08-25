#!/usr/bin/env julia
# Install and precompile all Julia packages needed for NNLC training.
# Used during automated builds to avoid a 20-minute wait on first run.

import Pkg

packages = [
    "CSV",
    "DataFrames",
    "StatsBase",
    "MultivariateStats",
    "Flux",
    "MLDataUtils",
    "MLUtils",
    "Statistics",
    "LinearAlgebra",
    "PyFormattedStrings",
    "Random",
    "ProgressMeter",
    "Zygote",
    "Optim",
    "FluxOptTools",
    "Plots",
    "BSON",
    "CategoricalArrays",
    "SharedArrays",
    "SplitApplyCombine",
    "InvertedIndices",
    "JSON",
    "Dates",
    "ArgParse",
    "TeeStreams",
]

# The packaged Windows GUI always runs the trainer in CPU mode.  CUDA pulls
# several gigabytes of NVIDIA runtime artifacts into the depot, while the
# training script already treats CUDA as an optional backend.  Keep the GPU
# packages for normal/manual installs, but omit them for the Windows bundle.
windows_cpu_build = get(ENV, "NNLC_WINDOWS_CPU_BUILD", "") == "1"
if !windows_cpu_build
    push!(packages, "CUDA")
    push!(packages, "ModelingToolkit")
end

Pkg.add(packages)

# Trigger precompilation
println("Precompiling all packages...")
for pkg in packages
    try
        Core.eval(Main, Meta.parse("using $pkg"))
        println("  ✓ $pkg")
    catch e
        println("  ⚠ $pkg ($(typeof(e)) — may work at runtime)")
    end
end

println("Package installation complete.")
