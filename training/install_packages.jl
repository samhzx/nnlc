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
    "CUDA",
    "ArgParse",
    "ModelingToolkit",
    "TeeStreams",
]

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
