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

precompile_only = get(ENV, "NNLC_PRECOMPILE_ONLY", "") == "1"
if !precompile_only
    Pkg.add(packages)
end

# Trigger precompilation unless the build is staging packages for a final
# relocatable bundle.  Those packages are precompiled after they reach their
# final path so Julia does not retain a second set of path-dependent caches.
skip_precompile = get(ENV, "NNLC_SKIP_PRECOMPILE", "") == "1"
if skip_precompile
    println("Package installation complete; final-location precompilation deferred.")
else
    println("Precompiling all packages...")
    failed = String[]
    for pkg in packages
        try
            Core.eval(Main, Meta.parse("using $pkg"))
            println("  ✓ $pkg")
        catch e
            push!(failed, pkg)
            println("  ⚠ $pkg ($(typeof(e)) — may work at runtime)")
        end
    end

    if !isempty(failed)
        error("Julia package precompilation failed for: $(join(failed, ", "))")
    end
    println("Package precompilation complete.")
end
