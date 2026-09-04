# import Pkg
# packages = [
#     "CSV",
#     "DataFrames",
#     "StatsBase",
#     "MultivariateStats",
#     "Flux",
#     "MLDataUtils",
#     "MLUtils",
#     "Statistics",
#     "LinearAlgebra",
#     "PyFormattedStrings",
#     "Random",
#     "ProgressMeter",
#     "Zygote",
#     "Optim",
#     "FluxOptTools",
#     "Plots",
#     "BSON",
#     "CategoricalArrays",
#     "SharedArrays",
#     "SplitApplyCombine",
#     "InvertedIndices",
#     "JSON",
#     # "Feather",  # No longer needed, using CSV
#     "Dates",
#     "ArgParse",
#     "ModelingToolkit",
#     "TeeStreams"
# ]

# for package in packages
#     Pkg.add(package)
# end

# Import packages
using CSV
using DataFrames
using StatsBase
using MultivariateStats
using Flux
using Flux: params, train!, mse
using Flux.Optimisers
using MLUtils: DataLoader
using MLDataUtils #: splitobs, rescale
using Statistics #: mean, std
using LinearAlgebra #: diag
using PyFormattedStrings
using Base.Threads
using Random
using ProgressMeter
using Zygote, Optim, FluxOptTools
using StatsBase: sample
using Plots
using Plots.PlotMeasures
using BSON: @save, @load
using CategoricalArrays
using SharedArrays
using SplitApplyCombine
using InvertedIndices
using JSON

using Dates
using ArgParse
using Optim

using TeeStreams

# Custom AdaGrad optimizer that uses Float32 literals
struct CustomAdaGrad <: Optimisers.AbstractRule
  eta::Float32
  epsilon::Float32
end

# Define the update rule
function Optimisers.apply!(o::CustomAdaGrad, state, x, Δ)
  η, ϵ = o.eta, o.epsilon
  acc = state
  @. acc += Δ^2
  @. Δ *= η / (√acc + ϵ)
  return (acc, Δ)
end

Optimisers.init(o::CustomAdaGrad, x::AbstractArray) = fill!(similar(x, Float32), 0f0)

# Define constants
const t_list = [-0.3f0 -0.2f0 -0.1f0 0.3f0 0.6f0 1f0 1.5f0]
const non_model_columns = Set([
  :torque_output,
  :combined_column,
  :v_ego_bins,
  :desired_lateral_accel_bins,
  :friction_input_bins,
  :roll_bins,
  :route_id,
  :timestamp,
])
const temporal_split_gap_rows = 180
const max_test_rows = 100_000

model_feature_names(data::DataFrame) = filter(name -> Symbol(name) ∉ non_model_columns, names(data))

function create_folder_with_iterator(path::AbstractString, folder_name::AbstractString; make_new=true)::String
  full_path = joinpath(path, folder_name)
  i = 1
  if isdir(full_path) && !make_new
      return full_path
  end
  while isdir(full_path)
      folder_name_with_iterator = string(folder_name, "_", i)
      full_path = joinpath(path, folder_name_with_iterator)
      i += 1
  end
  mkdir(full_path)
  return full_path
end

function describe(arr)::String
  n = length(arr)
  μ = mean(arr)
  σ = std(arr)
  minimum_value = minimum(arr)
  maximum_value = maximum(arr)
  quartiles = quantile(arr, [0.25f0, 0.5f0, 0.75f0])
  
  return f"n: {n}, mean: {μ:0.6f}, std: {σ:0.6f}, min: {minimum_value:0.6f}, max: {maximum_value:0.6f}, 25%: {quartiles[1]:0.6f}, 50%: {quartiles[2]:0.6f}, 75%: {quartiles[3]:0.6f}"

end

function load_data(infile::String, use_existing_data::Bool, outdir::String, out_streams)::DataFrame
  # Load the data into a DataFrame
  if use_existing_data
    infile = replace(infile, ".csv" => "_balanced.csv")

    println(out_streams, "Using existing data")
  end
  data = CSV.read(infile, DataFrame)
  if !use_existing_data
    println(out_streams, "Loading data...")
    # Remove rows with missing data
    data = data[completecases(data), :]

    # Filter out inactive and standstill rows (extractor outputs all rows)
    if "active" in names(data)
      old_nrows = nrow(data)
      data = filter(row -> row.active == true, data)
      println(out_streams, f"Filtered out {old_nrows - nrow(data)} inactive rows")
    end
    if "standstill" in names(data)
      old_nrows = nrow(data)
      data = filter(row -> row.standstill == false, data)
      println(out_streams, f"Filtered out {old_nrows - nrow(data)} standstill rows")
    end
    # Select only columns needed for training (extractor outputs many extra columns)
    # Model inputs: v_ego, desired_lateral_accel, friction_input, roll, temporal lat accels, temporal rolls
    # Target: torque_output
    # friction_input 由 extract_lateral_data.py 精确计算，与 nnlc.py 运行时 update_friction_input 逻辑一致
    temporal_lat_accel_cols = filter(c -> occursin(r"^desired_lateral_accel_t[mp]\d+$", c), names(data))
    temporal_roll_cols = filter(c -> occursin(r"^roll_t[mp]\d+$", c), names(data))
    metadata_cols = filter(c -> c in names(data), ["route_id", "timestamp"])
    keep_cols = vcat(["v_ego", "desired_lateral_accel", "friction_input", "roll", "torque_output"], temporal_lat_accel_cols, temporal_roll_cols, metadata_cols)
    select!(data, Symbol.(keep_cols))

    # 重排列顺序：与 nnlc.py 运行时 feedforward (nn_input) 输入完全一致
    # 列顺序：v_ego, desired_lateral_accel, friction_input, roll, torque_output,
    # 时序desired_lat(7), 时序roll(7)，最后保留不参与训练的路线划分元数据。
    select!(data, keep_cols)

    if nrow(data) == 0
      error("No valid training rows remain after active/standstill filtering")
    end

    println(out_streams, f"Loaded {nrow(data)} rows")
    println(out_streams, f"Data {data[sample(1:nrow(data), min(20, nrow(data))), :]}")
    for col in names(data)
      if typeof(data[1, col]) == Float64
        println(out_streams, "$col: $(describe(collect(data[:,col])))")
      end
    end

    println(out_streams, "Filtering out extreme values")
    min_vego = minimum(data[!, :v_ego])

    # setup bin ranges
    mm_v_ego = [0.1, 45.0]
    mm_torque_output = [-2.0, 2.0]
    mm_lateral_accel = [-4.1, 4.1]
    mm_friction_input = [-5.0, 5.0]
    mm_roll = [-0.20, 0.20]

    # setup bins
    nbins = 121
    # this is the max number of points that will go in each bin. it's low because lowspeed data is scarse.
    sample_size = 20
    step_v_ego = (mm_v_ego[2] - mm_v_ego[1]) / nbins
    step_torque_output = (mm_torque_output[2] - mm_torque_output[1]) / nbins
    step_lateral_accel = (mm_lateral_accel[2] - mm_lateral_accel[1]) / nbins
    step_friction_input = (mm_friction_input[2] - mm_friction_input[1]) / nbins
    step_roll = (mm_roll[2] - mm_roll[1]) / nbins

    function filter_columns(df::DataFrame, partial_match::String, tol)::DataFrame
        for col_name in names(df)
            if occursin(partial_match, col_name)
                # Apply your filter or transformation to the column here
                df = filter(row -> abs(row[col_name]) < tol, df)
            end
        end
        return df
    end

    # desired_lateral_accel_tp03 is kept as a temporal model input feature

    # filter data
    old_nrows = nrow(data)
    println(out_streams, f"{old_nrows} rows before filtering")
    data = filter(row -> mm_v_ego[1] < row.v_ego < mm_v_ego[2], data)
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with v_ego outside [{mm_v_ego[1]}, {mm_v_ego[2]}]")
    old_nrows = nrow(data)
    data = filter(row -> abs(row.torque_output) <= mm_torque_output[2], data)
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with torque_output outside [{-mm_torque_output[2]}, {mm_torque_output[2]}]")
    old_nrows = nrow(data)
    data = filter_columns(data, "lateral_accel", mm_lateral_accel[2])
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with lateral_accel outside [{-mm_lateral_accel[2]}, {mm_lateral_accel[2]}]")
    old_nrows = nrow(data)
    data = filter_columns(data, "friction_input", mm_friction_input[2])
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with friction_input outside [{-mm_friction_input[2]}, {mm_friction_input[2]}]")
    old_nrows = nrow(data)
    data = filter_columns(data, "roll", mm_roll[2])
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with roll outside [{-mm_roll[2]}, {mm_roll[2]}]")
    println(out_streams, f"{nrow(data)} rows after filtering")

    if nrow(data) == 0
      error("No training rows remain after range filtering")
    end

    for col in names(data)
      if typeof(data[1, col]) == Float64
        println(out_streams, "$col: $(describe(collect(data[:,col])))")
      end
    end

    println(out_streams, f"Calculating bins")
    data[!, :v_ego_bins] = cut(data[!, :v_ego], mm_v_ego[1]:step_v_ego:mm_v_ego[2])
    data[!, :desired_lateral_accel_bins] = cut(data[!, :desired_lateral_accel], mm_lateral_accel[1]:step_lateral_accel:mm_lateral_accel[2])
    data[!, :roll_bins] = cut(data[!, :roll], mm_roll[1]:step_roll:mm_roll[2])
    data[!, :friction_input_bins] = cut(data[!, :friction_input], mm_friction_input[1]:step_friction_input:mm_friction_input[2])

    # create a combined column for balancing
    data[!,:combined_column] = string.(data[!,:v_ego_bins], "_", data[!,:desired_lateral_accel_bins])
    
    if isempty(unique(data[!, :combined_column]))
      error("No training bins remain after preprocessing")
    end
    println(out_streams, f"Prepared {nrow(data)} rows across {length(unique(data[!, :combined_column]))} bins")
  else
    println(out_streams, "Loading preprocessed data...")
  end

  return data
end

# Keep complete routes on one side of the split. If there is only one usable
# route, use a chronological split with a gap large enough to separate the
# temporal input windows on both sides of the boundary.
function split_train_test_indices(data::DataFrame, out_streams; train_fraction::Float64=0.8)
  row_count = nrow(data)
  if row_count < 4
    error("至少需要 4 条有效数据，才能保证训练集和测试集各至少 2 条")
  end

  if :route_id in propertynames(data)
    route_groups = Dict{String, Vector{Int}}()
    for (index, route_id) in enumerate(data[!, :route_id])
      push!(get!(route_groups, string(route_id), Int[]), index)
    end

    if length(route_groups) >= 2
      target_test_count = clamp(round(Int, (1 - train_fraction) * row_count), 2, row_count - 2)
      route_ids = collect(keys(route_groups))
      rng = MersenneTwister(42)
      shuffle!(rng, route_ids)

      # Start with the single route closest to the requested test size, then
      # greedily add routes only while doing so improves the row-count ratio.
      test_route_ids = String[]
      candidate_route = nothing
      candidate_distance = typemax(Int)
      for route_id in route_ids
        route_count = length(route_groups[route_id])
        if route_count >= 2 && row_count - route_count >= 2
          distance = abs(route_count - target_test_count)
          if distance < candidate_distance
            candidate_route = route_id
            candidate_distance = distance
          end
        end
      end

      if candidate_route !== nothing
        push!(test_route_ids, candidate_route)
        test_count = length(route_groups[candidate_route])
        remaining_route_ids = filter(route_id -> route_id != candidate_route, route_ids)
        sort!(remaining_route_ids, by=route_id -> length(route_groups[route_id]))
        for route_id in remaining_route_ids
          route_count = length(route_groups[route_id])
          new_test_count = test_count + route_count
          if row_count - new_test_count >= 2 && abs(new_test_count - target_test_count) < abs(test_count - target_test_count)
            push!(test_route_ids, route_id)
            test_count = new_test_count
          end
        end

        test_routes = Set(test_route_ids)
        train_indices = Int[]
        test_indices = Int[]
        for (route_id, indices) in route_groups
          append!(route_id in test_routes ? test_indices : train_indices, indices)
        end
        if length(train_indices) >= 2 && length(test_indices) >= 2
          println(out_streams, "Route split: training=$(length(train_indices)) rows from $(length(route_groups) - length(test_routes)) routes; test=$(length(test_indices)) rows from $(length(test_routes)) routes")
          return train_indices, test_indices
        end
      end
    end
  end

  println(out_streams, "无法进行完整路线隔离，使用带时序隔离区的顺序划分")
  ordered_indices = :timestamp in propertynames(data) ? sortperm(data[!, :timestamp]) : collect(1:row_count)
  split_gap = min(temporal_split_gap_rows, row_count ÷ 5)
  usable_rows = row_count - split_gap
  train_count = clamp(round(Int, train_fraction * usable_rows), 2, usable_rows - 2)
  test_start = train_count + split_gap + 1
  train_indices = ordered_indices[1:train_count]
  test_indices = ordered_indices[test_start:end]
  if length(train_indices) < 2 || length(test_indices) < 2
    error("训练集和测试集都至少需要 2 条数据")
  end
  println(out_streams, "Sequential split: training=$(length(train_indices)); gap=$split_gap; test=$(length(test_indices))")
  return train_indices, test_indices
end

function limit_test_indices(test_indices::Vector{Int}, out_streams; row_limit::Int=max_test_rows)
  if length(test_indices) <= row_limit
    return test_indices
  end
  rng = MersenneTwister(43)
  limited_indices = sample(rng, test_indices, row_limit; replace=false)
  println(out_streams, "Test rows limited from $(length(test_indices)) to $(length(limited_indices)) to bound memory usage")
  return limited_indices
end

function balance_training_data(data::DataFrame, row_indices::Vector{Int}, out_streams; sample_size::Int=20)
  bin_indices = Dict{String, Vector{Int}}()
  for row_index in row_indices
    label = string(data[row_index, :combined_column])
    push!(get!(bin_indices, label, Int[]), row_index)
  end
  if isempty(bin_indices)
    error("No training bins remain after preprocessing")
  end
  prog = ProgressMeter.Progress(length(bin_indices), 1, "Balancing training bins:")
  println(out_streams, f"Balancing training data into {length(bin_indices)} bins (max {sample_size} rows/bin)")

  rng = MersenneTwister(44)
  sampled_indices = Int[]
  sizehint!(sampled_indices, min(length(row_indices), length(bin_indices) * sample_size))
  for indices in values(bin_indices)
    count = min(sample_size, length(indices))
    append!(sampled_indices, sample(rng, indices, count; replace=false))
    next!(prog)
  end
  shuffle!(rng, sampled_indices)
  balanced = data[sampled_indices, :]
  println(out_streams, f"Training rows after balancing: {nrow(balanced)}")
  return balanced
end

function train_model(working_dir::String, use_existing_model::Bool, data::DataFrame, out_streams; force_cpu::Bool=true, requested_batch_size::Int=16384)::NamedTuple{(:model, :input_mean, :input_std, :X_train, :y_train, :X_test, :y_test, :test_loss), Tuple{Flux.Chain, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Matrix{Float32}, Vector{Float32}, Float32}}
  model_path = joinpath(working_dir, Base.basename(working_dir))

  if nrow(data) < 4
    error("至少需要 4 条有效数据，才能保证训练集和测试集各至少 2 条")
  end

  feature_names = model_feature_names(data)

  # Keep only row indices until sampling is complete. This avoids materializing
  # full 80/20 DataFrame copies for large extracted datasets.
  train_indices, test_indices = split_train_test_indices(data, out_streams)
  test_indices = limit_test_indices(test_indices, out_streams)
  train = balance_training_data(data, train_indices, out_streams)
  test = data[test_indices, :]
  if nrow(train) == 0 || nrow(test) == 0
    error("Training/test split produced an empty partition")
  end

  # Compute normalization statistics from training data only. This avoids
  # leaking test-set distribution information into the model.
  input_mean = Matrix{Float32}(undef, 1, length(feature_names))
  input_std = Matrix{Float32}(undef, 1, length(feature_names))
  for (index, column_name) in enumerate(feature_names)
    column = Float32.(train[!, column_name])
    input_mean[1, index] = mean(column)
    column_std = std(column)
    input_std[1, index] = isfinite(column_std) && column_std > 0f0 ? Float32(column_std) : 1f0
  end

  # Keep only the numeric training columns. Symmetric examples are generated
  # per batch below instead of duplicating both DataFrames in memory.
  model_columns = vcat(feature_names, ["torque_output"])
  select!(train, Symbol.(model_columns))
  select!(test, Symbol.(model_columns))
  println(out_streams, "Training rows: $(nrow(train)); test rows: $(nrow(test))")

  # The packaged trainer intentionally supports CPU only.  Keeping this
  # decision explicit makes results and memory usage deterministic.
  device = cpu
  println(out_streams, "Using device: CPU")

  # Convert to Float32 before normalization so the retained arrays are compact.
  X_train = Matrix{Float32}(select(train, Not([:torque_output])))
  @fastmath @. X_train = (X_train - input_mean) / input_std
  X_train = X_train |> device
  
  y_train = Array{Float32}(train[:, "torque_output"]) |> device
  
  X_test = Matrix{Float32}(select(test, Not([:torque_output])))
  @fastmath @. X_test = (X_test - input_mean) / input_std
  X_test = X_test |> device
  
  y_test = Array{Float32}(test[:, "torque_output"]) |> device

  # DataFrames are no longer needed after conversion to compact numeric arrays.
  train = nothing
  test = nothing
  data = nothing
  GC.gc()

  input_dim = size(X_train, 2)

  # Define the model
  model = Chain(
      Dense(input_dim, 7, sigmoid),
      Dense(7, 13, sigmoid),
      Dense(13, 3),
      Dense(3, 1)
  ) |> device

  # Define the loss function, which includes penalties to enforce physically correct behavior

  # Define the range of values for each independent variable
  speed_len = 8
  other_len = 5
  v_ego_range = range(1f0, stop=40f0, length=speed_len)
  lateral_acceleration_range = range(-4f0, stop=4f0, length=other_len)
  lateral_acceleration_range_hi = range(-3.95f0, stop=4.05f0, length=other_len)
  lateral_jerk_range = range(-5f0, stop=5f0, length=other_len)
  lateral_jerk_range_hi = range(-4.87f0, stop=5.13f0, length=other_len)
  lateral_error_range = range(-5f0, stop=5f0, length=other_len)
  lateral_error_range_hi = range(-4.87f0, stop=5.13f0, length=other_len)
  roll_range = range(-0.2f0, stop=0.2f0, length=other_len)
  roll_range_hi = range(-0.17f0, stop=0.23f0, length=other_len)
  roll_rate_range = range(-0.4f0, stop=0.4f0, length=other_len)
  roll_rate_range_hi = range(-0.35f0, stop=0.45f0, length=other_len)

  function prepare_test_grid(lat_jerk_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range, lateral_error_range, roll_rate_range)
    num_test_samples = size(v_ego_range, 1) * size(lateral_acceleration_range, 1) * size(lateral_jerk_range, 1) * size(roll_range, 1) * size(lateral_error_range, 1) * size(roll_rate_range, 1)
    out_grid = Matrix{Float32}(undef, 4 + 2 * size(t_list,2), num_test_samples)
    i = 1
    
    # Pre-allocate arrays for better performance
    lat_accels = Vector{Float32}(undef, length(t_list))
    rolls = Vector{Float32}(undef, length(t_list))
    grid_tmp = Vector{Float32}(undef, 4 + 2 * length(t_list))
    
    for la in lateral_acceleration_range
      for lj in lateral_jerk_range
        for le in lateral_error_range
          for roll in roll_range
            for roll_rate in roll_rate_range
              for v_ego in v_ego_range
                # Use in-place operations to avoid allocations
                @inbounds for (idx, t) in enumerate(t_list)
                  lat_accels[idx] = lat_jerk_func(la, lj, t)
                  rolls[idx] = roll + roll_rate * t
                end
                
                # Construct grid_tmp without intermediate allocations
                # 列顺序与 nnlc.py 运行时 nn_input 完全一致：v_ego, desired_lateral_accel, friction_input, roll, 时序lat, 时序roll
                grid_tmp[1] = v_ego
                grid_tmp[2] = la
                # friction_input 近似：lat_accel_friction_factor*(setpoint-measurement) + lat_jerk_friction_factor*lookahead_lateral_jerk
                # setpoint-measurement ≈ lateral_error(le)，lookahead_lateral_jerk ≈ lateral_jerk(lj)
                # 系数 0.7/0.4 与 latcontrol_torque_ext_base.py 中 lat_accel_friction_factor/lat_jerk_friction_factor 一致
                grid_tmp[3] = 0.7f0 * le + 0.4f0 * lj
                grid_tmp[4] = roll
                
                @inbounds for idx in 1:length(t_list)
                  grid_tmp[4 + idx] = lat_accels[idx]
                  grid_tmp[4 + length(t_list) + idx] = rolls[idx]
                end
                
                # Normalize and store in out_grid
                @inbounds @fastmath for j in 1:length(grid_tmp)
                  out_grid[j, i] = (grid_tmp[j] - input_mean[j]) / input_std[j]
                end
                
                i += 1
              end
            end
          end
        end
      end
    end
    return out_grid
  end

  # Define the linear jerk function for better performance - use oftype for type stability
  lj_func = (la, lj, t) -> la + oftype(la, lj * t)
  
  # Prepare all the grids
  grid = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range, lateral_error_range, roll_rate_range)
  grid_da = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range_hi, lateral_jerk_range, roll_range, lateral_error_range, roll_rate_range)
  grid_dj = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range_hi, roll_range, lateral_error_range, roll_rate_range)
  grid_de = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range, lateral_error_range_hi, roll_rate_range)
  grid_dg = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range_hi, lateral_error_range, roll_rate_range)
  grid_dg_rate = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range_hi, lateral_error_range, roll_rate_range_hi)
  grid_odd_neg = prepare_test_grid(lj_func, v_ego_range, -lateral_acceleration_range, -lateral_jerk_range, -roll_range, -lateral_error_range, -roll_rate_range)
  # prepare_test_grid iterates over every range; wrap the origin values so
  # scalar zero values do not trigger a MethodError before training starts.
  grid_origin = prepare_test_grid(
    lj_func, v_ego_range, [0f0], [0f0], [0f0], [0f0], [0f0]
  )

  # Cache for physical constraint losses (recomputed every N epochs)
  cached_constraint_loss = Ref(0f0)
  cached_constraint_epoch = Ref(0)
  constraint_eval_interval = 10

  function physical_constraint_losses(x, y_pred, λ_monotonic::Float32, λ_odd::Float32, λ_origin::Float32) :: Float32
      if λ_monotonic ≈ 0f0 && λ_odd ≈ 0f0 && λ_origin ≈ 0f0
          return 0f0
      end

      # Only recompute every N epochs and cache the result
      if epoch % constraint_eval_interval != 0 || cached_constraint_epoch[] == epoch
        return cached_constraint_loss[]
      end
      
      monotonicity_loss = 0f0
      odd_loss = 0f0
      origin_loss = 0f0

      model_grid = model(grid)
      
      if λ_monotonic ≉ 0f0
          model_da = model(grid_da)
          model_dj = model(grid_dj)
          model_de = model(grid_de)
          model_dg = model(grid_dg)
          model_dgr = model(grid_dg_rate)

          # d(output)/d(lat accel/jerk) and d(output)/d(error) should be positive,
          # d(output)/d(roll) and d(output)/d(roll_rate) should be negative
          @fastmath monotonicity_loss = sum(abs2, max.(0f0, (model_da .- model_grid) .* -1f0)) +
                              sum(abs2, max.(0f0, (model_dj .- model_grid) .* -1f0)) +
                              sum(abs2, max.(0f0, (model_de .- model_grid) .* -1f0)) +
                              sum(abs2, max.(0f0, model_dg .- model_grid)) +
                              sum(abs2, max.(0f0, model_dgr .- model_grid))  
      end
      
      if λ_odd ≉ 0f0
          model_odd_neg = model(grid_odd_neg)
          @fastmath odd_loss = sum(abs2, model_grid .+ model_odd_neg)
      end

      if λ_origin ≉ 0f0
          @fastmath origin_loss = sum(abs2, model(grid_origin))
      end

      result = @fastmath λ_monotonic * monotonicity_loss + λ_odd * odd_loss + λ_origin * origin_loss
      cached_constraint_loss[] = result
      cached_constraint_epoch[] = epoch
      return result
  end

  function model_with_params(model, params)
      temp_model = deepcopy(model)
      Flux.loadparams!(temp_model, params)
      return temp_model
  end

  function get_scaling_vector(x, low, high)
      speed = @view x[:, 1]
      min_speed, max_speed = extrema(speed)
      speed_span = max_speed - min_speed
      if speed_span <= eps(Float32)
        # A constant-speed batch has no meaningful speed normalization. Use
        # the midpoint weight instead of creating NaN through division by 0.
        return fill((low + high) / 2f0, length(speed))
      end
      @fastmath normalized_speed = @. (speed - min_speed) / speed_span # normalize to [0, 1]
      @fastmath scaling_vector = @. low + (high - low) * normalized_speed # scale to [low, high]
      return scaling_vector
  end

  function custom_mse(y_true, y_pred, scaling_vector)
      @fastmath residuals = y_true .- y_pred
      @fastmath squared_errors = residuals .* residuals
      @fastmath weighted_squared_errors = scaling_vector .* squared_errors
      return sum(weighted_squared_errors) / oftype(weighted_squared_errors[1], length(y_true))
  end

  function combined_loss(x, y_true, y_pred, model, λ::Float32, λ_monotonic::Float32, λ_odd::Float32, λ_origin::Float32, low::Float32, high::Float32)
      mse = 0f0
      if low ≉ high
        scaling_vector = get_scaling_vector(x, low, high)
        mse = custom_mse(y_true, y_pred, scaling_vector)
      else
        mse = Flux.Losses.mse(y_true, y_pred)
      end
      
      l2 = λ ≈ 0f0 ? 0f0 : λ * sum(p -> sum(abs2, p), params(model))
      physical_constraints = physical_constraint_losses(x, y_pred, λ_monotonic, λ_odd, λ_origin)
      
      @fastmath return mse + l2 + physical_constraints
  end

  loss(x, y, model, λ::Float32=0f0, λ_monotonic::Float32=0f0, λ_odd::Float32=0f0, λ_origin::Float32=0f0, low::Float32=1f0, high::Float32=1f0) = combined_loss(x, y', model(x), model, λ, λ_monotonic, λ_odd, λ_origin, low, high)

  # pick an optimizer
  # opt = Flux.ADAM(0.001)
  # opt = Flux.Nesterov()
  opt = CustomAdaGrad(0.01f0, 1.0f-10)
  state_tree = Optimisers.setup(opt, model)

  # Train with a fixed-size batch to bound reverse-mode autodiff memory.
  tol = log10(size(X_train, 1)) > 7 ? 1f-4 : 1f-6
  Δtol = log10(size(X_train, 1)) > 7 ? 1f-4 : 5f-5
  logstep = device == gpu ? 50 : log10(size(X_train, 1)) > 7 ? 3 : 10
  logstepgrowth = 1
  logstepfloat = Float32(logstep)
  Δloss = Inf32
  ΔΔloss = Inf32
  Δloss_last = 0f0
  loss_last = Inf32
  loss_cur = 0f0
  stall_check_count = device == gpu ? 50000 : 15
  stall_count = 0
  epoch = 1
  ilog = logstep + 1
  epoch_max = device == gpu ? 10000 : log10(size(X_train, 1)) > 6 ? 150 : 1000
  epoch_min = 25
  if requested_batch_size <= 0
    error("batch size must be positive")
  end
  batch_size = min(requested_batch_size, size(y_train, 1))

  println(out_streams, size(X_train))
  println(out_streams, size(y_train))
  println(out_streams, "Batch size: $batch_size; symmetric augmentation is generated per batch")

  if use_existing_model
      old_model = "$model_path.bson"
      println(out_streams, "Loading old model, $old_model")
      @load old_model model
  else

    train_data_loader = DataLoader((X_train', y_train), batchsize=batch_size, shuffle=true)
    symmetry_signs = ones(Float32, input_dim)
    symmetry_signs[2:end] .= -1f0
    symmetry_signs = reshape(symmetry_signs, :, 1) |> device

    train_batch! = function(x, y, lambda, monotonic_lambda, odd_lambda, origin_lambda)
      batch_loss = 0f0
      gs = Flux.gradient(model) do current_model
        batch_loss = loss(x, y, current_model, lambda, monotonic_lambda, odd_lambda, origin_lambda)
      end
      state_tree, model = Optimisers.update!(state_tree, model, gs[1])
      return batch_loss
    end

    grid = grid |> device
    grid_da = grid_da |> device
    grid_dj = grid_dj |> device
    grid_de = grid_de |> device
    grid_dg = grid_dg |> device
    grid_dg_rate = grid_dg_rate |> device
    grid_odd_neg = grid_odd_neg |> device
    grid_origin = grid_origin |> device
    
    λmax = 0.000f0
    λ_monotonicmax = 0.0015f0
    λ_oddmax = 0.000001f0
    λ_originmax = 0.0001f0

    λ_start_epoch_fraction = 0.25f0
    λ_monotonic_start_epoch_fraction = 0.6f0
    λ_odd_start_epoch_fraction = 0.5f0
    λ_origin_start_epoch_fraction = 0.7f0

    start_time = now()
    last_log_time = start_time - Dates.Millisecond(40000)
    ptime(t) = Dates.format(t, "HH:MM:SS")
    losses = Float32[]
    lambdas = NTuple{4, Float32}[]
    ilog = 0
    epoch_last = 1

    while epoch < epoch_min || (epoch < epoch_max) # && (abs(Δloss) > tol || abs(ΔΔloss) > Δtol)
        # determine λ_monotonic and λ_odd. They stay at 0 until 25% of the way through the training, then increase linearly to their max values by 75% of the way through the training
        λ = λmax * min(1f0, max(0f0, epoch - epoch_max * λ_start_epoch_fraction) / (epoch_max * 0.7f0))
        λ_monotonic = λ_monotonicmax * min(1f0, max(0f0, epoch - epoch_max * λ_monotonic_start_epoch_fraction) / (epoch_max * 0.3f0))
        λ_odd = λ_oddmax * min(1f0, max(0f0, epoch - epoch_max * λ_odd_start_epoch_fraction) / (epoch_max * 0.4f0))
        λ_origin = λ_originmax * min(1f0, max(0f0, epoch - epoch_max * λ_origin_start_epoch_fraction) / (epoch_max * 0.2f0))
        epoch_loss_sum = 0f0
        epoch_sample_count = 0
        for (x, y) in train_data_loader
          epoch_loss_sum += Float32(train_batch!(x, y, λ, λ_monotonic, λ_odd, λ_origin)) * length(y)
          epoch_sample_count += length(y)
        end

        # Preserve the original sign-symmetry augmentation without retaining a
        # second copy of the complete training set.
        for (x, y) in train_data_loader
          epoch_loss_sum += Float32(train_batch!(x .* symmetry_signs, -y, λ, λ_monotonic, λ_odd, λ_origin)) * length(y)
          epoch_sample_count += length(y)
        end
        l = epoch_loss_sum / epoch_sample_count

        push!(losses, l)
        push!(lambdas, (λ, λ_monotonic, λ_odd, λ_origin))
      
        t = now()
        if (t - last_log_time) > Dates.Millisecond(10000) || epoch % logstep == 0 || epoch >= epoch_max
            loss_cur = l
            if isfinite(loss_last)
                Δloss = loss_cur - loss_last
                ΔΔloss = Δloss - Δloss_last
                if Δloss < -tol
                    stall_count = 0
                else
                    stall_count += 1
                end
            else
                Δloss = 0f0
                ΔΔloss = 0f0
                stall_count = 0
            end
            loss_last = loss_cur
            Δloss_last = Δloss
            if abs(Δloss) > tol || abs(ΔΔloss) > Δtol
                cur_time = Dates.format(now(), "HH:MM:SS")
                # predict time remaining
                time_str = "estimating remaining time..."
                if epoch / epoch_max > 0.05
                    elapsed = (t - last_log_time).value / 1000 # Milliseconds to seconds
                    epochs = epoch - epoch_last
                    epoch_remaining = epoch_max - epoch
                    recent_rate = epochs / elapsed
                    epoch_remaining_time = Dates.Millisecond(1000 * round(epoch_remaining / recent_rate))
                    total_time = epoch_remaining_time + Dates.Millisecond(1000 * round((t - start_time).value / 1000))
                    epoch_remaining_time_str = Dates.canonicalize(Dates.CompoundPeriod(epoch_remaining_time))
                    epoch_total_time_str = Dates.canonicalize(Dates.CompoundPeriod(total_time))
                    time_str = "$epoch_remaining_time_str remaining of $epoch_total_time_str total"
                end
                println(out_streams, f"{cur_time} Epoch {epoch:3d} of {epoch_max} ({time_str}); Loss: {loss_cur:.6f}, ΔLoss: {Δloss:.7f}, ΔΔLoss: {ΔΔloss:.9f}, λ: {λ:.6G}, λ_monotonic: {λ_monotonic:.6G}, λ_odd: {λ_odd:.6G}, λ_origin: {λ_origin:.6G}")
            end
            if epoch >= epoch_min && stall_count >= stall_check_count
                println(out_streams, "Stopped after $stall_count consecutive checks without loss improvement at epoch $epoch, loss $loss_cur")
                break
            end
            logstepfloat *= logstepgrowth
            logstep = round(Int, logstepfloat)
            last_log_time = t
            epoch_last = epoch
        end
        epoch += 1
        ilog += 1
    end
    loss_cur = loss(X_train', y_train, model)
    Δloss = loss_cur - loss_last
    cur_time = Dates.format(now(), "HH:MM:SS")
    println(out_streams, f"round 1 {cur_time} Epoch: {epoch:3d} (of {epoch_max}; Loss: {loss_cur:.6f}, ΔLoss: {Δloss:.7f}, ΔΔLoss: {ΔΔloss:.9f}")

    # save the model for easy loading back into Flux.jl
      # bring data back to cpu
    if device == gpu
      X_train = cpu(X_train)
      y_train = cpu(y_train)
      X_test = cpu(X_test)
      y_test = cpu(y_test)
      model = cpu(model)
      grid = cpu(grid)
      grid_da = cpu(grid_da)
      grid_dj = cpu(grid_dj)
      grid_de = cpu(grid_de)
      grid_dg = cpu(grid_dg)
      grid_dg_rate = cpu(grid_dg_rate)
      grid_odd_neg = cpu(grid_odd_neg)
      grid_origin = cpu(grid_origin)
    end

    @save "$model_path.bson" model

    # Create and save plot of training

    # x value is just the epoch number
    x = 1:length(losses)

    mean_loss = mean(losses)
    std_loss = std(losses)

    # Plot the loss values with a black line on the left axis
    p = plot(x, losses, label="Loss", color=:black, ylabel="Loss", xlabel="Epoch", yscale=:log10, legend=:topleft)

    # Iterate through each lamda and plot it with different colors
    colors = [:red, :blue, :green, :orange]
    loss_names = ["λ_l2_regularization", "λ_monotonic", "λ_odd", "λ_origin"]
    for i in 1:4
        lambda = [l[i] for l in lambdas]
        l_max = maximum(lambda)
        if l_max == 0f0
            continue
        end
        normalized_lambda = lambda ./ l_max
        plot!(twinx(), normalized_lambda, label=loss_names[i], color=colors[i], ylabel="Normalized Lamda",xticks=:none, legend=:bottomright)
    end
    savefig(p, "$model_path.training.png")
    
    println(out_streams, "Finished after $epoch epochs, Loss: $loss_cur, ΔLoss: $Δloss, Test loss: $(loss(X_test', y_test, model))")

  end

  function feedforward_function(input_data; zero_bias=false)
    # Scale the input data using the stored mean and standard deviation values
    input_data_scaled = (input_data .- input_mean) ./ input_std
    if zero_bias
      eval_model = deepcopy(model)
      for layer in eval_model.layers
        if layer isa Dense
          Flux.params(layer)[2] .= zeros(Float32, size(Flux.params(layer)[2]))
        end
      end
      steer_command = eval_model(input_data_scaled')
      return steer_command[1]
    else
      steer_command = model(input_data_scaled')
      return steer_command[1]
    end
  end

  function evaluate_manually(model, x; zero_bias=false)
    for layer in model.layers
      W, b = params(layer)
      W = W'
      b = b'
      if zero_bias
        b = zeros(Float32, size(b))
      end
      if layer.σ == σ
        x = σ.(x * W .+ b)
      elseif layer.σ == identity
        x = identity.(x * W .+ b)
      elseif layer.σ == tanh
        x = tanh.(x * W .+ b)
      elseif layer.σ == leakyrelu
        x = leakyrelu.(x * W .+ b)
      else
        try
          x = layer.σ.(x * W .+ b)
        catch e
          println(out_streams, "Unsupported activation function: $(layer.σ)")
          rethrow(e)
        end
      end
    end
    return x[1]
  end

  function feedforward_function_manual(input_data; zero_bias=false)
    # Scale the input data using the stored mean and standard deviation values
    input_data_scaled = (input_data .- input_mean) ./ input_std
    steer_command = evaluate_manually(model, input_data_scaled, zero_bias=zero_bias)
    return steer_command
  end

  function test_evaluate_manually(model; zero_bias=false)
    vego_range = 0f0:20f0:40f0
    lataccel_range = -4f0:4f0:4f0
    latjerk_range = -4f0:4f0:4f0
    roll_range = -0.2f0:0.2f0:0.2f0
    println(out_streams, "Testing manual model evaluation (as performed in OpenPilot)...")
    println(out_streams, "Testing with zero bias: $zero_bias")
    test_dict = Dict()
    for vego in vego_range
      for lataccel in lataccel_range
        for latjerk in latjerk_range
          for roll in roll_range
            lat_accels = [lataccel + t * latjerk for t in t_list]
            rolls = [roll for t in t_list]
            input_data = [vego lataccel latjerk roll]
            x = hcat(input_data, lat_accels, rolls)
            xstr = "[" * join(x, ",") * "]"
            result_model = feedforward_function(x, zero_bias=zero_bias)  # Model evaluation
            result_manual = feedforward_function_manual(x, zero_bias=zero_bias)  # Manual evaluation
            test_dict[xstr] = result_model
            if result_model ≉ result_manual
              println(out_streams, "Mismatch at input: $x")
              println(out_streams, "Model: $result_model, Manual: $result_manual")
              return false
            end
          end
        end
      end
    end

    println(out_streams, "Test passed: All outputs match!")
    return test_dict
  end

  test_dict_zero_bias = test_evaluate_manually(model, zero_bias=true)
  test_dict = test_evaluate_manually(model)

  current_date_and_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

  model_test_loss = loss(X_test', y_test, model)

  # save model to json for Python import
  function export_model_params_to_json(model::Chain, input_mean::Matrix{Float32}, input_std::Matrix{Float32}, filename::String, current_date_and_time, model_test_loss, input_vars)
      W, b = params(model.layers[1])
      input_size = size(W, 2)
      output_size = size(params(model.layers[end])[1], 1)
      params_dict = Dict{String, Any}("input_size" => input_size, "output_size" => output_size, "layers" => [], "input_mean" => input_mean, "input_std" => input_std, "current_date_and_time" => current_date_and_time, "model_test_loss" => model_test_loss, "input_vars" => input_vars)

      for (idx, layer) in enumerate(model.layers)
          if isa(layer, Dense)
              W, b = params(layer)
              params_dict["layers"] = push!(params_dict["layers"], Dict(
                  "dense_$(idx)_W" => Array{Float32}(W'),
                  "dense_$(idx)_b" => Array{Float32}(b'),
                  "activation" => string(layer.σ)
              ))
          end
      end

      open(filename, "w") do f
          write(f, JSON.json(params_dict))
      end
  end

  export_model_params_to_json(model, Matrix{Float32}(input_mean), Matrix{Float32}(input_std), "$model_path.json", current_date_and_time, model_test_loss, feature_names)


  # Evaluate the model on the test set 
  test_loss = loss(X_test', y_test, model)
  println(out_streams, "Test loss (MSE): ", test_loss)

  return (model=model, input_mean=input_mean, input_std=input_std, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, test_loss=test_loss)

end

function test_plot_model(model::Flux.Chain, plot_path::String, X_train::Matrix{Float32}, y_train::Vector{Float32}, X_test::Matrix{Float32}, y_test::Vector{Float32}, input_mean::Matrix{Float32}, input_std::Matrix{Float32}, x_var_names::String, out_streams, test_loss::Float32)

  car_name = Base.basename(plot_path)

  function feedforward_function(input_data)
    # Scale the input data using the stored mean and standard deviation values
    @fastmath input_data_scaled = @. (input_data - input_mean) / input_std
    steer_command = model(input_data_scaled')
    return steer_command[1]
  end

  max_abs_lat_jerk = 0.2f0
  max_abs_roll = 0.03f0

  # Create a function to filter the dataset based on speed
  function filter_data_by_speed(Xi, yi, speed, tolerance; no_jerk=false, no_roll=false, shuffle_data=true)
    indices = findall(@. abs(Xi[:, 1] - speed) < tolerance)
    X, y = Xi[indices, :], yi[indices]
    if no_jerk
      indices = findall(@. abs(X[:, 3]) < max_abs_lat_jerk)
      X, y = X[indices, :], y[indices]
    end
    if no_roll
      indices = findall(@. abs(X[:, 4]) < max_abs_roll)
      X, y = X[indices, :], y[indices]
    end
    if shuffle_data
      indices = shuffle(1:size(X, 1))
      X, y = X[indices, :], y[indices]
    end
    return X, y
  end

  @fastmath X_train_rescaled = @. X_train * input_std + input_mean
  @fastmath X_test_rescaled = @. X_test * input_std + input_mean

  scatter_points_desired = 1000

  # Iterate over the speed range and create a plot for each speed
  # first w.r.t. lateral jerk
  speed_step = 6f0
  speed_range = 3f0:speed_step:35f0
  lateral_acceleration_range = range(-4.0f0, 4.0f0, length=100)

  marker_alpha = 0.1f0
  test_alpha = 0.25f0
  train_color = :black
  test_color = :cadetblue
  cpalette = :Dark2_5
  line_width = 3

  plot_col_num = 1
  p = plot(layout = (size(collect(speed_range), 1), 3), legend=:bottomright, size=(2300, 2300), margin=8mm)

  # Pre-allocate arrays for better performance
  lat_accels = Vector{Float32}(undef, length(t_list))
  rolls = Vector{Float32}(undef, length(t_list))

  # Some speed bins can legitimately contain no samples.  Keep the empty
  # bin visible in the plot, but do not index an empty matrix at 1:end.
  function scatter_filtered!(plt, X_filtered, y_filtered, target_count; kwargs...)
    n = size(X_filtered, 1)
    if n > 0
      step = max(1, round(Int, n / target_count))
      indices = 1:step:n
      scatter!(plt, X_filtered[indices, 2], y_filtered[indices]; kwargs...)
    end
  end

  # Iterate over the speed range and create a plot for each speed
  for (si, speed) in enumerate(speed_range)
    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_roll=true)
    scatter_filtered!(p[si,plot_col_num], X_train_filtered, y_train_filtered, scatter_points_desired / 2,
      label="Training Data", markersize=2, markercolor=train_color, markeralpha=marker_alpha,
      xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_roll=true)
    scatter_filtered!(p[si,plot_col_num], X_test_filtered, y_test_filtered, scatter_points_desired,
      label="Test Data", markersize=2, markercolor=test_color, markeralpha=test_alpha,
      xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    vline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")
    
    # Plot "sustained error", so that the amount of lat accel error propogates backwards and forwards through time
    # (i.e. you're entering/exiting a turn at a constant rate that the car isn't keeping up with).
    # Here, we set a lateral jerk value and use that to compute the lateral acceleration at each time step.
    ci = 1
    for lj in [-1f0, -0.25f0, 0f0, 0.25f0, 1f0]
      x_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      y_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      
      for (i, la) in enumerate(lateral_acceleration_range)
          # Fill pre-allocated arrays
          fill!(lat_accels, la)
          fill!(rolls, 0f0)
          
          # Create input data
          input_data = [speed la lj 0f0]
          input_data = hcat(input_data, lat_accels', rolls')
          
          # Get model prediction
          steer_command = feedforward_function(input_data)
          x_model[i] = la
          y_model[i] = steer_command
      end
      
      plot!(p[si,plot_col_num], x_model, y_model, label="err = $lj", linewidth=line_width, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"{car_name}\nLateral acceleration/jerk error response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |roll| < {max_abs_roll:.2G}")
    else
      title!(p[si,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |roll| < {max_abs_roll:.2G}")
    end
    xlabel!(p[si,plot_col_num], "lateral acceleration (m/s²; [+] = right turn)")
    ylabel!(p[si,plot_col_num], "steer command\n([+] = pushing right)")
  end

  plot_col_num += 1

  for (si, speed) in enumerate(speed_range)

    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_roll=true)
    scatter_filtered!(p[si,plot_col_num], X_train_filtered, y_train_filtered, scatter_points_desired / 2,
      label="Training Data", markersize=2, markercolor=train_color, markeralpha=marker_alpha,
      xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_roll=true)
    scatter_filtered!(p[si,plot_col_num], X_test_filtered, y_test_filtered, scatter_points_desired,
      label="Test Data", markersize=2, markercolor=test_color, markeralpha=test_alpha,
      xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    vline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")
    
    # Plot "sustained error", so that the amount of lat accel error propogates backwards and forwards through time
    ci = 1
    for lj in -1f0:0.5f0:1f0
      x_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      y_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      
      for (i, la) in enumerate(lateral_acceleration_range)
          # Calculate lat_accels with lateral jerk
          @inbounds for (idx, t) in enumerate(t_list)
              lat_accels[idx] = la + lj * t
              rolls[idx] = 0f0
          end
          
          input_data = [speed la 0f0 0f0]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = la
          y_model[i] = steer_command
      end
      
      plot!(p[si,plot_col_num], x_model, y_model, label="lat. jerk = $lj", linewidth=line_width, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"{x_var_names}\nLateral jerk response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph @ |roll| < {max_abs_roll:.2G}")
    else
      title!(p[si,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |roll| < {max_abs_roll:.2G}")
    end
    xlabel!(p[si,plot_col_num], "lateral acceleration (m/s²; [+] = right turn)")
    ylabel!(p[si,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # Iterate over the speed range and create a plot for each speed

  plot_col_num += 1

  # now w.r.t. lateral gravitational acceleration

  for (si, speed) in enumerate(speed_range)

    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_jerk=true)
    scatter_filtered!(p[si,plot_col_num], X_train_filtered, y_train_filtered, scatter_points_desired / 2,
      label="Training Data", markersize=2, markercolor=train_color, markeralpha=marker_alpha,
      xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_jerk=true)
    scatter_filtered!(p[si,plot_col_num], X_test_filtered, y_test_filtered, scatter_points_desired,
      label="Test Data", markersize=2, markercolor=test_color, markeralpha=test_alpha,
      xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    vline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for gla in -0.1f0:0.05f0:0.1f0
      x_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      y_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      
      for (i, la) in enumerate(lateral_acceleration_range)
          fill!(lat_accels, la)
          fill!(rolls, gla)
          
          input_data = [speed la 0f0 gla]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = la
          y_model[i] = steer_command
      end
      
      plot!(p[si,plot_col_num], x_model, y_model, label="roll = $gla", linewidth=line_width, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"Model test loss: {test_loss:.2G}\nRoll compensation [+] = leaning to the right\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |lat jerk| < {max_abs_lat_jerk:.2G}")
    else
      title!(p[si,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |lat jerk| < {max_abs_lat_jerk:.2G}")
    end
    xlabel!(p[si,plot_col_num], "lateral acceleration (m/s²; [+] = right turn)")
    ylabel!(p[si,plot_col_num], "steer command\n([+] = pushing right)")
  end


  # Display the plot
  savefig(p, "$plot_path/$car_name-a.png")
  savefig(p, "$plot_path/$car_name-a.pdf")
  # display(p)

  # Now plot model response to error, lateral jerk, and roll.
  # Each row will be different speeds as above. The left column will show the model output
  # as a function of error (instantaneous lateral jerk) at different lateral accels (as different lines). 
  # The second column will show model output as a function of lateral jerk (i.e. past/future lateral accels).
  # The third column will be as a function of *constant* roll (all past/future the same), and the 
  # fourth column will be as a function of dynamic roll (past/future changing linearly).

  plot_col_num = 1
  lateral_accel_range = -2f0:1f0:2f0
  lateral_jerk_range = -3f0:0.1f0:3f0
  roll_range = -0.2f0:0.01f0:0.2f0

  p = plot(layout = (size(collect(speed_range), 1), 4), legend=:bottomright, size=(2500, 2300), margin=8mm)

  # first w.r.t. error
  for (plot_row_num, speed) in enumerate(speed_range)
    vline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for la in lateral_accel_range
      x_model = Vector{Float32}(undef, length(lateral_jerk_range))
      y_model = Vector{Float32}(undef, length(lateral_jerk_range))
      
      # Pre-fill lat_accels with constant values
      fill!(lat_accels, la)
      fill!(rolls, 0f0)
      
      for (i, lj) in enumerate(lateral_jerk_range)
          input_data = [speed la lj 0f0]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          x_model[i] = lj
          y_model[i] = steer_command
      end
      
      plot!(p[plot_row_num,plot_col_num], x_model, y_model, label="lat accel = $la", linewidth=line_width, xlims=(-3.0, 3.0), ylims=(-1.5,1.5), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if plot_row_num == 1
      title!(p[1,plot_col_num], f"Error response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    else
      title!(p[plot_row_num,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    end
    xlabel!(p[plot_row_num,plot_col_num], "lateral acceleration/jerk error (m/s²; [+] = correcting to the right)")
    ylabel!(p[plot_row_num,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # now plot model response to lateral jerk
  plot_col_num += 1
  for (plot_row_num, speed) in enumerate(speed_range)
    vline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for la in lateral_accel_range
      x_model = Vector{Float32}(undef, length(lateral_jerk_range))
      y_model = Vector{Float32}(undef, length(lateral_jerk_range))
      
      for (i, lj) in enumerate(lateral_jerk_range)
          # Calculate lat_accels with lateral jerk
          @inbounds for (idx, t) in enumerate(t_list)
              lat_accels[idx] = la + lj * t
              rolls[idx] = 0f0
          end
          
          input_data = [speed la 0f0 0f0]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = lj
          y_model[i] = steer_command
      end
      
      plot!(p[plot_row_num,plot_col_num], x_model, y_model, label="lat accel = $la", linewidth=line_width, xlims=(-3.0, 3.0), ylims=(-1.5,1.5), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if plot_row_num == 1
      title!(p[1,plot_col_num], f"{car_name}\nLateral jerk response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    else
      title!(p[plot_row_num,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    end
    xlabel!(p[plot_row_num,plot_col_num], "lateral jerk (m/s²; [+] = wheel moving to the right)")
    ylabel!(p[plot_row_num,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # now plot model response to constant roll
  plot_col_num += 1
  for (plot_row_num, speed) in enumerate(speed_range)
    vline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for la in lateral_accel_range
      x_model = Vector{Float32}(undef, length(roll_range))
      y_model = Vector{Float32}(undef, length(roll_range))
      
      # Pre-fill lat_accels with constant values
      fill!(lat_accels, la)
      
      for (i, ro) in enumerate(roll_range)
          # Fill rolls with constant value
          fill!(rolls, ro)
          
          input_data = [speed la 0f0 ro]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = ro
          y_model[i] = steer_command
      end
      
      plot!(p[plot_row_num,plot_col_num], x_model, y_model, label="lat accel = $la", linewidth=line_width, xlims=(-0.2, 0.2), ylims=(-1.5,1.5), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if plot_row_num == 1
      title!(p[1,plot_col_num], f"Roll response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    else
      title!(p[plot_row_num,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    end
    xlabel!(p[plot_row_num,plot_col_num], "Road roll (radians; [+] = leaning right)")
    ylabel!(p[plot_row_num,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # now plot model response to roll rate (same as previous, but
  # lateral acceleration lines are replaced with lines with different roll rates)
  roll_rate_range = -0.4f0:0.2f0:0.4f0
  plot_col_num += 1
  for (plot_row_num, speed) in enumerate(speed_range)
    vline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for rr in roll_rate_range
      x_model = Vector{Float32}(undef, length(roll_range))
      y_model = Vector{Float32}(undef, length(roll_range))
      
      # Pre-fill lat_accels with zeros
      fill!(lat_accels, 0f0)
      
      for (i, ro) in enumerate(roll_range)
          # Calculate rolls with roll rate
          @inbounds for (idx, t) in enumerate(t_list)
              rolls[idx] = ro + t * rr
          end
          
          input_data = [speed 0f0 0f0 ro]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = ro
          y_model[i] = steer_command
      end
      
      plot!(p[plot_row_num,plot_col_num], x_model, y_model, label="roll rate = $rr", linewidth=line_width, xlims=(-0.2, 0.2), ylims=(-1.5,1.5), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if plot_row_num == 1
      title!(p[1,plot_col_num], f"Roll rate [rad/s] response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    else
      title!(p[plot_row_num,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    end
    xlabel!(p[plot_row_num,plot_col_num], "Road roll (radians; [+] = leaning right)")
    ylabel!(p[plot_row_num,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # Display the plot
  savefig(p, "$plot_path/$car_name-b.png")
  savefig(p, "$plot_path/$car_name-b.pdf")
end

function multiline_string(strings::Vector{String}, n::Int; prefix="")::String
  # Pre-allocate for better performance
  lines = Vector{String}()
  sizehint!(lines, div(length(strings), 5) + 1)  # Estimate number of lines needed
  
  current_line = prefix
  
  for str in strings
      # If the current string fits on the current line, add it to the line
      if length(current_line) + length(str) + 2 <= n
          if isempty(current_line)
              current_line = str
          else
              current_line *= ", $str"
          end
      # Otherwise, start a new line with the current string
      else
          push!(lines, current_line)
          current_line = str
      end
  end
  
  # Add the last line to the list
  if !isempty(current_line)
      push!(lines, current_line)
  end
  
  # Join the lines with newline characters
  return join(lines, ",\n")
end

function create_model(in_file, out_dir_base; force_cpu::Bool=true, requested_batch_size::Int=16384)
  carname = replace(Base.basename(in_file), ".csv" => "")
  outdir = create_folder_with_iterator(out_dir_base, carname, make_new=true)
  logfile = open(outdir * "/$(carname)_log.txt", "a")  # Open log file in append mode
  out_streams = TeeStream(stdout, logfile)  # Create a Tee output stream
  preprocess_infile = replace(in_file, ".csv" => "_balanced.csv")
  use_existing_input = false
  if isfile(preprocess_infile) # && stat(in_file).mtime < stat(preprocess_infile).mtime
      #use_existing_input = true
      # return
  end
  
  data = load_data(in_file, use_existing_input, outdir, out_streams)

  feature_names = model_feature_names(data)
  model_file = "$outdir/$carname.bson"
  use_existing_input = false
  println(out_streams, "Model file: $model_file")
  if isfile(model_file) && stat(in_file).mtime < stat(model_file).mtime
      use_existing_input = true
      println(out_streams, "Using existing model file: $model_file")
      # return
  end

  model, input_mean, input_std, X_train, y_train, X_test, y_test, test_loss = train_model(outdir, use_existing_input, data, out_streams; force_cpu=force_cpu, requested_batch_size=requested_batch_size)
  
  test_plot_model(model, outdir, X_train, y_train, X_test, y_test, input_mean, input_std, multiline_string(feature_names, 60, prefix="Model input: "), out_streams, test_loss)
  close(logfile)
end

function main(in_dir; force_cpu::Bool=true, requested_batch_size::Int=16384)
  # Process only regular CSV files, excluding old balanced-data artifacts.
  csv_files = filter(
    file -> isfile(joinpath(in_dir, file)) && endswith(lowercase(file), ".csv") && !endswith(lowercase(file), "_balanced.csv"),
    readdir(in_dir),
  )

  results_dir = joinpath(in_dir, "training_results")
  mkpath(results_dir)

  # Process each file
  for in_file in csv_files
      println("Processing $in_file")
      create_model(joinpath(in_dir, in_file), results_dir; force_cpu=force_cpu, requested_batch_size=requested_batch_size)
  end
end

# Accept data directory as command-line argument, or default to ~/Downloads/rlogs/output/GENESIS
# GPU training is intentionally unsupported by this trainer.  Keep accepting
# --cpu for compatibility with existing scripts, but always select CPU.
force_cpu = true
batch_size_arg = findfirst(a -> startswith(a, "--batch-size="), ARGS)
batch_size_flag = findfirst(a -> a == "--batch-size", ARGS)
requested_batch_size = 16384
if batch_size_arg !== nothing
  requested_batch_size = try
    parse(Int, split(ARGS[batch_size_arg], "=", limit=2)[2])
  catch
    error("--batch-size must be a positive integer")
  end
elseif batch_size_flag !== nothing
  if batch_size_flag == length(ARGS)
    error("--batch-size must be followed by a positive integer")
  end
  requested_batch_size = try
    parse(Int, ARGS[batch_size_flag + 1])
  catch
    error("--batch-size must be a positive integer")
  end
end
if requested_batch_size <= 0
  error("--batch-size must be a positive integer")
end

positional_args = [
  argument for (index, argument) in enumerate(ARGS)
  if !startswith(argument, "--") && !(batch_size_flag !== nothing && index == batch_size_flag + 1)
]

println("CPU mode enabled")

if length(positional_args) > 0
  main(positional_args[1]; force_cpu=force_cpu, requested_batch_size=requested_batch_size)
else
  home_dir = ENV["HOME"]
  main("$home_dir/Downloads/rlogs/output/GENESIS"; force_cpu=force_cpu, requested_batch_size=requested_batch_size)
end
