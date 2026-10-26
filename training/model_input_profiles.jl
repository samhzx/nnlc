const temporal_offsets_by_input_dim = Dict(
  18 => Float32[-0.3, -0.2, -0.1, 0.3, 0.6, 1.0, 1.5],
  23 => Float32[-2.0, -1.0, -0.3, -0.2, -0.1, 0.3, 0.6, 1.0, 1.5],
)

function model_temporal_offsets(input_dim::Int)
  haskey(temporal_offsets_by_input_dim, input_dim) || error("input dimension must be 18 or 23")
  return temporal_offsets_by_input_dim[input_dim]
end

function model_temporal_suffix(offset)
  tenths = round(Int, offset * 10)
  return "_t" * (tenths < 0 ? "m" : "p") * lpad(string(abs(tenths)), 2, '0')
end

function model_input_features(input_dim::Int)
  offsets = model_temporal_offsets(input_dim)
  current = ["v_ego", "desired_lateral_accel", "friction_input", "roll"]
  input_dim == 23 && push!(current, "a_ego")
  return vcat(
    current,
    ["desired_lateral_accel" * model_temporal_suffix(offset) for offset in offsets],
    ["roll" * model_temporal_suffix(offset) for offset in offsets],
  )
end

function ordered_model_features(input_columns, input_dim::Int)
  features = model_input_features(input_dim)
  missing_features = filter(column -> !(column in input_columns), features)
  isempty(missing_features) || error(
    "模型输入列不完整或顺序错误：缺少 $(join(missing_features, ", "))"
  )
  return features
end

function model_base_row(v_ego, desired, friction, roll, input_dim::Int)
  values = [v_ego, desired, friction, roll]
  input_dim == 23 && push!(values, 0f0)
  return permutedims(values)
end

function model_symmetry_signs(input_dim::Int)
  signs = ones(Float32, input_dim)
  signs[2:end] .= -1f0
  input_dim == 23 && (signs[5] = 1f0)
  return signs
end

function stream_profile(input_dim::Int)
  offsets = model_temporal_offsets(input_dim)
  model_features = model_input_features(input_dim)
  all_temporal_features = [
    feature * model_temporal_suffix(offset)
    for offset in offsets for feature in ["actual_lateral_accel", "desired_lateral_accel", "roll"]
  ]
  required_columns = vcat(
    ["timestamp", "v_ego", "a_ego", "steering_angle_deg", "steering_rate_deg", "steering_torque", "steering_pressed", "standstill", "desired_curvature", "curvature", "active", "lateral_control_type", "actual_lateral_accel", "desired_lateral_accel", "torque_output", "saturated", "roll", "lane_change_state", "friction_input", "route_id"],
    all_temporal_features,
  )
  return (
    offsets=offsets,
    model_features=model_features,
    all_temporal_features=all_temporal_features,
    required_columns=required_columns,
    base_feature_count=length(model_features) - 2 * length(offsets),
  )
end
