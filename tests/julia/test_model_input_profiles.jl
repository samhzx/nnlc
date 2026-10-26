using CSV
using DataFrames
using Test

include(joinpath(@__DIR__, "..", "..", "training", "latmodel_temporal.jl"))

const expected_features = Dict(
  18 => [
    "v_ego", "desired_lateral_accel", "friction_input", "roll",
    "desired_lateral_accel_tm03", "desired_lateral_accel_tm02", "desired_lateral_accel_tm01",
    "desired_lateral_accel_tp03", "desired_lateral_accel_tp06", "desired_lateral_accel_tp10", "desired_lateral_accel_tp15",
    "roll_tm03", "roll_tm02", "roll_tm01", "roll_tp03", "roll_tp06", "roll_tp10", "roll_tp15",
  ],
  23 => [
    "v_ego", "desired_lateral_accel", "friction_input", "roll", "a_ego",
    "desired_lateral_accel_tm20", "desired_lateral_accel_tm10", "desired_lateral_accel_tm03", "desired_lateral_accel_tm02", "desired_lateral_accel_tm01",
    "desired_lateral_accel_tp03", "desired_lateral_accel_tp06", "desired_lateral_accel_tp10", "desired_lateral_accel_tp15",
    "roll_tm20", "roll_tm10", "roll_tm03", "roll_tm02", "roll_tm01", "roll_tp03", "roll_tp06", "roll_tp10", "roll_tp15",
  ],
)

function smoke_csv(path::String, input_dim::Int)
  row_count = 1200
  data = DataFrame()
  data[!, :timestamp] = collect(0:row_count-1) ./ 100
  data[!, :v_ego] = fill(15.0, row_count)
  data[!, :a_ego] = collect(range(-1.0, 1.0, length=row_count))
  data[!, :steering_angle_deg] = zeros(row_count)
  data[!, :steering_rate_deg] = zeros(row_count)
  data[!, :steering_torque] = zeros(row_count)
  data[!, :steering_pressed] = falses(row_count)
  data[!, :standstill] = falses(row_count)
  data[!, :desired_curvature] = zeros(row_count)
  data[!, :curvature] = zeros(row_count)
  data[!, :active] = trues(row_count)
  data[!, :lateral_control_type] = fill("torque", row_count)
  data[!, :actual_lateral_accel] = zeros(row_count)
  desired = [0.5 * sin(index / 20) for index in 1:row_count]
  data[!, :desired_lateral_accel] = desired
  data[!, :torque_output] = desired .* 0.1
  data[!, :saturated] = falses(row_count)
  data[!, :roll] = zeros(row_count)
  data[!, :lane_change_state] = zeros(Int, row_count)
  data[!, :friction_input] = desired .* 0.2

  for offset in model_temporal_offsets(input_dim)
    suffix = model_temporal_suffix(offset)
    data[!, Symbol("actual_lateral_accel" * suffix)] = zeros(row_count)
    data[!, Symbol("desired_lateral_accel" * suffix)] = desired
    data[!, Symbol("roll" * suffix)] = zeros(row_count)
  end
  data[!, :route_id] = fill("single-route", row_count)

  # Use the extractor's offset-major ordering rather than model-input ordering.
  CSV.write(path, data)
  return path
end

@testset "NNLC model input profiles" begin
  mktempdir() do temp_dir
    for input_dim in (18, 23)
      expected = expected_features[input_dim]
      @test model_input_features(input_dim) == expected
      @test model_symmetry_signs(input_dim)[1] == 1f0
      @test model_symmetry_signs(input_dim)[2:4] == fill(-1f0, 3)
      @test stream_profile(input_dim).model_features == expected
      @test ordered_model_features(reverse(expected), input_dim) == expected
      @test_throws ErrorException ordered_model_features(expected[2:end], input_dim)

      signs = model_symmetry_signs(input_dim)
      if input_dim == 23
        @test signs[5] == 1f0
        @test signs[6:end] == fill(-1f0, 18)
        @test vec(model_base_row(10f0, 0.2f0, 0.1f0, 0.01f0, 23))[5] == 0f0
      else
        @test signs[5:end] == fill(-1f0, 14)
      end

      csv_path = smoke_csv(joinpath(temp_dir, "sample_$(input_dim)d.csv"), input_dim)
      standard = load_data(csv_path, false, temp_dir, devnull, input_dim)
      @test model_feature_names(standard) == expected

      streaming_train, streaming_test = load_data_streaming(csv_path, devnull, input_dim)
      @test model_feature_names(streaming_train) == expected
      @test model_feature_names(streaming_test) == expected
      if input_dim == 23
        @test streaming_train.a_ego != fill(0f0, nrow(streaming_train))
      end
    end
  end
end
