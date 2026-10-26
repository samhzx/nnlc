import nnlc_auto_train
from nnlc_tools.model_inputs import INPUT_PROFILES, input_variables, temporal_offsets


EXPECTED_23D_VARS = [
    "v_ego",
    "desired_lateral_accel",
    "friction_input",
    "roll",
    "a_ego",
    "desired_lateral_accel_tm20",
    "desired_lateral_accel_tm10",
    "desired_lateral_accel_tm03",
    "desired_lateral_accel_tm02",
    "desired_lateral_accel_tm01",
    "desired_lateral_accel_tp03",
    "desired_lateral_accel_tp06",
    "desired_lateral_accel_tp10",
    "desired_lateral_accel_tp15",
    "roll_tm20",
    "roll_tm10",
    "roll_tm03",
    "roll_tm02",
    "roll_tm01",
    "roll_tp03",
    "roll_tp06",
    "roll_tp10",
    "roll_tp15",
]


def make_model_params(input_dim):
    layer_sizes = (input_dim, 7, 13, 3, 1)
    layers = []
    for index, (input_size, output_size) in enumerate(
        zip(layer_sizes, layer_sizes[1:]), start=1
    ):
        layers.append(
            {
                "activation": "σ" if index < 3 else "identity",
                f"dense_{index}_W": [[0.001] * input_size for _ in range(output_size)],
                f"dense_{index}_b": [[0.0] for _ in range(output_size)],
            }
        )
    return {
        "input_size": input_dim,
        "output_size": 1,
        "input_mean": [[0.0] for _ in range(input_dim)],
        "input_std": [[1.0] for _ in range(input_dim)],
        "input_vars": list(input_variables(input_dim)),
        "layers": layers,
        "model_test_loss": 0.01,
    }


def test_profiles_match_the_runtime_18d_and_23d_feature_contracts():
    assert temporal_offsets(18) == (-0.3, -0.2, -0.1, 0.3, 0.6, 1.0, 1.5)
    assert temporal_offsets(23) == (-2.0, -1.0, -0.3, -0.2, -0.1, 0.3, 0.6, 1.0, 1.5)
    assert len(input_variables(18)) == 18
    assert list(input_variables(23)) == EXPECTED_23D_VARS
    assert INPUT_PROFILES[18]["suffix"] == "18d"
    assert INPUT_PROFILES[23]["suffix"] == "23d"


def test_model_validator_accepts_both_input_profiles():
    for input_dim in (18, 23):
        assert nnlc_auto_train._validate_model_structure(
            make_model_params(input_dim), input_dim=input_dim
        ) == []


def test_model_validator_rejects_wrong_dimension_order_and_weights():
    params = make_model_params(23)
    assert any(
        "input_size" in issue
        for issue in nnlc_auto_train._validate_model_structure(params, input_dim=18)
    )

    params = make_model_params(23)
    params["input_vars"][4], params["input_vars"][5] = (
        params["input_vars"][5],
        params["input_vars"][4],
    )
    assert any(
        "input_vars 顺序或名称错误" in issue
        for issue in nnlc_auto_train._validate_model_structure(params, input_dim=23)
    )

    params = make_model_params(23)
    params["layers"][0]["dense_1_W"] = [[0.0] * 18 for _ in range(7)]
    assert any(
        "dense_1_W 形状" in issue
        for issue in nnlc_auto_train._validate_model_structure(params, input_dim=23)
    )


def test_step_validate_routes_23d_model_and_plots_to_23d_validation_dir(
    tmp_path, monkeypatch
):
    import json
    from pathlib import Path

    model_path = tmp_path / "BYD_TEST_23d.json"
    model_path.write_text(json.dumps(make_model_params(23)), encoding="utf-8")
    train_csv = tmp_path / "interventions_pruned_23d.csv"
    train_csv.write_text("timestamp\n1\n", encoding="utf-8")
    captured = {}

    def fake_run_command(command, description, **kwargs):
        captured["command"] = command
        output_dir = Path(command[command.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("lat_accel_vs_torque.png", "torque_vs_speed.png"):
            (output_dir / name).write_bytes(b"plot")

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(nnlc_auto_train, "run_command", fake_run_command)

    assert nnlc_auto_train.step_validate(
        str(model_path), str(train_csv), str(tmp_path), "python", input_dim=23
    )
    assert captured["command"][captured["command"].index("-o") + 1].endswith(
        "validation_23d"
    )


def test_23d_visualizer_uses_model_input_means_and_writes_both_plots(tmp_path):
    import json

    import numpy as np
    import pandas as pd

    from nnlc_tools.visualize_model import (
        NNModel,
        plot_lat_accel_vs_torque,
        plot_torque_vs_speed,
    )

    model_json = tmp_path / "model_23d.json"
    model_json.write_text(json.dumps(make_model_params(23)), encoding="utf-8")
    model = NNModel(model_json)
    assert model.make_input_at_means(3).shape == (3, 23)
    model.forward = lambda inputs: np.zeros((len(inputs), 1))

    data = pd.DataFrame(
        {
            "v_ego": [5.0, 10.0, 20.0],
            "desired_lateral_accel": [-0.5, 0.0, 0.5],
            "actual_lateral_accel": [-0.4, 0.0, 0.4],
            "torque_output": [-0.1, 0.0, 0.1],
            "active": [True, True, True],
            "standstill": [False, False, False],
        }
    )
    lat_plot = tmp_path / "lat.png"
    speed_plot = tmp_path / "speed.png"
    plot_lat_accel_vs_torque(model, data, str(lat_plot))
    plot_torque_vs_speed(model, data, str(speed_plot))
    assert lat_plot.is_file()
    assert speed_plot.is_file()


def test_dimension_specific_file_names_and_commands(tmp_path, monkeypatch):
    captured = {}

    def fake_run_command(command, description, **kwargs):
        captured["command"] = command
        output = command[command.index("-o") + 1]
        from pathlib import Path

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            handle.write("header\nrow\n")

    monkeypatch.setattr(nnlc_auto_train, "run_command", fake_run_command)
    output = nnlc_auto_train.step_extract(
        str(tmp_path), str(tmp_path / "out"), "python", input_dim=23
    )
    assert output.endswith("lateral_data_23d.csv")
    command = captured["command"]
    assert command[command.index("--input-dim") + 1] == "23"

    extracted = tmp_path / "interventions_pruned_23d.csv"
    extracted.write_text("header\nrow\n", encoding="utf-8")
    captured.clear()
    train_input = tmp_path / "out" / "training_input"
    train_input.mkdir(parents=True)
    old_18d_input = train_input / "BYD_TEST_18d.csv"
    old_18d_input.write_text("keep\n", encoding="utf-8")
    old_18d_results = train_input / "training_results" / "BYD_TEST_18d"
    old_18d_results.mkdir(parents=True)
    old_18d_marker = old_18d_results / "keep.txt"
    old_18d_marker.write_text("keep\n", encoding="utf-8")

    monkeypatch.setattr(nnlc_auto_train, "get_julia_exe", lambda: "julia")
    monkeypatch.setattr(nnlc_auto_train, "_forward_process_output", lambda *a, **k: 0)

    class Process:
        returncode = 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        result_dir = tmp_path / "out" / "training_input" / "training_results" / "BYD_TEST_23d"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "BYD_TEST_23d.json").write_text("{}", encoding="utf-8")
        return Process()

    monkeypatch.setattr(nnlc_auto_train.subprocess, "Popen", fake_popen)
    model_json, _result_dir = nnlc_auto_train.step_train(
        str(extracted), "BYD_TEST", str(tmp_path / "out"), input_dim=23
    )
    command = captured["command"]
    assert model_json.replace("\\", "/").endswith("training_results/BYD_TEST_23d/BYD_TEST_23d.json")
    assert "BYD_TEST_23d.csv" in str(list((tmp_path / "out" / "training_input").glob("*.csv"))[0])
    assert old_18d_input.read_text(encoding="utf-8") == "keep\n"
    assert old_18d_marker.read_text(encoding="utf-8") == "keep\n"
    assert "--input-dim=23" in command


def test_pruned_and_coverage_outputs_include_dimension(tmp_path, monkeypatch):
    import nnlc_auto_train

    def fake_prepare(input_csv, output_csv, **kwargs):
        from pathlib import Path

        Path(output_csv).write_text("data\n", encoding="utf-8")
        return {
            "initial_rows": 25000,
            "route_rows_dropped": 0,
            "saturated_dropped": 0,
            "lane_change_dropped": 0,
            "output_rows": 25000,
            "intervention_dropped_rows": 0,
            "intervention_event_count": 0,
        }

    monkeypatch.setattr(
        "nnlc_tools.prune_routes.prepare_training_csv", fake_prepare
    )
    final_csv, _stats = nnlc_auto_train.step_prepare_training(
        str(tmp_path / "lateral_data_23d.csv"), 60, str(tmp_path), input_dim=23
    )
    assert final_csv.endswith("interventions_pruned_23d.csv")

    captured = {}

    def fake_run(command, description, **kwargs):
        captured["command"] = command
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"png")

    from pathlib import Path

    monkeypatch.setattr(nnlc_auto_train, "run_command", fake_run)
    coverage = nnlc_auto_train.step_visualize(
        final_csv, str(tmp_path), "python", input_dim=23
    )
    assert coverage.endswith("coverage_23d.png")
    assert captured["command"][captured["command"].index("-o") + 1] == coverage

