# NNLC Julia 训练脚本

这些 Julia 脚本用于训练 NNLC（Neural Network Lateral Control，神经网络横向控制）在 openpilot/sunnypilot 中使用的神经网络前馈模型。

## 选择哪个脚本

| 脚本 | 说明 | 推荐程度 |
|---|---|---|
| `latmodel_temporal.jl` | 包含过去/未来上下文的时序模型 | **是，主模型** |
| `latmodel.jl` | 基础模型，不含时序特征 | 仅用于对比 |
| `latmodel_temporal_steer_angle.jl` | 时序模型 + 转向角输入 | 实验性 |
| `latmodel_NeuralPDE.jl` | 物理信息模型（Lux + NeuralPDE） | 实验性 |

请使用 `latmodel_temporal.jl`。该模型版本与 sunnypilot 中的生产 NNLC 代码一致。

## 输入格式

训练脚本读取由 `extract_lateral_data.py --temporal` 生成的 CSV 文件。所需列包括：

- `v_ego`：车速（m/s）
- `desired_lateral_accel`：期望横向加速度（m/s²）
- `friction_input`：横向加速度误差与前瞻横向 jerk 组合得到的摩擦补偿输入
- `roll`：道路横滚角（弧度）
- `torque_output`：模型训练目标
- `desired_lateral_accel` 和 `roll` 的时间偏移列：`-0.3`、`-0.2`、`-0.1`、`+0.3`、`+0.6`、`+1.0`、`+1.5` 秒

最终模型共有 18 个输入：4 个当前状态输入、7 个期望横向加速度时序输入和 7 个横滚角时序输入。`route_id` 和 `timestamp` 仅用于隔离训练集与测试集，不会写入模型输入。

CSV 文件路径应类似 `/path/to/latmodels/YOUR_CAR_NAME.csv`。

## 输出格式

训练脚本输出可供 sunnypilot `NNTorqueModel` 使用的 JSON 文件：

```json
{
  "input_size": 18,
  "output_size": 1,
  "input_mean": [15.2, 0.01, ...],
  "input_std": [8.5, 1.2, ...],
  "layers": [
    {
      "dense_1_W": [[...weights...]],
      "dense_1_b": [...biases...],
      "activation": "sigmoid"
    },
    ...
  ]
}
```

将 JSON 文件部署到：`sunnypilot/neural_network_data/neural_network_lateral_control/`

## 依赖

安装 Julia 1.9 或更高版本，以及所需软件包：

```julia
using Pkg
Pkg.add(["Flux", "MLUtils", "CSV", "DataFrames", "JSON", "Statistics",
         "StatsBase", "Plots", "ProgressMeter"])
```

训练脚本固定使用 CPU，不需要安装 CUDA 或 Metal GPU 软件包。

## 已知问题

### CPU 训练

使用 CustomAdaGrad 优化器时，CPU 训练运行稳定；小数据集训练 1000 个 epoch 约需 8 秒。训练脚本固定使用 CPU：

```bash
bash training/run.sh /path/to/latmodels/ --cpu
```

大型数据集建议使用“CPU 低内存模式”并适当减小批次大小，以控制内存峰值。

## 使用方法

```bash
# 1. 提取包含时序特征的训练数据
python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o /path/to/latmodels/my_car.csv --temporal

# 2. 运行 CPU 训练（推荐，会自动处理 juliaup PATH）
bash training/run.sh /path/to/latmodels/

# 或直接运行 Julia
cd training/
julia latmodel_temporal.jl /path/to/latmodels/

# CPU 训练（--cpu 为兼容旧命令保留）
bash training/run.sh /path/to/latmodels/ --cpu

# 3. 部署模型
cp /path/to/latmodels/my_car_model.json \
   /path/to/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/
```
