# NNLC Julia Training Scripts

These Julia scripts train the neural network feedforward model used by NNLC (Neural Network Lateral Control) in openpilot/sunnypilot.

## Which Script to Use

| Script | Description | Recommended |
|--------|-------------|-------------|
| `latmodel_temporal.jl` | Temporal model with past/future context | **Yes — primary** |
| `latmodel.jl` | Base model (no temporal features) | For comparison only |
| `latmodel_temporal_steer_angle.jl` | Temporal + steer angle input | Experimental |
| `latmodel_NeuralPDE.jl` | Physics-informed (Lux + NeuralPDE) | Experimental |

**Use `latmodel_temporal.jl`** — this is the model variant that matches the production NNLC code in sunnypilot.

## Input Format

The training script reads a CSV file produced by `extract_lateral_data.py --temporal`. Required columns:

- `v_ego` — vehicle speed (m/s)
- `actual_lateral_accel` — measured lateral acceleration (m/s²)
- `desired_lateral_accel` — desired lateral acceleration (m/s²)
- `roll` — road roll angle (radians)
- Temporal columns at offsets: `-0.3, -0.2, -0.1, +0.3, +0.6, +1.0, +1.5` seconds

The script expects the CSV at a path like `/path/to/latmodels/YOUR_CAR_NAME.csv`.

## Output Format

The training script outputs a JSON file compatible with sunnypilot's `NNTorqueModel`:

```json
{
  "input_size": 25,
  "output_size": 1,
  "input_mean": [15.2, 0.01, ...],
  "input_std": [8.5, 1.2, ...],
  "layers": [
    {
      "layer1_W": [[...weights...]],
      "layer1_b": [...biases...],
      "activation": "sigmoid"
    },
    ...
  ]
}
```

Deploy the JSON file to: `sunnypilot/neural_network_data/neural_network_lateral_control/`

## Dependencies

Install Julia 1.9+ and the required packages:

```julia
using Pkg
Pkg.add(["Flux", "MLUtils", "CSV", "DataFrames", "JSON", "Statistics",
         "StatsBase", "Plots", "ProgressMeter", "CUDA"])
```

For Apple Silicon (Metal GPU):
```julia
Pkg.add("Metal")
```

## Known Issues

### CPU Training

CPU training works reliably with the CustomAdaGrad optimizer (~8 seconds for 1000 epochs on small datasets). Use `--cpu` to force CPU mode:

```bash
bash training/run.sh /path/to/latmodels/ --cpu
```

GPU is still recommended for large datasets:
- **NVIDIA GPU with CUDA** — fastest and most reliable
- **Apple Silicon with Metal** — supported via `latmodel_temporal.jl`
- M1 Pro with 16GB RAM is insufficient for larger datasets — consider M2 Pro/Max or better

### AMD GPU

AMD GPU support via ROCm is not yet working with Julia's Flux. This is an open area of investigation — rgbacon has volunteered a 7900 XT for testing.

## Usage

```bash
# 1. Extract training data with temporal features
python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o /path/to/latmodels/my_car.csv --temporal

# 2. Run training (recommended — handles juliaup PATH automatically)
bash training/run.sh /path/to/latmodels/

# Or run Julia directly
cd training/
julia latmodel_temporal.jl /path/to/latmodels/

# Force CPU mode (no GPU required)
bash training/run.sh /path/to/latmodels/ --cpu

# 3. Deploy model
cp /path/to/latmodels/my_car_model.json \
   /path/to/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/
```
