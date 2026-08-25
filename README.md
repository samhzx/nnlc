# openpilot NNLC Training Tools

Tools for training NNLC (Neural Network Lateral Control) models for [openpilot](https://github.com/commaai/openpilot) / [sunnypilot](https://github.com/sunnypilot/sunnypilot).

NNLC replaces the standard torque lateral controller with a per-vehicle neural network that learns the relationship between desired lateral acceleration and steering torque. This produces smoother, more accurate steering — but training a model requires collecting driving data, processing it, and running the Julia training pipeline.

This repo consolidates the scattered, broken tooling into one place. Discussion and support on the [Sunnypilot forum](https://community.sunnypilot.ai/t/nnlc-tools-repo-for-complete-training-to-driving/3283).

## Prerequisites

- **Python 3.11+**
- **Julia 1.9+** — for model training (with CUDA or Metal GPU recommended)
- **comma device** — for collecting driving data (comma 3/3X)

## Existing Vehicle NNLC Models

As of Feb 2026 there are 113 NNLC models. I personally don't know the status of any of them.

<details>
<summary>View all 113 models</summary>

| Model | Training Date |
|---|---|
| ACURA_RDX_3G | 2023-08-05 |
| AUDI_A3_MK3 | 2023-08-05 |
| AUDI_Q3_MK2 | 2023-08-05 |
| BUICK_LACROSSE | 2023-08-31 |
| CHEVROLET_BOLT_EUV | 2023-08-05 |
| CHEVROLET_SILVERADO | 2023-08-05 |
| CHEVROLET_TRAILBLAZER | 2023-09-28 |
| CHEVROLET_VOLT | 2023-08-05 |
| CHRYSLER_PACIFICA_2017_HYBRID | 2023-08-05 |
| CHRYSLER_PACIFICA_2018_HYBRID | 2023-08-05 |
| CHRYSLER_PACIFICA_2019_HYBRID | 2023-08-05 |
| CHRYSLER_PACIFICA_2020 | 2023-08-05 |
| GENESIS_G70 | 2023-08-05 |
| GENESIS_GV60_EV_1ST_GEN | 2023-08-05 |
| GENESIS_GV70_1ST_GEN | 2023-09-01 |
| GMC_ACADIA | 2023-09-01 |
| HONDA_ACCORD | 2023-08-05 |
| HONDA_CIVIC | 2023-08-06 |
| HONDA_CIVIC_2022 | 2023-08-06 |
| HONDA_CIVIC_BOSCH | 2023-08-05 |
| HONDA_CLARITY | 2024-01-04 |
| HONDA_CRV_5G | 2023-08-06 |
| HONDA_CRV_HYBRID | 2023-08-06 |
| HONDA_HRV | 2023-08-06 |
| HONDA_ODYSSEY | 2023-08-06 |
| HONDA_PILOT | 2023-08-06 |
| HONDA_RIDGELINE | 2023-08-06 |
| HYUNDAI_ELANTRA_2021 | 2023-08-06 |
| HYUNDAI_ELANTRA_HEV_2021 | 2023-08-07 |
| HYUNDAI_GENESIS | 2023-08-07 |
| HYUNDAI_IONIQ_5 | 2023-08-07 |
| HYUNDAI_IONIQ_EV_LTD | 2023-08-07 |
| HYUNDAI_IONIQ_PHEV | 2023-08-07 |
| HYUNDAI_KONA_EV | 2023-08-07 |
| HYUNDAI_KONA_EV_2022 | 2023-09-01 |
| HYUNDAI_KONA_HEV | 2023-08-07 |
| HYUNDAI_PALISADE | 2023-08-07 |
| HYUNDAI_SANTA_FE | 2023-08-07 |
| HYUNDAI_SANTA_FE_2022 | 2023-08-07 |
| HYUNDAI_SANTA_FE_HEV_2022 | 2023-08-07 |
| HYUNDAI_SANTA_FE_PHEV_2022 | 2023-08-07 |
| HYUNDAI_SONATA | 2023-08-07 |
| HYUNDAI_SONATA_HYBRID | 2023-08-08 |
| HYUNDAI_SONATA_LF | 2023-08-07 |
| HYUNDAI_TUCSON_4TH_GEN | 2023-08-08 |
| JEEP_GRAND_CHEROKEE | 2023-08-08 |
| JEEP_GRAND_CHEROKEE_2019 | 2023-08-08 |
| KIA_CEED | 2023-09-02 |
| KIA_EV6 | 2023-08-08 |
| KIA_K5_2021 | 2023-08-08 |
| KIA_NIRO_EV | 2023-08-08 |
| KIA_NIRO_HEV_2021 | 2023-09-02 |
| KIA_NIRO_HEV_2ND_GEN | 2023-09-02 |
| KIA_NIRO_PHEV_2022 | 2025-07-15 |
| KIA_OPTIMA_G4_FL | 2023-08-08 |
| KIA_SELTOS | 2023-08-08 |
| KIA_SORENTO | 2023-08-08 |
| KIA_SORENTO_4TH_GEN | 2023-08-08 |
| KIA_SORENTO_HEV_4TH_GEN | 2023-08-08 |
| KIA_SPORTAGE_5TH_GEN | 2023-08-08 |
| KIA_STINGER | 2023-08-08 |
| KIA_STINGER_2022 | 2023-08-08 |
| LEXUS_ES_TSS2 | 2023-08-09 |
| LEXUS_IS | 2023-08-09 |
| LEXUS_NX | 2023-08-09 |
| LEXUS_NX_TSS2 | 2023-08-09 |
| LEXUS_RX | 2023-08-09 |
| LEXUS_RX_TSS2 | 2023-08-09 |
| MAZDA_CX5_2022 | 2023-08-09 |
| MAZDA_CX9 | 2023-07-16 |
| MAZDA_CX9_2021 | 2023-08-09 |
| RAM_1500_5TH_GEN | 2023-08-10 |
| RAM_HD_5TH_GEN | 2023-09-02 |
| SKODA_KAROQ_MK1 | 2023-07-16 |
| SKODA_KODIAQ_MK1 | 2023-08-10 |
| SKODA_OCTAVIA_MK3 | 2023-08-10 |
| SKODA_SUPERB_MK3 | 2023-08-10 |
| SUBARU_ASCENT | 2023-08-10 |
| SUBARU_FORESTER | 2023-08-10 |
| SUBARU_IMPREZA | 2023-08-10 |
| SUBARU_IMPREZA_2020 | 2023-08-10 |
| SUBARU_LEGACY | 2023-09-02 |
| SUBARU_LEGACY_PREGLOBAL | 2023-09-02 |
| SUBARU_OUTBACK | 2023-08-10 |
| TOYOTA_AVALON | 2023-08-10 |
| TOYOTA_AVALON_2019 | 2023-08-11 |
| TOYOTA_AVALON_TSS2 | 2023-08-11 |
| TOYOTA_CAMRY | 2023-08-11 |
| TOYOTA_CAMRY_TSS2 | 2023-08-11 |
| TOYOTA_CHR | 2023-08-11 |
| TOYOTA_CHR_TSS2 | 2023-08-11 |
| TOYOTA_COROLLA | 2023-08-11 |
| TOYOTA_COROLLA_TSS2 | 2023-08-11 |
| TOYOTA_HIGHLANDER | 2023-08-12 |
| TOYOTA_HIGHLANDER_TSS2 | 2023-08-12 |
| TOYOTA_MIRAI | 2023-08-12 |
| TOYOTA_PRIUS | 2023-08-12 |
| TOYOTA_PRIUS_TSS2 | 2023-08-12 |
| TOYOTA_PRIUS_V | 2023-09-03 |
| TOYOTA_RAV4 | 2023-08-12 |
| TOYOTA_RAV4H | 2023-08-12 |
| TOYOTA_RAV4_PRIME | 2025-05-15 |
| TOYOTA_RAV4_TSS2 | 2023-08-12 |
| TOYOTA_RAV4_TSS2_2022 | 2023-08-13 |
| TOYOTA_SIENNA | 2023-08-13 |
| TOYOTA_SIENNA_4TH_GEN | 2025-06-14 |
| VOLKSWAGEN_ARTEON_MK1 | 2023-08-13 |
| VOLKSWAGEN_ATLAS_MK1 | 2023-08-13 |
| VOLKSWAGEN_GOLF_MK7 | 2023-08-13 |
| VOLKSWAGEN_JETTA_MK7 | 2023-08-13 |
| VOLKSWAGEN_PASSAT_MK8 | 2023-08-13 |
| VOLKSWAGEN_PASSAT_NMS | 2023-07-17 |
| VOLKSWAGEN_TIGUAN_MK2 | 2023-08-14 |

</details>

## Installation

```bash
git clone https://github.com/amzoo/openpilot-nnlc-tools.git
cd openpilot-nnlc-tools

# Create virtual environment with uv (recommended)
uv venv
uv pip install -e .
```

Or use the setup script (handles everything including uv installation):
```bash
bash scripts/setup.sh
```

All CLI tools work with `uv run` (auto-discovers the `.venv`, no manual activation needed):
```bash
uv run nnlc-extract ./data -o output/lateral_data.csv --temporal
```

## Quick Start

The full pipeline: **sync → extract → score → prune routes → visualize → classify & prune → train → deploy**

### One-command pipeline

`scripts/prepare_training_data.sh` chains steps 1-6 into a single command:

```bash
# Full pipeline with device sync
bash scripts/prepare_training_data.sh -d 192.168.1.161

# Without sync (rlogs already in ./data)
bash scripts/prepare_training_data.sh

# Custom output dir and minimum score filter
bash scripts/prepare_training_data.sh -o ./my_output --min-score 70
```

Or run each step individually:

### 1. Sync rlogs from device

**Mac desktop shortcut:** Copy `scripts/sync_device.command` to your Desktop. Double-click it — a dialog prompts for the device IP (pre-filled with `192.168.1.161`), then syncs to `./data/`.

```bash
cp scripts/sync_device.command ~/Desktop/
```

Or run directly:

```bash
python3 -m nnlc_tools.sync_rlogs -d 192.168.1.161 -o ./data

# Dry run first to see what would be synced
python3 -m nnlc_tools.sync_rlogs -d 192.168.1.161 -o ./data --dry-run
```

### 2. Extract lateral data

```bash
# Basic extraction
python3 -m nnlc_tools.extract_lateral_data ./data -o ./output/lateral_data.csv

# With temporal features (required for training)
python3 -m nnlc_tools.extract_lateral_data ./data -o ./output/lateral_data.csv --temporal

# Parquet format (faster for large datasets)
python3 -m nnlc_tools.extract_lateral_data ./data -o ./output/lateral_data.parquet --format parquet
```

### 3. Score route quality

```bash
python3 -m nnlc_tools.score_routes ./data

# Or score from extracted CSV
python3 -m nnlc_tools.score_routes ./output/lateral_data.csv

# Only show routes scoring 70+
python3 -m nnlc_tools.score_routes ./output/lateral_data.csv --min-score 70
```

### 4. Prune routes

```bash
# Drop saturated and lane-change frames, no route exclusion (default)
uv run nnlc-prune-routes ./output/lateral_data.csv -o ./output/lateral_data_routes_pruned.csv

# Also exclude routes scoring below 60
uv run nnlc-prune-routes ./output/lateral_data.csv --min-score 60 -o ./output/lateral_data_routes_pruned.csv

# Keep saturated frames (opt out of frame-level filter)
uv run nnlc-prune-routes ./output/lateral_data.csv --keep-saturated -o ./output/lateral_data_routes_pruned.csv
```

### 5. Visualize data coverage

```bash
python3 -m nnlc_tools.visualize_coverage ./output/lateral_data_routes_pruned.csv -o ./output/coverage.png
```

This generates a 6-panel plot (2 rows):
- **Speed vs lateral accel heatmap** — shows data density, highlights gaps (<50 samples in red)
- **Lateral accel distribution** — shows balance of left/right turning data
- **Override rate by speed** — shows where the driver is fighting the controller
- **Override rate by lat accel** — where in the lat-accel range interventions cluster
- **Override density heatmap** — speed × lat-accel concentration of override events
- **Torque magnitude during overrides** — distribution of driver torque inputs when steering_pressed

### 6. Classify & prune interventions

```bash
# Prune all override frames (driver + mechanical) — default
uv run nnlc-interventions ./output/lateral_data_routes_pruned.csv \
    --prune-output ./output/lateral_data_pruned.csv

# Prune only mechanical disturbances (keep driver interventions)
uv run nnlc-interventions ./output/lateral_data_routes_pruned.csv \
    --prune mechanical --prune-output ./output/lateral_data_pruned.csv

# Optional: cascade feature diagnostic plot
uv run nnlc-interventions ./output/lateral_data_routes_pruned.csv --plot \
    --prune-output ./output/lateral_data_pruned.csv \
    -o ./output/interventions.png

# Optional: standalone feature explorer
uv run nnlc-sc-visualize ./output/lateral_data_routes_pruned.csv -o ./output/sc_features.png
```

The cascade classifier labels each `steering_pressed` event as a **driver** intervention or **mechanical** disturbance (pothole/bump). `--prune-output` writes active frames with the selected event type(s) removed. Default is `both` — removes all override frames for the cleanest training signal. Use `--prune mechanical` to keep driver corrections in the data.

### 7. Assess coverage and iterate

Review the coverage chart from step 4. If you see red bins (gaps with <50 samples), collect more driving data targeting those conditions before training. Common gaps:
- Low-speed tight turns (city driving)
- High-speed gentle curves (highway)
- One turning direction over the other

### 8. Train model

See [training/README.md](training/README.md) for Julia setup and training instructions.

```bash
# Recommended — handles juliaup PATH automatically
bash training/run.sh ./output/lateral_data_pruned.csv

# Or run Julia directly
cd training/
julia latmodel_temporal.jl ../output/lateral_data_pruned.csv

# Force CPU mode (no GPU required, slower for large datasets)
bash training/run.sh ./output/lateral_data_pruned.csv --cpu
```

### 9. Deploy model

Copy the output JSON to your openpilot install:

```bash
cp my_car_model.json /path/to/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/
```

The filename should match your car's fingerprint. See `sunnypilot/selfdrive/controls/lib/nnlc/helpers.py` for the naming convention.

## Driving Tips for Data Collection

Good training data is diverse and clean. Aim for:

- **Disable NNLC while collecting**: Use the stock torque controller during data collection so the torque signal reflects the base controller, not a previous model
- **Disable "lateral on blinker"**: Turn off any blinker-based lateral override settings to avoid noisy data during lane changes
- **Varied speeds**: City streets (5-15 m/s), suburban (15-25 m/s), highway (25-35 m/s)
- **Varied turns**: Gentle curves, tight turns, S-curves, on-ramps/off-ramps
- **Minimal overrides**: Let the controller drive — interventions corrupt the torque signal
- **Both directions**: Left and right turns in equal measure
- **Different road grades**: Flat, uphill, downhill — affects roll compensation
- **Multiple routes**: Don't just drive the same loop repeatedly — aim for 20-30 clean routes across different road types
- **Dry roads**: Wet/icy roads change tire grip and produce non-representative data

**How much data?** Start with 5-10 hours of clean driving across 20-30 routes. Check coverage gaps with `visualize_coverage` and fill them with targeted drives.

**What to avoid:**
- Heavy traffic (lots of standstill/stop-and-go)
- Construction zones (lane changes, overrides)
- Parking lots (low speed, lots of turning at standstill)

## Tool Reference

### sync_rlogs

```
python -m nnlc_tools.sync_rlogs [-h] -d DEVICE -o OUTPUT [-u USER] [-p PATH] [--dry-run] [--no-rsync]

  -d, --device     Device IP address
  -o, --output     Local output directory
  -u, --user       SSH username (default: comma)
  -p, --path       Device rlog path (default: /data/media/0/realdata/)
  --dry-run        Show what would be synced
  --no-rsync       Force SFTP mode
```

### extract_lateral_data

```
python -m nnlc_tools.extract_lateral_data [-h] [-o OUTPUT] [--format {csv,parquet}] [--temporal] [--filter-overrides] input

  input               Directory containing rlog files
  -o, --output        Output file path (default: lateral_data.csv)
  --format            Output format (default: inferred from extension)
  --temporal          Add temporal lag/lead columns for NNLC training
  --filter-overrides  Drop rows where driver overrides (steering_pressed=True)
```

### score_routes

```
python -m nnlc_tools.score_routes [-h] [--min-score MIN_SCORE] input

  input            CSV/Parquet file or directory of rlogs
  --min-score      Only show routes with score >= this value
```

Scoring criteria (100 base, deductions):

| Criterion | Penalty |
|-----------|---------|
| Override rate > 10% | -15 |
| Saturated > 5% | -20 |
| Inactive > 20% | -25 |
| Standstill > 30% | -15 |
| Lane change > 10% | -10 |
| < 2 min active driving | -20 |

### prune_routes

```
nnlc-prune-routes [-h] [-o OUTPUT] [--min-score MIN_SCORE]
                  [--keep-saturated] [--keep-lane-change]
                  input

  input                CSV/Parquet file from nnlc-extract
  -o, --output         Output path (default: pruned_routes.csv)
  --min-score N        Exclude routes scoring below N (default: 0, no exclusion)
  --keep-saturated     Do not drop saturated frames
  --keep-lane-change   Do not drop lane-change frames
```

Sits between `score_routes` and `visualize_coverage` in the pipeline. Does two things:
1. **Route-level**: exclude entire routes scoring below `--min-score`
2. **Frame-level**: drop saturated frames and lane-change frames (enabled by default)

### visualize_coverage

```
python -m nnlc_tools.visualize_coverage [-h] [-o OUTPUT] [--gap-threshold GAP_THRESHOLD] [--torque-scatter] [--max-points MAX_POINTS] input

  input              CSV/Parquet file or directory of rlogs
  -o, --output       Output image path (default: coverage.png)
  --gap-threshold    Highlight bins with fewer samples (default: 50)
  --torque-scatter   Generate a separate lat_accel vs torque scatter plot
  --max-points       Max data points per torque scatter subplot (random sample)
```

### visualize_model

```
python -m nnlc_tools.visualize_model [-h] [-o OUTPUT_DIR] model data

  model              Trained model JSON file
  data               Training data CSV/Parquet file
  -o, --output-dir   Output directory for plots (default: ./output/)
```

Generates two plot sets with model prediction curves overlaid on data:
- **lat_accel_vs_torque** — per-speed-bin scatter with viridis speed coloring + model curve
- **torque_vs_speed** — per-lat_accel-bin scatter with viridis lat_accel coloring + model curve

### analyze_interventions

```
nnlc-interventions [-h] [-o OUTPUT] [--plot] [--scatter]
                   [--gap-frames GAP_FRAMES] [--max-points MAX_POINTS]
                   [--torque-rate-mechanical FLOAT]
                   [--torque-rate-driver FLOAT]
                   [--max-pothole-length FLOAT]
                   [--prune-output PATH]
                   [--prune {mechanical,driver,both}]
                   input
```

Uses a 3-stage cascade classifier (11 features, F1–F11) to distinguish **driver** corrections from **mechanical** disturbances (potholes, bumps, curb impacts). Stage 1 decides on torque rate + duration alone (~10 ms); Stage 2 adds sign consistency, zero-crossing rate, kurtosis, and longitudinal shock (~50 ms); Stage 3 adds torque–lateral-accel correlation and frequency energy ratio (~150 ms).

| Arg | Default | Effect |
|-----|---------|--------|
| `--torque-rate-mechanical` | 80.0 Nm/s | Rate above which Stage 1 calls mechanical immediately |
| `--torque-rate-driver` | 20.0 Nm/s | Rate below which Stage 1 calls driver immediately |
| `--max-pothole-length` | 2.5 m | Pothole size estimate for speed-adaptive brevity |
| `--prune-output PATH` | (none) | Write pruned active frames to PATH (.csv or .parquet) |
| `--prune` | `both` | Remove `mechanical`, `driver`, or `both` event frames |

### nnlc-sc-visualize

```
nnlc-sc-visualize [-h] [-o OUTPUT] [--gap-frames GAP_FRAMES]
                  [--torque-rate-mechanical FLOAT]
                  [--torque-rate-driver FLOAT]
                  [--max-pothole-length FLOAT]
                  input
```

Standalone 3×3 feature diagnostic plot — histograms of all 11 classifier features split by driver/mechanical, speed-vs-duration scatter, speed band bar chart, and cascade stage distribution. Useful for threshold exploration without writing pruned output.

## Troubleshooting

### Out of memory during extraction

Process fewer rlogs at a time, or use `--format parquet` which is more memory-efficient. The extractor processes segments one at a time, but the final DataFrame concatenation can be large.

### rsync connection refused

The comma device may not have rsync installed. Use `--no-rsync` to fall back to SFTP:
```bash
python3 -m nnlc_tools.sync_rlogs -d 192.168.1.161 -o ./data --no-rsync
```

### No rlog files found

Check that your rlogs are in the expected directory structure:
```
./data/
  2024-01-15--12-30-45/
    0/rlog.zst
    1/rlog.zst
    ...
```

### Julia training on CPU

CPU training works — expect ~8 seconds for 1000 epochs on small datasets. Use `--cpu` to force CPU mode:

```bash
bash training/run.sh /path/to/latmodels/ --cpu
```

GPU (CUDA or Metal) is still recommended for large datasets due to speed. See [training/README.md](training/README.md).

## Source Attribution

This project builds on work from:
- [mmmorks/sunnypilot](https://github.com/mmmorks/sunnypilot) (`staging-merged` @ `8a9f0311`) — Python rlog processing tools
- [ryanomatic/rlog_aggregation](https://github.com/ryanomatic/rlog_aggregation) (`main` @ `26b1ea05`) — Rlog download tool
- [mmmorks/OP_ML_FF](https://github.com/mmmorks/OP_ML_FF) (`master` @ `0116b9e3`, forked from [twilsonco/OP_ML_FF](https://github.com/twilsonco/OP_ML_FF)) — Julia training scripts
- warren.2 — Testing, debugging, and pipeline feedback that shaped the tool design

## Roadmap

Derived from community feedback from the Sunnypilot tuning-nnlc Discord channel.

- [x] **Forum documentation** — Published guide to Sunnypilot forum
- [x] **Canonical repo** — Tools consolidated from 3 scattered repos into this one
- [x] **Simplified rlog processing** — Refactored to accept a single input directory, stripped multi-server logic
- [x] **Rlog syncing** — `nnlc-sync` with rsync (primary) and SFTP fallback, incremental sync
- [x] **Dependencies** — `pyproject.toml` with pinned versions, bundled cereal schemas (no openpilot checkout needed)
- [x] **CPU training** — Fixed with `CustomAdaGrad` optimizer and `--cpu` flag
- [x] **Coverage visualization** — `nnlc-visualize` generates 3-panel coverage chart (heatmap, histogram, override rate)
- [x] **Route quality scoring** — `nnlc-score` with 6-criteria scorer (100-point scale)
- [x] **Route pruning** — removes saturated and lane-change frames, optionally excludes low-scoring routes before training
- [x] **Driving guidance** — Documented in README (data collection tips, what to avoid)
- [x] **End-to-end guide** — README covers full pipeline: sync → extract → score → visualize → train → deploy
- [x] **Troubleshooting** — Common issues documented (OOM, rsync, rlogs, CPU training)
- [x] **HKG compatibility** — Fixed rlog parsing failures for Hyundai/Kia/Genesis
- [x] **Model validation plots** — `nnlc-validate` generates lat_accel_vs_torque and torque_vs_speed plots with model curves
- [x] **Steering input filtering** — 3-stage cascade classifier (`nnlc-interventions`) distinguishes driver corrections from mechanical disturbances; `--prune-output` removes unwanted frames before training
- [ ] **NNLC-on data quality** — Investigate whether collecting data with an existing NNLC model active degrades the next trained model
- [ ] **Temporal signal alignment** — Verify that all signals in each training row share the same timestamp and that lag/lead columns are correctly offset
- [ ] **AMD GPU support** — Port training to support ROCm (community member with 7900 XT available to test)
- [ ] **Honda/Acura EPS filtering** — Review and integrate `Micim987/opendbc` signal filtering
- [ ] **Mazda compatibility** — Investigate signal compatibility issues; does the HKG fix above address this too?
