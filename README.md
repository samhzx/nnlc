# openpilot NNLC 训练工具

用于训练 [openpilot](https://github.com/commaai/openpilot) / [sunnypilot](https://github.com/sunnypilot/sunnypilot) NNLC（Neural Network Lateral Control，神经网络横向控制）模型的工具。

NNLC 使用针对每辆车的神经网络替代标准扭矩横向控制器，学习期望横向加速度与转向扭矩之间的关系。这样可以获得更平顺、更准确的转向，但训练模型需要采集驾驶数据、处理数据并运行 Julia 训练流程。

本仓库将分散且无法正常工作的工具整合到了一处。相关讨论和支持请参见 [Sunnypilot 论坛](https://community.sunnypilot.ai/t/nnlc-tools-repo-for-complete-training-to-driving/3283)。

## 前置条件

- **Python 3.11+**
- **Julia 1.9+** — 用于 CPU 模型训练
- **comma 设备** — 用于采集驾驶数据（comma 3/3X）

如果只使用 Windows 发布包，不需要在目标电脑安装 Python、Julia 或项目依赖；请直接阅读下方的 [Windows 图形界面](#windows-图形界面) 章节。命令行开发和从源码训练仍需要安装 Python 与 Julia。

## 项目结构

```text
nnlc/
├── nnlc_gui.py                 # Tkinter 图形界面入口
├── nnlc_auto_train.py          # 提取、评分、剪枝、训练和验证的一键流程
├── nnlc_tools/                 # rlog 处理、评分、可视化和干预分类工具
├── training/
│   ├── latmodel_temporal.jl   # 主训练脚本（GUI 和自动流程使用）
│   ├── run.sh                 # Julia 训练启动脚本
│   └── 其他 *.jl              # 实验或对比模型，不参与默认流程
├── build_windows.ps1          # Windows one-dir 构建脚本
└── nnlc_windows.spec          # PyInstaller 配置
```

默认训练只使用 `training/latmodel_temporal.jl`。`latmodel.jl`、`latmodel_temporal_steer_angle.jl` 和 `latmodel_NeuralPDE.jl` 保留用于实验或结果对比，不会被 GUI 自动调用。

## Windows 图形界面

Windows 发布包提供 `NNLC_Trainer.exe` 图形界面，适合不想使用命令行的场景。启动后按以下顺序填写：

1. 选择包含 `reallog`/`rlog` 文件的输入目录。
2. 选择输出目录；中间 CSV、覆盖度图、模型验证图和最终 JSON 会保存在这里。
3. 输入车型 `carFingerprint`，例如 `BYD_TANG_DMI_24`。车型仅要求非空，不做额外格式限制。
4. 路线阈值保持“自动推荐”，或取消勾选后填写 `0-100` 的整数。
5. 训练模式选择“CPU 标准模式”或“CPU 低内存模式”。项目固定使用 CPU，不会启用 GPU；低内存模式使用更小批次，适合内存较小的电脑。
6. 日志处理模式默认是严格模式。若设备下载日志时可能仍在写入，可选择“容错模式（跳过损坏日志）”；被跳过的文件会写入运行日志，训练前应确认剩余数据量足够。

车型、训练模式和日志处理模式会自动记住上次选择。覆盖度图默认生成，也可以在界面中取消。

one-dir 程序无需安装目标电脑上的 Python、Julia 或依赖包，但必须整体保留 `NNLC_Trainer` 文件夹，不能只复制 `NNLC_Trainer.exe`。GitHub Actions 会将它打包为 `NNLC_Trainer-windows-x64-onedir.7z`；下载后使用 7-Zip、WinRAR 等工具完整解压，再运行 `NNLC_Trainer\NNLC_Trainer.exe`。

## 现有车型 NNLC 模型

截至 2026 年 2 月，共有 113 个 NNLC 模型。作者不了解这些模型目前各自的状态。

<details>
<summary>查看全部 113 个模型</summary>

| 模型 | 训练日期 |
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

## 安装

```bash
git clone https://github.com/samhzx/nnlc.git
cd nnlc

# 使用 uv 创建虚拟环境（推荐）
uv venv
uv pip install -e .
# 仅在需要读写 Parquet 时额外安装：
# uv pip install -e ".[parquet]"
```

所有命令行工具都可以通过 `uv run` 运行（会自动发现 `.venv`，无需手动激活）：
```bash
uv run nnlc-extract ./data -o output/lateral_data.csv --temporal
```

## 快速开始

完整流程：**提取 → 评分 → 剪枝路线 → 可视化 → 分类并剪枝 → 训练 → 部署**。请先将设备中的 rlog 文件复制到本地目录，再从下面的第 1 步开始。

也可以逐步运行：

### 1. 提取横向数据

```bash
# 基本提取
python3 -m nnlc_tools.extract_lateral_data ./data -o ./output/lateral_data.csv

# 包含时序特征（训练必需）
python3 -m nnlc_tools.extract_lateral_data ./data -o ./output/lateral_data.csv --temporal

# Parquet 格式（大型数据集更快）
python3 -m nnlc_tools.extract_lateral_data ./data -o ./output/lateral_data.parquet --format parquet
```

### 2. 评估路线质量

```bash
python3 -m nnlc_tools.score_routes ./data

# 或从提取出的 CSV 评分
python3 -m nnlc_tools.score_routes ./output/lateral_data.csv

# 仅显示评分达到 70 分的路线
python3 -m nnlc_tools.score_routes ./output/lateral_data.csv --min-score 70
```

### 3. 剪枝路线

```bash
# 删除饱和帧和变道帧，不排除路线（默认）
uv run nnlc-prune-routes ./output/lateral_data.csv -o ./output/lateral_data_routes_pruned.csv

# 同时排除评分低于 60 分的路线
uv run nnlc-prune-routes ./output/lateral_data.csv --min-score 60 -o ./output/lateral_data_routes_pruned.csv

# 保留饱和帧（停用帧级筛选）
uv run nnlc-prune-routes ./output/lateral_data.csv --keep-saturated -o ./output/lateral_data_routes_pruned.csv
```

### 4. 可视化数据覆盖度

```bash
python3 -m nnlc_tools.visualize_coverage ./output/lateral_data_routes_pruned.csv -o ./output/coverage.png
```

这会生成一个 6 面板图（2 行）：
- **车速与横向加速度热力图** — 显示数据密度，并用红色标出缺口（样本少于 50）
- **横向加速度分布** — 显示左右转向数据是否均衡
- **按车速统计的接管率** — 显示驾驶员与控制器对抗的位置
- **按横向加速度统计的接管率** — 显示干预集中在哪个横向加速度范围
- **接管密度热力图** — 接管事件在车速 × 横向加速度上的分布
- **接管期间的扭矩幅值** — `steering_pressed` 时驾驶员扭矩输入的分布

### 5. 分类并剪枝干预事件

```bash
# 剪除所有接管帧（驾驶员 + 机械干扰）——默认
uv run nnlc-interventions ./output/lateral_data_routes_pruned.csv \
    --prune-output ./output/lateral_data_pruned.csv

# 仅剪除机械干扰（保留驾驶员干预）
uv run nnlc-interventions ./output/lateral_data_routes_pruned.csv \
    --prune mechanical --prune-output ./output/lateral_data_pruned.csv

# 可选：级联特征诊断图
uv run nnlc-interventions ./output/lateral_data_routes_pruned.csv --plot \
    --prune-output ./output/lateral_data_pruned.csv \
    -o ./output/interventions.png

# 可选：独立特征探索器
uv run nnlc-sc-visualize ./output/lateral_data_routes_pruned.csv -o ./output/sc_features.png
```

级联分类器会将每个 `steering_pressed` 事件标记为**驾驶员**干预或**机械**干扰（坑洼/颠簸）。`--prune-output` 会将删除所选事件类型后的有效帧写入文件。默认值为 `both`，即删除所有接管帧，以获得最干净的训练信号；使用 `--prune mechanical` 可在数据中保留驾驶员修正。

### 6. 检查覆盖度并迭代

检查第 4 步生成的覆盖度图。如果看到红色区间（样本少于 50 的缺口），请在训练前针对这些条件继续采集驾驶数据。常见缺口包括：
- 低速急转弯（城市驾驶）
- 高速缓弯（高速公路）
- 某一侧转向数据明显多于另一侧

### 7. 训练模型

Julia 配置和训练说明请参见 [training/README.md](training/README.md)。

```bash
# 推荐方式——自动处理 juliaup PATH
mkdir -p ./output/latmodels
cp ./output/lateral_data_pruned.csv ./output/latmodels/YOUR_CAR.csv
bash training/run.sh ./output/latmodels

# 或直接运行 Julia
cd training/
julia latmodel_temporal.jl ../output/latmodels

# CPU 训练（低内存时可将训练模式切换为 CPU 低内存模式）
bash training/run.sh ./output/latmodels --cpu
```

### 8. 部署模型

将输出的 JSON 复制到 openpilot 安装目录：

```bash
cp my_car_model.json /path/to/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/
```

文件名应与车辆 fingerprint 匹配。命名规则请参见 `sunnypilot/selfdrive/controls/lib/nnlc/helpers.py`。

## 数据采集建议

良好的训练数据应当多样且干净，建议做到：

- **采集时关闭 NNLC**：使用原厂扭矩控制器，使扭矩信号反映基础控制器，而不是已有模型
- **关闭“转向灯触发横向控制”**：关闭基于转向灯的横向覆盖设置，避免变道时产生噪声数据
- **覆盖不同车速**：城市道路（5-15 m/s）、郊区道路（15-25 m/s）、高速公路（25-35 m/s）
- **覆盖不同转弯**：缓弯、急转弯、S 弯、上下匝道
- **尽量少接管**：让控制器驾驶，干预会污染扭矩信号
- **覆盖两个方向**：左右转弯数量尽量相等
- **覆盖不同坡度**：平路、上坡、下坡都会影响横滚补偿
- **使用多条路线**：不要反复驾驶同一个循环路线，建议在不同道路类型上采集 20-30 条干净路线
- **选择干燥路面**：湿滑或结冰路面会改变轮胎抓地力，产生不具代表性的数据

**需要多少数据？** 建议先在 20-30 条路线中采集 5-10 小时干净数据。使用 `visualize_coverage` 检查覆盖缺口，再针对缺口驾驶补充数据。

**应避免的场景：**
- 拥堵交通（大量停车和走走停停）
- 施工区域（频繁变道和接管）
- 停车场（低速且经常在静止状态转向）

## 工具参考

### extract_lateral_data

```
python -m nnlc_tools.extract_lateral_data [-h] [-o OUTPUT] [--format {csv,parquet}] [--temporal] [--filter-overrides] [--skip-corrupt] input

  input               包含 rlog 文件的目录
  -o, --output        输出文件路径（默认：lateral_data.csv）
  --format            输出格式（默认：根据扩展名推断）
  --temporal          添加 NNLC 训练所需的时序滞后/超前列
  --filter-overrides  删除驾驶员接管的行（steering_pressed=True）
  --skip-corrupt      跳过无法解析的损坏 rlog，继续处理其他文件并在末尾列出
```

提取器默认使用严格模式，遇到损坏日志会立即停止。确认部分日志可能在下载时仍处于写入状态时，可使用 `--skip-corrupt` 跳过这些文件；训练前应检查日志中的跳过清单，并确认剩余数据量充足。

### score_routes

```
python -m nnlc_tools.score_routes [-h] [--min-score MIN_SCORE] input

  input            CSV/Parquet 文件或 rlog 目录
  --min-score      仅显示评分 >= 此值的路线
```

评分标准（基础分 100 分，按以下项目扣分）：

| 项目 | 扣分 |
|-----------|---------|
| 接管率 > 10% | -15 |
| 饱和帧 > 5% | -20 |
| 非激活帧 > 20% | -25 |
| 静止帧 > 30% | -15 |
| 变道帧 > 10% | -10 |
| 有效驾驶少于 2 分钟 | -20 |

### prune_routes

```
nnlc-prune-routes [-h] [-o OUTPUT] [--min-score MIN_SCORE]
                  [--keep-saturated] [--keep-lane-change]
                  input

  input                nnlc-extract 生成的 CSV/Parquet 文件
  -o, --output         输出路径（默认：pruned_routes.csv）
  --min-score N        排除评分低于 N 的路线（默认：0，不排除）
  --keep-saturated     不删除饱和帧
  --keep-lane-change   不删除变道帧
```

它位于流程中的 `score_routes` 和 `visualize_coverage` 之间，完成两件事：
1. **路线级**：排除评分低于 `--min-score` 的整条路线
2. **帧级**：删除饱和帧和变道帧（默认启用）

### visualize_coverage

```
python -m nnlc_tools.visualize_coverage [-h] [-o OUTPUT] [--gap-threshold GAP_THRESHOLD] [--torque-scatter] [--max-points MAX_POINTS] input

  input              CSV/Parquet 文件或 rlog 目录
  -o, --output       输出图像路径（默认：coverage.png）
  --gap-threshold    突出显示样本较少的区间（默认：50）
  --torque-scatter   额外生成横向加速度与扭矩散点图
  --max-points       每个扭矩散点子图的最大数据点数（随机采样）
```

### visualize_model

```
python -m nnlc_tools.visualize_model [-h] [-o OUTPUT_DIR] model data

  model              已训练模型的 JSON 文件
  data               训练数据 CSV/Parquet 文件
  -o, --output-dir   图表输出目录（默认：./output/）
```

生成两组叠加模型预测曲线的数据图：
- **lat_accel_vs_torque** — 按车速分箱的散点图，使用 viridis 以车速着色，并叠加模型曲线
- **torque_vs_speed** — 按横向加速度分箱的散点图，使用 viridis 以横向加速度着色，并叠加模型曲线

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

使用包含 11 个特征（F1-F11）的三级级联分类器，区分**驾驶员**修正与**机械**干扰（坑洼、颠簸、路缘碰撞）。第 1 阶段仅根据扭矩变化率和持续时间决策（约 10 ms）；第 2 阶段加入符号一致性、过零率、峰度和纵向冲击（约 50 ms）；第 3 阶段加入扭矩-横向加速度相关性和频率能量比（约 150 ms）。

| 参数 | 默认值 | 作用 |
|-----|---------|--------|
| `--torque-rate-mechanical` | 80.0 Nm/s | 高于此变化率时，第 1 阶段立即判定为机械干扰 |
| `--torque-rate-driver` | 20.0 Nm/s | 低于此变化率时，第 1 阶段立即判定为驾驶员干预 |
| `--max-pothole-length` | 2.5 m | 用于按车速调整短事件阈值的坑洼尺寸估计 |
| `--prune-output PATH` | （无） | 将剪枝后的有效帧写入 PATH（.csv 或 .parquet） |
| `--prune` | `both` | 删除 `mechanical`、`driver` 或 `both` 类型的事件帧 |

### nnlc-sc-visualize

```
nnlc-sc-visualize [-h] [-o OUTPUT] [--gap-frames GAP_FRAMES]
                  [--torque-rate-mechanical FLOAT]
                  [--torque-rate-driver FLOAT]
                  [--max-pothole-length FLOAT]
                  input
```

独立的 3×3 特征诊断图：按驾驶员/机械类别分别绘制 11 个分类器特征的直方图、车速-持续时间散点图、车速区间柱状图和级联阶段分布图。适合探索阈值，无需写入剪枝后的输出文件。

## 故障排查

### 提取时内存不足

CSV 提取器会逐段处理 rlog，内存峰值主要由当前 rlog 和时序窗口决定。Parquet 输出仍需由 pandas 读取最终 DataFrame，超大数据集建议优先使用 CSV 或分批处理。

### 未找到 rlog 文件

检查 rlog 是否位于预期的目录结构中：
```
./data/
  2024-01-15--12-30-45/
    0/rlog.zst
    1/rlog.zst
    ...
```

### 使用 CPU 进行 Julia 训练

CPU 训练可以正常工作，小型数据集运行 1000 个 epoch 预计需要约 8 秒。项目当前固定使用 CPU；`--cpu` 参数仅为兼容旧命令保留：

```bash
bash training/run.sh /path/to/latmodels/ --cpu
```

当前项目固定使用 CPU 训练，不会自动探测或启用 GPU。

## 来源致谢

本项目基于以下项目的工作：
- [mmmorks/sunnypilot](https://github.com/mmmorks/sunnypilot) (`staging-merged` @ `8a9f0311`) — Python rlog 处理工具
- [ryanomatic/rlog_aggregation](https://github.com/ryanomatic/rlog_aggregation) (`main` @ `26b1ea05`) — rlog 下载工具
- [mmmorks/OP_ML_FF](https://github.com/mmmorks/OP_ML_FF) (`master` @ `0116b9e3`，fork 自 [twilsonco/OP_ML_FF](https://github.com/twilsonco/OP_ML_FF)) — Julia 训练脚本
- warren.2 — 参与测试、调试并提供流程反馈，帮助完善工具设计

## 路线图

根据 Sunnypilot tuning-nnlc Discord 频道的社区反馈整理。

- [x] **论坛文档** — 已在 Sunnypilot 论坛发布指南
- [x] **规范仓库** — 将 3 个分散仓库中的工具整合到本仓库
- [x] **简化 rlog 处理** — 重构为接收单个输入目录，移除多服务器逻辑
- [x] **依赖管理** — `pyproject.toml` 固定版本并内置 cereal schema（无需检出 openpilot）
- [x] **CPU 训练** — 使用 `CustomAdaGrad` 优化器并固定 CPU 设备
- [x] **覆盖度可视化** — `nnlc-visualize` 生成三面板覆盖度图（热力图、直方图、接管率）
- [x] **路线质量评分** — `nnlc-score` 使用六项标准评分（百分制）
- [x] **路线剪枝** — 删除饱和帧和变道帧，并可在训练前排除低评分路线
- [x] **驾驶指南** — README 已记录数据采集建议和应避免的场景
- [x] **端到端指南** — README 覆盖完整流程：提取 → 评分 → 可视化 → 训练 → 部署
- [x] **故障排查** — 已记录常见问题（OOM、rlog、CPU 训练）
- [x] **HKG 兼容性** — 修复 Hyundai/Kia/Genesis 的 rlog 解析失败
- [x] **模型验证图** — `nnlc-validate` 生成带模型曲线的 lat_accel_vs_torque 和 torque_vs_speed 图
- [x] **转向输入筛选** — 三级级联分类器（`nnlc-interventions`）区分驾驶员修正和机械干扰；`--prune-output` 可在训练前删除不需要的帧
- [ ] **NNLC 开启时的数据质量** — 研究采集数据时启用已有 NNLC 模型是否会降低下一模型的训练质量
- [ ] **时序信号对齐** — 验证每个训练行中的信号是否共享同一时间戳，以及滞后/超前列是否正确偏移
- [ ] **Honda/Acura EPS 筛选** — 评估并集成 `Micim987/opendbc` 的信号筛选
- [ ] **Mazda 兼容性** — 调查信号兼容性问题，并确认上述 HKG 修复是否也能解决
