# 转向事件分类器

**在 ADAS/自动驾驶接管事件日志中区分真实的驾驶员转向干预与道路机械干扰。**

当车辆自动驾驶系统检测到超过自身控制能力的转向扭矩时，会标记一个 `steering_pressed` 接管事件。但并非每次接管都是真实的驾驶员干预：坑洼、路面颠簸、路缘碰撞和粗糙路面都可能产生触发同一标记的扭矩尖峰。将坑洼误判为驾驶员接管（或反之）会污染驾驶员参与度指标、安全分析和系统调校。本分类器用于区分这两类事件。

---

## 工作原理

分类器从每个接管事件中提取 11 个信号特征，并通过一个在延迟和准确率之间权衡的**三级级联**进行处理。每个阶段都可以直接给出最终判断，也可以将不确定事件交给下一阶段获取更多证据。

```
Event frames (100 Hz)
        │
        ▼
┌─────────────────────────────┐
│  阶段 1 — 快速门控（10ms）   │  仅使用扭矩变化率和持续时间
│  可处理明显案例              │  → “确定为坑洼”或“确定为驾驶员”
└──────────┬──────────────────┘
           │ 不明确
           ▼
┌──────────────────────────────────┐
│  阶段 2 — 确认（50ms）           │  + 符号一致性、ZCR、峰度、
│  加权多特征评分                 │    纵向冲击、扭矩-转角相位
└──────────┬───────────────────────┘
           │ 仍不明确
           ▼
┌──────────────────────────────────────┐
│  阶段 3 — 上下文门控（200ms）        │  + 扭矩-横向加速度相关性、
│  跨信号相关特征                      │    频率能量比、横向加速度残差
│  处理边界案例                        │
└──────────────────────────────────────┘
           │
           ▼
   （“driver” | “mechanical”，置信度）
```

系统在无法确定时默认判定为 **“driver”**。将真实驾驶员干预误判为机械干扰，比反向误判具有更高的安全风险。

---

## 11 个特征

每个特征的设计都基于车辆动力学研究和 EPAS（Electric Power Assisted Steering，电动助力转向）控制文献。特征按可计算所需的延迟等级分组。

### 第 1 级 — 快速门控（1 个样本 / 10 ms）

**F1：扭矩峰值变化率（Nm/s）** —— 转向扭矩的一阶导数是最快的区分指标。人体神经肌肉动力学会对转向输入形成二阶低通滤波，使正常操作中的主动扭矩变化率大致不超过 20 Nm/s。道路冲击完全绕过驾驶员，通过转向连杆上的直接机械力产生 50–100+ Nm/s 的扭矩变化率。

原分类器使用 500 Nm/s 阈值，这一阈值约保守了 5–10 倍。根据文献校准，机械干扰/驾驶员干预的边界范围应为 50–80 Nm/s。

*来源：*
- Pick & Cole, "Neuromuscular dynamics in the driver–vehicle system," *Vehicle System Dynamics*, 44(sup1):511-522, 2006 — 建立了限制驾驶员扭矩变化率的神经肌肉带宽上限。
- Springer 关于仪器化方向盘特性的章节（2024）——测得无意识反射扭矩峰值为 0.13 ± 0.03 s，而有意识转向为 0.28 ± 0.11 s；反射扭矩仅约为最大自主扭矩的 20%。

**F2：事件持续时间（秒）** —— 道路冲击瞬态持续 20–200 ms，具体取决于车速和障碍物几何形状。车道保持修正持续 0.5–3 s；变道需要 2–5 s。紧急避让动作通常持续 0.3–2 s，并伴随较高且持续的扭矩。

*来源：*
- Schinkel 等，"Driver Intervention Detection via Real-Time Transfer Function Estimation," *IEEE Trans. ITS*, 2021 — 发现可靠的事件分类至少需要 0.2 s 数据。

### 第 2 级 — 确认门控（5 个样本 / 50 ms）

**F3：扭矩符号一致性** —— 主动转向具有单向性：驾驶员左转时会在整个动作中持续施加负（或正）扭矩。道路干扰会产生符号快速交替的振荡扭矩。主导符号样本数与总样本数之比可量化这一特征。

*来源：*
- Moreillon 等，"Hands On/Off Detection Based on EPS Sensors," *JTEKT Engineering Journal*, No. 1017E, 2020 — 测得无人握方向盘时在鹅卵石路面上有 ±1–2 Nm 的振荡扭矩，在沥青路面自动驾驶时有 ±3 Nm。其 Driver Torque Estimator 天然使用低通滤波抑制振荡成分，验证了符号一致性作为分类信号的有效性。

**F4：过零率（Hz）** —— 这是扭矩信号振荡频率的高效计算代理指标。道路干扰会以 8–20+ Hz 激励转向（产生 16–40 Hz 的 ZCR）；驾驶员输入通常低于 2–4 Hz（ZCR < 4 Hz）。该特征捕捉振荡频率，与只反映极性平衡的符号一致性形成互补。

*来源：*
- Cole 等，"Real-time characterisation of driver steering behaviour," *Vehicle System Dynamics*, 56(10), 2018 — 测得驾驶员主动转向频率为 0.2–2 Hz，而道路激励高于 2 Hz。
- Giacomin & Woo，"A study of the human ability to detect road surface type on the basis of steering wheel vibration feedback," *Proc. IMechE Part D*, 219(11), 2005 — 发现路面引起的方向盘振动集中在 10–60 Hz，其中 26–35 Hz 最具诊断性。

**F5：扭矩峰度** —— 超额峰度衡量扭矩分布的“尾部厚度”。单个尖锐冲击（撞击坑洼）会产生高峰度（>>3，尖峰厚尾）。平滑的驾驶员输入大致呈高斯或次高斯分布（峰度 ≈ 3）。持续的粗糙路面会产生介于两者之间的中等峰度。

*来源：*
- 这是振动分析和结构健康监测中使用的标准统计属性。此处类比道路激励文献中已经确立的脉冲型与平滑型信号形态。

**F6：纵向冲击** —— 短事件（<0.4 s）期间同时出现纵向加速度尖峰（|a_ego| > 1.5 m/s²），是垂直路面冲击的强指标。主动转向通常不会产生明显纵向加速度，除非同时制动；这种情况可以通过持续时间加以区分。

*来源：*
- Mednis 等，"Real Time Pothole Detection Using Android Smartphones with Accelerometers," *IEEE DCOSS*, 2011 — 使用垂直/纵向加速度阈值验证了基于加速度计的坑洼检测。
- Eriksson 等，"The Pothole Patrol," *MobiSys*, 2008 — 展示了用于道路异常检测的车速相关垂直加速度滤波。

**F7：扭矩-转角相位关系** —— 主动转向时，驾驶员*先*施加扭矩，方向盘随后响应转动（扭矩领先转角）。道路干扰时，外力*先*使方向盘转动，驾驶员随后感受到产生的扭矩（转角领先扭矩）。扭矩变化率和转角变化率在零延迟处的互相关，可将这种因果方向表示为连续的 [-1, 1] 数值。

*来源：*
- EPAS 的基本控制原理。以下文献对其进行了间接验证：
  - Chevrel、Mars 等，"Modelling human control of steering for the design of advanced driver assistance systems," *Annual Reviews in Control*, 47:249-261, 2019 — 赛博驾驶员模型表明，在主动控制中扭矩是航向变化的*原因*。
  - GM Patent US 8,954,235 (2015) — 比较实测扭矩与模型预期扭矩来检测车道居中的驾驶员接管，隐含依赖扭矩→转角的因果关系。

### 第 3 级 — 上下文门控（10–20 个样本 / 100–200 ms）

**F8：扭矩-横向加速度相关性** —— **这是原分类器缺失的影响最大的单一特征。** 驾驶员主动转向时，施加的扭矩会引起相应的横向加速度变化，两者紧密耦合（Pearson r > 0.6），且扭矩领先 50–200 ms。道路干扰时，扭矩会出现尖峰，*但不会*产生相称的横向加速度变化（r ≈ 0），因为干扰在改变车辆横向轨迹之前就被悬架和转向系统吸收了。

该特征之所以有效，是因为它利用了车辆物理规律，而不仅仅是扭矩信号的形状。坑洼可能产生在幅值甚至符号一致性上都“看起来”像主动操作的扭矩，但不会产生相关的横向加速度。

*来源：*
- comma.ai torqued lateral control 文档——确认了生产级 ADAS 控制所使用的扭矩-横向加速度基本关系。
- Hyundai Patent US2019/0077447 — 将驾驶员扭矩建模为与横向加速度成正比，用于 ADAS 冲突检测。
- Zhou 等，"Driver Steering Intention Prediction for Human-Machine Shared Systems of Intelligent Vehicles Based on CNN-GRU Network," *Sensors*, 25(10):3224, MDPI, 2025 — 特征重要性分析将转向扭矩和横向加速度列为意图预测中最具区分性的两个输入。

**F9：频率能量比（低频段 / 高频段）** —— 最强的*频谱*区分指标。驾驶员转向内容位于 0.5–3 Hz；道路干扰内容集中在 5–40 Hz。在短窗口内计算两个频段的 RMS 能量比，可以用一个标量直接衡量信号功率所在的频段。为保证数值稳定性，使用 SOS（二阶节）形式的二阶 Butterworth IIR 带通滤波器实现。

在 100 Hz 采样率（Nyquist = 50 Hz）下，可以完整区分驾驶员频段和道路频段。对于短于 200 ms（约 20 个样本）的事件，带通滤波器的边缘效应占主导，因此不计算该特征。

*来源：*
- Cole 等（2018）——驾驶员频段为 0.2–2 Hz。
- Giacomin & Woo（2005）——路面振动频段为 10–60 Hz。
- Giacomin 等，"Effect of steering wheel acceleration frequency distribution on detection of road type," *Ingeniería Mecánica, Tecnología y Desarrollo*, 2013 — 确认频率分布是区分路面类型的主要依据。
- 关于动态驾驶模拟器中转向反馈的 arXiv 论文（2024）——确认道路激励频段为 10–30 Hz，并指出低于 10 Hz 的成分会被感知为外部转向干预。
- Giacomin & Onesti，"Frequency weighting for the evaluation of steering wheel rotational vibration," *International Journal of Industrial Ergonomics*, 34(2):89-97, 2004 — 建立了人类对方向盘振动的频率相关敏感性。

**F10：车速自适应持续时间阈值** —— 替代原有固定的 0.15 s 和“高速短事件”（车速 > 20 m/s 且持续时间 < 0.4 s）规则。其物理关系很直接：穿越坑洼的时间等于坑洼长度除以车速（T = L/V）。在 10 m/s 下通过 2.5 m 坑洼需要 250 ms，而在 30 m/s 下只需要 83 ms。使用可校准的最大坑洼长度（默认 2.5 m），阈值可随车速连续调整，而不是使用二元截断。

*来源：*
- SAE Paper 2015-01-0637, "Simulation of Vehicle Pothole Test and Techniques Used" — 确认坑洼尺寸和车速是影响冲击严重程度的两个主要因素，持续时间按 L/V 缩放。
- ISO 8608:2016, "Mechanical vibration — Road surface profiles — Reporting of measured data" — 通过参考空间频率处的功率谱密度对路面粗糙度分类，并确立时间激励频率 = 空间频率 × 车速。
- Bridgelall & Tolliver，"Characterisation of road bumps using smartphones," *European Transport Research Review*, 8:13, 2016 — 展示了车速相关的道路异常检测阈值。

**F11：横向加速度残差** —— 事件期间实际横向加速度与期望横向加速度之间的最大绝对偏差。驾驶员接管自动驾驶并将车辆驶离规划路径时，会产生较大的残差（>0.5 m/s²）。道路干扰会产生扭矩尖峰，但车辆轨迹仍接近规划路径（残差较小），因为悬架会在垂直/横向扰动明显改变车辆横向动力学之前将其吸收。

*来源：*
- Toyota Patent EP3659878B1 (Mitsumoto, 2021) — 使用实际偏航率与模型预期偏航率之间的偏差检测外部横向干扰，这是横向加速度残差的旋转等价形式。
- Euro NCAP Assessment Protocol SA v10.4.1 (2024) — 规定施加不超过 3.5 Nm 即可覆盖车道保持辅助，说明驾驶员接管会产生可测量的路径偏差。

---

## 为什么不能只使用扭矩阈值？

原分类器和大多数生产级 ADAS 系统使用简单的扭矩或扭矩变化率阈值，但这会因以下三个原因失效：

1. **鹅卵石和粗糙路面会产生持续的 ±1–3 Nm 扭矩振荡** —— 即使无人触碰方向盘，也远高于 0.6–1.0 Nm 的手握检测阈值（Moreillon 等，JTEKT 2020）。只看幅值的分类器会把每条鹅卵石路都标记为驾驶员干预。

2. **紧急避让会产生 >50 Nm/s 的扭矩变化率和 <0.5 s 的持续时间**，在变化率和持续时间上都与机械干扰特征重叠。只有跨信号特征（扭矩-横向加速度相关性、扭矩-转角相位、横向加速度残差）才能解决这一歧义，因为紧急避让会*使车辆横向移动*，而坑洼不会。

3. **道路干扰幅值随车速增加而增大**，同时持续时间缩短。在高速下，坑洼冲击产生的峰值扭矩可能与轻微车道修正相当，但会压缩在 40 ms 内，并呈现振荡符号模式和高频能量。针对某一车速范围优化的固定阈值，在其他范围内会系统性地产生误判。

---

## 与车速相关的影响

道路干扰特征会通过以下三个相互耦合的机制随车速系统性变化：

**持续时间缩短** —— 穿越时间 T = L/V，因此同一个 1 m 坑洼在 10 m/s 下产生 100 ms 事件，在 30 m/s 下产生 33 ms 事件。

**峰值幅值增大** —— 冲击能量与 V² 成正比。低速时轮胎会顺应坑洼形状，使力在时间上分散；高速时轮胎在短暂接触期间表现得更刚性，使力集中为更尖锐的脉冲。两轮车研究表明，在 60 km/h 通过坑洼会出现“车轮跳起”（失去接地），而同一坑洼在 20 km/h 下只产生中等载荷。

**激励频率升高** —— 波长为 L 的固定空间不平整会产生 f = V/L 的时间激励。在 30 m/s 通过 0.5 m 的路面特征时，激励频率为 60 Hz，远高于驾驶员转向频段并进入车轮/悬架共振范围。因此，频率能量比（F9）在高速下具有更强的区分能力。

*来源：*
- ISO 8608:2016 — 时间频率 = 空间频率 × 车速的关系。
- Wang 等，"Influence of Road Excitation and Steering Wheel Input on Vehicle System Dynamic Responses," *Applied Sciences*, 2017 — 展示了道路激励对横向响应的车速相关耦合影响。

---

## 替代方案：随机森林分类器

级联启发式方法面向可解释性和实时生产部署。对于离线分析或潜在的生产升级，使用相同 11+ 个特征训练的随机森林，在车辆传感器分类任务中通常可达到 84–95% 的准确率。

实现中包含一个训练流程，使用 scikit-learn 的 `RandomForestClassifier`，配置 30 棵深度为 7 的树、平衡类别权重和 5 折分层交叉验证。训练后，特征重要性可以解释哪些信号权重最高，并用于反馈调整级联阈值。

*来源：*
- Das、Khan & Ahmed，"Deep Learning Approach for Detecting Lane Change Maneuvers Using SHRP2 Naturalistic Driving Data," *Transportation Research Record*, 2023 — XGBoost + ResNet-18 在自然驾驶数据上达到 98.8% 召回率和 95% 准确率。
- Zhou 等（MDPI *Sensors*，2025）——CNN-GRU 在驾驶员转向意图预测中的 RMSE 相比 BP 下降约 32%、相比 LSTM 下降约 21%、相比单独 CNN 下降约 25%。特征重要性为：转向扭矩 > 转向角 > 车速 > 横向加速度。
- 轻量级 CAN 总线驾驶员行为分类（*Sensors/PMC*，2020）——在 NVIDIA Jetson Nano 上部署了逐通道卷积 + 增强 RNN，实现实时推理。

---

## 相关标准

| 标准 | 相关性 |
|---|---|
| **ISO 8608:2016** | 通过参考空间频率处的 PSD 对路面轮廓进行分类（A-H 级）。为车速相关的激励频率和幅值缩放提供物理依据。 |
| **ISO 11270:2014** | 转向手感 —— 定义转向系统评估的测试流程，包括扭矩特性。 |
| **UN ECE R79** | 车道保持辅助限制：最大横向加速度 3 m/s²，最大横向加加速度 5 m/s³。脱手警告在 15 s → 30 s 后升级并停用。 |
| **Euro NCAP SA Protocol v10.4.1 (2024)** | 驾驶员接管要求：施加不超过 3.5 Nm 即可覆盖车道保持辅助。 |
| **SAE J3016** | 驾驶自动化等级；定义何时必须保留驾驶员接管权限。 |

---

## 训练和验证数据集

**commaSteeringControl**（comma.ai，Hugging Face）—— 包含 275+ 个车型、约 12,500 小时的 openpilot 开启状态驾驶数据。包含 `steeringPressed`（二值接管标记）、`steer`（归一化扭矩）、`steeringAngleDeg`、`vEgo`、`aEgo`、`latAccelDesired`、`latAccelSteeringAngle`。`steeringPressed` 事件可提供候选标签，但无法区分驾驶员意图和道路干扰，这正是本项目要解决的分类空白。

**comma2k19**（Schafer 等，arXiv:1812.05752）—— 包含 Honda Civic 和 Toyota RAV4 的 33+ 小时 CAN 总线数据及 9 轴 IMU 数据。

**SHRP2 Naturalistic Driving Study**（FHWA/Virginia Tech）—— 大规模自然驾驶数据集，包含 CAN 信号；已用于多项变道和驾驶员行为研究，但需要签署数据使用协议。

**智能手机坑洼数据集** —— 有多个用于基于加速度计检测道路异常的公开数据集（Kaggle 及不同研究团队），与 GPS 坐标交叉引用后可以提供路面真实标签。

*注意：* 没有任何单一公开数据集同时提供转向扭矩数据和带标签的路面冲击事件。构建真实标签需要结合 `a_ego` 的垂直加速度阈值、`steeringPressed` 标记、与 GPS 关联的道路质量数据库，以及对不明确案例进行选择性人工视频复核。

---

## 主要文献参考

### 驾驶员转向动力学和带宽
1. Cole, D.J. et al., "Real-time characterisation of driver steering behaviour," *Vehicle System Dynamics*, 56(10), 2018. — 驾驶员转向频率为 0.2–2 Hz。
2. Pick, A.J. & Cole, D.J., "Neuromuscular dynamics in the driver–vehicle system," *Vehicle System Dynamics*, 44(sup1):511-522, 2006. — 神经肌肉低通滤波限制扭矩变化率。
3. Timings, J.P. & Cole, D.J., "A review of human sensory dynamics for application to models of driver steering and speed control," *Biological Cybernetics*, 110(2-3), 2016. — 驾驶员反馈带宽约为 1–2 Hz。
4. Chevrel, P., Mars, F. et al., "Modelling human control of steering for the design of advanced driver assistance systems," *Annual Reviews in Control*, 47:249-261, 2019. — 具有视觉预判和补偿控制的赛博驾驶员模型。

### 路面振动和转向干扰
5. Giacomin, J. & Woo, Y.J., "A study of the human ability to detect road surface type on the basis of steering wheel vibration feedback," *Proc. IMechE Part D*, 219(11), 2005. — 路面振动位于 10–60 Hz，其中 26–35 Hz 最具诊断性。
6. Giacomin, J. et al., "Effect of steering wheel acceleration frequency distribution on detection of road type," *Ingeniería Mecánica, Tecnología y Desarrollo*, 2013. — 频率分布是识别路面类型的主要依据。
7. Giacomin, J. & Onesti, L., "Frequency weighting for the evaluation of steering wheel rotational vibration," *International Journal of Industrial Ergonomics*, 34(2):89-97, 2004. — 人类对转向振动的敏感性与频率有关。

### EPAS 手握检测和干扰抑制
8. Moreillon, L. et al., "Hands On/Off Detection Based on EPS Sensors," *JTEKT Engineering Journal*, No. 1017E, 2020. — 鹅卵石路面上存在 ±1–2 Nm 振荡扭矩；Driver Torque Estimator 架构。
9. Dornhege, C., Nolden, P. & Mayer, R., "Steering Torque Disturbance Rejection," *SAE Int. J. Veh. Dyn., Stab., and NVH*, 1(2):165-172, 2017. — 用于干扰识别的双齿条力模型。
10. Yamamoto, K. et al., "Driver torque estimation in Electric Power Steering system using an H∞/H2 Proportional Integral Observer," *IEEE CDC*, 2015. — 观测器带宽构成隐含的一致性时间尺度。

### 驾驶员意图和干预检测
11. Schinkel, W. et al., "Driver Intervention Detection via Real-Time Transfer Function Estimation," *IEEE Trans. ITS*, 2021. — 传递函数方法；可靠分类至少需要 0.2 s。
12. Zhou, Y. et al., "Driver Steering Intention Prediction for Human-Machine Shared Systems of Intelligent Vehicles Based on CNN-GRU Network," *Sensors*, 25(10):3224, MDPI, 2025. — CNN-GRU 相比 LSTM/Transformer 的 RMSE 降幅；特征重要性排序。
13. Das, A., Khan, M.N. & Ahmed, M.M., "Deep Learning Approach for Detecting Lane Change Maneuvers Using SHRP2 Naturalistic Driving Data," *Transportation Research Record*, 2023. — XGBoost + ResNet-18，召回率 98.8%。

### 道路异常检测
14. Mednis, A. et al., "Real Time Pothole Detection Using Android Smartphones with Accelerometers," *IEEE DCOSS*, 2011. — STDEV(Z) 算法，真正例率约 90%。
15. Bridgelall, R. & Tolliver, D., "Characterisation of road bumps using smartphones," *European Transport Research Review*, 8:13, 2016. — 车速相关的异常检测阈值。

### 车辆动力学和干扰建模
16. Abe, M. et al., "A yaw-moment control method based on a vehicle's lateral jerk information," *Vehicle System Dynamics*, 52(10), 2014. — 将横向加加速度作为主动操作指标。
17. ISO 8608:2016, "Mechanical vibration — Road surface profiles — Reporting of measured data."
18. Toyota Patent EP3659878B1 (Mitsumoto, 2021) — 使用偏航率残差检测外部干扰。
19. GM Patent US 8,954,235 (2015) — 改进自动车道居中期间的转向接管检测。

### 标准
20. UN ECE R79 — 车道保持辅助横向加速度/加加速度限制及脱手检测时序。
21. Euro NCAP Assessment Protocol SA v10.4.1 (2024) — 接管力要求。
22. ISO 11270:2014 — 转向手感测试流程。
