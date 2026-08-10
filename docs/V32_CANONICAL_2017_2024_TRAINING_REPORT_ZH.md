# V3.2 Canonical 2017-2024 训练与验收报告

更新时间：2026-08-10（Asia/Shanghai）

本文只记录 `/data/datasets/sentinel_translate_v32_2017_2024` 上已经完成的 canonical
训练链。旧 463 样本协议、早期 source-anchor 实验和后续 NC-OPC 消融都与本报告分开，
不得混用 checkpoint、指标或选模结论。

## 1. 最终状态

- canonical 训练链已经完成：physical 7k、codec 20k、detail 9k、flow 6k、
  phase transport 5k。
- 141 样本 `validation_temporal` 上，Physical gate 和 SAR visual gate 通过。
- Optical visual 的 RMSE、DISTS、Edge F1、PSD、越界率和逐场景风险通过；LPIPS
  只改善 `3.7111%`，未达到 `5%`，所以 Optical visual、visual 和 joint gate 失败。
- selection 只发布 `best_physical.pt`，没有 `best_visual.pt` 或 `best_joint.pt`。
- 三个封闭 test split 尚未运行。当前不能把 visual 结果写成最终 test 或 SOTA 结论。

权威文件：

- validation：`reports_v32_canonical_2017_2024/final_validation.json`
- selection：`checkpoints_v32_canonical_2017_2024/selection.json`
- protocol hash：`f72deee58e7c421bd6af9d96164a272717564f94b7c227e4b38fa4e915f61606`

GitHub 快照：

- [`final validation 141`](results/v32_canonical_2017_2024_final_validation_141.json)
- [`selection`](results/v32_canonical_2017_2024_selection.json)
- [`canonical panel`](assets/v32_canonical_2017_2024_final_000_sar2opt.png)

## 2. 数据集与防泄漏协议

### 2.1 数据范围

| 项目 | 数量与范围 |
| --- | --- |
| 原始训练候选 | 2,050 pair，2017-2022 |
| 接受的训练 pair | 1,947；103 pair 因有效 patch 不足被拒绝 |
| 训练 patch | 31,152，固定 `256 x 256`，每 pair 16 个 |
| 高频合格 patch | 14,622，registration audited |
| validation_temporal | 141 pair，全部 2023，固定中心 `256 x 256` crop |
| test_spatial | 39 pair，2023，封闭 |
| test_temporal | 131 pair，2024，封闭 |
| test_joint | 62 pair，2024，封闭 |

输入统一到 10 m 网格。Sentinel-2 使用 10 个 surface-reflectance 通道，顺序为
`blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22`；
Sentinel-1 使用 `VV, VH` 两个 dB backscatter 通道。

validation mask 只接受 SCL `2/4/5/6/7`。单位、通道顺序、中心 crop、mask 和 manifest
SHA-256 都写入 `manifests/validation_protocol.json`，并通过 protocol hash 绑定到
format-v4 checkpoint。旧 463 样本协议与本协议不兼容。

### 2.2 高频审计

高频监督必须同时满足：

- 年份属于 2017-2022 train；
- `delta_days <= 1`，其中 `0/1` 天权重为 `1.0/0.25`；更长时间差对 residual
  参数严格零梯度；
- local-structure NCC registration audit 的估计位移不超过 `0.5 px`；
- 有效比例至少 `0.8`；云和阴影比例不超过 `0.2`。

registration audit 使用局部结构梯度、`[-2, 2]` 像素搜索、NCC 至少 `0.10`，且相对
零位移提升至少 `0.05` 才报告非零位移。审计版本和阈值保存在
`hf_eligibility.json`。

## 3. 模型设置

共享 physical 编码器为 12 层 Transformer，hidden 768、12 heads，方向 residual
adapter rank 64。方向专用 decoder/radiometry head 分别输出 10 通道 Optical reflectance
和 2 通道 SAR dB backscatter。

高频组件为：

- shared residual codec：4 倍压缩、16-channel standardized latent，模态 I/O 独立；
- multiscale detail head：从完整 FPN 预测确定性 residual；
- Residual-DiT：hidden 512、8 层、8 heads；
- phase transport：hidden 128、三频带 gain caps `[0.5, 0.25, 0.1]`，启用
  cyclic-shift null calibration；
- phase Optical residual state：two-level Haar packet，48 channels；SAR 保留兼容的
  16-channel residual path。

canonical 模型总参数为 `195,533,271`。各阶段实际开放的参数如下：

| Stage | 可训练参数 | 核心目标 |
| --- | ---: | --- |
| physical | 109,238,278 | 低误差、光谱/角度、SAR 辐射与双向对齐 |
| codec | 1,073,493 | residual 重建、梯度、频谱和 SAR 局部统计 |
| detail | 895,755 | 可预测的多尺度高频 residual |
| flow | 36,790,705 | conditional residual rectified flow 的 velocity/endpoint/rollout |
| phase_transport | 652,166 | 受保护 anchor 外的可观测 Laplacian phase correction |

## 4. 实际训练过程

所有阶段使用 8x A100、BF16、EMA、4 worker/rank、persistent workers 和 prefetch 2。
physical 开启 activation checkpointing 和 PCGrad。表中 steps 是实际终点，不是配置上限。

| Stage | 实际 steps | 全局有效 batch | LR | 墙钟 |
| --- | ---: | ---: | --- | ---: |
| physical | 7,000 | 64 | encoder `2e-6`、main `1e-5`、adapter `1e-4` | 4:05:10 |
| codec | 20,000 | 64 | `1e-4` | 0:53:43 |
| detail | 9,000 | 64 | `1e-4` | 1:01:34 |
| flow | 6,000 | 64 | `1e-4` | 1:58:24 |
| phase_transport | 5,000 | 16 | `1e-4` | 0:35:57 |

正式 stage 墙钟约 8 小时 35 分；加上阶段间 calibration 和 artifact 检查，主链从
2026-08-09 23:17:54 到 2026-08-10 08:13:13，共约 8 小时 55 分。

阶段事实：

1. physical 在 step 4k 产生最佳候选，训练到 7k 后早停；后续所有高频阶段冻结它。
2. codec 在 20k 通过重建 gate：Optical MAE `0.002582`，SAR MAE `0.523176 dB`。
3. raw detail 在 9k 没通过全图 gate；confidence calibration 只在可信区域发布，Optical
   coverage 约 `0.216%`，因此不能把 standalone detail 写成已解决全图高频。
4. flow 在 6k 早停。最终 phase 配置将 Optical stochastic innovation 和 correction
   release 都设为零；SAR residual path 保持发布。
5. phase transport 训练 5k，最终 Optical visual 的改善来自确定性 anchor 和 physical
   phase detail，不来自随机彩色纹理。

## 5. 最终 141 样本结果

### 5.1 Physical

| 指标 | 结果 | 门槛 | 状态 |
| --- | ---: | ---: | --- |
| SAR -> Optical RMSE | `0.0326609` | `<= 0.03909` | 通过 |
| SAR -> Optical SAM | `5.58075 deg` | `<= 5.716 deg` | 通过 |
| Optical -> SAR RMSE | `4.50149 dB` | `<= 5.0 dB` | 通过 |
| Optical -> SAR signed bias | `0.01087 dB` | `<= 0.5 dB` | 通过 |

RGB-only physical RMSE 为 `0.02663915`。Physical 是论文中 RMSE、SAM 和辐射指标的
主输出，不能用 best-of-K visual 样本替代。

### 5.2 Optical visual

| 指标 | Physical | Visual | 状态 |
| --- | ---: | ---: | --- |
| RGB RMSE | `0.02663915` | `0.02695340` | ratio `1.01180`，通过 1.05 门槛 |
| LPIPS | `0.199744` | `0.192331` | 改善 `3.7111%`，未到 5% |
| DISTS | `0.204725` | `0.190849` | 改善 `6.7778%`，通过 |
| Edge F1 | `0.439870` | `0.523919` | 改善 |
| PSD distance | `0.00611677` | `0.00604207` | 改善 |
| pre-projection violation | - | `0.016716%` | 通过 0.1% 门槛 |

`94.33%` 场景 Edge 改善，`87.23%` 场景 DISTS 改善，`97.16%` 至少一项改善；
`7.09%` 场景的 RGB RMSE 退化超过 5%，低于 10% 上限。唯一 aggregate 硬失败项是
LPIPS，所以不能发布 Optical visual 或 joint checkpoint。

### 5.3 SAR visual

| 指标 | Physical/mean | Visual | 趋势 |
| --- | ---: | ---: | --- |
| signed bias | `0.01087 dB` | `0.01635 dB` | 均通过 |
| PSD distance | `0.65475` | `0.28154` | 改善 |
| ENL error | `0.18116` | `0.07391` | 改善 |
| histogram distance | `0.004823` | `0.001350` | 改善 |
| P01 error | `4.1113 dB` | `1.6178 dB` | 改善 |
| P99 error | `5.3851 dB` | `2.9792 dB` | 改善 |

SAR visual gate 通过。`scene_abs_bias` 约 `0.765 dB` 只作为离散场景诊断；正式 gate
按协议使用跨场景 signed mean 后取绝对值，避免把正负偏差错误地当作同号系统偏差。

## 6. 收尾故障与修复记录

主训练权重没有损坏。原 launcher 在训练后收尾阶段曾失败，根因有三项：

1. calibration 仍调用旧 `deterministic_detail/sample_residual` API，把 SAR 的 16-channel
   状态送入 Optical 48-channel flow；现已统一走 `visual_detail/sample_visual_residual`。
2. Optical calibration 原先计算 `sqrt(mean(scene MSE))`，SAR 原先计算 mean scene-abs
   bias，与 evaluator 口径不一致；现已改为逐场景 RMSE 平均，以及 signed scene mean
   聚合后取绝对值。
3. selection 直接 `torch.save` 到 `best_physical.pt` 软链接可能覆盖原始 step checkpoint；
   现已用同目录临时普通文件原子 replace。原始 physical step-4k SHA-256 前后保持
   `62d6b9140c407ab7d60429262474376939c102c1b3e65af7e046f37dab27f729`。

修复后 calibration 选择 Optical/SAR alpha 均为 `1.0`，正式评估和 selection 已成功完成。

## 7. Artifact 与复现

| Artifact | 路径 |
| --- | --- |
| 数据根 | `/data/datasets/sentinel_translate_v32_2017_2024` |
| canonical config | `configs/canonical_2017_2024_*.yaml` |
| launcher | `scripts/launch_canonical_2017_2024_8gpu.sh` |
| stage checkpoints | `checkpoints_v32_canonical_2017_2024/{physical,codec,detail,flow,phase_transport}` |
| calibrated checkpoint | `checkpoints_v32_canonical_2017_2024/final_calibrated.pt` |
| selected physical | `checkpoints_v32_canonical_2017_2024/best_physical.pt` |
| final validation | `reports_v32_canonical_2017_2024/final_validation.json` |
| panels | `reports_v32_canonical_2017_2024/final_validation_panels` |

重新评估 validation：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m sentinel_v3.cli \
  --config configs/canonical_2017_2024_phase_transport.yaml \
  evaluate \
  --checkpoint checkpoints_v32_canonical_2017_2024/final_calibrated.pt \
  --split validation_temporal \
  --output reports_v32_canonical_2017_2024/final_validation.json
```

只有 `best_joint.pt` 存在时才允许运行三个 test split。当前该文件不存在，因此不要运行
closed-test 命令。

## 8. 结论

本轮 canonical 训练已经完成了低误差 physical 和 SAR 高频统计恢复，也证明了 Optical
确定性边缘增强可以在很小 RMSE 代价下改善 DISTS、Edge 和 PSD。它尚未完成 Optical
感知门槛；当前发布物只能是 `best_physical.pt`。后续 NC-OPC 等实验必须继续使用同一
validation protocol，并且只有在 validation 选模完成后才可解封 test。
