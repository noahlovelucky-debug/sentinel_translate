# V3.2 Canonical 2017-2024 训练说明

## 状态摘要

- 原始数据源：`/data/data_disk/data_dir`。
- 规范化数据集目标目录：`/data/datasets/sentinel_translate_v32_2017_2024`。
- codec 修复提交：`c0639ba Stabilize SAR codec training`。启动器与报告基线提交：`17d225d`，不表示当前 HEAD。
- SAR codec 的常量局部方差数学奇点已经修复；全套测试为 `207 passed`。
- 8 卡 smoke 已运行至 320 step，越过此前约 220 step 的故障点并以 exit code 0 结束。
- 正式 canonical 链已于 `2026-08-09 23:15:30+08:00` 启动，tmux 会话为 `sentinel_v32_canonical_full`；launcher 初始 PID 为 `325000`，physical torchrun 初始 PID 为 `325517`。
- `23:17:54+08:00` 已切入 physical，8 个 rank 已确认；兼容加载为 520 tensors / new 177，启动验收 GPU 活动为 37-47%，无 OOM。
- external temporal prior PID `214730` 已恢复并与 physical/codec 并行。最终训练效果尚未产生；按用户要求，本报告不持续监测训练。

`scripts/launch_canonical_2017_2024_8gpu.sh` 会先检查训练索引、manifest 和已完成 registration audit 的 HF sidecar。temporal prior 索引不存在时，launcher 会启动或接入已有的 prior 预计算，使其与 bootstrap、physical 和 codec 并行；进入 detail 前强制等待 prior 完成并严格检查索引存在。

## 数据集与协议

原始 TIFF 不复制到训练目录。构建产物是排序的 manifest、协议 sidecar、HF eligibility sidecar 和规范化 patch shard。每个 patch 为 `256x256`，每个 SAR-S2 pair 最多 16 个确定性 patch。

| 项目 | 定义 |
| --- | --- |
| S2 输入 | 10 通道，严格顺序 `blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22` |
| Optical 物理输出 | RGB 3 通道 |
| SAR 输入/输出 | `vv, vh` 两通道 |
| 训练有效性 | joint-valid、clear fraction、网格/CRS/shape/transform 约束均已在 manifest 构建阶段检查 |
| HF 审计 | registration audit 后 14,622 个训练 patch 可作为高频候选 |

### 固定时空切分

切分不使用随机 patch split。`row==5` 或 `col==5` 为 buffer；其后 `row==6` 或 `col==6` 为 spatial holdout，其余为 core。

| 集合 | pair 数与时间 | 用途 |
| --- | --- | --- |
| train 候选 | 2,050：2017 306、2018 299、2019 474、2020 603、2021 230、2022 138 | core 训练候选 |
| train 通过 | 1,947 pair，31,152 patch | 有效 patch 检查通过的训练集 |
| train 拒绝 | 103 pair | `insufficient_valid` |
| validation_temporal | 141 pair，全部 2023 | 开发期唯一验证协议 |
| test_spatial | 39 pair，全部 2023 | 最终选模后封闭测试 |
| test_temporal | 131 pair，全部 2024 | 最终选模后封闭测试 |
| test_joint | 62 pair，全部 2024 | 最终选模后封闭测试 |
| buffer | 1,844 pair | 不训练 |
| unused_spatial | 1,058 pair | 不训练 |

三个 test split 在最终 checkpoint 选定前不得运行或参与超参数选择。新 `validation_temporal` protocol 的 manifest hash、样本数、中心裁剪、mask、单位和通道顺序绑定在 sidecar 与 format-v4 checkpoint 中；它与旧 463-sample protocol 不可混用，不得共享 report、gate、checkpoint resume 或 selection 结论。

## 模型与训练目标

共享 physical 主干为 12 层、hidden 768、12 heads 的编码器，配有 rank-64 的方向 adapter 和方向输出头。codec 使用共享 trunk、模态 I/O，执行 4x 下采样的 16-channel latent 表示。detail 为多尺度确定性高频头；ResidualDiT 为 hidden 512、depth 8、heads 8。

最终 phase 配置使用 48-channel two-level Haar packet residual state，以及 hidden 128 的 phase transport head。其 gain caps 为 `[0.5, 0.25, 0.1]`，并启用 null-calibrated 约束。

| Stage | 训练目标与主要损失 | 可训练参数 |
| --- | --- | ---: |
| physical | 跨模态物理基底的低误差确定性预测；重建、光谱/角度、SAR 偏差和对齐约束 | 109,238,278 |
| codec | 固定 physical 残差的紧凑编解码；masked Charbonnier、梯度、频谱、局部频谱，SAR 另含稳定局部 speckle/variance 项 | 1,073,493 |
| detail | source-supported 的确定性多尺度高频残差；重建、梯度、边缘、结构和稀疏约束 | 895,755 |
| flow | 条件 rectified/residual flow：从条件起点运输 latent residual，拟合速度、端点和 rollout 高频/失真约束 | 36,790,705 |
| phase_transport | 在受保护 source-aware anchor 上学习可观测的目标域 Laplacian detail；高频、phase alignment、gain utility、失真和周期性 perceptual 约束 | 652,166 |

flow 不是从纯噪声直接生成整幅图像。physical 基底先固定，codec/detail/flow/phase 仅对其残差高频建模；flow 通过条件化 latent residual 的 rectified transport 生成随机但受结构条件约束的部分。phase 进一步限制为受保护 anchor 之外、可观测且经 null calibration 支持的高频修正。

最终模型总参数为 **195,533,271**。上述 stage 参数数是该阶段优化器实际开放的参数量，而不是五者相加后的独立模型总量。

## 正式训练设置

所有正式阶段使用 8x A100、BF16、EMA、4 worker/rank、persistent worker、prefetch factor 2 和 activation checkpointing（physical 阶段启用）。除 phase 外，快速验证每 1,000 step；physical 的 full validation 为每 4,000 step，codec/detail/flow 为每 5,000 step；phase 的快速验证和保存每 500 step、full validation 在 5,000 step。physical/codec/detail/flow 的 early-stop patience 为 5，phase 为 10。

| Stage | Steps | 每卡 batch | Accumulation | 全局有效 batch | 学习率 |
| --- | ---: | ---: | ---: | ---: | --- |
| physical | 20,000 | 2 | 4 | 64 | encoder `2e-6`，physical `1e-5`，adapter `1e-4` |
| codec | 20,000 | 8 | 1 | 64 | `1e-4` |
| detail | 20,000 | 4 | 2 | 64 | `1e-4` |
| flow | 40,000 | 2 | 4 | 64 | `1e-4` |
| phase_transport | 5,000 | 1 | 2 | 16 | `1e-4` |

物理阶段还使用 `physical_alignment_every=4` 的无偏稀疏对齐近似和 PCGrad；flow 使用 2-step rollout；phase 使用 1-step rollout，并按配置进行周期性 perceptual 计算。EMA 为 `.999`，phase 为 `.99`。

## 质量门槛

下游阶段只能继承同一 validation protocol hash 且已通过 physical gate 的 format-v4 checkpoint。checkpoint selection 也会验证 report 和 checkpoint 的 protocol hash 一致。

**Physical gate** 必须同时满足：

- SAR-to-Optical RMSE `<= 0.03909`。
- SAR-to-Optical SAM `<= 5.716` 度。
- Optical-to-SAR RMSE `<= 5.0 dB`。
- Optical-to-SAR physical absolute bias `<= 0.5 dB`。

**Optical visual gate** 必须同时满足：

- visual RGB RMSE 不高于 physical RGB RMSE 的 `1.05x`。
- LPIPS improvement 与 DISTS improvement 均 `>= 0.05`。
- visual edge F1 高于 physical edge F1，且 optical PSD distance 低于 physical。
- pre-projection violation `<= 0.001`。
- `scene_edge_or_dists_improved_fraction >= 0.70`。
- `scene_rgb_rmse_degraded_over_5pct_fraction <= 0.10`。

**SAR visual gate** 要求 visual absolute bias `<= 0.5 dB`，且 visual PSD distance、ENL error、histogram distance、P01 error 和 P99 error 都低于对应的 SAR mean baseline。joint gate 要求 physical、Optical visual 和 SAR visual 三者全部通过。

## 路径与执行链

| 类型 | 路径 |
| --- | --- |
| 数据根 | `/data/datasets/sentinel_translate_v32_2017_2024` |
| 配置 | `configs/canonical_2017_2024_{physical,codec,detail,flow,phase_transport}.yaml` |
| Launcher | `scripts/launch_canonical_2017_2024_8gpu.sh` |
| Checkpoint 根 | `/data/code/sentinel_translat/v3.2/checkpoints_v32_canonical_2017_2024` |
| 日志 | `/data/code/sentinel_translat/v3.2/checkpoints_v32_canonical_2017_2024/logs/*.log` |
| Report 根 | `/data/code/sentinel_translat/v3.2/reports_v32_canonical_2017_2024` |
| 关键中间件 | `best_physical.pt`、`best_codec.pt`、`best_detail_calibrated.pt`、`flow_anchor_calibrated.pt`、`final_calibrated.pt` |

launcher 支持同阶段 checkpoint resume。初始 physical 权重默认来自 `/data/code/sentinel_translat/v3.2/checkpoints_v32_temporal/best_physical.pt`，并使用 EMA 初始化。训练日志位于 checkpoint 根下的 `logs/`；不要把旧 protocol 的日志、报告或 checkpoint 放入这条选择链。

按需人工查看即可：

```bash
tmux attach -t sentinel_v32_canonical_full
tail -n 100 -F /data/code/sentinel_translat/v3.2/checkpoints_v32_canonical_2017_2024/logs/physical.log
```

## 时间预算与后续动作

| 工作 | 估计时长 |
| --- | --- |
| temporal prior 预计算 | 约 4-6 小时，可与 physical/codec 并行 |
| physical | 约 5-10 小时 |
| codec | 约 2-4 小时 |
| detail | 约 2-4 小时 |
| flow | 约 12-20 小时 |
| phase、校准与最终验证 | 约 3-6 小时 |
| 总计 | 正式 GPU 链墙钟约 24-44 小时，保守预算 24-48 小时 |

早停可能缩短实际时长。总预算指正式 GPU 链墙钟；temporal prior 的大部分时间由 physical/codec 覆盖，但若 prior 异常缓慢，detail 前的强制等待会增加墙钟时间。本轮正式链已经启动；完成后根据最终 protocol-bound validation 和封闭 test report 做一次性验收，不在本报告中预先声明任何最终指标或视觉效果。
