# 稀疏配准锚点图生图 V2：完整设计与可行性分析

## 1. 问题重新定义

模型不再假设固定四张历史影像。一次请求由以下信息组成：

- 一对早于查询时刻、在同一地面网格上配准的真实影像：`source_anchor` 和 `target_anchor`；两侧保留各自的真实采集日期；
- 1 到 N 张源模态影像，每张都有采集时间、有效掩码和传感器描述；
- 一个待生成的目标时刻 `t=0` 和目标传感器；
- 训练时才存在的该时刻真实目标影像。

同一接口覆盖三种应用：

| 场景 | 源观测 | 配准对 | 任务含义 |
| --- | --- | --- | --- |
| 标准图生图 | 目标时刻有一张源图 | 一对历史 SAR/Opt | `translation`，把当时的源图变成目标图 |
| 少历史预测 | 1 到 3 张源历史 | 一对历史 SAR/Opt | `forecast`，生成晚于所有源图的目标图 |
| 多时相生成 | 4 张以上源历史或查询图 | 一对历史 SAR/Opt | 利用轨迹估计目标时刻状态 |

SAR→Optical 与 Optical→SAR 使用同一结构、不同方向 checkpoint。传感器描述符决定通道的物理含义，因此核心不固定为 Sentinel 的 2/10 通道，但新传感器仍需要配准样本做适配和验证。

这里的配准对是历史参考，不是查询答案。如果唯一一对 SAR/RGB 就是当前待翻译的那一对，把其真实目标侧作为 `target_anchor` 会直接泄漏标签。合法的最小部署输入是“一对更早的配准锚点 + 一张当前或历史源图”。如果没有额外源观测，模型只能复制目标锚点，不能称为图生图。

## 2. 为什么必须有“配准对”

如果只有一张孤立 SAR 和一张待生成 Optical，模型既要学习传感器映射，又要猜场景状态，容易退化为训练集平均值。V2 把任务写成变化传输：

```text
源模态变化 = source_observation(t) - source_anchor
目标物理输出 = target_anchor + Transport(源模态变化, 时间, 传感器描述)
```

配准对定义了同一地点在两种传感器下的共同坐标系。历史/查询源图只需要告诉模型“相对锚点发生了什么变化”。这比直接从所有图重建整幅目标更容易学习，也更适合少样本迁移。

历史观测可以早于或晚于配准锚点，但必须不晚于查询时刻，且不能包含查询目标资产。模型用有符号时间和相对锚点时间描述轨迹，不强制序列等间隔。

## 3. 模型框架

核心实现位于 `src/sentinel_v3/paired_temporal_v2.py`。

### 3.1 描述符条件的传感器适配器

每个通道带有模态、中心波长或雷达频率、极化、原始 GSD、目标网格 GSD 和 PSF 描述。共享适配器把不同通道数投到共同特征空间；动态投影器按目标通道描述生成输出。

这不能自动实现任意传感器零样本迁移，但允许固定时序骨干，只微调输入输出适配器、变化载体和辐射标定层。

### 3.2 可变长度集合融合

输入张量使用 batch 内最大帧数 `Tmax`，`observation_present` 是唯一可用性依据。padding 值不会参与注意力。时间特征包含：

- 距目标时刻天数；
- 距源锚点天数；源/目标锚点的两个真实时间都做因果校验；
- 周期时间编码；
- 是否恰好存在目标时刻源图。

空间注意力在每个 latent 像素上从 1 到 N 张源图中选择证据。局部无有效观测时安全回退，不产生 NaN。

### 3.3 配对变化载体与 physical

模型有一条短的可观测变化载体：先计算源图相对源锚点的像素变化，再用时空注意力聚合，最后投影到目标通道。较深的 physical head 学习非线性跨模态变化。两者之和通过有符号有界更新叠加到真实目标锚点。

所有最终投影层零初始化，因此未训练模型对 1 张、少量或多张输入都严格输出 `target_anchor`。这既是安全基线，也是判断时序证据是否真正有用的对照。

不确定性同时考虑模型预测、可用帧数和最后观测距目标时刻的间隔。单帧 forecast 的不确定性应高于目标时刻存在源图的 translation。

### 3.4 可预测细节与随机纹理

```text
physical = target_anchor + bounded physical delta
visual_base = physical + released deterministic detail
visual = visual_base + released conditional residual bridge
```

确定性细节只监督 `highpass(target - stopgrad(physical))`，负责可从源图定位的道路、屋顶、田块边缘。随机分支只学习剩余的零局部均值残差，不能改变 4×4 块的辐射均值。

随机分支是条件残差 flow/bridge，不是从纯噪声重新生成整幅图。起点分布由 physical 和配准上下文预测，速度场把它运输到 residual codec latent。这个分支用于条件上不可唯一确定的 Optical 细纹理、色彩微差及 SAR speckle/强散射尾部。

detail 和 texture 的释放系数初始为零。只有验证集固定 seed 下满足 RMSE 预算并改善高频指标，才写入 checkpoint；禁止 best-of-K。

## 4. 数据与防泄漏

新索引必须满足：

- 配准 anchor pair 的 SAR/Opt 日期差不超过 1 天；
- query、anchor、observations 全部属于同一 split、tile、orbit 和空间网格；同一 split 内允许不超过 180 天的跨年历史，自然年本身不是泄漏边界；
- 所有源观测日期 `<= query`，时间窗不超过 180 天；
- query target 的真实资产不能作为任何输入；
- 每个 index 只包含一个方向；
- translation 与 forecast 按实际时间推导，不靠人工标签覆盖；
- validation/test 的目标资产不进入训练或预训练。

一个 query 可在索引中配最多 3 个不同历史锚点，形成多个训练样本；一次前向仍只使用一对锚点，避免把多锚点堆叠成另一个固定输入假设。训练 split 内允许 180 天内跨自然年取历史，但不能跨 split。

数据层按样本保留完整 1..N 序列，batch collate 才 padding。训练先按 `0.40/0.35/0.25` 抽取 1 帧、2–3 帧和 4+ 帧 regime，再单独丢弃 query-time 源图，使同一条丰富序列同时产生 translation 和 forecast 条件。有效 loss mask 每步根据保留后的观测重新计算为 `target ∩ source_anchor ∩ target_anchor ∩ OR(retained observations)`。

确定性高频只使用有局部结构 NCC 证据支持配准的 translation crop。估计位移超过 0.5 px 或缺少足够相关结构都拒绝，不把“返回零位移但没有证据”误当成对齐。同日高频权重为 1.0，前一日为 0.25；forecast 或 query 源图在 dropout 后被移除时权重精确为 0。flow 可以在 forecast 上学习条件残差分布，但 deterministic detail 不逐像素猜未来纹理。

## 5. 训练闭环

训练分四阶段，每个方向独立进行：

1. `physical`：训练适配器、集合融合、变化载体和 physical head；初始对照是复制真实目标锚点。
2. `detail`：冻结 physical，仅训练确定性高频；目标是物理残差的多尺度高频。
3. `flow`：完整启用已训练 detail 作为基底，冻结 physical/detail，训练 residual codec 和条件 bridge。
4. `balance`：physical 和 detail head 始终冻结，小学习率调节 flow、codec 与两个释放幅度；验证集再做硬门槛搜索，避免 forecast 的像素 loss 反向污染确定性细节。

验证从每条真实序列构造确定性子集，而不是只按原始帧数贴标签：translation 子集必须保留 query 源图，forecast 子集必须删除 query 源图，且 padding 永不复活。所有可实现的以下六格分别报告：

```text
translation × {1帧, 2-3帧, >=4帧}
forecast    × {1帧, 2-3帧, >=4帧}
```

不能只报告汇总均值。最困难的 `forecast × 1帧` 是产品能力下限；`translation × 1帧` 是最接近普通图生图的主基线。

## 6. 完成标准

第一层是可行性门槛：

- 64 patch 固定样本能过拟合，所有阶段梯度有限；
- padding 前后单帧输出一致；
- 丢帧后至少保留一张真实源图，padding 不能被复活；
- null-change 反事实（把源观测替换为 source anchor，保持时间/掩码不变）必须明显损害结果，证明模型使用了真实源变化，而不是只复制 target anchor 或依赖时间先验；
- physical 相对复制 target anchor 的 RMSE 至少改善 5%。

第二层是 Sentinel 性能门槛：

- translation 一帧不能劣于现有 V3.2 单帧 physical；
- 每个 frame-count/task-mode 分层的 physical 都优于复制锚点；
- visual RMSE 不超过对应 physical 的 1.05 倍；
- LPIPS/DISTS 各改善至少 5%，Edge F1 与 PSD 同时改善；
- 至少 70% 场景改善 DISTS 或 Edge F1；
- RMSE 退化超过 5% 的场景不超过 10%；
- Optical 投影前越界不超过 0.1%；
- SAR bias 不超过 0.5 dB，PSD、ENL、CDF、P1/P99 尾部优于 physical。

第三层才是跨传感器结论：冻结时序骨干，只用少量新配准对适配 descriptor I/O 和 radiometric head，并与从头训练比较。没有这组实验，不能声称“任意传感器、任意分辨率”。

## 7. 运行顺序

先运行 64 样本 feasibility 链。脚本自动串联四阶段、从 `latest.pt` 恢复，并在上一阶段没有产生合格 `best_*.pt` 时停止：

```bash
cd /data/code/sentinel_translat/v3.2
DIRECTION=sar_to_optical \
  ./scripts/launch_paired_temporal_v2_pipeline_8gpu.sh
DIRECTION=optical_to_sar \
  ./scripts/launch_paired_temporal_v2_pipeline_8gpu.sh
```

只有两方向 feasibility 均过门槛后才切全量配置：

```bash
CONFIG_PATH=configs/paired_temporal_v2_full.yaml \
OUTPUT_ROOT=checkpoints_paired_temporal_v2_full \
DIRECTION=sar_to_optical \
  ./scripts/launch_paired_temporal_v2_pipeline_8gpu.sh
```

full 配置每 1k 做固定 32 样本 pilot，每 5k 做完整 validation；只有完整验证更新 best 和早停。checkpoint 同时绑定完整配置的 SHA-256、方向、模型 family 与 stage，旧 V3.2/V1 checkpoint 不能混入。

## 8. 主要难点

### 不可辨识未来

如果目标时刻没有源图，新增建筑、云、作物突变和随机散斑没有观测证据。模型只能学习条件分布，不能保证逐像素恢复唯一真值。解决方法是把 translation 与 forecast 分开报告，physical 负责条件期望，visual 表达受约束的不确定性。

### 配准误差会被当成高频

一像素偏移足以产生大面积伪边缘。高频训练必须拒绝位移超过 0.5 px、云影或有效率不足的 patch；配准对日期差也必须受限。

### 稀疏输入训练分布

如果总用完整序列训练，推理只有一帧时会崩。随机 observation dropout、query dropout 和分层验证是模型支持稀疏输入的必要条件，不只是数据增强。

### 多分辨率不是 resize 问题

更换 GSD 会改变 PSF、散斑尺度和可恢复频率。descriptor 可以告诉模型传感器差异，但不能创造输入不存在的空间信息。跨分辨率适配必须用共同物理网格、sensor-specific anti-alias/resampler 和真实配准对验证。

### 数据量

当前 manifest 在旧的固定四帧严格协议下只有约 753 个 SAR→Optical train query 和 832 个 Optical→SAR train query。V2 会产生不同 anchor/observation 子集，但独立地物数量没有增加。强随机模型很容易记忆 Beijing 地块，因此空间封闭测试和 scene-level 统计比 patch 数更重要。

## 9. 当前验证状态

已经完成的是代码逻辑验证，不是遥感效果结论：

- V2 模型的单帧、多帧、padding、双方向、translation/forecast、固定 seed 和 flow 反传测试通过；
- 训练工具的分层稀疏采样、动态有效 mask、四阶段冻结、固定随机验证、协议绑定和 checkpoint 兼容测试已经实现；
- 数据层的跨年因果历史、多锚点采样、配准证据诊断、0/0.25/1 高频权重和端到端 TIFF→collate→model 合同已经实现；
- 未训练状态严格回退到真实目标锚点；
- 旧 Temporal V1 保留为消融对照，不与 V2 checkpoint 互通。

尚未完成的是实际 Sentinel pilot、视觉面板和 8 卡全量训练。共享 `/data` 在本轮验证中多次进入 NFS I/O 等待，因此此时不能声称 V2 已改善真实 RMSE、视觉质量或达到 SOTA。存储稳定后先跑 64 样本四阶段 feasibility，达到门槛后再启动 8 卡全量训练。

## 10. 可行性判断

作为“目标时刻有源图”的双向图生图，这个方向可行性高：配准对提供跨模态标定，查询源图提供当时结构，问题比旧单图映射更受约束。

作为“只有历史源图”的未来目标生成，可行性中等：低频状态和持续变化可以改善，独有高频只能做合理采样，不能承诺真值重建。历史越少、时间间隔越长，上限越低。

作为“任意传感器与任意分辨率”的统一模型，可行性取决于是否允许少量配准对适配。少样本迁移是合理目标；完全零样本且保证物理正确，目前不应承诺。
