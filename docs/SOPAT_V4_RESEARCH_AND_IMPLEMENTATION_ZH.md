# SOPAT V4 双向 SAR-Optical 研究与实施规范

更新日期：2026-08-14

## 1. 研究目标与边界

SOPAT-Core（Sparse-Observation Paired-Anchor Transport）使用一个 checkpoint 完成
Sentinel-1 SAR 到 Sentinel-2 Optical 和反方向转换。输入至少包含一对历史配准
S1/S2 锚点，以及 1--N 张源模态观测；输出是在目标模态历史锚点上运输可观测变化后
得到的目标时刻影像。

第一篇论文只研究：

- Sentinel-1 VV/VH dB 与 Sentinel-2 十波段反射率；
- canonical 10 m 对齐网格；
- 一对历史配准锚点和 1--N 个无序源观测；
- translation 与 forecast 的双向确定性 physical 输出；
- 一个共享变化运输核心和传感器专用输入/辐射输出算子。

第一篇不声称任意传感器、任意 GSD，也不以 residual flow 为核心贡献。原生网格、
PSF/MTF renderer、外部传感器和少样本适配属于后续 SOPAT-Operator。

## 2. 可证伪的核心假设

历史配准对负责描述同一地表在两种传感器中的观测方式。后续源图只提供相对同模态
锚点的变化。若模型先分离锚点公共场景状态与传感器私有状态，再把 1--N 个源变化作为
无序集合运输到目标模态，它应当：

1. 同时优于复制目标锚点、单图 V3.2 和输入匹配的固定拼接基线；
2. 在变化区域上的收益大于未变化区域，且 source shuffle 后明显退化；
3. 在 forecast 或高变化样本上从 4+ observations 获得超过单 observation 的收益；
4. 用一个共享模型达到两个独立方向模型的 95% 置信区间，同时参数更少。

任一方向长期只能复制锚点、更多观测无收益或 source shuffle 不影响结果，都构成对核心
假设的反证，不能靠加入 diffusion 掩盖。

## 3. 数据与因果协议

### 3.1 当前可证明的协议

现有 2017--2024 manifest 的 acquisition metadata 只有日期，没有 UTC 时间。因此 V4
第一版的因果强度是 `date` precision：

- source/target anchor 日期必须严格早于 target 日期；
- 所有 observation 日期不得晚于 target 日期；
- translation 必须显式标记一个 query-source，其日期与 target 相差 0--1 天；
- forecast 不得包含 query-source，所有 observation 距 target 必须超过 1 天；
- target modality 的资产 ID 不得出现在任何输入 role；
- split、tile、orbit 和 canonical grid 必须一致；
- 观测集合的序列化顺序可稳定，但模型不得依赖 slot 顺序。

同日内的真实先后关系无法由当前数据证明。论文必须写成“日期级严格因果”；只有未来
manifest 补齐 UTC acquisition time 后，才能升级为时间戳级严格因果。

### 3.2 样本契约

```text
SOPATExampleV4
  sample_id, split, tile, direction, task_mode
  target_ref, target_date
  anchor_pair
    source_ref, source_date
    target_ref, target_date
    registration/provenance
  observations[1..N]
    source_ref, source_date, role = history | query_source
  query_source_id?              # translation only
  canonical_grid, sensor_schema_hash, normalization_version
  time_precision = date
```

训练以方向同质 microbatch 读取数据。每个 optimizer step 分别处理 SAR->Optical 和
Optical->SAR microbatch，再更新同一个 model、EMA、optimizer 和 global step。不同通道数
不得通过填充合并成一个普通 dense batch。

## 4. 模型

```text
registered SAR anchor ---- E_sar ----+---- Factorize ---- common state c_a
                                     |                    private p_sar
registered Optical anchor - E_opt ---+                    private p_opt

source observation_i ---- E_source ---- difference from E_source(anchor)
                                             |
                          four-scale masked set transport + relative time
                                             |
                                      transported change Delta_q
                                             |
target anchor + target-private + c_a + Delta_q -- target renderer --> physical
```

### 4.1 Sensor encoder

主配置使用完整四尺度 FPN 和 12 层、768 hidden 的共享 scene encoder。S1/S2 使用独立
channel projection、low-rank modality adapter；测试配置允许缩小宽度和深度。每个唯一
影像只编码一次，观测可分块编码并使用 activation checkpointing。

### 4.2 Paired-anchor factorization

锚点的 H/8 token 通过对称 cross-attention 得到公共地表状态。私有状态定义为各模态
锚点特征减去公共投影后的残差。训练包含：

- 两种锚点的 cross reconstruction；
- 公共状态对齐；
- 公共和私有状态去相关；
- private swap 诊断。

公共状态对齐不强迫 SAR speckle 与 Optical 颜色逐像素一致。

### 4.3 Anchor-relative set transport

每个尺度只输入显式变化：

```text
d_i^l = E_source^l(observation_i) - E_source^l(source_anchor)
```

relative-time embedding、valid mask 和 present mask 进入 point-wise set attention。集合聚合
必须对 permutation 不变；padding 的值、日期与 valid 内容均不得影响输出。共享 transport
trunk 预测公共变化，方向/目标传感器 adapter 将其转换成目标 latent update。

### 4.4 Radiometric rendering

S2 和 S1 使用独立输出头。确定性输出是目标锚点的受界增量：

```text
physical = bounded_update(target_anchor, delta_target)
```

输出投影零初始化，使未训练模型精确返回 target anchor。模型 forward 不接收 target label；
target 只在 loss 和 changed/unchanged metric 中使用。模型同时输出 log variance、集合注意力、
有效支持和投影前越界率。

## 5. 第一阶段损失

`factorizer` 阶段只学习锚点状态分解和 cross reconstruction。`physical` 阶段使用两个方向
的加权和：

```text
L = L_charbonnier
  + lambda_grad L_gradient
  + lambda_delta L_(prediction-anchor delta)
  + lambda_nll L_heteroscedastic
  + lambda_regret max(0, RMSE(pred)-RMSE(anchor)+margin)
  + lambda_null L(model(null-change), anchor)
  + lambda_perm L(model(permuted-set), model(set))
```

Optical 另外报告十波段 bias、SAM、ERGAS、NDVI；SAR 在真实 dB 单位报告 VV/VH RMSE、
bias 和 correlation。梯度冲突先只记录 cosine，不在首版加入 PCGrad；只有实验证明共享梯度
冲突持续且独立模型更优时才作为消融引入。

## 6. 基线、消融与评价

统一协议包含：

- `anchor_copy`；
- 现有单图 V3.2；
- 参数匹配的 fixed-concat 和 mean-pool；
- latest-only、source-null、source-shuffle；
- 去掉 anchor-relative difference；
- 去掉 factorization；
- shared transport 对比两个独立方向 transport；
- N=1、N=2--3、N=4--8；
- translation/forecast 与时间间隔分层。

fixed-concat 等是“style baseline”，除非逐项复现外部论文，否则不得标为官方论文复现。
changed/unchanged mask 只由 ground-truth target 与 target anchor 构造，仅用于评价，不得进入
forward 或训练 gate。

## 7. 分阶段门槛

### Connectivity / 64 samples

- 双方向同一 checkpoint 均有有限梯度；
- 初始化精确复制 anchor；
- permutation 误差和 padding 影响处于数值容差；
- null change 返回 anchor；
- target label 不在 forward signature；
- 两方向各自的 64 样本 overfit 都超过 anchor-copy，而不是只让平均值变好。

### Feasibility

- 每个 direction/task/N 核心 bucket 不劣于 anchor-copy；
- source evidence improvement 为正，source shuffle 明显变差；
- changed-region delta error 优于 fixed-concat；
- 任何方向失败都不得保存为 joint best。

### Full physical

- 两个方向显著超过最强输入匹配基线、anchor-copy 和 V3.2；
- 至少 70% 场景改善；
- 4+ observations 在 forecast 或高变化子集显著优于 one；
- shared model 与 independent models 的两个方向均处在同一 95% CI；
- validation 完成选模后，封闭测试只运行一次。

### Visual 与 downstream

只有 physical 通过后才实现可观测 deterministic detail；只有 detail 通过后才允许研究
不可辨识 residual sampling。最终 visual 要求 `RMSE <= 1.05 * physical RMSE`，并改善
LPIPS、DISTS、Edge 与 PSD。下游 SCL proxy 比较 SAR-only、synthetic Optical、
SAR+synthetic Optical 和 real Optical oracle；它不是独立 land-cover ground truth。

## 8. Checkpoint 与发布

V4 checkpoint 独立于 V3：

```text
sopat_v4_format = 1
family = sopat_v4
directions = [sar_to_optical, optical_to_sar]
stage = factorizer | physical
model, ema, optimizer, scheduler, RNG, global_step
model_config, train_config
protocol hashes for both direction indexes, cache, sensor schema
best validation metrics for both directions
initialization provenance
```

`--resume` 必须严格匹配 V4 architecture、双方向协议和 stage。旧 V3/V2 只允许显式
`--init-v3` 迁移白名单内、形状一致的 scene encoder 参数，不恢复 optimizer、scheduler、
release scale 或旧 physical/detail/flow 头。

## 9. 实现文件与运行顺序

V4 保持独立命名空间，不覆盖 V3.2 checkpoint：

- `src/sentinel_v4/model.py`：paired-anchor factorization、四尺度集合变化运输和双辐射头；
- `src/sentinel_v4/data.py`：V3 序列索引到 V4 显式角色索引的迁移与严格因果校验；
- `src/sentinel_v4/cache.py`：只读 chunk cache preflight，任何协议不一致均 fail closed；
- `src/sentinel_v4/training.py`：双方向单图 DDP objective、EMA 与 v4 checkpoint；
- `src/sentinel_v4/evaluation.py`：anchor、source-shuffle、task/N/change-region 分层评价；
- `scripts/build_sopat_v4_index.py`：原子发布 V4 role index 与四个精确投影的 V3 cache
  index；validation 在迁移前按 fixed-center 最小可评估支持筛选，train 绝不按标签像素筛选；
- `scripts/train_sopat_v4.py`：factorizer/physical 两阶段 8 卡训练；
- `scripts/evaluate_sopat_v4.py`：固定验证协议评估与可视化样本导出。

标准执行链：

```bash
# 0. 固化 validation 中心可评估协议；不会改写源 index
PYTHONPATH=src python scripts/filter_sopat_v4_center_evaluable.py \
  --manifest /home/noah/datasets/sentinel_translate_paired_v2_feasibility/manifests/pairs.jsonl \
  --source-root /home/noah/datasets/sentinel_translate_paired_v2_feasibility/indexes \
  --output-root /home/noah/datasets/sentinel_translate_paired_v2_feasibility/indexes_v4_center_evaluable

# 1. 64-sample factorizer feasibility
torchrun --standalone --nproc_per_node=8 scripts/train_sopat_v4.py \
  --config configs/sopat_v4_feasibility_local.yaml \
  --stage factorizer --init-v3 checkpoints_v32_canonical_2017_2024/best_physical.pt \
  --output checkpoints_sopat_v4_feasibility

# 2. 从最佳 factorizer 初始化 deterministic physical
torchrun --standalone --nproc_per_node=8 scripts/train_sopat_v4.py \
  --config configs/sopat_v4_feasibility_local.yaml \
  --stage physical \
  --init-checkpoint checkpoints_sopat_v4_feasibility/factorizer/best_factorizer.pt \
  --output checkpoints_sopat_v4_feasibility
```

等价的串联启动器为 `scripts/launch_sopat_v4_feasibility_8gpu.sh`。它只在 factorizer
产生非空 `best_factorizer.pt` 后启动 physical，随后自动生成同裁剪 V2/V4/anchor 比较和
固定色标 PNG。通过 feasibility 后，全量链使用
`scripts/launch_sopat_v4_full_chunk_8gpu.sh`（旧的
`scripts/launch_sopat_v4_full_8gpu.sh` 是兼容入口）。全量链严格按以下顺序执行：

```bash
tmux new-session -d -s sopat-v4-full \
  'cd /data/code/sentinel_translat/v3.2 && bash scripts/launch_sopat_v4_full_chunk_8gpu.sh'
```

开始任何数据写入前，启动器 fail-closed 读取 `FEASIBILITY_REPORT`（默认第二轮
feasibility 目录）。它要求 `.validation.selection` 的 `eligible=true`、
`phase="feasibility"`、空 `failures` 和有限 `score`，并要求
`.validation.selection_policy.version="sopat_v4_quality_gate_v2"`。其中序列化的
`effective` policy 至少必须保持双方向 feasibility scene-improved fraction `>=0.50`、
source-shuffle structural degradation `>=0.01`、optical 相对 anchor 的 SAM/NDVI 不回归且
Edge F1 不回归、SAR absolute bias `<=0.5 dB`、SAR Edge F1 regression cap `-0.02`。更严格的
policy 可以通过，旧式仅有 `eligible=true` 或任一较弱/缺失字段的报告会在 cache 构建和
`torchrun` 前直接退出。通过后，它在 `/data/datasets/sopat_v4_2017_2024` 发布：

- `index.jsonl` 与 `paired_indexes/{direction}/{split}.jsonl`，二者逐 sample projection
  完全相等，并由 `index_publication.json` 的内容 hash 绑定；
- `chunk_cache`，只通过显式的两个方向 train/validation V3 indexes 选取 role 所需
  acquisition，预算 `200 GiB`、预留可用空间 `80 GiB`，默认 8 个 cache workers；
- `configs/sopat_v4_full_chunk.yaml`，只读已完成 `.npy` mmap cache；cache publication 与
  V4 role index 不一致时 training preflight 必须失败，不能退回 TIFF/NFS。

全量训练固定为 crop `256`、最多 8 observations、8 GPU。启动器要求
`CUDA_VISIBLE_DEVICES` 恰好包含 8 个不同的物理 GPU，且与 YAML 的 `world_size=8` 一致。
启动前它查询每张卡的已用显存；默认超过 `8192 MiB` 则拒绝运行且不终止任何其他进程。
轻量服务可通过 `GPU_USED_MIB_LIMIT` 调整，重训练占用则需要先由资源所有者释放。该数值是
启动前的保守冲突检查，不代表训练所需的显存上限。

这条链不包含训练监控循环。以 64-patch feasibility 的实际速度为基准，256 patch 的像素量约为
16 倍；全量 cache 首次物化预计数小时，10k factorizer 加 30k physical 预计数日。实际首个完整
阶段结束后再记录吞吐和最终 wall-clock，不以此估计替代报告。

当前 feasibility 固定中心筛选保留 SAR->Optical `61/64` 个验证序列、
Optical->SAR `64/64` 个验证序列；被排除的 3 个序列在 query target 与历史 target anchor
之间没有任何共同有效中心像素。完整 sample ID 记录在数据目录的
`center_filter_report.json`，不是训练期间动态挑选。

只有两个方向的 feasibility gate 均通过，才允许按相同顺序运行
`configs/sopat_v4_full_chunk.yaml`。`publication_is_valid` 的 reusable fast path 是 full-only，
故明确固定 `validation_temporal`；feasibility 或其他 split 必须生成独立 publication，不能复用
此全量 marker。训练启动后不以人工监测改变选模或超参数。

## 10. 当前状态

- 设计规格：已冻结。
- 日期级 full index 审计：通过；25 个 2560x2560、10 m grids。
- V4 核心、数据、训练、评价代码和定向单测：已实现。
- 64 feasibility、同裁剪 V2/V4 对比、GitHub commit、full training：待执行。
- SOPAT-Operator、任意传感器/分辨率、residual flow：未实现，不属于当前发布主张。
