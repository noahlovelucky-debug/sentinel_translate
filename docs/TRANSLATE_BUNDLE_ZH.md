# Translate 服务器迁移包

`/data/code/translate` 是当前 Sentinel 图生图研究的独立迁移目录。它包含 V3.2
已通过门槛的物理底座、SOPAT V4 研究代码和继续训练所需的最小权重，不包含历史上
全部中间 checkpoint，也不把 3.2 TiB 原始影像复制进代码目录。

## 目录职责

```text
/data/code/translate
  src/ scripts/ configs/ tests/ docs/  # 完整源码与协议
  artifacts/checkpoints/v32/          # 已通过物理门槛的 V3.2 权重
  artifacts/checkpoints/v4/           # V4 factorizer 与研究初始化权重
  artifacts/eval_weights/             # LPIPS/DISTS 评测依赖
  artifacts/protocols/                # manifest、index 和 validation protocol
  artifacts/reports/                  # 对应选择/失败报告
  artifacts/SHA256SUMS                 # 包内大文件的字节级校验
  BUNDLE_INFO.json                     # 代码 commit、来源和发布状态
```

V3.2 `best_physical.pt` 是可用于确定性物理输出的 release 权重。
`final_calibrated.pt` 用于复现实验中的 V3.2 visual 路径。V4 physical 权重尚未通过
全局历史依赖门槛，只能作为下一轮 SOPAT 训练初始化，不能作为发布模型。

## 数据不放进代码目录

原始数据在源服务器保持为：

```text
/data/data_disk/data_dir/{2017,...,2024}
```

2017–2024 年目录内的精确数据清单是 `317,814` 个普通文件、
`3,430,323,306,974` bytes。数据根目录另有一个 11-byte 的
`.retry_incomplete_downloads.lock` 运行锁，不属于 Sentinel 数据，也不应传输：

| 年份 | 文件数 | bytes |
|---|---:|---:|
| 2017 | 36,983 | 368,293,879,352 |
| 2018 | 43,570 | 470,700,816,628 |
| 2019 | 51,271 | 567,618,127,487 |
| 2020 | 70,583 | 791,645,667,782 |
| 2021 | 33,795 | 373,645,371,189 |
| 2022 | 22,598 | 234,797,873,039 |
| 2023 | 22,233 | 231,324,770,449 |
| 2024 | 36,781 | 392,296,801,048 |

推荐在新服务器保持同一路径，避免重写 JSONL 中的资产路径。传输使用可恢复的
`rsync`。包内脚本默认只预演，确认目标地址后显式执行复制，再做 checksum 校验：

```bash
scripts/transfer_sentinel_2017_2024.sh \
  USER@HOST:/data/data_disk/data_dir --dry-run
scripts/transfer_sentinel_2017_2024.sh \
  USER@HOST:/data/data_disk/data_dir --execute
scripts/transfer_sentinel_2017_2024.sh \
  USER@HOST:/data/data_disk/data_dir --verify
```

对应的原始命令为：

```bash
rsync -aH --numeric-ids --partial --info=progress2 \
  --exclude='.retry_incomplete_downloads.lock' \
  /data/data_disk/data_dir/ USER@HOST:/data/data_disk/data_dir/

rsync -aHnc --numeric-ids --exclude='.retry_incomplete_downloads.lock' \
  /data/data_disk/data_dir/ USER@HOST:/data/data_disk/data_dir/
```

第二条命令不写目标端；无输出才表示内容一致。目标服务器还应运行快速聚合检查：

```bash
python /data/code/translate/scripts/verify_sentinel_transfer.py \
  /data/data_disk/data_dir
```

## 代码包校验与环境

```bash
cd /data/code/translate
sha256sum -c artifacts/SHA256SUMS
python -m pip install -e .
PYTHONPATH=src python -m pytest -q tests/test_sopat_v4_model.py \
  tests/test_sopat_v4_data.py tests/test_sopat_v4_training.py
```

完整 SOPAT mmap 数据缓存不随代码复制。数据到达后，使用
`scripts/launch_sopat_v4_full_chunk_8gpu.sh` 构建并校验缓存；feasibility report 未通过
V3 quality gate 时，launcher 必须在缓存构建和全量训练前退出。

## 必须保留的路径或显式修改

新服务器默认目录约定：

```text
/data/code/translate
/data/data_disk/data_dir
/data/datasets/sentinel_translate_v32_2017_2024
/data/datasets/sopat_v4_2017_2024
```

若新服务器使用其他挂载点，应只修改 `configs/*.yaml` 和 launcher 的 `DATA_ROOT`、
`CACHE_ROOT`、`V3_INIT` 等环境变量；不要手工修改已经发布的 index 内容。应重新运行
index/cache publication，使新的路径和协议 hash 一起写入 checkpoint。
