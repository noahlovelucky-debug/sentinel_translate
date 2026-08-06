from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .model import ModelConfig

DEFAULT: dict[str, Any] = {
    "paths": {
        "train_shards": "/data/sentinel_translate/data/shards_v2/train/index.json",
        "manifest": "/data/sentinel_translate/data/manifests/pairs.jsonl",
        "output": "/data/code/sentinel_translate_v3/checkpoints",
        "reports": "/data/code/sentinel_translate_v3/reports",
    },
    "model": asdict(ModelConfig()),
    "train": {
        "stage": "pretrain",
        "max_steps": 30000,
        "batch_size": 16,
        "gradient_accumulation": 1,
        "num_workers": 0,
        "learning_rate": 0.0001,
        "encoder_learning_rate": 0.00002,
        "weight_decay": 0.05,
        "warmup_steps": 2000,
        "gradient_clip": 1.0,
        "ema_decay": 0.9999,
        "init_use_ema": False,
        "channels_last": False,
        "save_final": True,
        "seed": 42,
        "amp": "bfloat16",
        "log_every": 20,
        "validate_every": 1000,
        "save_every": 2000,
        "task_probabilities": [0.35, 0.35, 0.15, 0.15],
        "physical_alignment_samples": 4,
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = _merge(DEFAULT, yaml.safe_load(handle) or {})
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    train = config["train"]
    if int(model["hidden"]) % int(model["heads"]):
        raise ValueError("model.hidden must be divisible by model.heads")
    if int(model["hidden"]) % 4:
        raise ValueError("model.hidden must be divisible by four")
    if train["stage"] not in {"overfit", "pretrain", "physical", "visual", "balance"}:
        raise ValueError("unsupported training stage")
    probabilities = [float(value) for value in train["task_probabilities"]]
    if len(probabilities) != 4 or abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("four task probabilities must sum to one")
    if int(train["num_workers"]) < 0 or int(train["batch_size"]) <= 0:
        raise ValueError("invalid data loader settings")
    if int(train["physical_alignment_samples"]) < 2:
        raise ValueError("train.physical_alignment_samples must be at least two")
    if train["amp"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("train.amp must be bfloat16, float16, or float32")
