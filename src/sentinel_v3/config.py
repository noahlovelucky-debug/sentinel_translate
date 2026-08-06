from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .model import ModelConfig
from .validation import validation_protocol_hash

DEFAULT: dict[str, Any] = {
    "paths": {
        "train_shards": "/data/sentinel_translate/data/shards_v2/train/index.json",
        "manifest": "/data/sentinel_translate/data/manifests/pairs.jsonl",
        "output": "/data/code/sentinel_translat/v3.2/checkpoints_v32",
        "reports": "/data/code/sentinel_translat/v3.2/reports_v32",
    },
    "model": asdict(ModelConfig()),
    "train": {
        "stage": "physical",
        "max_steps": 20000,
        "batch_size": 16,
        "gradient_accumulation": 1,
        "num_workers": 0,
        "learning_rate": 0.00001,
        "adapter_learning_rate": 0.0001,
        "encoder_learning_rate": 0.000002,
        "weight_decay": 0.05,
        "warmup_steps": 2000,
        "gradient_clip": 1.0,
        "ema_decay": 0.999,
        "init_use_ema": False,
        "channels_last": False,
        "save_final": True,
        "seed": 42,
        "amp": "bfloat16",
        "log_every": 20,
        "validate_every": 1000,
        "save_every": 2000,
        "task_probabilities": [0.5, 0.5],
        "physical_alignment_samples": 4,
        "physical_alignment_weight": 0.02,
        "optical_dists_weight": 0.1,
        "native_gsd_probability": 0.8,
        "full_validate_every": 5000,
        "early_stop_patience": 5,
        "pcgrad": True,
        "find_unused_parameters": True,
        "registration_audit": True,
        "require_physical_gate": True,
        "require_codec_gate": True,
        "require_detail_gate": True,
        "balance_learning_rates": {
            "encoder": 0.000002,
            "physical_detail": 0.00001,
            "dit": 0.00001,
        },
    },
    "validation": {
        "enabled": True,
        "split": "validation_temporal",
        "quick_samples": 32,
        "full_steps": [4000, 6000, 8000, 10000, 12000],
        "protocol_hash": "unresolved",
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
    if config["validation"].get("enabled", False):
        config["validation"]["protocol_hash"] = validation_protocol_hash(
            config["paths"]["manifest"]
        )
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    train = config["train"]
    if int(model["hidden"]) % int(model["heads"]):
        raise ValueError("model.hidden must be divisible by model.heads")
    if int(model["hidden"]) % 4:
        raise ValueError("model.hidden must be divisible by four")
    if int(model["dit_hidden"]) % int(model["dit_heads"]):
        raise ValueError("model.dit_hidden must be divisible by model.dit_heads")
    if train["stage"] not in {
        "overfit",
        "physical",
        "detail",
        "codec",
        "flow",
        "visual",
        "balance",
    }:
        raise ValueError("unsupported training stage")
    probabilities = [float(value) for value in train["task_probabilities"]]
    if len(probabilities) != 2 or abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("two direction probabilities must sum to one")
    if int(train["num_workers"]) < 0 or int(train["batch_size"]) <= 0:
        raise ValueError("invalid data loader settings")
    if int(train["physical_alignment_samples"]) < 2:
        raise ValueError("train.physical_alignment_samples must be at least two")
    if not 0.0 <= float(train["physical_alignment_weight"]):
        raise ValueError("train.physical_alignment_weight must be non-negative")
    if not 0.0 <= float(train["native_gsd_probability"]) <= 1.0:
        raise ValueError("train.native_gsd_probability must be in [0, 1]")
    if train["amp"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("train.amp must be bfloat16, float16, or float32")
