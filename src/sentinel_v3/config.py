from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .model import ModelConfig
from .validation import validation_protocol_hash

DEFAULT: dict[str, Any] = {
    "paths": {
        "train_shards": "/data/sentinel_translate/data/shards_v2/train/index.json",
        "temporal_prior_shards": "/data/sentinel_translate/data/shards_v32_temporal_prior/index.json",
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
        "persistent_workers": True,
        "prefetch_factor": 2,
        "activation_checkpointing": False,
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
        "physical_alignment_every": 1,
        "optical_dists_weight": 0.1,
        "flow_perceptual_every": 8,
        "flow_visual_pixel_weight": 0.05,
        "flow_visual_hf_weight": 0.05,
        "flow_visual_perceptual_weight": 0.025,
        "flow_rollout_every": 4,
        "flow_rollout_steps": 2,
        "flow_rollout_samples": 2,
        "flow_rollout_pixel_weight": 0.1,
        "flow_rollout_hf_weight": 0.1,
        "id_bridge_antithetic_weight": 0.0,
        "phase_transport_hf_weight": 0.05,
        "phase_transport_utility_weight": 0.10,
        "phase_transport_signed_alignment_weight": 0.0,
        "risk_flow_steps": 4,
        "bridge_flow_steps": 4,
        "codec_perceptual_every": 8,
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
    if model["id_bridge_state"] not in {"codec", "haar_packet"}:
        raise ValueError("model.id_bridge_state must be codec or haar_packet")
    if model["id_bridge_state"] == "haar_packet" and model["id_bridge_state_channels"] != 48:
        raise ValueError("model.id_bridge_state_channels must be 48 for haar_packet")
    for name in ("id_bridge_optical_state_scale", "id_bridge_sar_state_scale"):
        value = float(model[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"model.{name} must be finite and positive")
    for name in ("id_bridge_optical_innovation_scale", "id_bridge_sar_innovation_scale"):
        value = float(model[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"model.{name} must be finite and in [0, 1]")
    band_scales = model["id_bridge_optical_innovation_band_scales"]
    if not isinstance(band_scales, (list, tuple)) or len(band_scales) != 3:
        raise ValueError("model.id_bridge_optical_innovation_band_scales must contain three values")
    if any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in band_scales
    ):
        raise ValueError(
            "model.id_bridge_optical_innovation_band_scales must be finite and in [0, 1]"
        )
    for name in (
        "id_bridge_optical_mid_basis_scale",
        "id_bridge_optical_coarse_basis_scale",
    ):
        value = float(model[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"model.{name} must be finite and non-negative")
    for name in ("id_bridge_optical_correction_scale", "id_bridge_sar_correction_scale"):
        value = float(model[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"model.{name} must be finite and in [0, 1]")
    if not isinstance(model["phase_transport_enabled"], bool):
        raise TypeError("model.phase_transport_enabled must be a bool")
    hidden = model["phase_transport_hidden"]
    if isinstance(hidden, bool) or not isinstance(hidden, int):
        raise TypeError("model.phase_transport_hidden must be a positive integer")
    if hidden <= 0:
        raise ValueError("model.phase_transport_hidden must be positive")
    for name in ("phase_transport_gain_caps", "phase_transport_offset_caps_px"):
        values = model[name]
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"model.{name} must contain three values")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError(f"model.{name} must contain numeric values")
        normalized = [float(value) for value in values]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in normalized):
            raise ValueError(f"model.{name} must be finite and in [0, 1]")
        model[name] = normalized
    initial_gate = float(model["phase_transport_initial_gate"])
    if not math.isfinite(initial_gate) or not 0.0 < initial_gate < 1.0:
        raise ValueError("model.phase_transport_initial_gate must be finite and in (0, 1)")
    model["phase_transport_initial_gate"] = initial_gate
    if not isinstance(model["phase_transport_null_calibrated"], bool):
        raise TypeError("model.phase_transport_null_calibrated must be a bool")
    null_quantile = float(model["phase_transport_null_quantile"])
    if not math.isfinite(null_quantile) or not 0.0 < null_quantile < 1.0:
        raise ValueError("model.phase_transport_null_quantile must be finite and in (0, 1)")
    model["phase_transport_null_quantile"] = null_quantile
    support_epsilon = float(model["phase_transport_support_epsilon"])
    if not math.isfinite(support_epsilon) or support_epsilon <= 0.0:
        raise ValueError("model.phase_transport_support_epsilon must be finite and positive")
    model["phase_transport_support_epsilon"] = support_epsilon
    if model["phase_transport_carrier_mode"] not in {"physical_gain", "orthogonal_source"}:
        raise ValueError(
            "model.phase_transport_carrier_mode must be physical_gain or orthogonal_source"
        )
    for name in ("flow_noise_scale", "optical_flow_noise_scale", "sar_flow_noise_scale"):
        if model[name] is not None and float(model[name]) < 0.0:
            raise ValueError(f"model.{name} must be non-negative")
    if not 0.0 <= float(model["optical_texture_risk_threshold"]) <= 1.0:
        raise ValueError("model.optical_texture_risk_threshold must be in [0, 1]")
    if float(model["optical_bridge_density_threshold"]) <= 0.0:
        raise ValueError("model.optical_bridge_density_threshold must be positive")
    if train["stage"] not in {
        "overfit",
        "physical",
        "detail",
        "codec",
        "flow",
        "risk",
        "bridge",
        "id_bridge",
        "id_utility",
        "phase_transport",
        "visual",
        "balance",
    }:
        raise ValueError("unsupported training stage")
    probabilities = [float(value) for value in train["task_probabilities"]]
    if len(probabilities) != 2 or abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("two direction probabilities must sum to one")
    if int(train["num_workers"]) < 0 or int(train["batch_size"]) <= 0:
        raise ValueError("invalid data loader settings")
    if not isinstance(train["persistent_workers"], bool):
        raise TypeError("train.persistent_workers must be a bool")
    prefetch_factor = train["prefetch_factor"]
    if isinstance(prefetch_factor, bool) or not isinstance(prefetch_factor, int):
        raise TypeError("train.prefetch_factor must be a positive integer")
    if prefetch_factor < 1:
        raise ValueError("train.prefetch_factor must be positive")
    if not isinstance(train["activation_checkpointing"], bool):
        raise TypeError("train.activation_checkpointing must be a bool")
    if int(train["physical_alignment_samples"]) < 2:
        raise ValueError("train.physical_alignment_samples must be at least two")
    alignment_every = train["physical_alignment_every"]
    if isinstance(alignment_every, bool) or not isinstance(alignment_every, int):
        raise TypeError("train.physical_alignment_every must be a positive integer")
    if alignment_every < 1:
        raise ValueError("train.physical_alignment_every must be positive")
    if not 0.0 <= float(train["physical_alignment_weight"]):
        raise ValueError("train.physical_alignment_weight must be non-negative")
    for name in (
        "flow_visual_pixel_weight",
        "flow_visual_hf_weight",
        "flow_visual_perceptual_weight",
        "flow_rollout_pixel_weight",
        "flow_rollout_hf_weight",
        "id_bridge_antithetic_weight",
        "phase_transport_hf_weight",
        "phase_transport_utility_weight",
        "phase_transport_signed_alignment_weight",
    ):
        if not math.isfinite(float(train[name])) or float(train[name]) < 0.0:
            raise ValueError(f"train.{name} must be non-negative")
    for name in ("flow_rollout_every", "flow_rollout_steps", "flow_rollout_samples"):
        if int(train[name]) < 1:
            raise ValueError(f"train.{name} must be positive")
    if int(train["risk_flow_steps"]) < 1:
        raise ValueError("train.risk_flow_steps must be positive")
    if int(train["bridge_flow_steps"]) < 1:
        raise ValueError("train.bridge_flow_steps must be positive")
    optical_detail_override = train.get("optical_detail_confidence_threshold_override")
    if (
        optical_detail_override is not None
        and not 0.0 <= float(optical_detail_override) <= 1.01
    ):
        raise ValueError(
            "train.optical_detail_confidence_threshold_override must be in [0, 1.01]"
        )
    if not 0.0 <= float(train["native_gsd_probability"]) <= 1.0:
        raise ValueError("train.native_gsd_probability must be in [0, 1]")
    if train["amp"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("train.amp must be bfloat16, float16, or float32")
