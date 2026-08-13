"""Evaluate one bidirectional SOPAT V4 checkpoint on a fixed causal protocol.

This command intentionally reports only input-matched SOPAT V4 routes:
the learned checkpoint, target-anchor copy, and optional causal
counterfactuals.  Older V2/V3 checkpoints require a separately declared
adapter because their single-image inputs are not equivalent to SOPAT's
paired-anchor observation set.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sentinel_v4.evaluation import (
    VARIANT_NAMES,
    SOPATVariantConfig,
    evaluate_sopat_loaders,
    export_sopat_prediction_samples,
    select_sopat_candidate,
)
from sentinel_v4.model import SOPAT
from sentinel_v4.training import ModelEMA, SOPATTrainConfig, load_sopat_checkpoint

# The runner owns the common raw/chunk-cache protocol preparation.  Script
# execution puts this directory on sys.path; importing the same helpers keeps
# evaluation from accidentally accepting a less strict cache route.
try:
    from train_sopat_v4 import (  # type: ignore[import-not-found]
        _datasets,
        _device_generator,
        _load_config,
        _loaders,
        _model_config,
        _prepare_data_on_rank_zero,
        _selection_config,
    )
except ModuleNotFoundError:  # pragma: no cover - module-style test invocation
    from scripts.train_sopat_v4 import (
        _datasets,
        _device_generator,
        _load_config,
        _loaders,
        _model_config,
        _prepare_data_on_rank_zero,
        _selection_config,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(sorted(VARIANT_NAMES)),
        help="Input-matched V4 route; repeat to evaluate multiple routes.",
    )
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--panel-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--phase", choices=("feasibility", "full"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the checkpoint EMA state (default) or its raw model state.",
    )
    return parser


def _device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_train_config(path: Path) -> SOPATTrainConfig:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise TypeError("SOPAT evaluation checkpoint payload must be a mapping")
    values = payload.get("train_config")
    if not isinstance(values, Mapping):
        raise TypeError("SOPAT evaluation checkpoint is missing train_config")
    return SOPATTrainConfig.from_mapping(values)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _variants(requested: Sequence[str] | None) -> tuple[str, ...]:
    if requested is None:
        return ("sopat", "anchor_copy", "source_shuffle")
    unique = tuple(dict.fromkeys(str(value) for value in requested))
    if not unique:
        raise ValueError("SOPAT evaluation needs at least one variant")
    return unique


def evaluate(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    output: Path,
    config_base: Path,
    device: torch.device,
    variants: Sequence[str],
    limit_batches: int | None,
    panel_samples: int,
    seed: int,
    phase: str | None,
    use_ema: bool,
) -> dict[str, object]:
    """Run an auditable V4 validation evaluation without opening raw cache paths."""

    if limit_batches is not None and limit_batches <= 0:
        raise ValueError("SOPAT --limit-batches must be positive")
    if panel_samples <= 0:
        raise ValueError("SOPAT --panel-samples must be positive")
    prepared = _prepare_data_on_rank_zero(
        config,
        output=output / "prepared",
        config_base=config_base,
    )
    model_config = _model_config(config)
    train_config = _checkpoint_train_config(checkpoint)
    model = SOPAT(model_config).to(device)
    ema = ModelEMA.create(model, train_config.ema_decay)
    payload = load_sopat_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        ema=ema,
        scheduler=None,
        model_config=asdict(model_config),
        train_config=train_config,
        protocol_hashes=prepared.protocol_hashes,
        restore_rng=False,
    )
    train_datasets, validation_datasets = _datasets(
        config,
        prepared,
        seed=seed,
        stage="physical",
    )
    del train_datasets
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("SOPAT evaluation configuration requires training mapping")
    validation = config.get("validation")
    if not isinstance(validation, Mapping):
        raise TypeError("SOPAT evaluation configuration requires validation mapping")
    batch_size = int(validation.get("batch_size", training.get("batch_size", 1)))
    if batch_size < 2:
        raise ValueError("SOPAT validation.batch_size must be at least 2 for source_shuffle")
    num_workers = int(training.get("num_workers", 4))
    validation_loaders = _loaders(
        validation_datasets,
        batch_size=batch_size,
        num_workers=max(0, min(num_workers, 2)),
        device=device,
        rank=0,
        world_size=1,
        training=False,
        seed=seed,
    )
    context = ema.average_parameters(model) if use_ema else _null_context()
    reports: dict[str, object] = {}
    with context:
        for offset, name in enumerate(variants):
            generator = _device_generator(device, seed + 1_000_003 * offset)
            reports[name] = evaluate_sopat_loaders(
                model if name != "anchor_copy" else None,
                validation_loaders,
                variant=SOPATVariantConfig(name=name, seed=seed),  # type: ignore[arg-type]
                device=device,
                limit_batches=limit_batches,
                generator=generator,
            )
        if "sopat" in reports:
            export_sopat_prediction_samples(
                model,
                validation_loaders,
                output / "panels",
                variant=SOPATVariantConfig("sopat", seed=seed),
                device=device,
                limit_per_direction=panel_samples,
                generator=_device_generator(device, seed + 99),
            )
    selection_phase = phase or "full"
    selection = _selection_config(config, phase=selection_phase)
    sopat_report = reports.get("sopat")
    shuffle_report = reports.get("source_shuffle")
    decision = (
        select_sopat_candidate(
            sopat_report,
            selection,
            source_shuffle_report=shuffle_report if isinstance(shuffle_report, Mapping) else None,
        )
        if isinstance(sopat_report, Mapping)
        else None
    )
    result: dict[str, object] = {
        "family": "sopat_v4_evaluation_run",
        "checkpoint": str(checkpoint),
        "checkpoint_global_step": int(payload.get("global_step", 0)),
        "checkpoint_train_stage": train_config.stage,
        "protocol_hashes": prepared.protocol_hashes,
        "input_matched": True,
        "ema": bool(use_ema),
        "reports": reports,
        "selection": decision.to_dict() if decision is not None else None,
        "external_baselines": {
            "input_matched": False,
            "note": "V2/V3 single-image checkpoints are not evaluated here. "
            "Use an explicit paired-anchor adapter and disclose the input mismatch.",
        },
    }
    _write_json(output / "evaluation.json", result)
    return result


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _type: object, _value: object, _traceback: object) -> bool:
        return False


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    result = evaluate(
        config,
        checkpoint=args.checkpoint,
        output=args.output,
        config_base=args.config.parent.resolve(),
        device=_device(args.device),
        variants=_variants(args.variant),
        limit_batches=args.limit_batches,
        panel_samples=args.panel_samples,
        seed=args.seed,
        phase=args.phase,
        use_ema=args.use_ema,
    )
    print(json.dumps({"output": str(args.output / "evaluation.json"), "selection": result["selection"]}))


if __name__ == "__main__":
    main()
