from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .acceptance import acceptance_decision
from .baselines import evaluate_legacy_sar2opt
from .calibration import calibrate_checkpoint
from .config import load_config
from .evaluation import evaluate
from .model import ModelConfig, SentinelV3
from .selection import select_checkpoint
from .temporal_prior import configure_checkpoint_temporal_prior
from .training import train
from .validation import ValidationProtocol, validation_protocol_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel-v32")
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--resume")
    training.add_argument("--init-model")
    training.add_argument("--limit", type=int)
    training.add_argument(
        "--stage",
        choices=("overfit", "physical", "detail", "codec", "flow", "balance"),
    )
    training.add_argument("--max-steps", type=int)
    training.add_argument("--output")
    training.add_argument("--reports")
    training.add_argument("--batch-size", type=int)
    training.add_argument("--gradient-accumulation", type=int)
    training.add_argument("--warmup-steps", type=int)
    training.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=None
    )
    training.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=None)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--seed", type=int, default=42)
    commands.add_parser("model-info")
    commands.add_parser("validation-protocol")
    baseline = commands.add_parser("evaluate-baseline")
    baseline.add_argument("--kind", choices=("v1_mean", "v2_refiner"), required=True)
    baseline.add_argument("--checkpoint", required=True)
    baseline.add_argument("--mean-checkpoint")
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--limit", type=int)
    acceptance = commands.add_parser("check-report")
    acceptance.add_argument("--report", required=True)
    acceptance.add_argument(
        "--milestone", choices=("connectivity", "1k", "5k", "final"), required=True
    )
    acceptance.add_argument("--manual-visual-pass", action="store_true")
    calibration = commands.add_parser("calibrate-alpha")
    calibration.add_argument("--checkpoint", required=True)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--limit", type=int)
    calibration.add_argument("--seed", type=int, default=42)
    temporal = commands.add_parser("configure-temporal-prior")
    temporal.add_argument("--checkpoint", required=True)
    temporal.add_argument("--output", required=True)
    temporal.add_argument("--optical-amplitude-weight", type=float, default=0.75)
    temporal.add_argument("--sar-weight", type=float, default=0.80)
    selection = commands.add_parser("select-checkpoint")
    selection.add_argument("--checkpoint", required=True)
    selection.add_argument("--report", action="append", required=True)
    selection.add_argument("--output-dir", required=True)
    selection.add_argument("--baseline-rmse", type=float)
    selection.add_argument("--baseline-sam", type=float)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "train":
        if args.stage:
            config["train"]["stage"] = args.stage
        if args.max_steps is not None:
            config["train"]["max_steps"] = args.max_steps
        if args.output:
            config["paths"]["output"] = args.output
        if args.reports:
            config["paths"]["reports"] = args.reports
        if args.batch_size is not None:
            config["train"]["batch_size"] = args.batch_size
        if args.gradient_accumulation is not None:
            config["train"]["gradient_accumulation"] = args.gradient_accumulation
        if args.warmup_steps is not None:
            config["train"]["warmup_steps"] = args.warmup_steps
        if args.channels_last is not None:
            config["train"]["channels_last"] = args.channels_last
        if args.save_final is not None:
            config["train"]["save_final"] = args.save_final
        train(config, resume=args.resume, init_model=args.init_model, limit=args.limit)
    elif args.command == "evaluate":
        report = evaluate(
            args.checkpoint,
            config["paths"]["manifest"],
            args.split,
            args.output,
            limit=args.limit,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "model-info":
        model = SentinelV3(ModelConfig(**config["model"]))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(
            json.dumps(
                {
                    "parameters": parameters,
                    "trainable": trainable,
                    "cuda_devices": torch.cuda.device_count(),
                }
            )
        )
    elif args.command == "validation-protocol":
        protocol = ValidationProtocol()
        print(
            json.dumps(
                {
                    "name": protocol.name,
                    "split": protocol.split,
                    "samples": protocol.expected_samples,
                    "hash": validation_protocol_hash(config["paths"]["manifest"]),
                },
                indent=2,
            )
        )
    elif args.command == "evaluate-baseline":
        report = evaluate_legacy_sar2opt(
            args.kind,
            args.checkpoint,
            config["paths"]["manifest"],
            mean_checkpoint=args.mean_checkpoint,
            limit=args.limit,
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "check-report":
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        decision = acceptance_decision(
            report, args.milestone, manual_visual_pass=args.manual_visual_pass
        )
        print(json.dumps(decision, indent=2))
        if not decision["passed"]:
            raise SystemExit(2)
    elif args.command == "calibrate-alpha":
        result = calibrate_checkpoint(
            args.checkpoint,
            config["paths"]["manifest"],
            args.output,
            seed=args.seed,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "configure-temporal-prior":
        result = configure_checkpoint_temporal_prior(
            args.checkpoint,
            config["paths"]["manifest"],
            args.output,
            optical_amplitude_weight=args.optical_amplitude_weight,
            sar_weight=args.sar_weight,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "select-checkpoint":
        result = select_checkpoint(
            args.checkpoint,
            args.report,
            args.output_dir,
            baseline_rmse=args.baseline_rmse,
            baseline_sam=args.baseline_sam,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
