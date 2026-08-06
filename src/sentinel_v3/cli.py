from __future__ import annotations

import argparse
import json

import torch

from .config import load_config
from .evaluation import evaluate
from .model import ModelConfig, SentinelV3
from .selection import select_checkpoint
from .training import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel-v3")
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--resume")
    training.add_argument("--init")
    training.add_argument("--limit", type=int)
    training.add_argument("--stage", choices=("overfit", "pretrain", "physical", "visual", "balance"))
    training.add_argument("--max-steps", type=int)
    training.add_argument("--output")
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
        train(config, resume=args.resume, init=args.init, limit=args.limit)
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
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(json.dumps({"parameters": parameters, "trainable": trainable, "cuda_devices": torch.cuda.device_count()}))
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
