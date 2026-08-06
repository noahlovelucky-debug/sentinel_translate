from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from .data import StatefulShardSampler, V2ShardDataset, time_weights
from .losses import high_frequency_loss, highpass, latent_alignment, masked_mean, physical_loss
from .model import ModelConfig, SentinelV3
from .physics import physical_resample
from .sensors import SENTINEL1, SENTINEL2, SensorSpec


class JointObjective(nn.Module):
    task_names = ("sar2opt", "opt2sar", "opt_self", "sar_self")

    def __init__(
        self,
        model: SentinelV3,
        task_probabilities: list[float],
        physical_alignment_samples: int = 4,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("task_probabilities", torch.tensor(task_probabilities))
        self.physical_alignment_samples = physical_alignment_samples
        self.visual_joint = False

    @staticmethod
    def _masked_source(values: Tensor, stage: str) -> Tensor:
        if stage not in {"pretrain", "balance"}:
            return values
        batch, channels, height, width = values.shape
        keep_channels = (torch.rand(batch, channels, 1, 1, device=values.device) > 0.15).to(values.dtype)
        spatial = torch.ones(batch, 1, height, width, device=values.device, dtype=values.dtype)
        block = max(4, height // 8)
        for index in range(batch):
            if torch.rand((), device=values.device) < 0.5:
                top = int(torch.randint(0, max(1, height - block + 1), (), device=values.device))
                left = int(torch.randint(0, max(1, width - block + 1), (), device=values.device))
                spatial[index, :, top : top + block, left : left + block] = 0
        return values * keep_channels * spatial

    def _physical_task(
        self,
        batch: dict[str, object],
        indices: Tensor,
        target_key: str,
        target_spec: SensorSpec,
        physical_weights: Tensor,
        pyramid: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        target = batch[target_key][indices]  # type: ignore[index]
        valid = batch["valid"][indices]  # type: ignore[index]
        descriptors = self.model.descriptors(target_spec.channels, target.device)
        prediction, log_variance = self.model.decoder(
            pyramid,
            descriptors,
            target_spec.modality,
            target.shape[-2:],
            self.model.condition(
                len(indices),
                target.device,
                batch["input_gsd"][indices],  # type: ignore[index]
                batch["target_gsd"][indices],  # type: ignore[index]
                batch["metadata"][indices],  # type: ignore[index]
            )[:, -3:],
        )
        prediction = prediction * valid
        loss, metrics = physical_loss(
            prediction, log_variance, target, valid, target_spec.modality, physical_weights[indices]
        )
        return loss, metrics, pyramid

    @staticmethod
    def _translation_assignments(batch_size: int, device: torch.device) -> Tensor:
        tasks = torch.arange(batch_size, device=device) % 2
        return tasks[torch.randperm(batch_size, device=device)]

    def _encode_selected(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        source_spec: SensorSpec,
        stage: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model.encode(
            self._masked_source(batch[source_key][indices], stage),  # type: ignore[index]
            source_spec,
            valid=batch["valid"][indices],  # type: ignore[index]
            input_gsd=batch["input_gsd"][indices],  # type: ignore[index]
            target_gsd=batch["target_gsd"][indices],  # type: ignore[index]
            metadata=batch["metadata"][indices],  # type: ignore[index]
        )

    def _task_assignments(self, batch_size: int) -> Tensor:
        expected = self.task_probabilities * batch_size
        counts = torch.floor(expected).long()
        remainder = batch_size - int(counts.sum())
        if remainder:
            priorities = (expected - counts).argsort(descending=True)
            counts[priorities[:remainder]] += 1
        tasks = torch.repeat_interleave(
            torch.arange(len(counts), device=counts.device), counts
        )
        return tasks[torch.randperm(batch_size, device=tasks.device)]

    def _visual_task(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        high_frequency_weights: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        source = batch[source_key][indices]  # type: ignore[index]
        target = batch[target_key][indices]  # type: ignore[index]
        valid = batch["valid"][indices]  # type: ignore[index]
        physical_context = nullcontext() if self.visual_joint else torch.no_grad()
        with physical_context:
            physical, _, pyramid = self.model.physical(
                source,
                source_spec,
                target_spec,
                valid,
                input_gsd=batch["input_gsd"][indices],  # type: ignore[index]
                target_gsd=batch["target_gsd"][indices],  # type: ignore[index]
                metadata=batch["metadata"][indices],  # type: ignore[index]
            )
        if target_spec.modality == "optical":
            selected = [2, 1, 0]
            target_visual = target[:, selected]
            base = physical[:, selected]
        else:
            target_visual = target
            base = physical
        residual = highpass((target_visual - base.detach()) * valid)
        residual_latent = self.model.residual_dit.encode_residual(residual)
        noise = torch.randn_like(residual_latent)
        time_values = torch.rand(residual.shape[0], device=residual.device, dtype=residual.dtype)
        interpolated = (1 - time_values[:, None, None, None]) * noise + time_values[:, None, None, None] * residual_latent
        velocity = self.model.flow_velocity(
            interpolated, time_values, pyramid[-1], target_spec, target_visual.shape[1]
        )
        target_velocity = residual_latent - noise
        latent_mask = F.interpolate(valid, size=velocity.shape[-2:], mode="area")
        residual_weights = high_frequency_weights[indices]
        weight = residual_weights.view(-1, 1, 1, 1) * latent_mask
        flow = masked_mean((velocity - target_velocity).square(), weight)
        decoded = highpass(
            self.model.residual_dit.decode_residual(residual_latent, target_visual.shape[1])
        )
        frequency_loss, frequency_metrics = high_frequency_loss(
            decoded,
            residual,
            valid,
            target_spec.modality,
            sample_weight=residual_weights,
        )
        predicted_endpoint = interpolated + (
            1 - time_values[:, None, None, None]
        ) * velocity
        endpoint = highpass(
            self.model.residual_dit.decode_residual(
                predicted_endpoint, target_visual.shape[1]
            )
        )
        endpoint_loss, endpoint_metrics = high_frequency_loss(
            endpoint,
            residual,
            valid,
            target_spec.modality,
            sample_weight=residual_weights,
        )
        target_amplitude = F.avg_pool2d(
            (residual * valid).square(), 4, stride=4
        ).sqrt()
        amplitude_limit = (
            self.model.config.optical_residual_limit
            if target_spec.modality == "optical"
            else self.model.config.sar_residual_limit_db
        )
        target_amplitude = target_amplitude.clamp_max(amplitude_limit)
        predicted_amplitude = self.model.residual_amplitude(
            pyramid[-1], target_spec, target_visual.shape[1], target_visual.shape[-2:]
        )
        amplitude_mask = F.avg_pool2d(valid, 4, stride=4)
        amplitude_weight = residual_weights.view(-1, 1, 1, 1) * amplitude_mask
        amplitude_loss = masked_mean(
            (predicted_amplitude - target_amplitude).abs(), amplitude_weight
        )
        total = flow + 0.15 * frequency_loss + 0.1 * endpoint_loss + 0.2 * amplitude_loss
        frequency_metrics.update(
            {
                f"endpoint_{name}": value
                for name, value in endpoint_metrics.items()
            }
        )
        frequency_metrics["amplitude"] = amplitude_loss.detach()
        if self.visual_joint:
            physical_auxiliary = masked_mean((physical - target).square(), valid)
            if target_spec.modality == "sar":
                physical_auxiliary = 0.01 * physical_auxiliary
            total = total + 0.1 * physical_auxiliary
            frequency_metrics["physical_auxiliary"] = physical_auxiliary.detach()
        return total, {"flow": flow.detach(), **frequency_metrics}

    def forward(self, batch: dict[str, object], stage: str) -> tuple[Tensor, dict[str, Tensor]]:
        device = batch["s2"].device  # type: ignore[union-attr]
        batch_size = batch["s2"].shape[0]  # type: ignore[union-attr]
        if stage in {"physical", "visual"}:
            tasks = self._translation_assignments(batch_size, device)
        else:
            tasks = self._task_assignments(batch_size)
        physical_weights, high_frequency_weights = time_weights(batch["delta_days"])  # type: ignore[arg-type]
        total = torch.zeros((), device=device)
        metrics: dict[str, Tensor] = {}
        active = 0
        specifications = (
            ("sar_view", "s2_target", SENTINEL1, SENTINEL2),
            ("s2_view", "sar_target", SENTINEL2, SENTINEL1),
            ("s2_view", "s2_target", SENTINEL2, SENTINEL2),
            ("sar_view", "sar_target", SENTINEL1, SENTINEL1),
        )
        cached_pyramids: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        if stage not in {"physical", "visual"}:
            valid = batch["valid"]  # type: ignore[assignment]
            common = {
                "valid": valid,
                "input_gsd": batch["input_gsd"],
                "target_gsd": batch["target_gsd"],
                "metadata": batch["metadata"],
            }
            cached_pyramids["sar_view"] = self.model.encode(
                self._masked_source(batch["sar_view"], stage), SENTINEL1, **common
            )
            cached_pyramids["s2_view"] = self.model.encode(
                self._masked_source(batch["s2_view"], stage), SENTINEL2, **common
            )
        active_specifications = (
            specifications[:2] if stage in {"physical", "visual"} else specifications
        )
        task_pyramids: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        for task_index, (source_key, target_key, source_spec, target_spec) in enumerate(active_specifications):
            indices = torch.nonzero(tasks == task_index, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            active += 1
            if stage == "visual" and task_index < 2:
                loss, task_metrics = self._visual_task(
                    batch, indices, source_key, target_key, source_spec, target_spec, high_frequency_weights
                )
            else:
                if stage == "physical":
                    pyramid = self._encode_selected(
                        batch, indices, source_key, source_spec, stage
                    )
                    task_pyramids[task_index] = pyramid
                else:
                    pyramid = tuple(level[indices] for level in cached_pyramids[source_key])
                loss, task_metrics, _ = self._physical_task(
                    batch,
                    indices,
                    target_key,
                    target_spec,
                    physical_weights,
                    pyramid,
                )
                if stage == "balance" and task_index < 2:
                    visual_loss, visual_metrics = self._visual_task(
                        batch,
                        indices,
                        source_key,
                        target_key,
                        source_spec,
                        target_spec,
                        high_frequency_weights,
                    )
                    loss = loss + 0.5 * visual_loss
                    task_metrics.update(
                        {f"visual_{name}": value for name, value in visual_metrics.items()}
                    )
            total = total + loss
            for name, value in task_metrics.items():
                metrics[f"{self.task_names[task_index]}/{name}"] = value
        total = total / max(active, 1)

        if stage in {"overfit", "pretrain", "physical", "balance"}:
            if stage == "physical":
                # Translation batches only encode the source modality. Reuse the
                # SAR source features and encode a small paired optical subset for
                # alignment instead of encoding both modalities for every sample.
                sar_indices = torch.nonzero(tasks == 0, as_tuple=False).flatten()
                alignment_count = min(self.physical_alignment_samples, batch_size // 2)
                alignment_indices = sar_indices[:alignment_count]
                valid = batch["valid"][alignment_indices]  # type: ignore[index]
                sar_scene = task_pyramids[0][-1][:alignment_count]
                optical_scene = self._encode_selected(
                    batch, alignment_indices, "s2_view", SENTINEL2, stage
                )[-1]
            else:
                valid = batch["valid"]  # type: ignore[assignment]
                sar_scene = cached_pyramids["sar_view"][-1]
                optical_scene = cached_pyramids["s2_view"][-1]
            alignment, alignment_metrics = latent_alignment(sar_scene, optical_scene, valid)
            coefficient = 1.0 if stage == "pretrain" else 0.1
            total = total + coefficient * alignment
            metrics.update({f"latent/{name}": value for name, value in alignment_metrics.items()})

        if stage in {"pretrain", "balance"}:
            sample_count = min(2, batch_size)
            valid = batch["valid"][:sample_count]  # type: ignore[index]
            source = batch["sar_view"][:sample_count]  # type: ignore[index]
            metadata = batch["metadata"][:sample_count]  # type: ignore[index]
            output10 = self.model.physical(source, SENTINEL1, SENTINEL2, valid, metadata=metadata, target_gsd=10.0)[0]
            output20 = self.model.physical(source, SENTINEL1, SENTINEL2, valid, metadata=metadata, target_gsd=20.0)[0]
            degraded = physical_resample(output10, modality="optical", source_gsd_m=10.0, target_gsd_m=20.0)
            scale_loss = masked_mean((degraded - output20).abs(), valid)
            total = total + 0.1 * scale_loss
            metrics["scale/consistency"] = scale_loss.detach()
        metrics["loss"] = total.detach()
        return total, metrics


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.tracked = tuple(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        self.state = {
            name: value.detach().float().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        active = [name for name in self.tracked if parameters[name].grad is not None]
        averages = [self.state[name] for name in active]
        current = [parameters[name].detach().float() for name in active]
        if not active:
            return
        torch._foreach_lerp_(averages, current, 1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "state": self.state}

    def load_state_dict(
        self, payload: dict[str, object], device: torch.device, model: nn.Module
    ) -> None:
        self.decay = float(payload["decay"])
        state = payload["state"]  # type: ignore[assignment]
        model_state = model.state_dict()
        self.state = {}
        for name, value in state.items():
            restored = value.to(device)
            target = model_state.get(name)
            if target is not None and target.ndim == 4 and target.is_contiguous(
                memory_format=torch.channels_last
            ):
                restored = restored.contiguous(memory_format=torch.channels_last)
            self.state[name] = restored


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _set_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def _scheduler(optimizer: AdamW, warmup: int, maximum: int) -> LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return max(1, step) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, factor)


def _stage_learning_rates(
    train_config: dict[str, Any], stage: str
) -> tuple[float, float, float]:
    residual_lr = float(train_config["learning_rate"])
    decoder_lr = residual_lr
    encoder_lr = float(train_config["encoder_learning_rate"])
    if stage == "visual":
        decoder_lr *= 0.1
        encoder_lr *= 0.1
    elif stage == "balance":
        decoder_lr *= 0.1
        encoder_lr *= 0.1
        residual_lr *= 0.1
    return encoder_lr, decoder_lr, residual_lr


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_compatible_state(model: nn.Module, state: dict[str, Tensor]) -> tuple[int, int]:
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current and current[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(current) - len(compatible)


def _match_optimizer_layout(optimizer: AdamW) -> None:
    for parameter, state in optimizer.state.items():
        if parameter.ndim != 4 or not parameter.is_contiguous(
            memory_format=torch.channels_last
        ):
            continue
        for name, value in state.items():
            if isinstance(value, Tensor) and value.ndim == 4:
                state[name] = value.contiguous(memory_format=torch.channels_last)


def train(
    config: dict[str, Any], *, resume: str | None = None, init: str | None = None, limit: int | None = None
) -> None:
    if resume and init:
        raise ValueError("resume and init are mutually exclusive")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_config = config["train"]
    channels_last = bool(train_config.get("channels_last", False)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed = int(train_config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset: Any = V2ShardDataset(config["paths"]["train_shards"], augment=True, random_gsd=True)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    if isinstance(dataset, Subset):
        sampler = None
        shuffle = True
    else:
        sampler = StatefulShardSampler(dataset, replicas=world_size, rank=rank, seed=seed)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        sampler=sampler,
        shuffle=shuffle,
        num_workers=int(train_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    model = SentinelV3(ModelConfig(**config["model"])).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    objective = JointObjective(
        model,
        list(train_config["task_probabilities"]),
        int(train_config.get("physical_alignment_samples", 4)),
    ).to(device)
    stage = str(train_config["stage"])
    if stage in {"overfit", "pretrain", "physical"}:
        for parameter in model.residual_dit.parameters():
            parameter.requires_grad_(False)
    encoder_lr, main_lr, residual_lr = _stage_learning_rates(train_config, stage)
    groups = [
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.decoder.parameters(), "lr": main_lr},
        {"params": model.residual_dit.parameters(), "lr": residual_lr},
    ]
    optimizer = AdamW(
        groups,
        weight_decay=float(train_config["weight_decay"]),
        fused=device.type == "cuda",
    )
    scheduler = _scheduler(optimizer, int(train_config["warmup_steps"]), int(train_config["max_steps"]))
    ema = EMA(model, float(train_config["ema_decay"]))
    step = 0
    if init:
        initial = torch.load(init, map_location="cpu", weights_only=False)
        initial_state = dict(initial["model"])
        if bool(train_config.get("init_use_ema", False)) and "ema" in initial:
            initial_state.update(initial["ema"]["state"])
        loaded, initialized = _load_compatible_state(model, initial_state)
        if rank == 0:
            print(
                json.dumps(
                    {"init": str(init), "compatible_tensors": loaded, "new_tensors": initialized}
                ),
                flush=True,
            )
        ema = EMA(model, float(train_config["ema_decay"]))
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if channels_last:
            _match_optimizer_layout(optimizer)
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"], device, model)
        step = int(checkpoint["step"])
        states = checkpoint["rank_states"]
        state = states[rank] if rank < len(states) else states[0]
        _set_rng_state(state["rng"])
        if sampler is not None and state.get("sampler") is not None:
            sampler.load_state_dict(state["sampler"])
    objective.visual_joint = stage == "balance" or (stage == "visual" and step >= 10000)
    wrapped: nn.Module = objective
    if distributed:
        wrapped = DistributedDataParallel(
            objective,
            device_ids=[local_rank],
            find_unused_parameters=stage == "visual",
            static_graph=stage != "visual",
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            bucket_cap_mb=100,
        )
    amp_name = str(train_config["amp"])
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[amp_name]
    accumulation = int(train_config["gradient_accumulation"])
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    starting_step = step
    output = Path(config["paths"]["output"])
    while step < int(train_config["max_steps"]):
        if stage == "visual" and step == 10000:
            objective.visual_joint = True
        aggregate: dict[str, float] = {}
        for micro_step in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {
                key: (
                    value.to(device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                    if channels_last and value.ndim == 4
                    else value.to(device, non_blocking=True)
                )
                if isinstance(value, Tensor)
                else value
                for key, value in batch.items()
            }
            sync_context = wrapped.no_sync() if distributed and micro_step + 1 < accumulation else nullcontext()  # type: ignore[attr-defined]
            amp_context = torch.autocast(device.type, dtype=amp_dtype) if device.type == "cuda" and amp_dtype != torch.float32 else nullcontext()
            with sync_context, amp_context:
                loss, metrics = wrapped(batch, stage)  # type: ignore[operator]
                loss = loss / accumulation
            loss.backward()
            for name, value in metrics.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value) / accumulation
        for group in optimizer.param_groups:
            nn.utils.clip_grad_norm_(group["params"], float(train_config["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if rank == 0 and step % int(train_config["log_every"]) == 0:
            elapsed = time.time() - started
            processed = (step - starting_step) * int(train_config["batch_size"]) * world_size
            processed *= accumulation
            print(
                json.dumps(
                    {
                        "step": step,
                        "seconds": round(elapsed, 1),
                        "samples_per_second": round(processed / max(elapsed, 1e-6), 1),
                        **aggregate,
                    }
                ),
                flush=True,
            )
        save_final = bool(train_config.get("save_final", True))
        if step % int(train_config["save_every"]) == 0 or (
            save_final and step == int(train_config["max_steps"])
        ):
            local_state = {"rng": _rng_state(), "sampler": sampler.state_dict() if sampler is not None else None}
            if distributed:
                rank_states: list[object] = [None] * world_size
                dist.all_gather_object(rank_states, local_state)
            else:
                rank_states = [local_state]
            if rank == 0:
                payload = {
                    "format_version": 3,
                    "stage": stage,
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rank_states": rank_states,
                    "config": config,
                }
                stage_path = output / stage / f"step_{step:07d}.pt"
                _atomic_save(payload, stage_path)
                latest = output / "latest.pt"
                temporary_latest = output / ".latest.pt.tmp"
                temporary_latest.unlink(missing_ok=True)
                temporary_latest.symlink_to(os.path.relpath(stage_path, output))
                temporary_latest.replace(latest)
    if distributed:
        dist.destroy_process_group()
