from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    """Replace a destination entry without following an existing symlink."""
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def select_checkpoint(
    checkpoint: str | Path,
    report_paths: list[str | Path],
    output_dir: str | Path,
    *,
    baseline_rmse: float | None = None,
    baseline_sam: float | None = None,
) -> dict[str, Any]:
    del baseline_rmse, baseline_sam  # V3.2 uses fixed physical gates, not moving baselines.
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    if not reports:
        raise ValueError("at least one validation report is required")
    hashes = {report.get("protocol_hash") for report in reports}
    if None in hashes or len(hashes) != 1:
        raise ValueError("all reports must use the same non-null V3.2 validation protocol hash")
    protocol_hash = next(iter(hashes))
    physical_pass = all(
        bool(report.get("quality_gates", {}).get("physical")) for report in reports
    )
    visual_pass = all(bool(report.get("quality_gates", {}).get("visual")) for report in reports)
    joint_pass = (
        physical_pass
        and visual_pass
        and all(bool(report.get("quality_gates", {}).get("joint")) for report in reports)
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("format_version", 0)) != 4:
        raise RuntimeError("checkpoint selection requires V3.2 format-v4 input")
    if payload.get("validation_protocol_hash") != protocol_hash:
        raise RuntimeError("checkpoint validation_protocol_hash does not match selection reports")
    payload.setdefault("quality_gates", {}).update(
        {"physical": physical_pass, "visual": visual_pass, "joint": joint_pass}
    )
    payload["selection_reports"] = [str(Path(path).resolve()) for path in report_paths]
    for passed, name in (
        (physical_pass, "best_physical.pt"),
        (visual_pass, "best_visual.pt"),
        (joint_pass, "best_joint.pt"),
    ):
        if passed:
            _atomic_torch_save(payload, destination / name)
            selected.append(name)
    result = {
        "physical_pass": physical_pass,
        "visual_pass": visual_pass,
        "joint_pass": joint_pass,
        "protocol_hash": protocol_hash,
        "selected": selected,
    }
    (destination / "selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
