from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


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
    for passed, name in (
        (physical_pass, "best_physical.pt"),
        (visual_pass, "best_visual.pt"),
        (joint_pass, "best_joint.pt"),
    ):
        if passed:
            shutil.copy2(checkpoint, destination / name)
            selected.append(name)
    result = {
        "physical_pass": physical_pass,
        "visual_pass": visual_pass,
        "joint_pass": joint_pass,
        "protocol_hash": hashes.pop(),
        "selected": selected,
    }
    (destination / "selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
