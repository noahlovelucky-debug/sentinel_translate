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
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    if not reports:
        raise ValueError("at least one validation report is required")
    optical_gate_names = (
        "lpips_improves_5_percent", "dists_improves_5_percent", "edge_improves",
        "optical_psd_improves", "rgb_rmse_within_5_percent", "optical_bounds_within_0_1_percent",
    )
    sar_gate_names = (
        "sar_bias_within_0_5_db", "sar_psd_improves", "sar_enl_improves", "sar_histogram_improves"
    )
    optical_pass = all(
        all(bool(report.get("quality_gates", {}).get(name, False)) for name in optical_gate_names)
        for report in reports
    )
    sar_pass = all(
        all(bool(report.get("quality_gates", {}).get(name, False)) for name in sar_gate_names)
        for report in reports
    )
    physical_pass = True
    if baseline_rmse is not None:
        physical_pass &= all(report["sar2opt_rmse"] <= baseline_rmse * 1.02 for report in reports)
    if baseline_sam is not None:
        physical_pass &= all(report["sar2opt_sam"] <= baseline_sam * 1.02 for report in reports)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    if optical_pass and physical_pass:
        shutil.copy2(checkpoint, destination / "best_sar2opt.pt")
        selected.append("best_sar2opt.pt")
    if sar_pass:
        shutil.copy2(checkpoint, destination / "best_opt2sar.pt")
        selected.append("best_opt2sar.pt")
    if optical_pass and sar_pass and physical_pass and baseline_rmse is not None and baseline_sam is not None:
        shutil.copy2(checkpoint, destination / "best_joint.pt")
        selected.append("best_joint.pt")
    result = {
        "optical_pass": optical_pass,
        "sar_pass": sar_pass,
        "physical_pass": physical_pass,
        "baseline_supplied": baseline_rmse is not None and baseline_sam is not None,
        "selected": selected,
    }
    (destination / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
