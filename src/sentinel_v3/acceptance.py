from __future__ import annotations

from typing import Any


def acceptance_decision(
    report: dict[str, Any], milestone: str, *, manual_visual_pass: bool = False
) -> dict[str, Any]:
    if milestone not in {"connectivity", "1k", "5k", "final"}:
        raise ValueError("milestone must be connectivity, 1k, 5k, or final")
    if milestone == "connectivity":
        metrics = report["training_metrics"]
        checks = {
            "sar2opt_detail_mae_improves_30_percent": float(
                metrics["sar2opt/detail_mae_improvement"]
            )
            >= 0.30,
            "opt2sar_detail_mae_improves_30_percent": float(
                metrics["opt2sar/detail_mae_improvement"]
            )
            >= 0.30,
        }
        return {
            "milestone": milestone,
            "checks": checks,
            "passed": all(checks.values()),
            "manual_review_required": False,
        }
    physical_rmse = float(report["physical_rgb_rmse"])
    visual_rmse = float(report["visual_rgb_rmse"])
    checks: dict[str, bool] = {
        "physical_gate": bool(report.get("quality_gates", {}).get("physical", False)),
        "manual_visual_review": manual_visual_pass,
    }
    if milestone == "1k":
        checks.update(
            {
                "rgb_rmse_within_10_percent": visual_rmse <= 1.10 * physical_rmse,
                "pre_projection_violation_within_1_percent": float(
                    report["pre_projection_violation"]
                )
                <= 0.01,
            }
        )
    elif milestone == "5k":
        checks.update(
            {
                "rgb_rmse_within_5_percent": visual_rmse <= 1.05 * physical_rmse,
                "lpips_improves_3_percent": float(report["lpips_improvement"]) >= 0.03,
                "dists_improves_3_percent": float(report["dists_improvement"]) >= 0.03,
                "edge_improves": report["visual_edge_f1"] > report["physical_edge_f1"],
                "psd_improves": report["visual_optical_psd_distance"]
                < report["physical_optical_psd_distance"],
            }
        )
    else:
        checks.update(
            {
                "visual_gate": bool(report.get("quality_gates", {}).get("visual", False)),
                "joint_gate": bool(report.get("quality_gates", {}).get("joint", False)),
            }
        )
    return {
        "milestone": milestone,
        "checks": checks,
        "passed": all(checks.values()),
        "manual_review_required": True,
    }
