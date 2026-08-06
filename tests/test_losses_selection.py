from __future__ import annotations

import json
from pathlib import Path

import torch

from sentinel_v3.losses import latent_alignment
from sentinel_v3.selection import select_checkpoint


def test_latent_alignment_prefers_correct_pairs() -> None:
    scenes = torch.eye(4).view(4, 4, 1, 1)
    loss_aligned, _ = latent_alignment(scenes, scenes, torch.ones(4, 1, 1, 1))
    loss_permuted, _ = latent_alignment(scenes, scenes.roll(1, 0), torch.ones(4, 1, 1, 1))
    assert loss_aligned < loss_permuted


def test_joint_checkpoint_requires_baseline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "optical_edge_f1": 0.5,
                "optical_psd_distance": 0.1,
                "opt2sar_bias_db": 0.2,
                "sar_enl_error": 0.1,
                "sar2opt_rmse": 0.04,
                "sar2opt_sam": 0.1,
                "quality_gates": {
                    "lpips_improves_5_percent": True,
                    "dists_improves_5_percent": True,
                    "edge_improves": True,
                    "optical_psd_improves": True,
                    "rgb_rmse_within_5_percent": True,
                    "optical_bounds_within_0_1_percent": True,
                    "sar_bias_within_0_5_db": True,
                    "sar_psd_improves": True,
                    "sar_enl_improves": True,
                    "sar_histogram_improves": True,
                },
            }
        ),
        encoding="utf-8",
    )
    without = select_checkpoint(checkpoint, [report], tmp_path / "without")
    assert "best_joint.pt" not in without["selected"]
    with_baseline = select_checkpoint(
        checkpoint, [report], tmp_path / "with", baseline_rmse=0.041, baseline_sam=0.11
    )
    assert "best_joint.pt" in with_baseline["selected"]
