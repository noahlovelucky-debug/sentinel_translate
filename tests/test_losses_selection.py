from __future__ import annotations

import json
from pathlib import Path

import torch

from sentinel_v3.losses import latent_alignment, physical_loss
from sentinel_v3.selection import select_checkpoint


def test_latent_alignment_prefers_correct_pairs() -> None:
    scenes = torch.eye(4).view(4, 4, 1, 1)
    loss_aligned, _ = latent_alignment(scenes, scenes, torch.ones(4, 1, 1, 1))
    loss_permuted, _ = latent_alignment(scenes, scenes.roll(1, 0), torch.ones(4, 1, 1, 1))
    assert loss_aligned < loss_permuted


def test_physical_loss_balances_acceptance_scale_errors() -> None:
    mask = torch.ones(1, 1, 8, 8)
    signs = torch.ones(1, 1, 8, 8)
    signs[..., 4:, :] = -1
    optical_target = torch.full((1, 3, 8, 8), 0.5)
    sar_target = torch.full((1, 2, 8, 8), -20.0)
    optical_loss, _ = physical_loss(
        optical_target + 0.05 * signs,
        torch.zeros_like(optical_target),
        optical_target,
        mask,
        "optical",
        torch.ones(1),
    )
    sar_loss, _ = physical_loss(
        sar_target + 5.0 * signs,
        torch.zeros_like(sar_target),
        sar_target,
        mask,
        "sar",
        torch.ones(1),
    )
    ratio = float(optical_loss / sar_loss)
    assert 0.5 <= ratio <= 2.0


def test_hard_gate_checkpoint_selection(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "protocol_hash": "fixed-protocol",
                "quality_gates": {"physical": True, "visual": True, "joint": True},
            }
        ),
        encoding="utf-8",
    )
    result = select_checkpoint(checkpoint, [report], tmp_path / "selected")
    assert result["selected"] == [
        "best_physical.pt",
        "best_visual.pt",
        "best_joint.pt",
    ]


def test_selection_rejects_mixed_protocols(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    reports = []
    for index, protocol_hash in enumerate(("a", "b")):
        report = tmp_path / f"report_{index}.json"
        report.write_text(
            json.dumps({"protocol_hash": protocol_hash, "quality_gates": {}}),
            encoding="utf-8",
        )
        reports.append(report)
    try:
        select_checkpoint(checkpoint, reports, tmp_path / "selected")
    except ValueError as error:
        assert "protocol hash" in str(error)
    else:
        raise AssertionError("mixed protocols must be rejected")
