from __future__ import annotations

import json
from pathlib import Path

import pytest
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
    torch.save(
        {
            "format_version": 4,
            "quality_gates": {},
            "validation_protocol_hash": "fixed-protocol",
        },
        checkpoint,
    )
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
    selected = torch.load(
        tmp_path / "selected" / "best_physical.pt", weights_only=False
    )
    assert selected["quality_gates"] == {
        "physical": True,
        "visual": True,
        "joint": True,
    }


def test_selection_replaces_symlink_and_existing_destination_entries(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    torch.save(
        {
            "format_version": 4,
            "quality_gates": {},
            "validation_protocol_hash": "fixed-protocol",
            "selected_value": "candidate",
        },
        checkpoint,
    )
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
    selected_dir = tmp_path / "selected"
    selected_dir.mkdir()
    step_checkpoint = tmp_path / "physical" / "step_0004000.pt"
    step_checkpoint.parent.mkdir()
    torch.save({"sentinel": "must remain unchanged"}, step_checkpoint)
    original_step_bytes = step_checkpoint.read_bytes()
    best_physical = selected_dir / "best_physical.pt"
    best_physical.symlink_to(step_checkpoint)
    best_visual = selected_dir / "best_visual.pt"
    best_visual.write_bytes(b"old regular destination")
    best_joint = selected_dir / "best_joint.pt"

    select_checkpoint(checkpoint, [report], selected_dir)

    assert step_checkpoint.read_bytes() == original_step_bytes
    assert torch.load(step_checkpoint, weights_only=False) == {
        "sentinel": "must remain unchanged"
    }
    assert not best_physical.is_symlink()
    assert not best_visual.is_symlink()
    assert not best_joint.is_symlink()
    for selected_path in (best_physical, best_visual, best_joint):
        selected = torch.load(selected_path, weights_only=False)
        assert selected["selected_value"] == "candidate"
        assert selected["quality_gates"] == {
            "physical": True,
            "visual": True,
            "joint": True,
        }


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


def test_selection_rejects_checkpoint_protocol_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    torch.save(
        {
            "format_version": 4,
            "quality_gates": {},
            "validation_protocol_hash": "checkpoint-protocol",
        },
        checkpoint,
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"protocol_hash": "report-protocol", "quality_gates": {}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="validation_protocol_hash"):
        select_checkpoint(checkpoint, [report], tmp_path / "selected")
