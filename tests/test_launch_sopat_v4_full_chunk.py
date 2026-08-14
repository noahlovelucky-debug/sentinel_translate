from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "launch_sopat_v4_full_chunk_8gpu.sh"
COMPATIBILITY_LAUNCHER = ROOT / "scripts" / "launch_sopat_v4_full_8gpu.sh"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


POLICY_VERSION = "sopat_v4_quality_gate_v3"
_VALID_EFFECTIVE_POLICY: dict[str, object] = {
    "phase": "feasibility",
    "feasibility_scene_improved_fraction_min": 0.50,
    "feasibility_source_shuffle_min_degradation": 0.01,
    "feasibility_candidate_source_shuffle_min_degradation": 0.01,
    "optical_sam_anchor_delta_max": 0.0,
    "optical_ndvi_mae_anchor_delta_max": 0.0,
    "optical_edge_f1_anchor_delta_min": 0.0,
    "full_sar_db_bias_abs_max": 0.5,
    "sar_edge_f1_anchor_delta_min": -0.02,
}


def _global_counterfactual_variant() -> dict[str, object]:
    return {
        "name": "global_cross_tile",
        "planner": "global_cross_tile_hard_v1",
        "plan_hash": "a" * 64,
        "coverage": 1.0,
        "cross_tile_coverage": 1.0,
        "tier_counts": {"same_task_exact_n": 2},
    }


def _report(
    path: Path,
    *,
    eligible: bool,
    phase: str = "feasibility",
    failures: list[str] | None = None,
    score: float = 1.0,
    include_policy: bool = True,
    policy_version: str = POLICY_VERSION,
    effective_policy: dict[str, object] | None = None,
    source_shuffle_variant: dict[str, object] | None = None,
    include_source_shuffle: bool = True,
) -> Path:
    validation: dict[str, object] = {
        "selection": {
            "eligible": eligible,
            "phase": phase,
            "failures": [] if failures is None else failures,
            "score": score,
        }
    }
    if include_policy:
        validation["selection_policy"] = {
            "version": policy_version,
            "effective": (
                dict(_VALID_EFFECTIVE_POLICY)
                if effective_policy is None
                else effective_policy
            ),
        }
    if include_source_shuffle:
        validation["source_shuffle"] = {
            "variant": (
                _global_counterfactual_variant()
                if source_shuffle_variant is None
                else source_shuffle_variant
            )
        }
    return _write(
        path,
        json.dumps({"validation": validation}),
    )


def _full_config(path: Path, *, world_size: int = 8) -> Path:
    return _write(
        path,
        yaml.safe_dump({"training": {"world_size": world_size}}, sort_keys=True),
    )


def _stub_command(directory: Path, name: str, log: Path, *, body: str = "") -> Path:
    return _write(
        directory / name,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s:%s\\n' '{name}' \"$*\" >> '{log}'\n"
        f"{body}\n",
    ).chmod(0o755) or directory / name


def _environment(
    tmp_path: Path,
    *,
    eligible: bool,
    world_size: int = 8,
    report_options: dict[str, object] | None = None,
) -> dict[str, str]:
    stubs = tmp_path / "stubs"
    log = tmp_path / "commands.log"
    _stub_command(stubs, "nvidia-smi", log, body="printf '0\\n'")
    report = _report(tmp_path / "report.json", eligible=eligible, **(report_options or {}))
    full = _full_config(tmp_path / "full.yaml", world_size=world_size)
    for name in ("index.yaml", "cache.yaml", "v3.pt"):
        _write(tmp_path / name, "placeholder\n")
    return {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "ROOT": str(ROOT),
        "FULL_CONFIG": str(full),
        "INDEX_CONFIG": str(tmp_path / "index.yaml"),
        "CACHE_CONFIG": str(tmp_path / "cache.yaml"),
        "V3_INIT": str(tmp_path / "v3.pt"),
        "FEASIBILITY_REPORT": str(report),
        "DATA_ROOT": str(tmp_path / "data"),
        "OUTPUT": str(tmp_path / "output"),
        "TMPDIR": str(tmp_path / "tmp"),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "SOPAT_FULL_DRY_RUN": "1",
        "COMMAND_LOG": str(log),
    }


def _run(
    tmp_path: Path,
    *,
    eligible: bool,
    world_size: int = 8,
    report_options: dict[str, object] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = _environment(
        tmp_path,
        eligible=eligible,
        world_size=world_size,
        report_options=report_options,
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_false_feasibility_gate_executes_no_subsequent_commands(tmp_path: Path) -> None:
    result = _run(tmp_path, eligible=False)

    assert result.returncode != 0
    assert "feasibility gate is not eligible" in result.stderr
    assert not (tmp_path / "commands.log").exists()
    assert not (tmp_path / "data").exists()


def test_legacy_eligible_report_fails_closed_before_gpu_or_cache_work(tmp_path: Path) -> None:
    result = _run(tmp_path, eligible=True, report_options={"include_policy": False})

    assert result.returncode != 0
    assert "validation.selection_policy must be an object" in result.stderr
    assert not (tmp_path / "commands.log").exists()
    assert not (tmp_path / "data").exists()


def test_policy_missing_a_required_threshold_fails_closed_before_gpu_or_cache_work(
    tmp_path: Path,
) -> None:
    effective_policy = dict(_VALID_EFFECTIVE_POLICY)
    del effective_policy["sar_edge_f1_anchor_delta_min"]

    result = _run(
        tmp_path,
        eligible=True,
        report_options={"effective_policy": effective_policy},
    )

    assert result.returncode != 0
    assert "sar_edge_f1_anchor_delta_min must be a finite number" in result.stderr
    assert not (tmp_path / "commands.log").exists()
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("field", "weaker_value"),
    [
        ("feasibility_scene_improved_fraction_min", 0.49),
        ("feasibility_source_shuffle_min_degradation", 0.009),
        ("feasibility_candidate_source_shuffle_min_degradation", 0.009),
        ("optical_sam_anchor_delta_max", 0.001),
        ("optical_ndvi_mae_anchor_delta_max", 0.001),
        ("optical_edge_f1_anchor_delta_min", -0.001),
        ("full_sar_db_bias_abs_max", 0.501),
        ("sar_edge_f1_anchor_delta_min", -0.021),
    ],
)
def test_weaker_effective_policy_fails_closed_before_gpu_or_cache_work(
    tmp_path: Path, field: str, weaker_value: float
) -> None:
    effective_policy = dict(_VALID_EFFECTIVE_POLICY)
    effective_policy[field] = weaker_value

    result = _run(
        tmp_path,
        eligible=True,
        report_options={"effective_policy": effective_policy},
    )

    assert result.returncode != 0
    assert field in result.stderr
    assert not (tmp_path / "commands.log").exists()
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("report_options", "expected_error"),
    [
        ({"phase": "full"}, "validation.selection.phase must be feasibility"),
        ({"failures": ["source_shuffle_gate"]}, "validation.selection.failures"),
        ({"score": float("nan")}, "validation.selection.score must be a finite number"),
        (
            {"policy_version": "sopat_v4_quality_gate_v1"},
            "validation.selection_policy.version must be 'sopat_v4_quality_gate_v3'",
        ),
        (
            {"policy_version": "sopat_v4_quality_gate_v2"},
            "validation.selection_policy.version must be 'sopat_v4_quality_gate_v3'",
        ),
    ],
)
def test_incomplete_or_unversioned_feasibility_decision_fails_closed(
    tmp_path: Path, report_options: dict[str, object], expected_error: str
) -> None:
    result = _run(tmp_path, eligible=True, report_options=report_options)

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not (tmp_path / "commands.log").exists()
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("variant", "expected_error"),
    [
        ({"name": "source_shuffle"}, "validation.source_shuffle.variant.name"),
        (
            {**_global_counterfactual_variant(), "coverage": 0.99},
            "validation.source_shuffle.variant.coverage=0.99 must be >= 1",
        ),
        (
            {**_global_counterfactual_variant(), "cross_tile_coverage": 0.99},
            "validation.source_shuffle.variant.cross_tile_coverage=0.99 must be >= 1",
        ),
    ],
)
def test_global_counterfactual_metadata_fails_closed_before_gpu_or_cache_work(
    tmp_path: Path, variant: dict[str, object], expected_error: str
) -> None:
    result = _run(
        tmp_path,
        eligible=True,
        report_options={"source_shuffle_variant": variant},
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not (tmp_path / "commands.log").exists()
    assert not (tmp_path / "data").exists()


def _expected_gpu_checks() -> list[str]:
    return [
        "nvidia-smi:--id=0 --query-gpu=memory.used --format=csv,noheader,nounits",
        "nvidia-smi:--id=1 --query-gpu=memory.used --format=csv,noheader,nounits",
        "nvidia-smi:--id=2 --query-gpu=memory.used --format=csv,noheader,nounits",
        "nvidia-smi:--id=3 --query-gpu=memory.used --format=csv,noheader,nounits",
        "nvidia-smi:--id=4 --query-gpu=memory.used --format=csv,noheader,nounits",
        "nvidia-smi:--id=5 --query-gpu=memory.used --format=csv,noheader,nounits",
        "nvidia-smi:--id=6 --query-gpu=memory.used --format=csv,noheader,nounits",
        "nvidia-smi:--id=7 --query-gpu=memory.used --format=csv,noheader,nounits",
    ]


def test_true_gate_dry_run_preserves_required_chain_order(tmp_path: Path) -> None:
    result = _run(tmp_path, eligible=True)

    assert result.returncode == 0, result.stderr
    event_lines = [
        line
        for line in result.stdout.splitlines()
        if "RUN " in line or "GPU contract valid:" in line
    ]
    events = []
    for line in event_lines:
        if "GPU contract valid:" in line:
            events.append("gpu-check")
        elif "build_sopat_v4_index.py" in line:
            events.append("build-index")
        elif "--stage factorizer" in line:
            events.append("factorizer")
        elif "--stage physical" in line:
            events.append("physical")
        elif "--execute" in line:
            events.append("build-cache")
        elif "--verify" in line:
            events.append("verify-cache")

    assert events == [
        "gpu-check",
        "build-index",
        "build-cache",
        "verify-cache",
        "gpu-check",
        "factorizer",
        "gpu-check",
        "physical",
    ]

    run_lines = [line for line in result.stdout.splitlines() if "RUN " in line]
    assert len(run_lines) == 5
    assert "build_sopat_v4_index.py" in run_lines[0]
    assert "build_paired_temporal_chunk_cache.py" in run_lines[1]
    assert "--execute" in run_lines[1]
    assert "build_paired_temporal_chunk_cache.py" in run_lines[2]
    assert "--verify" in run_lines[2]
    assert "--stage factorizer" in run_lines[3]
    assert "--stage physical" in run_lines[4]
    assert (tmp_path / "commands.log").read_text(encoding="utf-8").splitlines() == (
        _expected_gpu_checks() * 3
    )


def test_launcher_refuses_world_size_or_visible_gpu_mismatch(tmp_path: Path) -> None:
    result = _run(tmp_path, eligible=True, world_size=7)

    assert result.returncode != 0
    assert "world_size=7 differs from visible GPU count=8" in result.stderr
    assert not (tmp_path / "commands.log").exists()


def test_launcher_script_is_bash_syntax_valid() -> None:
    result = subprocess.run(["bash", "-n", str(LAUNCHER)], text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr


def test_compatibility_launcher_delegates_to_chunk_chain() -> None:
    result = subprocess.run(
        ["bash", "-n", str(COMPATIBILITY_LAUNCHER)], text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "launch_sopat_v4_full_chunk_8gpu.sh" in COMPATIBILITY_LAUNCHER.read_text(
        encoding="utf-8"
    )
