from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "launch_sopat_v4_full_chunk_8gpu.sh"
COMPATIBILITY_LAUNCHER = ROOT / "scripts" / "launch_sopat_v4_full_8gpu.sh"


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _report(path: Path, *, eligible: bool) -> Path:
    return _write(
        path,
        json.dumps({"validation": {"selection": {"eligible": eligible}}}),
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


def _environment(tmp_path: Path, *, eligible: bool, world_size: int = 8) -> dict[str, str]:
    stubs = tmp_path / "stubs"
    log = tmp_path / "commands.log"
    _stub_command(stubs, "nvidia-smi", log, body="printf '0\\n'")
    report = _report(tmp_path / "report.json", eligible=eligible)
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


def _run(tmp_path: Path, *, eligible: bool, world_size: int = 8) -> subprocess.CompletedProcess[str]:
    environment = _environment(tmp_path, eligible=eligible, world_size=world_size)
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
