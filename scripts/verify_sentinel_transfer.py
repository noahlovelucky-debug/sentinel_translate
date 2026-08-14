#!/usr/bin/env python3
"""Verify the aggregate 2017-2024 Sentinel transfer inventory.

This intentionally performs a metadata-only pass. Use the rsync checksum pass
documented in ``docs/TRANSLATE_BUNDLE_ZH.md`` when byte-level verification is
required after copying the 3.2 TiB source tree.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EXPECTED = {
    "2017": {"files": 36_983, "bytes": 368_293_879_352},
    "2018": {"files": 43_570, "bytes": 470_700_816_628},
    "2019": {"files": 51_271, "bytes": 567_618_127_487},
    "2020": {"files": 70_583, "bytes": 791_645_667_782},
    "2021": {"files": 33_795, "bytes": 373_645_371_189},
    "2022": {"files": 22_598, "bytes": 234_797_873_039},
    "2023": {"files": 22_233, "bytes": 231_324_770_449},
    "2024": {"files": 36_781, "bytes": 392_296_801_048},
}


def _inventory(root: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for year in EXPECTED:
        directory = root / year
        if not directory.is_dir():
            result[year] = {"files": 0, "bytes": 0}
            continue
        files = 0
        total_bytes = 0
        for current, _, names in os.walk(directory):
            current_path = Path(current)
            for name in names:
                path = current_path / name
                if path.is_file() and not path.is_symlink():
                    files += 1
                    total_bytes += path.stat().st_size
        result[year] = {"files": files, "bytes": total_bytes}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing 2017 ... 2024")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    actual = _inventory(root)
    failures = {
        year: {"expected": EXPECTED[year], "actual": actual[year]}
        for year in EXPECTED
        if actual[year] != EXPECTED[year]
    }
    payload = {
        "root": str(root),
        "expected": EXPECTED,
        "actual": actual,
        "total_files": sum(value["files"] for value in actual.values()),
        "total_bytes": sum(value["bytes"] for value in actual.values()),
        "valid": not failures,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for year, value in actual.items():
            status = "PASS" if value == EXPECTED[year] else "FAIL"
            print(f"{year} {status} files={value['files']} bytes={value['bytes']}")
        print(
            f"total files={payload['total_files']} bytes={payload['total_bytes']} "
            f"valid={str(payload['valid']).lower()}"
        )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
