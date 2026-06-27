#!/usr/bin/env python3
"""Collect evidence files for the G1 Dex1 skill example.

This script does not run Isaac Sim. It checks whether the expected output
artifacts from the example commands exist and summarizes key JSON fields when
available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_FILES = {
    "gripper_smoke_json": "g1_dex1_runtime_assets_gripper_smoke.json",
    "gripper_smoke_md": "g1_dex1_runtime_assets_gripper_smoke.md",
    "reachability_json": "g1_dex1_runtime_assets_reachability_smoke.json",
    "reachability_md": "g1_dex1_runtime_assets_reachability_smoke.md",
    "mock_visuomotor_hdf5": "g1_dex1_runtime_assets_mock_pico_visuomotor.hdf5",
    "mock_visuomotor_json": "g1_dex1_runtime_assets_mock_pico_visuomotor.json",
}

EXPECTED_VIDEOS = {
    "ego_cam_mp4": "g1_dex1_runtime_assets_mock_pico_visuomotor_videos/demo_0_ego_cam.mp4",
    "left_wrist_cam_mp4": "g1_dex1_runtime_assets_mock_pico_visuomotor_videos/demo_0_left_wrist_cam.mp4",
    "right_wrist_cam_mp4": "g1_dex1_runtime_assets_mock_pico_visuomotor_videos/demo_0_right_wrist_cam.mp4",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"_error": str(exc)}
    return data if isinstance(data, dict) else {"_value": data}


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="out", help="Host output directory to inspect.")
    parser.add_argument("--write-json", help="Optional path to write the evidence summary.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary: dict[str, Any] = {
        "out_dir": str(out_dir),
        "files": {},
        "json_summaries": {},
        "gates": {},
        "overall_status": "unknown",
    }

    for key, rel_path in {**EXPECTED_FILES, **EXPECTED_VIDEOS}.items():
        summary["files"][key] = file_info(out_dir / rel_path)

    gripper = load_json(out_dir / EXPECTED_FILES["gripper_smoke_json"])
    reachability = load_json(out_dir / EXPECTED_FILES["reachability_json"])
    mock = load_json(out_dir / EXPECTED_FILES["mock_visuomotor_json"])

    if gripper is not None:
        summary["json_summaries"]["gripper_smoke"] = {
            "passed": gripper.get("passed"),
            "num_joints": gripper.get("num_joints"),
            "num_bodies": gripper.get("num_bodies"),
            "driven_joints": gripper.get("driven_joints"),
        }
    if reachability is not None:
        summary["json_summaries"]["reachability"] = {
            "passed": reachability.get("passed"),
            "action_terms": reachability.get("action_terms"),
            "action_term_dims": reachability.get("action_term_dims"),
        }
    if mock is not None:
        summary["json_summaries"]["mock_visuomotor"] = {
            "passed": mock.get("passed"),
            "camera_keys": mock.get("camera_keys"),
            "camera_content_ok": mock.get("camera_content_ok"),
            "max_cube_lift_m": mock.get("max_cube_lift_m"),
            "min_gripper_center_to_cube_m": mock.get("min_gripper_center_to_cube_m"),
        }

    summary["gates"] = {
        "asset_and_gripper": bool(gripper and gripper.get("passed") is True),
        "reachability": bool(reachability and reachability.get("passed") is True),
        "mock_visuomotor": bool(mock and mock.get("passed") is True),
        "camera_video_export": all(
            summary["files"][key]["exists"] and summary["files"][key]["bytes"] > 0
            for key in EXPECTED_VIDEOS
        ),
    }
    summary["overall_status"] = "passed" if all(summary["gates"].values()) else "incomplete"

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.write_json:
        write_path = Path(args.write_json)
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(text + "\n", encoding="utf-8")
    return 0 if summary["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
