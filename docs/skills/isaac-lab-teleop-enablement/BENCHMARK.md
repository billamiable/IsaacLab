# Benchmark

Use this benchmark to test whether the skill can drive a complete example run.

## Task

Run the G1 Dex1 Pico visuomotor example and produce an evidence summary.

## Required Commands

Follow `references/g1-dex1-example.md`:

1. Asset path checks.
2. Gripper smoke test.
3. Stack-cube reachability smoke.
4. Headless mock Pico visuomotor recording.
5. HDF5-to-MP4 camera export.
6. Evidence collection script.

## Passing Criteria

- `g1_dex1_runtime_assets_gripper_smoke.json` reports `passed: true`.
- `g1_dex1_runtime_assets_reachability_smoke.json` reports `passed: true`.
- `g1_dex1_runtime_assets_mock_pico_visuomotor.json` reports `passed: true`.
- Three MP4 files exist for `ego_cam`, `left_wrist_cam`, and `right_wrist_cam`.
- `collect_g1_dex1_evidence.py` reports `overall_status: passed`.

## Failure Classification

When the benchmark fails, classify the first red gate as:

- `asset_path`
- `asset_articulation`
- `gripper_control`
- `ik_reachability`
- `mock_retargeting`
- `headless_camera`
- `hdf5_video_export`
- `unknown`
