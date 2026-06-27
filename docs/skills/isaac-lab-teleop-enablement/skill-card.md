# Skill Card: Isaac Lab Teleop Enablement

## Purpose

Guide an agent through adapting a robot asset into an Isaac Lab teleoperation or visuomotor data-collection task using incremental validation gates.

## Scope

- USD/URDF runtime asset probing.
- Gripper and upper-body control slices.
- Pico/OpenXR motion-controller retargeting.
- Headless videos and screenshots.
- Scene, object, and success/reset setup.
- Camera attachment/FOV tuning.
- HDF5 recording and MP4 export.

## Example

The bundled example is G1 + Dex1 + Pico motion controllers with ego, left wrist, and right wrist RGB cameras. It uses the compact runtime assets under `docs/g1_dex1_runtime_assets/` and the commands in `references/g1-dex1-example.md`.

## Expected Evidence

- JSON/Markdown smoke reports.
- Headless rendered MP4s or screenshots.
- HDF5 recording with image observations.
- Per-camera MP4 exports.
- Evidence summary JSON from `scripts/collect_g1_dex1_evidence.py`.

## Known Limits

- The skill does not guarantee physical grasp tuning for arbitrary grippers.
- Real device testing still requires working OpenXR/Pico/CloudXR infrastructure.
- Some gates require a GPU Isaac Sim/Isaac Lab runtime and cannot be fully validated in a CPU-only environment.
