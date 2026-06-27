# Example: G1 Dex1 Pico Visuomotor Teleop

This example is the reference implementation produced during the G1 + Dex1 + Pico enablement work. Use it to test the workflow before applying the skill to a new robot.

## Runtime Assumptions

- Run from the Isaac Lab repo root inside the container: `/workspace/isaaclab`.
- Host output is mounted at `/workspace/host/out`.
- The container can see the compact runtime asset bundle:
  - `/workspace/isaaclab/docs/g1_dex1_runtime_assets/usd/g1_dex1_sim.usd`
  - `/workspace/isaaclab/docs/g1_dex1_runtime_assets/usd/g1_dex1_visuomotor.usda`
  - `/workspace/isaaclab/docs/g1_dex1_runtime_assets/kinematics/g1_dex1_kinematics.urdf`
  - `/workspace/isaaclab/docs/g1_dex1_runtime_assets/kinematics/meshes`

## Gate 0: Asset Path Check

```bash
docker exec isaac-lab-232 test -f /workspace/isaaclab/docs/g1_dex1_runtime_assets/usd/g1_dex1_sim.usd
docker exec isaac-lab-232 test -f /workspace/isaaclab/docs/g1_dex1_runtime_assets/usd/g1_dex1_visuomotor.usda
docker exec isaac-lab-232 test -f /workspace/isaaclab/docs/g1_dex1_runtime_assets/kinematics/g1_dex1_kinematics.urdf
docker exec isaac-lab-232 test -d /workspace/isaaclab/docs/g1_dex1_runtime_assets/kinematics/meshes
```

## Gate A/B: Gripper Smoke

```bash
docker exec -w /workspace/isaaclab isaac-lab-232 ./isaaclab.sh \
  -p scripts/tools/probe_g1_dex1_gripper_smoke.py \
  --headless --device cuda:0 \
  --out-json /workspace/host/out/g1_dex1_runtime_assets_gripper_smoke.json \
  --out-md /workspace/host/out/g1_dex1_runtime_assets_gripper_smoke.md
```

Expected:

- `passed: true`
- Four Dex1 joints exist:
  - `left_dex1_finger_joint_1`
  - `left_dex1_finger_joint_2`
  - `right_dex1_finger_joint_1`
  - `right_dex1_finger_joint_2`
- Target tracking error is near zero for the scripted open/close cases.

## Gate D: Reachability Smoke

```bash
docker exec -w /workspace/isaaclab isaac-lab-232 ./isaaclab.sh \
  -p scripts/tools/probe_g1_dex1_stack_cube_reachability_env.py \
  --headless --device cuda:0 --enable_pinocchio \
  --target-error-threshold 0.25 \
  --cube-distance-threshold 0.30 \
  --out-json /workspace/host/out/g1_dex1_runtime_assets_reachability_smoke.json \
  --out-md /workspace/host/out/g1_dex1_runtime_assets_reachability_smoke.md
```

Expected:

- `passed: true`
- Action terms are `upper_body_ik`, `left_gripper_action`, `right_gripper_action`.
- Action dimensions are `[14, 1, 1]`.
- Wrist body names resolve to `left_wrist_yaw_link` and `right_wrist_yaw_link`.

## Gate C/E: Headless Mock Pico Visuomotor Recording

```bash
docker exec -w /workspace/isaaclab isaac-lab-232 ./isaaclab.sh \
  -p scripts/tools/record_g1_dex1_mock_pico_visuomotor_demo.py \
  --headless --device cuda:0 --enable_pinocchio \
  --dataset-file /workspace/host/out/g1_dex1_runtime_assets_mock_pico_visuomotor.hdf5 \
  --summary /workspace/host/out/g1_dex1_runtime_assets_mock_pico_visuomotor.json \
  --frames 160 --warmup-steps 24 \
  --side right --cube cube_1
```

Expected:

- `passed: true`
- `camera_keys` contains `ego_cam`, `left_wrist_cam`, `right_wrist_cam`.
- `camera_content_ok` is true for all three cameras.
- The cube is lifted or the task trajectory visibly reaches the object.

## Gate E: HDF5 To MP4

```bash
docker exec -w /workspace/isaaclab isaac-lab-232 ./isaaclab.sh \
  -p scripts/tools/hdf5_to_mp4.py \
  --input_file /workspace/host/out/g1_dex1_runtime_assets_mock_pico_visuomotor.hdf5 \
  --output_dir /workspace/host/out/g1_dex1_runtime_assets_mock_pico_visuomotor_videos \
  --input_keys ego_cam left_wrist_cam right_wrist_cam \
  --framerate 30
```

Expected output videos:

- `/workspace/host/out/g1_dex1_runtime_assets_mock_pico_visuomotor_videos/demo_0_ego_cam.mp4`
- `/workspace/host/out/g1_dex1_runtime_assets_mock_pico_visuomotor_videos/demo_0_left_wrist_cam.mp4`
- `/workspace/host/out/g1_dex1_runtime_assets_mock_pico_visuomotor_videos/demo_0_right_wrist_cam.mp4`

## Evidence Summary

After running the gates, summarize artifacts from the host repo root:

```bash
python3 IsaacLab/docs/skills/isaac-lab-teleop-enablement/scripts/collect_g1_dex1_evidence.py \
  --out-dir out \
  --write-json out/g1_dex1_skill_example_evidence.json
```

Use the generated JSON as the example completion evidence.

## Real Pico Variant

For real device testing, use the task documented as `Isaac-G1-Dex1-FixedBase-StackCube-Visuomotor-v0` in `docs/RECORD_VIDEO_DATA.md`. The mock example proves the environment and data path; real Pico testing additionally validates OpenXR device input, handedness, height calibration, and operator success/reset behavior.
