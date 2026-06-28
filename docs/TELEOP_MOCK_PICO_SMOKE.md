# Mock Pico Motion Controller Teleop Smoke Tests

This note records the Isaac Lab 3.0-beta2 smoke tests used to validate the
Pico / motion-controller teleop stack before porting the G1 Dex1 task.

The goal is not to prove that a physical Pico device is connected.  The goal is
to validate the layers that can be checked without a headset:

- Isaac Lab 3 Docker local overlay is active.
- `isaaclab_teleop` and `isaacteleop` are installed and importable.
- Synthetic controller / trigger input can exercise retargeter logic.
- The official G1 motion-controller teleop pipeline can be constructed.
- The official fixed-base G1 teleop environment can reset and step headlessly.
- The non-GUI path for future MCAP replay is available.

## Container

Start the local overlay container from `IsaacLab3/`:

```bash
bash docker/teleop_dev.sh start
```

The expected container name is:

```text
isaac-lab-base-300b2
```

The local overlay maps host source into the running container, so edits under
`IsaacLab3/source`, `IsaacLab3/scripts`, `IsaacLab3/docs`, and
`IsaacLab3/tools` are visible under `/workspace/isaaclab/...` without rebuilding.

## Import Smoke

```bash
docker exec isaac-lab-base-300b2 bash -lc \
  'cd /workspace/isaaclab && ./isaaclab.sh -p - <<'"'"'PY'"'"'
import isaaclab_teleop
import isaacteleop
print("isaaclab_teleop", isaaclab_teleop.__file__)
print("isaacteleop", isaacteleop.__file__)
PY'
```

Expected result:

- `isaaclab_teleop` resolves to `/workspace/isaaclab/source/...`.
- `isaacteleop` resolves to the Isaac Sim Python site-packages path.

This proves the local source overlay and the installed Isaac Teleop package are
both visible inside the container.

## Pure Teleop Unit Tests

These tests do not require a physical Pico device.

```bash
docker exec isaac-lab-base-300b2 bash -lc \
  'cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest source/isaaclab_teleop/test/test_target_frame_rebase.py -q'

docker exec isaac-lab-base-300b2 bash -lc \
  'cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest source/isaaclab_teleop/test/test_cloudxr_lifecycle.py -q'
```

Observed result:

```text
17 passed
22 passed
```

Coverage:

- Target-frame rebasing math.
- Config-driven target frame selection.
- Mocked CloudXR lifecycle behavior.
- Non-GUI lifecycle code paths.

## Synthetic Motion-Controller Retargeter Test

```bash
docker exec isaac-lab-base-300b2 bash -lc \
  'cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest source/isaaclab_teleop/test/test_retargeters.py -q --tb=short'
```

Observed result:

```text
11 passed
```

This file includes deprecated OpenXR retargeter tests that directly feed mock
controller arrays such as:

```text
[pose, inputs]
```

where the input vector contains thumbstick, trigger, squeeze, and button values.
The useful checks for Pico-style motion controllers are:

- `G1LowerBodyStandingMotionControllerRetargeter`: thumbstick to locomotion.
- `G1TriHandUpperBodyMotionControllerGripperRetargeter`: trigger threshold to
  gripper open/close state.
- `G1TriHandUpperBodyMotionControllerRetargeter`: controller pose plus hand
  input to a 28D upper-body action.

This proves that synthetic controller values can exercise the mapping logic, but
it is not the full Isaac Teleop 3.0 session path.  The full path should use
`isaacteleop` pipelines, live CloudXR, or MCAP replay.

## Official Fixed-Base G1 Pipeline Smoke

The closest official reference for the current G1 Dex1 migration is:

```text
Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0
```

Its pipeline is defined in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg.py
```

Run a lightweight pipeline construction smoke:

```bash
docker exec isaac-lab-base-300b2 bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p - <<'"'"'PY'"'"'
from isaaclab_tasks.manager_based.locomanipulation.pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
    _build_g1_upper_body_pipeline,
)

pipeline, retargeters = _build_g1_upper_body_pipeline()
cfg = FixedBaseUpperBodyIKG1EnvCfg()

print("pipeline", type(pipeline))
print("retargeters", [type(r).__name__ for r in retargeters])
print("action_cfg", type(cfg.actions.upper_body_ik).__name__)
print("isaac_teleop", type(cfg.isaac_teleop).__name__, cfg.isaac_teleop is not None)
print("teleoperation_active_default", cfg.isaac_teleop.teleoperation_active_default)
PY'
```

Expected shape:

- Pipeline type: `OutputCombiner`.
- Retargeters: two `Se3AbsRetargeter` instances for left/right wrists.
- Action config: `PinkInverseKinematicsActionCfg`.
- Isaac Teleop config exists.
- `teleoperation_active_default` is `False`, so a live XR session still needs a
  START event from the client.

## Mock Controller to Real Pink IK Video Smoke

The unit tests above only validate pieces of the stack.  The stronger smoke test
is the scripted video render below: it creates a Pico-like right motion-controller
stream, maps it into the official 28D fixed-base G1 action, and lets the real
Isaac Lab action manager plus Pink IK solver move the robot.

Script:

```text
scripts/tools/render_mock_pico_g1_fixed_base_ik_video.py
```

Run from the host against the local overlay container:

```bash
docker exec isaac-lab-base-300b2 bash -lc '
cd /workspace/isaaclab
./isaaclab.sh -p scripts/tools/render_mock_pico_g1_fixed_base_ik_video.py \
  --headless --device cuda:0 --rendering_mode balanced \
  --frames 144 --fps 24 --width 960 --height 540 \
  --out-mp4 /workspace/host/out/isaaclab3/mock_pico_g1_fixed_base_ik/mock_pico_g1_fixed_base_ik.mp4 \
  --out-json /workspace/host/out/isaaclab3/mock_pico_g1_fixed_base_ik/mock_pico_g1_fixed_base_ik_summary.json \
  --out-frames /workspace/host/out/isaaclab3/mock_pico_g1_fixed_base_ik/frames
'
```

Output is intentionally namespaced under `out/isaaclab3/` so it does not mix
with Isaac Lab 2.3.2 artifacts.

The mock input model is:

- right controller absolute pose -> right wrist target pose
- right trigger scalar -> right TriHand close/open command
- left wrist is held at its reset pose

The script records, for sampled frames, the mock controller target, trigger
value, actual right wrist pose, wrist tracking error, and wrist motion delta.
This verifies the important integration point: a controller-like command becomes
an Isaac Lab action, then Pink IK solves and moves the robot.

Observed output from the first successful run:

```text
video:   out/isaaclab3/mock_pico_g1_fixed_base_ik/mock_pico_g1_fixed_base_ik.mp4
summary: out/isaaclab3/mock_pico_g1_fixed_base_ik/mock_pico_g1_fixed_base_ik_summary.json
frames:  out/isaaclab3/mock_pico_g1_fixed_base_ik/frames/
passed:  true
action_dim: 28
max_right_wrist_motion_delta_m: 0.2086
min_right_wrist_tracking_error_m: 0.0029
video: 6.0s, 960x540, 24 fps
```

This still is not a live `isaacteleop`/CloudXR session.  It is the practical
pre-MCAP integration slice: mock Pico semantics enter the same action shape that
the official fixed-base G1 teleop pipeline emits, and the actual Pink IK solver
is exercised in simulation.  The next stronger non-GUI test is MCAP replay from
a real Pico recording.

## Headless Environment Smoke

This validates that the official fixed-base G1 task can launch, reset, and step
in headless Kit.

```bash
docker exec isaac-lab-base-300b2 bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p - <<'"'"'PY'"'"'
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

task = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)

print("task", task)
print("isaac_teleop", type(env_cfg.isaac_teleop).__name__, env_cfg.isaac_teleop is not None)

env = gym.make(task, cfg=env_cfg).unwrapped
obs, _ = env.reset()
print("reset_ok", sorted(obs.keys()) if isinstance(obs, dict) else type(obs).__name__)
print("action_dim", env.action_manager.total_action_dim)

action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
for _ in range(3):
    env.step(action)
print("step_ok", 3)

env.close()
simulation_app.close()
PY'
```

Observed result:

- The environment launches headlessly.
- `isaac_teleop` exists on the env config.
- Action dimension is `28`.
- `env.reset()` and several `env.step()` calls complete.

Note: zero actions are not valid wrist-pose commands for Pink IK, so warnings
such as "IK quadratic solver could not find a solution" or "Solution to IK
contains NaN" are expected for this smoke.  They do not indicate a Docker or
teleop startup failure.

## Headless Versus GUI XR

In GUI mode, clicking **Start XR** creates the Kit/OpenXR session and exposes
OpenXR handles to Isaac Teleop.

In headless mode there is no button.  The equivalent session setup is driven by
command-line flags:

```bash
--xr --headless --cloudxr_env cloudxrjs
```

The CloudXR runtime is auto-launched by the teleop script in Isaac Lab
3.0-beta2.  A physical Pico/Quest client still needs to connect and send START
or STOP control events.

## MCAP Replay Path

For true non-GUI regression testing, record one live Pico session to MCAP and
then replay it headlessly.

Live recording example:

```bash
docker exec -it isaac-lab-base-300b2 bash -lc '
cd /workspace/isaaclab
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 \
  --num_demos 1 \
  --dataset_file /workspace/host/out/fixed_base_g1_demo.hdf5 \
  --mcap_record_path /workspace/host/out/fixed_base_g1_demo.mcap \
  --xr --headless --cloudxr_env cloudxrjs --device cuda:0
'
```

Replay example:

```bash
docker exec isaac-lab-base-300b2 bash -lc '
cd /workspace/isaaclab
./isaaclab.sh -p scripts/environments/teleoperation/teleop_replay_agent.py \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 \
  --replay_file /workspace/host/out/fixed_base_g1_demo.mcap \
  --stats_output_file /workspace/host/out/fixed_base_g1_replay_stats.json \
  --headless --device cuda:0
'
```

Replay mode uses `SessionMode.REPLAY`, so it does not need GUI Start XR or a
live Pico device.  The recorded MCAP is the input source.

## G1 Dex1 Headless Video Gate

This is the first Lab3 G1+Dex1 migration gate, not the official TriHand G1
smoke.  It loads the custom G1+Dex1 runtime assets, creates the registered
`Isaac-G1-Dex1-FixedBase-StackCube-v0` task, and sends a scripted mock Pico
right-controller stream into the real Lab3 action manager and Pink IK action.

The Lab3 action shape is `18`: left wrist pose `7`, right wrist pose `7`, and
Dex1 gripper joint targets `4`.  The right trigger maps linearly from Dex1 open
to close on the two right prismatic finger joints.

Lab3 asset and root-pose quaternions are `xyzw`; the G1 Dex1 env uses
`IDENTITY_QUAT_XYZW = (0, 0, 0, 1)` for robot, table, and cube init states.

Validated physical grasp command:

```bash
docker exec isaac-lab-base-300b2 bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh \
  -p scripts/tools/render_mock_pico_g1_dex1_stack_cube_video.py \
  --headless --device cuda:0 --rendering_mode balanced \
  --grasp-mode physical \
  --frames 192 --fps 24 --width 960 --height 540 --keyframe-every 48 \
  --out-mp4 /workspace/host/out/isaaclab3/g1_dex1_stack_cube/physical_grasp_v2/mock_pico_g1_dex1_stack_cube_physical_grasp_v2.mp4 \
  --out-json /workspace/host/out/isaaclab3/g1_dex1_stack_cube/physical_grasp_v2/mock_pico_g1_dex1_stack_cube_physical_grasp_v2.json \
  --out-frames /workspace/host/out/isaaclab3/g1_dex1_stack_cube/physical_grasp_v2/frames'
```

Observed physical result:

- `passed: true`
- `grasp_mode: physical`
- `used_assisted_grasp: false`
- `ever_assisted_attached: false`
- `max_lift_m: 0.07877188920974731`
- `max_consecutive_lift_frames: 41` with `required_physical_hold_frames: 18`
- `min_attach_distance_m: 0.011476660147309303`
- Output video: `out/isaaclab3/g1_dex1_stack_cube/physical_grasp_v2/mock_pico_g1_dex1_stack_cube_physical_grasp_v2.mp4`
- Output summary: `out/isaaclab3/g1_dex1_stack_cube/physical_grasp_v2/mock_pico_g1_dex1_stack_cube_physical_grasp_v2.json`

This is the required G1 Dex1 grasp gate.  In `physical` mode the script never
writes the cube root pose during grasp.  Success requires the cube to stay above
the lift threshold for a minimum number of consecutive frames.  The older
assisted behavior is still available with `--grasp-mode assisted`, but it should
only be used to debug the custom USD, kinematics URDF, Lab3 Pink IK action,
Dex1 gripper joints, headless rendering, and output-video path.

Contact notes:

- The task binds high-friction material to the blocks and table for the physical
  gate.
- Robot gripper collision friction currently relies on the asset defaults.  A
  root-level material bind is intentionally not used because many robot collision
  prims are instanced; stricter future tuning should use non-instanced Dex1
  collision prims or a more specific USD overlay.
- The quaternion audit for this custom G1 Dex1 path found no remaining Lab2
  identity quaternion literals; the task uses `xyzw` root-pose quaternions.

## Practical Interpretation

Use these checks as a staged gate:

1. Import smoke: container and overlay are correct.
2. Pure unit tests: teleop math and lifecycle code are importable and stable.
3. Deprecated retargeter mock: synthetic trigger/thumbstick mappings work.
4. Official G1 pipeline construction: Isaac Teleop 3.0 pipeline builder works.
5. Headless env smoke: Isaac Sim can load and step the official G1 teleop task.
6. G1 Dex1 physical grasp video gate: custom robot asset, Pink IK, Dex1 gripper tail, contact grasp, and headless video output work.
7. MCAP replay: future end-to-end non-GUI validation path for real Pico data.

For the G1 Dex1 migration, the most relevant reference is the fixed-base G1
pipeline.  The Dex1 task should replace the TriHand hand output with the Dex1
1-DoF gripper mapping while keeping the same staged validation strategy.
