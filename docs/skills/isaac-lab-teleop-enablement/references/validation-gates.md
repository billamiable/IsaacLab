# Validation Gates

Use this table to decide whether an adaptation is ready to move forward.

| Gate | Required evidence | Strong pass signal | Common failure owner |
| --- | --- | --- | --- |
| Asset path | File existence checks in the same container path used by the env. | All task defaults resolve without env var overrides. | Docker mount, asset packaging, path defaults. |
| Articulation | Joint/body/link report. | Expected joints, limits, and root prim are present. | USD export, composition, asset pruning. |
| Gripper | Scripted open/close JSON and MP4. | Open scalar maps monotonically to physical finger joints. | Joint names, limits, mimic/parallel-jaw abstraction. |
| IK/action | Reachability JSON and video. | Action terms and dimensions match the intended robot control slice. | URDF, Pinocchio model, frame names, action term config. |
| Mock teleop | Raw controller trace, retargeted target, action tensor, video. | Simulated trigger/pose changes move the intended hand/gripper. | Retargeter, handedness, absolute/relative mode, offsets. |
| Scene layout | Third-person MP4 and object pose report. | Robot/table/object are reachable and not intersecting. | Env cfg, table height, humanoid base pose. |
| Camera | HDF5 image keys and MP4. | Non-empty RGB tensors with useful task content. | Camera prim path, orientation, FOV, headless rendering. |
| Recording | HDF5 with demos and metadata. | Successful episode exports after success/reset. | Success criterion, recorder flags, export-succeeded-only. |

## Evidence Rules

- A log-only run is not enough for control or camera changes.
- A screenshot is enough for asset and layout sanity, but use MP4 for motion.
- HDF5 existence is not enough; inspect image keys, shapes, frame counts, and at least one converted video.
- If a gate fails, report the first owning layer rather than stacking speculative fixes across multiple layers.

## Completion Report Fields

Use this shape for final reports:

```text
status: passed | failed | blocked | needs_rerun
robot:
  sim_usd:
  kinematics_urdf:
  mesh_root:
task:
  task_id:
  reference_task:
  control_device:
  action_terms:
  observation_keys:
gates:
  - name:
    status:
    command:
    artifacts:
    notes:
changes:
  - file:
    reason:
next_work:
  - item:
```
