---
name: isaac-lab-teleop-enablement
description: Use when adapting a robot asset into an Isaac Lab teleoperation, imitation-data, or visuomotor recording task. Covers USD/URDF asset probing, gripper and arm control slices, Pico/OpenXR motion-controller retargeting, headless validation videos, task scene setup, camera attachment/FOV tuning, HDF5 recording, and success/reset checks.
license: Apache-2.0
allowed-tools:
  - Read
  - Shell
  - Write
metadata:
  author: Local Isaac Lab Teleop Enablement
  tags:
    - physical-ai
    - isaac-lab
    - teleoperation
    - robotics
    - visuomotor
  domain: robotics
  languages:
    - python
    - markdown
---

# Isaac Lab Teleop Enablement

Use this skill to turn a new robot embodiment, environment, or task into an Isaac Lab teleoperation pipeline that can be verified through concrete evidence: JSON reports, screenshots, MP4 videos, HDF5 demos, and minimal code diffs.

This is an orchestrator skill. Do not try to solve a new robot adaptation as one large code edit. Move through small validation gates and keep every gate reproducible.

## First Action

Classify the request:

1. **Run the example**: use the G1 Dex1 Pico visuomotor example to prove the local Docker/Isaac Lab/headless/data-export loop works.
2. **Adapt a new robot**: collect the robot USD/URDF, reference task, control device, target task, and expected observations.
3. **Extend an existing adapted task**: identify the current passing gate, then continue from the next gate instead of restarting.
4. **Debug a failure**: map the symptom to the owning layer: asset, action, retargeter, scene, camera/rendering, recorder, or success criterion.

Before editing code, read `references/workflow.md`. For the shipped example, also read `references/g1-dex1-example.md`.

## Required Workflow

Follow these gates in order unless the user explicitly scopes the task to a later gate.

| Gate | Goal | Evidence |
| --- | --- | --- |
| A. Asset probe | Load USD/URDF, list articulation root, joints, links, gripper limits, camera candidate links. | JSON/Markdown report plus one full-body screenshot or short video. |
| B. Control slice | Prove gripper, wrist/arm, then combined upper-body action can move without full teleop. | Scripted MP4 plus joint error/IK summary. |
| C. Mock teleop | Feed simulated controller poses/triggers through the same retargeter/action path used by real teleop. | JSON/HDF5/MP4 showing raw input to action to robot motion. |
| D. Scene/task | Add table, objects, robot pose, and success/reset logic. Resolve height and reachability. | Third-person MP4 and reachability/success report. |
| E. Camera/data | Attach or overlay ego/wrist cameras, tune FOV, record RGB observations, export videos from HDF5. | HDF5 plus per-camera MP4 or composite MP4. |
| F. Real device | Run the same env with real Pico/OpenXR controllers and record usable demos. | HDF5 demos, success/export logs, and converted videos. |

Stop at the first red gate unless the user asks for best-effort exploration. Report the failed owning layer and preserve the artifacts that prove the failure.

## Operating Rules

- Prefer the repo's existing Isaac Lab patterns, task registration style, action manager terms, device retargeters, and recorder scripts.
- Keep new robot enablement local to task/env config files until a core Isaac Lab change is proven generally useful.
- Treat USD and URDF as separate runtime contracts: USD drives simulation/rendering; URDF/Pinocchio drives kinematics and frame semantics.
- Keep original assets unchanged when possible. Use flattened runtime USDs and thin USDA overlays for task-specific cameras.
- Do not validate teleop only by logs. Produce visual evidence for each control or camera change.
- When headless camera recording fails, first reduce annotators to required RGB, keep active viewports/render products alive, and inspect image buffer sizes before changing task semantics.
- For grippers, define an explicit scalar contract such as `open01 in [0, 1] -> joint targets`. Clamp and rate-limit before writing joint targets.
- For motion controllers, distinguish raw controller data, retargeted task-space targets, and joint targets. Verify handedness and frame offsets visually.
- For success criteria, match the data-collection intent: task geometry, release/open state, and a short stable-step window.

## References

Read only the needed files:

- `references/workflow.md`: full adaptation workflow, inputs, outputs, and gate order.
- `references/g1-dex1-example.md`: runnable example based on G1 + Dex1 + Pico motion controllers + three RGB cameras.
- `references/validation-gates.md`: evidence checklist and failure ownership.
- `references/camera-visuomotor.md`: camera attachment, overlay USD, FOV, headless RGB recording, and HDF5-to-video checks.
- `references/troubleshooting.md`: common symptoms from this enablement work and likely fixes.

## Example

When the user asks to test the skill itself, run the G1 Dex1 example gate sequence in `references/g1-dex1-example.md`. The minimum passing example is:

1. Confirm the compact runtime assets exist.
2. Run the right gripper smoke test.
3. Run stack-cube reachability.
4. Run headless mock Pico visuomotor recording.
5. Convert the HDF5 camera streams to MP4.
6. Summarize evidence with `scripts/collect_g1_dex1_evidence.py`.

Do not mark the skill use complete until the artifacts listed by the example exist and the summary clearly says which gates passed, failed, or were skipped.

## Output Format

For every adaptation pass, return:

- Overall status: `passed`, `blocked`, `failed`, or `needs_rerun`.
- Robot/task summary: assets, reference task, control device, action space, observations, and success criterion.
- Gate table: each gate status, commands run, artifacts generated, and failure owner if any.
- Code/asset changes: files modified, runtime asset paths, env vars added, and any core Isaac Lab changes.
- Next work: only items needed to improve robustness or move to real data collection.
