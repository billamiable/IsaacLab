# Workflow

This reference is the canonical adaptation flow for a new Isaac Lab teleop embodiment or task.

## Inputs To Collect

Ask for or discover:

- Robot simulation USD and any source USD/URDF/assets used to build it.
- Kinematics URDF and mesh root for Pinocchio/Pink IK when task-space control is required.
- Reference Isaac Lab task or env closest to the target behavior.
- Target control device: Pico/OpenXR motion controllers, keyboard, mouse, scripted mock, or another device.
- Target action semantics: wrist pose, end-effector pose, gripper scalar, mobile base, waist, or locomotion.
- Target observations: proprioception only, RGB, RGB-D, wrist cameras, ego/chest/head cameras.
- Target task success: object lift, stack, place, release, stable hold, or data-only recording.

## Gate A: Asset Probe

Goal: prove the robot asset is loadable and inspectable before building a task around it.

Actions:

1. Load the USD headlessly with Isaac Lab/Isaac Sim.
2. Print articulation root, body names, joint names, joint limits, and candidate camera parent prims.
3. For grippers, identify all physical joints involved, even when the user describes it as "1 DoF".
4. Save one screenshot or MP4 with a camera positioned to show the whole robot.

Pass criteria:

- USD exists in the container path used by the task.
- Articulation has expected joints and bodies.
- No missing mesh, NaN, simulation explosion, or broken root prim.
- Visual evidence shows the full robot, not a cropped limb.

## Gate B: Control Slice

Goal: prove the smallest action path before full teleop.

Actions:

1. Move only the gripper through open/close targets.
2. Move one wrist or upper-body trajectory without object interaction.
3. Combine arm/wrist with gripper.
4. Record videos for each slice.

Pass criteria:

- Joint targets and measured positions agree within a reasonable tolerance.
- The end effector being controlled is the intended gripper/claw center, not only the wrist link.
- Left/right action mapping is visually correct.

## Gate C: Mock Teleop

Goal: run the same code path that real teleop will use, but with synthetic controller input.

Actions:

1. Define a raw input schema: controller handedness, pose, trigger/grip scalar, buttons, and start/reset events.
2. Feed scripted raw packets through the retargeter.
3. Verify the retargeted target pose and gripper scalar before writing actions.
4. Render the resulting robot motion.

Pass criteria:

- Raw controller data, retargeted targets, and final action tensors are all observable in logs or JSON.
- The mock path uses the same retargeter/action term as real device control.
- Trigger release produces an open gripper when success requires release.

## Gate D: Scene And Task

Goal: put the robot into a meaningful task scene without solving all manipulation physics at once.

Actions:

1. Place robot, table, and objects using humanoid-appropriate heights and offsets.
2. Avoid inheriting manipulator-only assumptions such as a table intersecting a humanoid torso.
3. Tune cube/object positions for reachability before tuning grasp physics.
4. Add success criteria late, after the visual task flow is plausible.

Pass criteria:

- Robot is not intersecting the table or environment.
- Object is reachable by the intended hand.
- Third-person video shows the intended task geometry clearly.
- Success/reset behavior matches the demonstration collection goal.

## Gate E: Camera And Data

Goal: convert the task into a visuomotor data source.

Actions:

1. Decide camera semantics: ego/chest/head for global context, wrist cameras for local manipulation.
2. Prefer a thin USDA overlay that references the simulation USD and adds camera prims under stable robot links.
3. Configure Isaac Lab camera sensors to bind existing camera prims.
4. Record HDF5 with camera observations and convert the image keys to MP4.

Pass criteria:

- HDF5 contains expected image keys with non-empty RGB tensors.
- Camera MP4s show useful content for the task.
- FOV and offsets are tuned with newly recorded data, not by post-processing old HDF5.

## Gate F: Real Device

Goal: use real controllers to produce a data artifact.

Actions:

1. Start the container with Isaac Lab repo mounted and an output directory mounted.
2. Run `record_demos.py` with the real teleop device and the adapted task id.
3. Verify control handedness, height calibration, gripper mapping, and start/reset events.
4. Convert any successful HDF5 demos to MP4 for review.

Pass criteria:

- Operator can control the intended hands/grippers.
- Recorder exports HDF5 when success is met, or logs clearly show why demos were not exported.
- Converted videos show the action and camera observations expected by the policy.
