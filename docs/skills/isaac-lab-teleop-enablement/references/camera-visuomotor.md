# Camera And Visuomotor Notes

Camera work is a first-class gate for visuomotor teleop tasks. Treat it as a control/data problem, not as a cosmetic rendering step.

## Attachment Strategy

Preferred order:

1. Attach to an existing stable USD prim/link when the robot USD exposes one.
2. Use a thin USDA overlay that references the simulation USD and defines camera prims under known robot links.
3. Use runtime-updated world cameras only as a debug fallback, because they do not prove the camera is actually attached to the robot.

For Isaac Lab sensors, if a camera prim already exists in the USD/overlay, use a `CameraCfg` that binds the existing prim instead of spawning a different camera elsewhere.

## Parent Link Selection

- Wrist cameras should attach near the wrist or gripper base, but their optical axis should see the claw/object interaction.
- Ego/chest cameras can attach under torso/head-like links. If a dedicated `head_link` or `d435_link` is absent from the composed USD, use `torso_link` with a fixed offset.
- A camera offset is not a physical shell. It is a virtual sensor pose relative to the parent link. This is acceptable for data collection if the pose is stable and documented.

## FOV

FOV is controlled by lens parameters:

- Smaller `focalLength` means wider FOV.
- Larger `horizontalAperture` means wider FOV.

Practical default:

- Wrist cameras: keep local, moderate FOV to preserve manipulation detail.
- Ego/chest camera: use a wider FOV so table, objects, and surrounding context are visible.

Important: old HDF5 recordings cannot be widened after the fact. Change the camera config, rerun recording, then export new MP4s.

## Headless RGB Recording

For headless recording:

- Enable cameras/rendering in the app launch path.
- Record only required data types first; RGB-only is a safer baseline than RGB+depth.
- If SyntheticData reports empty `LdrColorSD` or `DistanceToImagePlaneSD`, inspect which annotator is returning a zero-sized buffer before changing task logic.
- Keep a visual validation loop: HDF5 -> MP4 -> human review.

## Visual Acceptance

Each camera should have a defined purpose:

- Ego/chest: global task context and object layout.
- Left wrist: left-hand manipulation details.
- Right wrist: right-hand manipulation details.

If one wrist camera sees blank space during most of a right-hand task, that can be acceptable. The acting hand's camera must see useful content near the approach/grasp/lift phase.
