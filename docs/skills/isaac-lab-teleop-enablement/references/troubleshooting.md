# Troubleshooting

## USD file not found in Docker

Likely owner: Docker mount or default path.

Fix:

- Check the path inside the container, not only on the host.
- Prefer runtime asset bundles under the mounted Isaac Lab repo.
- Keep task defaults pointing at stable repo-relative paths, with env vars as overrides.

## Robot is visible but cropped in screenshots

Likely owner: render camera/viewer placement.

Fix:

- Compute or estimate the robot bounds and move the camera farther away.
- Save a full-body evidence screenshot before judging mesh quality.

## "1 DoF gripper" exposes multiple prismatic joints

Likely owner: hardware abstraction.

Fix:

- Treat physical finger joints as implementation details.
- Define one scalar command per gripper side, then map it to the required symmetric joint targets.
- Document which joints are driven together.

## Motion controller moves the wrist, not the claw center

Likely owner: retarget frame.

Fix:

- Decide whether the controller should align to wrist frame, gripper base, or claw center.
- Add a fixed transform offset if the user expects controller pose to correspond to the physical gripper/claw.
- Validate with a video where the claw, not only the wrist, approaches the object.

## Left and right controllers appear swapped

Likely owner: device handedness or operator handling.

Fix:

- First run a gripper-only handedness test using each controller trigger.
- Then inspect controller handedness labels in raw input.
- Do not blame asset pruning unless raw input and action mapping prove the code path is correct.

## Humanoid intersects the table

Likely owner: scene layout inherited from manipulator-only tasks.

Fix:

- Move the humanoid base and table as separate objects.
- Tune table height so the target object is reachable without excessive bending.
- Use a third-person screenshot/video before tuning grasp behavior.

## Headless camera gives empty buffers

Likely owner: render product/annotator/view lifecycle.

Fix:

- Start with RGB-only sensors.
- Keep render products/viewports alive through recording.
- Reduce camera count or resolution only after confirming the lifecycle is correct.
- Convert HDF5 to MP4 to verify actual data, not only tensor shapes.

## Demo controls work but no HDF5 success exports

Likely owner: success criterion.

Fix:

- Check whether recorder exports only succeeded episodes.
- Verify object geometry conditions and gripper release/open state.
- Add a stable-step window rather than a single-frame success pulse.
