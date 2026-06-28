# IsaacLab3 Teleop Docker Local Overlay

This is the first-stage Docker setup for migrating the G1 Dex1 Pico teleop work
to Isaac Lab 3.0-beta2.  It keeps all local changes under `IsaacLab3/`.

## Shape

- Image/container are managed by IsaacLab3's official `docker/container.py`.
- Local overlays are the official Isaac Lab development mounts:
  - `source/` -> `/workspace/isaaclab/source`
  - `scripts/` -> `/workspace/isaaclab/scripts`
  - `docs/` -> `/workspace/isaaclab/docs`
  - `tools/` -> `/workspace/isaaclab/tools`
- This project adds one extra output mount:
  - `${TELEOP_OUT_HOST}` -> `/workspace/host/out`
- The compose project name defaults to `isaaclab3teleop` so named volumes do
  not collide with older Isaac Lab 2.3 containers.

## Start

From the `IsaacLab3` directory:

```bash
bash docker/teleop_dev.sh start
```

The wrapper uses:

```text
container: isaac-lab-base-300b2
image:     isaac-lab-base-300b2:latest
out:       ../out -> /workspace/host/out
```

Enter the running container:

```bash
bash docker/teleop_dev.sh enter
```

Stop and remove it:

```bash
bash docker/teleop_dev.sh stop
```

## Verify Teleop Install

Inside the container:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p - <<'PY'
import isaaclab_teleop
import isaacteleop
print("isaaclab_teleop OK", isaaclab_teleop.__file__)
print("isaacteleop OK", isaacteleop.__file__)
PY
```

Check the official recorder CLI:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/tools/record_demos.py --help
```

## Verify Local Overlay

Edit any tracked file under `IsaacLab3/source`, `IsaacLab3/scripts`,
`IsaacLab3/docs`, or `IsaacLab3/tools` on the host, then read the same file from
`/workspace/isaaclab/...` inside the container.  The content should update
without rebuilding the image.

For example:

```bash
echo local-overlay-check > docs/.teleop_overlay_probe
docker exec isaac-lab-base-300b2 cat /workspace/isaaclab/docs/.teleop_overlay_probe
rm docs/.teleop_overlay_probe
```

## Notes

- Do not use the old Isaac Lab 2.x multi-container CloudXR compose path for this
  migration.  Isaac Lab 3.0-beta2 teleoperation runs in a single container and
  the CloudXR runtime is auto-launched by the teleop script.
- Do not bind-mount the whole IsaacLab3 repository over `/workspace/isaaclab`.
  Mounting the four development directories avoids hiding `_isaac_sim` and other
  image-prepared runtime state.
