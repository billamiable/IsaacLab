# G1 Dex1 USD/URDF Asset Relationship

This note documents the asset layout used by
`Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0` and the planned path for
flattening the simulation USD later.

## Runtime Entry Points

The current pipeline has two independent robot descriptions:

```text
Isaac Sim / PhysX articulation
  -> docs/g1_dex1_assets/g1_29dof_dex1_1_v4_test_good.usd

Pink IK / Pinocchio kinematics
  -> docs/g1_dex1_assets/g1_29dof_mode_15_with_dex1_1.urdf
  -> docs/g1_dex1_assets/meshes/
```

The USD creates the simulated articulation. The URDF is not used to spawn the
robot in Isaac Sim; it is used by Pink IK to build the kinematic tree and frame
model.

## Simulation USD Composition

`g1_29dof_dex1_1_v4_test_good.usd` is the simulation entry point, but it is not
a fully self-contained single-file asset. It relies on USD composition layers.
The current verified dependency bundle is:

```text
docs/g1_dex1_assets/
├── g1_29dof_dex1_1_v4_test_good.usd
├── g1_29dof_dex1_1_v3.usd
├── g1_29dof_dex1_1_v1.usd
├── dex1_1_gripper.usd
├── g1_29dof_mode_15_with_dex1_1/
│   ├── g1_29dof_mode_15_with_dex1_1.usd
│   └── configuration/*.usd
└── g1_29dof_rev_1_0_with_inspire_hand_FTP/
    ├── g1_29dof_no_hand.usd
    └── configuration/*.usd
```

Conceptually, the final assembled robot is:

```text
G1 no-hand body
  + Dex1 gripper subtree
  + wrist/gripper joint and physics relationships
  + final overrides saved in v1/v3/v4 layers
```

## What The Two Large USD Directories Mean

### `g1_29dof_rev_1_0_with_inspire_hand_FTP/`

This directory is the G1 no-hand body asset source. Its key entry is:

```text
g1_29dof_rev_1_0_with_inspire_hand_FTP/g1_29dof_no_hand.usd
```

Its `configuration/` layers provide the body hierarchy, physics schemas, and
articulation data needed by the no-hand G1 base robot. The 38M base layer in
this directory was tested explicitly: removing it breaks articulation creation
with a missing/invalid `root_joint`.

In the final assembled asset, this directory should be understood as the
authoritative source for the G1 body after the original hand was removed.

### `g1_29dof_mode_15_with_dex1_1/`

This directory is the USD output from importing the full
`g1_29dof_mode_15_with_dex1_1.urdf`. It contains a complete imported robot USD,
but in our assembled asset it should not be treated as the intended G1 body
source.

Its role is an intermediate donor/reference layer. The final `v4` asset uses
USD composition that depends on this imported layer, mainly for the Dex1/wrist
connection area:

- Dex1 gripper prims and related physics schema.
- Wrist/gripper joint relationships.
- Importer-generated USD configuration layers required for the composed stage.

This is why the directory name looks like "full G1 + Dex1", even though the
final assembly conceptually uses the no-hand G1 body plus the Dex1 gripper.
USD composition operates at layer/prim level; until the stage is flattened or
rebuilt, we cannot delete only the "unused body-looking" parts of this imported
USD safely.

## Current Pruning Result

The current checked-in asset bundle is about 90M. It keeps only the simulation
USD dependency layers verified by headless smoke tests plus the Pink IK URDF and
its 40 referenced STL meshes.

These were tested and removed because the current pipeline does not need them:

```text
g1_29dof_dex1_1_v2.usd
dex1_1/
old inspire-hand meshes and old root USD/URDF files
```

The final validation command was:

```bash
docker exec -w /workspace/isaaclab isaac-lab-232 ./isaaclab.sh \
  -p scripts/tools/probe_g1_dex1_stack_cube_reachability_env.py \
  --headless --device cuda:0 --enable_pinocchio --steps-per-command 20
```

Expected result:

```text
Overall result: PASS
action terms: ['upper_body_ik', 'left_gripper_action', 'right_gripper_action']
dims: [14, 1, 1]
```

## Future Flattened USD Plan

The goal of flattening is to replace the multi-layer simulation USD bundle with
one simulation USD file:

```text
g1_29dof_dex1_1_v4_test_good_flattened.usd
```

Flattening should reduce dependency complexity:

```text
Before:
  v4 USD
    -> v3/v1 USD
    -> no_hand USD + configuration layers
    -> imported G1+Dex1 USD + configuration layers
    -> dex1_1_gripper USD

After, if validated:
  flattened simulation USD
  Pink IK URDF
  URDF meshes
```

Important limitation: USD flattening resolves USD layers, references, and
payloads into one layer. It does not necessarily embed external mesh or texture
files. The Pink IK URDF and `meshes/` directory are still needed unless the IK
path is also changed.

Suggested enablement steps:

1. Generate a flattened USD from `g1_29dof_dex1_1_v4_test_good.usd`.
2. Add it next to the current asset bundle as
   `g1_29dof_dex1_1_v4_test_good_flattened.usd`.
3. Run the G1 Dex1 task with:

   ```bash
   G1_DEX1_USD_PATH=/workspace/isaaclab/docs/g1_dex1_assets/g1_29dof_dex1_1_v4_test_good_flattened.usd \
   ./isaaclab.sh -p scripts/tools/probe_g1_dex1_stack_cube_reachability_env.py \
     --headless --device cuda:0 --enable_pinocchio --steps-per-command 20
   ```

4. Compare the articulation against the current asset:
   - Joint count and names.
   - Presence of `left_wrist_yaw_joint` and `right_wrist_yaw_joint`.
   - Presence of all four Dex1 prismatic joints.
   - Action terms and dimensions.
   - Gripper open/close tracking.
   - Pink IK wrist reachability.
5. Temporarily move the old USD dependency directories out of the asset bundle
   and rerun the same smoke test.
6. If the flattened USD passes without those directories, update
   `G1_DEX1_USD_PATH` default and prune the old USD layers.

The acceptance criterion is not just that the file opens. The environment must
reset and step in headless mode, and the current stack-cube reachability smoke
test must pass.
