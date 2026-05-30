# G1 Dex1 Assets

This directory is the self-contained asset bundle used by `Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0`.

## Runtime Entry Points

The current pipeline uses two robot descriptions:

```text
Isaac Sim / PhysX articulation
  -> g1_29dof_dex1_1_v4_test_good.usd

Pink IK / Pinocchio kinematics
  -> g1_29dof_mode_15_with_dex1_1.urdf
  -> meshes/
```

The USD creates the simulated articulation. The URDF is not used to spawn the robot in Isaac Sim; it is used by Pink IK to build the kinematic tree, joint model, and frame model.

## Current Bundle

```text
docs/g1_dex1_assets/
├── g1_29dof_dex1_1_v4_test_good.usd
├── g1_29dof_dex1_1_v3.usd
├── g1_29dof_dex1_1_v1.usd
├── dex1_1_gripper.usd
├── g1_29dof_mode_15_with_dex1_1.urdf
├── meshes/*.STL
├── g1_29dof_mode_15_with_dex1_1/
│   ├── g1_29dof_mode_15_with_dex1_1.usd
│   └── configuration/*.usd
└── g1_29dof_rev_1_0_with_inspire_hand_FTP/
    ├── g1_29dof_no_hand.usd
    └── configuration/*.usd
```

The checked-in bundle is about 90M. `meshes/` has been pruned to the 40 STL files referenced by `g1_29dof_mode_15_with_dex1_1.urdf`; old inspire-hand meshes are intentionally not included.

## USD Composition Roles

`g1_29dof_dex1_1_v4_test_good.usd` is the simulation entry point, but it is not a fully self-contained single-file asset. It relies on USD composition layers.

Conceptually, the assembled robot is:

```text
G1 no-hand body
  + Dex1 gripper subtree
  + wrist/gripper joint and physics relationships
  + final overrides saved in v1/v3/v4 layers
```

### `g1_29dof_rev_1_0_with_inspire_hand_FTP/`

This is the G1 no-hand body asset source. Its key entry is:

```text
g1_29dof_rev_1_0_with_inspire_hand_FTP/g1_29dof_no_hand.usd
```

Its `configuration/` layers provide the body hierarchy, physics schemas, and articulation data needed by the no-hand G1 base robot. The 38M base layer in this directory was tested explicitly: removing it breaks articulation creation with a missing/invalid `root_joint`.

### `g1_29dof_mode_15_with_dex1_1/`

This is the USD output from importing the full `g1_29dof_mode_15_with_dex1_1.urdf`. It contains a complete imported robot USD, but in the assembled asset it should not be treated as the intended G1 body source.

Its role is an intermediate donor/reference layer. The final `v4` asset depends on this imported layer mainly for the Dex1/wrist connection area:

- Dex1 gripper prims and related physics schema.
- Wrist/gripper joint relationships.
- Importer-generated USD configuration layers required for the composed stage.

This is why the directory name looks like "full G1 + Dex1", even though the final assembly conceptually uses the no-hand G1 body plus the Dex1 gripper. USD composition operates at layer/prim level; until the stage is flattened or rebuilt, deleting only the "unused body-looking" parts of this imported USD is not safe.

## Removed Files

These were tested and removed because the current pipeline does not need them:

```text
g1_29dof_dex1_1_v2.usd
dex1_1/
old inspire-hand meshes and old root USD/URDF files
```

## Future Flattened USD Plan

The future cleanup target is to replace the multi-layer simulation USD bundle with one simulation USD file:

```text
g1_29dof_dex1_1_v4_test_good_flattened.usd
```

Expected structure after validation:

```text
docs/g1_dex1_assets/
├── g1_29dof_dex1_1_v4_test_good_flattened.usd
├── g1_29dof_mode_15_with_dex1_1.urdf
├── meshes/*.STL
└── README.md
```

Important limitation: USD flattening resolves USD layers, references, and payloads into one layer. It does not necessarily embed external mesh or texture files. The Pink IK URDF and `meshes/` directory are still needed unless the IK path is changed.

Suggested enablement steps:

1. Generate a flattened USD from `g1_29dof_dex1_1_v4_test_good.usd`.
2. Add it next to the current asset bundle as `g1_29dof_dex1_1_v4_test_good_flattened.usd`.
3. Temporarily point `G1_DEX1_USD_PATH` to the flattened USD.
4. Compare the articulation against the current asset: joint count and names, wrist yaw joints, all four Dex1 prismatic joints, gripper open/close tracking, and Pink IK wrist reachability.
5. Temporarily move the old USD dependency directories out of the asset bundle and rerun validation.
6. If the flattened USD passes without those directories, update the default `G1_DEX1_USD_PATH` and prune the old USD layers.

The acceptance criterion is not just that the file opens. The environment must reset and step in headless mode, and the G1 Dex1 action pipeline must still expose `upper_body_ik`, `left_gripper_action`, and `right_gripper_action`.
