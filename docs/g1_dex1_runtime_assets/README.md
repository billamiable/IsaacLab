# G1 Dex1 Runtime Assets

This directory is the compact runtime asset bundle for the G1 + Dex1 Isaac Lab teleop tasks. The older `docs/g1_dex1_assets/` directory is left in place as source/reference material.

## Entry Points

```text
g1_dex1_runtime_assets/
├── usd/
│   ├── g1_dex1_sim.usd          # flattened simulation USD
│   └── g1_dex1_visuomotor.usda  # camera overlay referencing g1_dex1_sim.usd
├── kinematics/
│   ├── g1_dex1_kinematics.urdf  # Pink IK / Pinocchio model
│   └── meshes/                  # STL meshes referenced by the URDF
├── manifest.json
└── README.md
```

## Default Consumers

The G1 Dex1 task configs now default to this bundle:

- `G1_DEX1_USD_PATH`: `usd/g1_dex1_sim.usd`
- `G1_DEX1_VISUOMOTOR_USD_PATH`: `usd/g1_dex1_visuomotor.usda`
- `G1_DEX1_KINEMATICS_URDF_PATH`: `kinematics/g1_dex1_kinematics.urdf`
- `G1_DEX1_KINEMATICS_MESH_PATH`: `kinematics/`

Environment variables with the same names can still override these paths.

## Validation Gates

Before treating this bundle as working, run:

1. Right Dex1 gripper open/close smoke.
2. G1 Dex1 stack-cube reachability smoke.
3. Headless mock Pico visuomotor HDF5 recording.
4. HDF5 camera-to-MP4 export.

The old asset directory should not be removed until the compact bundle passes those gates and downstream users have switched paths.
