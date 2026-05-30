# G1 Dex1 Assets

This directory is the self-contained asset bundle used by `Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0`.

For the full USD/URDF relationship and the future flattening plan, see
`../G1_DEX1_USD_URDF_ASSET_RELATIONSHIP.md`.

Runtime entry points:

- Simulation USD: `g1_29dof_dex1_1_v4_test_good.usd`
- Pink IK URDF: `g1_29dof_mode_15_with_dex1_1.urdf`
- Pink IK meshes: `meshes/`

Important dependency notes:

- The Python task config only points at the v4 USD and the URDF, but the v4 USD has indirect USD composition dependencies.
- Keep `g1_29dof_dex1_1_v1.usd`, `g1_29dof_dex1_1_v3.usd`, `dex1_1_gripper.usd`, and `g1_29dof_mode_15_with_dex1_1/`. Removing these can silently drop wrist-yaw or gripper joints from the articulation. `g1_29dof_dex1_1_v2.usd` and the original `dex1_1/` importer directory were tested separately and are not required by the current pipeline.
- Keep `g1_29dof_rev_1_0_with_inspire_hand_FTP/g1_29dof_no_hand.usd` and its `configuration/` directory. The v4 USD payload still references those configuration layers; the 38M base layer was tested and removing it breaks articulation creation.
- `meshes/` has been pruned to the 40 STL files referenced by `g1_29dof_mode_15_with_dex1_1.urdf`; old inspire-hand meshes are intentionally not included.

Validation command used after pruning:

```bash
docker exec -w /workspace/isaaclab isaac-lab-232 ./isaaclab.sh \
  -p scripts/tools/probe_g1_dex1_stack_cube_reachability_env.py \
  --headless --device cuda:0 --enable_pinocchio --steps-per-command 20
```
