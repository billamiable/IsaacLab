# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 Dex1 stack-cube reachability slice.

This task is intentionally not a full stacking task yet. It keeps the bilateral
Pico/Pink IK/Dex1 gripper action pipeline and replaces the simple single-cube
scene with a three-cube tabletop layout inspired by the Franka stack-cube task.
The first acceptance criterion is reachability: each wrist should be able to
move near a tabletop cube with the fixed-base humanoid height and table height.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_fixed_base_ik_scene_env_cfg import (
    ActionsCfg,
    G1Dex1FixedBaseIKSceneEnvCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    DEX1_GRIPPER_JOINTS,
    G1_DEX1_GRIPPER_ONLY_CFG,
    RIGHT_DEX1_OPEN,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp


BLOCK_HEIGHT_DIFF = 0.0468
BLOCK_CENTER_Z_OFFSET = 0.0203


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# The custom G1 Dex1 USD faces +X, so the table must start in front of the feet.
TABLE_CENTER = (
    _env_float("G1_DEX1_TABLE_CENTER_X", 0.58),
    _env_float("G1_DEX1_TABLE_CENTER_Y", 0.0),
    _env_float("G1_DEX1_TABLE_CENTER_Z", 0.90),
)
TABLE_SIZE = (
    _env_float("G1_DEX1_TABLE_SIZE_X", 0.60),
    _env_float("G1_DEX1_TABLE_SIZE_Y", 0.72),
    _env_float("G1_DEX1_TABLE_SIZE_Z", 0.06),
)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
BLOCK_CENTER_Z = TABLE_TOP_Z + BLOCK_CENTER_Z_OFFSET
CUBE_1_POS = (
    _env_float("G1_DEX1_CUBE_1_X", 0.50),
    _env_float("G1_DEX1_CUBE_1_Y", -0.16),
    _env_float("G1_DEX1_CUBE_1_Z", BLOCK_CENTER_Z),
)
CUBE_2_POS = (
    _env_float("G1_DEX1_CUBE_2_X", 0.50),
    _env_float("G1_DEX1_CUBE_2_Y", 0.16),
    _env_float("G1_DEX1_CUBE_2_Z", BLOCK_CENTER_Z),
)
CUBE_3_POS = (
    _env_float("G1_DEX1_CUBE_3_X", 0.64),
    _env_float("G1_DEX1_CUBE_3_Y", 0.0),
    _env_float("G1_DEX1_CUBE_3_Z", BLOCK_CENTER_Z),
)
RIGHT_REACH_CUBE = "cube_1"
LEFT_REACH_CUBE = "cube_2"


def _block_spawn(block_file: str, semantic_class: str) -> UsdFileCfg:
    return UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/{block_file}",
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        semantic_tags=[("class", semantic_class)],
    )


def g1_dex1_cubes_stacked(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.06,
    height_threshold: float = 0.015,
    height_diff: float = BLOCK_HEIGHT_DIFF,
):
    """Check blue-bottom, red-middle, green-top cube stacking success for teleop recording."""

    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    cube_3 = env.scene[cube_3_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    stacked = torch.logical_and(torch.abs(h_dist_c12 - height_diff) < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)
    stacked = torch.logical_and(torch.abs(h_dist_c23 - height_diff) < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c23[:, 2] < 0.0, stacked)
    return stacked


@configclass
class G1Dex1StackCubeReachabilitySceneCfg(InteractiveSceneCfg):
    """Scene with fixed-base G1 Dex1, table, three cubes, ground, and light."""

    robot = G1_DEX1_GRIPPER_ONLY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.47, 0.45), roughness=0.75),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_CENTER, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        spawn=_block_spawn("blue_block.usd", "cube_1"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_1_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        spawn=_block_spawn("red_block.usd", "cube_2"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_2_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        spawn=_block_spawn("green_block.usd", "cube_3"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_3_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(size=(8.0, 8.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        """Keep the robot base fixed for this reachability slice."""
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ObservationsCfg:
    """Observations used for reachability testing and video summaries."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DEX1_GRIPPER_JOINTS, preserve_order=True)},
        )
        cube_1_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_1")})
        cube_2_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_2")})
        cube_3_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_3")})
        cube_1_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_1")})
        cube_2_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_2")})
        cube_3_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_3")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Basic terminations for the stack-cube reachability scene."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)
    success = DoneTerm(func=g1_dex1_cubes_stacked)
    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_1")}
    )
    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_2")}
    )
    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_3")}
    )


@configclass
class G1Dex1FixedBaseStackCubeReachabilityEnvCfg(G1Dex1FixedBaseIKSceneEnvCfg):
    """Fixed-base G1 Dex1 scene for stack-cube reachability calibration."""

    scene: G1Dex1StackCubeReachabilitySceneCfg = G1Dex1StackCubeReachabilitySceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.gripper_joint_names = DEX1_GRIPPER_JOINTS
        self.gripper_open_val = RIGHT_DEX1_OPEN
        self.gripper_threshold = 0.005
        self.viewer.eye = (2.8, -2.4, 1.8)
        self.viewer.lookat = (0.34, 0.0, 0.92)
