# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 + Dex1 one-DoF gripper stack-cube slice for Isaac Lab 3.

This is the Lab3 port of the previously validated Isaac Lab 2.3.2 G1 Dex1
pipeline.  It intentionally follows the Lab3 G1 Pink IK action layout:

    [left_wrist_pose(7), right_wrist_pose(7), dex1_gripper_joints(4)]

The Dex1 gripper joints live in the Pink action "hand" tail.  That keeps this
task on the upstream Lab3 Pink action code path and avoids patching the core
action implementation for a zero-dimensional hand.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import LocalFrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp


G1_DEX1_ASSET_DIR = os.environ.get("G1_DEX1_ASSET_DIR", "/workspace/isaaclab/docs/g1_dex1_runtime_assets")
G1_DEX1_USD_PATH = os.environ.get("G1_DEX1_USD_PATH", os.path.join(G1_DEX1_ASSET_DIR, "usd", "g1_dex1_sim.usd"))
G1_DEX1_KINEMATICS_URDF_PATH = os.environ.get(
    "G1_DEX1_KINEMATICS_URDF_PATH",
    os.path.join(G1_DEX1_ASSET_DIR, "kinematics", "g1_dex1_kinematics.urdf"),
)
G1_DEX1_KINEMATICS_MESH_PATH = os.environ.get(
    "G1_DEX1_KINEMATICS_MESH_PATH", os.path.join(G1_DEX1_ASSET_DIR, "kinematics")
)

LEFT_DEX1_GRIPPER_JOINTS = ["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"]
RIGHT_DEX1_GRIPPER_JOINTS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]
DEX1_GRIPPER_JOINTS = LEFT_DEX1_GRIPPER_JOINTS + RIGHT_DEX1_GRIPPER_JOINTS
DEX1_OPEN = 0.02449999935925007
DEX1_CLOSE = -0.019999999552965164
IDENTITY_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

TABLE_CENTER = (0.58, 0.0, 0.90)
TABLE_SIZE = (0.60, 0.72, 0.06)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
BLOCK_HEIGHT_DIFF = 0.0468
BLOCK_CENTER_Z_OFFSET = 0.0203
BLOCK_CENTER_Z = TABLE_TOP_Z + BLOCK_CENTER_Z_OFFSET

CUBE_1_POS = (0.56, 0.0, BLOCK_CENTER_Z)  # blue, stack base
CUBE_2_POS = (0.52, -0.18, BLOCK_CENTER_Z)  # red, robot right
CUBE_3_POS = (0.52, 0.18, BLOCK_CENTER_Z)  # green, robot left


G1_DEX1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_DEX1_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=IDENTITY_QUAT_XYZW,
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=300.0,
            velocity_limit_sim=100.0,
            stiffness=80.0,
            damping=8.0,
            armature=0.001,
        ),
    },
)


G1_DEX1_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=len(DEX1_GRIPPER_JOINTS),
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTaskCfg(
            frame="left_wrist_yaw_link",
            base_link_frame_name="pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTaskCfg(
            frame="right_wrist_yaw_link",
            base_link_frame_name="pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTaskCfg(
            cost=0.5,
            lm_damping=1,
            controlled_frames=["left_wrist_yaw_link", "right_wrist_yaw_link"],
            controlled_joints=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            gain=0.3,
        ),
    ],
    fixed_input_tasks=[],
)


G1_DEX1_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_yaw_joint",
        "waist_.*_joint",
    ],
    hand_joint_names=DEX1_GRIPPER_JOINTS,
    target_eef_link_names={"left_wrist": "left_wrist_yaw_link", "right_wrist": "right_wrist_yaw_link"},
    asset_name="robot",
    controller=G1_DEX1_UPPER_BODY_IK_CONTROLLER_CFG,
)


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
    gripper_open_threshold: float = 0.01,
):
    """Check blue/red/green stacking success after both Dex1 grippers are open."""

    robot = env.scene[robot_cfg.name]
    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    cube_3 = env.scene[cube_3_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w.torch - cube_2.data.root_pos_w.torch
    pos_diff_c23 = cube_2.data.root_pos_w.torch - cube_3.data.root_pos_w.torch

    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    stacked = torch.logical_and(torch.abs(h_dist_c12 - height_diff) < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)
    stacked = torch.logical_and(torch.abs(h_dist_c23 - height_diff) < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c23[:, 2] < 0.0, stacked)

    gripper_joint_ids, _ = robot.find_joints(DEX1_GRIPPER_JOINTS, preserve_order=True)
    open_target = torch.full(
        (len(gripper_joint_ids),), DEX1_OPEN, dtype=torch.float32, device=env.device
    )
    gripper_joint_pos = robot.data.joint_pos.torch[:, gripper_joint_ids]
    grippers_open = torch.all(torch.abs(gripper_joint_pos - open_target) < gripper_open_threshold, dim=1)
    return torch.logical_and(stacked, grippers_open)


@configclass
class G1Dex1StackCubeSceneCfg(InteractiveSceneCfg):
    """Scene with fixed-base G1 Dex1, table, three cubes, ground, and light."""

    robot: ArticulationCfg = G1_DEX1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.47, 0.45), roughness=0.75),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_CENTER, rot=IDENTITY_QUAT_XYZW),
    )

    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        spawn=_block_spawn("blue_block.usd", "cube_1"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_1_POS, rot=IDENTITY_QUAT_XYZW),
    )

    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        spawn=_block_spawn("red_block.usd", "cube_2"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_2_POS, rot=IDENTITY_QUAT_XYZW),
    )

    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        spawn=_block_spawn("green_block.usd", "cube_3"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_3_POS, rot=IDENTITY_QUAT_XYZW),
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
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ActionsCfg:
    """Action tensor: left wrist pose, right wrist pose, Dex1 gripper joints."""

    upper_body_ik = G1_DEX1_UPPER_BODY_IK_ACTION_CFG


@configclass
class ObservationsCfg:
    """State observations for smoke tests and video summaries."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=base_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
        gripper_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DEX1_GRIPPER_JOINTS, preserve_order=True)},
        )
        cube_1_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_1")})
        cube_2_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_2")})
        cube_3_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_3")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Basic stack-cube terminations."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)
    success = DoneTerm(func=g1_dex1_cubes_stacked)
    cube_1_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_1")},
    )
    cube_2_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_2")},
    )
    cube_3_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_3")},
    )


@configclass
class G1Dex1FixedBaseStackCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-base G1 Dex1 stack-cube scene driven by Lab3 Pink IK."""

    scene: G1Dex1StackCubeSceneCfg = G1Dex1StackCubeSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 4
        self.viewer.eye = (2.8, -2.4, 1.8)
        self.viewer.lookat = (0.34, 0.0, 0.92)

        self.actions.upper_body_ik.controller.urdf_path = G1_DEX1_KINEMATICS_URDF_PATH
        self.actions.upper_body_ik.controller.mesh_path = G1_DEX1_KINEMATICS_MESH_PATH
