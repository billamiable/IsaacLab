# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base custom G1 Dex1 scene with upper-body Pink IK plus right gripper."""

from __future__ import annotations

import os

import isaaclab.envs.mdp as mdp
from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.dex1.g1_dex1_upper_body_motion_ctrl_retargeter import (
    G1Dex1UpperBodyMotionControllerRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.manipulator.gripper_trigger_or_pinch_retargeter import (
    GripperTriggerOrPinchRetargeterCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_fixed_base_scene_env_cfg import (
    G1Dex1FixedBaseSceneCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    RIGHT_DEX1_CLOSE,
    RIGHT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_OPEN,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp


G1_DEX1_KINEMATICS_URDF_PATH = os.environ.get(
    "G1_DEX1_KINEMATICS_URDF_PATH",
    "/workspace/host/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf",
)
G1_DEX1_KINEMATICS_MESH_PATH = os.environ.get(
    "G1_DEX1_KINEMATICS_MESH_PATH",
    "/workspace/host/unitree_ros/robots/g1_description",
)


G1_DEX1_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=0,
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "left_wrist_yaw_link",
            base_link_frame_name="pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "right_wrist_yaw_link",
            base_link_frame_name="pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
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
    hand_joint_names=[],
    target_eef_link_names={"left_wrist": "left_wrist_yaw_link", "right_wrist": "right_wrist_yaw_link"},
    asset_name="robot",
    controller=G1_DEX1_UPPER_BODY_IK_CONTROLLER_CFG,
)


@configclass
class ActionsCfg:
    """Upper-body IK plus a separate one-dimensional right gripper action."""

    upper_body_ik = G1_DEX1_UPPER_BODY_IK_ACTION_CFG

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_DEX1_GRIPPER_JOINTS,
        open_command_expr={
            "right_dex1_finger_joint_1": RIGHT_DEX1_OPEN,
            "right_dex1_finger_joint_2": RIGHT_DEX1_OPEN,
        },
        close_command_expr={
            "right_dex1_finger_joint_1": RIGHT_DEX1_CLOSE,
            "right_dex1_finger_joint_2": RIGHT_DEX1_CLOSE,
        },
    )


@configclass
class ObservationsCfg:
    """Observations used for smoke testing and video summaries."""

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
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True
                )
            },
        )
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Basic terminations for the fixed-base IK scene."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)
    cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube")}
    )


@configclass
class G1Dex1FixedBaseIKSceneEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-base task scene with upper-body IK and right Dex1 gripper."""

    scene: G1Dex1FixedBaseSceneCfg = G1Dex1FixedBaseSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    commands = None
    curriculum = None

    xr: XrCfg = XrCfg(anchor_pos=(0.0, 0.0, -0.45), anchor_rot=(1.0, 0.0, 0.0, 0.0))

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 4
        self.viewer.eye = (3.0, -3.0, 2.0)
        self.viewer.lookat = (0.0, 0.45, 1.0)

        self.actions.upper_body_ik.controller.urdf_path = G1_DEX1_KINEMATICS_URDF_PATH
        self.actions.upper_body_ik.controller.mesh_path = G1_DEX1_KINEMATICS_MESH_PATH

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True

        self.teleop_devices = DevicesCfg(
            devices={
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=[
                        G1Dex1UpperBodyMotionControllerRetargeterCfg(
                            bound_right_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
                            use_left_controller=False,
                            use_controller_orientation=False,
                            left_wrist_default_pose=(
                                0.20477421581745148,
                                0.1486508846282959,
                                1.0952297449111938,
                                0.9999998807907104,
                                3.006622864631936e-05,
                                2.755914829322137e-05,
                                -9.576302545610815e-05,
                            ),
                            right_wrist_default_pose=(
                                0.2047741860151291,
                                -0.14864102005958557,
                                1.0952297449111938,
                                0.9999998807907104,
                                -3.006685437867418e-05,
                                2.7558131478144787e-05,
                                9.56020230660215e-05,
                            ),
                            sim_device=self.sim.device,
                        ),
                        GripperTriggerOrPinchRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
                            controller_threshold=0.5,
                            sim_device=self.sim.device,
                        )
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
