# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal G1 Dex1 gripper-only task.

This environment is intentionally small: it keeps the manager-based Isaac Lab
pipeline (scene, action manager, observations, teleop device config) but removes
the manipulation object, IK controller, and task success logic. It is used as a
smoke-test bridge between standalone USD probing and the full G1 teleop task.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_trigger_or_pinch_retargeter import (
    GripperTriggerOrPinchRetargeterCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass


G1_DEX1_USD_PATH = os.environ.get(
    "G1_DEX1_USD_PATH",
    "/workspace/host/RobotLearningLab_Dataset/usecase/humanoid_teleop/g1_29dof_dex1_1_v4_test_good.usd",
)
RIGHT_DEX1_GRIPPER_JOINTS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]
RIGHT_DEX1_OPEN = 0.02449999935925007
RIGHT_DEX1_CLOSE = -0.019999999552965164


G1_DEX1_GRIPPER_ONLY_CFG = ArticulationCfg(
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
        rot=(1.0, 0.0, 0.0, 0.0),
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


@configclass
class G1Dex1GripperOnlySceneCfg(InteractiveSceneCfg):
    """Scene with only the custom G1 Dex1 robot, ground, and light."""

    robot: ArticulationCfg = G1_DEX1_GRIPPER_ONLY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(size=(8.0, 8.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the gripper-only MDP."""

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
    """Minimal observations for smoke testing."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True
                )
            },
        )
        gripper_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Only time-out termination is needed for this smoke task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class G1Dex1GripperOnlyEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based environment that exposes only the right Dex1 gripper action."""

    scene: G1Dex1GripperOnlySceneCfg = G1Dex1GripperOnlySceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    commands = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 4
        self.viewer.eye = (3.0, -3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True

        self.teleop_devices = DevicesCfg(
            devices={
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=[
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
