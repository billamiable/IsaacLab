# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 Dex1 scene slice for teleop pipeline validation.

This task keeps the lower body/root fixed and only exposes the right Dex1 gripper
action.  It adds a simple table and cube so that the visual scene is closer to
the real teleop environments, while intentionally avoiding locomotion and upper
body IK until the gripper/action path is stable in a task scene.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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

from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    G1_DEX1_GRIPPER_ONLY_CFG,
    RIGHT_DEX1_CLOSE,
    RIGHT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_OPEN,
)


@configclass
class G1Dex1FixedBaseSceneCfg(InteractiveSceneCfg):
    """Scene with fixed-base custom G1 Dex1, table, cube, ground, and light."""

    robot = G1_DEX1_GRIPPER_ONLY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.10, 0.80, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.47, 0.45), roughness=0.75),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.62, 0.67), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.16, 0.16),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.12, 0.05), roughness=0.45),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.30, 0.78), rot=(1.0, 0.0, 0.0, 0.0)),
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
        """Keep the robot base fixed for this scene slice."""
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ActionsCfg:
    """Action specifications for the fixed-base scene slice."""

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
    """Minimal observations for validating the task scene pipeline."""

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
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Basic task scene terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube")}
    )


@configclass
class G1Dex1FixedBaseSceneEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-base task scene that exposes only the right Dex1 gripper action."""

    scene: G1Dex1FixedBaseSceneCfg = G1Dex1FixedBaseSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    commands = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.45),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 4
        self.viewer.eye = (3.0, -3.0, 2.0)
        self.viewer.lookat = (0.0, 0.45, 1.0)

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
