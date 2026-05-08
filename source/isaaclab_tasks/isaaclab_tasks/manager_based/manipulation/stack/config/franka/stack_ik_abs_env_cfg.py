# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_trigger_or_pinch_retargeter import (
    GripperTriggerOrPinchRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_motion_controller_retargeter import (
    Se3AbsMotionControllerRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaCubeStackEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3AbsRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                # Controller EE + trigger-or-pinch gripper: semantic backport of Isaac Lab 3
                # ``stack_ik_abs_env_cfg._build_franka_stack_pipeline`` + ``IsaacTeleopCfg`` (no teleop_devices key there;
                # IL3 ``record_demos`` ignores --teleop_device when isaac_teleop is set). Device key matches IL2
                # ``locomanipulation_g1_env_cfg.motion_controllers``.
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=[
                        Se3AbsMotionControllerRetargeterCfg(
                            bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
                            zero_out_xy_rotation=False,
                            target_offset_roll_deg=90.0,
                            target_offset_pitch_deg=0.0,
                            target_offset_yaw_deg=0.0,
                            sim_device=self.sim.device,
                        ),
                        GripperTriggerOrPinchRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
                            controller_threshold=0.5,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
