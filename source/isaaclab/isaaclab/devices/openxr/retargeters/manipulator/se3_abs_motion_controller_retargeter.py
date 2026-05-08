# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Absolute SE(3) retargeting from OpenXR motion controller poses.

Backport target (semantic parity): Isaac Lab 3
``isaaclab_tasks/.../franka/stack_ik_abs_env_cfg.py::_build_franka_stack_pipeline`` uses isaacteleop
``Se3AbsRetargeter`` with ``ControllersSource.RIGHT``; see also ``IsaacTeleop``
``src/retargeters/se3_retargeter.py`` controller branch (Euler XYZ offsets on grip pose).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


class Se3AbsMotionControllerRetargeter(RetargeterBase):
    """Maps motion-controller grip pose to absolute EE pose.

    Matches isaacteleop ``Se3AbsRetargeter`` when ``input_device`` is a controller.
    OpenXR pose row is ``[x, y, z, qw, qx, qy, qz]`` per ``OpenXRDevice._query_controller``.
    Output quaternion is ``w, x, y, z`` like :class:`Se3AbsRetargeter` for IL2 differential IK.
    """

    def __init__(self, cfg: Se3AbsMotionControllerRetargeterCfg):
        super().__init__(cfg)
        if cfg.bound_controller not in (
            DeviceBase.TrackingTarget.CONTROLLER_LEFT,
            DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
        ):
            raise ValueError("bound_controller must be CONTROLLER_LEFT or CONTROLLER_RIGHT")

        self._cfg = cfg
        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation

        self._target_offset_rot = Rotation.from_euler(
            "XYZ",
            [cfg.target_offset_roll_deg, cfg.target_offset_pitch_deg, cfg.target_offset_yaw_deg],
            degrees=True,
        )
        self._target_offset_pos = np.array([cfg.target_offset_x, cfg.target_offset_y, cfg.target_offset_z], dtype=np.float64)

        self._last_output = np.concatenate(
            [np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
        )

        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal_mc"))
            self._goal_marker.set_visibility(True)

    def retarget(self, data: dict) -> torch.Tensor:
        controller_data = data.get(self._cfg.bound_controller)

        if not self._controller_pose_available(controller_data):
            return torch.tensor(self._last_output, dtype=torch.float32, device=self._sim_device)

        pose_row = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        position = pose_row[:3].astype(np.float64)
        base_rot = Rotation.from_quat([pose_row[4], pose_row[5], pose_row[6], pose_row[3]])

        final_rot = base_rot * self._target_offset_rot
        position = position + base_rot.apply(self._target_offset_pos)

        if self._zero_out_xy_rotation:
            z, _, _ = final_rot.as_euler("ZYX")
            final_rot = Rotation.from_euler("ZYX", np.array([z, 0.0, 0.0])) * Rotation.from_euler(
                "X", np.pi, degrees=False
            )

        quat_xyzw = final_rot.as_quat()
        rotation_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        ee_command_np = np.concatenate([position.astype(np.float32), rotation_wxyz])
        self._last_output = ee_command_np

        if self._enable_visualization:
            trans = np.array([ee_command_np[:3]])
            rot = np.array([rotation_wxyz])
            self._goal_marker.visualize(translations=trans, orientations=rot)

        return torch.tensor(ee_command_np, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    @staticmethod
    def _controller_pose_available(controller_data: np.ndarray | None) -> bool:
        if controller_data is None or getattr(controller_data, "size", 0) == 0:
            return False
        row = DeviceBase.MotionControllerDataRowIndex.POSE.value
        return len(controller_data) > row and len(controller_data[row]) >= 7


@dataclass
class Se3AbsMotionControllerRetargeterCfg(RetargeterCfg):
    """Configuration for absolute motion-controller SE(3) retargeting."""

    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    zero_out_xy_rotation: bool = False
    target_offset_roll_deg: float = 90.0
    target_offset_pitch_deg: float = 0.0
    target_offset_yaw_deg: float = 0.0
    target_offset_x: float = 0.0
    target_offset_y: float = 0.0
    target_offset_z: float = 0.0
    enable_visualization: bool = False
    retargeter_type: type[RetargeterBase] = Se3AbsMotionControllerRetargeter
