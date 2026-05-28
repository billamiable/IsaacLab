# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class G1Dex1UpperBodyMotionControllerRetargeter(RetargeterBase):
    """Map Pico/OpenXR motion-controller poses to G1 Dex1 upper-body Pink IK targets.

    The output is only the Pink IK part of the environment action:

    ``[left_wrist_pose(7), right_wrist_pose(7)]``.

    Dex1 gripper trigger is intentionally handled by
    :class:`GripperTriggerOrPinchRetargeter`, so the env receives:

    ``[left wrist 7, right wrist 7] + [right gripper 1] = 15``.
    """

    def __init__(self, cfg: G1Dex1UpperBodyMotionControllerRetargeterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._left_default_pose = np.asarray(cfg.left_wrist_default_pose, dtype=np.float32)
        self._right_default_pose = np.asarray(cfg.right_wrist_default_pose, dtype=np.float32)
        self._left_controller_origin_pose = np.asarray(cfg.left_controller_origin_pose, dtype=np.float32)
        self._right_controller_origin_pose = np.asarray(cfg.right_controller_origin_pose, dtype=np.float32)

        for name, pose in (
            ("left_wrist_default_pose", self._left_default_pose),
            ("right_wrist_default_pose", self._right_default_pose),
            ("left_controller_origin_pose", self._left_controller_origin_pose),
            ("right_controller_origin_pose", self._right_controller_origin_pose),
        ):
            if pose.shape != (7,):
                raise ValueError(f"{name} must be a 7D pose [x, y, z, qw, qx, qy, qz], got {pose.shape}")

    def retarget(self, data: dict) -> torch.Tensor:
        """Return 14D Pink IK wrist target action from controller raw data."""
        left_pose = self._retarget_controller(
            data.get(self._cfg.bound_left_controller),
            self._left_default_pose,
            self._left_controller_origin_pose,
            enabled=self._cfg.use_left_controller,
        )
        right_pose = self._retarget_controller(
            data.get(self._cfg.bound_right_controller),
            self._right_default_pose,
            self._right_controller_origin_pose,
            enabled=True,
        )
        command = np.concatenate([left_pose, right_pose]).astype(np.float32)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _retarget_controller(
        self,
        controller_data: np.ndarray | None,
        default_pose: np.ndarray,
        origin_pose: np.ndarray,
        *,
        enabled: bool,
    ) -> np.ndarray:
        """Map one controller pose to one wrist pose.

        The first version uses relative controller translation around a configured
        origin. This keeps the robot wrist near its configured default pose and
        avoids requiring exact OpenXR world coordinates during headless tests.
        """
        if not enabled or not self._controller_pose_available(controller_data):
            return default_pose.copy()

        pose = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value].astype(np.float32)
        output = default_pose.copy()
        output[:3] = default_pose[:3] + (pose[:3] - origin_pose[:3]) * float(self._cfg.position_scale)
        if self._cfg.use_controller_orientation:
            output[3:] = pose[3:]
        return output

    @staticmethod
    def _controller_pose_available(controller_data: np.ndarray | None) -> bool:
        if controller_data is None or getattr(controller_data, "size", 0) == 0:
            return False
        row = DeviceBase.MotionControllerDataRowIndex.POSE.value
        return len(controller_data) > row and len(controller_data[row]) >= 7


@dataclass
class G1Dex1UpperBodyMotionControllerRetargeterCfg(RetargeterCfg):
    """Configuration for G1 Dex1 motion-controller wrist retargeting."""

    bound_left_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_LEFT
    bound_right_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    use_left_controller: bool = False
    use_controller_orientation: bool = False
    position_scale: float = 1.0
    left_wrist_default_pose: tuple[float, float, float, float, float, float, float] = (
        0.205,
        0.149,
        1.095,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    right_wrist_default_pose: tuple[float, float, float, float, float, float, float] = (
        0.205,
        -0.149,
        1.095,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    left_controller_origin_pose: tuple[float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    right_controller_origin_pose: tuple[float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    retargeter_type: type[RetargeterBase] = G1Dex1UpperBodyMotionControllerRetargeter
