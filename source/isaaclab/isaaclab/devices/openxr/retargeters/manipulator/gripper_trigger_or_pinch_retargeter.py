# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper: controller trigger priority, else hand pinch (isaacteleop-compatible).

Matches ``IsaacTeleop/src/retargeters/gripper_retargeter.py`` branching; pairs with motion-controller EE like
Isaac Lab 3 Franka ``stack_ik_abs_env_cfg`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class GripperTriggerOrPinchRetargeter(RetargeterBase):
    """Trigger controls grip when controller inputs exist; else thumb-index pinch with hysteresis."""

    GRIPPER_CLOSE_METERS: Final[float] = 0.03
    GRIPPER_OPEN_METERS: Final[float] = 0.05

    def __init__(self, cfg: GripperTriggerOrPinchRetargeterCfg):
        super().__init__(cfg)
        if cfg.bound_hand not in (DeviceBase.TrackingTarget.HAND_LEFT, DeviceBase.TrackingTarget.HAND_RIGHT):
            raise ValueError("bound_hand must be HAND_LEFT or HAND_RIGHT")
        if cfg.bound_controller not in (
            DeviceBase.TrackingTarget.CONTROLLER_LEFT,
            DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
        ):
            raise ValueError("bound_controller must be CONTROLLER_LEFT or CONTROLLER_RIGHT")

        self._bound_hand = cfg.bound_hand
        self._bound_controller = cfg.bound_controller
        self._controller_threshold = cfg.controller_threshold
        self._close_m = cfg.gripper_close_meters
        self._open_m = cfg.gripper_open_meters
        self._previous_gripper_command = False

    def retarget(self, data: dict) -> torch.Tensor:
        controller_data = data.get(self._bound_controller)

        if self._controller_inputs_available(controller_data):
            inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
            trigger = float(inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value])
            self._previous_gripper_command = trigger > self._controller_threshold
        else:
            hand_data = data[self._bound_hand]
            thumb_tip = hand_data["thumb_tip"]
            index_tip = hand_data["index_tip"]
            self._calculate_gripper_command(thumb_tip[:3], index_tip[:3])

        gripper_value = -1.0 if self._previous_gripper_command else 1.0
        return torch.tensor([gripper_value], dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [
            RetargeterBase.Requirement.HAND_TRACKING,
            RetargeterBase.Requirement.MOTION_CONTROLLER,
        ]

    def _calculate_gripper_command(self, thumb_pos: np.ndarray, index_pos: np.ndarray) -> None:
        distance = np.linalg.norm(thumb_pos - index_pos)
        if distance > self._open_m:
            self._previous_gripper_command = False
        elif distance < self._close_m:
            self._previous_gripper_command = True

    @staticmethod
    def _controller_inputs_available(controller_data: np.ndarray | None) -> bool:
        if controller_data is None or getattr(controller_data, "size", 0) == 0:
            return False
        ri = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
        if len(controller_data) <= ri:
            return False
        inputs = controller_data[ri]
        return len(inputs) > DeviceBase.MotionControllerInputIndex.TRIGGER.value


@dataclass
class GripperTriggerOrPinchRetargeterCfg(RetargeterCfg):
    """Configuration for trigger-or-pinch gripper retargeting."""

    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT
    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    controller_threshold: float = 0.5
    gripper_close_meters: float = GripperTriggerOrPinchRetargeter.GRIPPER_CLOSE_METERS
    gripper_open_meters: float = GripperTriggerOrPinchRetargeter.GRIPPER_OPEN_METERS
    retargeter_type: type[RetargeterBase] = GripperTriggerOrPinchRetargeter
