# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the known-good G1 Dex1 contact-lift slice with warehouse and robot cameras."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0"
DEFAULT_OUT_MP4 = "/workspace/host/out/g1_dex1_contact_lift_warehouse_cameras_pipeline.mp4"
DEFAULT_OUT_SUMMARY = "/workspace/host/out/g1_dex1_contact_lift_warehouse_cameras_pipeline.json"
DEFAULT_OUT_DIR = "/workspace/host/out/g1_dex1_contact_lift_warehouse_cameras_pipeline_frames"


parser = argparse.ArgumentParser(description="Render the known-good scripted G1 Dex1 contact lift with warehouse and robot cameras.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-mp4", default=DEFAULT_OUT_MP4, help="Output MP4 path.")
parser.add_argument("--summary", default=DEFAULT_OUT_SUMMARY, help="Output JSON summary path.")
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for key-frame PNGs.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--frames", type=int, default=192)
parser.add_argument("--warmup-steps", type=int, default=32)
parser.add_argument("--global-width", type=int, default=960)
parser.add_argument("--global-height", type=int, default=540)
parser.add_argument("--close-width", type=int, default=480, help="Kept for CLI compatibility with the original script.")
parser.add_argument("--close-height", type=int, default=540, help="Kept for CLI compatibility with the original script.")
parser.add_argument("--with-warehouse", action=argparse.BooleanOptionalAction, default=True, help="Load the Isaac Sim Simple_Warehouse environment while preserving the known-good task layout.")
parser.add_argument(
    "--dump-wrist-frames",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Save every sampled left/right wrist camera frame from policy observations and raw sensor outputs.",
)
parser.add_argument("--dump-wrist-every", type=int, default=1, help="Save wrist debug frames every N rendered frames.")
parser.add_argument(
    "--right-wrist-camera-pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override right wrist camera fixed offset position relative to right_wrist_yaw_link.",
)
parser.add_argument(
    "--right-wrist-camera-rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Override right wrist camera fixed offset quaternion relative to right_wrist_yaw_link.",
)
parser.add_argument(
    "--right-wrist-camera-convention",
    choices=("ros", "world", "opengl"),
    default=None,
    help="Convention for the right wrist camera override.",
)
parser.add_argument(
    "--drive-right-wrist-camera-from-link",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Drive the right wrist camera from right_wrist_yaw_link with a fixed local transform each frame.",
)
parser.add_argument("--side", choices=("left", "right"), default="right", help="Which gripper performs the grasp.")
parser.add_argument("--cube", choices=("cube_1", "cube_2", "cube_3"), default="cube_1", help="Cube to grasp.")
parser.add_argument("--pregrasp-height", type=float, default=0.12, help="Vertical offset above grasp pose in meters.")
parser.add_argument("--lift-height", type=float, default=0.12, help="Vertical lift distance after closing in meters.")
parser.add_argument("--retreat-x", type=float, default=-0.04, help="Retreat offset along env x after lift in meters.")
parser.add_argument("--grasp-center-x-offset", type=float, default=0.0, help="Extra cube-relative gripper-center x offset.")
parser.add_argument("--grasp-center-y-offset", type=float, default=0.0, help="Extra cube-relative gripper-center y offset.")
parser.add_argument("--grasp-center-z-offset", type=float, default=0.0, help="Extra cube-relative gripper-center z offset.")
parser.add_argument("--lift-success-threshold", type=float, default=0.025, help="Cube lift threshold in meters.")
parser.add_argument("--center-success-threshold", type=float, default=0.045, help="Allowed gripper-center to cube distance.")
parser.add_argument("--close-error-threshold", type=float, default=0.006, help="Allowed gripper close joint error.")
parser.add_argument(
    "--assist-gripper-grasp",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Attach the cube only after the Dex1 gripper closes around it; intended for assisted grasp demos.",
)
parser.add_argument("--assist-trigger-threshold", type=float, default=0.5, help="Trigger threshold for assisted grasp.")
parser.add_argument("--assist-cube-size", type=float, default=0.0468, help="Approximate cube side length used for assisted grasp checks.")
parser.add_argument("--assist-gap-tolerance", type=float, default=0.008, help="Allowed finger-gap margin over cube size.")
parser.add_argument("--assist-center-threshold", type=float, default=0.07, help="Allowed gripper-center to cube distance before attach.")
parser.add_argument("--assist-close-error-threshold", type=float, default=0.006, help="Allowed gripper close joint error before attach.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Import Pinocchio before AppLauncher, matching Pink IK teleop/recording entrypoints.",
)
parser.add_argument(
    "--skip-kit-cleanup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit directly after writing video to avoid Kit shutdown hangs in headless CI.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

if not args_cli.enable_cameras:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.retargeters.humanoid.unitree.dex1.g1_dex1_upper_body_motion_ctrl_retargeter import (
    G1Dex1UpperBodyMotionControllerRetargeter,
)
from isaaclab.devices.openxr.retargeters.manipulator.gripper_trigger_or_pinch_retargeter import (
    GripperTriggerOrPinchRetargeter,
)
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    LEFT_DEX1_CLOSE,
    LEFT_DEX1_GRIPPER_JOINTS,
    LEFT_DEX1_OPEN,
    RIGHT_DEX1_CLOSE,
    RIGHT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_OPEN,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_stack_cube_visuomotor_env_cfg import (
    XformPrimCfg,
    _camera_spawn,
    spawn_xform_with_default_xform_command,
)
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
LEFT_DEX1_BODIES = ["left_dex1_base_link", "left_dex1_finger_link_1", "left_dex1_finger_link_2"]
RIGHT_DEX1_BODIES = ["right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2"]
# Centers of the Dex1 collision pads in each finger-link local frame, computed from dex1_col_*.stl bounds.
DEX1_FINGER_1_PAD_CENTER = (0.1082509, -0.0315503, 0.0)
DEX1_FINGER_2_PAD_CENTER = (0.1097630, 0.0313668, 0.0)


def add_warehouse(env_cfg) -> None:
    if not args_cli.with_warehouse:
        return
    env_cfg.scene.ground = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/Warehouse",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        debug_vis=False,
    )


def add_video_cameras(env_cfg) -> None:
    camera_spawn = sim_utils.PinholeCameraCfg(
        focal_length=20.0,
        focus_distance=400.0,
        horizontal_aperture=24.0,
        clipping_range=(0.01, 100.0),
    )
    env_cfg.scene.global_view_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoGlobalCamera",
        update_period=0,
        height=args_cli.global_height,
        width=args_cli.global_width,
        data_types=["rgb"],
        spawn=camera_spawn,
    )

    if not all(hasattr(env_cfg.scene, name) for name in ("ego_cam", "left_wrist_cam", "right_wrist_cam")):
        wrist_camera_rot = (0.5, -0.5, 0.5, -0.5)
        ego_camera_rot = (0.5, -0.5, 0.5, -0.5)
        env_cfg.scene.ego_cam_mount = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/ego_cam",
            spawn=XformPrimCfg(func=spawn_xform_with_default_xform_command),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0576, 0.0175, 0.4299)),
        )
        env_cfg.scene.ego_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/ego_cam/camera",
            update_period=0,
            height=args_cli.global_height,
            width=args_cli.global_width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_camera_spawn(),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=ego_camera_rot, convention="ros"),
        )
        env_cfg.scene.left_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/left_wrist_cam",
            update_period=0,
            height=args_cli.global_height,
            width=args_cli.global_width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_camera_spawn(),
            offset=CameraCfg.OffsetCfg(pos=(0.075, 0.035, 0.035), rot=wrist_camera_rot, convention="ros"),
        )
        env_cfg.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/right_wrist_cam",
            update_period=0,
            height=args_cli.global_height,
            width=args_cli.global_width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=_camera_spawn(),
            offset=CameraCfg.OffsetCfg(pos=(0.075, -0.035, 0.035), rot=wrist_camera_rot, convention="ros"),
        )
    apply_right_wrist_camera_override(env_cfg)
    if args_cli.drive_right_wrist_camera_from_link:
        if not hasattr(env_cfg.scene, "right_wrist_cam"):
            raise RuntimeError("Cannot drive right wrist camera because env_cfg.scene.right_wrist_cam is missing.")
        # Keep the camera's local transform fixed to the wrist, but place the Camera prim outside
        # the robot hierarchy. On the composed G1 USD, child Camera prims under wrist links can
        # drift in headless rendering even when the wrist body pose itself is valid.
        env_cfg.scene.right_wrist_cam.prim_path = "{ENV_REGEX_NS}/DrivenRightWristCamera"
        env_cfg.scene.right_wrist_cam.offset = CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        )
    for camera_name in ("global_view_cam", "ego_cam", "left_wrist_cam", "right_wrist_cam"):
        if hasattr(env_cfg.scene, camera_name):
            getattr(env_cfg.scene, camera_name).update_latest_camera_pose = True
    env_cfg.scene.lazy_sensor_update = False
    env_cfg.sim.render_interval = 1


def apply_right_wrist_camera_override(env_cfg) -> None:
    if not any(
        value is not None
        for value in (
            args_cli.right_wrist_camera_pos,
            args_cli.right_wrist_camera_rot,
            args_cli.right_wrist_camera_convention,
        )
    ):
        return
    if not hasattr(env_cfg.scene, "right_wrist_cam"):
        raise RuntimeError("Cannot override right wrist camera because env_cfg.scene.right_wrist_cam is missing.")
    camera_cfg = env_cfg.scene.right_wrist_cam
    offset = camera_cfg.offset
    if args_cli.right_wrist_camera_pos is not None and hasattr(env_cfg.scene, "right_wrist_cam_mount"):
        env_cfg.scene.right_wrist_cam_mount.init_state.pos = tuple(float(value) for value in args_cli.right_wrist_camera_pos)
        pos = (0.0, 0.0, 0.0)
    else:
        pos = tuple(float(value) for value in (args_cli.right_wrist_camera_pos or offset.pos))
    rot = tuple(float(value) for value in (args_cli.right_wrist_camera_rot or offset.rot))
    convention = args_cli.right_wrist_camera_convention or offset.convention
    camera_cfg.offset = CameraCfg.OffsetCfg(pos=pos, rot=rot, convention=convention)


def camera_offset_summary(camera: Camera) -> dict:
    offset = camera.cfg.offset
    return {
        "pos": [float(value) for value in offset.pos],
        "rot": [float(value) for value in offset.rot],
        "convention": offset.convention,
    }


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def interpolate_keyframes(frame_index: int, num_frames: int, keyframes):
    if num_frames <= 1:
        return keyframes[0][1], keyframes[0][2], keyframes[0][3], keyframes[0][0]
    t = frame_index / float(num_frames - 1)
    for next_index in range(1, len(keyframes)):
        t0, offset0, trigger0, phase0 = keyframes[next_index - 1]
        t1, offset1, trigger1, phase1 = keyframes[next_index]
        if t <= t1 or next_index == len(keyframes) - 1:
            alpha = _smoothstep((t - t0) / max(t1 - t0, 1.0e-6))
            offset = [float(v0 + (v1 - v0) * alpha) for v0, v1 in zip(offset0, offset1)]
            trigger = float(trigger0 + (trigger1 - trigger0) * alpha)
            phase = phase1 if alpha > 0.5 else phase0
            return offset, trigger, phase, t
    return keyframes[-1][1], keyframes[-1][2], keyframes[-1][3], t


def _controller_packet(position: list[float], quat: list[float], trigger: float) -> np.ndarray:
    pose = np.array([*position, *quat], dtype=np.float32)
    inputs = np.zeros(7, dtype=np.float32)
    inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value] = float(trigger)
    return np.stack([pose, inputs])


def make_controller_data(active_position, active_quat, active_trigger, inactive_position, inactive_quat) -> dict:
    if args_cli.side == "left":
        left = _controller_packet(active_position, active_quat, active_trigger)
        right = _controller_packet(inactive_position, inactive_quat, 0.0)
    else:
        left = _controller_packet(inactive_position, inactive_quat, 0.0)
        right = _controller_packet(active_position, active_quat, active_trigger)
    return {DeviceBase.TrackingTarget.CONTROLLER_LEFT: left, DeviceBase.TrackingTarget.CONTROLLER_RIGHT: right}


def body_pose_env_frame(env, robot, body_id: int) -> tuple[torch.Tensor, list[float]]:
    pos = robot.data.body_pos_w[0, body_id] - env.unwrapped.scene.env_origins[0]
    quat = robot.data.body_quat_w[0, body_id]
    return pos.detach().clone(), [float(v) for v in quat.detach().cpu().tolist()]


def make_retargeters_from_env_cfg(env_cfg):
    retargeter_cfgs = env_cfg.teleop_devices.devices["motion_controllers"].retargeters
    retargeters = [cfg.retargeter_type(cfg) for cfg in retargeter_cfgs]
    wrist_retargeter = next(
        retargeter for retargeter in retargeters if isinstance(retargeter, G1Dex1UpperBodyMotionControllerRetargeter)
    )
    gripper_retargeters = [
        retargeter for retargeter in retargeters if isinstance(retargeter, GripperTriggerOrPinchRetargeter)
    ]
    left_gripper = next(
        (
            retargeter
            for retargeter in gripper_retargeters
            if getattr(retargeter, "_bound_controller", None) == DeviceBase.TrackingTarget.CONTROLLER_LEFT
        ),
        None,
    )
    right_gripper = next(
        (
            retargeter
            for retargeter in gripper_retargeters
            if getattr(retargeter, "_bound_controller", None) == DeviceBase.TrackingTarget.CONTROLLER_RIGHT
        ),
        None,
    )
    if left_gripper is None or right_gripper is None:
        raise RuntimeError("Expected both left and right GripperTriggerOrPinchRetargeter instances.")
    return wrist_retargeter, left_gripper, right_gripper, [type(retargeter).__name__ for retargeter in retargeters]


def retarget_action(wrist_retargeter, left_gripper, right_gripper, raw_data: dict) -> torch.Tensor:
    return torch.cat(
        [wrist_retargeter.retarget(raw_data), left_gripper.retarget(raw_data), right_gripper.retarget(raw_data)],
        dim=-1,
    ).unsqueeze(0)


def cube_pos_env(env, name: str) -> torch.Tensor:
    cube = env.unwrapped.scene[name]
    return cube.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]


def maybe_assist_gripper_grasp(
    env,
    robot,
    body_ids: list[int],
    joint_ids: list[int],
    close_target: float,
    cube_name: str,
    trigger: float,
    attached_delta_w,
):
    info = {
        "assist_trigger_ready": False,
        "assist_close_ready": False,
        "assist_gap_ready": False,
        "assist_center_ready": False,
        "assist_close_error_m": None,
    }
    if not args_cli.assist_gripper_grasp:
        return attached_delta_w, False, info

    cube = env.unwrapped.scene[cube_name]
    center_w = gripper_center_world(robot, body_ids)
    finger_gap = float(gripper_gap_world(robot, body_ids).item())
    center_to_cube = float(torch.linalg.norm(center_w - cube.data.root_pos_w[0]).item())
    close_error = max_abs_error(to_list(robot.data.joint_pos[0, joint_ids]), close_target)

    gap_error = abs(finger_gap - args_cli.assist_cube_size)
    gap_ready = finger_gap <= args_cli.assist_cube_size + args_cli.assist_gap_tolerance
    info.update(
        {
            "assist_trigger_ready": bool(trigger >= args_cli.assist_trigger_threshold),
            "assist_close_ready": bool(gap_ready),
            "assist_gap_ready": bool(gap_ready),
            "assist_center_ready": bool(center_to_cube <= args_cli.assist_center_threshold),
            "assist_close_error_m": float(close_error),
            "assist_gap_error_m": float(gap_error),
        }
    )

    if trigger < args_cli.assist_trigger_threshold:
        return None, False, info

    if attached_delta_w is None:
        ready = info["assist_trigger_ready"] and info["assist_gap_ready"] and info["assist_center_ready"]
        if not ready:
            return None, False, info
        # Assisted demos lock the cube to the midpoint between the Dex1 front collision pads,
        # so the visual grasp is driven by the gripper claws, not by the wrist target.
        attached_delta_w = torch.zeros_like(center_w)

    root_pose = torch.cat([center_w + attached_delta_w, cube.data.root_quat_w[0]], dim=0).unsqueeze(0)
    cube.write_root_pose_to_sim(root_pose)
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device))
    return attached_delta_w, True, info


def quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_xyz = quat[1:]
    q_w = quat[0]
    return vec + 2.0 * torch.cross(q_xyz, torch.cross(q_xyz, vec, dim=0) + q_w * vec, dim=0)


def quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def drive_camera_from_body_fixed_transform(
    camera: Camera, robot, body_id: int, local_pos: torch.Tensor, local_quat_world: torch.Tensor
) -> None:
    body_pos_w = robot.data.body_pos_w[0, body_id]
    body_quat_w = robot.data.body_quat_w[0, body_id]
    camera_pos_w = body_pos_w + quat_apply_wxyz(body_quat_w, local_pos)
    camera_quat_w = quat_multiply_wxyz(body_quat_w, local_quat_world)
    camera_quat_w = camera_quat_w / torch.linalg.norm(camera_quat_w)
    camera.set_world_poses(camera_pos_w.unsqueeze(0), camera_quat_w.unsqueeze(0), convention="world")


def _local_vec(robot, values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=robot.device)


def finger_contact_points_world(robot, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    finger_1_pos = robot.data.body_pos_w[0, body_ids[1]]
    finger_2_pos = robot.data.body_pos_w[0, body_ids[2]]
    finger_1_quat = robot.data.body_quat_w[0, body_ids[1]]
    finger_2_quat = robot.data.body_quat_w[0, body_ids[2]]
    finger_1_contact = finger_1_pos + quat_apply_wxyz(finger_1_quat, _local_vec(robot, DEX1_FINGER_1_PAD_CENTER))
    finger_2_contact = finger_2_pos + quat_apply_wxyz(finger_2_quat, _local_vec(robot, DEX1_FINGER_2_PAD_CENTER))
    return finger_1_contact, finger_2_contact


def gripper_center_world(robot, body_ids: list[int]) -> torch.Tensor:
    finger_1_contact, finger_2_contact = finger_contact_points_world(robot, body_ids)
    return 0.5 * (finger_1_contact + finger_2_contact)


def gripper_gap_world(robot, body_ids: list[int]) -> torch.Tensor:
    finger_1_contact, finger_2_contact = finger_contact_points_world(robot, body_ids)
    return torch.linalg.norm(finger_1_contact - finger_2_contact)


def gripper_center_env(env, robot, body_ids: list[int]) -> torch.Tensor:
    return gripper_center_world(robot, body_ids) - env.unwrapped.scene.env_origins[0]


def to_list(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.detach().cpu().tolist()]


def max_abs_error(values: list[float], target: float) -> float:
    return max(abs(value - target) for value in values)


def compute_scene_bounds(env, robot) -> tuple[torch.Tensor, float]:
    positions = [robot.data.body_pos_w[0]]
    for rigid_object in env.unwrapped.scene.rigid_objects.values():
        positions.append(rigid_object.data.root_pos_w)
    body_pos = torch.cat(positions, dim=0)
    minimum = body_pos.min(dim=0).values
    maximum = body_pos.max(dim=0).values
    center = (minimum + maximum) * 0.5
    center[2] = torch.clamp(center[2], min=1.0)
    radius = max(torch.linalg.norm(maximum - minimum).item() * 0.5, 1.0)
    return center, radius


def set_global_camera_pose(env, robot, camera: Camera) -> None:
    center, radius = compute_scene_bounds(env, robot)
    view_dir = torch.tensor([1.6, -2.2, 0.75], dtype=torch.float32, device=robot.device)
    view_dir = view_dir / torch.linalg.norm(view_dir)
    eye = center + view_dir * max(radius * 3.2, 3.0)
    camera.set_world_poses_from_view(eye.unsqueeze(0), center.unsqueeze(0))


def set_gripper_camera_pose(env, robot, body_ids: list[int], camera: Camera) -> None:
    target = gripper_center_world(robot, body_ids)
    side_y = 0.58 if args_cli.side == "left" else -0.58
    view_dir = torch.tensor([0.78, side_y, 0.30], dtype=torch.float32, device=robot.device)
    view_dir = view_dir / torch.linalg.norm(view_dir)
    eye = target + view_dir * 0.55
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))


def set_cube_camera_pose(env, robot, body_ids: list[int], cube_name: str, camera: Camera) -> None:
    cube = env.unwrapped.scene[cube_name]
    target = 0.55 * cube.data.root_pos_w[0] + 0.45 * gripper_center_world(robot, body_ids)
    side_y = -0.70 if args_cli.side == "right" else 0.70
    view_dir = torch.tensor([0.82, side_y, 0.42], dtype=torch.float32, device=robot.device)
    view_dir = view_dir / torch.linalg.norm(view_dir)
    eye = target + view_dir * 0.62
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))


def rgb_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()[..., :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def image_array_stats(image: np.ndarray) -> dict:
    return {
        "shape": list(image.shape),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "range": int(image.max()) - int(image.min()),
    }


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image.copy()
    y = 30
    for line in lines:
        cv2.putText(output, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
        y += 28
    return output


def side_config(robot):
    if args_cli.side == "left":
        wrist_ids, wrist_names = robot.find_bodies([LEFT_WRIST_BODY], preserve_order=True)
        body_ids, body_names = robot.find_bodies(LEFT_DEX1_BODIES, preserve_order=True)
        joint_ids, joint_names = robot.find_joints(LEFT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        close_target = LEFT_DEX1_CLOSE
        open_target = LEFT_DEX1_OPEN
    else:
        wrist_ids, wrist_names = robot.find_bodies([RIGHT_WRIST_BODY], preserve_order=True)
        body_ids, body_names = robot.find_bodies(RIGHT_DEX1_BODIES, preserve_order=True)
        joint_ids, joint_names = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        close_target = RIGHT_DEX1_CLOSE
        open_target = RIGHT_DEX1_OPEN
    if len(wrist_ids) != 1 or len(body_ids) != 3 or len(joint_ids) != 2:
        raise RuntimeError(f"Could not resolve {args_cli.side} gripper entities.")
    return wrist_ids[0], wrist_names, body_ids, body_names, joint_ids, joint_names, close_target, open_target


def inactive_pose(env, robot) -> tuple[list[float], list[float]]:
    wrist_name = RIGHT_WRIST_BODY if args_cli.side == "left" else LEFT_WRIST_BODY
    gripper_bodies = RIGHT_DEX1_BODIES if args_cli.side == "left" else LEFT_DEX1_BODIES
    wrist_ids, _ = robot.find_bodies([wrist_name], preserve_order=True)
    body_ids, _ = robot.find_bodies(gripper_bodies, preserve_order=True)
    _, quat = body_pose_env_frame(env, robot, wrist_ids[0])
    return to_list(gripper_center_env(env, robot, body_ids)), quat


def measure(env, robot, wrist_id: int, joint_ids: list[int], body_ids: list[int], cube_name: str) -> dict:
    cube_env = cube_pos_env(env, cube_name)
    wrist_env = robot.data.body_pos_w[0, wrist_id] - env.unwrapped.scene.env_origins[0]
    wrist_quat_w = robot.data.body_quat_w[0, wrist_id]
    center_env = gripper_center_env(env, robot, body_ids)
    joint_pos = robot.data.joint_pos[0, joint_ids]
    finger_gap = gripper_gap_world(robot, body_ids)
    finger_1_contact, finger_2_contact = finger_contact_points_world(robot, body_ids)
    return {
        "wrist_pos_env": to_list(wrist_env),
        "wrist_quat_w": to_list(wrist_quat_w),
        "gripper_center_env": to_list(center_env),
        "cube_pos_env": to_list(cube_env),
        "cube_height_env_m": float(cube_env[2].item()),
        "gripper_joint_pos": to_list(joint_pos),
        "finger_body_separation_m": float(finger_gap.item()),
        "finger_1_contact_env": to_list(finger_1_contact - env.unwrapped.scene.env_origins[0]),
        "finger_2_contact_env": to_list(finger_2_contact - env.unwrapped.scene.env_origins[0]),
        "center_to_cube_m": float(torch.linalg.norm(center_env - cube_env).item()),
        "center_cube_xy_m": float(torch.linalg.norm((center_env - cube_env)[:2]).item()),
        "center_minus_cube_z_m": float((center_env - cube_env)[2].item()),
    }


def make_trajectory(default_wrist: torch.Tensor, grasp_wrist: torch.Tensor) -> list[tuple[float, list[float], float, str]]:
    pregrasp_wrist = grasp_wrist + torch.tensor([0.0, 0.0, args_cli.pregrasp_height], device=grasp_wrist.device)
    lift_wrist = grasp_wrist + torch.tensor([0.0, 0.0, args_cli.lift_height], device=grasp_wrist.device)
    retreat_wrist = lift_wrist + torch.tensor([args_cli.retreat_x, 0.0, 0.0], device=grasp_wrist.device)

    def target(wrist_target: torch.Tensor) -> list[float]:
        return to_list(wrist_target)

    return [
        (0.00, target(default_wrist), 0.0, "home_open"),
        (0.20, target(pregrasp_wrist), 0.0, "pregrasp_above_open"),
        (0.45, target(grasp_wrist), 0.0, "descend_to_cube_open"),
        (0.58, target(grasp_wrist), 1.0, "close_on_cube"),
        (0.78, target(lift_wrist), 1.0, "lift_closed"),
        (0.88, target(retreat_wrist), 1.0, "retreat_closed"),
        (1.00, target(retreat_wrist), 0.0, "reopen_after_lift"),
    ]


def resize_panel(image: np.ndarray) -> np.ndarray:
    if image.shape[1] == args_cli.global_width and image.shape[0] == args_cli.global_height:
        return image
    return cv2.resize(image, (args_cli.global_width, args_cli.global_height), interpolation=cv2.INTER_LINEAR)


def camera_rgb(camera: Camera, title: str, lines: list[str], policy_obs: dict | None = None, obs_name: str | None = None) -> np.ndarray:
    if policy_obs is not None and obs_name is not None and obs_name in policy_obs:
        image = rgb_tensor_to_uint8(policy_obs[obs_name][0])
    else:
        image = rgb_tensor_to_uint8(camera.data.output["rgb"][0])
    return annotate(resize_panel(image), [title, *lines])


def camera_rgb_stats(camera: Camera) -> dict:
    rgb = rgb_tensor_to_uint8(camera.data.output["rgb"][0])
    distance = camera.data.output.get("distance_to_image_plane")
    finite_fraction = None
    if distance is not None:
        finite_fraction = float(torch.isfinite(distance).float().mean().item())
    return {
        "shape": list(rgb.shape),
        "min": int(rgb.min()),
        "max": int(rgb.max()),
        "mean": float(rgb.mean()),
        "std": float(rgb.std()),
        "range": int(rgb.max()) - int(rgb.min()),
        "distance_finite_fraction": finite_fraction,
    }


def camera_content_ok(stats: dict[str, dict]) -> dict[str, bool]:
    result = {}
    for name, item in stats.items():
        has_rgb_variation = item["std"] > 1.0 or item["range"] > 8.0
        finite_depth = item.get("distance_finite_fraction")
        has_depth = finite_depth is None or finite_depth > 0.01
        result[name] = bool(has_rgb_variation and has_depth)
    return result


def camera_pose_debug(camera: Camera, target_w: torch.Tensor) -> dict:
    pos_w = camera.data.pos_w[0]
    quat_w_world = camera.data.quat_w_world[0]
    forward_w = quat_apply_wxyz(quat_w_world, torch.tensor([1.0, 0.0, 0.0], device=pos_w.device))
    to_target_w = target_w - pos_w
    to_target_norm = torch.linalg.norm(to_target_w)
    if float(to_target_norm.item()) > 1.0e-6:
        to_target_unit = to_target_w / to_target_norm
        forward_alignment = torch.dot(forward_w / torch.linalg.norm(forward_w), to_target_unit)
    else:
        forward_alignment = torch.tensor(float("nan"), device=pos_w.device)
    return {
        "pos_w": to_list(pos_w),
        "quat_w_world": to_list(quat_w_world),
        "forward_w": to_list(forward_w),
        "target_w": to_list(target_w),
        "target_distance_m": float(to_target_norm.item()),
        "forward_target_dot": float(forward_alignment.item()),
    }


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def dump_wrist_debug_frames(
    frame: int,
    phase: str,
    policy_obs: dict,
    left_wrist_camera: Camera,
    right_wrist_camera: Camera,
    debug_dir: Path,
) -> dict:
    result = {"frame": int(frame), "phase": phase, "images": {}}
    for side_name, camera in (("left", left_wrist_camera), ("right", right_wrist_camera)):
        obs_name = f"{side_name}_wrist_cam"
        if obs_name in policy_obs:
            policy_image = rgb_tensor_to_uint8(policy_obs[obs_name][0])
            policy_path = debug_dir / f"frame_{frame:03d}_{obs_name}_policy.png"
            save_rgb(policy_path, policy_image)
            result["images"][f"{obs_name}_policy"] = {
                "path": str(policy_path),
                **image_array_stats(policy_image),
            }
        sensor_image = rgb_tensor_to_uint8(camera.data.output["rgb"][0])
        sensor_path = debug_dir / f"frame_{frame:03d}_{obs_name}_sensor.png"
        save_rgb(sensor_path, sensor_image)
        result["images"][f"{obs_name}_sensor"] = {
            "path": str(sensor_path),
            **image_array_stats(sensor_image),
        }
    return result


def make_composite(global_camera, ego_camera, left_wrist_camera, right_wrist_camera, policy_obs, frame, progress, phase, trigger, action, measured, cube_lift):
    common = [
        "Known-good contact-lift pipeline" + (" (assisted Dex1 grasp)" if args_cli.assist_gripper_grasp else ""),
        f"{args_cli.side} gripper on {args_cli.cube}",
        f"frame {frame:03d} progress {progress:.2f}",
        f"phase: {phase}",
    ]
    global_rgb = camera_rgb(global_camera, "Third-person / warehouse", common)
    ego_rgb = camera_rgb(
        ego_camera,
        "Ego camera: torso_link/ego_cam",
        [f"trigger: {trigger:.2f}", f"cube lift: {cube_lift:+.4f} m", f"center-cube: {measured['center_to_cube_m']:.4f} m"],
        policy_obs,
        "ego_cam",
    )
    left_wrist_rgb = camera_rgb(
        left_wrist_camera,
        "Left wrist camera",
        [f"action shape: {tuple(action.shape)}", f"phase: {phase}", "policy_obs left_wrist_cam"],
        policy_obs,
        "left_wrist_cam",
    )
    right_wrist_rgb = camera_rgb(
        right_wrist_camera,
        "Right wrist camera",
        ["Dex1 front-claw contact lift", "joints: {:.4f}, {:.4f} m".format(*measured["gripper_joint_pos"]), f"finger sep: {measured['finger_body_separation_m']:.4f} m"],
        policy_obs,
        "right_wrist_cam",
    )
    top = np.concatenate([global_rgb, ego_rgb], axis=1)
    bottom = np.concatenate([left_wrist_rgb, right_wrist_rgb], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("Video recording currently expects --num-envs 1.")
    out_mp4 = Path(args_cli.out_mp4)
    out_summary = Path(args_cli.summary)
    out_dir = Path(args_cli.out_dir)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrist_debug_dir = out_dir / "wrist_debug_frames"
    if args_cli.dump_wrist_frames:
        wrist_debug_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    add_warehouse(env_cfg)
    add_video_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    try:
        robot = env.unwrapped.scene["robot"]
        global_camera = env.unwrapped.scene.sensors["global_view_cam"]
        ego_camera = env.unwrapped.scene.sensors["ego_cam"]
        left_wrist_camera = env.unwrapped.scene.sensors["left_wrist_cam"]
        right_wrist_camera = env.unwrapped.scene.sensors["right_wrist_cam"]
        cameras = {
            "global_view_cam": global_camera,
            "ego_cam": ego_camera,
            "left_wrist_cam": left_wrist_camera,
            "right_wrist_cam": right_wrist_camera,
        }
        wrist_id, wrist_names, body_ids, body_names, joint_ids, joint_names, close_target, open_target = side_config(robot)
        driven_right_wrist_camera_local_pos = None
        driven_right_wrist_camera_local_rot = None
        if args_cli.drive_right_wrist_camera_from_link:
            if args_cli.right_wrist_camera_pos is None or args_cli.right_wrist_camera_rot is None:
                raise ValueError(
                    "--drive-right-wrist-camera-from-link requires --right-wrist-camera-pos and "
                    "--right-wrist-camera-rot."
                )
            if args_cli.right_wrist_camera_convention != "world":
                raise ValueError("--drive-right-wrist-camera-from-link currently expects convention='world'.")
            driven_right_wrist_camera_local_pos = torch.tensor(
                args_cli.right_wrist_camera_pos, dtype=torch.float32, device=robot.device
            )
            driven_right_wrist_camera_local_rot = torch.tensor(
                args_cli.right_wrist_camera_rot, dtype=torch.float32, device=robot.device
            )
            driven_right_wrist_camera_local_rot = driven_right_wrist_camera_local_rot / torch.linalg.norm(
                driven_right_wrist_camera_local_rot
            )
        default_wrist, active_quat = body_pose_env_frame(env, robot, wrist_id)
        default_center = gripper_center_env(env, robot, body_ids)
        inactive_default_position, inactive_controller_quat = inactive_pose(env, robot)
        wrist_retargeter, left_gripper, right_gripper, configured_retargeters = make_retargeters_from_env_cfg(env_cfg)

        set_global_camera_pose(env, robot, global_camera)
        for _ in range(args_cli.warmup_steps):
            raw = make_controller_data(to_list(default_center), active_quat, 0.0, inactive_default_position, inactive_controller_quat)
            env.step(retarget_action(wrist_retargeter, left_gripper, right_gripper, raw))

        default_wrist, active_quat = body_pose_env_frame(env, robot, wrist_id)
        default_center = gripper_center_env(env, robot, body_ids)
        cube_initial = cube_pos_env(env, args_cli.cube)
        center_to_wrist = default_center - default_wrist
        requested_center = cube_initial + torch.tensor(
            [args_cli.grasp_center_x_offset, args_cli.grasp_center_y_offset, args_cli.grasp_center_z_offset],
            dtype=torch.float32,
            device=cube_initial.device,
        )
        grasp_wrist = requested_center - center_to_wrist
        keyframes = make_trajectory(default_wrist, grasp_wrist)
        initial = measure(env, robot, wrist_id, joint_ids, body_ids, args_cli.cube)

        video_size = (args_cli.global_width * 2, args_cli.global_height * 2)
        video = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.fps, video_size)
        if not video.isOpened():
            raise RuntimeError(f"Could not open video writer: {out_mp4}")

        key_frame_indices = {0, args_cli.frames // 4, args_cli.frames // 2, (args_cli.frames * 3) // 4, args_cli.frames - 1}
        attached_cube_delta_w = None
        assist_info = {}
        frame_records = []
        wrist_debug_records = []
        for frame in range(args_cli.frames):
            active_position, active_trigger, phase, progress = interpolate_keyframes(frame, args_cli.frames, keyframes)
            active_controller_position = [
                float(active_position[i] + center_to_wrist[i].item()) for i in range(3)
            ]
            raw = make_controller_data(
                active_controller_position, active_quat, active_trigger, inactive_default_position, inactive_controller_quat
            )
            action = retarget_action(wrist_retargeter, left_gripper, right_gripper, raw)
            set_global_camera_pose(env, robot, global_camera)
            env.step(action)
            attached_cube_delta_w, cube_assist_attached, assist_info = maybe_assist_gripper_grasp(
                env, robot, body_ids, joint_ids, close_target, args_cli.cube, active_trigger, attached_cube_delta_w
            )
            if args_cli.drive_right_wrist_camera_from_link:
                drive_camera_from_body_fixed_transform(
                    right_wrist_camera,
                    robot,
                    wrist_id,
                    driven_right_wrist_camera_local_pos,
                    driven_right_wrist_camera_local_rot,
                )
                env.unwrapped.sim.render()
                right_wrist_camera.update(env.unwrapped.sim.get_physics_dt(), force_recompute=True)
            measured = measure(env, robot, wrist_id, joint_ids, body_ids, args_cli.cube)
            cube_lift = measured["cube_height_env_m"] - float(cube_initial[2].item())
            policy_obs = dict(env.unwrapped.obs_buf.get("policy", {}))
            if args_cli.drive_right_wrist_camera_from_link:
                policy_obs.pop("right_wrist_cam", None)
            if args_cli.dump_wrist_frames and frame % max(args_cli.dump_wrist_every, 1) == 0:
                wrist_debug_records.append(
                    dump_wrist_debug_frames(
                        frame,
                        phase,
                        policy_obs,
                        left_wrist_camera,
                        right_wrist_camera,
                        wrist_debug_dir,
                    )
                )
            composite = make_composite(
                global_camera,
                ego_camera,
                left_wrist_camera,
                right_wrist_camera,
                policy_obs,
                frame,
                progress,
                phase,
                active_trigger,
                action,
                measured,
                cube_lift,
            )
            video.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            if frame in key_frame_indices:
                cv2.imwrite(str(out_dir / f"frame_{frame:03d}.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            frame_records.append(
                {
                    "frame": frame,
                    "progress": float(progress),
                    "phase": phase,
                    "mock_active_wrist_target_position": [float(v) for v in active_position],
                    "mock_active_controller_position": active_controller_position,
                    "mock_active_controller_offset": [
                        float(active_controller_position[i] - default_center[i].item()) for i in range(3)
                    ],
                    "mock_active_trigger": float(active_trigger),
                    "assist_cube_attached": bool(cube_assist_attached),
                    **assist_info,
                    "cube_lift_m": float(cube_lift),
                    "right_wrist_camera_debug": camera_pose_debug(
                        right_wrist_camera, env.unwrapped.scene[args_cli.cube].data.root_pos_w[0]
                    ),
                    "retargeted_action_shape": list(action.shape),
                    **measured,
                }
            )

        video.release()
        if args_cli.dump_wrist_frames:
            (wrist_debug_dir / "wrist_debug_stats.json").write_text(
                json.dumps(wrist_debug_records, indent=2), encoding="utf-8"
            )
        if not out_mp4.exists() or out_mp4.stat().st_size == 0:
            raise RuntimeError(f"Video file was not written: {out_mp4}")

        max_cube_lift = max(record["cube_lift_m"] for record in frame_records)
        final_cube_lift = frame_records[-1]["cube_lift_m"]
        min_center_to_cube = min(record["center_to_cube_m"] for record in frame_records)
        close_records = [record for record in frame_records if record["phase"] in ("close_on_cube", "lift_closed", "retreat_closed")]
        close_error = min(max_abs_error(record["gripper_joint_pos"], close_target) for record in close_records)
        open_error = max_abs_error(frame_records[-1]["gripper_joint_pos"], open_target)
        action_shape_ok = tuple(frame_records[0]["retargeted_action_shape"]) == tuple(env.action_space.shape)
        attached_frames = sum(1 for record in frame_records if record.get("assist_cube_attached", False))
        contact_lift_mode = not args_cli.assist_gripper_grasp
        contact_lift_passed = (
            contact_lift_mode
            and attached_frames == 0
            and action_shape_ok
            and min_center_to_cube <= args_cli.center_success_threshold
            and max_cube_lift >= args_cli.lift_success_threshold
        )
        assisted_lift_passed = (
            bool(args_cli.assist_gripper_grasp)
            and action_shape_ok
            and min_center_to_cube <= args_cli.center_success_threshold
            and max_cube_lift >= args_cli.lift_success_threshold
        )
        passed = contact_lift_passed or assisted_lift_passed
        camera_stats = {name: camera_rgb_stats(camera) for name, camera in cameras.items()}
        camera_ok = camera_content_ok(camera_stats)
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "passed": bool(passed),
            "task": args_cli.task,
            "side": args_cli.side,
            "cube": args_cli.cube,
            "out_mp4": str(out_mp4),
            "out_dir": str(out_dir),
            "fps": args_cli.fps,
            "frames": args_cli.frames,
            "video_size": list(video_size),
            "warehouse_enabled": bool(args_cli.with_warehouse),
            "warehouse_usd": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd" if args_cli.with_warehouse else None,
            "camera_prim_paths": {name: camera.cfg.prim_path for name, camera in cameras.items()},
            "camera_offsets": {name: camera_offset_summary(camera) for name, camera in cameras.items() if hasattr(camera.cfg, "offset")},
            "robot_camera_image_source": "policy_obs when available, sensor.data.output fallback",
            "drive_right_wrist_camera_from_link": bool(args_cli.drive_right_wrist_camera_from_link),
            "driven_right_wrist_camera_local_pos": (
                [float(value) for value in args_cli.right_wrist_camera_pos]
                if args_cli.drive_right_wrist_camera_from_link and args_cli.right_wrist_camera_pos is not None
                else None
            ),
            "driven_right_wrist_camera_local_rot_world": (
                [float(value) for value in args_cli.right_wrist_camera_rot]
                if args_cli.drive_right_wrist_camera_from_link and args_cli.right_wrist_camera_rot is not None
                else None
            ),
            "wrist_debug_enabled": bool(args_cli.dump_wrist_frames),
            "wrist_debug_every": int(args_cli.dump_wrist_every),
            "wrist_debug_dir": str(wrist_debug_dir) if args_cli.dump_wrist_frames else None,
            "wrist_debug_stats_json": str(wrist_debug_dir / "wrist_debug_stats.json") if args_cli.dump_wrist_frames else None,
            "wrist_debug_record_count": len(wrist_debug_records),
            "camera_stats": camera_stats,
            "camera_content_ok": camera_ok,
            "configured_motion_controller_retargeters": configured_retargeters,
            "assist_gripper_grasp": bool(args_cli.assist_gripper_grasp),
            "assist_trigger_threshold": float(args_cli.assist_trigger_threshold),
            "assist_cube_size": float(args_cli.assist_cube_size),
            "assist_gap_tolerance": float(args_cli.assist_gap_tolerance),
            "assist_center_threshold": float(args_cli.assist_center_threshold),
            "assist_close_error_threshold": float(args_cli.assist_close_error_threshold),
            "assist_attach_target": "dex1_front_pad_midpoint",
            "attached_frames": int(attached_frames),
            "contact_lift_mode": bool(contact_lift_mode),
            "contact_lift_passed": bool(contact_lift_passed),
            "assisted_lift_passed": bool(assisted_lift_passed),
            "action_space": str(env.action_space),
            "action_shape_ok": bool(action_shape_ok),
            "action_terms": env.unwrapped.action_manager.active_terms,
            "action_term_dims": env.unwrapped.action_manager.action_term_dim,
            "wrist_names": wrist_names,
            "gripper_body_names": body_names,
            "gripper_joint_names": joint_names,
            "default_wrist_env": to_list(default_wrist),
            "default_gripper_center_env": to_list(default_center),
            "center_to_wrist_env": to_list(center_to_wrist),
            "cube_initial_env": to_list(cube_initial),
            "requested_gripper_center_env": to_list(requested_center),
            "computed_grasp_wrist_env": to_list(grasp_wrist),
            "keyframes": [
                {"t": t, "offset": offset, "trigger": trigger, "phase": phase}
                for t, offset, trigger, phase in keyframes
            ],
            "initial": initial,
            "midpoint": frame_records[len(frame_records) // 2],
            "final": frame_records[-1],
            "max_cube_lift_m": float(max_cube_lift),
            "final_cube_lift_m": float(final_cube_lift),
            "min_gripper_center_to_cube_m": float(min_center_to_cube),
            "best_close_max_abs_error_m": float(close_error),
            "final_open_max_abs_error_m": float(open_error),
            "lift_success_threshold_m": args_cli.lift_success_threshold,
            "center_success_threshold_m": args_cli.center_success_threshold,
            "close_error_threshold_m": args_cli.close_error_threshold,
            "frame_records": frame_records,
            "bytes": out_mp4.stat().st_size,
        }
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote G1 Dex1 stack-cube grasp/lift video: {out_mp4}")
        print(f"[INFO] Wrote summary: {out_summary}")
        print(f"[INFO] Overall result: {'PASS' if passed else 'FAIL'}")
        print(
            "[INFO] metrics: "
            f"max_cube_lift={max_cube_lift:.6f} m, "
            f"min_center_to_cube={min_center_to_cube:.6f} m, "
            f"best_close_error={close_error:.6f} m"
        )
    finally:
        if not args_cli.skip_kit_cleanup:
            env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if args_cli.skip_kit_cleanup:
            os._exit(exit_code)
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(exit_code)
