# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render bilateral mock Pico motion-controller retargeting for fixed-base G1 Dex1."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-IK-Scene-v0"
DEFAULT_OUT_MP4 = "/workspace/host/out/g1_dex1_bimanual_motion_controller_retargeter_pipeline.mp4"
DEFAULT_OUT_SUMMARY = "/workspace/host/out/g1_dex1_bimanual_motion_controller_retargeter_pipeline.json"
DEFAULT_OUT_DIR = "/workspace/host/out/g1_dex1_bimanual_motion_controller_retargeter_pipeline_frames"


parser = argparse.ArgumentParser(description="Render bilateral G1 Dex1 mock Pico motion-controller retargeter pipeline.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-mp4", default=DEFAULT_OUT_MP4, help="Output MP4 path.")
parser.add_argument("--summary", default=DEFAULT_OUT_SUMMARY, help="Output JSON summary path.")
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for key-frame PNGs.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--frames", type=int, default=168)
parser.add_argument("--warmup-steps", type=int, default=24)
parser.add_argument("--global-width", type=int, default=960)
parser.add_argument("--global-height", type=int, default=540)
parser.add_argument("--close-width", type=int, default=480)
parser.add_argument("--close-height", type=int, default=540)
parser.add_argument("--left-offset-x", type=float, default=0.08)
parser.add_argument("--left-offset-y", type=float, default=0.04)
parser.add_argument("--left-offset-z", type=float, default=0.06)
parser.add_argument("--right-offset-x", type=float, default=0.10)
parser.add_argument("--right-offset-y", type=float, default=-0.02)
parser.add_argument("--right-offset-z", type=float, default=0.07)
parser.add_argument(
    "--trajectory",
    choices=("smooth", "grasp", "bimanual"),
    default="bimanual",
    help="Mock controller trajectory. bimanual tests left, right, then both grippers.",
)
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
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
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    DEX1_GRIPPER_JOINTS,
    LEFT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_GRIPPER_JOINTS,
)
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
LEFT_DEX1_BODIES = ["left_dex1_base_link", "left_dex1_finger_link_1", "left_dex1_finger_link_2"]
RIGHT_DEX1_BODIES = ["right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2"]


def add_video_cameras(env_cfg) -> None:
    if args_cli.global_height != args_cli.close_height:
        raise ValueError("Global and close camera heights must match.")
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
    env_cfg.scene.left_dex1_close_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoLeftDex1CloseCamera",
        update_period=0,
        height=args_cli.close_height,
        width=args_cli.close_width,
        data_types=["rgb"],
        spawn=camera_spawn,
    )
    env_cfg.scene.right_dex1_close_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoRightDex1CloseCamera",
        update_period=0,
        height=args_cli.close_height,
        width=args_cli.close_width,
        data_types=["rgb"],
        spawn=camera_spawn,
    )
    env_cfg.scene.lazy_sensor_update = False
    env_cfg.sim.render_interval = 1


def smooth_profile(frame_index: int, num_frames: int) -> float:
    if num_frames <= 1:
        return 0.0
    phase = frame_index / float(num_frames - 1)
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def interpolate_keyframes(frame_index: int, num_frames: int, keyframes):
    if num_frames <= 1:
        return keyframes[0][1], keyframes[0][2], keyframes[0][3], keyframes[0][4], keyframes[0][5], keyframes[0][0]
    t = frame_index / float(num_frames - 1)
    for next_index in range(1, len(keyframes)):
        t0, left0, right0, left_trigger0, right_trigger0, phase0 = keyframes[next_index - 1]
        t1, left1, right1, left_trigger1, right_trigger1, phase1 = keyframes[next_index]
        if t <= t1 or next_index == len(keyframes) - 1:
            alpha = _smoothstep((t - t0) / max(t1 - t0, 1.0e-6))
            left = [float(v0 + (v1 - v0) * alpha) for v0, v1 in zip(left0, left1)]
            right = [float(v0 + (v1 - v0) * alpha) for v0, v1 in zip(right0, right1)]
            left_trigger = float(left_trigger0 + (left_trigger1 - left_trigger0) * alpha)
            right_trigger = float(right_trigger0 + (right_trigger1 - right_trigger0) * alpha)
            phase = phase1 if alpha > 0.5 else phase0
            return left, right, left_trigger, right_trigger, phase, t
    return keyframes[-1][1], keyframes[-1][2], keyframes[-1][3], keyframes[-1][4], keyframes[-1][5], t


def trajectory_command(frame_index: int, num_frames: int, left_peak: torch.Tensor, right_peak: torch.Tensor):
    zero = [0.0, 0.0, 0.0]
    left_x, left_y, left_z = [float(v) for v in left_peak.tolist()]
    right_x, right_y, right_z = [float(v) for v in right_peak.tolist()]

    if args_cli.trajectory == "smooth":
        progress = smooth_profile(frame_index, num_frames)
        right = (right_peak * progress).tolist()
        return zero, [float(v) for v in right], 0.0, float(progress), "right_smooth_open_close", float(progress)

    if args_cli.trajectory == "grasp":
        keyframes = [
            (0.00, zero, zero, 0.0, 0.0, "home_open"),
            (0.18, zero, [0.0, right_y * 0.70, right_z * 0.15], 0.0, 0.0, "right_lateral_align_open"),
            (0.38, zero, [right_x * 0.75, right_y * 0.90, right_z * 0.35], 0.0, 0.0, "right_forward_approach_open"),
            (0.56, zero, [right_x, right_y, right_z], 0.0, 0.0, "right_descend_to_grasp_open"),
            (0.68, zero, [right_x, right_y, right_z], 0.0, 1.0, "right_close_gripper"),
            (0.82, zero, [right_x, right_y, right_z * 0.45], 0.0, 1.0, "right_lift_closed"),
            (1.00, zero, [right_x * 0.35, right_y * 0.35, right_z * 0.20], 0.0, 0.0, "right_retreat_reopen"),
        ]
        return interpolate_keyframes(frame_index, num_frames, keyframes)

    keyframes = [
        (0.00, zero, zero, 0.0, 0.0, "home_open"),
        (0.16, [left_x, left_y, left_z], zero, 0.0, 0.0, "left_reach_open"),
        (0.28, [left_x, left_y, left_z], zero, 1.0, 0.0, "left_close"),
        (0.42, zero, [right_x, right_y, right_z], 0.0, 0.0, "right_reach_open"),
        (0.54, zero, [right_x, right_y, right_z], 0.0, 1.0, "right_close"),
        (0.72, [left_x, left_y, left_z], [right_x, right_y, right_z], 0.0, 0.0, "both_reach_open"),
        (0.86, [left_x, left_y, left_z], [right_x, right_y, right_z], 1.0, 1.0, "both_close"),
        (1.00, [left_x * 0.25, left_y * 0.25, left_z * 0.25], [right_x * 0.25, right_y * 0.25, right_z * 0.25], 0.0, 0.0, "retreat_reopen"),
    ]
    return interpolate_keyframes(frame_index, num_frames, keyframes)


def _controller_packet(position: list[float], quat: list[float], trigger: float) -> np.ndarray:
    pose = np.array([*position, *quat], dtype=np.float32)
    inputs = np.zeros(7, dtype=np.float32)
    inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value] = float(trigger)
    return np.stack([pose, inputs])


def make_controller_data(left_position, left_quat, left_trigger, right_position, right_quat, right_trigger) -> dict:
    return {
        DeviceBase.TrackingTarget.CONTROLLER_LEFT: _controller_packet(left_position, left_quat, left_trigger),
        DeviceBase.TrackingTarget.CONTROLLER_RIGHT: _controller_packet(right_position, right_quat, right_trigger),
    }


def body_pose_env_frame(env, robot, body_id: int) -> tuple[list[float], list[float]]:
    pos = robot.data.body_pos_w[0, body_id] - env.unwrapped.scene.env_origins[0]
    quat = robot.data.body_quat_w[0, body_id]
    return [float(v) for v in pos.detach().cpu().tolist()], [float(v) for v in quat.detach().cpu().tolist()]


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
        (retargeter for retargeter in gripper_retargeters if getattr(retargeter, "_bound_controller", None) == DeviceBase.TrackingTarget.CONTROLLER_LEFT),
        None,
    )
    right_gripper = next(
        (retargeter for retargeter in gripper_retargeters if getattr(retargeter, "_bound_controller", None) == DeviceBase.TrackingTarget.CONTROLLER_RIGHT),
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


def set_close_camera_pose(robot, body_ids: list[int], camera: Camera, side: str) -> None:
    target = robot.data.body_pos_w[0, body_ids].mean(dim=0)
    view_dir = torch.tensor([0.75, 0.55 if side == "left" else -0.55, 0.28], dtype=torch.float32, device=robot.device)
    view_dir = view_dir / torch.linalg.norm(view_dir)
    eye = target + view_dir * 0.58
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))


def rgb_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()[..., :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image.copy()
    y = 30
    for line in lines:
        cv2.putText(output, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
        y += 28
    return output


def measure_side(robot, wrist_body_id: int, joint_ids: list[int], body_ids: list[int]) -> dict:
    body_pos = robot.data.body_pos_w[0, body_ids]
    return {
        "wrist_pos_w": [float(v) for v in robot.data.body_pos_w[0, wrist_body_id].detach().cpu().tolist()],
        "gripper_joint_pos": [float(v) for v in robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()],
        "finger_body_separation_m": float(torch.linalg.norm(body_pos[1] - body_pos[2]).item()),
    }


def measure(robot, left_wrist_id, right_wrist_id, left_joint_ids, right_joint_ids, left_body_ids, right_body_ids) -> dict:
    return {
        "left": measure_side(robot, left_wrist_id, left_joint_ids, left_body_ids),
        "right": measure_side(robot, right_wrist_id, right_joint_ids, right_body_ids),
    }


def make_composite(global_camera, left_camera, right_camera, frame, progress, phase, left_trigger, right_trigger, action, measured):
    global_rgb = annotate(
        rgb_tensor_to_uint8(global_camera.data.output["rgb"][0]),
        [
            "Mock Pico bilateral motion-controller retargeter",
            "Global scene view",
            f"frame {frame:03d} progress {progress:.2f}",
            f"phase: {phase}",
        ],
    )
    left_rgb = annotate(
        rgb_tensor_to_uint8(left_camera.data.output["rgb"][0]),
        [
            "Left wrist + Dex1",
            f"trigger: {left_trigger:.2f}",
            "joints: {:.4f}, {:.4f} m".format(*measured["left"]["gripper_joint_pos"]),
            f"sep: {measured['left']['finger_body_separation_m']:.4f} m",
        ],
    )
    right_rgb = annotate(
        rgb_tensor_to_uint8(right_camera.data.output["rgb"][0]),
        [
            "Right wrist + Dex1",
            f"action shape: {tuple(action.shape)}",
            f"trigger: {right_trigger:.2f}",
            "joints: {:.4f}, {:.4f} m".format(*measured["right"]["gripper_joint_pos"]),
            f"sep: {measured['right']['finger_body_separation_m']:.4f} m",
        ],
    )
    return np.concatenate([global_rgb, left_rgb, right_rgb], axis=1)


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("Video recording currently expects --num-envs 1.")
    out_mp4 = Path(args_cli.out_mp4)
    out_summary = Path(args_cli.summary)
    out_dir = Path(args_cli.out_dir)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    add_video_cameras(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    try:
        robot = env.unwrapped.scene["robot"]
        global_camera = env.unwrapped.scene.sensors["global_view_cam"]
        left_camera = env.unwrapped.scene.sensors["left_dex1_close_cam"]
        right_camera = env.unwrapped.scene.sensors["right_dex1_close_cam"]
        left_wrist_ids, left_wrist_names = robot.find_bodies([LEFT_WRIST_BODY], preserve_order=True)
        right_wrist_ids, right_wrist_names = robot.find_bodies([RIGHT_WRIST_BODY], preserve_order=True)
        left_joint_ids, left_joint_names = robot.find_joints(LEFT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        right_joint_ids, right_joint_names = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        left_body_ids, left_body_names = robot.find_bodies(LEFT_DEX1_BODIES, preserve_order=True)
        right_body_ids, right_body_names = robot.find_bodies(RIGHT_DEX1_BODIES, preserve_order=True)
        left_default_pos, left_quat = body_pose_env_frame(env, robot, left_wrist_ids[0])
        right_default_pos, right_quat = body_pose_env_frame(env, robot, right_wrist_ids[0])
        wrist_retargeter, left_gripper, right_gripper, configured_retargeters = make_retargeters_from_env_cfg(env_cfg)
        left_offset_peak = torch.tensor([args_cli.left_offset_x, args_cli.left_offset_y, args_cli.left_offset_z], dtype=torch.float32)
        right_offset_peak = torch.tensor([args_cli.right_offset_x, args_cli.right_offset_y, args_cli.right_offset_z], dtype=torch.float32)

        set_global_camera_pose(env, robot, global_camera)
        for _ in range(args_cli.warmup_steps):
            raw = make_controller_data(left_default_pos, left_quat, 0.0, right_default_pos, right_quat, 0.0)
            env.step(retarget_action(wrist_retargeter, left_gripper, right_gripper, raw))
        initial = measure(robot, left_wrist_ids[0], right_wrist_ids[0], left_joint_ids, right_joint_ids, left_body_ids, right_body_ids)

        video_size = (args_cli.global_width + args_cli.close_width * 2, args_cli.global_height)
        video = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.fps, video_size)
        if not video.isOpened():
            raise RuntimeError(f"Could not open video writer: {out_mp4}")

        key_frames = {0, args_cli.frames // 4, args_cli.frames // 2, (args_cli.frames * 3) // 4, args_cli.frames - 1}
        frame_records = []
        for frame in range(args_cli.frames):
            left_offset, right_offset, left_trigger, right_trigger, phase, progress = trajectory_command(
                frame, args_cli.frames, left_offset_peak, right_offset_peak
            )
            left_target_pos = [left_default_pos[i] + float(left_offset[i]) for i in range(3)]
            right_target_pos = [right_default_pos[i] + float(right_offset[i]) for i in range(3)]
            raw = make_controller_data(left_target_pos, left_quat, left_trigger, right_target_pos, right_quat, right_trigger)
            action = retarget_action(wrist_retargeter, left_gripper, right_gripper, raw)
            set_close_camera_pose(robot, left_body_ids, left_camera, "left")
            set_close_camera_pose(robot, right_body_ids, right_camera, "right")
            env.step(action)
            measured = measure(robot, left_wrist_ids[0], right_wrist_ids[0], left_joint_ids, right_joint_ids, left_body_ids, right_body_ids)
            composite = make_composite(
                global_camera, left_camera, right_camera, frame, progress, phase, left_trigger, right_trigger, action, measured
            )
            video.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            if frame in key_frames:
                cv2.imwrite(str(out_dir / f"frame_{frame:03d}.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            frame_records.append(
                {
                    "frame": frame,
                    "progress": float(progress),
                    "phase": phase,
                    "mock_left_trigger": float(left_trigger),
                    "mock_right_trigger": float(right_trigger),
                    "mock_left_controller_offset": [float(v) for v in left_offset],
                    "mock_right_controller_offset": [float(v) for v in right_offset],
                    "mock_left_controller_position": [float(v) for v in left_target_pos],
                    "mock_right_controller_position": [float(v) for v in right_target_pos],
                    "retargeted_action_shape": list(action.shape),
                    **measured,
                }
            )

        video.release()
        if not out_mp4.exists() or out_mp4.stat().st_size == 0:
            raise RuntimeError(f"Video file was not written: {out_mp4}")

        left_distances = [
            torch.linalg.norm(torch.tensor(record["left"]["wrist_pos_w"]) - torch.tensor(initial["left"]["wrist_pos_w"])).item()
            for record in frame_records
        ]
        right_distances = [
            torch.linalg.norm(torch.tensor(record["right"]["wrist_pos_w"]) - torch.tensor(initial["right"]["wrist_pos_w"])).item()
            for record in frame_records
        ]
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task": args_cli.task,
            "out_mp4": str(out_mp4),
            "out_dir": str(out_dir),
            "trajectory": args_cli.trajectory,
            "requested_left_offset_peak": [float(v) for v in left_offset_peak.tolist()],
            "requested_right_offset_peak": [float(v) for v in right_offset_peak.tolist()],
            "fps": args_cli.fps,
            "frames": args_cli.frames,
            "video_size": list(video_size),
            "configured_motion_controller_retargeters": configured_retargeters,
            "action_space": str(env.action_space),
            "action_terms": env.unwrapped.action_manager.active_terms,
            "action_term_dims": env.unwrapped.action_manager.action_term_dim,
            "left_wrist_names": left_wrist_names,
            "right_wrist_names": right_wrist_names,
            "left_gripper_joint_names": left_joint_names,
            "right_gripper_joint_names": right_joint_names,
            "all_gripper_joint_names": DEX1_GRIPPER_JOINTS,
            "left_gripper_body_names": left_body_names,
            "right_gripper_body_names": right_body_names,
            "initial": initial,
            "midpoint": frame_records[len(frame_records) // 2],
            "final": frame_records[-1],
            "max_left_wrist_displacement_m": float(max(left_distances)),
            "max_right_wrist_displacement_m": float(max(right_distances)),
            "frame_records": frame_records,
            "bytes": out_mp4.stat().st_size,
        }
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote bilateral motion-controller retargeter video: {out_mp4}")
        print(f"[INFO] Wrote summary: {out_summary}")
        print(
            "[INFO] max wrist displacement: "
            f"left={summary['max_left_wrist_displacement_m']:.6f} m, "
            f"right={summary['max_right_wrist_displacement_m']:.6f} m"
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
