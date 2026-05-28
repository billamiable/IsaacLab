# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render mock Pico motion-controller retargeting for fixed-base G1 Dex1."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-IK-Scene-v0"
DEFAULT_OUT_MP4 = "/workspace/host/out/g1_dex1_motion_controller_retargeter_pipeline.mp4"
DEFAULT_OUT_SUMMARY = "/workspace/host/out/g1_dex1_motion_controller_retargeter_pipeline.json"
DEFAULT_OUT_DIR = "/workspace/host/out/g1_dex1_motion_controller_retargeter_pipeline_frames"


parser = argparse.ArgumentParser(description="Render G1 Dex1 mock Pico motion-controller retargeter pipeline.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-mp4", default=DEFAULT_OUT_MP4, help="Output MP4 path.")
parser.add_argument("--summary", default=DEFAULT_OUT_SUMMARY, help="Output JSON summary path.")
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for key-frame PNGs.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--frames", type=int, default=144)
parser.add_argument("--warmup-steps", type=int, default=24)
parser.add_argument("--global-width", type=int, default=960)
parser.add_argument("--global-height", type=int, default=540)
parser.add_argument("--close-width", type=int, default=640)
parser.add_argument("--close-height", type=int, default=540)
parser.add_argument("--right-offset-x", type=float, default=0.10)
parser.add_argument("--right-offset-y", type=float, default=-0.02)
parser.add_argument("--right-offset-z", type=float, default=0.07)
parser.add_argument(
    "--trajectory",
    choices=("smooth", "grasp"),
    default="smooth",
    help="Mock controller trajectory. grasp adds lateral alignment, approach, descend, close, lift, and retreat.",
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
    RIGHT_DEX1_GRIPPER_JOINTS,
)
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
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


def interpolate_keyframes(
    frame_index: int,
    num_frames: int,
    keyframes: list[tuple[float, list[float], float, str]],
) -> tuple[list[float], float, str, float]:
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


def trajectory_command(frame_index: int, num_frames: int, peak: torch.Tensor) -> tuple[list[float], float, str, float]:
    if args_cli.trajectory == "smooth":
        progress = smooth_profile(frame_index, num_frames)
        offset = (peak * progress).tolist()
        return [float(v) for v in offset], float(progress), "smooth_open_close", float(progress)

    peak_x, peak_y, peak_z = [float(v) for v in peak.tolist()]
    keyframes = [
        (0.00, [0.0, 0.0, 0.0], 0.0, "home_open"),
        (0.18, [0.0, peak_y * 0.70, peak_z * 0.15], 0.0, "lateral_align_open"),
        (0.38, [peak_x * 0.75, peak_y * 0.90, peak_z * 0.35], 0.0, "forward_approach_open"),
        (0.56, [peak_x, peak_y, peak_z], 0.0, "descend_to_grasp_open"),
        (0.68, [peak_x, peak_y, peak_z], 1.0, "close_gripper"),
        (0.82, [peak_x, peak_y, peak_z * 0.45], 1.0, "lift_closed"),
        (1.00, [peak_x * 0.35, peak_y * 0.35, peak_z * 0.20], 0.0, "retreat_reopen"),
    ]
    return interpolate_keyframes(frame_index, num_frames, keyframes)

def make_controller_data(position: list[float], quat: list[float], trigger: float) -> dict:
    pose = np.array([*position, *quat], dtype=np.float32)
    inputs = np.zeros(7, dtype=np.float32)
    inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value] = float(trigger)
    return {DeviceBase.TrackingTarget.CONTROLLER_RIGHT: np.stack([pose, inputs])}


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
    gripper_retargeter = next(
        retargeter for retargeter in retargeters if isinstance(retargeter, GripperTriggerOrPinchRetargeter)
    )
    return wrist_retargeter, gripper_retargeter, [type(retargeter).__name__ for retargeter in retargeters]


def retarget_action(wrist_retargeter, gripper_retargeter, raw_data: dict) -> torch.Tensor:
    return torch.cat([wrist_retargeter.retarget(raw_data), gripper_retargeter.retarget(raw_data)], dim=-1).unsqueeze(0)


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


def set_close_camera_pose(robot, body_ids: list[int], camera: Camera) -> None:
    target = robot.data.body_pos_w[0, body_ids].mean(dim=0)
    view_dir = torch.tensor([0.75, -0.55, 0.28], dtype=torch.float32, device=robot.device)
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
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
        y += 30
    return output


def measure(robot, right_body_id: int, gripper_joint_ids: list[int], gripper_body_ids: list[int]) -> dict:
    body_pos = robot.data.body_pos_w[0, gripper_body_ids]
    return {
        "right_wrist_pos_w": [
            float(v) for v in robot.data.body_pos_w[0, right_body_id].detach().cpu().tolist()
        ],
        "gripper_joint_pos": [
            float(v) for v in robot.data.joint_pos[0, gripper_joint_ids].detach().cpu().tolist()
        ],
        "finger_body_separation_m": float(torch.linalg.norm(body_pos[1] - body_pos[2]).item()),
    }


def make_composite(
    global_camera: Camera,
    close_camera: Camera,
    frame: int,
    progress: float,
    phase: str,
    trigger: float,
    action: torch.Tensor,
    measured: dict,
) -> np.ndarray:
    global_rgb = annotate(
        rgb_tensor_to_uint8(global_camera.data.output["rgb"][0]),
        [
            "Mock Pico motion-controller retargeter",
            "Global scene view",
            f"frame {frame:03d} progress {progress:.2f}",
            f"phase: {phase}",
        ],
    )
    close_rgb = annotate(
        rgb_tensor_to_uint8(close_camera.data.output["rgb"][0]),
        [
            "Right wrist + Dex1 close-up",
            f"retargeted action shape: {tuple(action.shape)}",
            f"mock trigger: {trigger:.2f}",
            "gripper joints: {:.4f}, {:.4f} m".format(*measured["gripper_joint_pos"]),
            "finger separation: {:.4f} m".format(measured["finger_body_separation_m"]),
        ],
    )
    return np.concatenate([global_rgb, close_rgb], axis=1)

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
        close_camera = env.unwrapped.scene.sensors["right_dex1_close_cam"]
        left_ids, _ = robot.find_bodies([LEFT_WRIST_BODY], preserve_order=True)
        right_ids, right_names = robot.find_bodies([RIGHT_WRIST_BODY], preserve_order=True)
        joint_ids, joint_names = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        body_ids, body_names = robot.find_bodies(RIGHT_DEX1_BODIES, preserve_order=True)
        _, right_quat = body_pose_env_frame(env, robot, right_ids[0])
        wrist_retargeter, gripper_retargeter, configured_retargeters = make_retargeters_from_env_cfg(env_cfg)
        right_offset_peak = torch.tensor(
            [args_cli.right_offset_x, args_cli.right_offset_y, args_cli.right_offset_z], dtype=torch.float32
        )

        set_global_camera_pose(env, robot, global_camera)
        for _ in range(args_cli.warmup_steps):
            raw = make_controller_data([0.0, 0.0, 0.0], right_quat, trigger=0.0)
            env.step(retarget_action(wrist_retargeter, gripper_retargeter, raw))
        initial = measure(robot, right_ids[0], joint_ids, body_ids)

        video_size = (args_cli.global_width + args_cli.close_width, args_cli.global_height)
        video = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.fps, video_size)
        if not video.isOpened():
            raise RuntimeError(f"Could not open video writer: {out_mp4}")

        key_frames = {0, args_cli.frames // 2, args_cli.frames - 1}
        frame_records = []
        for frame in range(args_cli.frames):
            offset, trigger, phase, progress = trajectory_command(frame, args_cli.frames, right_offset_peak)
            raw = make_controller_data(offset, right_quat, trigger=trigger)
            action = retarget_action(wrist_retargeter, gripper_retargeter, raw)
            set_close_camera_pose(robot, body_ids, close_camera)
            env.step(action)
            measured = measure(robot, right_ids[0], joint_ids, body_ids)
            composite = make_composite(global_camera, close_camera, frame, progress, phase, trigger, action, measured)
            video.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            if frame in key_frames:
                cv2.imwrite(str(out_dir / f"frame_{frame:03d}.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            frame_records.append(
                {
                    "frame": frame,
                    "progress": float(progress),
                    "phase": phase,
                    "mock_trigger": float(trigger),
                    "mock_right_controller_offset": [float(v) for v in offset],
                    "retargeted_action_shape": list(action.shape),
                    **measured,
                }
            )

        video.release()
        if not out_mp4.exists() or out_mp4.stat().st_size == 0:
            raise RuntimeError(f"Video file was not written: {out_mp4}")

        midpoint = frame_records[len(frame_records) // 2]
        displacement = torch.linalg.norm(
            torch.tensor(midpoint["right_wrist_pos_w"]) - torch.tensor(initial["right_wrist_pos_w"])
        ).item()
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task": args_cli.task,
            "out_mp4": str(out_mp4),
            "out_dir": str(out_dir),
            "trajectory": args_cli.trajectory,
            "requested_right_offset_peak": [float(v) for v in right_offset_peak.tolist()],
            "fps": args_cli.fps,
            "frames": args_cli.frames,
            "video_size": list(video_size),
            "configured_motion_controller_retargeters": configured_retargeters,
            "action_space": str(env.action_space),
            "action_terms": env.unwrapped.action_manager.active_terms,
            "action_term_dims": env.unwrapped.action_manager.action_term_dim,
            "right_body_names": right_names,
            "gripper_joint_names": joint_names,
            "gripper_body_names": body_names,
            "initial": initial,
            "midpoint": midpoint,
            "final": frame_records[-1],
            "midpoint_wrist_displacement_m": float(displacement),
            "frame_records": frame_records,
            "bytes": out_mp4.stat().st_size,
        }
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote motion-controller retargeter video: {out_mp4}")
        print(f"[INFO] Wrote summary: {out_summary}")
        print(f"[INFO] midpoint wrist displacement: {displacement:.6f} m")
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
