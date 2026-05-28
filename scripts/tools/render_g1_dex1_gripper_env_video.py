# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the manager-based G1 Dex1 gripper-only teleop pipeline.

This script records the formal Isaac Lab environment path:

mock Pico right trigger -> existing gripper retargeter -> env.step(action) -> Action Manager -> Dex1 joints

The MP4 is a two-view composite:

* left: global full-body camera, so the capture is not misleadingly cropped
* right: right Dex1 close-up camera, so the gripper open/close motion is visible

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/render_g1_dex1_gripper_env_video.py \
        --headless --enable_cameras --task Isaac-G1-Dex1-Gripper-Only-v0

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-Gripper-Only-v0"
DEFAULT_OUT_MP4 = "/workspace/host/out/g1_dex1_gripper_only_env_pipeline.mp4"
DEFAULT_OUT_SUMMARY = "/workspace/host/out/g1_dex1_gripper_only_env_pipeline.json"
DEFAULT_OUT_DIR = "/workspace/host/out/g1_dex1_gripper_only_env_pipeline_frames"
RIGHT_DEX1_BODIES = ["right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2"]


parser = argparse.ArgumentParser(description="Render the G1 Dex1 gripper-only manager-based env pipeline.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-mp4", default=DEFAULT_OUT_MP4, help="Output MP4 path.")
parser.add_argument("--summary", default=DEFAULT_OUT_SUMMARY, help="Output JSON summary path.")
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for key-frame PNGs.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments. Video expects 1.")
parser.add_argument("--fps", type=int, default=24, help="Video frames per second.")
parser.add_argument("--frames", type=int, default=120, help="Number of frames to render.")
parser.add_argument("--warmup-steps", type=int, default=24, help="Initial env steps to settle the open command.")
parser.add_argument("--controller-threshold", type=float, default=0.5, help="Mock Pico trigger threshold.")
parser.add_argument("--global-width", type=int, default=960)
parser.add_argument("--global-height", type=int, default=540)
parser.add_argument("--close-width", type=int, default=640)
parser.add_argument("--close-height", type=int, default=540)
parser.add_argument("--global-distance-scale", type=float, default=3.2, help="Distance multiplier for full-body view.")
parser.add_argument("--close-distance", type=float, default=0.55, help="Distance from right gripper for close-up view.")
parser.add_argument(
    "--skip-kit-cleanup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit directly after writing video to avoid Kit shutdown hangs in headless CI.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from isaaclab.devices.openxr.retargeters.manipulator.gripper_trigger_or_pinch_retargeter import (
    GripperTriggerOrPinchRetargeter,
    GripperTriggerOrPinchRetargeterCfg,
)
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    RIGHT_DEX1_CLOSE,
    RIGHT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_OPEN,
)
from isaaclab_tasks.utils import parse_env_cfg


def add_video_cameras(env_cfg) -> None:
    """Attach two RGB camera sensors to the same scene used by the gym environment."""
    if args_cli.global_height != args_cli.close_height:
        raise ValueError("Global and close camera heights must match for side-by-side video composition.")

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


def make_mock_pico_controller_data(trigger: float) -> dict:
    """Build OpenXRDevice-style raw data for the right Pico motion controller."""
    pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    inputs = np.zeros(7, dtype=np.float32)
    inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value] = float(trigger)
    return {DeviceBase.TrackingTarget.CONTROLLER_RIGHT: np.stack([pose, inputs])}


def make_gripper_retargeter(device: str) -> GripperTriggerOrPinchRetargeter:
    """Create the same trigger retargeter configured by the motion-controller device."""
    cfg = GripperTriggerOrPinchRetargeterCfg(
        bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
        bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
        controller_threshold=args_cli.controller_threshold,
        sim_device=device,
    )
    return GripperTriggerOrPinchRetargeter(cfg)


def smooth_close_open_trigger(frame_index: int, num_frames: int) -> float:
    """Return a smooth mock trigger profile: open trigger 0 -> close trigger 1 -> open trigger 0."""
    if num_frames <= 1:
        return 0.0
    phase = frame_index / float(num_frames - 1)
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)


def compute_scene_bounds(env, robot) -> tuple[torch.Tensor, float]:
    """Compute a conservative target and radius from robot bodies plus scene object roots."""
    positions = [robot.data.body_pos_w[0]]
    for rigid_object in env.unwrapped.scene.rigid_objects.values():
        positions.append(rigid_object.data.root_pos_w)
    body_pos = torch.cat(positions, dim=0)
    minimum = body_pos.min(dim=0).values
    maximum = body_pos.max(dim=0).values
    center = (minimum + maximum) * 0.5
    center[2] = torch.clamp(center[2], min=1.0)
    diagonal = torch.linalg.norm(maximum - minimum).item()
    radius = max(diagonal * 0.5, 1.0)
    return center, radius


def set_global_camera_pose(env, robot, camera: Camera) -> dict:
    """Place the global camera far enough to see the robot and task scene."""
    center, radius = compute_scene_bounds(env, robot)
    view_dir = torch.tensor([1.6, -2.2, 0.75], dtype=torch.float32, device=robot.device)
    view_dir = view_dir / torch.linalg.norm(view_dir)
    eye = center + view_dir * max(radius * args_cli.global_distance_scale, 3.0)
    camera.set_world_poses_from_view(eye.unsqueeze(0), center.unsqueeze(0))
    return {
        "eye": [float(value) for value in eye.detach().cpu().tolist()],
        "target": [float(value) for value in center.detach().cpu().tolist()],
        "radius": float(radius),
    }


def gripper_target(robot, body_ids: list[int]) -> torch.Tensor:
    """Return the current average world position of the right Dex1 bodies."""
    return robot.data.body_pos_w[0, body_ids].mean(dim=0)


def set_close_camera_pose(robot, body_ids: list[int], camera: Camera) -> dict:
    """Place the close-up camera to look at the right Dex1 gripper."""
    target = gripper_target(robot, body_ids)
    view_dir = torch.tensor([0.75, -0.55, 0.28], dtype=torch.float32, device=robot.device)
    view_dir = view_dir / torch.linalg.norm(view_dir)
    eye = target + view_dir * args_cli.close_distance
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
    return {
        "eye": [float(value) for value in eye.detach().cpu().tolist()],
        "target": [float(value) for value in target.detach().cpu().tolist()],
    }


def rgb_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert Isaac Lab camera RGB tensor to uint8 RGB numpy array."""
    array = image.detach().cpu().numpy()
    array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a compact text overlay on an RGB image."""
    output = image.copy()
    x, y = 16, 30
    for line in lines:
        cv2.putText(output, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
        y += 30
    return output


def make_composite(
    global_camera: Camera,
    close_camera: Camera,
    frame_index: int,
    trigger: float,
    retargeter_output: float,
    env_action: float,
    target_values: list[float],
    measured_values: list[float],
    visual_state: str,
) -> np.ndarray:
    """Create one RGB composite frame."""
    global_rgb = rgb_tensor_to_uint8(global_camera.data.output["rgb"][0])
    close_rgb = rgb_tensor_to_uint8(close_camera.data.output["rgb"][0])
    global_rgb = annotate(
        global_rgb,
        [
            "Isaac Lab env pipeline",
            "Global full-body view",
            f"frame {frame_index:03d}",
        ],
    )
    close_rgb = annotate(
        close_rgb,
        [
            "Right Dex1 gripper close-up",
            f"mock Pico right trigger: {trigger:.2f}",
            f"retargeter output: {retargeter_output:+.1f}",
            f"env action: {env_action:+.1f}  state: {visual_state}",
            f"target: {target_values[0]:.4f}, {target_values[1]:.4f} m",
            f"measured: {measured_values[0]:.4f}, {measured_values[1]:.4f} m",
        ],
    )
    return np.concatenate([global_rgb, close_rgb], axis=1)


def step_env_command(env, action_value: float, steps: int = 1) -> None:
    """Step the environment through its action manager."""
    action = torch.full(env.action_space.shape, float(action_value), device=env.unwrapped.device)
    for _ in range(steps):
        env.step(action)


def measure_gripper(robot, joint_ids: list[int], body_ids: list[int]) -> tuple[list[float], float]:
    """Measure gripper joint positions and finger-body separation."""
    measured = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
    body_pos = robot.data.body_pos_w[0, body_ids]
    if len(body_ids) >= 3:
        separation = torch.linalg.norm(body_pos[1] - body_pos[2]).item()
    else:
        separation = 0.0
    return [float(value) for value in measured], float(separation)


def write_video_summary(summary_path: Path, summary: dict) -> None:
    """Write the JSON summary."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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

        joint_ids, joint_names = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        if len(joint_ids) != len(RIGHT_DEX1_GRIPPER_JOINTS):
            raise RuntimeError(f"Could not resolve gripper joints. resolved={joint_names}")
        body_ids, body_names = robot.find_bodies(RIGHT_DEX1_BODIES, preserve_order=True)
        if len(body_ids) != len(RIGHT_DEX1_BODIES):
            raise RuntimeError(f"Could not resolve right Dex1 bodies. resolved={body_names}")

        open_target = [RIGHT_DEX1_OPEN, RIGHT_DEX1_OPEN]
        close_target = [RIGHT_DEX1_CLOSE, RIGHT_DEX1_CLOSE]
        retargeter = make_gripper_retargeter(env.unwrapped.device)

        global_camera_info = set_global_camera_pose(env, robot, global_camera)
        close_camera_info = set_close_camera_pose(robot, body_ids, close_camera)

        step_env_command(env, 1.0, args_cli.warmup_steps)
        open_measured, open_separation = measure_gripper(robot, joint_ids, body_ids)
        step_env_command(env, -1.0, args_cli.warmup_steps)
        close_measured, close_separation = measure_gripper(robot, joint_ids, body_ids)
        step_env_command(env, 1.0, args_cli.warmup_steps)

        open_visual_state = "open" if open_separation >= close_separation else "closed"
        close_visual_state = "closed" if open_separation >= close_separation else "open"

        video_size = (args_cli.global_width + args_cli.close_width, args_cli.global_height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(str(out_mp4), fourcc, args_cli.fps, video_size)
        if not video.isOpened():
            raise RuntimeError(f"Could not open video writer: {out_mp4}")

        key_frame_indices = {0, args_cli.frames // 2, args_cli.frames - 1}
        frame_records = []
        for frame_index in range(args_cli.frames):
            trigger = smooth_close_open_trigger(frame_index, args_cli.frames)
            retargeter_output = float(
                retargeter.retarget(make_mock_pico_controller_data(trigger))[0].detach().cpu().item()
            )
            env_action = 1.0 if retargeter_output >= 0.0 else -1.0
            target_values = open_target if env_action >= 0.0 else close_target
            visual_state = open_visual_state if env_action >= 0.0 else close_visual_state

            set_close_camera_pose(robot, body_ids, close_camera)
            step_env_command(env, env_action, 1)
            measured_values, separation = measure_gripper(robot, joint_ids, body_ids)

            composite = make_composite(
                global_camera,
                close_camera,
                frame_index,
                trigger,
                retargeter_output,
                env_action,
                target_values,
                measured_values,
                visual_state,
            )
            video.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            if frame_index in key_frame_indices:
                cv2.imwrite(str(out_dir / f"frame_{frame_index:03d}.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            frame_records.append(
                {
                    "frame": frame_index,
                    "mock_pico_right_trigger": float(trigger),
                    "retargeter_output": float(retargeter_output),
                    "env_action": float(env_action),
                    "target": [float(value) for value in target_values],
                    "measured": measured_values,
                    "finger_body_separation_m": separation,
                    "visual_state": visual_state,
                }
            )

        video.release()
        if not out_mp4.exists() or out_mp4.stat().st_size == 0:
            raise RuntimeError(f"Video file was not written: {out_mp4}")

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task": args_cli.task,
            "out_mp4": str(out_mp4),
            "out_dir": str(out_dir),
            "fps": args_cli.fps,
            "frames": args_cli.frames,
            "video_size": list(video_size),
            "control_path": "mock Pico right trigger -> GripperTriggerOrPinchRetargeter -> env.step(action) -> Action Manager -> right Dex1 joints",
            "controller_threshold": args_cli.controller_threshold,
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "joint_names": joint_names,
            "joint_ids": [int(joint_id) for joint_id in joint_ids],
            "body_names": body_names,
            "body_ids": [int(body_id) for body_id in body_ids],
            "open_target": open_target,
            "close_target": close_target,
            "open_probe": {"measured": open_measured, "finger_body_separation_m": open_separation},
            "close_probe": {"measured": close_measured, "finger_body_separation_m": close_separation},
            "open_visual_state": open_visual_state,
            "close_visual_state": close_visual_state,
            "global_camera": global_camera_info,
            "close_camera_initial": close_camera_info,
            "frame_records": frame_records,
            "bytes": out_mp4.stat().st_size,
        }
        write_video_summary(out_summary, summary)
        print(f"[INFO] Wrote env pipeline video: {out_mp4}")
        print(f"[INFO] Wrote key frames: {out_dir}")
        print(f"[INFO] Wrote summary: {out_summary}")
        print(
            "[INFO] open separation={:.6f} m, close separation={:.6f} m, inferred states: open_probe={}, close_probe={}".format(
                open_separation,
                close_separation,
                open_visual_state,
                close_visual_state,
            )
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
