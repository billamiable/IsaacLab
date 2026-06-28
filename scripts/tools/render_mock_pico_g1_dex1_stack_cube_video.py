# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render mock Pico input driving the Lab3 G1 + Dex1 stack-cube slice.

The script is a migration gate for the G1 Dex1 teleop enablement work.  It uses
the real Lab3 manager-based environment and Pink IK action term, but replaces a
physical Pico device with a scripted controller stream:

* mock right-controller position -> right wrist target
* mock right trigger scalar -> Dex1 prismatic gripper joint targets

The cube is assisted once the front Dex1 claws close around it.  This keeps the
acceptance test deterministic while still verifying the actual robot, action
manager, Pink IK, gripper joints, rendering, and output-video path.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-StackCube-v0"
DEFAULT_OUT_DIR = "/workspace/host/out/isaaclab3/g1_dex1_stack_cube"
DEFAULT_OUT_MP4 = f"{DEFAULT_OUT_DIR}/mock_pico_g1_dex1_stack_cube.mp4"
DEFAULT_OUT_JSON = f"{DEFAULT_OUT_DIR}/mock_pico_g1_dex1_stack_cube_summary.json"
DEFAULT_OUT_FRAMES = f"{DEFAULT_OUT_DIR}/frames"


parser = argparse.ArgumentParser(description="Render mock Pico input driving G1 Dex1 stack-cube in Isaac Lab 3.")
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--out-mp4", default=DEFAULT_OUT_MP4)
parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
parser.add_argument("--out-frames", default=DEFAULT_OUT_FRAMES)
parser.add_argument("--frames", type=int, default=192)
parser.add_argument("--warmup-steps", type=int, default=32)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--cube", choices=("cube_1", "cube_2", "cube_3"), default="cube_2")
parser.add_argument("--cube-x", type=float, default=0.36)
parser.add_argument("--cube-y", type=float, default=-0.02)
parser.add_argument("--cube-z", type=float, default=0.9535)
parser.add_argument("--pregrasp-height", type=float, default=0.12)
parser.add_argument("--grasp-z-offset", type=float, default=0.012)
parser.add_argument("--lift-height", type=float, default=0.14)
parser.add_argument("--retreat-x", type=float, default=-0.03)
parser.add_argument("--grasp-center-x-offset", type=float, default=0.0)
parser.add_argument("--grasp-center-y-offset", type=float, default=0.0)
parser.add_argument("--grasp-center-z-offset", type=float, default=0.0)
parser.add_argument("--attach-trigger-threshold", type=float, default=0.65)
parser.add_argument("--attach-distance-threshold", type=float, default=0.085)
parser.add_argument("--lift-success-threshold", type=float, default=0.04)
parser.add_argument("--keyframe-every", type=int, default=48)
parser.add_argument("--skip-kit-cleanup", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
args_cli.enable_cameras = True
if args_cli.rendering_mode is None:
    args_cli.rendering_mode = "balanced"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_stack_cube_env_cfg import (
    DEX1_CLOSE,
    DEX1_OPEN,
    LEFT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_GRIPPER_JOINTS,
)
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST = "left_wrist_yaw_link"
RIGHT_WRIST = "right_wrist_yaw_link"
RIGHT_DEX1_BODIES = ["right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2"]

# Same wrist-to-front-claw-center offset used in the validated 2.3.2 retargeting path.
RIGHT_WRIST_TO_GRIPPER_CENTER = (0.15050695836544037, -6.29723072052002e-05, -8.344650268554688e-06)
DEX1_FINGER_1_PAD_CENTER = (0.1082509, -0.0315503, 0.0)
DEX1_FINGER_2_PAD_CENTER = (0.1097630, 0.0313668, 0.0)


def _ensure_output_dirs() -> None:
    Path(args_cli.out_mp4).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out_frames).mkdir(parents=True, exist_ok=True)


def _configure_scene(env_cfg) -> None:
    getattr(env_cfg.scene, args_cli.cube).init_state.pos = [args_cli.cube_x, args_cli.cube_y, args_cli.cube_z]
    env_cfg.scene.mock_pico_front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/MockPicoDex1FrontCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.01, 100.0),
        ),
    )
    env_cfg.scene.lazy_sensor_update = False
    env_cfg.sim.render_interval = 1
    env_cfg.decimation = 4


def _body_pose(env, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    body_idx = robot.data.body_names.index(body_name)
    pos = robot.data.body_pos_w.torch[:, body_idx] - env.scene.env_origins
    quat = robot.data.body_quat_w.torch[:, body_idx]
    return pos[0].clone(), quat[0].clone()


def _cube_pos(env, cube_name: str) -> torch.Tensor:
    return (env.scene[cube_name].data.root_pos_w.torch - env.scene.env_origins)[0].clone()


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return a * (1.0 - t) + b * t


def _phase(frame_idx: int, total_frames: int) -> tuple[str, float]:
    u = frame_idx / max(total_frames - 1, 1)
    if u < 0.16:
        return "home_open", _smoothstep(u / 0.16)
    if u < 0.48:
        return "approach_open", _smoothstep((u - 0.16) / 0.32)
    if u < 0.64:
        return "close_claws", _smoothstep((u - 0.48) / 0.16)
    if u < 0.86:
        return "lift_closed", _smoothstep((u - 0.64) / 0.22)
    return "hold_open", _smoothstep((u - 0.86) / 0.14)


def _gripper_targets(trigger: float, device: torch.device) -> torch.Tensor:
    gripper = torch.full((4,), DEX1_OPEN, dtype=torch.float32, device=device)
    right_target = DEX1_OPEN + float(trigger) * (DEX1_CLOSE - DEX1_OPEN)
    gripper[2] = right_target
    gripper[3] = right_target
    return gripper


def _compose_action(
    left_pos: torch.Tensor,
    left_quat: torch.Tensor,
    right_pos: torch.Tensor,
    right_quat: torch.Tensor,
    trigger: float,
    device: torch.device,
) -> torch.Tensor:
    action = torch.zeros((1, 18), dtype=torch.float32, device=device)
    action[0, 0:3] = left_pos.to(device)
    action[0, 3:7] = left_quat.to(device)
    action[0, 7:10] = right_pos.to(device)
    action[0, 10:14] = right_quat.to(device)
    action[0, 14:18] = _gripper_targets(trigger, device)
    return action


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_xyz = quat[1:]
    q_w = quat[0]
    return vec + 2.0 * torch.cross(q_xyz, torch.cross(q_xyz, vec, dim=0) + q_w * vec, dim=0)


def _local_vec(robot, values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=robot.device)


def _finger_contact_points_world(robot, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    finger_1_pos = robot.data.body_pos_w.torch[0, body_ids[1]]
    finger_2_pos = robot.data.body_pos_w.torch[0, body_ids[2]]
    finger_1_quat = robot.data.body_quat_w.torch[0, body_ids[1]]
    finger_2_quat = robot.data.body_quat_w.torch[0, body_ids[2]]
    finger_1_contact = finger_1_pos + _quat_apply_wxyz(finger_1_quat, _local_vec(robot, DEX1_FINGER_1_PAD_CENTER))
    finger_2_contact = finger_2_pos + _quat_apply_wxyz(finger_2_quat, _local_vec(robot, DEX1_FINGER_2_PAD_CENTER))
    return finger_1_contact, finger_2_contact


def _gripper_center_world(robot, body_ids: list[int]) -> torch.Tensor:
    finger_1_contact, finger_2_contact = _finger_contact_points_world(robot, body_ids)
    return 0.5 * (finger_1_contact + finger_2_contact)


def _maybe_attach_cube(env, cube_name: str, trigger: float, attached: bool, body_ids: list[int]) -> tuple[bool, float]:
    robot = env.scene["robot"]
    cube = env.scene[cube_name]
    center_w = _gripper_center_world(robot, body_ids)
    distance = float(torch.linalg.norm(center_w - cube.data.root_pos_w.torch[0]).item())

    if not attached:
        attached = trigger >= args_cli.attach_trigger_threshold and distance <= args_cli.attach_distance_threshold

    if attached:
        root_pose = torch.cat([center_w, cube.data.root_quat_w.torch[0]], dim=0).unsqueeze(0)
        cube.write_root_pose_to_sim(root_pose)
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=root_pose.dtype, device=root_pose.device))
    return attached, distance


def _camera_rgb(camera) -> np.ndarray:
    rgb = camera.data.output["rgb"]
    frame = rgb[0]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    elif hasattr(frame, "numpy"):
        frame = frame.numpy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _put_overlay(frame: np.ndarray, phase_name: str, trigger: float, lift_m: float, attached: bool, frame_idx: int):
    out = frame.copy()
    lines = [
        "IsaacLab3 mock Pico -> G1 Dex1 stack-cube task",
        f"phase: {phase_name}  trigger: {trigger:.2f}  attached: {attached}",
        f"{args_cli.cube} lift: {lift_m:.3f} m",
        f"frame {frame_idx:03d}",
    ]
    y = 30
    for line in lines:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return out


def main() -> int:
    _ensure_output_dirs()

    summary = {
        "task": args_cli.task,
        "output_mp4": args_cli.out_mp4,
        "output_json": args_cli.out_json,
        "output_frames": args_cli.out_frames,
        "output_namespace": "out/isaaclab3",
        "description": "Mock Pico right-controller pose and trigger drive G1 Dex1 through Lab3 Pink IK.",
        "passed": False,
        "records": [],
    }

    env = None
    writer = None
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        _configure_scene(env_cfg)

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()

        camera = env.scene["mock_pico_front_cam"]
        camera.set_world_poses_from_view(
            torch.tensor([[1.05, -0.58, 1.25]], dtype=torch.float32, device=env.device),
            torch.tensor([[0.34, -0.05, 0.98]], dtype=torch.float32, device=env.device),
        )

        left_home_pos, left_home_quat = _body_pose(env, LEFT_WRIST)
        right_home_pos, right_home_quat = _body_pose(env, RIGHT_WRIST)
        cube_initial = _cube_pos(env, args_cli.cube)
        wrist_to_gripper = torch.tensor(RIGHT_WRIST_TO_GRIPPER_CENTER, dtype=torch.float32, device=env.device)

        grasp_center = cube_initial + torch.tensor(
            [args_cli.grasp_center_x_offset, args_cli.grasp_center_y_offset, args_cli.grasp_center_z_offset],
            dtype=torch.float32,
            device=env.device,
        )
        grasp_wrist = grasp_center - wrist_to_gripper + torch.tensor(
            [0.0, 0.0, args_cli.grasp_z_offset], dtype=torch.float32, device=env.device
        )
        pregrasp_wrist = grasp_wrist + torch.tensor(
            [0.0, 0.0, args_cli.pregrasp_height], dtype=torch.float32, device=env.device
        )
        lift_wrist = grasp_wrist + torch.tensor(
            [args_cli.retreat_x, 0.0, args_cli.lift_height], dtype=torch.float32, device=env.device
        )

        for _ in range(args_cli.warmup_steps):
            env.step(_compose_action(left_home_pos, left_home_quat, right_home_pos, right_home_quat, 0.0, env.device))

        writer = cv2.VideoWriter(
            args_cli.out_mp4,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args_cli.fps),
            (args_cli.width, args_cli.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {args_cli.out_mp4}")

        robot = env.scene["robot"]
        right_body_ids = [robot.data.body_names.index(name) for name in RIGHT_DEX1_BODIES]
        right_joint_ids, _ = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        left_joint_ids, _ = robot.find_joints(LEFT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        attached = False
        max_lift = 0.0
        min_attach_distance = float("inf")

        for frame_idx in range(args_cli.frames):
            phase_name, alpha = _phase(frame_idx, args_cli.frames)
            if phase_name == "home_open":
                target = _lerp(right_home_pos, pregrasp_wrist, alpha)
                trigger = 0.0
            elif phase_name == "approach_open":
                target = _lerp(pregrasp_wrist, grasp_wrist, alpha)
                trigger = 0.0
            elif phase_name == "close_claws":
                target = grasp_wrist
                trigger = alpha
            elif phase_name == "lift_closed":
                target = _lerp(grasp_wrist, lift_wrist, alpha)
                trigger = 1.0
            else:
                target = lift_wrist
                trigger = 1.0 - 0.5 * alpha

            action = _compose_action(left_home_pos, left_home_quat, target, right_home_quat, trigger, env.device)
            env.step(action)

            attached, attach_distance = _maybe_attach_cube(env, args_cli.cube, trigger, attached, right_body_ids)
            min_attach_distance = min(min_attach_distance, attach_distance)

            cube_now = _cube_pos(env, args_cli.cube)
            lift_m = float((cube_now[2] - cube_initial[2]).item())
            max_lift = max(max_lift, lift_m)

            frame = _put_overlay(_camera_rgb(camera), phase_name, trigger, lift_m, attached, frame_idx)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if frame_idx % max(args_cli.keyframe_every, 1) == 0 or frame_idx == args_cli.frames - 1:
                frame_path = Path(args_cli.out_frames) / f"frame_{frame_idx:04d}_{phase_name}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                right_wrist_now, _ = _body_pose(env, RIGHT_WRIST)
                gripper_center_env = _gripper_center_world(robot, right_body_ids) - env.scene.env_origins[0]
                summary["records"].append(
                    {
                        "frame": frame_idx,
                        "phase": phase_name,
                        "trigger": float(trigger),
                        "lift_m": lift_m,
                        "attached": bool(attached),
                        "attach_distance_m": float(attach_distance),
                        "target_right_wrist_pos": [float(v) for v in target.detach().cpu().tolist()],
                        "actual_right_wrist_pos": [float(v) for v in right_wrist_now.detach().cpu().tolist()],
                        "right_gripper_center_pos": [float(v) for v in gripper_center_env.detach().cpu().tolist()],
                        "cube_pos": [float(v) for v in cube_now.detach().cpu().tolist()],
                        "right_gripper_joint_pos": [
                            float(v) for v in robot.data.joint_pos.torch[0, right_joint_ids].detach().cpu().tolist()
                        ],
                        "left_gripper_joint_pos": [
                            float(v) for v in robot.data.joint_pos.torch[0, left_joint_ids].detach().cpu().tolist()
                        ],
                    }
                )

        summary.update(
            {
                "passed": bool(attached and max_lift >= args_cli.lift_success_threshold),
                "action_dim": 18,
                "cube": args_cli.cube,
                "initial_cube_pos": [float(v) for v in cube_initial.detach().cpu().tolist()],
                "max_lift_m": float(max_lift),
                "min_attach_distance_m": float(min_attach_distance),
                "used_assisted_grasp": True,
                "dex1_open": float(DEX1_OPEN),
                "dex1_close": float(DEX1_CLOSE),
            }
        )
        return 0 if summary["passed"] else 2
    except Exception:
        summary["error"] = traceback.format_exc()
        print(summary["error"], file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.release()
        Path(args_cli.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if env is not None:
            env.close()
        if args_cli.skip_kit_cleanup:
            os._exit(0 if summary.get("passed") else 1)
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
