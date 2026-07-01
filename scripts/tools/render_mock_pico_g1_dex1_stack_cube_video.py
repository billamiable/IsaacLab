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

By default the cube must be lifted through contact physics.  An assisted mode is
available for debugging the teleop/control path, but the physical mode never
writes the cube root pose during grasp.
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
parser.add_argument("--table-z-offset", type=float, default=0.0)
parser.add_argument("--object-usd", default=None)
parser.add_argument("--object-scale", type=float, default=1.0)
parser.add_argument("--object-mass", type=float, default=None, help="Optional spawned object mass override in kg.")
parser.add_argument("--object-roll-deg", type=float, default=0.0)
parser.add_argument("--object-pitch-deg", type=float, default=0.0)
parser.add_argument("--object-yaw-deg", type=float, default=0.0)
parser.add_argument("--block-scale-x", type=float, default=1.0)
parser.add_argument("--block-scale-y", type=float, default=1.0)
parser.add_argument("--block-scale-z", type=float, default=1.0)
parser.add_argument("--use-teleop-right-target-quat", action="store_true")
parser.add_argument("--right-target-roll-deg", type=float, default=-135.0)
parser.add_argument("--right-target-pitch-deg", type=float, default=0.0)
parser.add_argument("--right-target-yaw-deg", type=float, default=90.0)
parser.add_argument("--pregrasp-height", type=float, default=0.12)
parser.add_argument("--grasp-z-offset", type=float, default=0.012)
parser.add_argument("--lift-height", type=float, default=0.08)
parser.add_argument("--retreat-x", type=float, default=-0.02)
parser.add_argument("--grasp-center-x-offset", type=float, default=0.0)
parser.add_argument("--grasp-center-y-offset", type=float, default=0.0)
parser.add_argument("--grasp-center-z-offset", type=float, default=0.0)
parser.add_argument(
    "--grasp-depth-x-offset",
    type=float,
    default=0.0,
    help="Extra world-X offset for front-claw insertion depth. Negative is shallower for the right-hand setup.",
)
parser.add_argument("--grasp-mode", choices=("physical", "assisted"), default="physical")
parser.add_argument("--attach-trigger-threshold", type=float, default=0.65)
parser.add_argument("--attach-distance-threshold", type=float, default=0.085)
parser.add_argument("--lift-success-threshold", type=float, default=0.04)
parser.add_argument("--physical-hold-frames", type=int, default=18)
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
from scipy.spatial.transform import Rotation

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
DEX1_NOMINAL_PAD_DEPTH_M = 0.5 * (DEX1_FINGER_1_PAD_CENTER[0] + DEX1_FINGER_2_PAD_CENTER[0])
DEX1_NOMINAL_PAD_SEPARATION_M = DEX1_FINGER_2_PAD_CENTER[1] - DEX1_FINGER_1_PAD_CENTER[1]


def _euler_xyz_quat_xyzw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Return an Isaac Lab xyzw quaternion matching isaacteleop's XYZ Euler convention."""

    quat = Rotation.from_euler("XYZ", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat()
    return tuple(float(v) for v in quat)


def _ensure_output_dirs() -> None:
    Path(args_cli.out_mp4).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out_frames).mkdir(parents=True, exist_ok=True)


def _configure_scene(env_cfg) -> None:
    if abs(args_cli.table_z_offset) > 1.0e-9:
        table_pos = list(env_cfg.scene.table.init_state.pos)
        table_pos[2] += args_cli.table_z_offset
        env_cfg.scene.table.init_state.pos = table_pos

    object_cfg = getattr(env_cfg.scene, args_cli.cube)
    object_cfg.init_state.pos = [args_cli.cube_x, args_cli.cube_y, args_cli.cube_z]
    object_cfg.init_state.rot = _euler_xyz_quat_xyzw(
        args_cli.object_roll_deg, args_cli.object_pitch_deg, args_cli.object_yaw_deg
    )
    if args_cli.object_usd is not None:
        object_cfg.spawn.usd_path = args_cli.object_usd
        object_cfg.spawn.scale = (args_cli.object_scale, args_cli.object_scale, args_cli.object_scale)
    if args_cli.object_mass is not None:
        object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=args_cli.object_mass)
    elif (args_cli.block_scale_x, args_cli.block_scale_y, args_cli.block_scale_z) != (1.0, 1.0, 1.0):
        object_cfg.spawn.scale = (args_cli.block_scale_x, args_cli.block_scale_y, args_cli.block_scale_z)
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
    if u < 0.14:
        return "home_open", _smoothstep(u / 0.14)
    if u < 0.42:
        return "approach_open", _smoothstep((u - 0.14) / 0.28)
    if u < 0.62:
        return "close_claws", _smoothstep((u - 0.42) / 0.20)
    if u < 0.88:
        return "lift_closed", _smoothstep((u - 0.62) / 0.26)
    return "hold_closed", _smoothstep((u - 0.88) / 0.12)


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


def _quat_apply_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_xyz = quat[:3]
    q_w = quat[3]
    return vec + 2.0 * torch.cross(q_xyz, torch.cross(q_xyz, vec, dim=0) + q_w * vec, dim=0)


def _local_vec(robot, values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=robot.device)


def _finger_contact_points_world(robot, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    finger_1_pos = robot.data.body_pos_w.torch[0, body_ids[1]]
    finger_2_pos = robot.data.body_pos_w.torch[0, body_ids[2]]
    finger_1_quat = robot.data.body_quat_w.torch[0, body_ids[1]]
    finger_2_quat = robot.data.body_quat_w.torch[0, body_ids[2]]
    finger_1_contact = finger_1_pos + _quat_apply_xyzw(finger_1_quat, _local_vec(robot, DEX1_FINGER_1_PAD_CENTER))
    finger_2_contact = finger_2_pos + _quat_apply_xyzw(finger_2_quat, _local_vec(robot, DEX1_FINGER_2_PAD_CENTER))
    return finger_1_contact, finger_2_contact


def _gripper_center_world(robot, body_ids: list[int]) -> torch.Tensor:
    finger_1_contact, finger_2_contact = _finger_contact_points_world(robot, body_ids)
    return 0.5 * (finger_1_contact + finger_2_contact)


def _maybe_attach_cube(env, cube_name: str, trigger: float, attached: bool, body_ids: list[int]) -> tuple[bool, float]:
    robot = env.scene["robot"]
    cube = env.scene[cube_name]
    center_w = _gripper_center_world(robot, body_ids)
    distance = float(torch.linalg.norm(center_w - cube.data.root_pos_w.torch[0]).item())

    if args_cli.grasp_mode != "assisted":
        return False, distance

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
        f"mode: {args_cli.grasp_mode}  phase: {phase_name}",
        f"trigger: {trigger:.2f}  assisted_attached: {attached}",
        f"{args_cli.cube} lift: {lift_m:.3f} m  frame {frame_idx:03d}",
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
            torch.tensor([[1.65, -1.20, 1.45]], dtype=torch.float32, device=env.device),
            torch.tensor([[0.28, -0.03, 0.92]], dtype=torch.float32, device=env.device),
        )

        left_home_pos, left_home_quat = _body_pose(env, LEFT_WRIST)
        right_home_pos, right_home_quat = _body_pose(env, RIGHT_WRIST)
        if args_cli.use_teleop_right_target_quat:
            right_home_quat = torch.tensor(
                _euler_xyz_quat_xyzw(
                    args_cli.right_target_roll_deg, args_cli.right_target_pitch_deg, args_cli.right_target_yaw_deg
                ),
                dtype=torch.float32,
                device=env.device,
            )
        cube_initial = _cube_pos(env, args_cli.cube)
        wrist_to_gripper = torch.tensor(RIGHT_WRIST_TO_GRIPPER_CENTER, dtype=torch.float32, device=env.device)

        grasp_center = cube_initial + torch.tensor(
            [
                args_cli.grasp_center_x_offset + args_cli.grasp_depth_x_offset,
                args_cli.grasp_center_y_offset,
                args_cli.grasp_center_z_offset,
            ],
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

        # Re-plan from the settled object pose.  Substituted USD assets may drop to the table during warmup.
        left_home_pos, left_home_quat = _body_pose(env, LEFT_WRIST)
        right_home_pos, right_home_quat = _body_pose(env, RIGHT_WRIST)
        if args_cli.use_teleop_right_target_quat:
            right_home_quat = torch.tensor(
                _euler_xyz_quat_xyzw(
                    args_cli.right_target_roll_deg, args_cli.right_target_pitch_deg, args_cli.right_target_yaw_deg
                ),
                dtype=torch.float32,
                device=env.device,
            )
        cube_initial = _cube_pos(env, args_cli.cube)
        grasp_center = cube_initial + torch.tensor(
            [
                args_cli.grasp_center_x_offset + args_cli.grasp_depth_x_offset,
                args_cli.grasp_center_y_offset,
                args_cli.grasp_center_z_offset,
            ],
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
        consecutive_lift_frames = 0
        max_consecutive_lift_frames = 0

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
                trigger = 1.0 if args_cli.grasp_mode == "physical" else 1.0 - 0.5 * alpha

            action = _compose_action(left_home_pos, left_home_quat, target, right_home_quat, trigger, env.device)
            env.step(action)

            attached, attach_distance = _maybe_attach_cube(env, args_cli.cube, trigger, attached, right_body_ids)
            min_attach_distance = min(min_attach_distance, attach_distance)

            cube_now = _cube_pos(env, args_cli.cube)
            lift_m = float((cube_now[2] - cube_initial[2]).item())
            max_lift = max(max_lift, lift_m)
            if lift_m >= args_cli.lift_success_threshold:
                consecutive_lift_frames += 1
            else:
                consecutive_lift_frames = 0
            max_consecutive_lift_frames = max(max_consecutive_lift_frames, consecutive_lift_frames)

            frame = _put_overlay(_camera_rgb(camera), phase_name, trigger, lift_m, attached, frame_idx)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if frame_idx % max(args_cli.keyframe_every, 1) == 0 or frame_idx == args_cli.frames - 1:
                frame_path = Path(args_cli.out_frames) / f"frame_{frame_idx:04d}_{phase_name}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                right_wrist_now, _ = _body_pose(env, RIGHT_WRIST)
                finger_1_contact_w, finger_2_contact_w = _finger_contact_points_world(robot, right_body_ids)
                finger_1_contact_env = finger_1_contact_w - env.scene.env_origins[0]
                finger_2_contact_env = finger_2_contact_w - env.scene.env_origins[0]
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
                        "right_finger_1_pad_pos": [float(v) for v in finger_1_contact_env.detach().cpu().tolist()],
                        "right_finger_2_pad_pos": [float(v) for v in finger_2_contact_env.detach().cpu().tolist()],
                        "right_finger_pad_separation_m": float(
                            torch.linalg.norm(finger_1_contact_w - finger_2_contact_w).item()
                        ),
                        "cube_pos": [float(v) for v in cube_now.detach().cpu().tolist()],
                        "right_gripper_joint_pos": [
                            float(v) for v in robot.data.joint_pos.torch[0, right_joint_ids].detach().cpu().tolist()
                        ],
                        "left_gripper_joint_pos": [
                            float(v) for v in robot.data.joint_pos.torch[0, left_joint_ids].detach().cpu().tolist()
                        ],
                    }
                )

        physical_passed = max_consecutive_lift_frames >= args_cli.physical_hold_frames
        assisted_passed = attached and max_lift >= args_cli.lift_success_threshold
        summary.update(
            {
                "passed": bool(physical_passed if args_cli.grasp_mode == "physical" else assisted_passed),
                "action_dim": 18,
                "cube": args_cli.cube,
                "table_z_offset": float(args_cli.table_z_offset),
                "object_usd": args_cli.object_usd,
                "object_scale": float(args_cli.object_scale),
                "object_mass": None if args_cli.object_mass is None else float(args_cli.object_mass),
                "block_scale_xyz": [
                    float(args_cli.block_scale_x),
                    float(args_cli.block_scale_y),
                    float(args_cli.block_scale_z),
                ],
                "object_rpy_deg": [
                    float(args_cli.object_roll_deg),
                    float(args_cli.object_pitch_deg),
                    float(args_cli.object_yaw_deg),
                ],
                "grasp_center_offset": [
                    float(args_cli.grasp_center_x_offset),
                    float(args_cli.grasp_center_y_offset),
                    float(args_cli.grasp_center_z_offset),
                ],
                "grasp_depth_x_offset": float(args_cli.grasp_depth_x_offset),
                "use_teleop_right_target_quat": bool(args_cli.use_teleop_right_target_quat),
                "right_target_rpy_deg": [
                    float(args_cli.right_target_roll_deg),
                    float(args_cli.right_target_pitch_deg),
                    float(args_cli.right_target_yaw_deg),
                ],
                "grasp_mode": args_cli.grasp_mode,
                "initial_cube_pos": [float(v) for v in cube_initial.detach().cpu().tolist()],
                "max_lift_m": float(max_lift),
                "lift_success_threshold_m": float(args_cli.lift_success_threshold),
                "max_consecutive_lift_frames": int(max_consecutive_lift_frames),
                "required_physical_hold_frames": int(args_cli.physical_hold_frames),
                "min_attach_distance_m": float(min_attach_distance),
                "used_assisted_grasp": bool(args_cli.grasp_mode == "assisted"),
                "ever_assisted_attached": bool(attached),
                "dex1_open": float(DEX1_OPEN),
                "dex1_close": float(DEX1_CLOSE),
                "dex1_nominal_pad_depth_m": float(DEX1_NOMINAL_PAD_DEPTH_M),
                "dex1_nominal_pad_separation_m": float(DEX1_NOMINAL_PAD_SEPARATION_M),
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
