# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a headless mock-Pico visuomotor G1 Dex1 demo to HDF5.

The script simulates Pico/OpenXR motion-controller packets, sends them through
G1 Dex1's existing motion-controller retargeters, steps the normal Isaac Lab
environment, and writes a single episode containing actions, joint state, cube
state, controller raw data, and three robot camera RGB observations.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-StackCube-Visuomotor-v0"
DEFAULT_DATASET = "/workspace/host/out/g1_dex1_mock_pico_visuomotor_demo.hdf5"
DEFAULT_SUMMARY = "/workspace/host/out/g1_dex1_mock_pico_visuomotor_demo.json"

parser = argparse.ArgumentParser(description="Record a mock Pico G1 Dex1 visuomotor HDF5 demo.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--dataset-file", default=DEFAULT_DATASET, help="Output HDF5 dataset path.")
parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="Output JSON summary path.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--frames", type=int, default=192)
parser.add_argument("--warmup-steps", type=int, default=32)
parser.add_argument("--side", choices=("left", "right"), default="right", help="Which mock Pico controller grasps.")
parser.add_argument("--cube", choices=("cube_1", "cube_2", "cube_3"), default="cube_1", help="Cube to grasp.")
parser.add_argument("--pregrasp-height", type=float, default=0.12)
parser.add_argument("--lift-height", type=float, default=0.12)
parser.add_argument("--retreat-x", type=float, default=-0.04)
parser.add_argument("--grasp-center-x-offset", type=float, default=0.0)
parser.add_argument("--grasp-center-y-offset", type=float, default=0.0)
parser.add_argument("--grasp-center-z-offset", type=float, default=0.0)
parser.add_argument("--lift-success-threshold", type=float, default=0.025)
parser.add_argument("--center-success-threshold", type=float, default=0.045)
parser.add_argument("--close-error-threshold", type=float, default=0.006)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Import Pinocchio before AppLauncher, matching Pink IK teleop entrypoints.",
)
parser.add_argument(
    "--skip-kit-cleanup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit directly after writing outputs to avoid Kit shutdown hangs in headless runs.",
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

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.retargeters.humanoid.unitree.dex1.g1_dex1_upper_body_motion_ctrl_retargeter import (
    G1Dex1UpperBodyMotionControllerRetargeter,
)
from isaaclab.devices.openxr.retargeters.manipulator.gripper_trigger_or_pinch_retargeter import (
    GripperTriggerOrPinchRetargeter,
)
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    LEFT_DEX1_CLOSE,
    LEFT_DEX1_GRIPPER_JOINTS,
    LEFT_DEX1_OPEN,
    RIGHT_DEX1_CLOSE,
    RIGHT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_OPEN,
)
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
LEFT_DEX1_BODIES = ["left_dex1_base_link", "left_dex1_finger_link_1", "left_dex1_finger_link_2"]
RIGHT_DEX1_BODIES = ["right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2"]
DEX1_FINGER_1_PAD_CENTER = (0.1082509, -0.0315503, 0.0)
DEX1_FINGER_2_PAD_CENTER = (0.1097630, 0.0313668, 0.0)
CAMERA_KEYS = ("ego_cam", "left_wrist_cam", "right_wrist_cam")


def to_list(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.detach().cpu().tolist()]


def max_abs_error(values: list[float], target: float) -> float:
    return max(abs(value - target) for value in values)


def quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(v.shape[:-1] + (1,), dtype=v.dtype, device=v.device)
    v_quat = torch.cat((zeros, v), dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(q, v_quat), quat_conjugate_wxyz(q))[..., 1:]


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


def cube_pos_env(env, name: str) -> torch.Tensor:
    cube = env.unwrapped.scene[name]
    return cube.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]


def body_pose_env_frame(env, robot, body_id: int) -> tuple[torch.Tensor, list[float]]:
    pos = robot.data.body_pos_w[0, body_id] - env.unwrapped.scene.env_origins[0]
    quat = robot.data.body_quat_w[0, body_id]
    return pos.detach().clone(), [float(v) for v in quat.detach().cpu().tolist()]


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


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def interpolate_keyframes(frame_index: int, num_frames: int, keyframes):
    if num_frames <= 1:
        return keyframes[0][1], keyframes[0][2], keyframes[0][3], keyframes[0][0]
    t = frame_index / float(num_frames - 1)
    for next_index in range(1, len(keyframes)):
        t0, pos0, trigger0, phase0 = keyframes[next_index - 1]
        t1, pos1, trigger1, phase1 = keyframes[next_index]
        if t <= t1 or next_index == len(keyframes) - 1:
            alpha = _smoothstep((t - t0) / max(t1 - t0, 1.0e-6))
            pos = [float(v0 + (v1 - v0) * alpha) for v0, v1 in zip(pos0, pos1)]
            trigger = float(trigger0 + (trigger1 - trigger0) * alpha)
            phase = phase1 if alpha > 0.5 else phase0
            return pos, trigger, phase, t
    return keyframes[-1][1], keyframes[-1][2], keyframes[-1][3], t


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


def measure(env, robot, wrist_id: int, joint_ids: list[int], body_ids: list[int], cube_name: str) -> dict:
    cube_env = cube_pos_env(env, cube_name)
    wrist_env = robot.data.body_pos_w[0, wrist_id] - env.unwrapped.scene.env_origins[0]
    center_env = gripper_center_env(env, robot, body_ids)
    joint_pos = robot.data.joint_pos[0, joint_ids]
    finger_gap = gripper_gap_world(robot, body_ids)
    return {
        "wrist_pos_env": to_list(wrist_env),
        "gripper_center_env": to_list(center_env),
        "cube_pos_env": to_list(cube_env),
        "cube_height_env_m": float(cube_env[2].item()),
        "gripper_joint_pos": to_list(joint_pos),
        "finger_body_separation_m": float(finger_gap.item()),
        "center_to_cube_m": float(torch.linalg.norm(center_env - cube_env).item()),
    }


def add_tensor(episode: EpisodeData, key: str, value: torch.Tensor) -> None:
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    episode.add(key, value.detach().clone())


def add_obs(episode: EpisodeData, policy_obs: dict, keys: list[str]) -> None:
    for key in keys:
        if key in policy_obs:
            add_tensor(episode, f"obs/{key}", policy_obs[key])


def tensor_image_stats(value: torch.Tensor) -> dict:
    if value.ndim == 4:
        value = value[0]
    image = value.detach().cpu().numpy()[..., :3]
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return {
        "shape": list(image.shape),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "range": int(image.max()) - int(image.min()),
    }


def camera_content_ok(stats: dict[str, dict]) -> dict[str, bool]:
    return {name: bool(item["std"] > 1.0 or item["range"] > 8.0) for name, item in stats.items()}


def write_dataset(dataset_file: Path, episode: EpisodeData, env_name: str, metadata: dict) -> None:
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    handler = HDF5DatasetFileHandler()
    handler.create(str(dataset_file), env_name=env_name)
    handler.add_env_args(metadata)
    episode.pre_export()
    handler.write_episode(episode)
    handler.flush()
    handler.close()


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("Mock Pico recorder currently expects --num-envs 1.")
    dataset_file = Path(args_cli.dataset_file)
    summary_path = Path(args_cli.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.scene.lazy_sensor_update = False
    env_cfg.sim.render_interval = 1

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    try:
        robot = env.unwrapped.scene["robot"]
        wrist_id, wrist_names, body_ids, body_names, joint_ids, joint_names, close_target, open_target = side_config(robot)
        default_wrist, active_quat = body_pose_env_frame(env, robot, wrist_id)
        default_center = gripper_center_env(env, robot, body_ids)
        inactive_default_position, inactive_controller_quat = inactive_pose(env, robot)
        wrist_retargeter, left_gripper, right_gripper, configured_retargeters = make_retargeters_from_env_cfg(env_cfg)

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

        episode = EpisodeData()
        episode.success = True
        frame_records = []
        last_policy_obs = None
        obs_keys = [
            "actions",
            "robot_joint_pos",
            "right_eef_pos",
            "right_eef_quat",
            "left_eef_pos",
            "left_eef_quat",
            "gripper_joint_pos",
            "cube_1_pos",
            "cube_2_pos",
            "cube_3_pos",
            "cube_1_rot",
            "cube_2_rot",
            "cube_3_rot",
            *CAMERA_KEYS,
        ]

        for frame in range(args_cli.frames):
            active_position, active_trigger, phase, progress = interpolate_keyframes(frame, args_cli.frames, keyframes)
            active_controller_position = [float(active_position[i] + center_to_wrist[i].item()) for i in range(3)]
            raw = make_controller_data(
                active_controller_position,
                active_quat,
                active_trigger,
                inactive_default_position,
                inactive_controller_quat,
            )
            action = retarget_action(wrist_retargeter, left_gripper, right_gripper, raw)
            obs, _, _, _, _ = env.step(action)
            policy_obs = dict(obs["policy"])
            last_policy_obs = policy_obs

            add_tensor(episode, "actions", action)
            add_obs(episode, policy_obs, obs_keys)
            episode.add("controller/left_raw", torch.tensor(raw[DeviceBase.TrackingTarget.CONTROLLER_LEFT], device=robot.device))
            episode.add("controller/right_raw", torch.tensor(raw[DeviceBase.TrackingTarget.CONTROLLER_RIGHT], device=robot.device))
            episode.add("controller/active_trigger", torch.tensor([active_trigger], dtype=torch.float32, device=robot.device))
            episode.add(
                "controller/active_position",
                torch.tensor(active_controller_position, dtype=torch.float32, device=robot.device),
            )
            episode.add("phase_index", torch.tensor([frame], dtype=torch.int32, device=robot.device))

            measured = measure(env, robot, wrist_id, joint_ids, body_ids, args_cli.cube)
            cube_lift = measured["cube_height_env_m"] - float(cube_initial[2].item())
            frame_records.append(
                {
                    "frame": frame,
                    "progress": float(progress),
                    "phase": phase,
                    "mock_active_trigger": float(active_trigger),
                    "mock_active_controller_position": active_controller_position,
                    "retargeted_action_shape": list(action.shape),
                    "cube_lift_m": float(cube_lift),
                    **measured,
                }
            )

        max_cube_lift = max(record["cube_lift_m"] for record in frame_records)
        final_cube_lift = frame_records[-1]["cube_lift_m"]
        min_center_to_cube = min(record["center_to_cube_m"] for record in frame_records)
        close_records = [record for record in frame_records if record["phase"] in ("close_on_cube", "lift_closed", "retreat_closed")]
        close_error = min(max_abs_error(record["gripper_joint_pos"], close_target) for record in close_records)
        open_error = max_abs_error(frame_records[-1]["gripper_joint_pos"], open_target)
        action_shape_ok = tuple(frame_records[0]["retargeted_action_shape"]) == tuple(env.action_space.shape)
        passed = bool(
            action_shape_ok
            and min_center_to_cube <= args_cli.center_success_threshold
            and max_cube_lift >= args_cli.lift_success_threshold
        )
        episode.success = passed

        camera_stats = {name: tensor_image_stats(last_policy_obs[name]) for name in CAMERA_KEYS if name in last_policy_obs}
        camera_ok = camera_content_ok(camera_stats)
        metadata = {
            "task": args_cli.task,
            "mock_pico": True,
            "teleop_device_semantics": "motion_controllers",
            "image_obs_keys": list(CAMERA_KEYS),
            "dataset_kind": "mock_pico_visuomotor_smoke",
        }
        write_dataset(dataset_file, episode, args_cli.task, metadata)
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "dataset_file": str(dataset_file),
            "task": args_cli.task,
            "side": args_cli.side,
            "cube": args_cli.cube,
            "frames": int(args_cli.frames),
            "camera_keys": list(CAMERA_KEYS),
            "camera_stats": camera_stats,
            "camera_content_ok": camera_ok,
            "configured_motion_controller_retargeters": configured_retargeters,
            "action_space": str(env.action_space),
            "action_shape_ok": bool(action_shape_ok),
            "wrist_names": wrist_names,
            "gripper_body_names": body_names,
            "gripper_joint_names": joint_names,
            "initial": initial,
            "default_wrist_env": to_list(default_wrist),
            "default_gripper_center_env": to_list(default_center),
            "center_to_wrist_env": to_list(center_to_wrist),
            "cube_initial_env": to_list(cube_initial),
            "requested_gripper_center_env": to_list(requested_center),
            "computed_grasp_wrist_env": to_list(grasp_wrist),
            "keyframes": [
                {"t": t, "position": position, "trigger": trigger, "phase": phase}
                for t, position, trigger, phase in keyframes
            ],
            "max_cube_lift_m": float(max_cube_lift),
            "final_cube_lift_m": float(final_cube_lift),
            "min_gripper_center_to_cube_m": float(min_center_to_cube),
            "best_close_max_abs_error_m": float(close_error),
            "final_open_max_abs_error_m": float(open_error),
            "lift_success_threshold_m": float(args_cli.lift_success_threshold),
            "center_success_threshold_m": float(args_cli.center_success_threshold),
            "close_error_threshold_m": float(args_cli.close_error_threshold),
            "midpoint": frame_records[len(frame_records) // 2],
            "final": frame_records[-1],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote mock Pico visuomotor HDF5: {dataset_file}")
        print(f"[INFO] Wrote summary: {summary_path}")
        print(f"[INFO] Overall result: {'PASS' if passed else 'FAIL'}")
        print(
            "[INFO] metrics: "
            f"max_cube_lift={max_cube_lift:.6f} m, "
            f"min_center_to_cube={min_center_to_cube:.6f} m, "
            f"best_close_error={close_error:.6f} m"
        )
    finally:
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
