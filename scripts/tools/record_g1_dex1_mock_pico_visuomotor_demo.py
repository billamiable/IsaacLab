# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a headless mock-Pico visuomotor demo for the Lab3 G1 Dex1 task.

This script is intentionally narrower than ``record_demos.py``: it does not use
real CloudXR/Pico input.  Instead, it feeds a scripted motion-controller-like
stream into the same 18D task action layout used by the G1 Dex1 teleop task:

    [left_wrist_pose(7), right_wrist_pose(7), dex1_gripper_joints(4)]

The output is an HDF5 demo containing low-dimensional policy observations plus
three RGB camera streams: ego/chest, left wrist, and right wrist.  It is used as
a headless migration gate before testing with a physical Pico device.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-StackCube-Visuomotor-v0"
DEFAULT_OUT_DIR = "/workspace/host/out/isaaclab3/g1_dex1_stack_cube/visuomotor_mock"
DEFAULT_DATASET = f"{DEFAULT_OUT_DIR}/g1_dex1_mock_pico_visuomotor_demo.hdf5"
DEFAULT_SUMMARY = f"{DEFAULT_OUT_DIR}/g1_dex1_mock_pico_visuomotor_demo.json"

parser = argparse.ArgumentParser(description="Record mock Pico G1 Dex1 visuomotor data in Isaac Lab 3.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task name.")
parser.add_argument("--dataset-file", default=DEFAULT_DATASET, help="Output HDF5 dataset file.")
parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="Output JSON summary file.")
parser.add_argument("--frames", type=int, default=128, help="Recorded frames after warmup.")
parser.add_argument("--warmup-steps", type=int, default=24, help="Initial hold steps before recording.")
parser.add_argument("--side", choices=("left", "right"), default="right", help="Mock controller side to move.")
parser.add_argument("--cube", choices=("cube_1", "cube_2", "cube_3"), default="cube_2")
parser.add_argument("--cube-x", type=float, default=0.36)
parser.add_argument("--cube-y", type=float, default=-0.02)
parser.add_argument("--cube-z", type=float, default=0.9535)
parser.add_argument("--pregrasp-height", type=float, default=0.12)
parser.add_argument("--grasp-z-offset", type=float, default=0.012)
parser.add_argument("--lift-height", type=float, default=0.08)
parser.add_argument("--retreat-x", type=float, default=-0.02)
parser.add_argument("--min-wrist-motion", type=float, default=0.03)
parser.add_argument("--gripper-close-tolerance", type=float, default=0.018)
parser.add_argument("--min-gripper-motion", type=float, default=0.01)
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

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_stack_cube_env_cfg import (
    DEX1_CLOSE,
    DEX1_OPEN,
    LEFT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_GRIPPER_JOINTS,
)
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST = "left_wrist_yaw_link"
RIGHT_WRIST = "right_wrist_yaw_link"
LEFT_WRIST_TO_GRIPPER_CENTER = (0.15050695836544037, 6.29723072052002e-05, -8.344650268554688e-06)
RIGHT_WRIST_TO_GRIPPER_CENTER = (0.15050695836544037, -6.29723072052002e-05, -8.344650268554688e-06)
CAMERA_KEYS = ("ego_cam", "left_wrist_cam", "right_wrist_cam")


def _configure_scene(env_cfg) -> None:
    getattr(env_cfg.scene, args_cli.cube).init_state.pos = [args_cli.cube_x, args_cli.cube_y, args_cli.cube_z]
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
    if u < 0.18:
        return "home_open", _smoothstep(u / 0.18)
    if u < 0.46:
        return "approach_open", _smoothstep((u - 0.18) / 0.28)
    if u < 0.66:
        return "close_claws", _smoothstep((u - 0.46) / 0.20)
    if u < 0.90:
        return "lift_closed", _smoothstep((u - 0.66) / 0.24)
    return "hold_closed", _smoothstep((u - 0.90) / 0.10)


def _gripper_targets(left_trigger: float, right_trigger: float, device: torch.device) -> torch.Tensor:
    gripper = torch.full((4,), DEX1_OPEN, dtype=torch.float32, device=device)
    left_target = DEX1_OPEN + float(left_trigger) * (DEX1_CLOSE - DEX1_OPEN)
    right_target = DEX1_OPEN + float(right_trigger) * (DEX1_CLOSE - DEX1_OPEN)
    gripper[0] = left_target
    gripper[1] = left_target
    gripper[2] = right_target
    gripper[3] = right_target
    return gripper


def _compose_action(
    left_pos: torch.Tensor,
    left_quat: torch.Tensor,
    right_pos: torch.Tensor,
    right_quat: torch.Tensor,
    left_trigger: float,
    right_trigger: float,
    device: torch.device,
) -> torch.Tensor:
    action = torch.zeros((1, 18), dtype=torch.float32, device=device)
    action[0, 0:3] = left_pos.to(device)
    action[0, 3:7] = left_quat.to(device)
    action[0, 7:10] = right_pos.to(device)
    action[0, 10:14] = right_quat.to(device)
    action[0, 14:18] = _gripper_targets(left_trigger, right_trigger, device)
    return action


def _normalize_obs_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach()
    if value.shape[0] == 1:
        value = value[0]
    if value.dtype == torch.float32 and value.ndim >= 3:
        value = torch.clamp(value, 0, 255).to(torch.uint8)
    return value.cpu()


def _camera_stats(policy_obs: dict[str, torch.Tensor]) -> dict[str, dict[str, float | list[int]]]:
    stats = {}
    for key in CAMERA_KEYS:
        tensor = policy_obs[key]
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        tensor_f = tensor.detach().float()
        stats[key] = {
            "shape": [int(v) for v in tensor.shape],
            "min": float(tensor_f.min().item()) if tensor.numel() else 0.0,
            "max": float(tensor_f.max().item()) if tensor.numel() else 0.0,
            "mean": float(tensor_f.mean().item()) if tensor.numel() else 0.0,
            "std": float(tensor_f.std().item()) if tensor.numel() > 1 else 0.0,
        }
    return stats


def _obs_cameras_ok(policy_obs: dict[str, torch.Tensor]) -> bool:
    for key in CAMERA_KEYS:
        tensor = policy_obs.get(key)
        if tensor is None or tensor.numel() == 0:
            return False
        if tensor.shape[-1] not in (3, 4):
            return False
    return True


def _write_episode(dataset_file: str, env_name: str, episode: EpisodeData, metadata: dict) -> None:
    handler = HDF5DatasetFileHandler()
    handler.create(dataset_file, env_name=env_name)
    handler.add_env_args(metadata)
    episode.pre_export()
    handler.write_episode(episode, dataset_compression=True)
    handler.flush()
    handler.close()


def main() -> int:
    Path(args_cli.dataset_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.summary).parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "task": args_cli.task,
        "dataset_file": args_cli.dataset_file,
        "summary": args_cli.summary,
        "output_namespace": "out/isaaclab3",
        "description": "Headless mock Pico stream driving G1 Dex1 visuomotor env and recording three RGB cameras.",
        "passed": False,
        "records": [],
    }

    env = None
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        _configure_scene(env_cfg)

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()

        left_home_pos, left_home_quat = _body_pose(env, LEFT_WRIST)
        right_home_pos, right_home_quat = _body_pose(env, RIGHT_WRIST)
        cube_initial = _cube_pos(env, args_cli.cube)
        side_is_right = args_cli.side == "right"
        active_home = right_home_pos if side_is_right else left_home_pos
        active_quat = right_home_quat if side_is_right else left_home_quat
        wrist_to_gripper = torch.tensor(
            RIGHT_WRIST_TO_GRIPPER_CENTER if side_is_right else LEFT_WRIST_TO_GRIPPER_CENTER,
            dtype=torch.float32,
            device=env.device,
        )

        grasp_wrist = cube_initial - wrist_to_gripper + torch.tensor(
            [0.0, 0.0, args_cli.grasp_z_offset], dtype=torch.float32, device=env.device
        )
        pregrasp_wrist = grasp_wrist + torch.tensor(
            [0.0, 0.0, args_cli.pregrasp_height], dtype=torch.float32, device=env.device
        )
        lift_wrist = grasp_wrist + torch.tensor(
            [args_cli.retreat_x, 0.0, args_cli.lift_height], dtype=torch.float32, device=env.device
        )

        hold_action = _compose_action(
            left_home_pos, left_home_quat, right_home_pos, right_home_quat, 0.0, 0.0, env.device
        )
        for _ in range(args_cli.warmup_steps):
            env.step(hold_action)

        robot = env.scene["robot"]
        left_joint_ids, _ = robot.find_joints(LEFT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        right_joint_ids, _ = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        active_joint_ids = right_joint_ids if side_is_right else left_joint_ids
        initial_active_gripper_pos = robot.data.joint_pos.torch[0, active_joint_ids].clone()

        episode = EpisodeData()
        max_wrist_motion = 0.0
        max_trigger = 0.0
        last_policy_obs = None

        for frame_idx in range(args_cli.frames):
            phase_name, alpha = _phase(frame_idx, args_cli.frames)
            if phase_name == "home_open":
                active_target = _lerp(active_home, pregrasp_wrist, alpha)
                trigger = 0.0
            elif phase_name == "approach_open":
                active_target = _lerp(pregrasp_wrist, grasp_wrist, alpha)
                trigger = 0.0
            elif phase_name == "close_claws":
                active_target = grasp_wrist
                trigger = alpha
            elif phase_name == "lift_closed":
                active_target = _lerp(grasp_wrist, lift_wrist, alpha)
                trigger = 1.0
            else:
                active_target = lift_wrist
                trigger = 1.0

            if side_is_right:
                action = _compose_action(
                    left_home_pos, left_home_quat, active_target, active_quat, 0.0, trigger, env.device
                )
                active_now, _ = _body_pose(env, RIGHT_WRIST)
            else:
                action = _compose_action(
                    active_target, active_quat, right_home_pos, right_home_quat, trigger, 0.0, env.device
                )
                active_now, _ = _body_pose(env, LEFT_WRIST)

            obs, _, _, _, _ = env.step(action)
            policy_obs = obs["policy"]
            last_policy_obs = policy_obs
            active_now, _ = _body_pose(env, RIGHT_WRIST if side_is_right else LEFT_WRIST)
            wrist_motion = float(torch.linalg.norm(active_now - active_home).item())
            max_wrist_motion = max(max_wrist_motion, wrist_motion)
            max_trigger = max(max_trigger, float(trigger))

            episode.add("actions", action[0].detach().cpu())
            for obs_key, obs_value in policy_obs.items():
                episode.add(f"obs/{obs_key}", _normalize_obs_tensor(obs_value), clone=False)
            episode.add("mock/side", torch.tensor([1 if side_is_right else 0], dtype=torch.int32))
            episode.add("mock/trigger", torch.tensor([trigger], dtype=torch.float32))
            episode.add("mock/controller_gripper_center", (active_target + wrist_to_gripper).detach().cpu())
            episode.add("mock/controller_wrist_target", active_target.detach().cpu())

            if frame_idx in (0, args_cli.frames // 2, args_cli.frames - 1):
                gripper_pos = robot.data.joint_pos.torch[0, active_joint_ids]
                cube_now = _cube_pos(env, args_cli.cube)
                summary["records"].append(
                    {
                        "frame": int(frame_idx),
                        "phase": phase_name,
                        "trigger": float(trigger),
                        "active_wrist_motion_m": wrist_motion,
                        "active_wrist_pos": [float(v) for v in active_now.detach().cpu().tolist()],
                        "target_wrist_pos": [float(v) for v in active_target.detach().cpu().tolist()],
                        "active_gripper_joint_pos": [float(v) for v in gripper_pos.detach().cpu().tolist()],
                        "cube_pos": [float(v) for v in cube_now.detach().cpu().tolist()],
                    }
                )

        if last_policy_obs is None:
            raise RuntimeError("No observations recorded.")

        active_gripper_pos = robot.data.joint_pos.torch[0, active_joint_ids]
        gripper_close_error = float(torch.max(torch.abs(active_gripper_pos - DEX1_CLOSE)).item())
        gripper_motion_m = float(torch.max(torch.abs(active_gripper_pos - initial_active_gripper_pos)).item())
        cameras_ok = _obs_cameras_ok(last_policy_obs)
        motion_ok = max_wrist_motion >= args_cli.min_wrist_motion
        gripper_motion_ok = gripper_motion_m >= args_cli.min_gripper_motion
        passed = bool(cameras_ok and motion_ok and gripper_motion_ok and max_trigger > 0.95)
        episode.success = passed

        metadata = {
            "env_name": args_cli.task,
            "type": 2,
            "generator": "record_g1_dex1_mock_pico_visuomotor_demo.py",
            "lab_version": "3.0.0-beta2",
            "mock_side": args_cli.side,
            "camera_keys": list(CAMERA_KEYS),
        }
        _write_episode(args_cli.dataset_file, args_cli.task, episode, metadata)

        summary.update(
            {
                "passed": passed,
                "action_dim": 18,
                "frames": int(args_cli.frames),
                "mock_side": args_cli.side,
                "cube": args_cli.cube,
                "initial_cube_pos": [float(v) for v in cube_initial.detach().cpu().tolist()],
                "max_wrist_motion_m": float(max_wrist_motion),
                "min_wrist_motion_m": float(args_cli.min_wrist_motion),
                "max_trigger": float(max_trigger),
                "active_gripper_close_error": gripper_close_error,
                "gripper_close_tolerance": float(args_cli.gripper_close_tolerance),
                "active_gripper_motion_m": gripper_motion_m,
                "min_gripper_motion_m": float(args_cli.min_gripper_motion),
                "gripper_motion_ok": gripper_motion_ok,
                "cameras_ok": cameras_ok,
                "camera_stats": _camera_stats(last_policy_obs),
                "dex1_open": float(DEX1_OPEN),
                "dex1_close": float(DEX1_CLOSE),
            }
        )
        return 0 if passed else 2
    except Exception:
        summary["error"] = traceback.format_exc()
        print(summary["error"], file=sys.stderr)
        return 1
    finally:
        Path(args_cli.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if env is not None:
            env.close()
        if args_cli.skip_kit_cleanup:
            os._exit(0 if summary.get("passed") else 1)
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
