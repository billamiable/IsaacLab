# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test mock Pico motion-controller retargeting for fixed-base G1 Dex1."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-IK-Scene-v0"
DEFAULT_OUT_JSON = "/workspace/host/out/g1_dex1_motion_controller_retargeter_smoke.json"
DEFAULT_OUT_MD = "/workspace/host/out/g1_dex1_motion_controller_retargeter_smoke.md"


parser = argparse.ArgumentParser(description="Smoke-test G1 Dex1 mock Pico motion-controller retargeter path.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for JSON report.")
parser.add_argument("--out-md", default=DEFAULT_OUT_MD, help="Path for Markdown report.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps-per-command", type=int, default=60, help="Environment steps per command.")
parser.add_argument("--wrist-motion-threshold", type=float, default=0.015, help="Required right wrist movement in m.")
parser.add_argument("--gripper-error-threshold", type=float, default=0.006, help="Allowed gripper joint error in m.")
parser.add_argument("--right-offset-x", type=float, default=0.10)
parser.add_argument("--right-offset-y", type=float, default=-0.02)
parser.add_argument("--right-offset-z", type=float, default=0.07)
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
    help="Exit directly after writing reports to avoid Kit shutdown hangs in headless CI.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

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
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    RIGHT_DEX1_CLOSE,
    RIGHT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_OPEN,
)
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
RIGHT_DEX1_BODIES = ["right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2"]


def body_pose_env_frame(env, robot, body_id: int) -> tuple[list[float], list[float]]:
    pos = robot.data.body_pos_w[0, body_id] - env.unwrapped.scene.env_origins[0]
    quat = robot.data.body_quat_w[0, body_id]
    return [float(v) for v in pos.detach().cpu().tolist()], [float(v) for v in quat.detach().cpu().tolist()]


def make_controller_data(position: list[float], quat: list[float], trigger: float) -> dict:
    pose = np.array([*position, *quat], dtype=np.float32)
    inputs = np.zeros(7, dtype=np.float32)
    inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value] = float(trigger)
    return {DeviceBase.TrackingTarget.CONTROLLER_RIGHT: np.stack([pose, inputs])}


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
    action = torch.cat([wrist_retargeter.retarget(raw_data), gripper_retargeter.retarget(raw_data)], dim=-1)
    return action.unsqueeze(0)


def step_command(env, action: torch.Tensor, steps: int) -> None:
    for _ in range(steps):
        env.step(action)


def measure(robot, right_body_id: int, gripper_joint_ids: list[int], gripper_body_ids: list[int]) -> dict:
    gripper_body_pos = robot.data.body_pos_w[0, gripper_body_ids]
    separation = torch.linalg.norm(gripper_body_pos[1] - gripper_body_pos[2]).item()
    return {
        "right_wrist_pos_w": [
            float(v) for v in robot.data.body_pos_w[0, right_body_id].detach().cpu().tolist()
        ],
        "gripper_joint_pos": [
            float(v) for v in robot.data.joint_pos[0, gripper_joint_ids].detach().cpu().tolist()
        ],
        "finger_body_separation_m": float(separation),
    }


def write_reports(report: dict) -> None:
    json_path = Path(args_cli.out_json)
    md_path = Path(args_cli.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# G1 Dex1 Motion Controller Retargeter Smoke Test",
                "",
                f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
                f"- Task: `{report['task']}`",
                f"- Action space: `{report['action_space']}`",
                f"- Retargeted action shape: `{report['retargeted_action_shape']}`",
                f"- Action terms: `{report['action_terms']}`",
                f"- Action term dims: `{report['action_term_dims']}`",
                f"- Right wrist displacement: `{report['right_wrist_displacement_m']:.6f} m`",
                f"- Gripper close max error: `{report['close_gripper_max_abs_error']:.6f} m`",
                f"- Gripper reopen max error: `{report['reopen_gripper_max_abs_error']:.6f} m`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    try:
        robot = env.unwrapped.scene["robot"]
        left_body_ids, left_body_names = robot.find_bodies([LEFT_WRIST_BODY], preserve_order=True)
        right_body_ids, right_body_names = robot.find_bodies([RIGHT_WRIST_BODY], preserve_order=True)
        gripper_joint_ids, gripper_joint_names = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        gripper_body_ids, gripper_body_names = robot.find_bodies(RIGHT_DEX1_BODIES, preserve_order=True)
        if len(left_body_ids) != 1 or len(right_body_ids) != 1:
            raise RuntimeError(f"Could not resolve wrist bodies: {left_body_names=} {right_body_names=}")

        left_pos, left_quat = body_pose_env_frame(env, robot, left_body_ids[0])
        right_pos, right_quat = body_pose_env_frame(env, robot, right_body_ids[0])
        left_pose = [*left_pos, *left_quat]
        right_pose = [*right_pos, *right_quat]
        wrist_retargeter, gripper_retargeter, configured_retargeters = make_retargeters_from_env_cfg(env_cfg)

        open_raw = make_controller_data([0.0, 0.0, 0.0], right_quat, trigger=0.0)
        open_action = retarget_action(wrist_retargeter, gripper_retargeter, open_raw)
        step_command(env, open_action, args_cli.steps_per_command)
        initial = measure(robot, right_body_ids[0], gripper_joint_ids, gripper_body_ids)

        offset = [args_cli.right_offset_x, args_cli.right_offset_y, args_cli.right_offset_z]
        close_raw = make_controller_data(offset, right_quat, trigger=1.0)
        close_action = retarget_action(wrist_retargeter, gripper_retargeter, close_raw)
        step_command(env, close_action, args_cli.steps_per_command)
        moved_closed = measure(robot, right_body_ids[0], gripper_joint_ids, gripper_body_ids)

        reopen_raw = make_controller_data([0.0, 0.0, 0.0], right_quat, trigger=0.0)
        reopen_action = retarget_action(wrist_retargeter, gripper_retargeter, reopen_raw)
        step_command(env, reopen_action, args_cli.steps_per_command)
        reopened = measure(robot, right_body_ids[0], gripper_joint_ids, gripper_body_ids)

        initial_wrist = torch.tensor(initial["right_wrist_pos_w"])
        moved_wrist = torch.tensor(moved_closed["right_wrist_pos_w"])
        right_wrist_displacement = torch.linalg.norm(moved_wrist - initial_wrist).item()

        close_target = [RIGHT_DEX1_CLOSE, RIGHT_DEX1_CLOSE]
        open_target = [RIGHT_DEX1_OPEN, RIGHT_DEX1_OPEN]
        close_errors = [
            measured - target for measured, target in zip(moved_closed["gripper_joint_pos"], close_target)
        ]
        reopen_errors = [measured - target for measured, target in zip(reopened["gripper_joint_pos"], open_target)]
        close_max_abs_error = max(abs(value) for value in close_errors)
        reopen_max_abs_error = max(abs(value) for value in reopen_errors)

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "passed": (
                tuple(open_action.shape) == env.action_space.shape
                and right_wrist_displacement >= args_cli.wrist_motion_threshold
                and close_max_abs_error <= args_cli.gripper_error_threshold
                and reopen_max_abs_error <= args_cli.gripper_error_threshold
            ),
            "task": args_cli.task,
            "device": env.unwrapped.device,
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "retargeted_action_shape": list(open_action.shape),
            "open_action": [float(v) for v in open_action[0].detach().cpu().tolist()],
            "close_action": [float(v) for v in close_action[0].detach().cpu().tolist()],
            "reopen_action": [float(v) for v in reopen_action[0].detach().cpu().tolist()],
            "action_terms": env.unwrapped.action_manager.active_terms,
            "action_term_dims": env.unwrapped.action_manager.action_term_dim,
            "configured_motion_controller_retargeters": configured_retargeters,
            "left_body_names": left_body_names,
            "right_body_names": right_body_names,
            "gripper_joint_names": gripper_joint_names,
            "gripper_body_names": gripper_body_names,
            "mock_right_controller_offset": offset,
            "left_wrist_hold_pose": left_pose,
            "right_wrist_default_pose": right_pose,
            "initial": initial,
            "moved_closed": moved_closed,
            "reopened": reopened,
            "right_wrist_displacement_m": float(right_wrist_displacement),
            "close_gripper_errors": close_errors,
            "close_gripper_max_abs_error": close_max_abs_error,
            "reopen_gripper_errors": reopen_errors,
            "reopen_gripper_max_abs_error": reopen_max_abs_error,
        }
        write_reports(report)
        print(f"[INFO] Wrote JSON report: {args_cli.out_json}")
        print(f"[INFO] Wrote Markdown report: {args_cli.out_md}")
        print(f"[INFO] Overall result: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"[INFO] retargeted action shape: {tuple(open_action.shape)}")
        print(f"[INFO] right wrist displacement: {right_wrist_displacement:.6f} m")
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
