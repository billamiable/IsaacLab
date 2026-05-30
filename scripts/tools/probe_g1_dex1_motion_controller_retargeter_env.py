# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test bilateral mock Pico motion-controller retargeting for fixed-base G1 Dex1."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-IK-Scene-v0"
DEFAULT_OUT_JSON = "/workspace/host/out/g1_dex1_bimanual_motion_controller_retargeter_smoke.json"
DEFAULT_OUT_MD = "/workspace/host/out/g1_dex1_bimanual_motion_controller_retargeter_smoke.md"


parser = argparse.ArgumentParser(description="Smoke-test bilateral G1 Dex1 mock Pico motion-controller retargeter path.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for JSON report.")
parser.add_argument("--out-md", default=DEFAULT_OUT_MD, help="Path for Markdown report.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps-per-command", type=int, default=60, help="Environment steps per command.")
parser.add_argument("--wrist-motion-threshold", type=float, default=0.015, help="Required wrist movement in m.")
parser.add_argument("--gripper-error-threshold", type=float, default=0.006, help="Allowed gripper joint error in m.")
parser.add_argument("--left-offset-x", type=float, default=0.08)
parser.add_argument("--left-offset-y", type=float, default=0.04)
parser.add_argument("--left-offset-z", type=float, default=0.06)
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


def body_pose_env_frame(env, robot, body_id: int) -> tuple[list[float], list[float]]:
    pos = robot.data.body_pos_w[0, body_id] - env.unwrapped.scene.env_origins[0]
    quat = robot.data.body_quat_w[0, body_id]
    return [float(v) for v in pos.detach().cpu().tolist()], [float(v) for v in quat.detach().cpu().tolist()]


def _controller_packet(position: list[float], quat: list[float], trigger: float) -> np.ndarray:
    pose = np.array([*position, *quat], dtype=np.float32)
    inputs = np.zeros(7, dtype=np.float32)
    inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value] = float(trigger)
    return np.stack([pose, inputs])


def make_controller_data(
    left_position: list[float],
    left_quat: list[float],
    left_trigger: float,
    right_position: list[float],
    right_quat: list[float],
    right_trigger: float,
) -> dict:
    return {
        DeviceBase.TrackingTarget.CONTROLLER_LEFT: _controller_packet(left_position, left_quat, left_trigger),
        DeviceBase.TrackingTarget.CONTROLLER_RIGHT: _controller_packet(right_position, right_quat, right_trigger),
    }


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
    action = torch.cat(
        [wrist_retargeter.retarget(raw_data), left_gripper.retarget(raw_data), right_gripper.retarget(raw_data)],
        dim=-1,
    )
    return action.unsqueeze(0)


def step_command(env, action: torch.Tensor, steps: int) -> None:
    for _ in range(steps):
        env.step(action)


def measure_side(robot, wrist_body_id: int, gripper_joint_ids: list[int], gripper_body_ids: list[int]) -> dict:
    gripper_body_pos = robot.data.body_pos_w[0, gripper_body_ids]
    return {
        "wrist_pos_w": [float(v) for v in robot.data.body_pos_w[0, wrist_body_id].detach().cpu().tolist()],
        "gripper_joint_pos": [float(v) for v in robot.data.joint_pos[0, gripper_joint_ids].detach().cpu().tolist()],
        "finger_body_separation_m": float(torch.linalg.norm(gripper_body_pos[1] - gripper_body_pos[2]).item()),
    }


def measure(robot, left_body_id, right_body_id, left_joint_ids, right_joint_ids, left_body_ids, right_body_ids) -> dict:
    return {
        "left": measure_side(robot, left_body_id, left_joint_ids, left_body_ids),
        "right": measure_side(robot, right_body_id, right_joint_ids, right_body_ids),
    }


def max_abs_error(values: list[float], target: float) -> float:
    return max(abs(value - target) for value in values)


def write_reports(report: dict) -> None:
    json_path = Path(args_cli.out_json)
    md_path = Path(args_cli.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# G1 Dex1 Bilateral Motion Controller Retargeter Smoke Test",
                "",
                f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
                f"- Task: `{report['task']}`",
                f"- Action space: `{report['action_space']}`",
                f"- Retargeted action shape: `{report['retargeted_action_shape']}`",
                f"- Action terms: `{report['action_terms']}`",
                f"- Action term dims: `{report['action_term_dims']}`",
                f"- Left wrist displacement: `{report['left_wrist_displacement_m']:.6f} m`",
                f"- Right wrist displacement: `{report['right_wrist_displacement_m']:.6f} m`",
                f"- Left close max error: `{report['left_close_max_abs_error']:.6f} m`",
                f"- Right close max error: `{report['right_close_max_abs_error']:.6f} m`",
                f"- Left reopen max error: `{report['left_reopen_max_abs_error']:.6f} m`",
                f"- Right reopen max error: `{report['right_reopen_max_abs_error']:.6f} m`",
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
        left_wrist_ids, left_wrist_names = robot.find_bodies([LEFT_WRIST_BODY], preserve_order=True)
        right_wrist_ids, right_wrist_names = robot.find_bodies([RIGHT_WRIST_BODY], preserve_order=True)
        left_joint_ids, left_joint_names = robot.find_joints(LEFT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        right_joint_ids, right_joint_names = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        left_body_ids, left_body_names = robot.find_bodies(LEFT_DEX1_BODIES, preserve_order=True)
        right_body_ids, right_body_names = robot.find_bodies(RIGHT_DEX1_BODIES, preserve_order=True)
        if len(left_wrist_ids) != 1 or len(right_wrist_ids) != 1:
            raise RuntimeError(f"Could not resolve wrist bodies: {left_wrist_names=} {right_wrist_names=}")

        left_default_pos, left_quat = body_pose_env_frame(env, robot, left_wrist_ids[0])
        right_default_pos, right_quat = body_pose_env_frame(env, robot, right_wrist_ids[0])
        wrist_retargeter, left_gripper, right_gripper, configured_retargeters = make_retargeters_from_env_cfg(env_cfg)

        left_offset = [args_cli.left_offset_x, args_cli.left_offset_y, args_cli.left_offset_z]
        right_offset = [args_cli.right_offset_x, args_cli.right_offset_y, args_cli.right_offset_z]
        left_target_pos = [left_default_pos[i] + left_offset[i] for i in range(3)]
        right_target_pos = [right_default_pos[i] + right_offset[i] for i in range(3)]

        open_raw = make_controller_data(left_default_pos, left_quat, 0.0, right_default_pos, right_quat, 0.0)
        open_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, open_raw)
        step_command(env, open_action, args_cli.steps_per_command)
        initial = measure(robot, left_wrist_ids[0], right_wrist_ids[0], left_joint_ids, right_joint_ids, left_body_ids, right_body_ids)

        left_close_raw = make_controller_data(left_target_pos, left_quat, 1.0, right_default_pos, right_quat, 0.0)
        left_close_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, left_close_raw)
        step_command(env, left_close_action, args_cli.steps_per_command)
        left_closed = measure(robot, left_wrist_ids[0], right_wrist_ids[0], left_joint_ids, right_joint_ids, left_body_ids, right_body_ids)

        right_close_raw = make_controller_data(left_default_pos, left_quat, 0.0, right_target_pos, right_quat, 1.0)
        right_close_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, right_close_raw)
        step_command(env, right_close_action, args_cli.steps_per_command)
        right_closed = measure(robot, left_wrist_ids[0], right_wrist_ids[0], left_joint_ids, right_joint_ids, left_body_ids, right_body_ids)

        both_close_raw = make_controller_data(left_target_pos, left_quat, 1.0, right_target_pos, right_quat, 1.0)
        both_close_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, both_close_raw)
        step_command(env, both_close_action, args_cli.steps_per_command)
        both_closed = measure(robot, left_wrist_ids[0], right_wrist_ids[0], left_joint_ids, right_joint_ids, left_body_ids, right_body_ids)

        reopen_raw = make_controller_data(left_default_pos, left_quat, 0.0, right_default_pos, right_quat, 0.0)
        reopen_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, reopen_raw)
        step_command(env, reopen_action, args_cli.steps_per_command)
        reopened = measure(robot, left_wrist_ids[0], right_wrist_ids[0], left_joint_ids, right_joint_ids, left_body_ids, right_body_ids)

        left_displacement = torch.linalg.norm(
            torch.tensor(left_closed["left"]["wrist_pos_w"]) - torch.tensor(initial["left"]["wrist_pos_w"])
        ).item()
        right_displacement = torch.linalg.norm(
            torch.tensor(right_closed["right"]["wrist_pos_w"]) - torch.tensor(initial["right"]["wrist_pos_w"])
        ).item()

        left_close_error = max_abs_error(left_closed["left"]["gripper_joint_pos"], LEFT_DEX1_CLOSE)
        right_close_error = max_abs_error(right_closed["right"]["gripper_joint_pos"], RIGHT_DEX1_CLOSE)
        both_left_close_error = max_abs_error(both_closed["left"]["gripper_joint_pos"], LEFT_DEX1_CLOSE)
        both_right_close_error = max_abs_error(both_closed["right"]["gripper_joint_pos"], RIGHT_DEX1_CLOSE)
        left_reopen_error = max_abs_error(reopened["left"]["gripper_joint_pos"], LEFT_DEX1_OPEN)
        right_reopen_error = max_abs_error(reopened["right"]["gripper_joint_pos"], RIGHT_DEX1_OPEN)

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "passed": (
                tuple(open_action.shape) == env.action_space.shape
                and left_displacement >= args_cli.wrist_motion_threshold
                and right_displacement >= args_cli.wrist_motion_threshold
                and left_close_error <= args_cli.gripper_error_threshold
                and right_close_error <= args_cli.gripper_error_threshold
                and both_left_close_error <= args_cli.gripper_error_threshold
                and both_right_close_error <= args_cli.gripper_error_threshold
                and left_reopen_error <= args_cli.gripper_error_threshold
                and right_reopen_error <= args_cli.gripper_error_threshold
            ),
            "task": args_cli.task,
            "device": env.unwrapped.device,
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "retargeted_action_shape": list(open_action.shape),
            "open_action": [float(v) for v in open_action[0].detach().cpu().tolist()],
            "left_close_action": [float(v) for v in left_close_action[0].detach().cpu().tolist()],
            "right_close_action": [float(v) for v in right_close_action[0].detach().cpu().tolist()],
            "both_close_action": [float(v) for v in both_close_action[0].detach().cpu().tolist()],
            "reopen_action": [float(v) for v in reopen_action[0].detach().cpu().tolist()],
            "action_terms": env.unwrapped.action_manager.active_terms,
            "action_term_dims": env.unwrapped.action_manager.action_term_dim,
            "configured_motion_controller_retargeters": configured_retargeters,
            "left_wrist_names": left_wrist_names,
            "right_wrist_names": right_wrist_names,
            "left_gripper_joint_names": left_joint_names,
            "right_gripper_joint_names": right_joint_names,
            "left_gripper_body_names": left_body_names,
            "right_gripper_body_names": right_body_names,
            "mock_left_controller_offset": left_offset,
            "mock_right_controller_offset": right_offset,
            "mock_left_controller_default_position": left_default_pos,
            "mock_right_controller_default_position": right_default_pos,
            "mock_left_controller_target_position": left_target_pos,
            "mock_right_controller_target_position": right_target_pos,
            "initial": initial,
            "left_closed": left_closed,
            "right_closed": right_closed,
            "both_closed": both_closed,
            "reopened": reopened,
            "left_wrist_displacement_m": float(left_displacement),
            "right_wrist_displacement_m": float(right_displacement),
            "left_close_max_abs_error": float(left_close_error),
            "right_close_max_abs_error": float(right_close_error),
            "both_left_close_max_abs_error": float(both_left_close_error),
            "both_right_close_max_abs_error": float(both_right_close_error),
            "left_reopen_max_abs_error": float(left_reopen_error),
            "right_reopen_max_abs_error": float(right_reopen_error),
        }
        write_reports(report)
        print(f"[INFO] Wrote JSON report: {args_cli.out_json}")
        print(f"[INFO] Wrote Markdown report: {args_cli.out_md}")
        print(f"[INFO] Overall result: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"[INFO] retargeted action shape: {tuple(open_action.shape)}")
        print(f"[INFO] left/right wrist displacement: {left_displacement:.6f} m / {right_displacement:.6f} m")
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
