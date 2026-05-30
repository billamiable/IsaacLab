# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe fixed-base G1 Dex1 stack-cube reachability with mock Pico controllers."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0"
DEFAULT_OUT_JSON = "/workspace/host/out/g1_dex1_stack_cube_reachability_smoke.json"
DEFAULT_OUT_MD = "/workspace/host/out/g1_dex1_stack_cube_reachability_smoke.md"

parser = argparse.ArgumentParser(description="Probe G1 Dex1 fixed-base stack-cube reachability.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for JSON report.")
parser.add_argument("--out-md", default=DEFAULT_OUT_MD, help="Path for Markdown report.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps-per-command", type=int, default=80)
parser.add_argument("--approach-height", type=float, default=0.12, help="Wrist target height above cube center.")
parser.add_argument("--target-error-threshold", type=float, default=0.14, help="Allowed wrist-to-target error in meters.")
parser.add_argument("--cube-distance-threshold", type=float, default=0.20, help="Allowed wrist-to-cube distance in meters.")
parser.add_argument("--gripper-error-threshold", type=float, default=0.006, help="Allowed gripper joint error in meters.")
parser.add_argument("--right-cube", default="cube_1", choices=("cube_1", "cube_2", "cube_3"))
parser.add_argument("--left-cube", default="cube_2", choices=("cube_1", "cube_2", "cube_3"))
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


def body_pose_env_frame(env, robot, body_id: int) -> tuple[torch.Tensor, list[float]]:
    pos = robot.data.body_pos_w[0, body_id] - env.unwrapped.scene.env_origins[0]
    quat = robot.data.body_quat_w[0, body_id]
    return pos.detach().clone(), [float(v) for v in quat.detach().cpu().tolist()]


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
        raise RuntimeError("Expected both left and right gripper retargeters.")
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


def cube_pos_env(env, name: str) -> torch.Tensor:
    cube = env.unwrapped.scene[name]
    return cube.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]


def to_list(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.detach().cpu().tolist()]


def max_abs_error(values: list[float], target: float) -> float:
    return max(abs(value - target) for value in values)


def measure_side(robot, wrist_id: int, joint_ids: list[int], body_ids: list[int]) -> dict:
    body_pos = robot.data.body_pos_w[0, body_ids]
    return {
        "wrist_pos_w": to_list(robot.data.body_pos_w[0, wrist_id]),
        "gripper_joint_pos": to_list(robot.data.joint_pos[0, joint_ids]),
        "finger_body_separation_m": float(torch.linalg.norm(body_pos[1] - body_pos[2]).item()),
    }


def make_target(cube_pos: torch.Tensor, approach_height: float) -> torch.Tensor:
    return cube_pos + torch.tensor([0.0, 0.0, approach_height], dtype=torch.float32, device=cube_pos.device)


def target_metrics(env, wrist_pos_w: list[float], cube_name: str, target_env: torch.Tensor) -> dict:
    wrist_env = torch.tensor(wrist_pos_w, dtype=torch.float32, device=target_env.device) - env.unwrapped.scene.env_origins[0]
    cube_env = cube_pos_env(env, cube_name)
    delta_target = wrist_env - target_env
    delta_cube = wrist_env - cube_env
    return {
        "wrist_pos_env": to_list(wrist_env),
        "cube_pos_env": to_list(cube_env),
        "target_pos_env": to_list(target_env),
        "wrist_to_target_m": float(torch.linalg.norm(delta_target).item()),
        "wrist_to_cube_m": float(torch.linalg.norm(delta_cube).item()),
        "wrist_cube_xy_m": float(torch.linalg.norm(delta_cube[:2]).item()),
        "wrist_minus_cube_z_m": float(delta_cube[2].item()),
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
                "# G1 Dex1 Stack-Cube Reachability Smoke Test",
                "",
                f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
                f"- Task: `{report['task']}`",
                f"- Action space: `{report['action_space']}`",
                f"- Action terms: `{report['action_terms']}`",
                f"- Action term dims: `{report['action_term_dims']}`",
                f"- Retargeted action shape: `{report['retargeted_action_shape']}`",
                f"- Left cube: `{report['left_cube']}`, target error: `{report['left_approach_metrics']['wrist_to_target_m']:.6f} m`, wrist-cube: `{report['left_approach_metrics']['wrist_to_cube_m']:.6f} m`",
                f"- Right cube: `{report['right_cube']}`, target error: `{report['right_approach_metrics']['wrist_to_target_m']:.6f} m`, wrist-cube: `{report['right_approach_metrics']['wrist_to_cube_m']:.6f} m`",
                f"- Left close max error: `{report['left_close_max_abs_error']:.6f} m`",
                f"- Right close max error: `{report['right_close_max_abs_error']:.6f} m`",
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
        left_default_pos, left_quat = body_pose_env_frame(env, robot, left_wrist_ids[0])
        right_default_pos, right_quat = body_pose_env_frame(env, robot, right_wrist_ids[0])
        wrist_retargeter, left_gripper, right_gripper, configured_retargeters = make_retargeters_from_env_cfg(env_cfg)

        left_cube_pos = cube_pos_env(env, args_cli.left_cube)
        right_cube_pos = cube_pos_env(env, args_cli.right_cube)
        left_target = make_target(left_cube_pos, args_cli.approach_height)
        right_target = make_target(right_cube_pos, args_cli.approach_height)
        left_offset = to_list(left_target - left_default_pos)
        right_offset = to_list(right_target - right_default_pos)
        left_default_position = to_list(left_default_pos)
        right_default_position = to_list(right_default_pos)
        left_target_position = to_list(left_target)
        right_target_position = to_list(right_target)

        open_raw = make_controller_data(left_default_position, left_quat, 0.0, right_default_position, right_quat, 0.0)
        open_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, open_raw)
        step_command(env, open_action, args_cli.steps_per_command)

        right_raw = make_controller_data(left_default_position, left_quat, 0.0, right_target_position, right_quat, 0.0)
        right_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, right_raw)
        step_command(env, right_action, args_cli.steps_per_command)
        right_approach = measure_side(robot, right_wrist_ids[0], right_joint_ids, right_body_ids)
        right_metrics = target_metrics(env, right_approach["wrist_pos_w"], args_cli.right_cube, right_target)

        left_raw = make_controller_data(left_target_position, left_quat, 0.0, right_default_position, right_quat, 0.0)
        left_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, left_raw)
        step_command(env, left_action, args_cli.steps_per_command)
        left_approach = measure_side(robot, left_wrist_ids[0], left_joint_ids, left_body_ids)
        left_metrics = target_metrics(env, left_approach["wrist_pos_w"], args_cli.left_cube, left_target)

        both_close_raw = make_controller_data(left_target_position, left_quat, 1.0, right_target_position, right_quat, 1.0)
        both_close_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, both_close_raw)
        step_command(env, both_close_action, args_cli.steps_per_command)
        left_closed = measure_side(robot, left_wrist_ids[0], left_joint_ids, left_body_ids)
        right_closed = measure_side(robot, right_wrist_ids[0], right_joint_ids, right_body_ids)
        left_closed_metrics = target_metrics(env, left_closed["wrist_pos_w"], args_cli.left_cube, left_target)
        right_closed_metrics = target_metrics(env, right_closed["wrist_pos_w"], args_cli.right_cube, right_target)

        left_close_error = max_abs_error(left_closed["gripper_joint_pos"], LEFT_DEX1_CLOSE)
        right_close_error = max_abs_error(right_closed["gripper_joint_pos"], RIGHT_DEX1_CLOSE)

        reopen_raw = make_controller_data(left_default_position, left_quat, 0.0, right_default_position, right_quat, 0.0)
        reopen_action = retarget_action(wrist_retargeter, left_gripper, right_gripper, reopen_raw)
        step_command(env, reopen_action, args_cli.steps_per_command)
        left_reopened = measure_side(robot, left_wrist_ids[0], left_joint_ids, left_body_ids)
        right_reopened = measure_side(robot, right_wrist_ids[0], right_joint_ids, right_body_ids)
        left_reopen_error = max_abs_error(left_reopened["gripper_joint_pos"], LEFT_DEX1_OPEN)
        right_reopen_error = max_abs_error(right_reopened["gripper_joint_pos"], RIGHT_DEX1_OPEN)

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "passed": (
                tuple(open_action.shape) == env.action_space.shape
                and left_closed_metrics["wrist_to_target_m"] <= args_cli.target_error_threshold
                and right_closed_metrics["wrist_to_target_m"] <= args_cli.target_error_threshold
                and left_closed_metrics["wrist_to_cube_m"] <= args_cli.cube_distance_threshold
                and right_closed_metrics["wrist_to_cube_m"] <= args_cli.cube_distance_threshold
                and left_close_error <= args_cli.gripper_error_threshold
                and right_close_error <= args_cli.gripper_error_threshold
                and left_reopen_error <= args_cli.gripper_error_threshold
                and right_reopen_error <= args_cli.gripper_error_threshold
            ),
            "task": args_cli.task,
            "device": env.unwrapped.device,
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "action_terms": env.unwrapped.action_manager.active_terms,
            "action_term_dims": env.unwrapped.action_manager.action_term_dim,
            "configured_motion_controller_retargeters": configured_retargeters,
            "retargeted_action_shape": list(open_action.shape),
            "left_wrist_names": left_wrist_names,
            "right_wrist_names": right_wrist_names,
            "left_gripper_joint_names": left_joint_names,
            "right_gripper_joint_names": right_joint_names,
            "left_gripper_body_names": left_body_names,
            "right_gripper_body_names": right_body_names,
            "left_cube": args_cli.left_cube,
            "right_cube": args_cli.right_cube,
            "approach_height_m": args_cli.approach_height,
            "left_default_wrist_pos_env": to_list(left_default_pos),
            "right_default_wrist_pos_env": to_list(right_default_pos),
            "left_controller_offset": left_offset,
            "right_controller_offset": right_offset,
            "left_controller_default_position": left_default_position,
            "right_controller_default_position": right_default_position,
            "left_controller_target_position": left_target_position,
            "right_controller_target_position": right_target_position,
            "cube_positions_env": {
                "cube_1": to_list(cube_pos_env(env, "cube_1")),
                "cube_2": to_list(cube_pos_env(env, "cube_2")),
                "cube_3": to_list(cube_pos_env(env, "cube_3")),
            },
            "right_approach": right_approach,
            "left_approach": left_approach,
            "left_approach_metrics": left_metrics,
            "right_approach_metrics": right_metrics,
            "left_closed": left_closed,
            "right_closed": right_closed,
            "left_closed_metrics": left_closed_metrics,
            "right_closed_metrics": right_closed_metrics,
            "left_close_max_abs_error": float(left_close_error),
            "right_close_max_abs_error": float(right_close_error),
            "left_reopen_max_abs_error": float(left_reopen_error),
            "right_reopen_max_abs_error": float(right_reopen_error),
        }
        write_reports(report)
        print(f"[INFO] Wrote JSON report: {args_cli.out_json}")
        print(f"[INFO] Wrote Markdown report: {args_cli.out_md}")
        print(f"[INFO] Overall result: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"[INFO] action terms: {report['action_terms']} dims={report['action_term_dims']}")
        print(
            "[INFO] closed wrist-to-target: "
            f"left={left_closed_metrics['wrist_to_target_m']:.6f} m, "
            f"right={right_closed_metrics['wrist_to_target_m']:.6f} m"
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
