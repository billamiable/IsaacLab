# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test the manager-based G1 Dex1 gripper-only task.

This script validates the formal Isaac Lab env/action-manager path:

mock Pico right trigger -> existing gripper retargeter -> env action -> action_manager -> Dex1 joints
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-Gripper-Only-v0"
DEFAULT_OUT_JSON = "/workspace/host/out/g1_dex1_gripper_only_env_smoke.json"
DEFAULT_OUT_MD = "/workspace/host/out/g1_dex1_gripper_only_env_smoke.md"


parser = argparse.ArgumentParser(description="Smoke-test G1 Dex1 gripper-only manager-based environment.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for JSON report.")
parser.add_argument("--out-md", default=DEFAULT_OUT_MD, help="Path for Markdown report.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps-per-command", type=int, default=40, help="Environment steps per gripper command.")
parser.add_argument("--error-threshold", type=float, default=0.005, help="Allowed joint target tracking error in meters.")
parser.add_argument(
    "--skip-kit-cleanup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit directly after writing reports to avoid Kit shutdown hangs in headless CI.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from isaaclab.devices.openxr.retargeters.manipulator.gripper_trigger_or_pinch_retargeter import (
    GripperTriggerOrPinchRetargeter,
    GripperTriggerOrPinchRetargeterCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    RIGHT_DEX1_CLOSE,
    RIGHT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_OPEN,
)
from isaaclab_tasks.utils import parse_env_cfg


def make_mock_pico_controller_data(trigger: float) -> dict:
    """Build OpenXRDevice-style raw data for the right Pico motion controller."""
    pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    inputs = np.zeros(7, dtype=np.float32)
    inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value] = float(trigger)
    return {DeviceBase.TrackingTarget.CONTROLLER_RIGHT: np.stack([pose, inputs])}


def make_gripper_retargeter(device: str) -> GripperTriggerOrPinchRetargeter:
    """Create the same trigger retargeter configured by the env's motion_controllers device."""
    cfg = GripperTriggerOrPinchRetargeterCfg(
        bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
        bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
        controller_threshold=0.5,
        sim_device=device,
    )
    return GripperTriggerOrPinchRetargeter(cfg)


def step_command(env, robot, joint_ids: list[int], action_value: float, target_values: list[float]) -> dict:
    """Step the env action manager and collect final joint state."""
    action = torch.full(env.action_space.shape, float(action_value), device=env.unwrapped.device)
    for _ in range(args_cli.steps_per_command):
        env.step(action)

    measured = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
    error = [float(measured_value - target_value) for measured_value, target_value in zip(measured, target_values)]
    max_abs_error = max(abs(value) for value in error)
    return {
        "action": float(action_value),
        "target": [float(value) for value in target_values],
        "measured": [float(value) for value in measured],
        "error": error,
        "max_abs_error": max_abs_error,
        "passed": max_abs_error <= args_cli.error_threshold,
    }


def write_reports(report: dict) -> None:
    """Write JSON and Markdown reports."""
    json_path = Path(args_cli.out_json)
    md_path = Path(args_cli.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# G1 Dex1 Gripper-Only Env Smoke Test",
        "",
        f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
        f"- Task: `{report['task']}`",
        f"- Device: `{report['device']}`",
        f"- Action space: `{report['action_space']}`",
        f"- Observation space: `{report['observation_space']}`",
        f"- Joints: {', '.join(f'`{name}`' for name in report['joint_names'])}",
        f"- Open target: {report['open_target']}",
        f"- Close target: {report['close_target']}",
        "",
        "## Action Manager Commands",
        "",
        "| Name | Action | Target | Measured | Max Abs Error | Result |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for result in report["direct_action_results"]:
        lines.append(
            "| {name} | {action:.1f} | {target} | {measured} | {max_abs_error:.9f} | {status} |".format(
                name=result["name"],
                action=result["action"],
                target=", ".join(f"{value:.6f}" for value in result["target"]),
                measured=", ".join(f"{value:.6f}" for value in result["measured"]),
                max_abs_error=result["max_abs_error"],
                status="PASS" if result["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Mock Pico Trigger",
            "",
            "| Trigger | Retargeter Output | Joint Target | Visual State |",
            "| ---: | ---: | --- | --- |",
        ]
    )
    for result in report["mock_pico_results"]:
        lines.append(
            "| {trigger:.2f} | {retargeter_output:+.1f} | {target} | {state} |".format(
                trigger=result["trigger"],
                retargeter_output=result["retargeter_output"],
                target=", ".join(f"{value:.6f}" for value in result["joint_target"]),
                state=result["visual_state"],
            )
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    try:
        robot = env.unwrapped.scene["robot"]
        joint_ids, joint_names = robot.find_joints(RIGHT_DEX1_GRIPPER_JOINTS, preserve_order=True)
        if len(joint_ids) != len(RIGHT_DEX1_GRIPPER_JOINTS):
            raise RuntimeError(f"Could not resolve all gripper joints. resolved={joint_names}")

        open_target = [RIGHT_DEX1_OPEN, RIGHT_DEX1_OPEN]
        close_target = [RIGHT_DEX1_CLOSE, RIGHT_DEX1_CLOSE]

        direct_action_results = []
        direct_action_results.append(
            {"name": "open_positive_action", **step_command(env, robot, joint_ids, 1.0, open_target)}
        )
        direct_action_results.append(
            {"name": "close_negative_action", **step_command(env, robot, joint_ids, -1.0, close_target)}
        )
        direct_action_results.append(
            {"name": "open_positive_action_again", **step_command(env, robot, joint_ids, 1.0, open_target)}
        )

        retargeter = make_gripper_retargeter(env.unwrapped.device)
        mock_pico_results = []
        for trigger in [0.0, 1.0, 0.0]:
            retargeter_output = float(retargeter.retarget(make_mock_pico_controller_data(trigger))[0].cpu().item())
            joint_target = open_target if retargeter_output >= 0.0 else close_target
            visual_state = "open/upper" if retargeter_output >= 0.0 else "closed/lower"
            mock_pico_results.append(
                {
                    "trigger": trigger,
                    "retargeter_output": retargeter_output,
                    "joint_target": joint_target,
                    "visual_state": visual_state,
                }
            )

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "passed": all(result["passed"] for result in direct_action_results),
            "task": args_cli.task,
            "device": env.unwrapped.device,
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "joint_ids": [int(joint_id) for joint_id in joint_ids],
            "joint_names": joint_names,
            "open_target": open_target,
            "close_target": close_target,
            "direct_action_results": direct_action_results,
            "mock_pico_results": mock_pico_results,
        }
        write_reports(report)
        print(f"[INFO] Wrote JSON report: {args_cli.out_json}")
        print(f"[INFO] Wrote Markdown report: {args_cli.out_md}")
        print(f"[INFO] Overall result: {'PASS' if report['passed'] else 'FAIL'}")
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
