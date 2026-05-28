# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test fixed-base G1 Dex1 upper-body IK plus right gripper action."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-IK-Scene-v0"
DEFAULT_OUT_JSON = "/workspace/host/out/g1_dex1_fixed_base_ik_gripper_smoke.json"
DEFAULT_OUT_MD = "/workspace/host/out/g1_dex1_fixed_base_ik_gripper_smoke.md"


parser = argparse.ArgumentParser(description="Smoke-test G1 Dex1 fixed-base IK plus gripper task.")
parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for JSON report.")
parser.add_argument("--out-md", default=DEFAULT_OUT_MD, help="Path for Markdown report.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps-per-command", type=int, default=60, help="Environment steps per command.")
parser.add_argument("--wrist-motion-threshold", type=float, default=0.015, help="Required right wrist movement in m.")
parser.add_argument("--gripper-error-threshold", type=float, default=0.006, help="Allowed gripper joint error in m.")
parser.add_argument(
    "--skip-kit-cleanup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit directly after writing reports to avoid Kit shutdown hangs in headless CI.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Import Pinocchio before AppLauncher, matching the Pink IK teleop/recording entrypoints.",
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
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    RIGHT_DEX1_CLOSE,
    RIGHT_DEX1_GRIPPER_JOINTS,
    RIGHT_DEX1_OPEN,
)
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
RIGHT_DEX1_BODIES = ["right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2"]


def body_pose_env_frame(env, robot, body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return body pose in the environment-origin frame expected by Pink action."""
    pos = robot.data.body_pos_w[:, body_id] - env.unwrapped.scene.env_origins
    quat = robot.data.body_quat_w[:, body_id]
    return pos, quat


def build_action(env, robot, left_body_id: int, right_body_id: int, right_offset: torch.Tensor, gripper: float):
    """Build [left wrist pose, right wrist pose, gripper] action."""
    left_pos, left_quat = body_pose_env_frame(env, robot, left_body_id)
    right_pos, right_quat = body_pose_env_frame(env, robot, right_body_id)
    right_pos = right_pos + right_offset.unsqueeze(0)
    wrist_action = torch.cat([left_pos, left_quat, right_pos, right_quat], dim=1)
    gripper_action = torch.full((env.unwrapped.num_envs, 1), float(gripper), device=env.unwrapped.device)
    return torch.cat([wrist_action, gripper_action], dim=1)


def step_command(env, action: torch.Tensor, steps: int) -> None:
    for _ in range(steps):
        env.step(action)


def measure(robot, right_body_id: int, gripper_joint_ids: list[int], gripper_body_ids: list[int]) -> dict:
    body_pos = robot.data.body_pos_w[0]
    gripper_body_pos = robot.data.body_pos_w[0, gripper_body_ids]
    separation = torch.linalg.norm(gripper_body_pos[1] - gripper_body_pos[2]).item()
    return {
        "right_wrist_pos_w": [float(v) for v in body_pos[right_body_id].detach().cpu().tolist()],
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

    lines = [
        "# G1 Dex1 Fixed-Base IK + Gripper Smoke Test",
        "",
        f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
        f"- Task: `{report['task']}`",
        f"- Action space: `{report['action_space']}`",
        f"- Action terms: `{report['action_terms']}`",
        f"- Action term dims: `{report['action_term_dims']}`",
        f"- Right wrist displacement: `{report['right_wrist_displacement_m']:.6f} m`",
        f"- Gripper close max error: `{report['close_gripper_max_abs_error']:.6f} m`",
        f"- Gripper reopen max error: `{report['reopen_gripper_max_abs_error']:.6f} m`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


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
        if len(gripper_joint_ids) != len(RIGHT_DEX1_GRIPPER_JOINTS):
            raise RuntimeError(f"Could not resolve gripper joints: {gripper_joint_names=}")

        device = env.unwrapped.device
        left_body_id = left_body_ids[0]
        right_body_id = right_body_ids[0]

        zero_offset = torch.zeros(3, dtype=torch.float32, device=device)
        move_offset = torch.tensor([0.10, -0.02, 0.07], dtype=torch.float32, device=device)

        open_action = build_action(env, robot, left_body_id, right_body_id, zero_offset, 1.0)
        step_command(env, open_action, args_cli.steps_per_command)
        initial = measure(robot, right_body_id, gripper_joint_ids, gripper_body_ids)

        move_close_action = build_action(env, robot, left_body_id, right_body_id, move_offset, -1.0)
        step_command(env, move_close_action, args_cli.steps_per_command)
        moved_closed = measure(robot, right_body_id, gripper_joint_ids, gripper_body_ids)

        reopen_action = build_action(env, robot, left_body_id, right_body_id, zero_offset, 1.0)
        step_command(env, reopen_action, args_cli.steps_per_command)
        reopened = measure(robot, right_body_id, gripper_joint_ids, gripper_body_ids)

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
                right_wrist_displacement >= args_cli.wrist_motion_threshold
                and close_max_abs_error <= args_cli.gripper_error_threshold
                and reopen_max_abs_error <= args_cli.gripper_error_threshold
            ),
            "task": args_cli.task,
            "device": env.unwrapped.device,
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "action_terms": env.unwrapped.action_manager.active_terms,
            "action_term_dims": env.unwrapped.action_manager.action_term_dim,
            "left_body_names": left_body_names,
            "right_body_names": right_body_names,
            "gripper_joint_names": gripper_joint_names,
            "gripper_body_names": gripper_body_names,
            "move_offset_env_frame": [float(v) for v in move_offset.detach().cpu().tolist()],
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
