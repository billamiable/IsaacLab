# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a mock Pico motion-controller slice through the real G1 Pink IK env.

This is a Lab3 smoke test for the migration work.  It does not require a
physical Pico headset.  Instead it creates a Pico-like right-controller stream:

* absolute controller pose -> right wrist target pose
* trigger scalar -> right TriHand close/open command

The resulting 28D action is sent into the official
``Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`` environment, so the robot
motion is produced by the real Isaac Lab action manager and Pink IK solver.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
DEFAULT_OUT_DIR = "/workspace/host/out/isaaclab3/mock_pico_g1_fixed_base_ik"
DEFAULT_OUT_MP4 = f"{DEFAULT_OUT_DIR}/mock_pico_g1_fixed_base_ik.mp4"
DEFAULT_OUT_JSON = f"{DEFAULT_OUT_DIR}/mock_pico_g1_fixed_base_ik_summary.json"
DEFAULT_OUT_FRAMES = f"{DEFAULT_OUT_DIR}/frames"


parser = argparse.ArgumentParser(
    description="Render mock Pico controller input driving the official fixed-base G1 Pink IK task."
)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--out-mp4", default=DEFAULT_OUT_MP4)
parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
parser.add_argument("--out-frames", default=DEFAULT_OUT_FRAMES)
parser.add_argument("--frames", type=int, default=180)
parser.add_argument("--warmup-steps", type=int, default=24)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--object-x", type=float, default=0.24)
parser.add_argument("--object-y", type=float, default=0.46)
parser.add_argument("--object-z", type=float, default=0.6996)
parser.add_argument("--pregrasp-y-offset", type=float, default=-0.16)
parser.add_argument("--grasp-y-offset", type=float, default=-0.07)
parser.add_argument("--pregrasp-z-offset", type=float, default=0.18)
parser.add_argument("--grasp-z-offset", type=float, default=0.11)
parser.add_argument("--retreat-y", type=float, default=-0.18)
parser.add_argument("--keyframe-every", type=int, default=45)
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
from isaaclab_tasks.utils import parse_env_cfg


LEFT_WRIST = "left_wrist_yaw_link"
RIGHT_WRIST = "right_wrist_yaw_link"


def _ensure_parent_dirs() -> None:
    Path(args_cli.out_mp4).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out_frames).mkdir(parents=True, exist_ok=True)


def _configure_scene(env_cfg) -> None:
    """Make the official task more readable for this visual smoke test."""
    # Put the object on the right side so the right controller/arm has a simple
    # forward reach target.  Keep the official asset and table.
    env_cfg.scene.object.init_state.pos = [args_cli.object_x, args_cli.object_y, args_cli.object_z]

    # Add one fixed front/side camera for video validation.  We set the final
    # view after env creation via set_world_poses_from_view().
    env_cfg.scene.mock_pico_front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/MockPicoFrontCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.01, 100.0),
        ),
    )

    # Keep the video relatively short and responsive.
    env_cfg.sim.render_interval = 2
    env_cfg.decimation = 4


def _body_pose(env, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    body_idx = robot.data.body_names.index(body_name)
    pos = robot.data.body_pos_w.torch[:, body_idx] - env.scene.env_origins
    quat = robot.data.body_quat_w.torch[:, body_idx]
    return pos[0].clone(), quat[0].clone()


def _object_pos(env) -> torch.Tensor:
    return (env.scene["object"].data.root_pos_w.torch - env.scene.env_origins)[0].clone()


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return a * (1.0 - t) + b * t


def _phase(frame_idx: int, total_frames: int) -> tuple[str, float]:
    u = frame_idx / max(total_frames - 1, 1)
    if u < 0.18:
        return "home_open", _smoothstep(u / 0.18)
    if u < 0.48:
        return "approach_open", _smoothstep((u - 0.18) / 0.30)
    if u < 0.65:
        return "close_trigger", _smoothstep((u - 0.48) / 0.17)
    if u < 0.84:
        return "retreat_closed", _smoothstep((u - 0.65) / 0.19)
    return "return_open", _smoothstep((u - 0.84) / 0.16)


def _right_hand_targets(trigger: float, device: torch.device) -> torch.Tensor:
    """Map a Pico-like trigger scalar to the official 14D TriHand action tail."""
    hand = torch.zeros(14, dtype=torch.float32, device=device)

    # Action order from fixed_base_upper_body_ik_g1_env_cfg:
    # [L index0, L middle0, L thumb0, R index0, R middle0, R thumb0,
    #  L index1, L middle1, L thumb1, R index1, R middle1, R thumb1,
    #  L thumb2, R thumb2]
    hand[3] = 1.15 * trigger   # right index proximal
    hand[4] = 1.15 * trigger   # right middle proximal
    hand[5] = 0.20 * trigger   # right thumb base
    hand[9] = 1.25 * trigger   # right index distal
    hand[10] = 1.25 * trigger  # right middle distal
    hand[11] = -0.45 * trigger # right thumb middle
    hand[13] = -1.15 * trigger # right thumb tip
    return hand


def _compose_action(
    left_pos: torch.Tensor,
    left_quat: torch.Tensor,
    right_pos: torch.Tensor,
    right_quat: torch.Tensor,
    trigger: float,
    device: torch.device,
) -> torch.Tensor:
    action = torch.zeros((1, 28), dtype=torch.float32, device=device)
    action[0, 0:3] = left_pos.to(device)
    action[0, 3:7] = left_quat.to(device)
    action[0, 7:10] = right_pos.to(device)
    action[0, 10:14] = right_quat.to(device)
    action[0, 14:28] = _right_hand_targets(trigger, device)
    return action


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


def _put_overlay(frame: np.ndarray, phase_name: str, trigger: float, error_m: float, frame_idx: int) -> np.ndarray:
    out = frame.copy()
    lines = [
        "IsaacLab3 mock Pico -> fixed-base G1 Pink IK",
        f"phase: {phase_name}  trigger: {trigger:.2f}",
        f"right wrist tracking error: {error_m:.3f} m",
        f"frame {frame_idx:03d}",
    ]
    y = 28
    for line in lines:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return out


def main() -> int:
    _ensure_parent_dirs()

    summary = {
        "task": args_cli.task,
        "output_mp4": args_cli.out_mp4,
        "output_json": args_cli.out_json,
        "output_frames": args_cli.out_frames,
        "output_namespace": "out/isaaclab3",
        "description": "Mock Pico right-controller absolute pose and trigger drive the official fixed-base G1 Pink IK action.",
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
            torch.tensor([[1.05, -0.42, 1.18]], dtype=torch.float32, device=env.device),
            torch.tensor([[0.10, 0.40, 0.78]], dtype=torch.float32, device=env.device),
        )

        left_home_pos, left_home_quat = _body_pose(env, LEFT_WRIST)
        right_home_pos, right_home_quat = _body_pose(env, RIGHT_WRIST)
        object_pos = _object_pos(env)

        pregrasp = object_pos + torch.tensor(
            [0.0, args_cli.pregrasp_y_offset, args_cli.pregrasp_z_offset],
            dtype=torch.float32,
            device=object_pos.device,
        )
        grasp = object_pos + torch.tensor(
            [0.0, args_cli.grasp_y_offset, args_cli.grasp_z_offset],
            dtype=torch.float32,
            device=object_pos.device,
        )
        retreat = grasp + torch.tensor([0.0, args_cli.retreat_y, 0.10], dtype=torch.float32, device=object_pos.device)

        summary.update(
            {
                "initial_left_wrist_pos": left_home_pos.detach().cpu().tolist(),
                "initial_right_wrist_pos": right_home_pos.detach().cpu().tolist(),
                "object_pos": object_pos.detach().cpu().tolist(),
                "pregrasp_target": pregrasp.detach().cpu().tolist(),
                "grasp_target": grasp.detach().cpu().tolist(),
                "retreat_target": retreat.detach().cpu().tolist(),
                "action_dim": int(env.action_manager.total_action_dim),
            }
        )

        # Warm up at the current pose, which gives Pink IK a valid initial target.
        for _ in range(args_cli.warmup_steps):
            action = _compose_action(left_home_pos, left_home_quat, right_home_pos, right_home_quat, 0.0, env.device)
            env.step(action)

        writer = cv2.VideoWriter(
            args_cli.out_mp4,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args_cli.fps,
            (args_cli.width, args_cli.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {args_cli.out_mp4}")

        max_delta = 0.0
        min_error = float("inf")
        start_right_pos = _body_pose(env, RIGHT_WRIST)[0]

        for frame_idx in range(args_cli.frames):
            phase_name, alpha = _phase(frame_idx, args_cli.frames)
            trigger = 0.0
            if phase_name == "home_open":
                target = right_home_pos
            elif phase_name == "approach_open":
                target = _lerp(right_home_pos, pregrasp, alpha)
            elif phase_name == "close_trigger":
                target = _lerp(pregrasp, grasp, alpha)
                trigger = alpha
            elif phase_name == "retreat_closed":
                target = _lerp(grasp, retreat, alpha)
                trigger = 1.0
            else:
                target = _lerp(retreat, right_home_pos, alpha)
                trigger = 1.0 - alpha

            action = _compose_action(left_home_pos, left_home_quat, target, right_home_quat, trigger, env.device)
            env.step(action)

            actual_right_pos, _ = _body_pose(env, RIGHT_WRIST)
            tracking_error = torch.linalg.norm(actual_right_pos - target).item()
            moved_delta = torch.linalg.norm(actual_right_pos - start_right_pos).item()
            max_delta = max(max_delta, moved_delta)
            min_error = min(min_error, tracking_error)

            camera.update(0.0, force_recompute=True)
            rgb = _camera_rgb(camera)
            rgb = _put_overlay(rgb, phase_name, trigger, tracking_error, frame_idx)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if frame_idx % max(args_cli.keyframe_every, 1) == 0 or frame_idx == args_cli.frames - 1:
                frame_path = str(Path(args_cli.out_frames) / f"frame_{frame_idx:04d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if frame_idx % 5 == 0 or frame_idx == args_cli.frames - 1:
                summary["records"].append(
                    {
                        "frame": frame_idx,
                        "phase": phase_name,
                        "mock_controller_right_pos": target.detach().cpu().tolist(),
                        "mock_controller_right_quat": right_home_quat.detach().cpu().tolist(),
                        "mock_controller_trigger": float(trigger),
                        "actual_right_wrist_pos": actual_right_pos.detach().cpu().tolist(),
                        "right_wrist_tracking_error_m": float(tracking_error),
                        "right_wrist_motion_delta_m": float(moved_delta),
                    }
                )

        summary["max_right_wrist_motion_delta_m"] = float(max_delta)
        summary["min_right_wrist_tracking_error_m"] = float(min_error)
        summary["passed"] = bool(max_delta > 0.05 and min_error < 0.25)
        return 0 if summary["passed"] else 2
    except Exception as exc:
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        print(summary["traceback"], file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.release()
        with open(args_cli.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if env is not None:
            env.close()
        if not args_cli.skip_kit_cleanup:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
