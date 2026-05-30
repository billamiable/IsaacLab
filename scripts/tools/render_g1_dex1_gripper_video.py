# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a headless open/close video for the custom G1 + Dex1 right gripper.

The output video is a two-view composite:

* left: full-body global view, to avoid misleading cropped viewport captures
* right: right Dex1 gripper close-up, to make the open/close motion visible

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/render_g1_dex1_gripper_video.py \
        --headless --enable_cameras \
        --usd /workspace/isaaclab/docs/g1_dex1_assets/g1_29dof_dex1_1_v4_test_good.usd

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_USD_PATH = (
    "/workspace/isaaclab/docs/g1_dex1_assets/"
    "g1_29dof_dex1_1_v4_test_good.usd"
)
DEFAULT_OUT_MP4 = "/workspace/host/out/g1_dex1_right_gripper_open_close.mp4"
DEFAULT_OUT_SUMMARY = "/workspace/host/out/g1_dex1_right_gripper_open_close_video.json"
DEFAULT_OUT_DIR = "/workspace/host/out/g1_dex1_right_gripper_open_close_frames"
RIGHT_DEX1_JOINTS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]
RIGHT_DEX1_BODIES = ["right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2"]


parser = argparse.ArgumentParser(description="Render G1 Dex1 right gripper open/close video in Isaac Lab.")
parser.add_argument("--usd", default=DEFAULT_USD_PATH, help="Path to the custom G1 Dex1 USD inside this runtime.")
parser.add_argument("--out-mp4", default=DEFAULT_OUT_MP4, help="Output MP4 path.")
parser.add_argument("--summary", default=DEFAULT_OUT_SUMMARY, help="Output JSON summary path.")
parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for key-frame PNGs.")
parser.add_argument("--fps", type=int, default=24, help="Video frames per second.")
parser.add_argument("--frames", type=int, default=144, help="Number of frames to render.")
parser.add_argument("--physics-steps-per-frame", type=int, default=3, help="Physics steps per rendered frame.")
parser.add_argument("--steps-per-settle", type=int, default=120, help="Steps used to settle lower/upper probe states.")
parser.add_argument(
    "--control-source",
    choices=["direct_joint_target", "mock_pico_trigger"],
    default="direct_joint_target",
    help="Drive the gripper directly, or route a mock Pico right trigger through the existing retargeter.",
)
parser.add_argument(
    "--controller-threshold",
    type=float,
    default=0.5,
    help="Trigger threshold used by GripperTriggerOrPinchRetargeter in mock_pico_trigger mode.",
)
parser.add_argument("--global-width", type=int, default=960)
parser.add_argument("--global-height", type=int, default=540)
parser.add_argument("--close-width", type=int, default=640)
parser.add_argument("--close-height", type=int, default=540)
parser.add_argument("--sim-dt", type=float, default=1.0 / 120.0, help="Physics timestep.")
parser.add_argument("--global-distance-scale", type=float, default=3.2, help="Distance multiplier for full-body view.")
parser.add_argument("--close-distance", type=float, default=0.55, help="Distance from right gripper for close-up view.")
parser.add_argument(
    "--skip-kit-cleanup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit directly after writing video to avoid Kit shutdown hangs in headless CI.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.retargeters.manipulator.gripper_trigger_or_pinch_retargeter import (
    GripperTriggerOrPinchRetargeter,
    GripperTriggerOrPinchRetargeterCfg,
)
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext


def build_robot_cfg(usd_path: str) -> ArticulationCfg:
    """Build a minimal articulation config for the visual gripper test."""
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=300.0,
                velocity_limit_sim=100.0,
                stiffness=80.0,
                damping=8.0,
                armature=0.001,
            ),
        },
    )


def build_camera(name: str, width: int, height: int) -> Camera:
    """Create a world-space RGB camera sensor."""
    camera_cfg = CameraCfg(
        prim_path=f"/World/{name}",
        update_period=0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.01, 100.0),
        ),
    )
    return Camera(cfg=camera_cfg)


def design_scene(usd_path: str) -> tuple[Articulation, Camera, Camera]:
    """Create robot, lights, ground, and cameras."""
    ground_cfg = sim_utils.GroundPlaneCfg(size=(8.0, 8.0))
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    dome_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.8, 0.8, 0.8))
    dome_cfg.func("/World/DomeLight", dome_cfg)
    distant_cfg = sim_utils.DistantLightCfg(intensity=1600.0, color=(0.95, 0.95, 0.9))
    distant_cfg.func("/World/KeyLight", distant_cfg)

    robot = Articulation(cfg=build_robot_cfg(usd_path))
    global_camera = build_camera("GlobalViewCamera", args_cli.global_width, args_cli.global_height)
    close_camera = build_camera("RightDex1CloseCamera", args_cli.close_width, args_cli.close_height)
    return robot, global_camera, close_camera


def reset_robot(sim: SimulationContext, robot: Articulation) -> None:
    """Reset robot root and joint state."""
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    robot.reset()
    sim.step()
    robot.update(sim.get_physics_dt())


def compute_body_bounds(robot: Articulation) -> tuple[torch.Tensor, float]:
    """Compute a conservative full-body target and radius from body origins."""
    body_pos = robot.data.body_pos_w[0]
    minimum = body_pos.min(dim=0).values
    maximum = body_pos.max(dim=0).values
    center = (minimum + maximum) * 0.5
    center[2] = torch.clamp(center[2], min=1.0)
    diagonal = torch.linalg.norm(maximum - minimum).item()
    radius = max(diagonal * 0.5, 1.0)
    return center, radius


def set_global_camera_pose(sim: SimulationContext, robot: Articulation, camera: Camera) -> dict:
    """Place the global camera far enough to see the whole robot."""
    center, radius = compute_body_bounds(robot)
    view_dir = torch.tensor([1.6, -2.2, 0.75], dtype=torch.float32, device=sim.device)
    view_dir = view_dir / torch.linalg.norm(view_dir)
    eye = center + view_dir * max(radius * args_cli.global_distance_scale, 3.0)
    camera.set_world_poses_from_view(eye.unsqueeze(0), center.unsqueeze(0))
    return {
        "eye": [float(v) for v in eye.detach().cpu().tolist()],
        "target": [float(v) for v in center.detach().cpu().tolist()],
        "radius": float(radius),
    }


def gripper_target(robot: Articulation, body_ids: list[int]) -> torch.Tensor:
    """Return the current average world position of the right Dex1 bodies."""
    return robot.data.body_pos_w[0, body_ids].mean(dim=0)


def set_close_camera_pose(sim: SimulationContext, robot: Articulation, body_ids: list[int], camera: Camera) -> dict:
    """Place the close-up camera to look at the right Dex1 gripper."""
    target = gripper_target(robot, body_ids)
    view_dir = torch.tensor([0.75, -0.55, 0.28], dtype=torch.float32, device=sim.device)
    view_dir = view_dir / torch.linalg.norm(view_dir)
    eye = target + view_dir * args_cli.close_distance
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
    return {
        "eye": [float(v) for v in eye.detach().cpu().tolist()],
        "target": [float(v) for v in target.detach().cpu().tolist()],
    }


def command_gripper(sim: SimulationContext, robot: Articulation, joint_ids: list[int], values: list[float]) -> None:
    """Set gripper joint position target."""
    target_tensor = torch.tensor([values], dtype=torch.float32, device=sim.device)
    robot.set_joint_position_target(target_tensor, joint_ids=joint_ids)


def step_world(
    sim: SimulationContext,
    robot: Articulation,
    cameras: tuple[Camera, Camera],
    steps: int,
) -> None:
    """Advance physics and camera buffers."""
    sim_dt = sim.get_physics_dt()
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
    for camera in cameras:
        camera.update(sim_dt)


def settle_and_measure(
    sim: SimulationContext,
    robot: Articulation,
    cameras: tuple[Camera, Camera],
    joint_ids: list[int],
    body_ids: list[int],
    values: list[float],
) -> dict:
    """Settle a target and measure joint and finger-body separation."""
    command_gripper(sim, robot, joint_ids, values)
    step_world(sim, robot, cameras, args_cli.steps_per_settle)
    body_pos = robot.data.body_pos_w[0, body_ids]
    if len(body_ids) >= 3:
        separation = torch.linalg.norm(body_pos[1] - body_pos[2]).item()
    else:
        separation = 0.0
    measured = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
    return {
        "target": [float(v) for v in values],
        "measured": [float(v) for v in measured],
        "finger_body_separation_m": float(separation),
    }



def build_mock_pico_gripper_retargeter(sim_device: str) -> GripperTriggerOrPinchRetargeter:
    """Use the same trigger retargeter as the Franka motion-controller task."""
    cfg = GripperTriggerOrPinchRetargeterCfg(
        bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
        bound_controller=DeviceBase.TrackingTarget.CONTROLLER_RIGHT,
        controller_threshold=args_cli.controller_threshold,
        sim_device=sim_device,
    )
    return GripperTriggerOrPinchRetargeter(cfg)


def make_mock_pico_controller_data(trigger: float) -> dict:
    """Build the OpenXRDevice-style raw data row for a right Pico motion controller."""
    pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    inputs = np.zeros(7, dtype=np.float32)
    inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value] = float(trigger)
    return {DeviceBase.TrackingTarget.CONTROLLER_RIGHT: np.stack([pose, inputs])}


def resolve_gripper_command(
    alpha: float,
    lower: list[float],
    upper: list[float],
    retargeter: GripperTriggerOrPinchRetargeter | None,
) -> tuple[list[float], dict]:
    """Resolve either direct joint target or mock Pico trigger into Dex1 joint targets."""
    if args_cli.control_source == "direct_joint_target":
        command_values = [
            lower_value + alpha * (upper_value - lower_value)
            for lower_value, upper_value in zip(lower, upper)
        ]
        return command_values, {
            "control_source": "direct_joint_target",
            "alpha": float(alpha),
            "trigger": None,
            "retargeter_output": None,
            "binary_action_convention": None,
        }

    if retargeter is None:
        raise RuntimeError("mock_pico_trigger mode requires a retargeter")

    trigger = float(alpha)
    raw_data = make_mock_pico_controller_data(trigger)
    retargeter_output = float(retargeter.retarget(raw_data)[0].detach().cpu().item())
    command_values = upper if retargeter_output >= 0.0 else lower
    return list(command_values), {
        "control_source": "mock_pico_trigger",
        "alpha": float(alpha),
        "trigger": trigger,
        "retargeter_output": retargeter_output,
        "binary_action_convention": "positive=open, negative=close",
    }


def rgb_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert Isaac Lab camera RGB tensor to uint8 RGB numpy array."""
    array = image.detach().cpu().numpy()
    array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a compact text overlay on an RGB image."""
    output = image.copy()
    x, y = 16, 30
    for line in lines:
        cv2.putText(output, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
        y += 30
    return output


def make_composite(
    global_camera: Camera,
    close_camera: Camera,
    frame_index: int,
    alpha: float,
    command_values: list[float],
    visual_state: str,
    control_lines: list[str] | None = None,
) -> np.ndarray:
    """Create one RGB composite frame."""
    global_rgb = rgb_tensor_to_uint8(global_camera.data.output["rgb"][0])
    close_rgb = rgb_tensor_to_uint8(close_camera.data.output["rgb"][0])
    global_rgb = annotate(
        global_rgb,
        [
            "Global full-body view",
            f"frame {frame_index:03d}  command alpha {alpha:.2f}",
        ],
    )
    close_lines = [
        "Right Dex1 gripper close-up",
        f"state: {visual_state}",
        f"target: {command_values[0]:.4f}, {command_values[1]:.4f} m",
    ]
    if control_lines is not None:
        close_lines.extend(control_lines)
    close_rgb = annotate(close_rgb, close_lines)
    return np.concatenate([global_rgb, close_rgb], axis=1)


def smooth_open_close_alpha(frame_index: int, num_frames: int) -> float:
    """Return a smooth lower->upper->lower command alpha."""
    if num_frames <= 1:
        return 0.0
    phase = frame_index / float(num_frames - 1)
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)


def write_video_summary(summary_path: Path, summary: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    usd_path = Path(args_cli.usd)
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD path does not exist in this runtime: {usd_path}")

    out_mp4 = Path(args_cli.out_mp4)
    out_summary = Path(args_cli.summary)
    out_dir = Path(args_cli.out_dir)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    robot, global_camera, close_camera = design_scene(str(usd_path))

    sim.reset()
    reset_robot(sim, robot)

    joint_ids, joint_names = robot.find_joints(RIGHT_DEX1_JOINTS, preserve_order=True)
    if len(joint_ids) != len(RIGHT_DEX1_JOINTS):
        raise RuntimeError(f"Could not resolve right Dex1 joints. resolved={joint_names}, all={robot.joint_names}")

    body_ids, body_names = robot.find_bodies(RIGHT_DEX1_BODIES, preserve_order=True)
    if len(body_ids) != len(RIGHT_DEX1_BODIES):
        raise RuntimeError(f"Could not resolve right Dex1 bodies. resolved={body_names}, all={robot.body_names}")

    limits = robot.data.joint_pos_limits[0, joint_ids, :].detach().cpu().tolist()
    lower = [float(limit[0]) for limit in limits]
    upper = [float(limit[1]) for limit in limits]

    global_camera_info = set_global_camera_pose(sim, robot, global_camera)
    close_camera_info = set_close_camera_pose(sim, robot, body_ids, close_camera)
    step_world(sim, robot, (global_camera, close_camera), 12)

    lower_probe = settle_and_measure(sim, robot, (global_camera, close_camera), joint_ids, body_ids, lower)
    upper_probe = settle_and_measure(sim, robot, (global_camera, close_camera), joint_ids, body_ids, upper)
    if upper_probe["finger_body_separation_m"] > lower_probe["finger_body_separation_m"]:
        lower_visual_state = "closed/lower"
        upper_visual_state = "open/upper"
    else:
        lower_visual_state = "open/lower"
        upper_visual_state = "closed/upper"

    retargeter = None
    if args_cli.control_source == "mock_pico_trigger":
        retargeter = build_mock_pico_gripper_retargeter(sim.device)

    initial_command = upper if args_cli.control_source == "mock_pico_trigger" else lower
    command_gripper(sim, robot, joint_ids, initial_command)
    step_world(sim, robot, (global_camera, close_camera), args_cli.steps_per_settle)

    video_size = (args_cli.global_width + args_cli.close_width, max(args_cli.global_height, args_cli.close_height))
    if args_cli.global_height != args_cli.close_height:
        raise ValueError("Global and close camera heights must match for simple side-by-side video composition.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(out_mp4), fourcc, args_cli.fps, video_size)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_mp4}")

    key_frame_indices = {0, args_cli.frames // 2, args_cli.frames - 1}
    frame_records = []
    for frame_index in range(args_cli.frames):
        alpha = smooth_open_close_alpha(frame_index, args_cli.frames)
        command_values, control_record = resolve_gripper_command(alpha, lower, upper, retargeter)
        command_gripper(sim, robot, joint_ids, command_values)
        step_world(sim, robot, (global_camera, close_camera), args_cli.physics_steps_per_frame)
        set_close_camera_pose(sim, robot, body_ids, close_camera)

        if args_cli.control_source == "direct_joint_target":
            visual_state = upper_visual_state if alpha >= 0.5 else lower_visual_state
        else:
            visual_state = upper_visual_state if command_values == upper else lower_visual_state
        control_lines = [f"source: {args_cli.control_source}"]
        if args_cli.control_source == "mock_pico_trigger":
            control_lines.extend(
                [
                    f"mock right trigger: {control_record['trigger']:.2f}",
                    f"retargeter output: {control_record['retargeter_output']:+.1f}",
                ]
            )
        composite = make_composite(
            global_camera,
            close_camera,
            frame_index,
            alpha,
            command_values,
            visual_state,
            control_lines,
        )
        video.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        if frame_index in key_frame_indices:
            cv2.imwrite(str(out_dir / f"frame_{frame_index:03d}.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        frame_records.append(
            {
                "frame": frame_index,
                **control_record,
                "joint_target": [float(value) for value in command_values],
                "visual_state": visual_state,
            }
        )

    video.release()
    if not out_mp4.exists() or out_mp4.stat().st_size == 0:
        raise RuntimeError(f"Video file was not written: {out_mp4}")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "usd_path": str(usd_path),
        "out_mp4": str(out_mp4),
        "out_dir": str(out_dir),
        "fps": args_cli.fps,
        "frames": args_cli.frames,
        "control_source": args_cli.control_source,
        "controller_threshold": args_cli.controller_threshold,
        "video_size": list(video_size),
        "joint_names": joint_names,
        "joint_ids": [int(joint_id) for joint_id in joint_ids],
        "body_names": body_names,
        "body_ids": [int(body_id) for body_id in body_ids],
        "lower_probe": lower_probe,
        "upper_probe": upper_probe,
        "lower_visual_state": lower_visual_state,
        "upper_visual_state": upper_visual_state,
        "global_camera": global_camera_info,
        "close_camera_initial": close_camera_info,
        "frame_records": frame_records,
        "bytes": out_mp4.stat().st_size,
    }
    write_video_summary(out_summary, summary)
    print(f"[INFO] Wrote video: {out_mp4}")
    print(f"[INFO] Wrote key frames: {out_dir}")
    print(f"[INFO] Wrote summary: {out_summary}")
    print(
        "[INFO] lower separation={:.6f} m, upper separation={:.6f} m, inferred states: lower={}, upper={}".format(
            lower_probe["finger_body_separation_m"],
            upper_probe["finger_body_separation_m"],
            lower_visual_state,
            upper_visual_state,
        )
    )


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
