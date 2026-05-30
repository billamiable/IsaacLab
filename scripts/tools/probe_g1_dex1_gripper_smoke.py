# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test the custom G1 + Dex1 gripper USD through Isaac Lab.

This script intentionally stays below the full task/env layer. It verifies that
Isaac Lab can load the custom USD as an articulation and command the right Dex1
prismatic finger joints from a standalone headless run.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/probe_g1_dex1_gripper_smoke.py \
        --headless \
        --usd /workspace/isaaclab/docs/g1_dex1_assets/g1_29dof_dex1_1_v4_test_good.usd

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_USD_PATH = (
    "/workspace/isaaclab/docs/g1_dex1_assets/"
    "g1_29dof_dex1_1_v4_test_good.usd"
)
DEFAULT_OUT_JSON = "/workspace/host/out/g1_dex1_right_gripper_smoke.json"
DEFAULT_OUT_MD = "/workspace/host/out/g1_dex1_right_gripper_smoke.md"
RIGHT_DEX1_JOINTS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]


parser = argparse.ArgumentParser(description="Smoke-test custom G1 Dex1 right gripper control in Isaac Lab.")
parser.add_argument("--usd", default=DEFAULT_USD_PATH, help="Path to the custom G1 Dex1 USD inside this runtime.")
parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for the JSON report.")
parser.add_argument("--out-md", default=DEFAULT_OUT_MD, help="Path for the Markdown report.")
parser.add_argument("--joint-names", nargs="+", default=RIGHT_DEX1_JOINTS, help="Right gripper joint names to drive.")
parser.add_argument("--steps-per-target", type=int, default=120, help="Simulation steps to settle each target.")
parser.add_argument("--error-threshold", type=float, default=0.005, help="Per-joint position error threshold in meters.")
parser.add_argument("--sim-dt", type=float, default=1.0 / 120.0, help="Physics timestep.")
parser.add_argument(
    "--skip-kit-cleanup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exit the Python process directly after writing reports. This avoids Kit shutdown hangs in headless CI.",
)
parser.add_argument("--fix-root", action=argparse.BooleanOptionalAction, default=True, help="Fix the articulation root.")
parser.add_argument(
    "--disable-gravity",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable gravity for this joint-control smoke test.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


def _tensor_row(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.detach().cpu().flatten().tolist()]


def _is_finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def build_robot_cfg(usd_path: str) -> ArticulationCfg:
    """Build a minimal articulation config for semantic joint-control probing."""
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=args_cli.disable_gravity,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=args_cli.fix_root,
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


def design_scene(usd_path: str) -> Articulation:
    """Create the light and robot articulation."""
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/Light", light_cfg)
    return Articulation(cfg=build_robot_cfg(usd_path))


def reset_robot(sim: SimulationContext, robot: Articulation) -> None:
    """Reset the articulation to its configured default state."""
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


def step_target(
    sim: SimulationContext,
    robot: Articulation,
    joint_ids: list[int],
    target_values: list[float],
    steps: int,
) -> dict:
    """Command a target and return the settled joint state."""
    target_tensor = torch.tensor([target_values], dtype=torch.float32, device=sim.device)
    robot.set_joint_position_target(target_tensor, joint_ids=joint_ids)

    sim_dt = sim.get_physics_dt()
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

    measured_tensor = robot.data.joint_pos[:, joint_ids]
    velocity_tensor = robot.data.joint_vel[:, joint_ids]
    measured = _tensor_row(measured_tensor)
    velocity = _tensor_row(velocity_tensor)
    errors = [measured_value - target_value for measured_value, target_value in zip(measured, target_values)]
    max_abs_error = max(abs(error) for error in errors)
    return {
        "target": [float(value) for value in target_values],
        "measured": measured,
        "velocity": velocity,
        "error": errors,
        "max_abs_error": max_abs_error,
        "finite": _is_finite(measured + velocity + errors),
        "passed": _is_finite(measured + velocity + errors) and max_abs_error <= args_cli.error_threshold,
    }


def build_targets(limits: list[list[float]]) -> list[tuple[str, list[float]]]:
    """Create same-sign and opposed-sign probe targets from the USD/PhysX limits."""
    lower = [joint_limit[0] for joint_limit in limits]
    upper = [joint_limit[1] for joint_limit in limits]
    zero = [0.0 for _ in limits]
    center = [(joint_limit[0] + joint_limit[1]) * 0.5 for joint_limit in limits]

    targets = [
        ("center", center),
        ("lower_same", lower),
        ("zero_same", zero),
        ("upper_same", upper),
    ]
    if len(limits) == 2:
        targets.extend(
            [
                ("opposed_joint1_lower_joint2_upper", [lower[0], upper[1]]),
                ("opposed_joint1_upper_joint2_lower", [upper[0], lower[1]]),
            ]
        )
    return targets


def write_reports(report: dict) -> None:
    """Write JSON and Markdown reports."""
    json_path = Path(args_cli.out_json)
    md_path = Path(args_cli.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# G1 Dex1 Right Gripper Smoke Test",
        "",
        f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
        f"- USD: `{report['usd_path']}`",
        f"- Device: `{report['device']}`",
        f"- Joint count: {report['num_joints']}",
        f"- Body count: {report['num_bodies']}",
        f"- Steps per target: {report['steps_per_target']}",
        f"- Error threshold: {report['error_threshold_m']} m",
        f"- Fix root: {report['fix_root']}",
        f"- Disable gravity: {report['disable_gravity']}",
        "",
        "## Driven Joints",
        "",
        "| ID | Name | Lower | Upper |",
        "| --- | --- | ---: | ---: |",
    ]
    for joint in report["driven_joints"]:
        lines.append(f"| {joint['id']} | `{joint['name']}` | {joint['lower']:.9f} | {joint['upper']:.9f} |")

    lines.extend(
        [
            "",
            "## Target Tracking",
            "",
            "| Target | Command | Measured | Error | Max Abs Error | Result |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for result in report["targets"]:
        lines.append(
            "| {name} | {target} | {measured} | {error} | {max_abs_error:.9f} | {status} |".format(
                name=result["name"],
                target=", ".join(f"{value:.6f}" for value in result["target"]),
                measured=", ".join(f"{value:.6f}" for value in result["measured"]),
                error=", ".join(f"{value:.6f}" for value in result["error"]),
                max_abs_error=result["max_abs_error"],
                status="PASS" if result["passed"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            "## Direction Notes",
            "",
            report["direction_note"],
            "",
            "## Next Use",
            "",
            (
                "For the first teleop experiment, map the right Pico trigger to a scalar and then to these two "
                "right-side prismatic joints. The visual open/close convention should be confirmed once with a "
                "viewport or recorded image, because the numeric smoke test only proves commandability and limit "
                "tracking."
            ),
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    usd_path = Path(args_cli.usd)
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD path does not exist in this runtime: {usd_path}")

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, -3.0, 2.0], target=[0.0, 0.0, 1.0])

    robot = design_scene(str(usd_path))
    sim.reset()
    reset_robot(sim, robot)

    joint_ids, joint_names = robot.find_joints(args_cli.joint_names, preserve_order=True)
    if len(joint_ids) != len(args_cli.joint_names):
        raise RuntimeError(
            "Could not resolve all requested joints. "
            f"requested={args_cli.joint_names}, resolved={joint_names}, all_joints={robot.joint_names}"
        )

    limits_tensor = robot.data.joint_pos_limits[:, joint_ids, :][0]
    limits = [[float(pair[0]), float(pair[1])] for pair in limits_tensor.detach().cpu().tolist()]

    target_results = []
    for name, target_values in build_targets(limits):
        result = step_target(sim, robot, joint_ids, target_values, args_cli.steps_per_target)
        result["name"] = name
        target_results.append(result)
        print(
            f"[INFO] {name}: target={result['target']} measured={result['measured']} "
            f"max_abs_error={result['max_abs_error']:.6f} passed={result['passed']}"
        )

    lower_result = next((result for result in target_results if result["name"] == "lower_same"), None)
    upper_result = next((result for result in target_results if result["name"] == "upper_same"), None)
    if lower_result is not None and upper_result is not None:
        deltas = [
            upper_value - lower_value
            for upper_value, lower_value in zip(upper_result["measured"], lower_result["measured"])
        ]
        direction_note = (
            "Increasing the commanded target from lower limit to upper limit increased the measured right gripper "
            f"joint positions by {', '.join(f'{delta:.6f} m' for delta in deltas)}. "
            "This establishes the numeric command direction. Whether upper means visually open or closed should be "
            "confirmed with one rendered frame before locking the Pico trigger convention."
        )
    else:
        direction_note = "Lower/upper comparison was not available."

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(result["passed"] for result in target_results),
        "usd_path": str(usd_path),
        "device": sim.device,
        "num_joints": robot.num_joints,
        "num_bodies": robot.num_bodies,
        "all_joint_names": robot.joint_names,
        "steps_per_target": args_cli.steps_per_target,
        "error_threshold_m": args_cli.error_threshold,
        "fix_root": args_cli.fix_root,
        "disable_gravity": args_cli.disable_gravity,
        "driven_joints": [
            {
                "id": int(joint_id),
                "name": joint_name,
                "lower": limits[index][0],
                "upper": limits[index][1],
            }
            for index, (joint_id, joint_name) in enumerate(zip(joint_ids, joint_names))
        ],
        "actuator_cfg": {
            "type": "ImplicitActuatorCfg",
            "joint_names_expr": [".*"],
            "effort_limit_sim": 300.0,
            "velocity_limit_sim": 100.0,
            "stiffness": 80.0,
            "damping": 8.0,
            "armature": 0.001,
        },
        "targets": target_results,
        "direction_note": direction_note,
    }
    write_reports(report)
    print(f"[INFO] Wrote JSON report: {args_cli.out_json}")
    print(f"[INFO] Wrote Markdown report: {args_cli.out_md}")
    print(f"[INFO] Overall result: {'PASS' if report['passed'] else 'FAIL'}")


if __name__ == "__main__":
    import os
    import sys
    import traceback

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
