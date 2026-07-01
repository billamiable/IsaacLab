# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the spawned object USD/PhysX properties in the G1 Dex1 stack-cube env."""

import argparse

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-G1-Dex1-FixedBase-StackCube-v0"

parser = argparse.ArgumentParser(description="Inspect object PhysX properties in the G1 Dex1 stack-cube env.")
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--cube", choices=("cube_1", "cube_2", "cube_3"), default="cube_2")
parser.add_argument("--cube-x", type=float, default=0.50)
parser.add_argument("--cube-y", type=float, default=-0.16)
parser.add_argument("--cube-z", type=float, default=0.9535)
parser.add_argument("--object-usd", default=None)
parser.add_argument("--object-scale", type=float, default=1.0)
parser.add_argument("--object-roll-deg", type=float, default=0.0)
parser.add_argument("--object-pitch-deg", type=float, default=0.0)
parser.add_argument("--object-yaw-deg", type=float, default=0.0)
parser.add_argument("--block-scale-x", type=float, default=1.0)
parser.add_argument("--block-scale-y", type=float, default=1.0)
parser.add_argument("--block-scale-z", type=float, default=1.0)
parser.add_argument("--warmup-steps", type=int, default=8)
parser.add_argument("--out-json", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
if args_cli.rendering_mode is None:
    args_cli.rendering_mode = "balanced"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import os
from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils  # noqa: F401
import isaaclab_tasks  # noqa: F401
import omni.usd
import torch
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

from isaaclab_tasks.utils import parse_env_cfg


def _euler_xyz_quat_xyzw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    quat = Rotation.from_euler("XYZ", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat()
    return tuple(float(v) for v in quat)


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "pathString"):
        return value.pathString
    if hasattr(value, "__iter__"):
        try:
            return [_json_value(v) for v in value]
        except TypeError:
            pass
    return str(value)


def _is_relevant_attr(name: str) -> bool:
    lowered = name.lower()
    needles = (
        "physics",
        "physx",
        "collision",
        "collider",
        "mass",
        "density",
        "material",
        "friction",
        "restitution",
        "rigid",
        "contact",
        "solver",
    )
    return any(needle in lowered for needle in needles)


def _authored_relevant_attrs(prim: Usd.Prim) -> dict[str, object]:
    attrs = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not _is_relevant_attr(name):
            continue
        if not attr.HasAuthoredValueOpinion():
            continue
        try:
            attrs[name] = _json_value(attr.Get())
        except Exception as exc:
            attrs[name] = f"<error: {exc}>"
    return attrs


def _relationships(prim: Usd.Prim) -> dict[str, list[str]]:
    rels = {}
    for rel in prim.GetRelationships():
        name = rel.GetName()
        if not _is_relevant_attr(name):
            continue
        rels[name] = [target.pathString for target in rel.GetTargets()]
    return rels


def _bound_info(bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> dict[str, object]:
    try:
        bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn = bound.GetMin()
        mx = bound.GetMax()
        dims = mx - mn
        return {
            "min": [float(mn[0]), float(mn[1]), float(mn[2])],
            "max": [float(mx[0]), float(mx[1]), float(mx[2])],
            "dims": [float(dims[0]), float(dims[1]), float(dims[2])],
        }
    except Exception as exc:
        return {"error": str(exc)}


def _material_binding(prim: Usd.Prim) -> dict[str, object]:
    try:
        material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        return {
            "material_path": material.GetPath().pathString if material else None,
            "relationship": relationship.GetName() if relationship else None,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _prim_record(bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> dict[str, object]:
    schemas = list(prim.GetAppliedSchemas())
    attrs = _authored_relevant_attrs(prim)
    rels = _relationships(prim)
    record = {
        "path": prim.GetPath().pathString,
        "type": prim.GetTypeName(),
        "active": prim.IsActive(),
        "applied_schemas": schemas,
        "attrs": attrs,
        "relationships": rels,
        "bound_material": _material_binding(prim),
    }
    if schemas or attrs or rels or prim.GetPath().pathString.endswith("/Cube_2") or prim.GetTypeName() in ("Cube", "Mesh"):
        record["world_bound"] = _bound_info(bbox_cache, prim)
    return record


def _configure_scene(env_cfg):
    object_cfg = getattr(env_cfg.scene, args_cli.cube)
    object_cfg.init_state.pos = [args_cli.cube_x, args_cli.cube_y, args_cli.cube_z]
    object_cfg.init_state.rot = _euler_xyz_quat_xyzw(
        args_cli.object_roll_deg, args_cli.object_pitch_deg, args_cli.object_yaw_deg
    )
    if args_cli.object_usd is not None:
        object_cfg.spawn.usd_path = args_cli.object_usd
        object_cfg.spawn.scale = (args_cli.object_scale, args_cli.object_scale, args_cli.object_scale)
    elif (args_cli.block_scale_x, args_cli.block_scale_y, args_cli.block_scale_z) != (1.0, 1.0, 1.0):
        object_cfg.spawn.scale = (args_cli.block_scale_x, args_cli.block_scale_y, args_cli.block_scale_z)
    env_cfg.sim.render_interval = 1
    env_cfg.decimation = 4


def main() -> int:
    env = None
    summary = {
        "task": args_cli.task,
        "cube": args_cli.cube,
        "object_usd": args_cli.object_usd,
        "object_scale": args_cli.object_scale,
        "block_scale_xyz": [args_cli.block_scale_x, args_cli.block_scale_y, args_cli.block_scale_z],
        "object_rpy_deg": [args_cli.object_roll_deg, args_cli.object_pitch_deg, args_cli.object_yaw_deg],
        "cube_pos": [args_cli.cube_x, args_cli.cube_y, args_cli.cube_z],
    }
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        _configure_scene(env_cfg)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        zero_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), dtype=torch.float32, device=env.device)
        for _ in range(args_cli.warmup_steps):
            env.step(zero_action)

        stage = omni.usd.get_context().get_stage()
        object_path = f"/World/envs/env_0/{args_cli.cube.replace('_', ' ').title().replace(' ', '_')}"
        object_prim = stage.GetPrimAtPath(object_path)
        if not object_prim:
            raise RuntimeError(f"Object prim not found: {object_path}")

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        records = []
        for prim in Usd.PrimRange(object_prim):
            record = _prim_record(bbox_cache, prim)
            if (
                record["applied_schemas"]
                or record["attrs"]
                or record["relationships"]
                or record["type"] in ("Cube", "Mesh")
                or record["path"] == object_path
            ):
                records.append(record)

        rigid_body_paths = []
        collider_paths = []
        material_paths = []
        for record in records:
            schemas = record["applied_schemas"]
            if "PhysicsRigidBodyAPI" in schemas or "PhysxRigidBodyAPI" in schemas:
                rigid_body_paths.append(record["path"])
            if "PhysicsCollisionAPI" in schemas or "PhysxCollisionAPI" in schemas:
                collider_paths.append(record["path"])
            if "PhysicsMaterialAPI" in schemas or "PhysxMaterialAPI" in schemas:
                material_paths.append(record["path"])

        summary.update(
            {
                "object_prim_path": object_path,
                "object_world_bound": _bound_info(bbox_cache, object_prim),
                "rigid_body_paths": rigid_body_paths,
                "collider_paths": collider_paths,
                "material_paths": material_paths,
                "prim_records": records,
            }
        )
        return 0
    finally:
        Path(args_cli.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if env is not None:
            env.close()
        os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
