# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create a USD overlay that adds fixed wrist/ego camera prims to the G1 Dex1 asset."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_SOURCE_USD = "/workspace/isaaclab/docs/g1_dex1_assets/g1_29dof_dex1_1_v4_test_good.usd"
DEFAULT_OUT_USD = "/workspace/isaaclab/docs/g1_dex1_assets/g1_29dof_dex1_1_v4_with_cameras.usda"

parser = argparse.ArgumentParser(description="Create a G1 Dex1 USD overlay with built-in cameras.")
parser.add_argument("--source-usd", default=DEFAULT_SOURCE_USD, help="Source robot USD to reference.")
parser.add_argument("--out-usd", default=DEFAULT_OUT_USD, help="Output overlay USD/USDA path.")
parser.add_argument("--right-wrist-pos", type=float, nargs=3, default=(0.10, -0.08, 0.06))
parser.add_argument("--right-wrist-rot-world", type=float, nargs=4, default=(0.85128069, -0.11272602, 0.20273128, 0.47065280))
parser.add_argument("--left-wrist-pos", type=float, nargs=3, default=(0.10, 0.08, 0.06))
parser.add_argument("--left-wrist-rot-world", type=float, nargs=4, default=(0.85128069, 0.11272602, 0.20273128, -0.47065280))
parser.add_argument("--ego-pos", type=float, nargs=3, default=(0.0576, 0.0175, 0.4299))
parser.add_argument("--ego-rot-ros", type=float, nargs=4, default=(0.5, -0.5, 0.5, -0.5))
parser.add_argument("--focal-length", type=float, default=18.0)
parser.add_argument("--focus-distance", type=float, default=400.0)
parser.add_argument("--horizontal-aperture", type=float, default=20.955)
parser.add_argument("--clipping-min", type=float, default=0.02)
parser.add_argument("--clipping-max", type=float, default=5.0)
parser.add_argument("--skip-kit-cleanup", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import posixpath
import sys
import traceback
from pathlib import Path

import torch
from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.utils.math import convert_camera_frame_orientation_convention


CAMERA_SPECS = {
    "left_wrist_cam": {
        "parent_link": "left_wrist_yaw_link",
        "mount_name": "left_wrist_cam_mount",
        "camera_name": "camera",
        "pos_attr": "left_wrist_pos",
        "rot_attr": "left_wrist_rot_world",
        "rot_origin": "world",
    },
    "right_wrist_cam": {
        "parent_link": "right_wrist_yaw_link",
        "mount_name": "right_wrist_cam_mount",
        "camera_name": "camera",
        "pos_attr": "right_wrist_pos",
        "rot_attr": "right_wrist_rot_world",
        "rot_origin": "world",
    },
    "ego_cam": {
        "parent_link": "torso_link",
        "mount_name": "ego_cam",
        "camera_name": "camera",
        "pos_attr": "ego_pos",
        "rot_attr": "ego_rot_ros",
        "rot_origin": "ros",
    },
}


def _find_unique_prim(stage: Usd.Stage, prim_name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == prim_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one prim named {prim_name!r}, found {len(matches)}: {[str(p.GetPath()) for p in matches[:10]]}")
    return matches[0]


def _quat_to_gf_rotation(quat_wxyz: tuple[float, float, float, float]) -> Gf.Quatf:
    return Gf.Quatf(float(quat_wxyz[0]), Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])))


def _convert_to_opengl(quat_wxyz: tuple[float, float, float, float], origin: str) -> tuple[float, float, float, float]:
    quat = torch.tensor([quat_wxyz], dtype=torch.float32)
    converted = convert_camera_frame_orientation_convention(quat, origin=origin, target="opengl")[0]
    converted = converted / torch.linalg.norm(converted)
    return tuple(float(value) for value in converted.tolist())


def _set_xform(prim: Usd.Prim, translate=None, orient=None) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate = translate if translate is not None else (0.0, 0.0, 0.0)
    orient = orient if orient is not None else (1.0, 0.0, 0.0, 0.0)
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in translate)))
    xform.AddOrientOp().Set(_quat_to_gf_rotation(orient))
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _set_camera_attrs(camera: UsdGeom.Camera) -> None:
    camera.GetFocalLengthAttr().Set(float(args_cli.focal_length))
    camera.GetFocusDistanceAttr().Set(float(args_cli.focus_distance))
    camera.GetHorizontalApertureAttr().Set(float(args_cli.horizontal_aperture))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(args_cli.clipping_min), float(args_cli.clipping_max)))


def _relative_reference(out_usd: Path, source_usd: Path) -> str:
    return posixpath.relpath(source_usd.resolve().as_posix(), out_usd.resolve().parent.as_posix())


def create_overlay() -> dict:
    source_usd = Path(args_cli.source_usd)
    out_usd = Path(args_cli.out_usd)
    out_usd.parent.mkdir(parents=True, exist_ok=True)

    source_stage = Usd.Stage.Open(str(source_usd))
    if source_stage is None:
        raise RuntimeError(f"Could not open source USD: {source_usd}")
    source_root = source_stage.GetDefaultPrim()
    if not source_root.IsValid():
        raise RuntimeError(f"Source USD has no valid defaultPrim: {source_usd}")

    stage = Usd.Stage.CreateNew(str(out_usd))
    root = stage.DefinePrim(source_root.GetPath(), source_root.GetTypeName() or "Xform")
    root.GetReferences().AddReference(_relative_reference(out_usd, source_usd), source_root.GetPath())
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))

    report = {
        "source_usd": str(source_usd),
        "out_usd": str(out_usd),
        "source_default_prim": str(source_root.GetPath()),
        "cameras": {},
    }

    for camera_key, spec in CAMERA_SPECS.items():
        source_link = _find_unique_prim(source_stage, spec["parent_link"])
        # Author opinions at the same absolute prim path as the referenced robot layer.
        link_path = source_link.GetPath()
        mount_path = link_path.AppendChild(spec["mount_name"])
        camera_path = mount_path.AppendChild(spec["camera_name"])
        mount = stage.DefinePrim(mount_path, "Xform")
        camera_prim = stage.DefinePrim(camera_path, "Camera")

        local_pos = tuple(float(value) for value in getattr(args_cli, spec["pos_attr"]))
        local_rot_source = tuple(float(value) for value in getattr(args_cli, spec["rot_attr"]))
        local_rot_opengl = _convert_to_opengl(local_rot_source, spec["rot_origin"])

        _set_xform(mount, translate=local_pos)
        _set_xform(camera_prim, orient=local_rot_opengl)
        _set_camera_attrs(UsdGeom.Camera(camera_prim))

        report["cameras"][camera_key] = {
            "parent_link": str(link_path),
            "mount_path": str(mount_path),
            "camera_path": str(camera_path),
            "local_pos": list(local_pos),
            "local_rot_source": list(local_rot_source),
            "local_rot_source_convention": spec["rot_origin"],
            "local_rot_opengl": list(local_rot_opengl),
        }

    stage.GetRootLayer().Save()
    return report


def main() -> None:
    report = create_overlay()
    print(json.dumps(report, indent=2))


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
