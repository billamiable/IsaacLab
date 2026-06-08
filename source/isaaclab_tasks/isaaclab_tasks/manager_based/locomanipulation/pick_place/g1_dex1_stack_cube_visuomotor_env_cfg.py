# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 Dex1 stack-cube visuomotor teleop scene.

This task extends the G1 Dex1 stack-cube reachability scene with three RGB cameras
for visuomotor data recording: left wrist, right wrist, and an ego/head view.
"""

from __future__ import annotations

import os

import omni.kit.commands
from pxr import Sdf, Usd

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.sensors import sensors as sensor_spawners
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import change_prim_property, clone, get_current_stage
from isaaclab.sim.utils.transforms import standardize_xform_ops
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, to_camel_case
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import G1_DEX1_ASSET_DIR
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_stack_cube_reachability_env_cfg import (
    G1Dex1FixedBaseStackCubeReachabilityEnvCfg,
    ObservationsCfg as ReachabilityObservationsCfg,
)


CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
CAMERA_UPDATE_PERIOD = 0.0333
WAREHOUSE_EDGE_SCENE_OFFSET = (2.5, 0.0, 0.0)
G1_DEX1_VISUOMOTOR_USD_PATH = os.environ.get(
    "G1_DEX1_VISUOMOTOR_USD_PATH",
    os.path.join(G1_DEX1_ASSET_DIR, "usd", "g1_dex1_visuomotor.usda"),
)


@configclass
class XformPrimCfg(SpawnerCfg):
    func: callable = None


@clone
def spawn_xform_with_default_xform_command(
    prim_path: str,
    cfg: XformPrimCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    success, _ = omni.kit.commands.execute(
        "CreatePrimWithDefaultXform",
        prim_type="Xform",
        prim_path=prim_path,
        select_new_prim=False,
    )
    prim = stage.GetPrimAtPath(prim_path)
    if not success or not prim.IsValid():
        raise RuntimeError(f"Failed to create xform prim at path: '{prim_path}'.")
    standardize_xform_ops(
        prim, translation=_as_float_tuple(translation), orientation=_as_float_tuple(orientation)
    )
    return prim


def _as_float_tuple(values):
    if values is None:
        return None
    return tuple(float(value) for value in values)


@clone
def spawn_camera_with_default_xform_command(
    prim_path: str,
    cfg: sim_utils.PinholeCameraCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a camera using Kit's prim creation command.

    Some imported/composed G1 link prims reject raw ``stage.DefinePrim`` children,
    while Kit's ``CreatePrimWithDefaultXform`` can create valid child camera prims
    under the same links. The rest of this function mirrors Isaac Lab's default
    pinhole camera spawner so the CameraCfg behavior remains standard.
    """

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    success, _ = omni.kit.commands.execute(
        "CreatePrimWithDefaultXform",
        prim_type="Camera",
        prim_path=prim_path,
        select_new_prim=False,
    )
    prim = stage.GetPrimAtPath(prim_path)
    if not success or not prim.IsValid():
        raise RuntimeError(f"Failed to create camera prim at path: '{prim_path}'.")

    standardize_xform_ops(
        prim, translation=_as_float_tuple(translation), orientation=_as_float_tuple(orientation)
    )

    if cfg.lock_camera:
        change_prim_property(
            prop_path=f"{prim_path}.omni:kit:cameraLock",
            value=True,
            stage=stage,
            type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
        )

    if cfg.projection_type == "pinhole":
        attribute_types = sensor_spawners.CUSTOM_PINHOLE_CAMERA_ATTRIBUTES
    else:
        attribute_types = sensor_spawners.CUSTOM_FISHEYE_CAMERA_ATTRIBUTES

    for attr_name, attr_type in attribute_types.values():
        if prim.GetAttribute(attr_name).Get() is None:
            prim.CreateAttribute(attr_name, attr_type)

    non_usd_cfg_param_names = {
        "func",
        "copy_from_source",
        "lock_camera",
        "visible",
        "semantic_tags",
        "from_intrinsic_matrix",
    }
    for param_name, param_value in cfg.__dict__.items():
        if param_value is None or param_name in non_usd_cfg_param_names:
            continue
        if param_name in attribute_types:
            prim_prop_name = attribute_types[param_name][0]
        else:
            prim_prop_name = to_camel_case(param_name, to="cC")
        prim.GetAttribute(prim_prop_name).Set(param_value)

    return prim


def _camera_spawn(focal_length: float = 18.0) -> sim_utils.PinholeCameraCfg:
    return sim_utils.PinholeCameraCfg(
        func=spawn_camera_with_default_xform_command,
        focal_length=focal_length,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.02, 5.0),
    )


def _offset_init_state_pos(asset_cfg, offset: tuple[float, float, float]) -> None:
    pos = asset_cfg.init_state.pos
    asset_cfg.init_state.pos = (
        float(pos[0]) + float(offset[0]),
        float(pos[1]) + float(offset[1]),
        float(pos[2]) + float(offset[2]),
    )


def _move_task_assets(scene_cfg, offset: tuple[float, float, float]) -> None:
    for asset_name in ("robot", "table", "cube_1", "cube_2", "cube_3"):
        _offset_init_state_pos(getattr(scene_cfg, asset_name), offset)


@configclass
class VisuomotorPolicyObsCfg(ReachabilityObservationsCfg.PolicyCfg):
    """Policy observations with RGB images for HDF5 visuomotor recording."""

    ego_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("ego_cam"), "data_type": "rgb", "normalize": False},
    )
    left_wrist_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"), "data_type": "rgb", "normalize": False},
    )
    right_wrist_cam = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ObservationsCfg(ReachabilityObservationsCfg):
    """Observation groups for G1 Dex1 visuomotor recording."""

    policy: VisuomotorPolicyObsCfg = VisuomotorPolicyObsCfg()


@configclass
class G1Dex1FixedBaseStackCubeVisuomotorEnvCfg(G1Dex1FixedBaseStackCubeReachabilityEnvCfg):
    """G1 Dex1 stack-cube scene with three on-robot cameras."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.spawn.usd_path = G1_DEX1_VISUOMOTOR_USD_PATH

        self.scene.ground = None
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/Warehouse",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            debug_vis=False,
        )
        _move_task_assets(self.scene, WAREHOUSE_EDGE_SCENE_OFFSET)

        self.scene.left_wrist_cam = CameraCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/dex1_1_gripper/g1_29dof_mode_15/"
                "left_wrist_yaw_link/left_wrist_cam_mount/camera"
            ),
            update_period=CAMERA_UPDATE_PERIOD,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=None,
        )
        self.scene.right_wrist_cam = CameraCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/dex1_1_gripper/g1_29dof_mode_15/"
                "right_wrist_yaw_link/right_wrist_cam_mount/camera"
            ),
            update_period=CAMERA_UPDATE_PERIOD,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=None,
        )
        self.scene.ego_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/ego_cam/camera",
            update_period=CAMERA_UPDATE_PERIOD,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=None,
        )
        for camera_name in ("ego_cam", "left_wrist_cam", "right_wrist_cam"):
            getattr(self.scene, camera_name).update_latest_camera_pose = True

        self.scene.lazy_sensor_update = False
        self.num_rerenders_on_reset = 3
        self.sim.render_interval = 1
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["ego_cam", "left_wrist_cam", "right_wrist_cam"]
