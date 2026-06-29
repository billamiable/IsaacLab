# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Dex1 stack-cube visuomotor scene for Isaac Lab 3.

This task extends the fixed-base G1 Dex1 stack-cube task with three RGB cameras:
ego/chest, left wrist, and right wrist.  The camera prims are provided by the
thin USD overlay under ``docs/g1_dex1_runtime_assets/usd/g1_dex1_visuomotor.usda``.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_stack_cube_env_cfg import (
    G1_DEX1_ASSET_DIR,
    G1Dex1FixedBaseStackCubeEnvCfg,
    ObservationsCfg as StackCubeObservationsCfg,
)


CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
CAMERA_UPDATE_PERIOD = 0.0
G1_DEX1_VISUOMOTOR_USD_PATH = os.environ.get(
    "G1_DEX1_VISUOMOTOR_USD_PATH",
    os.path.join(G1_DEX1_ASSET_DIR, "usd", "g1_dex1_visuomotor.usda"),
)


@configclass
class VisuomotorPolicyObsCfg(StackCubeObservationsCfg.PolicyCfg):
    """Low-dimensional G1 Dex1 observations plus three RGB camera streams."""

    ego_cam = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("ego_cam"), "data_type": "rgb", "normalize": False},
    )
    left_wrist_cam = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"), "data_type": "rgb", "normalize": False},
    )
    right_wrist_cam = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ObservationsCfg(StackCubeObservationsCfg):
    """Observation groups for G1 Dex1 visuomotor recording."""

    policy: VisuomotorPolicyObsCfg = VisuomotorPolicyObsCfg()


@configclass
class G1Dex1FixedBaseStackCubeVisuomotorEnvCfg(G1Dex1FixedBaseStackCubeEnvCfg):
    """G1 Dex1 stack-cube task with overlay cameras and RGB observations."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.spawn.usd_path = G1_DEX1_VISUOMOTOR_USD_PATH

        # Add visual context for headless ego/wrist camera checks.
        self.scene.ground = None
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/Warehouse",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            debug_vis=False,
        )

        self.scene.ego_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/ego_cam/camera",
            update_period=CAMERA_UPDATE_PERIOD,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=None,
        )
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
        for camera_name in ("ego_cam", "left_wrist_cam", "right_wrist_cam"):
            getattr(self.scene, camera_name).update_latest_camera_pose = True

        self.scene.lazy_sensor_update = False
        self.num_rerenders_on_reset = 3
        self.sim.render_interval = 1
        self.image_obs_list = ["ego_cam", "left_wrist_cam", "right_wrist_cam"]
