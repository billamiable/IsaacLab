# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 Dex1 stack-cube reachability slice.

This task is intentionally not a full stacking task yet. It keeps the bilateral
Pico/Pink IK/Dex1 gripper action pipeline and replaces the simple single-cube
scene with a three-cube tabletop layout inspired by the Franka stack-cube task.
The first acceptance criterion is reachability: each wrist should be able to
move near a tabletop cube with the fixed-base humanoid height and table height.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_fixed_base_ik_scene_env_cfg import (
    ActionsCfg,
    G1Dex1FixedBaseIKSceneEnvCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex1_gripper_only_env_cfg import (
    DEX1_GRIPPER_JOINTS,
    G1_DEX1_GRIPPER_ONLY_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp


CUBE_SIZE = 0.08
TABLE_CENTER = (0.36, 0.0, 0.78)
TABLE_SIZE = (0.82, 0.72, 0.06)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] * 0.5
CUBE_CENTER_Z = TABLE_TOP_Z + CUBE_SIZE * 0.5
RIGHT_REACH_CUBE = "cube_1"
LEFT_REACH_CUBE = "cube_2"


@configclass
class G1Dex1StackCubeReachabilitySceneCfg(InteractiveSceneCfg):
    """Scene with fixed-base G1 Dex1, table, three cubes, ground, and light."""

    robot = G1_DEX1_GRIPPER_ONLY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.47, 0.45), roughness=0.75),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_CENTER, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.20, 0.95), roughness=0.45),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, -0.16, CUBE_CENTER_Z), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.08, 0.05), roughness=0.45),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, 0.16, CUBE_CENTER_Z), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.75, 0.20), roughness=0.45),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.48, 0.0, CUBE_CENTER_Z), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(size=(8.0, 8.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        """Keep the robot base fixed for this reachability slice."""
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ObservationsCfg:
    """Observations used for reachability testing and video summaries."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DEX1_GRIPPER_JOINTS, preserve_order=True)},
        )
        cube_1_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_1")})
        cube_2_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_2")})
        cube_3_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_3")})
        cube_1_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_1")})
        cube_2_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_2")})
        cube_3_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_3")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Basic terminations for the stack-cube reachability scene."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)
    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_1")}
    )
    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_2")}
    )
    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("cube_3")}
    )


@configclass
class G1Dex1FixedBaseStackCubeReachabilityEnvCfg(G1Dex1FixedBaseIKSceneEnvCfg):
    """Fixed-base G1 Dex1 scene for stack-cube reachability calibration."""

    scene: G1Dex1StackCubeReachabilitySceneCfg = G1Dex1StackCubeReachabilitySceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.viewer.eye = (2.8, -2.4, 1.8)
        self.viewer.lookat = (0.34, 0.0, 0.92)
