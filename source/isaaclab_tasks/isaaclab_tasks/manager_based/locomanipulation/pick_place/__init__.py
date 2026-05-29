# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that are specific to the locomanipulation environments."""

import gymnasium as gym
import os

from . import (
    agents,
    fixed_base_upper_body_ik_g1_env_cfg,
    g1_dex1_fixed_base_ik_scene_env_cfg,
    g1_dex1_fixed_base_scene_env_cfg,
    g1_dex1_gripper_only_env_cfg,
    g1_dex1_stack_cube_reachability_env_cfg,
    locomanipulation_g1_env_cfg,
)

gym.register(
    id="Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": locomanipulation_g1_env_cfg.LocomanipulationG1EnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": fixed_base_upper_body_ik_g1_env_cfg.FixedBaseUpperBodyIKG1EnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-G1-Dex1-Gripper-Only-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_dex1_gripper_only_env_cfg.G1Dex1GripperOnlyEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-G1-Dex1-FixedBase-Scene-Gripper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_dex1_fixed_base_scene_env_cfg.G1Dex1FixedBaseSceneEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-G1-Dex1-FixedBase-IK-Scene-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_dex1_fixed_base_ik_scene_env_cfg.G1Dex1FixedBaseIKSceneEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-G1-Dex1-FixedBase-StackCube-Reachability-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_dex1_stack_cube_reachability_env_cfg.G1Dex1FixedBaseStackCubeReachabilityEnvCfg,
    },
    disable_env_checker=True,
)

