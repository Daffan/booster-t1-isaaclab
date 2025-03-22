# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Humanoid robot."""

from __future__ import annotations


import os

# Get path to T1 usd file
file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)

usd_relative_path = '../assets/usd'
usd_path = os.path.abspath(os.path.join(file_dir, usd_relative_path))

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg, RemotizedPDActuatorCfg

##
# Configuration
##

"""
Configuration for the Booster T1 Humanoid robot
action_dims = 23
['AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_pitch', 
'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 
'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Elbow_Yaw', 
'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'Left_Knee_Pitch', 'Right_Knee_Pitch', 
'Left_Ankle_Pitch', 'Right_Ankle_Pitch', 'Left_Ankle_Roll', 'Right_Ankle_Roll']
"""
T1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=os.path.join(usd_path, "T1_Sim_fixed_arms.usd"),
        usd_path=os.path.join(usd_path, "t1_whole_body.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,  # what does this affect?
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=True
        ),
        copy_from_source=False,
    ),
    soft_joint_pos_limit_factor=0.95,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "Right_Shoulder_Pitch": 0.2,
            "Right_Shoulder_Roll": 1.35,
            "Right_Elbow_Pitch": 0.0,
            "Right_Elbow_Yaw": 0.5,
            "Left_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Left_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            'Waist': 0.0,
            'Left_Hip_Pitch': -0.2,
            'Left_Hip_Roll': 0.0,
            'Left_Hip_Yaw': 0.0,
            'Left_Knee_Pitch': 0.4,
            'Left_Ankle_Pitch': -0.25,
            'Left_Ankle_Roll': 0.0,
            'Right_Hip_Pitch': -0.2,
            'Right_Hip_Roll': 0.0,
            'Right_Hip_Yaw': 0.0,
            'Right_Knee_Pitch': 0.4,
            'Right_Ankle_Pitch': -0.25,
            'Right_Ankle_Roll': 0.0,
        }
    ),
    actuators={
        "head": IdealPDActuatorCfg(
            joint_names_expr=[
                "AAHead_yaw",
                "Head_pitch",
            ],
            stiffness=20,
            damping=0.2,
            armature=0.2,
            friction=0.1
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
            ],
            stiffness=20,
            damping=0.5,
            armature=0.2,
            friction=0.1
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["Waist"],
            stiffness=150,
            damping=5,
            armature=0.2,
            friction=0.1
        ),
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
            ],
            stiffness=200,
            damping=5,
            armature=0.1,
            friction=0.2,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[
                "Left_Ankle_Pitch",
                "Right_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Ankle_Roll"
            ],
            stiffness=50,
            damping=1,
            armature=0.1,
            friction=0.2,
        )
    },
)