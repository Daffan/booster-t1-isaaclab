import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.humanoid_tasks.robots.t1 import T1_FIXED_ARMS_CFG  # isort: skip
from isaaclab.humanoid_tasks.robots.t1_wb import T1_CFG  # isort: skip
from isaaclab.humanoid_tasks.robots.ball import BALL_CFG  # isort: skip
import isaaclab.humanoid_tasks.mdp as hmdp
import isaaclab.humanoid_tasks.kick_mdp as kmdp
from isaaclab.humanoid_tasks.envs import HumanoidRLEnvCfg


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot. Defines terrains, robots, visuals and sensors"""
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # could also be "plane"
        terrain_generator=hmdp.FLAT_ROAD_CFG,  # or none
        max_init_terrain_level=hmdp.FLAT_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,  # show origin of each environment
    )
    # robots
    robot: ArticulationCfg = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    ball: RigidObjectCfg = BALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Ball")
    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.4,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            visible=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2))
    )  # this is a virtual goal without collision

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class CommandsCfg:
    """Curriculum terms for the MDP."""

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))  # [1]
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))  # [3]
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )  # [3]
        ball_root_pos = ObsTerm(
            func=kmdp.ball_rel_pos,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "ball_asset_cfg": SceneEntityCfg("ball"),
            }
        )  # [3]
        ball_root_vel = ObsTerm(
            func=kmdp.ball_root_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={
                "ball_asset_cfg": SceneEntityCfg("ball"),
            }
        )  # [3]
        goal_root_pos = ObsTerm(
            func=kmdp.goal_rel_pos,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "goal_asset_cfg": SceneEntityCfg("goal"),
            }
        )  # [2]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005))  # [10]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01), scale=0.1)  # [10]
        actions = ObsTerm(func=mdp.last_action, scale=0.1)  # [10]
        contact_pattern = ObsTerm(
            func=hmdp.contact_pattern,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            }
        )  # [2]
        phase_time_clock = ObsTerm(
            func=hmdp.time_clock,
            params={}
        )  # [3] internal time clock of the motion

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    ball_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )
    add_ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=".*"),
            "mass_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )
    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3)
            },
            "velocity_range": {
                "x": (-.5, .5),
                "y": (-0.3, 0.3),
                "z": (-0.2, 0.2),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_ball_goal = EventTerm(
        func=kmdp.reset_ball_goal_pos,
        mode="reset",
        params={
            "ball_asset_cfg": SceneEntityCfg("ball"),
            "goal_asset_cfg": SceneEntityCfg("goal"),
            "ball_pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
            },
            "goal_pose_range": {
                "x": (-4.0, 4.0),
                "y": (-4.0, 4.0),
            },
            "minimum_distance": 0.3,
        },
    )
    reset_robot_joints = EventTerm(
        func=hmdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    humanoid_fall = DoneTerm(
        func=hmdp.humanoid_fall,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_pitch": 0.7,
            "max_roll": 0.7,
            "min_height": 0.3,
        })
    ball_reach_goal = DoneTerm(
        func=kmdp.ball_reach_goal,
        params={
            "ball_asset_cfg": SceneEntityCfg("ball"),
            "goal_asset_cfg": SceneEntityCfg("goal"),
            "goal_radius": 0.3,
        })

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

@configclass
class RewardsCfg:
    approach_ball_pos = RewardTermCfg(
        func=kmdp.approach_ball_pos,
        weight=40.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "ball_asset_cfg": SceneEntityCfg("ball"),
        },
    )
    approach_ball_lin_vel = RewardTermCfg(
        func=kmdp.approach_ball_lin_vel,
        weight=10.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "ball_asset_cfg": SceneEntityCfg("ball"),
        },
    )
    approach_ball_yaw = RewardTermCfg(
        func=kmdp.approach_ball_yaw,
        weight=5.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "ball_asset_cfg": SceneEntityCfg("ball"),
            "std": 0.5,
        },
    )
    standstill = RewardTermCfg(
        func=kmdp.standstill,
        weight=4.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "ball_asset_cfg": SceneEntityCfg("ball"),
        },
    )
    ball_target = RewardTermCfg(
        func=kmdp.ball_target,
        weight=50.0,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "ball_asset_cfg": SceneEntityCfg("ball"),
            "goal_asset_cfg": SceneEntityCfg("goal"),
        },
    )
    # regularization reward
    action_smoothness = RewardTermCfg(
        func=hmdp.action_rate1_reward,
        weight=-1.0e-6
    )
    action_smoothness2 = RewardTermCfg(
        func=hmdp.action_rate2_reward,
        weight=-1.0e-5
    )
    joint_torques = RewardTermCfg(
        func=hmdp.joint_torques,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )  # penaltize large torques for all joints
    joint_pos_limits = RewardTermCfg(
        func=hmdp.joint_position_limit_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*")
        },
    )  # penalize joint limits
    joint_vel_penalty = RewardTermCfg(
        func=hmdp.joint_velocity_penalty,
        weight=-1.0e-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_accel_penalty = RewardTermCfg(
        func=hmdp.joint_acceleration_penalty,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    orientation_penalty = RewardTermCfg(
        func=hmdp.base_orientation_penalty,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewardTermCfg(
        func=hmdp.base_height_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.70, "std": 0.25},
    )


@configclass
class RLEnvCfg(HumanoidRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    # actions: ActionsCfg = ActionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Viewer
    viewer = ViewerCfg(eye=(12.5, 12.5, 7.5), origin_type="env", env_index=2046, asset_name="robot")

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10 # 100 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        
        # no height scan
        self.scene.height_scanner = None

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False