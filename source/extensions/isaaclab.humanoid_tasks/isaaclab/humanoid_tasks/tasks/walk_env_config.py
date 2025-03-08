import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.humanoid_tasks.robots.t1 import T1_LOCOMOTION_CFG  # isort: skip
from isaaclab.humanoid_tasks.robots.t1_wb import T1_CFG  # isort: skip
import isaaclab.humanoid_tasks.mdps.base as bmdp
import isaaclab.humanoid_tasks.mdps.walk as wmdp
from isaaclab.humanoid_tasks.envs import HumanoidRLEnvCfg


@configclass
class LowerBodySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot. Defines terrains, robots, visuals and sensors"""
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # could also be "plane"
        terrain_generator=bmdp.FLAT_ROAD_CFG,  # or none
        max_init_terrain_level=bmdp.FLAT_ROAD_CFG.num_rows - 1,
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
    robot: ArticulationCfg = T1_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class WholeBodySceneCfg(LowerBodySceneCfg):
    """Configuration for the terrain scene with a legged robot. Defines terrains, robots, visuals and sensors"""
    robot: ArticulationCfg = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = wmdp.UniformVelocityFreqCommandCfg(
        class_type=wmdp.UniformVelocityFreqCommand,
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=wmdp.UniformVelocityFreqCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), gait_frequency=(1.0, 2.0)
        ),
        filter_weight=0.4
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos=mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        clip={".*": (-1.0, 1.0)},
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Gnoise(mean=0.0, std=0.01),
        )  # [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(mean=0.0, std=0.1))  # [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # [4]
        gait_progress = ObsTerm(
            func=wmdp.gait_progress_obs,
            params={}
        )  # [2] internal time clock of the motion
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(mean=0.0, std=0.01), )  # [10]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(mean=0.0, std=0.1), scale=0.1)  # [12]
        actions = ObsTerm(func=mdp.last_action, scale=1.0)  # [12]

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

history_length = 50
@configclass
class ObservationsHistoryCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PrivilegeCfg(ObsGroup):
        """Observations for privilege group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")},
        )
        base_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")},
        )
        rigid_body_mass = ObsTerm(func=bmdp.rigid_body_mass, params={"asset_cfg": SceneEntityCfg("robot")})
        body_height = ObsTerm(func=bmdp.body_height, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_stiffness = ObsTerm(func=bmdp.joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_damping = ObsTerm(func=bmdp.joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})
        # joint_friction = ObsTerm(func=bmdp.joint_friction, params={"asset_cfg": SceneEntityCfg("robot")}, scale=0.2)

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Gnoise(mean=0.0, std=0.01),
            history_length=history_length,
            flatten_history_dim=False
        )  # [3]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(mean=0.0, std=0.1), history_length=history_length, flatten_history_dim=False)  # [3]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=history_length, flatten_history_dim=False)  # [4]
        gait_progress = ObsTerm(
            func=wmdp.gait_progress_obs,
            params={},
            history_length=history_length,
            flatten_history_dim=False
        )  # [2] internal time clock of the motion
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(mean=0.0, std=0.01), history_length=history_length, flatten_history_dim=False)  # [10]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(mean=0.0, std=0.1), scale=0.1, history_length=history_length, flatten_history_dim=False)  # [12]
        actions = ObsTerm(func=mdp.last_action, scale=1.0, history_length=history_length, flatten_history_dim=False)  # [12]

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privilege: PrivilegeCfg = PrivilegeCfg()


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

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    add_other_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-0.005, 0.015),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # randomize_joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "friction_distribution_params": (0.1, 0.7),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )

    # """Configuration for purterbation."""

    # # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-2.0, 2.0),
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

    """Configuration for reset."""

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

    reset_robot_joints = EventTerm(
        func=bmdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    humanoid_fall = DoneTerm(
        func=bmdp.humanoid_fall,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_pitch": 0.7,
            "max_roll": 0.7,
            "min_height": 0.3,
        })
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

@configclass
class RewardsCfg:
    survival = RewardTermCfg(
        func=bmdp.survival_reward,
        weight=2.0,
    )
    base_angular_velocity = RewardTermCfg(
        func=bmdp.tracking_ang_vel_reward,
        weight=4.0,
        params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=bmdp.tracking_lin_vel_reward,
        weight=8.0,
        params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    feet_swing = RewardTermCfg(
        func=wmdp.feet_swing_height,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "swing_period": 0.2,
            "target_height": 0.05
        }
    )
    action_smoothness = RewardTermCfg(
        func=bmdp.action_rate1_reward,
        weight=-1.0e-5
    )
    action_smoothness2 = RewardTermCfg(
        func=bmdp.action_rate2_reward,
        weight=-1.0e-5
    )
    joint_torques = RewardTermCfg(
        func=bmdp.joint_torques,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )  # penaltize large torques for all joints
    joint_pos_limits = RewardTermCfg(
        func=bmdp.joint_position_limit_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*")
        },
    )  # penalize joint limits
    joint_vel_penalty = RewardTermCfg(
        func=bmdp.joint_velocity_penalty,
        weight=-1.0e-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_accel_penalty = RewardTermCfg(
        func=bmdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    orientation_penalty = RewardTermCfg(
        func=bmdp.base_orientation_penalty,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewardTermCfg(
        func=wmdp.base_height,
        weight=-40.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.67},
    )
    base_z_velocity = RewardTermCfg(
        func=wmdp.base_z_velocity,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    collision = RewardTermCfg(
        func=wmdp.collision,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces", 
            body_names=[
                'Trunk', 'H1', 'H2', 'AL1', 'AL2', 'AL3', \
                'AR1', 'AR2', 'AR3', 'Waist', 'Hip_Pitch_Left', \
                'Hip_Roll_Left', 'Hip_Yaw_Left', 'Shank_Left', \
                'Ankle_Cross_Left', 'Hip_Pitch_Right', 'Hip_Roll_Right', \
                'Hip_Yaw_Right', 'Shank_Right', 'Ankle_Cross_Right']
        )}
    )
    # standstill = RewardTermCfg(
    #     func=wmdp.standstill,
    #     weight=0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    #     }
    # )
    joint_position = RewardTermCfg(
        func=wmdp.joint_position,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    foot_slip = RewardTermCfg(
        func=bmdp.foot_slip_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 1.0,
        },
    )
    jointReg_pb = RewardTermCfg(
        func=bmdp.joint_regularization,
        weight=-10.0,
        params={},
    )
    feet_distance = RewardTermCfg(
        func=wmdp.feet_distance,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"), "feet_distance_ref": 0.20}
    )
    feet_roll = RewardTermCfg(
        func=wmdp.feet_roll,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")}
    )

@configclass
class LowerBodyRLEnvCfg(HumanoidRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: LowerBodySceneCfg = LowerBodySceneCfg(num_envs=4096, env_spacing=2.5)
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
    only_positive_rewards: bool = True

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10 # 50 Hz
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.002
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


# replace the scene with the whole body robot. May not work right away
@configclass
class WholeBodyRLEnvCfg(LowerBodyRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: WholeBodySceneCfg = WholeBodySceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class LowerBodyHistoryRLEnvCfg(LowerBodyRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment with history."""

    # Basic settings
    observations: ObservationsHistoryCfg = ObservationsHistoryCfg()


@configclass
class WholeBodyHistoryRLEnvCfg(WholeBodyRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment with history."""

    # Basic settings
    observations: ObservationsHistoryCfg = ObservationsHistoryCfg()