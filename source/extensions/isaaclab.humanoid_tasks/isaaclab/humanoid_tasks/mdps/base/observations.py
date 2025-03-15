import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from isaaclab.sensors import ContactSensor

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def contact_pattern(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """access the ground truth contact pattern of the feet with floor"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # [envs, 2]
    in_contact = torch.gt(contact_force_z, 0.0).int()
    in_contact = torch.cat((in_contact[:, 0].unsqueeze(1), in_contact[:, 1].unsqueeze(1)), dim=1)
    return in_contact

###  Observations for Privileged Information  ###

def rigid_body_mass(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """access the ground truth mass of the body
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_masses().to(asset.device)

def body_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """access the ground truth height of the body
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]

def joint_stiffness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """access the ground truth stiffness of the joints
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    stiffness = torch.cat([v.stiffness for v in asset.actuators.values()], dim=-1).to(asset.device)
    default_stiffness = asset.data.default_joint_stiffness
    return (stiffness - default_stiffness) / default_stiffness

def joint_damping(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """access the ground truth damping of the joints
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    damping = torch.cat([v.damping for v in asset.actuators.values()], dim=-1).to(asset.device)
    default_damping = asset.data.default_joint_damping
    return (damping - default_damping) / default_damping

def joint_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """access the ground truth friction of the joints
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction_coeff