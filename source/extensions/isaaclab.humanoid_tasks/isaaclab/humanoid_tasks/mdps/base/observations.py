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