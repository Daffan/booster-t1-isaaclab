import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from omni.isaac.lab.sensors import ContactSensor

# if TYPE_CHECKING:
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def contact_pattern(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """access the ground truth contact pattern of the feet with floor"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # [envs, 2]
    in_contact = torch.gt(contact_force_z, 0.0).int()
    in_contact = torch.cat((in_contact[:, 0].unsqueeze(1), in_contact[:, 1].unsqueeze(1)), dim=1)
    return in_contact

def time_clock(env: ManagerBasedRLEnv, eps: float=1e-4) -> torch.Tensor:
    """access the time clock phase of the motion"""
    if not hasattr(env, 'phase_time'):
        env.phase_time = torch.zeros(env.num_envs, device=env.device)
    phase = 2. * torch.pi * env.phase_time * env.cfg.phase_freq
    smooth_sqr_wave = torch.sin(phase) / \
            (2 * torch.sqrt(torch.sin(phase) ** 2. + eps ** 2.)) + 1. / 2.
    output = torch.stack(
        [
            smooth_sqr_wave,
            torch.sin(phase),
            torch.cos(phase),
        ],
    dim=1)
    return output