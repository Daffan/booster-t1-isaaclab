import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from omni.isaac.lab.sensors import ContactSensor

from isaaclab.humanoid_tasks.envs import HumanoidRLEnv

def time_clock(env: HumanoidRLEnv, eps: float=1e-4) -> torch.Tensor:
    """access the time clock phase of the motion"""
    if not hasattr(env, 'phase_time'):
        env.phase_time = torch.zeros(env.num_envs, device=env.device)
    phase = 2. * torch.pi * env.phase_time * env.cfg.phase_freq
    output = torch.stack(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
    dim=1)
    return output