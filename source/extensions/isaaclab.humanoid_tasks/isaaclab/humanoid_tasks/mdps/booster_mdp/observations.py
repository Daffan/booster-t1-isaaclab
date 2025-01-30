import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from omni.isaac.lab.sensors import ContactSensor

from isaaclab.humanoid_tasks.envs import HumanoidRLEnv
from isaaclab.humanoid_tasks.mdps.booster_mdp.commands import UniformVelocityFreqCommand

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

def gait_progress_obs(env: HumanoidRLEnv) -> torch.Tensor:
    """access the time clock phase of the motion"""
    command_term: UniformVelocityFreqCommand = env.command_manager.get_term("base_velocity")
    gait_progress = command_term.gait_progress
    gait_frequency = command_term.gait_frequency

    return torch.cat([
        (torch.cos(2 * torch.pi * gait_progress) * (gait_frequency > 1.0e-8).float()).unsqueeze(-1),
        (torch.sin(2 * torch.pi * gait_progress) * (gait_frequency > 1.0e-8).float()).unsqueeze(-1),
    ], dim=-1)
