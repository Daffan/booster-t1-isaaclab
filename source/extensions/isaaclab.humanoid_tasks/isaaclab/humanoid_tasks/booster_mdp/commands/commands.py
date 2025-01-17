from collections.abc import Sequence
# import pandas as pd
import numpy as np
import torch
from scipy.signal import savgol_filter
from uuid import uuid4
from tqdm import tqdm
from easydict import EasyDict
import yaml

from .command_cfg import UniformVelocityFreqCommandCfg
from omni.isaac.lab.envs.mdp import UniformVelocityCommand
from isaaclab.humanoid_tasks.envs import HumanoidRLEnv

class UniformVelocityFreqCommand(UniformVelocityCommand):
    cfg: UniformVelocityFreqCommandCfg

    def __init__(self, cfg: UniformVelocityFreqCommandCfg, env: HumanoidRLEnv):
        super().__init__(cfg, env)
        self.gait_frequency = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return torch.cat(self.vel_command_b, self.gait_frequency.unsqueeze(-1), dim=-1)
    
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        self.gait_frequency[env_ids] = r.uniform_(*self.cfg.ranges.gait_frequency)