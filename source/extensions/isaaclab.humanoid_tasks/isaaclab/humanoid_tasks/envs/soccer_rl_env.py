from typing import Sequence
import torch
from collections import deque

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

@configclass
class SoccerRLEnvCfg(ManagerBasedRLEnvCfg):
    phase_freq: float = 1.0
    only_positive_rewards: bool = False


class SoccerRLEnv(ManagerBasedRLEnv):
    """ Soccer RL manager-based environment.
    """
    cfg: SoccerRLEnvCfg
    def __init__(self, cfg: SoccerRLEnvCfg, render_mode: str | None = None, **kwargs):
        super(SoccerRLEnv, self).__init__(cfg, render_mode, **kwargs)
        self.phase_time = torch.zeros(self.num_envs, device=self.device)

    def reset(self, seed: int | None = None, options = None):
        output = super(SoccerRLEnv, self).reset(seed, options)
        self.phase_time = torch.rand(self.phase_time.shape, device=self.phase_time.device)
        return output

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.phase_time[env_ids] = torch.rand(len(env_ids), device=self.phase_time.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        self.phase_time = torch.fmod(self.phase_time + self.step_dt / self.cfg.phase_freq, 1.0)
        if self.cfg.only_positive_rewards:
            self.reward_buf = torch.clamp(self.reward_buf, min=0.0)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras