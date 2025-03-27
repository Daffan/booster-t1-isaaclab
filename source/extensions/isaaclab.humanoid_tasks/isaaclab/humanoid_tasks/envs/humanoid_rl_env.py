from typing import Sequence
import torch
from collections import deque

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

@configclass
class HumanoidRLEnvCfg(ManagerBasedRLEnvCfg):
    phase_freq: float = 1.0
    only_positive_rewards: bool = False


class HumanoidRLEnv(ManagerBasedRLEnv):
    """ Humanoid RL manager-based environment.
    - Internal phase clock
    - Only positive rewards
    - Last 3 timestep action histories
    """
    cfg: HumanoidRLEnvCfg
    def __init__(self, cfg: HumanoidRLEnvCfg, render_mode: str | None = None, **kwargs):
        super(HumanoidRLEnv, self).__init__(cfg, render_mode, **kwargs)
        self.phase_time = torch.zeros(self.num_envs, device=self.device)
        self.action_histories = deque(maxlen=3)

    def reset(self, seed: int | None = None, options = None):
        output = super(HumanoidRLEnv, self).reset(seed, options)
        self.phase_time = torch.rand(self.phase_time.shape, device=self.phase_time.device)
        return output

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.phase_time[env_ids] = torch.rand(len(env_ids), device=self.phase_time.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        self.action_histories.append(action.clone())  # consider using action after clipping
        self.phase_time = torch.fmod(self.phase_time + self.step_dt / self.cfg.phase_freq, 1.0)
        if self.cfg.only_positive_rewards:
            self.reward_buf = torch.clamp(self.reward_buf, min=0.0)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras