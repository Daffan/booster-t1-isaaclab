from typing import Sequence
import torch
from collections import deque

from omni.isaac.lab.envs.common import VecEnvStepReturn
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

def sqrdexp(x, sigma):
    """ shorthand helper for squared exponential
    """
    return torch.exp(-torch.square(x)/sigma)

@configclass
class HumanoidRLEnvCfg(ManagerBasedRLEnvCfg):
    phase_freq: float = 1.0


class HumanoidRLEnv(ManagerBasedRLEnv):
    """ Humanoid RL manager-based environment.
    the environment maintains properties:
        1. reference phase time of the motion
        2. pervious joint regularization error for computing potential-based reward
        3. action history for computing action smoothness loss
    """
    cfg: HumanoidRLEnvCfg
    def __init__(self, cfg: HumanoidRLEnvCfg, render_mode: str | None = None, **kwargs):
        super(HumanoidRLEnv, self).__init__(cfg, render_mode, **kwargs)
        self.phase_time = torch.zeros(self.num_envs, device=self.device)
        self.rwd_jointRegPrev = torch.zeros(self.num_envs, device=self.device)
        self.action_histories = deque(maxlen=3)

    def reset(self, seed: int | None = None, options = None):
        output = super(HumanoidRLEnv, self).reset(seed, options)
        self.phase_time = torch.rand(self.phase_time.shape, device=self.phase_time.device) / 2 / torch.pi / self.cfg.phase_freq
        return output

    def _reset_idx(self, env_ids: Sequence[int]):
        out = super()._reset_idx(env_ids)
        self.phase_time[env_ids] = torch.rand(len(env_ids), device=self.phase_time.device) / 2 / torch.pi / self.cfg.phase_freq

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # compute rwd_jointRegPrev for potential-based reward
        self.action_histories.append(action.clone())
        self.rwd_jointRegPrev = self._joint_regularization()

        super().step(action)
        self.phase_time = torch.fmod(self.phase_time + self.step_dt, 1.0 / self.cfg.phase_freq)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _joint_regularization(self) -> torch.Tensor:
        """Penalize joint symmetry. The joint indices are hard-coded for the T1 humanoid robot."""
        # extract the used quantities (to enable type-hinting)
        asset = self.scene["robot"]
        # regularize joint positions around default
        error = 0.
        # Yaw joints regularization around 0
        error += sqrdexp(
            (asset.data.joint_pos[:, 4]), 0.25)
        error += sqrdexp(
            (asset.data.joint_pos[:, 5]), 0.25)
        # Ab/ad joint symmetry
        error += sqrdexp(
            (asset.data.joint_pos[:, 2] + asset.data.joint_pos[:, 3]), 0.25)
        # Pitch joint symmetry
        error += sqrdexp(
            (asset.data.joint_pos[:, 0] - asset.data.joint_pos[:, 1]), 0.25)
        return error / 3