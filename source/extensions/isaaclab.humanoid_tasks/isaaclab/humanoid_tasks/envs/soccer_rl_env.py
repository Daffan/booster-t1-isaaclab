from typing import Sequence
import torch
from collections import deque

from isaaclab.assets import Articulation, RigidObject
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
        self.action_histories = deque(maxlen=3)
        self.rel_robot_ball_pos = torch.zeros((self.num_envs, 2, 2), device=self.device)
        self.rel_ball_target_pos = torch.zeros((self.num_envs, 2, 2), device=self.device)

    def reset(self, seed: int | None = None, options = None):
        output = super(SoccerRLEnv, self).reset(seed, options)
        self.phase_time = torch.rand(self.phase_time.shape, device=self.phase_time.device)
        self.rel_robot_ball_pos = torch.zeros((self.num_envs, 2, 2), device=self.device)
        self.rel_ball_target_pos = torch.zeros((self.num_envs, 2, 2), device=self.device)
        return output

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.phase_time[env_ids] = torch.rand(len(env_ids), device=self.phase_time.device)
        self.rel_robot_ball_pos[env_ids] = torch.zeros((len(env_ids), 2, 2), device=self.device)
        self.rel_ball_target_pos[env_ids] = torch.zeros((len(env_ids), 2, 2), device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        self.phase_time = torch.fmod(self.phase_time + self.step_dt / self.cfg.phase_freq, 1.0)
        self.action_histories.append(action.clone())

        r_asset: Articulation = self.scene["robot"]
        b_asset: RigidObject = self.scene["ball"]
        g_asset: RigidObject = self.scene["goal"]

        rel_robot_ball_pos = b_asset.data.root_pos_w[:, :2] - r_asset.data.root_pos_w[:, :2]
        rel_ball_goal_pos = g_asset.data.root_pos_w[:, :2] - b_asset.data.root_pos_w[:, :2]
        self.rel_robot_ball_pos[:, 1] = self.rel_robot_ball_pos[:, 0]
        self.rel_robot_ball_pos[:, 0] = rel_robot_ball_pos
        self.rel_ball_target_pos[:, 1] = self.rel_ball_target_pos[:, 0]
        self.rel_ball_target_pos[:, 0] = rel_ball_goal_pos

        if self.cfg.only_positive_rewards:
            self.reward_buf = torch.clamp(self.reward_buf, min=0.0)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras