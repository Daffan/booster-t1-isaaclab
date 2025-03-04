import numpy as np
import torch

from .command_cfg import UniformVelocityFreqCommandCfg
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.humanoid_tasks.envs import HumanoidRLEnv

class UniformVelocityFreqCommand(UniformVelocityCommand):
    cfg: UniformVelocityFreqCommandCfg

    def __init__(self, cfg: UniformVelocityFreqCommandCfg, env: HumanoidRLEnv):
        super().__init__(cfg, env)
        self.gait_frequency = torch.zeros(self.num_envs, device=self.device)
        self.gait_progress = torch.rand(self.num_envs, device=self.device)

        self.filtered_lin_vel = torch.zeros_like(self.robot.data.root_lin_vel_w)
        self.filtered_ang_vel = torch.zeros_like(self.robot.data.root_ang_vel_w)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return torch.cat([self.vel_command_b, self.gait_frequency.unsqueeze(-1)], dim=-1)
    
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.filtered_lin_vel[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.filtered_ang_vel[:, 2]) / max_command_step
        )
    
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        self.gait_frequency[env_ids] = r.uniform_(*self.cfg.ranges.gait_frequency)
        self.gait_progress[env_ids] = r.uniform_(0, 1)

        self.filtered_ang_vel[env_ids] = self.robot.data.root_ang_vel_w[env_ids]
        self.filtered_lin_vel[env_ids] = self.robot.data.root_lin_vel_w[env_ids]

    def set_command(self, lin_x: float, lin_y: float, yaw: float, gait_freq: float):
        self.vel_command_b = torch.tensor([lin_x, lin_y, yaw], device=self.device)
        self.gait_frequency = torch.tensor(gait_freq, device=self.device)

    def _update_command(self):
        super()._update_command()
        # 0 for robot standing still
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.gait_frequency[standing_env_ids] = 0.0

        # filter the linear and angular velocity
        self.filtered_lin_vel = self.cfg.filter_weight * self.robot.data.root_lin_vel_w + \
                                (1 - self.cfg.filter_weight) * self.filtered_lin_vel
        self.filtered_ang_vel = self.cfg.filter_weight * self.robot.data.root_ang_vel_w + \
                                (1 - self.cfg.filter_weight) * self.filtered_ang_vel

        self.gait_progress = torch.fmod(self.gait_progress + self.gait_frequency * self._env.step_dt, 1.0)
        
    def get_filtered_velocities(self):
        return self.filtered_lin_vel, self.filtered_ang_vel
    
    def reset(self, env_ids = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.filtered_lin_vel[env_ids] = torch.zeros_like(self.robot.data.root_lin_vel_w[env_ids])
        self.filtered_ang_vel[env_ids] = torch.zeros_like(self.robot.data.root_ang_vel_w[env_ids])
        return super().reset(env_ids)
