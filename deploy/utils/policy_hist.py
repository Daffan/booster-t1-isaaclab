import os
import numpy as np
import torch
from collections import deque

class Policy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()
        self.history_len = 50
        self.obss = deque(maxlen=self.history_len)
        self.obs_log = []

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = 0.0
        else:
            self.gait_frequency = self.cfg["policy"]["gait_frequency"]

        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[6] = (
            self.smoothed_commands[0] * self.cfg["policy"]["normalization"]["lin_vel"]
        )
        self.obs[7] = (
            self.smoothed_commands[1] * self.cfg["policy"]["normalization"]["lin_vel"]
        )
        self.obs[8] = (
            self.smoothed_commands[2] * self.cfg["policy"]["normalization"]["ang_vel"]
        )
        # self.obs[9] = self.gait_frequency
        self.obs[10-1] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[11-1] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[12-1:24-1] = (dof_pos - self.default_dof_pos)[11:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[24-1:36-1] = dof_vel[11:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[36-1:48-1] = self.actions

        # self.actions[:] = self.policy.act(torch.from_numpy(self.obs).unsqueeze(0)).loc.detach().numpy()
        if len(self.obss) == 0:
            self.obss.extend([np.zeros_like(self.obs)] * self.history_len)
        self.obss.append(self.obs.copy())
            
        self.actions[:] = self.policy.act_mj(torch.from_numpy(np.stack(self.obss)).unsqueeze(0)).squeeze(0).detach().numpy()
        self.actions = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        with open(os.path.join("debug", "obs_roa.txt"), "ab") as file:
            file.write(b"\n")
            np.savetxt(file, self.obs.flatten(), fmt="%.4f", newline=", ")

        return self.dof_targets
