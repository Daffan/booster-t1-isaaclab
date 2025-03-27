import torch
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):

    def __init__(self, num_act, num_obs, num_privileged_obs):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

    def act(self, obs):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)

isaac_gym_joint_names = [
    'Left_Hip_Pitch',
    'Left_Hip_Roll',
    'Left_Hip_Yaw',
    'Left_Knee_Pitch',
    'Left_Ankle_Pitch',
    'Left_Ankle_Roll',
    'Right_Hip_Pitch',
    'Right_Hip_Roll',
    'Right_Hip_Yaw',
    'Right_Knee_Pitch',
    'Right_Ankle_Pitch',
    'Right_Ankle_Roll'
]

isaac_lab_joint_names = [
    'Left_Hip_Pitch',
    'Right_Hip_Pitch',
    'Left_Hip_Roll',
    'Right_Hip_Roll',
    'Left_Hip_Yaw',
    'Right_Hip_Yaw',
    'Left_Knee_Pitch',
    'Right_Knee_Pitch',
    'Left_Ankle_Pitch',
    'Right_Ankle_Pitch',
    'Left_Ankle_Roll',
    'Right_Ankle_Roll'
]

gym2lab_mapping = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
lab2gym_mapping = [gym2lab_mapping.index(i) for i in range(12)]


class IsaacLabActorCritic(torch.nn.Module):

    def __init__(self, num_act, num_obs, num_privileged_obs):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

    def process_obs(self, obs):
        # gait_frequency = (torch.norm(obs[:, 6:9]) < 0.1).float() * 2.0
        obs = torch.cat(
            (
                obs[:, :12],
                obs[:, 12:24][:, gym2lab_mapping],
                obs[:, 24:36][:, gym2lab_mapping],
                obs[:, 36:48][:, gym2lab_mapping]
            ),
            dim=-1,
        )
        return obs

    def act(self, obs):
        obs = self.process_obs(obs)
        action_mean = self.actor(obs)
        action_mean = action_mean[:, lab2gym_mapping]
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)
