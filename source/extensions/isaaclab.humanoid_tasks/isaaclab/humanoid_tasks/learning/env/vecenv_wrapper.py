import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as IsaaclabRslRlVecEnvWrapper

class RslRlVecEnvWrapper(IsaaclabRslRlVecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def process_obs_dict(self, obs_dict: dict):
        """Processes the observations dictionary."""
        if isinstance(obs_dict["policy"], dict):
            # each key has shape (batch_size, num_hist, num_features)
            values = list(obs_dict["policy"].values())
            obs_hist = torch.cat(values, dim=-1) # (batch_size, num_hist, total_num_features)
            obs = obs_hist[:, -1, :] # (batch_size, total_num_features)
            if "privilege" in obs_dict:
                privilege = obs_dict["privilege"]
                # overwrite critic
                obs_dict["critic"] = torch.cat([obs, privilege], dim=-1)
                # new policy obs
                obs = torch.cat([obs, privilege, obs_hist.flatten(start_dim=1)], dim=-1)
            else:
                obs = torch.cat([obs, obs_hist.flatten(start_dim=1)], dim=-1)
        else:
            obs = obs_dict["policy"]
        return obs, {"observations": obs_dict}

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        return self.process_obs_dict(obs_dict)

    def get_obs_info(self):
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        if isinstance(obs_dict["policy"], dict):
            # each key has shape (batch_size, num_hist, num_features)
            values = list(obs_dict["policy"].values())
            obs_hist = torch.cat(values, dim=-1) # (batch_size, num_hist, total_num_features)
            _, num_hist, num_prop = obs_hist.shape
            obs = obs_hist[:, -1, :] # (batch_size, total_num_features)
            if "privilege" in obs_dict:
                privilege = obs_dict["privilege"]
                _, num_priv = privilege.shape
                # overwrite critic
                obs_dict["critic"] = torch.cat([obs, privilege], dim=-1)
                obs = torch.cat([obs, privilege, obs_hist.flatten(start_dim=1)], dim=-1)
            else:
                obs = torch.cat([obs, obs_hist.flatten(start_dim=1)], dim=-1)
        else:
            obs = obs_dict["policy"]
            num_priv, num_hist, num_prop = 0, 0, 0

        num_obs = obs.shape[-1]
        num_critic_obs = obs_dict["critic"].shape[-1] if "critic" in obs_dict else num_obs
        
        return num_obs, num_critic_obs, num_priv, num_hist, num_prop
    
    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environment instances."""
        obs_dict, _ = self.env.reset()

        return self.process_obs_dict(obs_dict)
    
    def step(self, actions):
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs, extras_obs = self.process_obs_dict(obs_dict)
        extras.update(extras_obs)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras
    
