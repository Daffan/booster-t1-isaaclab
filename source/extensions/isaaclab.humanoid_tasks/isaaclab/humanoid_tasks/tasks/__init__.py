import gymnasium as gym
from . import walking_env_config, ppo_config

gym.register(
    id="RL-Booster-t1-walking-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walking_env_config.RLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1WalkFlatPPORunnerCfg,
    },
)