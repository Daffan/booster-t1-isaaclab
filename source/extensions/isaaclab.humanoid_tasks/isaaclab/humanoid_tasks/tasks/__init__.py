import gymnasium as gym
from . import walking_env_config, kick_env_config, booster_walking_env_config, ppo_config

gym.register(
    id="RL-Custom-t1-walking-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walking_env_config.LowerBodyRLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1WalkFlatPPORunnerCfg,
    },
)

gym.register(
    id="RL-Wholebody-t1-walking-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walking_env_config.WholeBodyRLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1WholeBodyWalkFlatPPORunnerCfg,
    },
)

gym.register(
    id="RL-Booster-t1-walking-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": booster_walking_env_config.RLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1BoosterWalkFlatPPORunnerCfg,
    },
)

gym.register(
    id="RL-Booster-t1-kick-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": kick_env_config.WholeBodyKick,
        "rsl_rl_cfg_entry_point": ppo_config.T1KickPPORunnerCfg,
    },
)

gym.register(
    id="RL-LowerBody-t1-kick-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": kick_env_config.LowerBodyKick,
        "rsl_rl_cfg_entry_point": ppo_config.T1KickPPORunnerCfg,
    },
)

gym.register(
    id="RL-LowerBody-t1-run-kick-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": kick_env_config.LowerBodyRunKick,
        "rsl_rl_cfg_entry_point": ppo_config.T1KickPPORunnerCfg,
    },
)

