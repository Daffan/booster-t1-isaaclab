import gymnasium as gym
from . import walk_env_config, ppo_config

gym.register(
    id="Booster-T1-lb-walk-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_config.LowerBodyRLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1WalkFlatPPORunnerCfg,
    },
)

gym.register(
    id="Booster-T1-wb-walk-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_config.WholeBodyRLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1WholeBodyWalkFlatPPORunnerCfg,
    },
)

# gym.register(
#     id="Booster-T1-wb-kick-v0",
#     entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": kick_env_config.LowerBodyKick,
#         "rsl_rl_cfg_entry_point": ppo_config.T1KickPPORunnerCfg,
#     },
# )

# gym.register(
#     id="Booster-T1-wb-kick-v0",
#     entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": kick_env_config.WholeBodyKick,
#         "rsl_rl_cfg_entry_point": ppo_config.T1KickPPORunnerCfg,
#     },
# )