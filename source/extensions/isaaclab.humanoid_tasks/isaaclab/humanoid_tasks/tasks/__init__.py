import gymnasium as gym
from . import walk_env_config, kick_env_config
from . import ppo_config, roa_ppo_config

#### Walk tasks ####

gym.register(
    id="Booster-T1-lb-walk-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_config.LowerBodyRLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1WalkPPORunnerCfg,
    },
)

gym.register(
    id="Booster-T1-lb-priv-walk-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_config.LowerPrivRLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1WalkPPORunnerCfg,
    },
)

gym.register(
    id="Booster-T1-wb-walk-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_config.WholeBodyRLEnvCfg,
        "rsl_rl_cfg_entry_point": roa_ppo_config.T1BoosterWalkROAPPORunnerCfg,
    },
)

gym.register(
    id="Booster-T1-lb-RMA-walk-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_config.LowerBodyHistoryRLEnvCfg,
        "rsl_rl_cfg_entry_point": roa_ppo_config.T1BoosterWalkROAPPORunnerCfg(
            experiment_name="t1-lb-RMA-walk",  # for logging purposes
        ),
    },
)

gym.register(
    id="Booster-T1-lb-RMA-walk-v1",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_config.LowerBodyHistoryV1RLEnvCfg,
        "rsl_rl_cfg_entry_point": roa_ppo_config.T1BoosterWalkROAPPORunnerCfg,
    },
)

gym.register(
    id="Booster-T1-wb-RMA-walk-v0",
    entry_point="isaaclab.humanoid_tasks.envs:HumanoidRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_config.WholeBodyHistoryRLEnvCfg,
        "rsl_rl_cfg_entry_point": ppo_config.T1WholeBodyWalkFlatPPORunnerCfg,
    },
)

#### Kick tasks ####

gym.register(
    id="Booster-T1-lb-pre-kick-priv-v0",
    entry_point="isaaclab.humanoid_tasks.envs:SoccerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": kick_env_config.LowerBodyPreKickPriv,
        "rsl_rl_cfg_entry_point": ppo_config.T1WalkPPORunnerCfg(
            experiment_name="t1-lb-pre-kick-priv",  # for logging purposes
        ),
    },
)

gym.register(
    id="Booster-T1-lb-kick-priv-v0",
    entry_point="isaaclab.humanoid_tasks.envs:SoccerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": kick_env_config.LowerBodyKickPriv,
        "rsl_rl_cfg_entry_point": ppo_config.T1KickPPORunnerCfg(
            experiment_name="t1-lb-kick-priv",  # for logging purposes
        ),
    },
)