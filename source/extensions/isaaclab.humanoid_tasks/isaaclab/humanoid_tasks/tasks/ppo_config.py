# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class T1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "t1_human_reward"
    logger = "wandb"
    wandb_project = "RL-Booster"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class T1WalkFlatPPORunnerCfg(T1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "t1_walking_flat"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        self.algorithm.value_loss_coef = 2.0


@configclass
class T1BoosterWalkFlatPPORunnerCfg(T1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "t1_booster_walking_flat"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        self.algorithm.value_loss_coef = 2.0

@configclass
class T1WholeBodyWalkFlatPPORunnerCfg(T1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "t1_booster_wholebody_walking"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        self.algorithm.value_loss_coef = 2.0


@configclass
class T1KickPPORunnerCfg(T1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "t1_kick"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        self.algorithm.value_loss_coef = 2.0